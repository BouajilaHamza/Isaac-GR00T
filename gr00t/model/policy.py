# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1

COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        attn_implementation: str = "auto", # Added: "auto", "eager", "flash_attention_2"
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            attn_implementation (str): Preferred attention implementation. "auto" selects flash_attention_2
                                       if available on GPU, otherwise falls back to "eager".
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.attn_implementation_preference = attn_implementation # Store preference
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path)
        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # let the get_action handles both batch and single input
        is_batch = self._check_state_is_batched(observations)
        if not is_batch:
            observations = unsqueeze_dict_values(observations)

        normalized_input = unsqueeze_dict_values
        # Apply transforms
        normalized_input = self.apply_transforms(observations)

        normalized_action = self._get_action_from_normalized_input(normalized_input)
        unnormalized_action = self._get_unnormalized_action(normalized_action)

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)
        return unnormalized_action

    def to_onnx(
        self,
        output_path: str,
        example_normalized_input: Dict[str, torch.Tensor],
        opset_version: int = 17, # Common opset, might need adjustment
        verbose: bool = False,
        **kwargs,
    ):
        """
        Exports the core Gr00t model to ONNX format.

        IMPORTANT: This method exports ONLY the underlying torch.nn.Module (`self.model`).
        The preprocessing (`apply_transforms`) and postprocessing (`unapply_transforms`)
        steps are NOT included in the exported ONNX graph. You must reimplement
        these steps in your deployment environment.

        Args:
            output_path (str): Path to save the exported ONNX model file.
            example_normalized_input (Dict[str, torch.Tensor]): A dictionary containing example
                input tensors *after* applying `apply_transforms`. All tensors must be
                on the same device as the model (`self.device`). The batch size of the
                example input will determine the fixed batch size in the ONNX graph
                unless dynamic axes are specified.
            opset_version (int): The ONNX opset version to use for export.
            verbose (bool): If True, prints detailed ONNX export information.
            **kwargs: Additional keyword arguments passed directly to `torch.onnx.export`.
                      Useful for specifying `dynamic_axes`, `input_names`, `output_names`, etc.

        Example Usage:

        .. code-block:: python

            # Assume 'policy' is an initialized Gr00tPolicy instance
            # Assume 'observations' is a sample observation dictionary (unnormalized)

            # 1. Prepare normalized example input (MUST match model's device)
            if not policy._check_state_is_batched(observations):
                 observations = unsqueeze_dict_values(observations) # Add batch dim if needed
            normalized_input = policy.apply_transforms(observations)
            # Ensure tensors are on the correct device
            normalized_input_device = {
                k: v.to(policy.device) if isinstance(v, torch.Tensor) else v
                for k, v in normalized_input.items()
            }

            # 2. Define input/output names (match keys in normalized_input and model output)
            input_names = list(normalized_input_device.keys())
            # Assuming model.get_action returns {'action_pred': tensor}
            output_names = ["action_pred"]

            # 3. Define dynamic axes (optional, allows variable batch size)
            dynamic_axes = {name: {0: 'batch_size'} for name in input_names}
            dynamic_axes[output_names[0]] = {0: 'batch_size'}

            # 4. Export
            policy.to_onnx(
                "gr00t_model.onnx",
                normalized_input_device,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=True
            )
            print("ONNX export complete. Remember to handle pre/post-processing separately!")

        """
        # Ensure model is in eval mode
        self.model.eval()

        # Check if example input tensors are on the correct device
        for key, tensor in example_normalized_input.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != self.model.device:
                warnings.warn(
                    f"Input tensor '{key}' is on device {tensor.device} but model is on {self.model.device}. "
                    f"Moving tensor to {self.model.device} for export. Ensure future inputs match this device.",
                    UserWarning
                )
                example_normalized_input[key] = tensor.to(self.model.device)

        # Default input/output names if not provided in kwargs
        if "input_names" not in kwargs:
            kwargs["input_names"] = list(example_normalized_input.keys())
            print(f"Using default input names: {kwargs['input_names']}")
        if "output_names" not in kwargs:
            # Assuming the model's get_action primarily returns 'action_pred'
            # This might need adjustment based on the actual GR00T_N1 implementation
            kwargs["output_names"] = ["action_pred"]
            print(f"Using default output names: {kwargs['output_names']}")

        print(f"Starting ONNX export to {output_path} with opset {opset_version}...")
        print("IMPORTANT: Preprocessing (normalization) and Postprocessing (denormalization) are NOT included.")

        try:
            # We attempt to export the `get_action` method directly.
            # This might require specific model structure or ONNX opset support.
            # If this fails, exporting `self.model.forward` might be an alternative,
            # depending on how `get_action` uses `forward`.
            torch.onnx.export(
                self.model, # Export the underlying nn.Module
                (example_normalized_input,), # Input needs to be a tuple/args for export
                output_path,
                opset_version=opset_version,
                verbose=verbose,
                **kwargs,
            )
            print(f"ONNX model successfully exported to {output_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            print("Troubleshooting tips:")
            print("- Check if the `example_normalized_input` structure and dtypes exactly match what `self.model.get_action` expects.")
            print("- Try a different `opset_version` (e.g., 14, 16).")
            print("- Ensure all operations within `self.model.get_action` are supported by the chosen ONNX opset.")
            print("- Check the `dynamic_axes` definitions if used.")
            print("- Consider exporting `self.model.forward` instead if `get_action` has complex logic unsuitable for tracing.")
            raise # Re-raise the exception

    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path):
        model = GR00T_N1.from_pretrained(
            model_path,
            torch_dtype=COMPUTE_DTYPE,
            attn_implementation=self.attn_implementation_preference # Pass preference
        )
        model.eval()  # Set model to eval mode
        model.to(device=self.device)  # type: ignore
        print(f"model moved to device {self.device}")

        self.model = model
    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"


#######################################################################################################


# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data
