import torch
import numpy as np
import onnx
import onnxruntime as ort

class Gr00tONNXWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, **inputs):
        obs = {k: v for k, v in inputs.items()}
        obs = self.policy.apply_transforms(obs)
        action = self.policy.model.get_action(obs)["action_pred"].float()
        return action

def export_policy_to_onnx(policy, onnx_path: str, example_input: dict):
    """
    Export the policy's model to ONNX format using the wrapper.
    """
    policy.model.eval()
    wrapper = Gr00tONNXWrapper(policy)
    input_tensors = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in example_input.items()}
    torch.onnx.export(
        wrapper,
        input_tensors,
        onnx_path,
        input_names=list(input_tensors.keys()),
        output_names=["action_pred"],
        dynamic_axes={k: {0: "batch"} for k in input_tensors.keys()},
        opset_version=17,
    )
    print(f"Exported policy model to ONNX at {onnx_path}")

def inference_onnx(onnx_path: str, input_data: dict) -> dict:
    """
    Run inference using ONNX Runtime (CPU) on the exported ONNX model.
    """
    ort_inputs = {k: v if isinstance(v, np.ndarray) else v.cpu().numpy() for k, v in input_data.items()}
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outputs = session.run(None, ort_inputs)
    return {"action": outputs[0]}
