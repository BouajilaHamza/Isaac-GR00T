import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
import cv2

# Import the functions you implemented (adjust import path)
from gr00t.utils.video import get_frames_by_indices, get_frames_by_timestamps, get_all_frames


class TestGetFramesByIndices:

    def test_decord_backend_returns_frames(self):
        mock_frames = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8)
        with patch("decord.VideoReader") as mock_vr:
            mock_reader = MagicMock()
            mock_reader.get_batch.return_value.asnumpy.return_value = mock_frames
            mock_vr.return_value = mock_reader

            indices = [0, 5, 10]
            result = get_frames_by_indices("video.mp4", indices, video_backend="decord")

            mock_vr.assert_called_once_with("video.mp4")
            mock_reader.get_batch.assert_called_once_with(indices)
            np.testing.assert_array_equal(result, mock_frames)

    def test_opencv_backend_reads_frames(self):
        with patch("cv2.VideoCapture") as mock_cap:
            mock_capture = MagicMock()
            mock_capture.isOpened.return_value = True
            # Simulate successful reads for 3 frames
            frames = [np.ones((224, 224, 3), dtype=np.uint8) * i for i in range(3)]
            mock_capture.read.side_effect = [(True, frame) for frame in frames]
            mock_cap.return_value = mock_capture

            indices = [0, 10, 20]
            result = get_frames_by_indices("video.mp4", indices, video_backend="opencv")

            mock_cap.assert_called_once_with("video.mp4")
            # Check that cap.set was called for each index
            expected_calls = [call(cv2.CAP_PROP_POS_FRAMES, idx) for idx in indices]
            mock_capture.set.assert_has_calls(expected_calls, any_order=False)
            assert mock_capture.read.call_count == len(indices)
            mock_capture.release.assert_called_once()
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == len(indices)

    def test_opencv_backend_read_failure_raises(self):
        with patch("cv2.VideoCapture") as mock_cap:
            mock_capture = MagicMock()
            mock_capture.isOpened.return_value = True
            # Fail reading the first frame
            mock_capture.read.return_value = (False, None)
            mock_cap.return_value = mock_capture

            with pytest.raises(ValueError, match="Unable to read frame at index 0"):
                get_frames_by_indices("video.mp4", [0], video_backend="opencv")

    def test_unsupported_backend_raises(self):
        with pytest.raises(NotImplementedError):
            get_frames_by_indices("video.mp4", [0], video_backend="unknown")


class TestGetFramesByTimestamps:

    def test_decord_backend_returns_frames(self):
        with patch("decord.VideoReader") as mock_vr:
            mock_reader = MagicMock()
            mock_reader.__len__.return_value = 5
            # Simulate frame timestamps shape (num_frames, 1)
            timestamps_array = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
            mock_reader.get_frame_timestamp.return_value = timestamps_array
            mock_reader.get_batch.return_value.asnumpy.return_value = np.zeros((2, 224, 224, 3), dtype=np.uint8)
            mock_vr.return_value = mock_reader

            timestamps = [0.5, 3.2]
            result = get_frames_by_timestamps("video.mp4", timestamps, video_backend="decord")

            mock_vr.assert_called_once_with("video.mp4")
            mock_reader.get_frame_timestamp.assert_called_once_with(range(5))
            mock_reader.get_batch.assert_called_once()
            assert isinstance(result, np.ndarray)

    def test_opencv_backend_reads_frames(self):
        with patch("cv2.VideoCapture") as mock_cap:
            mock_capture = MagicMock()
            mock_capture.isOpened.return_value = True
            # Mock CAP_PROP_FRAME_COUNT and FPS
            mock_capture.get.side_effect = lambda x: {cv2.CAP_PROP_FRAME_COUNT: 5, cv2.CAP_PROP_FPS: 1.0}[x]
            # Simulate successful reads for 2 frames
            frames = [np.ones((224, 224, 3), dtype=np.uint8) * i for i in range(2)]
            mock_capture.read.side_effect = [(True, frame) for frame in frames]
            mock_cap.return_value = mock_capture

            timestamps = [0.5, 3.2]
            result = get_frames_by_timestamps("video.mp4", timestamps, video_backend="opencv")

            mock_cap.assert_called_once_with("video.mp4")
            assert mock_capture.isOpened.call_count >= 1
            assert mock_capture.read.call_count == len(timestamps)
            mock_capture.release.assert_called_once()
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == len(timestamps)

    def test_opencv_backend_file_not_opened_raises(self):
        with patch("cv2.VideoCapture") as mock_cap:
            mock_capture = MagicMock()
            mock_capture.isOpened.return_value = False
            mock_cap.return_value = mock_capture

            with pytest.raises(ValueError, match="Unable to open video file"):
                get_frames_by_timestamps("video.mp4", [0.5], video_backend="opencv")

    def test_torchvision_av_backend_reads_frames_and_seeks(self):
        # Patch torchvision set_video_backend and VideoReader
        with patch("torchvision.set_video_backend") as mock_set_backend, \
             patch("torchvision.io.VideoReader") as mock_reader_class:

            # Create a mock reader instance
            mock_reader = MagicMock()
            # Create frames with 'pts' and 'data' keys
            frames = [
                {"pts": 0.0, "data": np.zeros((3, 224, 224), dtype=np.uint8)},
                {"pts": 1.0, "data": np.ones((3, 224, 224), dtype=np.uint8)},
                {"pts": 2.0, "data": np.ones((3, 224, 224), dtype=np.uint8) * 2},
            ]
            mock_reader.__iter__.return_value = iter(frames)
            mock_reader.container = MagicMock()
            mock_reader_class.return_value = mock_reader

            timestamps = [0.5, 1.5]
            result = get_frames_by_timestamps("video.mp4", timestamps, video_backend="torchvision_av")

            mock_set_backend.assert_called_once_with("pyav")
            mock_reader.seek.assert_called_once_with(timestamps[0], keyframes_only=True)
            mock_reader.container.close.assert_called_once()
            assert isinstance(result, np.ndarray)
            # Check shape is (num_frames, height, width, channels)
            assert result.shape[0] <= len(frames)
            assert result.shape[3] == 3  # RGB channels

    def test_torchvision_av_empty_timestamps_raises(self):
        with patch("torchvision.set_video_backend"), patch("torchvision.io.VideoReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.__iter__.return_value = iter([])
            mock_reader.container = MagicMock()
            mock_reader_class.return_value = mock_reader

            with pytest.raises(IndexError):
                # Your code might raise IndexError if timestamps is empty and you access timestamps[0]
                get_frames_by_timestamps("video.mp4", [], video_backend="torchvision_av")

    def test_unsupported_backend_raises(self):
        with pytest.raises(NotImplementedError):
            get_frames_by_timestamps("video.mp4", [0.5], video_backend="unknown")


class TestGetAllFrames:

    def test_decord_backend_returns_all_frames(self):
        with patch("decord.VideoReader") as mock_vr:
            mock_reader = MagicMock()
            mock_reader.__len__.return_value = 3
            mock_reader.get_batch.return_value.asnumpy.return_value = np.zeros((3, 224, 224, 3), dtype=np.uint8)
            mock_vr.return_value = mock_reader

            result = get_all_frames("video.mp4", video_backend="decord")

            mock_vr.assert_called_once_with("video.mp4")
            mock_reader.get_batch.assert_called_once_with(range(3))
            assert isinstance(result, np.ndarray)

    def test_torchvision_av_backend_returns_all_frames(self):
        with patch("torchvision.set_video_backend") as mock_set_backend, \
             patch("torchvision.io.VideoReader") as mock_reader_class:

            mock_reader = MagicMock()
            frames = [
                {"data": np.zeros((3, 224, 224), dtype=np.uint8)},
                {"data": np.ones((3, 224, 224), dtype=np.uint8)},
            ]
            mock_reader.__iter__.return_value = iter(frames)
            mock_reader_class.return_value = mock_reader

            result = get_all_frames("video.mp4", video_backend="torchvision_av")

            mock_set_backend.assert_called_once_with("pyav")
            mock_reader_class.assert_called_once_with("video.mp4", "video")
            assert isinstance(result, np.ndarray)
            assert result.shape[3] == 3  # RGB channels

    def test_pyav_backend_returns_all_frames(self):
        # Patch av.open to raise NotImplementedError immediately to avoid FileNotFoundError
        with patch("av.open", side_effect=NotImplementedError), \
            pytest.raises(NotImplementedError):
            get_all_frames("video.mp4", video_backend="pyav")

    def test_unsupported_backend_raises(self):
        with pytest.raises(NotImplementedError):
            get_all_frames("video.mp4", video_backend="unknown")

    def test_resize_frames(self):
        with patch("decord.VideoReader") as mock_vr, patch("cv2.resize") as mock_resize:
            mock_reader = MagicMock()
            mock_reader.__len__.return_value = 2
            frames = np.random.randint(0, 255, (2, 480, 640, 3), dtype=np.uint8)
            mock_reader.get_batch.return_value.asnumpy.return_value = frames
            mock_vr.return_value = mock_reader

            resized_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            mock_resize.side_effect = [resized_frame, resized_frame]

            result = get_all_frames("video.mp4", video_backend="decord", resize_size=(224, 224))

            assert mock_resize.call_count == 2
            for call_args in mock_resize.call_args_list:
                args, kwargs = call_args
                # Check resize called with correct size
                assert args[1] == (224, 224)
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 224, 224, 3)
