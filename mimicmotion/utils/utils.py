import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _write_video_torchvision(save_path, frames_array, fps):
    """Legacy path: torchvision.io.write_video was removed in newer torchvision."""
    from torchvision.io import write_video  # noqa: F401
    write_video(str(save_path), frames_array, fps=int(fps))


def _write_video_imageio(save_path, frames_array, fps):
    """imageio path; imageio is a transitive dep of moviepy so it's usually available."""
    import imageio

    imageio.mimsave(
        str(save_path),
        list(frames_array),
        fps=fps,
        codec="libx264",
        quality=8,
    )


def _write_video_cv2(save_path, frames_array, fps):
    """cv2 fallback when neither torchvision nor imageio is available."""
    import cv2

    height, width = frames_array.shape[1], frames_array.shape[2]
    # mp4v is broadly compatible; libx264 needs the ffmpeg backend which cv2
    # may not ship with on every platform.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(
            f"cv2.VideoWriter failed to open {save_path}; check that the "
            "platform supports the mp4v codec."
        )
    try:
        for frame in frames_array:
            # OpenCV expects BGR.
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_to_mp4(frames, save_path, fps=7):
    """Persist a video tensor to disk using the best available backend.

    Args:
        frames: Tensor shaped ``(num_frames, channels, height, width)``.
        save_path: Destination path. Parent directories are created as needed.
        fps: Output frames-per-second.
    """
    # (f, c, h, w) -> (f, h, w, c); cast to uint8 RGB frames.
    frames = frames.permute((0, 2, 3, 1))
    frames_array = frames.detach().cpu().numpy()
    if frames_array.dtype != np.uint8:
        frames_array = frames_array.astype(np.uint8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Try backends in order: torchvision (fast, old installs) -> imageio ->
    # cv2. Skip a backend cleanly when it isn't available so the user still
    # gets a working video writer.
    for backend in (_write_video_torchvision, _write_video_imageio, _write_video_cv2):
        try:
            backend(save_path, frames_array, fps)
            return
        except ImportError:
            continue
        except Exception as e:  # noqa: BLE001
            logger.warning("save_to_mp4 backend %s failed: %s", backend.__name__, e)
            continue

    raise RuntimeError(
        "No working video writer found. Please install one of: torchvision<0.22, "
        "imageio, or opencv-python."
    )