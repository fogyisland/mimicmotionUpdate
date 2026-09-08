import os

import numpy as np
import torch

from .wholebody import Wholebody

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DWposeDetector:
    """
    A pose detect method for image-like data.

    Parameters:
        model_det: (str) serialized ONNX format model path,
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx
        model_pose: (str) serialized ONNX format model path,
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx
        device: (str) 'cpu' or 'cuda:{device_id}'
    """
    def __init__(self, model_det, model_pose, device='cpu'):
        self.pose_estimation = Wholebody(model_det=model_det, model_pose=model_pose, device=device)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1

            # un_visible = subset < 0.3
            # candidate[un_visible] = -1

            # foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose


# Lazy singleton: defer model instantiation until first use so the module can
# be imported without the `dwpose` env var set. This means callers that don't
# actually run pose detection (e.g. ComfyUI loading the plugin's node list)
# won't crash on import.
_dwpose_detector_singleton = None


def _get_dwpose_detector():
    global _dwpose_detector_singleton
    if _dwpose_detector_singleton is None:
        if "dwpose" not in os.environ:
            raise RuntimeError(
                "Environment variable `dwpose` is not set. The MimicMotion "
                "plugin sets it automatically on import; if you are calling "
                "this module directly, export it to the DWPose weights "
                "directory before importing."
            )
        dwpose_dir = os.environ["dwpose"]
        _dwpose_detector_singleton = DWposeDetector(
            model_det=os.path.join(dwpose_dir, "yolox_l.onnx"),
            model_pose=os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx"),
            device=device,
        )
    return _dwpose_detector_singleton


class _LazyDWposeDetector:
    """Proxy that forwards attribute access and calls to the lazy singleton."""

    def __call__(self, *args, **kwargs):
        return _get_dwpose_detector()(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_get_dwpose_detector(), name)


dwpose_detector = _LazyDWposeDetector()
