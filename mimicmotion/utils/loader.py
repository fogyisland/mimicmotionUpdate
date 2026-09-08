import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline

logger = logging.getLogger(__name__)

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        # `from_config` does not accept `use_safetensors` / `variant` — those are
        # loading kwargs for `from_pretrained`. The unet structure is initialised
        # here with random weights, then `load_state_dict` overwrites it with the
        # MimicMotion checkpoint weights below.
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"),
        )
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", use_safetensors=True, variant="fp16",
        ).half()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", use_safetensors=True, variant="fp16",
        )
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler",
        )
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor",
        )
        # pose_net
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])

def _safe_torch_load(path, map_location):
    """Load a checkpoint, preferring `weights_only=True` for safety.

    PyTorch 2.x warns when `weights_only` is unspecified. We try the safe
    path first and fall back to the legacy behavior for checkpoints that
    embed arbitrary Python objects.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch that doesn't accept `weights_only`.
        return torch.load(path, map_location=map_location)
    except Exception:
        # Checkpoint may include non-tensor metadata (e.g. OmegaConf objects).
        logger.warning(
            "Falling back to weights_only=False for %s; the checkpoint "
            "contains objects that require the legacy loader.",
            path,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def create_pipeline(infer_config, device):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str):
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """
    mimicmotion_models = MimicMotionModel(infer_config.base_model_path).to(device=device).eval()
    # `strict=False` silently swallows mismatched keys; raise loudly when the
    # checkpoint is incompatible instead of producing garbage outputs.
    ckpt = _safe_torch_load(infer_config.ckpt_path, map_location=device)
    result = mimicmotion_models.load_state_dict(ckpt, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(
            "MimicMotion checkpoint has %d missing keys and %d unexpected keys.",
            len(result.missing_keys),
            len(result.unexpected_keys),
        )
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae,
        image_encoder=mimicmotion_models.image_encoder,
        unet=mimicmotion_models.unet,
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor,
        pose_net=mimicmotion_models.pose_net
    )
    return pipeline

