import os
from typing import Callable, List, Optional, Union
import PIL
import numpy as np
import torch

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.utils import logging
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines import StableDiffusionPipeline
from control_lora.models.control import ControlLoRAContainer
from control_lora.models.lora import ControlLoRACrossAttnProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



def prepare_control_image(control_image):
    if isinstance(control_image, list) and len(control_image) > 0 and isinstance(control_image[0], list):
        control_image = torch.cat([prepare_control_image(img) for img in control_image], 0)
    if isinstance(control_image, torch.Tensor):
        # Batch single image
        if control_image.ndim == 3:
            assert control_image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            control_image = control_image.unsqueeze(0)

        # Image as float32
        control_image = control_image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(control_image, (PIL.Image.Image, np.ndarray)):
            control_image = [control_image]

        # we always concat to last color channels and output batch size 1 control image
        if isinstance(control_image, list) and isinstance(control_image[0], PIL.Image.Image):
            control_image = [np.array(i.convert("RGB"))[None, :] for i in control_image]
            control_image = np.concatenate(control_image, axis=-1) 
        elif isinstance(control_image, list) and isinstance(control_image[0], np.ndarray):
            control_image = np.concatenate([i[None, :] for i in control_image], axis=-1)

        control_image = control_image.transpose(0, 3, 1, 2)
        control_image = torch.from_numpy(control_image).to(dtype=torch.float32) / 127.5 - 1.0

    return control_image


class StableDiffusionControlLoRAPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        control_lora: ControlLoRAContainer,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=requires_safety_checker)
        
        self.register_modules(control_lora=control_lora)
        control_lora.set_as_unet_processor(unet)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        control_lora = kwargs.pop('control_lora_pretrained_model_name_or_path', None)
        if control_lora is not None:
            control_lora = ControlLoRAContainer.from_pretrained(control_lora)
            kwargs['control_lora'] = control_lora

        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @torch.no_grad()
    def __call__(
            self, 
            *args, 
            control_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image], List[List[PIL.Image.Image]]] = None,
            **kwargs):
        
        device = self._execution_device

        control_lora: ControlLoRAContainer = self.control_lora
        if control_image is None:
            processor_layer: ControlLoRACrossAttnProcessor = control_lora.processor_layers[0][0]
            if getattr(processor_layer, 'control_states', 1) is None:
                raise ValueError("`control_image` input cannot be undefined.")

            logger.warning(
                "`control_image` input cannot be undefined."
                " Pipeline will use old cached control states."
            )
        else:
            control_image = prepare_control_image(control_image).to(device)
            _ = control_lora(control_image).output

        return super().__call__(*args, **kwargs)

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[Callable] = None) -> None:
        super().set_use_memory_efficient_attention_xformers(valid, attention_op)

        # unet will clear the processors.
        # we are not sure about the order to enable xformers.
        # so we manually set the processors.
        unet: UNet2DConditionModel = self.unet
        control_lora: ControlLoRAContainer = self.control_lora
        control_lora.set_as_unet_processor(unet)
