from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from math import sqrt
from collections import defaultdict
from diffusers.utils import is_xformers_available
from diffusers.utils.outputs import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.cross_attention import (
    CrossAttention, CrossAttnProcessor, XFormersCrossAttnProcessor,
    LoRALinearLayer, LoRACrossAttnProcessor, LoRAXFormersCrossAttnProcessor)
from diffusers.models import UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers


if TYPE_CHECKING:
    from control_lora.models.control import ControlLoRAContainer


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def parse_lora_from_layers(layers: Union[UNet2DConditionModel, AttnProcsLayers]):
    if isinstance(layers, dict):
        return layers
    if isinstance(layers, UNet2DConditionModel):
        layers = AttnProcsLayers(layers.attn_processors)

    state_dict = layers.state_dict()

    # fill attn processors
    attn_processors = {}

    is_lora = all("lora" in k for k in state_dict.keys())

    if is_lora:
        lora_grouped_dict = defaultdict(dict)
        for key, value in state_dict.items():
            attn_processor_key, sub_key = ".".join(
                key.split(".")[:-3]), ".".join(key.split(".")[-3:])
            lora_grouped_dict[attn_processor_key][sub_key] = value

        for key, value_dict in lora_grouped_dict.items():
            rank = value_dict["to_k_lora.down.weight"].shape[0]
            cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
            hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

            attn_processors[key] = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
            )
            attn_processors[key].load_state_dict(value_dict)
    else:
        raise ValueError(
            f"Input module does not seem to be in the correct format expected by LoRA training.")

    # set correct dtype & device
    attn_processors = {k: v.to(device=layers.device, dtype=layers.dtype)
                       for k, v in attn_processors.items()}

    return attn_processors


@dataclass
class LoRAContainerOutput(BaseOutput):
    output: torch.FloatTensor


class LoRAContainer(ModelMixin, ConfigMixin):
    '''
        LoRA layers wrapped like ControlLoRAContainer, 
        so we could train it in the control lora trainer.
    '''
    @register_to_config
    def __init__(
        self,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        cross_attention_dims: Tuple[List[int]] = (
            [None, 768, None, 768, None, 768, None, 768, None, 768],
            [None, 768, None, 768, None, 768, None, 768, None, 768],
            [None, 768, None, 768, None, 768, None, 768, None, 768],
            [None, 768]
        ),
        rank=4,
        encoder_only=False
    ):
        super().__init__()
        self.block_out_channels = block_out_channels
        self.cross_attention_dims = cross_attention_dims
        self.rank = rank
        self.encoder_only = encoder_only
        self._use_memory_efficient_attention_xformers = False

        self.processor_layers = nn.ModuleList()
        for block_out_channel, cross_attention_dim_list in zip(block_out_channels, cross_attention_dims):
            self.processor_layers.append(
                nn.ModuleList([
                    LoRACrossAttnProcessor(
                        block_out_channel,
                        cross_attention_dim=cross_attention_dim,
                        rank=rank)
                    for cross_attention_dim in cross_attention_dim_list
                ])
            )
        self.cached_unets: List[UNet2DConditionModel] = []

    def forward(self, x: torch.FloatTensor, return_dict: bool = True) -> Union[LoRAContainerOutput, Tuple]:
        if not return_dict:
            return (x, )
        return LoRAContainerOutput(output=x)

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        if valid == self._use_memory_efficient_attention_xformers:
            return

        device = self.device
        dtype = self.dtype
        block_out_channels = self.block_out_channels
        cross_attention_dims = self.cross_attention_dims
        rank = self.rank
        processors_state_dict = self.processor_layers.state_dict()
        processor_cls = (
            LoRAXFormersCrossAttnProcessor
            if valid
            else LoRACrossAttnProcessor)
        kwargs = dict(attention_op=attention_op) if valid else dict()

        self.processor_layers = nn.ModuleList()
        for block_out_channel, cross_attention_dim_list in zip(block_out_channels, cross_attention_dims):
            self.processor_layers.append(
                nn.ModuleList([
                    processor_cls(
                        block_out_channel,
                        cross_attention_dim=cross_attention_dim,
                        rank=rank,
                        **kwargs)
                    for cross_attention_dim in cross_attention_dim_list
                ])
            )

        self.processor_layers.to(device=device, dtype=dtype)
        self.processor_layers.load_state_dict(processors_state_dict)
        self._use_memory_efficient_attention_xformers = valid
        for unet in self.cached_unets:
            self.set_as_unet_processor(unet)

    def set_as_unet_processor(self, unet: UNet2DConditionModel):
        n_ch = len(self.block_out_channels)
        control_ids = [i for i in range(n_ch)]
        lora_attn_procs = {}
        processor_layers_list = list(
            [list(layer_list) for layer_list in self.processor_layers])
        for name in unet.attn_processors.keys():
            if name.startswith("mid_block"):
                control_id = control_ids[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                control_id = list(reversed(control_ids))[block_id]
                if self.encoder_only:
                    lora_attn_procs[name] = (
                        XFormersCrossAttnProcessor if
                        self._use_memory_efficient_attention_xformers else
                        CrossAttnProcessor
                    )()
                    continue
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                control_id = control_ids[block_id]

            processor_layers = processor_layers_list[control_id]
            if len(processor_layers) != 0:
                processor_layer = processor_layers.pop(0)
                lora_attn_procs[name] = processor_layer

        unet.set_attn_processor(lora_attn_procs)
        if unet not in self.cached_unets:
            self.cached_unets.append(unet)


class ControlLoRACrossAttnProcessor(nn.Module):
    def __init__(self, control_size, hidden_size, rank=4, concat_hidden=False):
        super().__init__()
        self.control_size = control_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.control_states: torch.Tensor = None
        self.concat_hidden = concat_hidden

        control_in = control_size + (hidden_size if concat_hidden else 0)
        self.to_control_lora = LoRALinearLayer(control_in, hidden_size, rank)

    def set_control_states(self, control_states):
        self.control_states = control_states

    def postprocess_hidden_states(self, hidden_states):
        assert self.control_states is not None

        control_states = self.control_states.to(hidden_states.dtype)
        if hidden_states.ndim == 3 and control_states.ndim == 4:
            batch, _, height, width = control_states.shape
            control_states = control_states.permute(
                0, 2, 3, 1).reshape(batch, height * width, -1)
            self.control_states = control_states

        b1, b2 = control_states.shape[0], hidden_states.shape[0]
        if b1 != b2:  # classifier free guidance
            control_states = torch.cat([control_states]*(b2 // b1))

        if self.concat_hidden:
            control_states = torch.cat([hidden_states, control_states], -1)

        return self.to_control_lora(control_states)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, skip_control=False
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states_proj = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states_proj)
        # add control
        if not skip_control:
            hidden_states = hidden_states + scale * \
                self.postprocess_hidden_states(hidden_states)

        return hidden_states


class ControlLoRAXFormersCrossAttnProcessor(nn.Module):
    def __init__(self, control_size, hidden_size, rank=4, concat_hidden=False, attention_op: Optional[Callable] = None):
        super().__init__()
        self.control_size = control_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.control_states: torch.Tensor = None
        self.concat_hidden = concat_hidden
        self.attention_op = attention_op

        control_in = control_size + (hidden_size if concat_hidden else 0)
        self.to_control_lora = LoRALinearLayer(control_in, hidden_size, rank)

    def set_control_states(self, control_states):
        self.control_states = control_states

    def postprocess_hidden_states(self, hidden_states):
        assert self.control_states is not None

        control_states = self.control_states.to(hidden_states.dtype)
        if hidden_states.ndim == 3 and control_states.ndim == 4:
            batch, _, height, width = control_states.shape
            control_states = control_states.permute(
                0, 2, 3, 1).reshape(batch, height * width, -1)
            self.control_states = control_states

        b1, b2 = control_states.shape[0], hidden_states.shape[0]
        if b1 != b2:  # classifier free guidance
            control_states = torch.cat([control_states]*(b2 // b1))

        if self.concat_hidden:
            control_states = torch.cat([hidden_states, control_states], -1)

        return self.to_control_lora(control_states)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0, skip_control=False
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states_proj = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states_proj)
        # add control
        if not skip_control:
            hidden_states = hidden_states + scale * \
                self.postprocess_hidden_states(hidden_states)

        return hidden_states


class MultiLoRACrossAttnProcessor(nn.Module):
    def __init__(
            self,
            lora_layers: Union[List[LoRACrossAttnProcessor],
                               List[LoRAXFormersCrossAttnProcessor]] = [],
            lora_scales: List[Union[float, torch.Tensor]] = [],
            control_lora_layers: Union[List[ControlLoRACrossAttnProcessor],
                                       List[ControlLoRAXFormersCrossAttnProcessor]] = [],
            control_lora_scales: List[Union[float, torch.Tensor]] = []):
        super().__init__()

        assert len(lora_layers) == len(lora_scales)
        assert len(control_lora_layers) == len(control_lora_scales)

        self.lora_layers = nn.ModuleList(lora_layers)
        self.lora_scales = lora_scales
        self.control_lora_layers = nn.ModuleList(control_lora_layers)
        self.control_lora_scales = control_lora_scales
        self.encoder_hidden_states: List[torch.Tensor] = [
            None] * len(self.lora_layers)
        self._use_memory_efficient_attention_xformers = False
        self.cached_unets: List[UNet2DConditionModel] = []

    def set_lora_scales(self, lora_scales: List[Union[float, torch.Tensor]]):
        self.lora_scales = lora_scales

    def set_control_lora_scales(self, lora_scales: List[Union[float, torch.Tensor]]):
        self.control_lora_scales = lora_scales

    def set_encoder_hidden_states(self, encoder_hidden_states: List[torch.Tensor] = []):
        self.encoder_hidden_states = encoder_hidden_states

    def resize_lora_scale(self, hidden_states, lora_scale):
        if isinstance(lora_scale, float):
            return hidden_states

        lora_scale = lora_scale.to(hidden_states.dtype)
        if hidden_states.ndim == 3 and lora_scale.ndim == 4:
            batch, _, height, width = lora_scale.shape
            scale = sqrt(height * width // hidden_states.shape[1])
            lora_scale = F.interpolate(
                lora_scale, scale_factor=1 / scale, mode='bilinear', align_corners=False)

            lora_scale = lora_scale.permute(
                0, 2, 3, 1).reshape(batch, height * width, -1)

        return lora_scale

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0
    ):
        lora_layer: Union[LoRACrossAttnProcessor,
                          LoRAXFormersCrossAttnProcessor]
        control_lora_layer: Union[ControlLoRACrossAttnProcessor,
                                  ControlLoRAXFormersCrossAttnProcessor]

        hidden_states_list = []
        for lora_layer, lora_scale, cached_encoder_hidden_states in zip(
                self.lora_layers, self.lora_scales, self.encoder_hidden_states):
            cross_hidden_states = encoder_hidden_states
            if lora_layer.cross_attention_dim is not None and cached_encoder_hidden_states is not None:
                cross_hidden_states = cached_encoder_hidden_states
            hidden_states_out = self.resize_lora_scale(hidden_states, lora_scale) * lora_layer(
                attn, hidden_states, cross_hidden_states, attention_mask, scale)
            hidden_states_list.append(hidden_states_out)

        if len(self.lora_layers) != 0:
            hidden_states = sum(hidden_states_list)
        elif len(self.control_lora_layers) > 0:
            hidden_states = self.control_lora_layers[0](
                attn, hidden_states, encoder_hidden_states, attention_mask, scale, skip_control=True)

        # add control
        hidden_states_list = [0]
        for control_lora_layer, control_lora_scale in zip(
                self.control_lora_layers, self.control_lora_scales):
            hidden_states_out = self.resize_lora_scale(
                hidden_states, control_lora_scale) * control_lora_layer.postprocess_hidden_states(hidden_states)
            hidden_states_list.append(hidden_states_out)
        hidden_states = hidden_states + sum(hidden_states_list)

        return hidden_states

    @classmethod
    def set_as_unet_processor(
            cls,
            unet: UNet2DConditionModel,
            lora_layers: Union[List[AttnProcsLayers],
                               List[Dict[str, LoRACrossAttnProcessor]]] = [],
            lora_scales: List[Union[float, torch.Tensor]] = [],
            control_lora_layers: List['ControlLoRAContainer'] = [],
            control_lora_scales: List[Union[float, torch.Tensor]] = []):
        lora_layers = [parse_lora_from_layers(
            layers) for layers in lora_layers]
        n_ch = len(unet.config.block_out_channels)
        control_ids = [i for i in range(n_ch)]
        lora_attn_procs = {}
        control_layers_list = [
            list(
                [list(layer_list) for layer_list in processor_layers])
            for processor_layers in control_lora_layers]
        for name in unet.attn_processors.keys():
            if name.startswith("mid_block"):
                control_id = control_ids[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                control_id = list(reversed(control_ids))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                control_id = control_ids[block_id]

            lora_layer_list = [layer[name] for layer in lora_layers]
            control_lora_layer_list = []
            for processor_layers_list in control_layers_list:
                processor_layers = processor_layers_list[control_id]
                if len(processor_layers) != 0:
                    processor_layer = processor_layers.pop(0)
                    control_lora_layer_list.append(processor_layer)
            lora_attn_procs[name] = cls(
                lora_layer_list,
                lora_scales,
                control_lora_layer_list,
                control_lora_scales)

        unet.set_attn_processor(lora_attn_procs)

        return lora_attn_procs
