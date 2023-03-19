import torch
import torch.nn as nn

from typing import Callable, List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers.utils.outputs import BaseOutput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models import UNet2DConditionModel
from diffusers.models.cross_attention import CrossAttnProcessor, XFormersCrossAttnProcessor
from control_lora.models.block import get_down_block
from control_lora.models.lora import ControlLoRACrossAttnProcessor, ControlLoRAXFormersCrossAttnProcessor


@dataclass
class ControlLoRAOutput(BaseOutput):
    output: Tuple[List[torch.FloatTensor]]


class ControlLoRAContainer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str] = (
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
        ),
        down_block_out_channels: Tuple[int] = (32, 64, 128, 128),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        post_down_block_types: Tuple[str] = (
            None,
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
        ),
        post_layers_per_block: int = 1,
        pre_control_types: Tuple[str] = (
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
            "SimpleDownEncoderBlock2D",
        ),
        pre_control_per_processor: int = None,
        pre_control_kernel_size: int = 1,
        control_num_processors: Tuple[int] = (10, 10, 10, 2),
        control_sizes: Tuple[int] = (128, 128, 128, 128),
        control_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        rank: int = 4,
        concat_hidden: bool = False,
        encoder_only: bool = False
    ):
        super().__init__()

        assert control_sizes[0] == down_block_out_channels[-1]

        self.layers_per_block = layers_per_block
        self.post_layers_per_block = post_layers_per_block
        self.pre_control_per_processor = pre_control_per_processor
        self.control_num_processors = control_num_processors
        self.control_sizes = control_sizes
        self.control_out_channels = control_out_channels
        self.rank = rank
        self.concat_hidden = concat_hidden
        self.encoder_only = encoder_only
        self._use_memory_efficient_attention_xformers = False

        self.conv_in = torch.nn.Conv2d(
            in_channels, down_block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        self.pre_control_layers = nn.ModuleList([])
        self.processor_layers = nn.ModuleList([])

        # down
        down_blocks = []
        output_channel = down_block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = down_block_out_channels[i]
            is_final_block = i == len(down_block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            down_blocks.append(down_block)

        self.down_blocks.append(nn.Sequential(*down_blocks))
        self.pre_control_layers.append(
            nn.ModuleList([
                get_down_block(
                    pre_control_types[0],
                    num_layers=self.pre_control_per_processor,
                    in_channels=control_sizes[0],
                    out_channels=control_sizes[0],
                    add_downsample=False,
                    resnet_eps=1e-6,
                    downsample_padding=0,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attn_num_head_channels=None,
                    temb_channels=None,
                    resnet_kernel_size=pre_control_kernel_size,
                ) if self.pre_control_per_processor is not None else nn.Identity()
                for _ in range(control_num_processors[0])
            ])
        )
        self.processor_layers.append(
            nn.ModuleList([
                ControlLoRACrossAttnProcessor(
                    control_sizes[0],
                    control_out_channels[0],
                    rank=rank,
                    concat_hidden=concat_hidden)
                for _ in range(control_num_processors[0])
            ])
        )

        # post down
        output_channel = control_sizes[0]
        for i, down_block_type in enumerate(post_down_block_types):
            if i == 0:
                continue
            input_channel = output_channel
            output_channel = control_sizes[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=self.post_layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=True,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

            self.pre_control_layers.append(
                nn.ModuleList([
                    get_down_block(
                        pre_control_types[i],
                        num_layers=self.pre_control_per_processor,
                        in_channels=control_sizes[i],
                        out_channels=control_sizes[i],
                        add_downsample=False,
                        resnet_eps=1e-6,
                        downsample_padding=0,
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        attn_num_head_channels=None,
                        temb_channels=None,
                        resnet_kernel_size=pre_control_kernel_size,
                    ) if self.pre_control_per_processor is not None else nn.Identity()
                    for _ in range(control_num_processors[i])
                ])
            )

            self.processor_layers.append(
                nn.ModuleList([
                    ControlLoRACrossAttnProcessor(
                        control_sizes[i],
                        control_out_channels[i],
                        rank=rank,
                        concat_hidden=concat_hidden)
                    for _ in range(control_num_processors[i])
                ])
            )

        self.cached_unets: List[UNet2DConditionModel] = []

    def forward(self, x: torch.FloatTensor, return_dict: bool = True) -> Union[ControlLoRAOutput, Tuple]:
        h = x.to(self.conv_in.weight)
        h = self.conv_in(h)
        control_states_list = []

        # down
        for down_block, pre_control_list, processor_layer_list in zip(
                self.down_blocks, self.pre_control_layers, self.processor_layers):
            h = down_block(h)
            if isinstance(h, tuple):
                h = h[0]

            control_states = []
            pre_control_layer: nn.Module
            processor_layer: ControlLoRACrossAttnProcessor
            for pre_control_layer, processor_layer in zip(
                    pre_control_list, processor_layer_list):
                h2 = pre_control_layer(h)
                processor_layer.set_control_states(h2)
                control_states.append(h2)
            control_states_list.append(control_states)

        control_states = tuple(control_states_list)

        if not return_dict:
            return control_states

        return ControlLoRAOutput(output=control_states)

    def set_use_memory_efficient_attention_xformers(
        self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:
        if valid == self._use_memory_efficient_attention_xformers:
            return

        device = self.device
        dtype = self.dtype
        control_num_processors = self.control_num_processors
        control_sizes = self.control_sizes
        control_out_channels = self.control_out_channels
        rank = self.rank
        concat_hidden = self.concat_hidden
        processors_state_dict = self.processor_layers.state_dict()
        processor_cls = (
            ControlLoRAXFormersCrossAttnProcessor
            if valid
            else ControlLoRACrossAttnProcessor)
        kwargs = dict(attention_op=attention_op) if valid else dict()

        self.processor_layers = nn.ModuleList()
        for control_num_processor, control_size, control_out_channel in zip(
            control_num_processors, control_sizes, control_out_channels
        ):
            self.processor_layers.append(
                nn.ModuleList([
                    processor_cls(
                        control_size,
                        control_out_channel,
                        rank=rank,
                        concat_hidden=concat_hidden,
                        **kwargs)
                    for _ in range(control_num_processor)
                ])
            )

        self.processor_layers.to(device=device, dtype=dtype)
        self.processor_layers.load_state_dict(processors_state_dict)
        self._use_memory_efficient_attention_xformers = valid
        for unet in self.cached_unets:
            self.set_as_unet_processor(unet)

    def set_as_unet_processor(self, unet: UNet2DConditionModel):
        n_ch = len(self.control_out_channels)
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
