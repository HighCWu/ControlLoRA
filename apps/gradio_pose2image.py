from diffusers import utils
from diffusers.utils import deprecation_utils
from diffusers.models import cross_attention
utils.deprecate = lambda *arg, **kwargs: None
deprecation_utils.deprecate = lambda *arg, **kwargs: None
cross_attention.deprecate = lambda *arg, **kwargs: None

import os
import sys
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MAIN_DIR)
os.chdir(MAIN_DIR)


import cv2
import gradio as gr
import numpy as np
import torch
import random

from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from models import ControlLoRA, ControlLoRACrossAttnProcessor


apply_openpose = OpenposeDetector()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipeline = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', safety_checker=None
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)
unet: UNet2DConditionModel = pipeline.unet

control_lora = ControlLoRA.from_pretrained('HighCWu/ControlLoRA', subfolder="sd-mpii-pose-model-control-lora")
control_lora = control_lora.to(device)


# load control lora attention processors
lora_attn_procs = {}
lora_layers_list = list([list(layer_list) for layer_list in control_lora.lora_layers])
n_ch = len(unet.config.block_out_channels)
control_ids = [i for i in range(n_ch)]
for name in pipeline.unet.attn_processors.keys():
    cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    if name.startswith("mid_block"):
        control_id = control_ids[-1]
    elif name.startswith("up_blocks"):
        block_id = int(name[len("up_blocks.")])
        control_id = list(reversed(control_ids))[block_id]
    elif name.startswith("down_blocks"):
        block_id = int(name[len("down_blocks.")])
        control_id = control_ids[block_id]

    lora_layers = lora_layers_list[control_id]
    if len(lora_layers) != 0:
        lora_layer: ControlLoRACrossAttnProcessor = lora_layers.pop(0)
        lora_attn_procs[name] = lora_layer

unet.set_attn_processor(lora_attn_procs)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, sample_steps, scale, seed, eta):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map[...,::-1].copy().transpose([2,0,1])).float().to(device)[None] / 127.5 - 1
        _ = control_lora(control).control_states

        if seed == -1:
            seed = random.randint(0, 65535)

        # run inference
        generator = torch.Generator(device=device).manual_seed(seed)
        images = []
        for i in range(num_samples):
            _ = control_lora(control).control_states
            image = pipeline(
                prompt + ', ' + a_prompt, negative_prompt=n_prompt, 
                num_inference_steps=sample_steps, guidance_scale=scale, eta=eta,
                generator=generator, height=H, width=W).images[0]
            images.append(np.asarray(image))

        results = images
    return [detected_map] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Human Pose")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=256)
                detect_resolution = gr.Slider(label="OpenPose Resolution", minimum=128, maximum=1024, value=512, step=1)
                sample_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=30, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, sample_steps, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0')
