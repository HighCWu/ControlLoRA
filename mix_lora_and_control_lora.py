from diffusers import utils
from diffusers.utils import deprecation_utils
from diffusers.models import cross_attention
utils.deprecate = lambda *arg, **kwargs: None
deprecation_utils.deprecate = lambda *arg, **kwargs: None
cross_attention.deprecate = lambda *arg, **kwargs: None

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from datasets.load import load_dataset
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from models import LoRACrossAttnProcessor, ControlLoRA, ControlLoRACrossAttnProcessor
from huggingface_hub import hf_hub_url, hf_hub_download

@torch.inference_mode()
def main():
    # args = 
    resolution = 512
    dataset_name = "HighCWu/diffusiondb_2m_first_5k_canny" # "HighCWu/mpii_100_openpose" 
    output_dir = "mix"
    validation_prompt = ("portrait of male HighCWu")
    num_validation_images = 16
    inject_pre_lora = True
    inject_post_lora = False
    
    dataset = load_dataset(dataset_name)

    data_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    torch.manual_seed(0)
    
    def preprocess_train(examples):
        guides = []
        for guide in examples["guide"]:
            guide = guide.convert("RGB")
            guide = data_transforms(guide)
            c, h, w = guide.shape
            y1, x1 = 0, 0
            if h != resolution:
                y1 = torch.randint(0, h - resolution, (1, )).item()
            elif w != resolution:
                x1 = torch.randint(0, w - resolution, (1, )).item()
            y2, x2 = y1 + resolution, x1 + resolution
            guide = guide[:,y1:y2,x1:x2]
            guides.append(guide)
        examples["guide_values"] = guides
        return examples
    
    def collate_fn(examples):
        guide_values = torch.stack([example["guide_values"] for example in examples])
        guide_values = guide_values.to(memory_format=torch.contiguous_format).float()
        return {"guide_values": guide_values}

    # Set the training transforms
    dataset = dataset["train"].with_transform(preprocess_train)
    val_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=0,
    )
    val_iter = iter(val_dataloader)

    pipeline = DiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    unet: UNet2DConditionModel = pipeline.unet

    control_lora = ControlLoRA.from_pretrained(
        "HighCWu/ControlLoRA",
        subfolder="sd-diffusiondb-canny-model-control-lora"
        # subfolder="sd-mpii-pose-model-control-lora"
    )
    control_lora = control_lora.to('cuda')

    n_ch = len(unet.config.block_out_channels)
    control_ids = [i for i in range(n_ch)]

    # load attention processors
    lora_attn_procs = {}
    lora_layers_list = list([list(layer_list) for layer_list in control_lora.lora_layers])
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            control_id = control_ids[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            control_id = list(reversed(control_ids))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            control_id = control_ids[block_id]

        lora_layers = lora_layers_list[control_id]
        if len(lora_layers) != 0:
            lora_layer: ControlLoRACrossAttnProcessor = lora_layers.pop(0)
            other_lora = LoRACrossAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            ).cuda()
            lora_attn_procs[name] = other_lora
            if inject_pre_lora:
                lora_layer.inject_pre_lora(other_lora)
            if inject_post_lora:
                lora_layer.inject_post_lora(other_lora)

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_path = hf_hub_download(
        "HighCWu/ControlLoRA", 
        "diffusion_pytorch_model.bin",
        subfolder="sd-highcwu_v1-model-lora"
    )
    lora_layers.load_state_dict(torch.load(lora_path))
    
    # load control lora attention processors
    lora_attn_procs = {}
    lora_layers_list = list([list(layer_list) for layer_list in control_lora.lora_layers])
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

    generator = torch.Generator(device='cuda').manual_seed(0)
    os.makedirs(os.path.join("samples", output_dir), exist_ok=True)
    for i in range(num_validation_images):
        with torch.no_grad():
            try:
                batch = next(val_iter)
            except:
                val_iter = iter(val_dataloader)
                batch = next(val_iter)
            guide = batch["guide_values"].to('cuda')
            _ = control_lora(guide).control_states
            image = pipeline(validation_prompt, num_inference_steps=30, generator=generator).images[0]
            guide = np.uint8(((guide + 1) * 127.5)[0].permute(1,2,0).cpu().numpy())
            guide = Image.fromarray(guide).convert('RGB').resize(image.size)
            image_cat = Image.new('RGB', (image.size[0]*2,image.size[1]), (0,0,0))
            image_cat.paste(guide,(0,0))
            image_cat.paste(image,(image.size[0], 0))
            image = image_cat
            image.save(os.path.join("samples", output_dir, f"{i}.png"))


if __name__ == '__main__':
    main()