import torch

from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import is_xformers_available
from diffusers.configuration_utils import ConfigMixin
from diffusers.models import ModelMixin
from control_lora.models import ControlLoRAContainer
from control_lora.pipelines import StableDiffusionControlLoRAPipeline


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)


def hook_control_lora(self: ControlLoRAContainer, inputs, results):
    print('ControlLoRa is being called.')

    # processor override the __call__ function,so it would not trigger
    # `forward_hook`` after `forward`,
    # so we call it in the control_lora forward hook.
    hook_control_processor(self.processor_layers[0][0], None, None)


def hook_control_processor(self, inputs, results):
    print(f'{type(self).__name__} is being called')


control_lora = ControlLoRAContainer(control_num_processors=(4, 4, 4, 2), encoder_only=True)
control_lora.register_forward_hook(hook_control_lora)
control_lora.to_json_file('tmp/base.json')
control_lora.save_pretrained('tmp/control_lora')
control_lora.save_pretrained('tmp/control_lora')
StableDiffusionControlLoRAPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    control_lora_pretrained_model_name_or_path='tmp/control_lora'
)
pipeline = StableDiffusionControlLoRAPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    control_lora=control_lora,
    torch_dtype=torch.float16
)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(device)
if is_xformers_available():
    pipeline.set_use_memory_efficient_attention_xformers(True)
generator = torch.Generator(device=device).manual_seed(233)
images = []
with torch.no_grad():
    guide = Image.new('RGB', (512,512), (0,0,0))
    guide = [[guide], [guide]]
    images = pipeline(
        ["Potrait of 1 beautiful girl, 8 k, ray tracing", "Potrait of 3 beautiful girls, 8 k, ray tracing"], 
        num_inference_steps=30, 
        generator=generator, 
        control_image=guide).images
    for i, img in enumerate(images):
        img.save(f'tmp/{i}.png')
