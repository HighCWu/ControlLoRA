import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

INSTANCE_DIR = os.path.join('HighCWu', 'highcwu_v1').replace('\\', '/')

validation_prompt = ("portrait of female HighCWu as a cute pink hair girl")

cmd = rf'''accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
  --instance_data_dir="{INSTANCE_DIR}" \
  --output_dir="ckpts/sd-highcwu_v1-model-lora" \
  --instance_prompt="portrait of male HighCWu" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 --resume_from_checkpoint="latest"\
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="{validation_prompt}" \
  --validation_epochs=50 \
  --seed="0" '''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
