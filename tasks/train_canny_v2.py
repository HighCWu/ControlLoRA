import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("1girl, 8 k, unreal engine")

cmd = rf'''accelerate launch --mixed_precision="fp16" train_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="process/diffusiondb_canny" --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=30 --checkpointing_steps=5000 --resume_from_checkpoint="latest" \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 --validation_epochs=5 \
  --output_dir="ckpts/sd-diffusiondb-canny-v2-model-control-lora" \
  --control_lora_config="configs/diffusiondb-canny-v2.json" \
  --validation_prompt="{validation_prompt}" --report_to="wandb"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
