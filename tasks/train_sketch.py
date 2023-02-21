import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("1girl")

cmd = rf'''accelerate launch --mixed_precision="fp16" train_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="ckpt/anything-v3-vae-swapped" \
  --dataset_name="process/danbooru_sketch" --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=6 --checkpointing_steps=5000 --resume_from_checkpoint="latest" \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="ckpts/sd-danbooru-sketch-model-control-lora" \
  --control_lora_config="configs/danbooru-sketch.json" \
  --validation_prompt="{validation_prompt}" --report_to="wandb"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
