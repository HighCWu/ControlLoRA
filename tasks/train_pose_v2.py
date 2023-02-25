import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("girls are playing with a frisbee in a field, "
"2009 cinematography, trending on artforum, running pose, "
"bruce springsteen, connected to heart machines, with tattoos, "
"beautiful - n 9, by Eric Dinyer, young child, midlands")

cmd = rf'''accelerate launch --mixed_precision="fp16" train_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="process/mpii_pose" --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=8 --checkpointing_steps=5000 --resume_from_checkpoint="latest" \
  --learning_rate=1e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 --validation_epochs=1 \
  --output_dir="ckpts/sd-mpii-pose-v2-model-control-lora" \
  --control_lora_config="configs/mpii-pose-v2.json" \
  --validation_prompt="{validation_prompt}" --report_to="wandb"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
