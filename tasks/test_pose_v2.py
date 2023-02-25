import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("girls are playing with a frisbee in a field, "
"2009 cinematography, trending on artforum, running pose, "
"bruce springsteen, connected to heart machines, with tattoos, "
"beautiful - n 9, by Eric Dinyer, young child, midlands")

cmd = rf'''accelerate launch --mixed_precision="fp16" test_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="ckpt/anything-v3-vae-swapped" \
  --dataset_name="process/mpii_pose" --caption_column="text" \
  --resolution=512 \
  --num_train_epochs=2 --resume_from_checkpoint="latest" \
  --lr_scheduler="constant" \
  --seed=42 \
  --num_validation_images 100 \
  --output_dir="ckpts/sd-mpii-pose-v2-model-control-lora" \
  --control_lora_config="configs/mpii-pose-v2.json" \
  --validation_prompt="{validation_prompt}"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
