import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("boys are playing with a frisbee in a field, "
"2009 cinematography, trending on artforum, running pose, "
"bruce springsteen, connected to heart machines, with tattoos, "
"beautiful - n 9, by Eric Dinyer, young child, midlands")

cmd = rf'''accelerate launch --mixed_precision="fp16" test_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="HighCWu/mpii_100_openpose" --caption_column="text" \
  --resolution=512 \
  --num_train_epochs=300 --resume_from_checkpoint="latest" \
  --lr_scheduler="constant" \
  --seed=42 \
  --output_dir="ckpts/sd-mpii-pose-model-control-lora" \
  --control_lora_config="configs/mpii-pose.json" \
  --validation_prompt="{validation_prompt}"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
