import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

cmd = r'''accelerate launch --mixed_precision="fp16" test_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="HighCWu/fill50k" --caption_column="text" \
  --resolution=512 \
  --num_train_epochs=100 --resume_from_checkpoint="latest" \
  --lr_scheduler="constant" \
  --seed=42 \
  --output_dir="ckpts/sd-fill50k-model-control-lora" \
  --control_lora_config="configs/fill50k.json" \
  --validation_prompt="pale red rod circle with old green background"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
