import os

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

validation_prompt = ("portrait of a dancing eagle woman, "
"beautiful blonde haired lakota sioux goddess, "
"intricate, highly detailed art by james jean, "
"ray tracing, digital painting, artstation, "
"concept art, smooth, sharp focus, illustration, "
"artgerm and greg rutkowski and alphonse mucha, "
"vladimir kush, giger, roger dean, 8 k")

cmd = rf'''accelerate launch --mixed_precision="fp16" test_text_to_image_control_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --dataset_name="HighCWu/diffusiondb_2m_first_5k_canny" --caption_column="text" \
  --resolution=512 \
  --num_train_epochs=100 --resume_from_checkpoint="latest" \
  --lr_scheduler="constant" \
  --seed=42 \
  --output_dir="ckpts/sd-diffusiondb-canny-model-control-lora" \
  --control_lora_config="configs/diffusiondb-canny.json" \
  --validation_prompt="{validation_prompt}"'''

os.system(cmd.replace('\\', ' ').replace('\r\n', '\n').replace('\n', ''))
