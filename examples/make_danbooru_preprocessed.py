import os
import cv2
import glob
import math
import torch
import datasets
import scipy.io
import jsonlines
import numpy as np

from tqdm import tqdm
from PIL import Image
from clip_interrogator import Config, Interrogator
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


MAIN_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')

img_dir = f'{MAIN_DIR}/data/danbooru-2020-512'

jlname = f"{MAIN_DIR}/data/danbooru-2020-512.jsonl"
def make_prompt():
    imgs = sorted(glob.glob(f'{MAIN_DIR}/data/danbooru-2020-512/**/*.jpg', recursive=True) + \
                  glob.glob(f'{MAIN_DIR}/data/danbooru-2020-512/**/*.png', recursive=True))[:20000]
    if not os.path.exists(jlname + '.done'):
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        with jsonlines.open(jlname, 'w', flush=True) as w:
            for img_path in tqdm(imgs):
                with torch.no_grad():
                    image = Image.open(img_path).convert('RGB')
                    prompt = ci.interrogate_fast(image)
                    w.write({
                        "image": img_path, 
                        "text": prompt}
                    )
        with open(jlname + '.done', 'w', encoding='utf-8') as f:
            f.write('Done.')

make_prompt()

for name in ['erika', 'illyasviel', 'infor', 'muko']:
    with jsonlines.open(jlname, 'r') as r:
        with jsonlines.open(jlname.replace('/data/danbooru-2020-512', f'/data/danbooru-2020-512-{name}'), 'w', flush=True) as w:
            for item in r.iter():
                w.write({
                    "image": item["image"].replace('/data/danbooru-2020-512', f'/data/danbooru-2020-512-{name}')
                })
