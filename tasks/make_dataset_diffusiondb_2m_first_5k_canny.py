
import os
import cv2
import random
import datasets

from PIL import Image, ImageFilter
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets.load import load_dataset

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

os.makedirs('HighCWu', exist_ok=True)
os.makedirs(os.path.join('data', 'diffusiondb_2m_first_5k_canny'), exist_ok=True)

random.seed(42)

def map_fn(kwargs):
    def _map_fn(image, prompt, **kwargs):
        image = image["path"]
        basename = os.path.basename(os.path.dirname(image)) + '_' + os.path.basename(image)
        savepath = os.path.join('data', 'diffusiondb_2m_first_5k_canny', basename)
        low_threshold = random.randint(1, 10)
        high_threshold = random.randint(130, 150)
        if not os.path.exists(savepath):
            cv2.imwrite(savepath, cv2.Canny(cv2.imread(image), low_threshold, high_threshold))
        guide = savepath
        text = prompt
        return dict(image=image, guide=guide, text=text)
    return _map_fn(**kwargs)

diffusiondb_dataset: Dataset
diffusiondb_dataset = load_dataset('poloclub/diffusiondb', '2m_first_5k')['train']
diffusiondb_dataset = diffusiondb_dataset.cast_column("image", datasets.Image(decode=False))
diffusiondb_dataset = diffusiondb_dataset.map(map_fn)
columns_to_remove = [f for f in diffusiondb_dataset.features if f not in ['image', 'guide', 'text']]
diffusiondb_dataset = diffusiondb_dataset.remove_columns(columns_to_remove)
diffusiondb_dataset = diffusiondb_dataset.cast_column("image", datasets.Image(decode=True))
diffusiondb_dataset = diffusiondb_dataset.cast_column("guide", datasets.Image(decode=True))
diffusiondb_dataset = DatasetDict(train=diffusiondb_dataset)
diffusiondb_dataset.save_to_disk("HighCWu/diffusiondb_2m_first_5k_canny")
