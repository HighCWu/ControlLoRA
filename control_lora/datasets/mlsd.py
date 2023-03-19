
import os
import cv2
import glob
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from control_lora.datasets.base import BaseDataset
from control_lora.datasets.annotator.util import resize_image, HWC3
from control_lora.datasets.annotator.mlsd import MLSDdetector


class MLSDDataset(BaseDataset):
    def __init__(self, seeds=[233], zfill=16, **kwargs):
        super().__init__(**kwargs)

        self.seeds = seeds
        self.zfill = zfill

        for seed in seeds:
            guide_dir = os.path.join(self.root_dir, self.base_name + '-mlsd-seed-' + str(seed))
            if not os.path.exists(guide_dir) or len(glob.glob(guide_dir + '/*.png')) != len(self.data):
                os.makedirs(guide_dir, exist_ok=True)

                apply_mlsd = MLSDdetector()
                finished = len(glob.glob(guide_dir + '/*.png'))
                rng = np.random.RandomState(seed)
                print('Generate dataset of MLSD guide.')
                for i in tqdm(range(len(self.data))):
                    thr_v, thr_d = rng.uniform(0.01, 2.0), rng.uniform(0.01, 20.0)
                    if i < finished:
                        continue
                    image = self.item_image(i)
                    h, w = image.shape[:2]
                    if min(h, w) != self.resolution:
                        ratio = self.resolution / min(h, w)
                        h, w = int(round(h * ratio)), int(round(w * ratio))
                        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                    ori_h, ori_w = image.shape[:2]
                    detected_map = apply_mlsd(resize_image(image, self.resolution), thr_v, thr_d)
                    detected_map = HWC3(detected_map)
                    detected_map = cv2.resize(detected_map, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
                    guide = Image.fromarray(detected_map)
                    guide.save(os.path.join(guide_dir, str(i).zfill(zfill)) + '.png')

    def item_guide(self, index: int) -> np.ndarray:
        seed_index = torch.randint(0, len(self.seeds), (1, )).item()
        seed = self.seeds[seed_index]
        guide_dir = os.path.join(self.root_dir, self.base_name + '-mlsd-seed-' + str(seed))
        guide_path = os.path.join(guide_dir, str(index).zfill(self.zfill)) + '.png'
        guide = Image.open(guide_path).convert('RGB')

        return np.asarray(guide)
