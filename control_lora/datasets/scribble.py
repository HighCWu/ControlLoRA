
import os
import cv2
import glob
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from control_lora.datasets.base import BaseDataset
from control_lora.datasets.hed import HEDDataset
from control_lora.datasets.annotator.util import resize_image, HWC3
from control_lora.datasets.annotator.hed import nms


class ScribbleDataset(BaseDataset):
    def __init__(self, zfill=16, **kwargs):
        super().__init__(**kwargs)

        self.zfill = zfill

        guide_dir = os.path.join(self.root_dir, self.base_name + '-scribble')
        if not os.path.exists(guide_dir) or len(glob.glob(guide_dir + '/*.png')) != len(self.data):
            os.makedirs(guide_dir, exist_ok=True)

            hed_dataset = HEDDataset(zfill=zfill, **kwargs)
            finished = len(glob.glob(guide_dir + '/*.png'))
            print('Generate dataset of Scribble guide.')
            for i in tqdm(range(len(self.data))):
                if i < finished:
                    continue
                image = hed_dataset.item_guide(i)
                h, w = image.shape[:2]
                if min(h, w) != self.resolution:
                    ratio = self.resolution / min(h, w)
                    h, w = int(round(h * ratio)), int(round(w * ratio))
                    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                ori_h, ori_w = image.shape[:2]
                image = resize_image(image, self.resolution)
                detected_map = nms(image, 127, 3.0)
                detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
                detected_map[detected_map > 4] = 255
                detected_map[detected_map < 255] = 0
                detected_map = HWC3(detected_map)
                detected_map = cv2.resize(detected_map, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
                guide = Image.fromarray(detected_map)
                guide.save(os.path.join(guide_dir, str(i).zfill(zfill)) + '.png')

    def item_guide(self, index: int) -> np.ndarray:
        guide_dir = os.path.join(self.root_dir, self.base_name + '-scribble')
        guide_path = os.path.join(guide_dir, str(index).zfill(self.zfill)) + '.png'
        guide = Image.open(guide_path).convert('RGB')

        return np.asarray(guide)
