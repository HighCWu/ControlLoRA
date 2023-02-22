import os
import cv2
import torch
import numpy as np

from PIL import Image
from process import base
from datasets.load import load_dataset


class Dataset(base.Dataset):
    def __init__(self, tokenizer, resolution=512, use_crop=True, **kwargs):
        self.tokenizer = tokenizer

        self.dataset = load_dataset('poloclub/diffusiondb', '2m_random_1k')['train']
        self.size = resolution
        self.use_crop = use_crop

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        img = item['image']

        if self.use_crop:
            w, h = img.size
            x0 = torch.randint(0, w - self.size, (1, )).item() if w > self.size else 0
            y0 = torch.randint(0, h - self.size, (1, )).item() if h > self.size else 0
            x1 = x0 + self.size
            y1 = y0 + self.size
            img = img.crop((x0,y0,x1,y1))
            
        w, h = img.size
        img = np.asarray(img)

        low_threshold = torch.randint(1, 255, (1, )).item()
        high_threshold = torch.randint(1, 255, (1, )).item()
        guide_img = cv2.Canny(img, low_threshold, high_threshold)

        guide_npy_img = np.asarray(guide_img).astype('float32') / 127.5 - 1

        npy_img = img.astype('float32')
        img = torch.tensor(npy_img.copy()).float().permute(2,0,1) / 127.5 - 1
        guide_img = torch.tensor(guide_npy_img.copy())[None].repeat(3,1,1).float()
        input_ids = self.tokenizer({"text": [item["prompt"]]})[0]

        return { "pixel_values": img, "guide_values": guide_img, "input_ids": input_ids }

    @staticmethod
    def control_channel():
        return 3

    @staticmethod
    def cat_input(image: Image.Image, target: torch.Tensor, guide: torch.Tensor):
        target = np.uint8(((target + 1) * 127.5)[0].permute(1,2,0).cpu().numpy().clip(0,255))
        guide = np.uint8(((guide + 1) * 127.5)[0].permute(1,2,0).cpu().numpy().clip(0,255))
        target = Image.fromarray(target).convert('RGB').resize(image.size)
        guide = Image.fromarray(guide).convert('RGB').resize(image.size)
        image_cat = Image.new('RGB', (image.size[0]*3,image.size[1]), (0,0,0))
        image_cat.paste(target,(0,0))
        image_cat.paste(guide,(image.size[0],0))
        image_cat.paste(image,(image.size[0]*2, 0))

        return image_cat

Dataset.register_cls('diffusiondb_canny')
