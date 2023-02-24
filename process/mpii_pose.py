import os
import torch
import jsonlines
import numpy as np

from PIL import Image
from process import base


class Dataset(base.Dataset):
    def __init__(self, tokenizer, resolution=512, use_crop=True, **kwargs):
        self.tokenizer = tokenizer

        with jsonlines.open("data/mpii/prompt.jsonl", 'r') as f:
            self.items = [item for item in f]
        self.size = resolution
        self.use_crop = use_crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img_file = os.path.join('./data/mpii', item['image'])
        guide_file = os.path.join('./data/mpii', item['guide'])
        img = Image.open(img_file).convert('RGB')
        guide_img = Image.open(guide_file).convert('RGB')

        if self.use_crop:
            w, h = img.size
            x0 = torch.randint(0, w - self.size, (1, )).item() if w > self.size else 0
            y0 = torch.randint(0, h - self.size, (1, )).item() if h > self.size else 0
            x1 = x0 + self.size
            y1 = y0 + self.size
            img = img.crop((x0,y0,x1,y1))
            guide_img = guide_img.crop((x0,y0,x1,y1))

        w, h = img.size

        guide_npy_img = np.asarray(guide_img).astype('float32') / 127.5 - 1

        npy_img = np.asarray(img).astype('float32')
        img = torch.tensor(npy_img.copy()).float().permute(2,0,1) / 127.5 - 1
        guide_img = torch.tensor(guide_npy_img.copy()).permute(2,0,1)
        input_ids = self.tokenizer({"text": [item["text"]]})[0]

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

Dataset.register_cls('mpii_pose')
