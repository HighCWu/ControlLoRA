import os
import torch
import random
import jsonlines
import numpy as np

from PIL import Image
from process import base


class Dataset(base.Dataset):
    def __init__(self, tokenizer, resolution=512, use_crop=True, **kwargs):
        self.tokenizer = tokenizer

        with open('./data/danbooru-2020-512-prompt.jsonl',"r+")as f:
            self.items = [item for item in jsonlines.Reader(f)]
        self.sketch_styles = [
            'illyasviel', 
            'muko',
            'erika', 
            'infor'
        ]
        self.size = resolution
        self.use_crop = use_crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        img_file = os.path.join('./data', item['image'])
        guide_style = random.choice(self.sketch_styles)
        guide_img_file = img_file.replace('danbooru-2020-512', 'danbooru-2020-512-' + guide_style)
        img = Image.open(img_file).convert('RGB')
        guide_img = Image.open(guide_img_file).convert('L')
        if self.use_crop:
            w, h = img.size
            x0 = random.randint(0, w - self.size) if w > self.size else 0
            y0 = random.randint(0, h - self.size) if h > self.size else 0
            x1 = x0 + self.size
            y1 = y0 + self.size
            img = img.crop((x0,y0,x1,y1))
            guide_img = guide_img.crop((x0,y0,x1,y1))

        w, h = img.size
        
        # for the guide image
        if random.randint(0, 1):
            interp_alg_lr = random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS])
            interp_alg_sr = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
            lr_scale = random.uniform(2, 6)
            lr_w, lr_h = int(round(w / lr_scale)), int(round(h / lr_scale))
            guide_img = guide_img.resize([lr_w, lr_h], interp_alg_lr)
            guide_img = guide_img.resize([w, h], interp_alg_sr)

        # random erase on the guide image
        guide_npy_img = np.asarray(guide_img).astype('float32') / 127.5 - 1
        if random.randint(0, 1):
            guide_npy_img = guide_npy_img + \
                random.uniform(0.02, 0.1) * np.random.randn(*guide_npy_img.shape).astype('float32')
            if random.randint(0, 1):
                a = random.randint(8, 16)
                h, w = guide_npy_img.shape[:2]
                alpha = np.asarray(
                    Image.fromarray(
                        np.random.randint(0,256,[a,a],dtype='uint8')
                    ).resize([w,h], Image.NEAREST)
                ).astype('float32') / 255.0
                alpha = (alpha > 0.25).astype('float32')
                guide_npy_img = guide_npy_img * alpha + (1) * (1 - alpha)

        # random hint points from the target image
        ratio = random.choice([1, 2, 4, 8])
        hint_npy_img = np.asarray(img.resize([int(round(s*(1/ratio))) for s in img.size])).astype('float32') / 127.5 - 1
        hint_mask = np.zeros_like(hint_npy_img, dtype=np.float32)[...,:1]
        base_pts_num = random.randint(self.size//4, self.size)
        y = np.random.randint(0, hint_mask.shape[0], size=base_pts_num//ratio, dtype=np.int64)
        x = np.random.randint(0, hint_mask.shape[1], size=base_pts_num//ratio, dtype=np.int64)
        hint_mask[y, x] = 1
        hint_npy_img = hint_npy_img * hint_mask + (-10) * (1 - hint_mask)

        npy_img = np.asarray(img).astype('float32')
        img = torch.tensor(npy_img.copy()).float().permute(2,0,1) / 127.5 - 1
        guide_img = torch.tensor(guide_npy_img.copy())[None].float()
        color_img = torch.tensor(hint_npy_img.copy()).float().permute(2,0,1)
        color_img = torch.nn.functional.interpolate(color_img[None], size=guide_img.shape[-2:])[0].detach()
        cond_img = torch.cat([guide_img, color_img], 0).float()
        input_ids = self.tokenizer({"text": [item["text"]]})[0]

        return { "pixel_values": img, "guide_values": cond_img, "input_ids": input_ids }

    @staticmethod
    def control_channel():
        return 1 + 3

    @staticmethod
    def cat_input(image: Image.Image, guide: torch.Tensor):
        guide = np.uint8(((guide + 1) * 127.5)[0].permute(1,2,0).cpu().numpy().clip(0,255))
        guide_sketch = Image.fromarray(guide[..., 0]).convert('RGB').resize(image.size)
        guide_color = Image.fromarray(guide[..., 1:]).convert('RGB').resize(image.size)
        image_cat = Image.new('RGB', (image.size[0]*3,image.size[1]), (0,0,0))
        image_cat.paste(guide_sketch,(0,0))
        image_cat.paste(guide_color,(image.size[0],0))
        image_cat.paste(image,(image.size[0]*2, 0))

        return image_cat

Dataset.register_cls('danbooru_sketch')
