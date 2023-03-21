import os
import cv2
import torch
import datasets
import numpy as np

from typing import Callable, List
from PIL import Image
from torch.utils import data
from datasets import load_dataset, load_from_disk
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict


def zoom_out_compare(x, y):
    min_v = min(x, y)
    max_v = max(x, y)
    if max_v / min_v < 2:
        max_v = min_v * 2

    return torch.randint(int(max_v / 1.5), max_v + 1, (1, )).item()


def zoom_in_compare(x, y):
    min_v = min(x, y)

    return torch.randint(int(min_v / 1.5), int(min_v / 1.2), (1, )).item()


class BaseDataset(data.Dataset):
    def load_dataset(self):
        path = self.path
        if path is None or path == '':
            return None
        if self.is_imagefolder:
            dataset = load_dataset(
                'imagefolder', data_dir=path)
        elif os.path.exists(path):
            if os.path.isdir(path):
                dataset = load_from_disk(path)
            else:
                if path.endswith('.json') or path.endswith('.jsonl') or path.endswith('.jsonlines'):
                    dataset = Dataset.from_json(path)
                elif path.endswith('.parquet'):
                    dataset = Dataset.from_parquet(path)
                elif path.endswith('.csv'):
                    dataset = Dataset.from_csv(path)
                else:
                    raise ValueError(f"Unsupported path file `{path}`")
        else:
            dataset = load_dataset(path)
        if isinstance(dataset, DatasetDict):
            if self.mode in dataset:
                dataset = dataset[self.mode]
            else:
                dataset = list(dataset.values())[0]

        if self.image_name in dataset.features:
            dataset = dataset.cast_column(
                self.image_name, datasets.Image(decode=True))

        if self.text_name not in dataset.features: # for imagefolder
            dataset = dataset.cast_column(
                self.image_name, datasets.Image(decode=False))
            dataset = dataset.map(lambda example: {self.text_name: os.path.basename(
                os.path.dirname(example[self.image_name]['path']))})
            dataset = dataset.cast_column(
                self.image_name, datasets.Image(decode=True))

        return dataset

    def __init__(
            self,
            path: str = None,
            data: Dataset = None,
            mode: str = 'train',
            root_dir: str = 'data',
            base_name: str = None,
            text_name: str = 'text',
            image_name: str = 'image',
            guide_name: str = 'guide',
            resolution: int = 512,
            enable_zoom_out: bool = False,
            enable_zoom_in: bool = False,
            index_in_mix: int = -1,
            negative_mix: bool = False,
            transparent_color_in_mix: List[int] = None,
            always_keep_in_mix: bool = False,
            control_channel: int = 3,
            instance_text: str = None,
            instance_text_only: bool = True,
            skip_guide: bool = False,
            is_imagefolder: bool = False,
            resample_thres: int = 0,
            random_rotate: bool = False,
            **kwargs):

        self.path = path
        self.mode = mode
        self.root_dir = root_dir
        self.text_name = text_name
        self.image_name = image_name
        self.guide_name = guide_name
        self.resolution = resolution
        self.enable_zoom_out = enable_zoom_out
        self.enable_zoom_in = enable_zoom_in
        self.index_in_mix = index_in_mix
        self.negative_mix = negative_mix
        self.transparent_color_in_mix = transparent_color_in_mix
        self.always_keep_in_mix = always_keep_in_mix
        self.control_channel = control_channel
        self.instance_text = instance_text
        self.instance_text_only = instance_text_only
        self.skip_guide = skip_guide
        self.is_imagefolder = is_imagefolder
        self.resample_thres = resample_thres
        self.random_rotate = random_rotate
        self.tokenizer: Callable = None

        self.data = data
        if data is None:
            self.data = self.load_dataset()

        self.base_name = base_name
        if base_name is None and self.path is not None:
            self.base_name = os.path.basename(path)
            base_name = self.base_name
            if '.' in self.base_name:
                self.base_name = base_name[:base_name.rfind('.')]

    def set_tokenizer(self, tokenizer: Callable):
        assert callable(tokenizer)
        self.tokenizer = tokenizer

    def item_text(self, index: int, extra_text: str = '') -> np.ndarray:
        assert self.tokenizer is not None
        if self.instance_text is not None:
            text = self.instance_text
            if not self.instance_text_only:
                text += ", " + self.data[index][self.text_name]
        else:
            text = self.data[index][self.text_name]
        return self.tokenizer(text + extra_text)

    def item_image(self, index: int) -> np.ndarray:
        return np.asarray(self.data[index][self.image_name].convert('RGB'))

    def item_guide(self, index: int) -> np.ndarray:
        return np.asarray(self.data[index][self.guide_name].convert('RGB'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        guide = None if self.skip_guide else self.item_guide(index)
        if not self.skip_guide and guide.sum() < self.resample_thres:
            return self[(index + 1) % len(self)]
        
        image = self.item_image(index)
        extra_text = ''

        if self.random_rotate:
            rot_rd = torch.randint(0, 3, (1, )).item()
            if rot_rd == 1: # 90°
                extra_text += ', rotate 90°'
                image = image.transpose([1,0,2])
                if not self.skip_guide:
                    guide = guide.transpose([1,0,2])
                if torch.randint(0, 2, (1, )).item():
                    image = image[:,::-1]
                    if not self.skip_guide:
                        guide = guide[:,::-1]
            elif rot_rd == 2: # up down flip
                extra_text += ', up down flip'
                image = image[::-1]
                if not self.skip_guide:
                    guide = guide[::-1]

        zoom_rd = torch.randint(0, 3, (1, )).item()
        zoom_out = self.enable_zoom_out and zoom_rd == 1
        zoom_in = self.enable_zoom_in and zoom_rd == 2
        compare = zoom_out_compare if zoom_out else (
            zoom_in_compare if zoom_in else min)
        h, w = image.shape[:2]
        if compare(h, w) != self.resolution:
            ratio = self.resolution / compare(h, w)
            h, w = int(round(h * ratio)), int(round(w * ratio))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            if not self.skip_guide:
                guide = cv2.resize(guide, (w, h), interpolation=cv2.INTER_AREA)

        h, w = image.shape[:2]
        if min(h, w) < self.resolution:
            extra_text += ', with white border'
            canvas = 255 + \
                np.zeros([self.resolution, self.resolution, 3], dtype=np.uint8)
            if not self.skip_guide:
                guide_canvas = canvas.copy()
            y1 = (self.resolution - h) // 2
            y2 = self.resolution - (self.resolution - h - y1)
            x1 = (self.resolution - w) // 2
            x2 = self.resolution - (self.resolution - w - x1)
            if y1 < 0:
                y1, y2 = 0, self.resolution
                _y1 = torch.randint(0, image.shape[0] - self.resolution, (1, )).item()
                _y2 = _y1 + self.resolution
                image = image[_y1:_y2]
                if not self.skip_guide:
                    guide = guide[_y1:_y2]
            if x1 < 0:
                x1, x2 = 0, self.resolution
                _x1 = torch.randint(0, image.shape[1] - self.resolution, (1, )).item()
                _x2 = _x1 + self.resolution
                image = image[:, _x1:_x2]
                if not self.skip_guide:
                    guide = guide[:, _x1:_x2]
            canvas[y1:y2, x1:x2] = image
            image = canvas.copy()
            if not self.skip_guide:
                guide_canvas[y1:y2, x1:x2] = guide
                guide = guide_canvas.copy()
        elif min(h, w) > self.resolution:
            extra_text += ', zoom in'

        h, w = image.shape[:2]
        x1 = torch.randint(0, w - self.resolution, (1, )
                           ).item() if w > self.resolution else 0
        x2 = x1 + self.resolution
        y1 = torch.randint(0, h - self.resolution, (1, )
                           ).item() if h > self.resolution else 0
        y2 = y1 + self.resolution
        image = image[y1:y2, x1:x2]
        if not self.skip_guide:
            guide = guide[y1:y2, x1:x2]

        if not self.skip_guide and guide.sum() < self.resample_thres:
            return self[(index + 1) % len(self)]
        
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 127.5 - 1
        if not self.skip_guide:
            guide = torch.from_numpy(guide.transpose(2, 0, 1).copy()).float() / 127.5 - 1
        

        text = self.item_text(index, extra_text)

        result = dict(
            input_ids=text,
            pixel_values=image,
            guide_values=guide if not self.skip_guide else 0.0
        )

        return result

    @torch.no_grad()
    def cat_for_show(self, image: Image.Image, target: torch.Tensor=None, guide: torch.Tensor=None):
        if self.skip_guide:
            return image
        
        assert self.control_channel == 1 or self.control_channel == 3

        if guide.shape[1] == 1:
            guide = guide.repeat(1,3,1,1)
        
        target = np.uint8(((target.float().clamp(-1, 1) + 1) * 127.5)
                          [0].permute(1, 2, 0).cpu().numpy().clip(0, 255))
        guide = np.uint8(((guide.float().clamp(-1, 1) + 1) * 127.5)
                         [0].permute(1, 2, 0).cpu().numpy().clip(0, 255))
        target = Image.fromarray(target).convert('RGB').resize(image.size)
        guide = Image.fromarray(guide).convert('RGB').resize(image.size)
        image_cat = Image.new(
            'RGB', (image.size[0]*3, image.size[1]), (0, 0, 0))
        image_cat.paste(target, (0, 0))
        image_cat.paste(guide, (image.size[0], 0))
        image_cat.paste(image, (image.size[0]*2, 0))

        return image_cat
