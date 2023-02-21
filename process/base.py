import torch
import numpy as np

from PIL import Image
from torch.utils import data


class Dataset(data.Dataset):
    DATASET_TYPE_DICT = {}

    def __init__(self, tokenizer, resolution=512, use_crop=True, **kwargs):
        pass

    @classmethod
    def register_cls(cls, name: str):
        Dataset.DATASET_TYPE_DICT['process/' + name] = cls

    @staticmethod
    def from_name(name: str):
        dataset_cls: Dataset = Dataset.DATASET_TYPE_DICT[name]
        return dataset_cls

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
        image_cat.paste(guide,(image.size[0], 0))
        image_cat.paste(image,(image.size[0]*2, 0))

        return image_cat
