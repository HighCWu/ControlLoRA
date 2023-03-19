import os
import cv2
import torch
import datasets
import numpy as np

from typing import Callable, Dict, List
from omegaconf import DictConfig
from datasets.arrow_dataset import Dataset
from control_lora.utils.config import instantiate_from_config
from control_lora.datasets.base import BaseDataset


class MultiDataset(BaseDataset):
    def __init__(
            self,
            data_types: List[DictConfig] = [],
            **kwargs):

        super().__init__(**kwargs)

        self.datasets: List[BaseDataset] = [
            instantiate_from_config(data_type, **kwargs)
            for data_type in data_types]
        
        self.cached_id = -1
        self.cached_dataset = None
        
    def __len__(self):
        return max([len(d) for d in self.datasets])
    
    def set_tokenizer(self, tokenizer: Callable):
        assert callable(tokenizer)
        self.tokenizer = tokenizer
        for dataset in self.datasets:
            dataset.set_tokenizer(tokenizer)

    def item_dataset(self, func_name: str, index: int, **kwargs) -> np.ndarray:
        if index != self.cached_id:
            dataset_id: torch.LongTensor = torch.randint(0, len(self.datasets), (1, )).item()
            self.cached_id = index
            self.cached_dataset = self.datasets[dataset_id]
        index = index % len(self.cached_dataset)

        return getattr(self.cached_dataset, func_name)(index, **kwargs)

    def item_text(self, index: int, extra_text: str = '') -> np.ndarray:
        return self.item_dataset('item_text', index, extra_text=extra_text)

    def item_image(self, index: int) -> np.ndarray:
        return self.item_dataset('item_image', index)
        
    def item_guide(self, index: int) -> np.ndarray:
        return self.item_dataset('item_guide', index)
