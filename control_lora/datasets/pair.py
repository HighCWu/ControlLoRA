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


class PairDataset(BaseDataset):
    def __init__(
            self,
            guide_types: List[DictConfig] = [],
            **kwargs):

        super().__init__(**kwargs)

        assert len(guide_types) > 0
        self.guide_datasets: List[BaseDataset] = [
            instantiate_from_config(data_type, **kwargs)
            for data_type in guide_types]
        
    def item_guide(self, index: int) -> np.ndarray:
        select_id = torch.randint(0, len(self.guide_datasets), (1,)).item()
        dataset: BaseDataset = self.guide_datasets[select_id]
        return dataset.item_image(index)
