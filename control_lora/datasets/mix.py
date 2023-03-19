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


class MixDataset(BaseDataset):
    def __init__(
            self,
            data_types: List[DictConfig] = [],
            **kwargs):

        super().__init__(**kwargs)

        kwargs['data'] = self.data
        datasets: List[BaseDataset] = [
            instantiate_from_config(data_type, **kwargs)
            for data_type in data_types]
        mix_id_dataset_map: Dict[int, List[BaseDataset]] = { dataset.index_in_mix: [] for dataset in datasets }
        for dataset in datasets:
            mix_id_dataset_map[dataset.index_in_mix].append(dataset)
        self.datasets = mix_id_dataset_map
        
    def item_guide(self, index: int) -> np.ndarray:
        source_keys = sorted(list(self.datasets.keys()))
        source_num = 1 # torch.randint(1, len(source_keys), (1, )).item() if len(source_keys) > 1 else 1
        source_ids_pt: torch.LongTensor = torch.randint(0, len(source_keys), (source_num, ))
        source_ids_np: np.ndarray = source_ids_pt.numpy()
        source_ids: List[int] = sorted(source_ids_np.tolist())
        guide = None
        datasets = { k: [] for k in self.datasets.keys() }
        for source_id in source_ids:
            dataset_list = self.datasets[source_keys[source_id]]
            dataset_id: int = torch.randint(0, len(dataset_list), (1, )).item()
            dataset = dataset_list[dataset_id]
            datasets[source_keys[source_id]] = [dataset]

        for key, dataset_list in self.datasets.items():
            for dataset in dataset_list:
                if dataset.always_keep_in_mix:
                    if dataset not in datasets[key]:
                        datasets[key].append(dataset)

        for key in sorted(datasets.keys()):
            dataset_list = datasets[key]
            dataset: BaseDataset
            for dataset in dataset_list:
                _guide = dataset.item_guide(index)
                alpha = None
                if dataset.transparent_color_in_mix is not None:
                    alpha_r = _guide[...,0:1] == dataset.transparent_color_in_mix[0]
                    alpha_g = _guide[...,1:2] == dataset.transparent_color_in_mix[1]
                    alpha_b = _guide[...,2:3] == dataset.transparent_color_in_mix[2]
                    _alpha = 1 - np.float32(alpha_r * alpha_g * alpha_b)
                if dataset.negative_mix:
                    _guide = 255 - _guide
                _guide = _guide.astype(np.float32)
                if guide is None:
                    guide = _guide
                else:
                    alpha = 1
                    # h, w = guide.shape[:2]
                    # alpha = torch.randint(0, 2, (4, 4), dtype=torch.uint8).numpy().astype(np.float32)
                    # alpha = cv2.resize(alpha, [w, h], interpolation=cv2.INTER_NEAREST)[...,None]
                    if dataset.transparent_color_in_mix is not None:
                        alpha = alpha * _alpha
                    if dataset.always_keep_in_mix:
                        alpha = _alpha
                    guide = _guide * alpha + guide * (1 - alpha)

        return guide.astype(np.float32)
