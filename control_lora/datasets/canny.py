import cv2
import numpy as np
import torch
from control_lora.datasets.base import BaseDataset
from control_lora.datasets.annotator.canny import CannyDetector


class CannyDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_canny = CannyDetector()
    
    def item_guide(self, index: int) -> np.ndarray:
        image = self.item_image(index)
        h, w = image.shape[:2]
        if min(h, w) != self.resolution:
            ratio = self.resolution / min(h, w)
            h, w = int(round(h * ratio)), int(round(w * ratio))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        
        low_threshold = torch.randint(0, 255 + 1, (1, )).item()
        high_threshold = torch.randint(0, 255 + 1, (1, )).item()

        guide = self.apply_canny(image, low_threshold, high_threshold)

        return np.tile(guide[...,None], [1, 1, 3])
