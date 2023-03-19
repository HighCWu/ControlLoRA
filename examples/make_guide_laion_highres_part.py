import os
import random
import sys

from PIL import Image

MAIN_DIR = os.path.abspath(os.path.dirname(__file__) + '/..')
sys.path.insert(0, MAIN_DIR)

from control_lora.datasets.canny import CannyDataset
from control_lora.datasets.mlsd import MLSDDataset
from control_lora.datasets.hed import HEDDataset
from control_lora.datasets.scribble import ScribbleDataset
from control_lora.datasets.openpose import OpenposeDataset
from control_lora.datasets.uniformer import UniformerDataset
from control_lora.datasets.midas import MidasDataset

CannyDataset(path='data/laion-high-resolution-part')
MLSDDataset(path='data/laion-high-resolution-part')
HEDDataset(path='data/laion-high-resolution-part')
ScribbleDataset(path='data/laion-high-resolution-part')
OpenposeDataset(path='data/laion-high-resolution-part')
UniformerDataset(path='data/laion-high-resolution-part')
MidasDataset(path='data/laion-high-resolution-part', guide_type='depth')
MidasDataset(path='data/laion-high-resolution-part', guide_type='normal')
