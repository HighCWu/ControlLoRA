import os

from datasets import load_from_disk

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DISK_PATH = os.path.join(MAIN_DIR, 'data/laion-high-resolution-part')

dataset = load_from_disk(DISK_PATH)
print(dataset)
print(type(dataset['train'].features['image']))
