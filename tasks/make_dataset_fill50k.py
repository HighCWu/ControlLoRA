
import os
import datasets

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict

os.chdir(os.path.join(os.path.dirname(__file__), '..'))

os.makedirs('HighCWu', exist_ok=True)

def map_fn(kwargs):
    def _map_fn(source, target, prompt):
        image = os.path.join('data', 'fill50k', target)
        guide = os.path.join('data', 'fill50k', source)
        text = prompt
        return dict(image=image, guide=guide, text=text)
    return _map_fn(**kwargs)


# download from https://huggingface.co/lllyasviel/ControlNet/blob/main/training/fill50k.zip
fill50k_dataset = Dataset.from_json('data/fill50k/prompt.json') 
fill50k_dataset = fill50k_dataset.map(map_fn)
fill50k_dataset = fill50k_dataset.remove_columns(['source', 'target', 'prompt'])
fill50k_dataset = fill50k_dataset.cast_column("image", datasets.Image(decode=True))
fill50k_dataset = fill50k_dataset.cast_column("guide", datasets.Image(decode=True))
fill50k_dataset = DatasetDict(train=fill50k_dataset)
fill50k_dataset.save_to_disk("HighCWu/fill50k")
