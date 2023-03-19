import os
import glob
import datasets
import jsonlines

from multiprocessing import freeze_support

from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from huggingface_hub import hf_hub_download
from img2dataset import download

MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PARQUET_PATH = os.path.join(MAIN_DIR, 'data/laion-high-resolution-part-extracted.parquet')


path = hf_hub_download(
    "laion/laion-high-resolution", 
    "part-00000-5d6701c4-b238-4c0a-84e4-fe8e9daea963-c000.snappy.parquet",
    repo_type="dataset"
)
dataset = Dataset.from_parquet(path)
dataset = dataset.filter(lambda data: data['LANGUAGE'] == 'en')
dataset = dataset.shuffle(233).train_test_split(20000 / dataset.num_rows)['test']
dataset.to_parquet(PARQUET_PATH)


def main():
    download(
        processes_count=8,
        thread_count=2,
        url_list=PARQUET_PATH,
        image_size=512,
        output_folder=os.path.join(MAIN_DIR, 'data/laion-high-resolution-part-extracted'),
        output_format="files",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=True,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
        resize_mode="keep_ratio"
    )


def save_to_disk():
    files = glob.glob(
        os.path.join(
            MAIN_DIR, 
            'data/laion-high-resolution-part-extracted', 
            '**', '*.txt'), 
        recursive=True)
    files = sorted(files)
    json_file = os.path.join(
        MAIN_DIR, 
        'data/laion-high-resolution-part.jsonl')
    with jsonlines.open(json_file, 'w') as f:
        for file in files:
            with open(file, 'r', encoding='utf-8') as f2:
                text = f2.read()
            img = file[:-3] + 'jpg'
            f.write({ 'image': img, 'text': text })

    dataset = Dataset.from_json(json_file)
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    dataset = DatasetDict(train=dataset)
    dataset.save_to_disk(os.path.join(
        MAIN_DIR, 
        'data/laion-high-resolution-part'))


if __name__ == '__main__':
    freeze_support()
    main()
    save_to_disk()
    