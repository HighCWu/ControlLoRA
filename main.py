import click

from control_lora.utils.config import instantiate_from_config
from control_lora.commands.trainer import Trainer
from control_lora.models import ControlLoRAContainer
from control_lora.datasets import BaseDataset


@click.command()
@click.option('--config', default='configs/base.yaml', help='Config file path')
@click.option('--running_mode', default='train', help='Trainer running mode, [train | sample]')
@click.option('--validation_prompt', default=None, help='Replace the default validation prompt')
@click.option('--pretrained_model_name_or_path', default=None, help='Replace the default pretrained model')
def main(config, running_mode, validation_prompt, pretrained_model_name_or_path):
    objs = instantiate_from_config(config)
    trainer: Trainer
    model: ControlLoRAContainer
    dataset: BaseDataset
    trainer, model, dataset = objs['trainer'], objs['model'], objs['dataset']
    trainer.args.running_mode = running_mode
    if validation_prompt is not None:
        trainer.args.validation_prompt = validation_prompt
    if pretrained_model_name_or_path is not None:
        trainer.args.pretrained_model_name_or_path = pretrained_model_name_or_path
    trainer.run(model=model, dataset=dataset)

if __name__ == "__main__":
    main()
