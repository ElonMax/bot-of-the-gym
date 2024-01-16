import os

import torch.distributed
import pandas as pd

from pyhocon import ConfigFactory
from torch.nn.parallel import DistributedDataParallel as DDP

from bots.model.load import for_train
from bots.trainer.train import train


def setup(rank, world_size):
    os.environ['MASTED_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def run(rank, world_size):
    setup(rank, world_size)

    train_data = pd.read_csv("/s/ls4/users/cappukan/projects/bot-of-the-gym/data/prep/ru_instruct_gpt4.csv", sep=';')
    config = ConfigFactory.parse_file("/s/ls4/users/cappukan/projects/bot-of-the-gym/configs/mistral-7B-exp.conf")["ru_instruct_gpt4"]

    model, tokenizer = for_train(config)
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    ddp_model, tokenizer = train(
        train_data=train_data,
        model=ddp_model,
        config=config,
        tokenizer=tokenizer,
        device=rank
    )

    if rank == 0:
        ddp_model.save_pretrained(config["save_path"])
        tokenizer.save_pretrained(config["save_path"])


if __name__ == "__main__":
    run()