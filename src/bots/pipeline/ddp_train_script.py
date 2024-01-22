import os
import pathlib
import argparse

import torch.distributed
import pandas as pd

from pyhocon import ConfigFactory
from torch.nn.parallel import DistributedDataParallel as DDP

from bots.model.load import for_train
from bots.trainer.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train script for tuning language models"
    )

    parser.add_argument('--config',
                        type=str,
                        help="Experiment config path")

    parser.add_argument('--namespace',
                        type=str,
                        help="Config name in *.conf file")

    return parser.parse_args()


def run():
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()

    args = parse_args()
    project_path = pathlib.Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath(args.config)
    config = ConfigFactory.parse_file(config_path)[args.namespace]
    config["data_path"] = project_path.joinpath(config["data_path"])

    train_data = pd.read_csv(config["data_path"], sep=';')

    model, tokenizer = for_train(config)
    model.to(device_id)

    optimizer = torch.optim.Adam(params=model.parameters(), **config["optimizer"])

    model = DDP(model, device_ids=[device_id])

    train(
        train_data=train_data,
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer=tokenizer,
        device=device_id
    )


if __name__ == "__main__":
    run()