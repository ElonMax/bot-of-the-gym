import pathlib
import argparse

import torch.distributed
import pandas as pd
import pyarrow.parquet as pq

from pyhocon import ConfigFactory
from torch.nn.parallel import DistributedDataParallel as DDP

from bots.model.model_for_reward import for_train
from bots.trainer.reward_train import train


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
    device_id = 0

    args = parse_args()
    project_path = pathlib.Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath(args.config)
    config = ConfigFactory.parse_file(config_path)[args.namespace]

    train_path = "/home/elnmax/Projects/bot-of-the-gym/data/raw/oasst1_pairwise_rlhf_reward/data/train-00000-of-00001-5466fcbe20f5c0ef.parquet"
    valid_path = "/home/elnmax/Projects/bot-of-the-gym/data/raw/oasst1_pairwise_rlhf_reward/data/validation-00000-of-00001-6855d7506403041c.parquet"

    train_table = pq.read_pandas(train_path).to_pandas()
    train_table = train_table[train_table["lang"] == "ru"].reset_index(drop=True)

    valid_table = pq.read_pandas(valid_path).to_pandas()
    valid_table = valid_table[valid_table["lang"] == "ru"].reset_index(drop=True)

    model, tokenizer = for_train(config)
    model.to(device_id)

    optimizer = torch.optim.AdamW(params=model.parameters(), **config["optimizer"])

    train(
        model,
        tokenizer,
        None,
        optimizer,
        config,
        train_table,
        valid_table,
        device_id
    )


if __name__ == "__main__":
    run()
