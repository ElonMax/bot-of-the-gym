import pathlib
import argparse

import pandas as pd

from pyhocon import ConfigFactory

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
    args = parse_args()

    project_path = pathlib.Path(__file__).resolve().parents[3]
    config_path = project_path.joinpath(args.config)

    config = ConfigFactory.parse_file(config_path)[args.namespace]
    config["pretrained_path"] = project_path.joinpath(config["pretrained_path"])
    config["data_path"] = project_path.joinpath(config["data_path"])
    config["save_path"] = project_path.joinpath(config["save_path"])

    train_data = pd.read_csv(config["data_path"], sep=';')

    model, tokenizer, device = for_train(config)

    model, tokenizer = train(
        train_data=train_data,
        model=model,
        config=config,
        tokenizer=tokenizer,
        device=device
    )

    model.save_pretrained(config["save_path"])
    tokenizer.save_pretrained(config["save_path"])


if __name__ == "__main__":
    run()