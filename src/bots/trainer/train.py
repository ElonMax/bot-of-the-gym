import time
import datetime
import random

import mlflow
import pandas as pd
import pyhocon
import torch
import torch.distributed
import numpy as np
import transformers

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from bots.data.dataset import BotDataset


def train(
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel,
        config: pyhocon.ConfigTree | dict,
        tokenizer: transformers.PreTrainedTokenizer,
        device: int | str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None
):

    torch.distributed.barrier()
    stamp = datetime.datetime.now().strftime("%d%m%Y-%H%M")

    mlflow.set_tracking_uri(config["log_path"])

    if device == 0:
        print("Create experiment...")

        try:
            mlflow.create_experiment(config["experiment_name"])
        except mlflow.exceptions.MlflowException as e:
            print(e)

    torch.distributed.barrier()
    mlflow.set_experiment(config["experiment_name"])

    run_name = "{}-rank:{}".format(stamp, device)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config)

        # reproducibility
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # load data
        train_dataset = BotDataset(dataframe=train_data, tokenizer=tokenizer, text_length=config["max_tokens"])
        valid_dataset = BotDataset(dataframe=valid_data, tokenizer=tokenizer, text_length=config["max_tokens"])

        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)

        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, **config["train_loader"])
        valid_dataloader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, **config["valid_loader"])

        max_epochs = config["train_epochs"]
        for epoch in range(max_epochs):
            model.train()
            torch.distributed.barrier()

            for step, data in enumerate(train_dataloader, 1):
                start_time = time.time()

                input_ids = data["input_ids"].to(device)
                attn_mask = data["attention_mask"].to(device)
                labels = data["labels"].to(device)
                labels[labels == tokenizer.pad_token_id] = -100

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels
                )

                loss = outputs[0]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

                mlflow.log_metric("Loss/train", loss, step + len(train_dataloader) * epoch)

                print("Train Epoch: {}/{}\t Step: {}/{}\t Loss: {:.3f}\t Time: {:.3f}".format(epoch+1,
                                                                                              max_epochs,
                                                                                              step,
                                                                                              len(train_dataloader),
                                                                                              loss.item(),
                                                                                              time.time() - start_time
                                                                                              ))

            model.eval()
            with torch.no_grad():
                eval_losses = []
                for step, data in enumerate(valid_dataloader, 1):
                    input_ids = data["input_ids"].to(device)
                    attn_mask = data["attention_mask"].to(device)
                    labels = data["labels"].to(device)
                    labels[labels == tokenizer.pad_token_id] = -100

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        labels=labels
                    )

                    print("Eval Step: {}/{}".format(step, len(valid_dataloader)))

                    eval_losses.append(outputs[0])

                loss1 = sum(eval_losses) / len(eval_losses)

                torch.distributed.barrier()
                torch.distributed.all_reduce(loss1, op=torch.distributed.ReduceOp.AVG)
                if device == 0:
                    mlflow.log_metric("Loss/eval", loss1, epoch)

            if device == 0:
                model.module.save_pretrained(config["save_path"])

    return model, tokenizer
