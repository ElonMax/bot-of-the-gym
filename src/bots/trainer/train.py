import pathlib
import random
import datetime

import torch
import torch.distributed
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from bots.data.dataset import BotDataset


def train(train_data, model, optimizer, config, tokenizer, device):
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(config["logdir"]),
        record_shapes=True,
        profile_memory=True,
        use_cuda=True
    )

    log_path = pathlib.Path(config["logdir"]) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    writer = SummaryWriter(log_path.__str__())

    metrics = {
        "Loss/train": None
    }

    # reproducibility
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # load data
    train_dataset = BotDataset(dataframe=train_data, tokenizer=tokenizer, text_length=config["max_tokens"])

    sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(dataset=train_dataset, sampler=sampler, **config["train_loader"])

    max_epochs = config["train_epochs"]
    prof.start()
    for epoch in range(max_epochs):
        prof.step()
        model.train()
        torch.distributed.barrier()

        for step, data in enumerate(train_dataloader, 1):
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

            writer.add_scalar("Loss/train", loss, step)

            # if step % 10 == 0:
            print("Epoch: {}/{}\t Step: {}/{}\t Loss: {:.3f}\t".format(epoch+1,
                                                                       max_epochs,
                                                                       step,
                                                                       len(train_dataloader),
                                                                       loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if device == 0:
            model.module.save_pretrained(config["save_path"])

    prof.stop()
    writer.add_hparams(hparam_dict=config, metric_dict=metrics)
    writer.flush()
    writer.close()

    return model, tokenizer
