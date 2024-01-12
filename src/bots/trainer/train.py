import random

import torch
import numpy as np

from torch.utils.data import DataLoader

from bots.data.dataset import BotDataset


def train(train_data, model, config, tokenizer, device):
    model.to(device)

    # reproducibility
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # load data
    train_dataset = BotDataset(dataframe=train_data, tokenizer=tokenizer, text_length=config["max_tokens"])
    train_dataloader = DataLoader(train_dataset, **config["train_loader"])

    # set optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), **config["optimizer"])

    max_epochs = config["train_epochs"]
    for epoch in range(max_epochs):
        model.train()

        for step, data in enumerate(train_dataloader):
            input_ids = data["input"].to(device)
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

    return model, tokenizer
