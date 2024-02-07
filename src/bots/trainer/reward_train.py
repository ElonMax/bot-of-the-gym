import random

import torch
import numpy as np

from torch.utils.data import DataLoader

from bots.data.reward_dataset import RewardDataset


def train(model, tokenizer, scheduler, optimizer, config, train_data, valid_data, device):

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    train_dataset = RewardDataset(train_data, tokenizer, config["max_length"])
    valid_dataset = RewardDataset(valid_data, tokenizer, config["max_length"])

    train_dataloader = DataLoader(train_dataset, **config["train_loader"])
    valid_dataloader = DataLoader(valid_dataset, **config["valid_loader"])

    max_epochs = config["train_epochs"]
    for epoch in range(max_epochs):
        model.train()

        for step, data in enumerate(train_dataloader, 1):
            chosen_ids = data["input_ids_chosen"].to(device)
            chosen_mask = data["attn_mask_chosen"].to(device)

            rejected_ids = data["input_ids_rejected"].to(device)
            rejected_mask = data["attn_mask_rejected"].to(device)

            r1 = model(input_ids=chosen_ids, attention_mask=chosen_mask).logits
            r2 = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits

            loss = -torch.nn.functional.logsigmoid(r1-r2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            print("Train Epoch: {}/{}\t Step: {}/{}\t Loss: {:.3f}".format(epoch + 1,
                                                                           max_epochs,
                                                                           step,
                                                                           len(train_dataloader),
                                                                           loss.item()))

            model.save_pretrained(config["save_path"])
