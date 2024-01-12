import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.bots.data.dataset import BotDataset


data_path = "/home/elnmax/Projects/bot-of-the-gym/data/prep/ru_instruct_gpt4.csv"

pretrained_dir = "/home/elnmax/llm/Mistral-7B-Instruct-v0.2/"


device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
model = AutoModelForCausalLM.from_pretrained(pretrained_dir)
model.to(device)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True


df = pd.read_csv(data_path, sep=';')
dataset = BotDataset(dataframe=df, tokenizer=tokenizer)
dataloader = DataLoader(dataset, **{"batch_size": 8, "shuffle": False})

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
epochs = 5

for epoch in range(epochs):

    model.train()

    for step, data in enumerate(dataloader):
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

        if step % 10 == 0:
            print("Epoch: {}/{}\t Step: {}/{}\t Loss: {:.3f}".format(epoch + 1, epochs, step, len(dataloader), loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
