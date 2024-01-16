import numpy as np
import torch

from torch.utils.data import Dataset


class BotDataset(Dataset):
    def __init__(self,
                 dataframe,
                 tokenizer,
                 text_length: int = 1000,
                 truncation: bool = True,
                 padding: bool = True,
                 only_target_loss: bool = True):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text_length = text_length - 1  # place for eos_token
        self.truncation = truncation
        self.padding = padding
        self.only_target_loss = only_target_loss

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        inp_text = self.data["input_text"][item]
        out_text = self.data["output_text"][item]

        inp_enc = self.encode(inp_text)
        out_enc = self.encode(out_text)

        if self.only_target_loss:
            lab_enc = [self.tokenizer.pad_token_id for _ in inp_enc]
        else:
            lab_enc = inp_enc

        input_ids = inp_enc + out_enc
        labels = lab_enc + out_enc

        seq_length = len(input_ids)
        # truncation
        if self.truncation and seq_length >= self.text_length:
            input_ids = input_ids[:self.text_length] + [self.tokenizer.eos_token_id]
            labels = labels[:self.text_length] + [self.tokenizer.eos_token_id]
        else:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        attention_mask = np.ones(len(input_ids))

        # padding
        if self.padding and seq_length < self.text_length:
            pad_num = self.text_length - seq_length
            pad_ids = [self.tokenizer.pad_token_id for _ in range(pad_num)]
            pad_attn = np.zeros(pad_num)

            input_ids = pad_ids + input_ids
            attention_mask = np.concatenate((pad_attn, attention_mask))
            labels = pad_ids + labels

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        labels = torch.LongTensor(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def encode(self, text):
        tensor = self.tokenizer(
            text,
            add_special_tokens=False,
        )

        return tensor.input_ids
