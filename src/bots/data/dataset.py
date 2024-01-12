import json

from torch.utils.data import Dataset


class BotDataset(Dataset):

    def __init__(self, dataframe, tokenizer, text_length: int = 4000):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text_length = text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_item = json.loads(self.data["input_text"][item])
        output_item = self.data["output_text"][item]

        input_chat_template = self.tokenizer.apply_chat_template([input_item], tokenize=False)
        input_encode = self.tokenizer.batch_encode_plus(
            [input_chat_template],
            add_special_tokens=False,
            max_length=self.text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        output_encode = self.tokenizer.batch_encode_plus(
            [output_item + self.tokenizer.eos_token],
            add_special_tokens=False,
            max_length=self.text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': input_encode.input_ids.squeeze(),
            'attention_mask': input_encode.attention_mask.squeeze(),
            'labels': output_encode.input_ids.squeeze()
        }
