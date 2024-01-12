from torch.utils.data import Dataset


class BotDataset(Dataset):

    def __init__(self, dataframe, tokenizer, text_length: int = 2000):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text_length = text_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_item = self.data["input_text"][item]
        output_item = self.data["output_text"][item]

        input_encode = self.batch_encode(input_item)
        output_encode = self.batch_encode(output_item)

        return {
            'input_ids': input_encode.input_ids.squeeze(),
            'attention_mask': input_encode.attention_mask.squeeze(),
            'labels': output_encode.input_ids.squeeze()
        }

    def batch_encode(self, item):
        encoded = self.tokenizer.batch_encode_plus(
            [item],
            add_special_tokens=False,
            max_length=self.text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return encoded
