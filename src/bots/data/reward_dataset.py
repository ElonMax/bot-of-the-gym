import pandas as pd
import transformers
from torch.utils.data import Dataset


class RewardDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 1024):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def encode(self, data):
        return self.tokenizer.batch_encode_plus(
            data,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )

    def __getitem__(self, item):
        chosen = self.data["prompt"][item] + " assistant: " + self.data["chosen"][item]
        rejected = self.data["prompt"][item] + " assistant: " + self.data["rejected"][item]

        chosen_encoded = self.encode([chosen])
        rejected_encoded = self.encode([rejected])

        return {
            "input_ids_chosen": chosen_encoded.input_ids,
            "attn_mask_chosen": chosen_encoded.attention_mask,
            "input_ids_rejected": rejected_encoded.input_ids,
            "attn_mask_rejected": rejected_encoded.attention_mask
        }


if __name__ == "__main__":
    import pandas as pd
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    data_path = "/home/elnmax/Projects/bot-of-the-gym/data/raw/oasst1_pairwise_rlhf_reward/data/train-00000-of-00001-5466fcbe20f5c0ef.parquet"
    table = pq.read_pandas(data_path).to_pandas()
    table = table[table["lang"] == "ru"].reset_index(drop=True)

    tokenizer_path = "/home/elnmax/Projects/bot-of-the-gym/models/rugpt3large_based_on_gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokens = []
    for p, c, r in table[["prompt", "chosen", "rejected"]].values:
        chosen = p + " assistant: " + c
        rejected = p + " assistant: " + r

        ids = len(tokenizer(p).input_ids)
        tokens.append(ids)
        # ids = len(tokenizer(chosen).input_ids)
        # tokens.append(ids)
        # ids = len(tokenizer(rejected).input_ids)
        # tokens.append(ids)

    bigger = []
    for i in tokens:
        if i > 1024:
            bigger.append(i)

    print(len(bigger))
