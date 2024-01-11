from transformers import AutoTokenizer


pretrained_dir = "/s/ls4/groups/g0126/transformers_models/mistralai/Mistral-7B-Instruct-v0.2/"


tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

