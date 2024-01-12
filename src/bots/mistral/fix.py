from transformers import AutoTokenizer, AutoModelForCausalLM


pretrained_dir = "/home/elnmax/llm/Mistral-7B-Instruct-v0.2/"


tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
model = AutoModelForCausalLM.from_pretrained(pretrained_dir)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<unk>'})
    model.resize_token_embeddings(len(tokenizer))


tokenizer.save_pretrained(pretrained_dir)
model.save_pretrained(pretrained_dir)
