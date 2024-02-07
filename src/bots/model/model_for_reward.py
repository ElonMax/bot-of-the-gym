import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from bots.exceptions.model import ModelTypeError


def for_train(config):
    pretrained = config["pretrained_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    match config["model_type"]:
        case "default":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained,
                num_labels=1
            )
        case "bfloat16":
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained,
                num_labels=1,
                torch_dtype=torch.bfloat16
            )
        case _:
            raise ModelTypeError(config["model_type"])

    return model, tokenizer
