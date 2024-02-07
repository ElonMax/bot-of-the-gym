import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from bots.exceptions.model import ModelTypeError


def for_train(config):
    pretrained = config["pretrained_path"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    match config["model_type"]:
        case "default":
            model = AutoModelForCausalLM.from_pretrained(
                pretrained
            )
        case "bfloat16":
            model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16
            )
        case "flash_attn":
            model = AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        case _:
            raise ModelTypeError(config["model_type"])

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<unk>'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    if config["lora"]:
        from peft import get_peft_model, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **config["lora_config"]
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer
