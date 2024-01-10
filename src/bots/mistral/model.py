import argparse

import torch
import deepspeed

from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
parser = deepspeed.add_config_arguments(parser)

cmd_args = parser.parse_args()


pretrained_dir = "/s/ls4/groups/g0126/transformers_models/mistralai/Mistral-7B-Instruct-v0.2/"


tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
model = AutoModelForCausalLM.from_pretrained(pretrained_dir)


model_engine, optimizer, _, _ = deepspeed.initialize(
    args=cmd_args,
    model=model,
    model_parameters=model.parameters()
)
