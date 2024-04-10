import gzip
import json
import pathlib
import argparse

import pandas as pd

from transformers import AutoTokenizer


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_from_gpt4",
        default=False,
        action="store_true",
        help="Preprocessing for lksy/ru_instruct_gpt4 dataset"
    )

    parser.add_argument(
        "--oasst2",
        default=False,
        action="store_true",
        help="Preprocessing for OpenAssistant/oasst2 dataset"
    )

    return parser.parse_args()


def ru_instruct_gpt4_process_mistral_7b(
        path: pathlib.Path,
        pretrained: str = "models/Mistral-7B-Instruct-v0.2"
) -> (list, list):
    """
    processes jsonlines from https://huggingface.co/datasets/lksy/ru_instruct_gpt4 into 2 lists - "inputs", "outputs"
    used for Mistral-7B-Instruct-v0.2
    :param path: path to dataset
    :param pretrained: path to tokenizer
    :return:
        (list of json for input text, list of json for output text)
    """
    mistral_tokenizer_path = pretrained
    tokenizer = AutoTokenizer.from_pretrained(mistral_tokenizer_path)

    with open(path, 'r') as file:
        data = list(
            map(
                lambda x: json.loads(x), file.readlines()
            )
        )

    df = pd.DataFrame(data)

    inputs = []
    outputs = []
    for inst, inp, out, full in df.values:
        user = ' '.join([inst, inp])

        if full and len(full) > len(out):
            assistant = full
        else:
            continue

        messages = [
            {"role": "user", "content": user}
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        output_text = assistant

        inputs.append(input_text)
        outputs.append(output_text)

    return inputs, outputs


def oasst2_process_mistral_7b(
        path: pathlib.Path,
        pretrained: str = "models/Mistral-7B-Instruct-v0.2"
) -> (list, list):
    """

    :param path:
    :param pretrained:
    :return:
    """
    mistral_tokenizer_path = pretrained
    tokenizer = AutoTokenizer.from_pretrained(mistral_tokenizer_path)

    with gzip.open(path, mode="tr") as file:
        data = list(
            map(
                lambda x: json.loads(x), file.readlines()
            )
        )

    df = pd.DataFrame(data)
    df = df[df["lang"] == "ru"]
    trees = df["message_tree_id"].unique()

    for tree in trees:
        print(df[df["message_tree_id"] == tree][["message_id", "parent_id", "text", "role"]])

    return [], []


def save_dataset(input_text: list, output_text: list, save_path: pathlib.Path) -> pd.DataFrame:
    """
    creates Dataframe from 2 list - input_text, output_text and save it
    :param input_text: list of input_text
    :param output_text: list of output_text
    :param save_path: save path
    :return:
        Dataframe with next columns - "input_text", "output_text"
    """
    prep_dataset = pd.DataFrame(
        {
            "input_text": input_text,
            "output_text": output_text
        }
    )

    if not save_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    prep_dataset.to_csv(save_path, sep=';', index=False)

    return prep_dataset


if __name__ == "__main__":
    project_path = pathlib.Path(__file__).parents[3]

    args = parse_arguments()

    # tokenizer_path = "/s/ls4/groups/g0126/transformers_models/mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_path = "models/Mistral-7B-Instruct-v0.2"

    if args.data_from_gpt4:
        load_path = project_path.joinpath("data/raw/ru_instruct_gpt4/ru_instruct_gpt4.jsonl")
        save_path = project_path.joinpath("data/prep/ru_instruct_gpt4.csv")
        input_data, output_data = ru_instruct_gpt4_process_mistral_7b(load_path, pretrained=tokenizer_path)

    elif args.oasst2:
        load_path = project_path.joinpath("data/raw/OpenAssistant/oasst2/2023-11-05_oasst2_all.messages.jsonl.gz")
        save_path = project_path.joinpath("data/prep/oasst2.csv")
        input_data, output_data = oasst2_process_mistral_7b(load_path, pretrained=tokenizer_path)
    else:
        input_data, output_data, save_path = None, None, None

    if input_data and output_data and save_path:
        df = save_dataset(input_data, output_data, save_path)
