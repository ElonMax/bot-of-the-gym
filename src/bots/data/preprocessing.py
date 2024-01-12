import json
import pathlib

import pandas as pd


project_path = pathlib.Path(__file__).parents[3]


def ru_instruct_gpt4_process(path: pathlib.Path) -> (list, list):
    """
    processes jsonlines from https://huggingface.co/datasets/lksy/ru_instruct_gpt4 into 2 lists - "inputs", "outputs"
    :param path: path to dataset
    :return:
        (list of json for input text, list of json for output text)
    """
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
        assistant = full if full else out

        input_text = json.dumps({
            "role": "user",
            "content": user
        })

        inputs.append(input_text)
        outputs.append(assistant)

    return inputs, outputs


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
    # ru_instruct_gpt4 processing
    ru_instruct_gpt4_path = project_path.joinpath("data/raw/ru_instruct_gpt4.jsonl")
    ru_instruct_gpt4_save = project_path.joinpath("data/prep/ru_instruct_gpt4.csv")
    rig4_inputs, rig4_outputs = ru_instruct_gpt4_process(ru_instruct_gpt4_path)
    save_dataset(rig4_inputs, rig4_outputs, ru_instruct_gpt4_save)
