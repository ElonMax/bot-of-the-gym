import argparse
import pathlib

from huggingface_hub import snapshot_download


project_path = pathlib.Path(__file__).parents[3]

models_path = project_path.joinpath("models")
models_path.mkdir(exist_ok=True, parents=True)

data_path = project_path.joinpath("data/raw")
data_path.mkdir(exist_ok=True, parents=True)


# data format
# {
#     "arg-name":[
#         "huggingface-name",
#         ["igonre-file1", "*.igonre_files", 'etc.'],
#         "model/dataset"
#     ]
# }

dict_for_args = {
    "mistral_instruct": [
        "mistralai/Mistral-7B-Instruct-v0.2",
        ["pytorch_model.bin.index.json", "*.bin", ".gitattributes"],
        "model"
    ],
    "mistral_base": [
        "mistralai/Mistral-7B-v0.1",
        ["pytorch_model.bin.index.json", "*.bin", ".gitattributes"],
        "model"
    ],
    "rugpt3large": [
        "ai-forever/rugpt3large_based_on_gpt2",
        ["*.msgpack", ".gitattributes"],
        "model"
    ],

    "data_from_gpt4": [
        "lksy/ru_instruct_gpt4",
        [".gitattributes", ],
        "dataset"
    ],
    "oasst1_rlhf": [
        "tasksource/oasst1_pairwise_rlhf_reward",
        [".gitattributes", ],
        "dataset"
    ],
    "oasst2": [
        "OpenAssistant/oasst2",
        [".gitattributes", ],
        "dataset"
    ]
}


def download():
    parser = argparse.ArgumentParser()

    for arg, desc in dict_for_args.items():

        match desc[-1]:
            case "model":
                arg_help = "Download model {}".format(desc[0])
            case "dataset":
                arg_help = "Download data {}".format(desc[0])
            case _:
                continue

        parser.add_argument(
            "--{}".format(arg),
            default=False,
            action="store_true",
            help=arg_help
        )

    args = parser.parse_args()

    for arg, use in args.__dict__.items():

        if use:
            desc = dict_for_args[arg]
        else:
            continue

        match desc[-1]:
            case "model":
                m_path = models_path.joinpath(desc[0])
                m_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=desc[0],
                    local_dir=m_path,
                    local_dir_use_symlinks=False,
                    ignore_patterns=desc[1]
                )
            case "dataset":
                d_path = data_path.joinpath(desc[0])
                d_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=desc[0],
                    repo_type="dataset",
                    local_dir=d_path,
                    local_dir_use_symlinks=False,
                    ignore_patterns=desc[1]
                )


if __name__ == '__main__':
    download()
