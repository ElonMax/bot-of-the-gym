import argparse
import pathlib

from huggingface_hub import snapshot_download, hf_hub_download


project_path = pathlib.Path(__file__).parents[3]

models_path = project_path.joinpath("models")
models_path.mkdir(exist_ok=True, parents=True)

data_path = project_path.joinpath("data/raw")
data_path.mkdir(exist_ok=True, parents=True)


def download():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument(
        "mistral_instruct",
        default=False,
        action="store_true",
        help="Download model Mistral-7B-Instruct-v0.2"
    )

    parser.add_argument(
        "mistral_base",
        default=False,
        action="store_true",
        help="Download model Mistral-7B-v0.1"
    )

    parser.add_argument(
        "data_from_gpt4",
        default=False,
        action="store_true",
        help="Download dataset ru_instruct_gpt4"
    )

    args = parser.parse_args()

    if args.mistral_instruct:
        mistral_instruct_path = models_path.joinpath("Mistral-7B-Instruct-v0.2")
        snapshot_download(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            local_dir=mistral_instruct_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["pytorch_model.bin.index.json", "*.bin", ".gitattributes"]
        )

    if args.mistral_base:
        mistral_path = models_path.joinpath("Mistral-7B-v0.1")
        snapshot_download(
            repo_id="mistralai/Mistral-7B-v0.1",
            local_dir=mistral_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["pytorch_model.bin.index.json", "*.bin", ".gitattributes"]
        )

    if args.data_from_gpt4:
        hf_hub_download(
            repo_id="lksy/ru_instruct_gpt4",
            repo_type="dataset",
            local_dir=data_path,
            local_dir_use_symlinks=False,
            filename="ru_instruct_gpt4.jsonl"
        )


if __name__ == '__main__':
    download()
