import argparse
import pathlib

from huggingface_hub import snapshot_download


project_path = pathlib.Path(__file__).parents[3]

models_path = project_path.joinpath("models")
models_path.mkdir(exist_ok=True, parents=True)

data_path = project_path.joinpath("data/raw")
data_path.mkdir(exist_ok=True, parents=True)


def download():
    parser = argparse.ArgumentParser(

    )

    parser.add_argument(
        "--mistral_instruct",
        default=False,
        action="store_true",
        help="Download model Mistral-7B-Instruct-v0.2"
    )

    parser.add_argument(
        "--mistral_base",
        default=False,
        action="store_true",
        help="Download model Mistral-7B-v0.1"
    )

    parser.add_argument(
        "--rugpt3large",
        default=False,
        action="store_true",
        help="Download model rugpt3large_based_on_gpt2"
    )

    parser.add_argument(
        "--data_from_gpt4",
        default=False,
        action="store_true",
        help="Download dataset ru_instruct_gpt4"
    )

    parser.add_argument(
        "--oasst1_rlhf",
        default=False,
        action="store_true",
        help="Download dataset oasst1_pairwise_rlhf_reward"
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

    if args.rugpt3large:
        rugpt3large_path = models_path.joinpath("rugpt3large_based_on_gpt2")
        snapshot_download(
            repo_id="ai-forever/rugpt3large_based_on_gpt2",
            local_dir=rugpt3large_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.msgpack", ".gitattributes"]
        )

    if args.data_from_gpt4:
        ru_instruct_gpt4_path = data_path.joinpath("ru_instruct_gpt4")
        snapshot_download(
            repo_id="lksy/ru_instruct_gpt4",
            repo_type="dataset",
            local_dir=ru_instruct_gpt4_path,
            local_dir_use_symlinks=False,
            ignore_patterns=[".gitattributes"]
        )

    if args.oasst1_rlhf:
        oasst1_rlhf_path = data_path.joinpath("oasst1_pairwise_rlhf_reward")
        snapshot_download(
            repo_id="tasksource/oasst1_pairwise_rlhf_reward",
            repo_type="dataset",
            local_dir=oasst1_rlhf_path,
            local_dir_use_symlinks=False,
            ignore_patterns=[".gitattributes"]
        )


if __name__ == '__main__':
    download()
