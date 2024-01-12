import pathlib

from huggingface_hub import snapshot_download, hf_hub_download


project_path = pathlib.Path(__file__).parents[3]

models_path = project_path.joinpath("models")
models_path.mkdir(exist_ok=True, parents=True)

data_path = project_path.joinpath("data/raw")
data_path.mkdir(exist_ok=True, parents=True)


def download():
    mistral_path = models_path.joinpath("Mistral-7B-Instruct-v0.2")
    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        local_dir=mistral_path,
        ignore_patterns=["pytorch_model.bin.index.json", "*.bin", ".gitattributes"]
    )

    hf_hub_download(
        repo_id="lksy/ru_instruct_gpt4",
        repo_type="dataset",
        local_dir=data_path,
        filename="ru_instruct_gpt4.jsonl"
    )


if __name__ == '__main__':
    download()
