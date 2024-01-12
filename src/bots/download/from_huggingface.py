import pathlib

from huggingface_hub import snapshot_download


project_path = pathlib.Path(__file__).parents[3]
models_path = project_path.joinpath("models")
models_path.mkdir(exist_ok=True, parents=True)


mistral_path = models_path.joinpath("Mistral-7B-Instruct-v0.2")
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir=mistral_path,
    ignore_patterns=["pytorch_model.bin.index.json", "*.bin", ".gitattributes"]
)
