# bot-of-the-gym

## Datasets

- Instructions from GPT-4 - https://huggingface.co/datasets/lksy/ru_instruct_gpt4
- Human-generated - https://huggingface.co/datasets/OpenAssistant/oasst2

## Models

- Mistral-7B-Instruct-v0.2 - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

## Setup

```shell
conda create -n bots python=3.10;
conda activate bots;
pip install -r requirements.txt;
pip install .
```

## Train


```shell
# Скачиваем данные (на голове)
python src/bots/download/from_huggingface.py --data_from_gpt4
```

```shell
# Готовим данные для обучения модели (на узле!!!)
python src/bots/data/preprocessing.py --data_from_gpt4
```

Поменять пути в конфиге на свои `configs/mistral-7B-exp.conf`

P.S.: Модель уже есть на кластере в папке groups - до нее путь оставить как есть

```shell
# Учить тоже на узле!!!
# Учим модель, --nnodes - количество узлов кластера (всегда 1!!!), --nproc_per_node - количество видеокарт (максимум 2)
torchrun --nnodes=1 --nproc_per_node=1 src/bots/pipeline/ddp_train_script.py --config configs/mistral-7B-exp.conf --namespace ru_instruct_gpt4_ddp
```

Посмотреть логи обучения
```shell
# Если на узле, то указать хост узла
mlflow server --backend-store-uri log/mistral-7B-lora --host 127.0.0.1 --port 6006
```