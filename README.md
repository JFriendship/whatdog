# Whatdog

Whatdog is a dog-breed classifier built with PyTorch Lightning and served through FastAPI. It fine-tunes the classification head of an ImageNet-pretrained ResNet-18 to recognize the 120 breeds in the Stanford Dogs dataset, then returns the most likely breeds for an uploaded image.

## Features

- Crops training images using the Stanford Dogs bounding-box annotations
- Uses a frozen ResNet-18 backbone with a trainable classification head
- Applies reproducible 70/15/15 training, validation, and test splits
- Saves the best and latest Lightning checkpoints during training
- Exposes top-k predictions through a versioned REST API
- Accepts JPEG and PNG uploads and validates invalid image data

## Tech stack

- Python
- PyTorch and torchvision
- Lightning
- TorchMetrics
- FastAPI and Pydantic
- Pillow
- pytest

## Getting started

Clone the repository, create a virtual environment, and install the dependencies:

```bash
git clone https://github.com/JFriendship/whatdog.git
cd whatdog

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

> The first training run may download the pretrained ResNet-18 weights.

## Dataset

Whatdog trains on the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), which contains 20,580 images across 120 dog breeds. A packaged version is also available from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset/data).

Download and extract both the images and annotations into the following layout:

```text
whatdog/
└── data/
    ├── Images/
    │   ├── n02085620-Chihuahua/
    │   └── ...
    └── Annotation/
        ├── n02085620-Chihuahua/
        └── ...
```

Each annotation file must have the same base name as its corresponding `.jpg` file. Samples without a matching annotation are skipped.

## Training

Run training with the default dataset paths and hyperparameters:

```bash
python -m app.training.train
```

By default, training runs for 10 epochs with a batch size of 32. Artifacts are written to `app/training/artifacts/`:

```text
artifacts/
├── checkpoints/
│   ├── last.ckpt
│   └── whatdog-<epoch>-<val_loss>.ckpt
├── class_names.json
└── training_logs/
```

Common options include:

```bash
python -m app.training.train \
  --images-dir data/Images \
  --annotations-dir data/Annotation \
  --output-dir app/training/artifacts \
  --batch-size 32 \
  --learning-rate 0.001 \
  --max-epochs 10 \
  --accelerator auto
```

To continue from a checkpoint, add `--resume-from path/to/last.ckpt`. Run `python -m app.training.train --help` to see every option.

## Running the API

The API loads a Lightning checkpoint and its matching class-name file at startup. Set their paths before starting the server:

```bash
export CHECKPOINT_PATH="app/training/artifacts/checkpoints/last.ckpt"
export CLASS_NAMES_PATH="app/training/artifacts/class_names.json"

uvicorn app.api.main:app --reload
```

The service is then available at `http://127.0.0.1:8000`:

- Interactive API docs: `http://127.0.0.1:8000/docs`
- Health check: `GET /health`
- Predictions: `POST /api/v2/predict`

### Make a prediction

Upload a JPEG or PNG using the multipart field named `file`. `top_k` is optional, defaults to `3`, and accepts values from 1 through 10.

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v2/predict?top_k=3" \
  -H "accept: application/json" \
  -F "file=@/path/to/dog.jpg"
```

Example response:

```json
{
  "predictions": [
    {"label": "golden retriever", "confidence": 0.82},
    {"label": "Labrador retriever", "confidence": 0.11},
    {"label": "cocker spaniel", "confidence": 0.03}
  ]
}
```

Confidence values are softmax probabilities between 0 and 1. If fewer classes exist than requested by `top_k`, the API returns one result per available class.

## Testing

Run the test suite from the repository root:

```bash
pytest
```

The tests cover dataset discovery and cropping, deterministic data splits, data-loader behavior, model freezing, training and evaluation steps, optimizer configuration, and a Lightning development run. Model tests use untrained weights, so they do not require a checkpoint or a dataset download.

## Project structure

```text
app/
├── api/
│   ├── inference.py       # Checkpoint loading, transforms, and prediction
│   ├── main.py            # FastAPI app and model lifespan
│   ├── routes.py          # Prediction endpoint
│   └── schemas.py         # API response models
├── tests/
│   ├── test_data_ingestion.py
│   └── test_model.py
└── training/
    ├── data_ingestion.py  # Dataset and Lightning data module
    ├── model.py           # Lightning ResNet-18 classifier
    └── train.py           # Training CLI
```

The previous v1 implementation, including its notebooks and Docker setup, is retained in [`v1_legacy_pytorch/`](v1_legacy_pytorch/README.md) for reference. The commands in this README describe the current v2 application.

## Model details

Training freezes the pretrained ResNet-18 backbone and replaces its final fully connected layer with a 120-class head. Images are cropped to the annotated dog, resized, center-cropped to 224 × 224, and normalized with ImageNet statistics. Training also applies horizontal flips, rotation, and brightness augmentation.

This project is currently under development.
