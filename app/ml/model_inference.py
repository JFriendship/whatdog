import torch
import torchvision.models as models
from PIL import Image
from app.ml.data_pipeline import get_transformations, load_imagefolder_dataset, clean_label_mapping
from app.ml.model_pipeline import load_and_finetune_resnet18
import os

def model_inference(image, device="cpu"):
    _, test_transform = get_transformations()
    img = Image.open(image)
    transformed_image = test_transform(img).unsqueeze(0)

    model = load_and_finetune_resnet18()

    checkpoint = torch.load("trained_models/resnet18-ft-classification-layer-75-percent-accuracy.pt")
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(transformed_image)
        _, classification = output.max(1)

    return classification.item()

def get_dataset_label_mapping():
    current_dir = os.getcwd()
    root_dir = current_dir + "/Images"
    dataset = load_imagefolder_dataset(root_dir=root_dir)
    return clean_label_mapping(dataset=dataset)