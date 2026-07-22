import os
import pytest
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from training import CroppedStanfordDogsDataset, WhatdogDataModule
from torch.utils.data.sampler import RandomSampler, SequentialSampler

@pytest.fixture
def mock_stanford_dogs_dir(tmp_path):
    """
    Fixture to create a temporary mock Stanford Dogs dataset structure:
    tmp_path/
      Images/
        n02085620-Chihuahua/
          dog1.jpg
      Annotations/
        n02085620-Chihuahua/
          dog1
    """
    images_dir = tmp_path / "Images"
    annotations_dir = tmp_path / "Annotations"
    
    breed_img_dir = images_dir / "n02085620-Chihuahua"
    breed_ann_dir = annotations_dir / "n02085620-Chihuahua"
    
    breed_img_dir.mkdir(parents=True)
    breed_ann_dir.mkdir(parents=True)

    # 1. Create a dummy 100x100 solid color image
    img_path = breed_img_dir / "dog1.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    # 2. Create the corresponding XML annotation file
    # Bounding box: xmin=10, ymin=20, xmax=50, ymax=60 (Width=40, Height=40)
    xml_content = """<annotation>
        <object>
            <bndbox>
                <xmin>10</xmin>
                <ymin>20</ymin>
                <xmax>50</xmax>
                <ymax>60</ymax>
            </bndbox>
        </object>
    </annotation>"""
    xml_path = breed_ann_dir / "dog1"
    xml_path.write_text(xml_content)

    return str(images_dir), str(annotations_dir)

def test_dataset_indexing_and_length(mock_stanford_dogs_dir):
    """
    Ensures the dataset finds the files, correctly maps labels, and returns a valid length.
    """
    images_dir, annotations_dir = mock_stanford_dogs_dir
    transform = transforms.ToTensor()
    
    dataset = CroppedStanfordDogsDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transform=transform
    )

    assert len(dataset) == 1, "Dataset length should reflect the 1 valid sample."
    
    image_tensor, label = dataset[0]
    
    assert isinstance(image_tensor, torch.Tensor), "Output should be a PyTorch Tensor."
    assert label == 0, "The first alphabetical folder should map to integer label 0."

def test_bounding_box_cropping_math(mock_stanford_dogs_dir):
    """
    Verifies that the bounding box logic accurately parses the XML and crops the PIL image.
    """
    images_dir, annotations_dir = mock_stanford_dogs_dir
    
    # Initialize without PyTorch transforms so we can inspect raw PIL Dimensions
    dataset = CroppedStanfordDogsDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transform=None
    )

    cropped_img, _ = dataset[0]
    
    # Expected Width: 50 (xmax) - 10 (xmin) = 40
    # Expected Height: 60 (ymax) - 20 (ymin) = 40
    expected_width = 40
    expected_height = 40

    assert cropped_img.size == (expected_width, expected_height), \
        f"Expected size {(expected_width, expected_height)}, got {cropped_img.size}"

def test_handling_of_missing_annotations(mock_stanford_dogs_dir):
    """
    Ensures that an image without a matching XML file is safely skipped rather than crashing.
    """
    images_dir, annotations_dir = mock_stanford_dogs_dir
    
    # Add an image without a matching XML file (an orphan image)
    orphan_img_path = os.path.join(images_dir, "n02085620-Chihuahua", "orphan.jpg")
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(orphan_img_path)

    dataset = CroppedStanfordDogsDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir
    )

    # Dataset length should STILL be 1 (the original dog1.jpg), and the orphan is safely ignored
    assert len(dataset) == 1, "Orphan image without XML file should be excluded."

def test_dataloader_batching_compatibility(mock_stanford_dogs_dir):
    """
    Ensures the dataset integrates properly with PyTorch's DataLoader for batching.
    """
    images_dir, annotations_dir = mock_stanford_dogs_dir
    
    # Apply standard normalization and resizing required for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = CroppedStanfordDogsDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    images_batch, labels_batch = next(iter(dataloader))
    
    assert images_batch.shape == (1, 3, 224, 224), "Batch tensor shape must match (B, C, H, W)."
    assert labels_batch.shape == (1,), "Target labels batch shape must match (B,)."


# =============================
# LIGHTNING DATA MODULE TESTING
# =============================

def test_datamodule_split_proportions(mocker):
    """
    Verifies that the 70/15/15 split math calculates correctly using mocker.
    """
    mock_dataset_class = mocker.patch("training.data_ingestion.CroppedStanfordDogsDataset")
    
    mock_instance = mock_dataset_class.return_value
    mock_instance.__len__.return_value = 100

    module_instance = WhatdogDataModule(
        images_dir="dummy/Images",
        annotations_dir="dummy/Annotations",
        batch_size=16,
        num_workers=0
    )
    module_instance.setup()

    assert len(module_instance.train_data) == 70, "Train split should be exactly 70%."
    assert len(module_instance.val_data) == 15, "Validation split should be exactly 15%."
    assert len(module_instance.test_data) == 15, "Test split should contain the remainder."


def test_no_data_leakage(mocker):
    """
    Verifies that no single image index exists in more than one subset using mocker.
    """
    mock_dataset_class = mocker.patch("training.data_ingestion.CroppedStanfordDogsDataset")
    mock_instance = mock_dataset_class.return_value
    mock_instance.__len__.return_value = 1000

    module_instance = WhatdogDataModule(
        images_dir="dummy/Images",
        annotations_dir="dummy/Annotations",
        batch_size=16,
        num_workers=0
    )
    module_instance.setup()

    train_indices = set(module_instance.train_data.indices)
    val_indices = set(module_instance.val_data.indices)
    test_indices = set(module_instance.test_data.indices)

    assert train_indices.isdisjoint(val_indices), "Leakage: Train and Val sets share images!"
    assert train_indices.isdisjoint(test_indices), "Leakage: Train and Test sets share images!"
    assert val_indices.isdisjoint(test_indices), "Leakage: Val and Test sets share images!"


def test_dataloader_configuration(mocker):
    """
    Verifies DataLoaders are configured correctly and only training is shuffled using mocker.
    """
    mock_dataset_class = mocker.patch("training.data_ingestion.CroppedStanfordDogsDataset")
    mock_instance = mock_dataset_class.return_value
    mock_instance.__len__.return_value = 100

    module_instance = WhatdogDataModule(
        images_dir="dummy/Images",
        annotations_dir="dummy/Annotations",
        batch_size=16,
        num_workers=0
    )
    module_instance.setup()

    train_loader = module_instance.train_dataloader()
    val_loader = module_instance.val_dataloader()
    test_loader = module_instance.test_dataloader()

    assert train_loader.batch_size == 16
    assert val_loader.batch_size == 16
    assert test_loader.batch_size == 16

    assert isinstance(train_loader.sampler, RandomSampler), "Train loader MUST be shuffled."
    assert isinstance(val_loader.sampler, SequentialSampler), "Validation loader MUST NOT be shuffled."
    assert isinstance(test_loader.sampler, SequentialSampler), "Test loader MUST NOT be shuffled."