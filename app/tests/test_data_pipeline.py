import pytest
from torchvision.datasets.folder import ImageFolder
import ml.data_pipeline as dp
from torchvision import transforms
import torch

def test_load_image_folder_dataset(dummy_dataset):
    dummy_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = dp.load_imagefolder_dataset(root_dir=dummy_dataset, transform=dummy_transform)

    # Length of the dataset
    #   3 images for each of the 2 classes = 6
    assert len(dataset) == 6

    # Classes and Class to Index
    assert dataset.classes == ["dog_breed_1", "dog_breed_2"]
    assert dataset.class_to_idx == {"dog_breed_1": 0, "dog_breed_2": 1}

    # Item Retrieval
    img, label = dataset[0]

    # Data type and shape
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)

    # Label check
    assert label in [0, 1]


# Data splitting
def test_split_sizes(golden_dataset: ImageFolder):
    train, val, test = dp.split_dataset(
        dataset=golden_dataset, train_percentage=0.6, val_percentage=0.2
    )

    assert len(train) == 3
    assert len(val) == 1
    assert len(test) == 1

def test_split__total_size_preserved(golden_dataset: ImageFolder):
    train, val, test = dp.split_dataset(
        dataset=golden_dataset, train_percentage=0.6, val_percentage=0.2
    )

    assert len(train) + len(val) + len(test) == len(golden_dataset)

def test_split_no_overlap(golden_dataset: ImageFolder) -> None:
    train, val, test = dp.split_dataset(
        dataset=golden_dataset, train_percentage=0.6, val_percentage=0.2
    )
        
# Transformations
def test_get_transformations_breaks():
    # length 4 and length 2
    mean, std = [0.5, 0.5, 0.5, 0.5], [0.5, 0.5]
    with pytest.raises(ValueError):
        dp.get_transformations(mean=mean, std=std)
    # length 2 and length 4
    mean, std = [0.5, 0.5], [0.5, 0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        dp.get_transformations(mean=mean, std=std)

# def test_create_dataloaders():
#     pass

# def test_load_one_image():
#     pass

# def test_clean_label_mapping():
#     pass
