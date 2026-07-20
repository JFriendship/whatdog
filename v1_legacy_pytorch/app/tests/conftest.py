import pytest
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

@pytest.fixture
def dummy_dataset(tmp_path):
    dataset_dir = tmp_path / "fake_dataset"
    dataset_dir.mkdir()

    for class_name in ["dog_breed_1", "dog_breed_2"]:
        class_dir = dataset_dir / class_name
        class_dir.mkdir()

        for i in range(3):
            img_path = class_dir / f"img_{i}.png"
            img = Image.new('RGB', (256, 256))
            img.save(img_path)
    
    return dataset_dir

@pytest.fixture(scope="session")
def golden_dataset():
    # Probably need to add some transforms here
    golden_transform = transforms.ToTensor()
    gold_dataset = ImageFolder("tests/golden_dataset", golden_transform)
    return gold_dataset
