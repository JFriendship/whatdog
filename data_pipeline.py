import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path

def load_imagefolder_dataset(root_dir: str, transform=None):
    """
    Loads the dataset from the provided root directory.
    Args:
        root_dir (string): The directory that contains the images.
        transform (Compose): A sequence of transformations to apply to the entire dataset.
    Returns:
        ImageFolder: the image folder dataset created from the provided directory.
    """

    image_folder_dataset = datasets.ImageFolder(root=root_dir, transform=None)
    return image_folder_dataset

def split_dataset(dataset, train_percentage: float = 0.7, val_percentage: float = 0.15, seed: int = 24):
    """
    Splits the dataset into train/test splits
    Args:
        dataset: The dataset that will be split
        train_percentage (float): The percentage of the dataset that will be used for training
        val_percentage (float): The percentage of the dataset that will be used for validation
        seed (int): The random seed used for the generator
    Returns:
        train_dataset (Subset): The portion of the dataset for training.  
        val_dataset (Subset): The portion of the dataset for validation.   
        test_dataset (Subset): The portion of the dataset for testing.   
    """

    # Type Checks
    if not isinstance(train_percentage, float) or not isinstance(val_percentage, float):
        print("TYPE ERROR: train_percentage and val_percentage have to be floats.")
        return None, None, None
    
    if not isinstance(seed, int):
        print("TYPE ERROR: seed has to be an int.")
        return None, None, None

    # Value Checks
    if (train_percentage + val_percentage > 1.0 or 
        train_percentage <= 0.0 or 
        val_percentage <= 0.0):
        print("===== Value Error =====")
        print("train_percentage must be within (0,1]")
        print("val_percentage must be between (0,1]")
        print("train_percentage + val_percentage must be <= 1.0")

    # Use a generator for reproducability
    generator = torch.Generator().manual_seed(seed)

    # Calculate the sizes of the train/val/test datasets
    train_size = int(train_percentage * len(dataset))
    val_size = int(val_percentage * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset into train/val/test subsets
    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset=dataset, 
        lengths=[train_size, val_size, test_size], 
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset

def get_transformations(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Returns the transformations for the train/test set. 
    The data will be normalized based on the mean and standard deviation of the Imagenet1k dataset by default.
    Args:
        mean(list): A list of RGB mean values based on the dataset.
        std(list): A list of RGB standard deviation values based on the dataset.
    Returns:
        train_transform (Compose): The torchvision transformations for the train dataset.  
        test_val_transform (Compose): The torchvision transformations for the validation and test datasets.  
    """
    if not isinstance(mean, list) or not isinstance(std, list):
        print("both mean and std have to be lists.")
        return None, None

    if not len(mean) == 3 or not len(std) == 3:
        print("mean and std must both have a length of 3.")
        return None, None

    train_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomRotation(0.15),
        transforms.ColorJitter(brightness=0.2),

        transforms.ToTensor(),
        # Normalize based on Imagenet1k dataset
        transforms.Normalize(mean=mean, std=std)
    ])

    # The test_val transform will be used for the test dataset and the validation dataset
    test_val_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalize based on Imagenet1k dataset
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_val_transform

def create_dataloaders(train_dataset, val_dataset, test_dataset, train_batch_size: int=32,
                       val_batch_size: int=64, test_batch_size: int=64, train_shuffle: bool=True):
    """
    Creates pytorch DataLoaders for the provided train/val/test datasets.
    Args:
        train_dataset (dataset): The training dataset.
        val_dataset (dataset): The validation dataset.
        test_dataset (dataset): the test dataset.
        train_batch_size (int): The size of training batches.
        val_batch_size (int): The size of validation batches.
        test_batch_size (int): The size of test batches.
        train_shuffle (bool): Whether or not to shuffle when loading data from the training dataset.
    Returns:
        train_loader (DataLoader): DataLoader for training.
        val_loader (DataLoader): DataLoader for validation.
        test_loader (DataLoader): DataLoader for testing.
    """

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    return train_loader, val_loader, test_loader

def load_one_image(dataloader):
    """
    Loads one image from the provided DataLoader. Best used with a dataloader that shuffles.
    Args:
        dataloader (DataLoader): A data loader that can load images.
    Returns:
        int: Label that feeds into an int to string label mapping.
    """
    # Load one image
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()

    # Denormalize the image
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    denorm_img = img * std_t + mean_t

    # Display the image
    label = train_labels[0]
    img_data_transposed = denorm_img.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_data_transposed)
    plt.show()

    return label

def clean_label_mapping(dataset):
    """
    Cleans the class folder names in the label:index mapping provided by PyTorch Datasets.
    Args:
        dataset (Dataset): A PyTorch Dataset.
    Returns:
        dictionary: a cleaned class index to label mapping using the dataset's class_to_idx dictionary.

    """

    label_mapping = dataset.class_to_idx

    label_mapping = {i: label[label.find('-')+1:] for label, i in label_mapping.items()}

    return label_mapping

class TransformedSubset(Dataset):
    """
    A class used to apply image transformations to subsets of a dataset
    """
    def __init__(self, subset, transform=None):
        """
        Initializes a TransformedSubset object.

        Args:
            subset: A PyTorch Subset object that has a portion of a dataset.
            transform (callable, optional): An optional transform that will be applied to the subset images.
        """

        self.subset = subset
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the subset.
        """

        return len(self.subset)
    
    def __getitem__(self, index):
        """
        Returns the image, label pair at the requested index and performs transforms if available.

        Args:
            index (int): the index of the image label pair that is being requested.

        Returns:
            Tensor or Image: The tensor representation of the requested image or the PIL Image representation if there is no ToTensor() transformation.  
            int: The integer class label associated with the image.  
        """

        image, label = self.subset[index]

        if self.transform:
            image = self.transform(image)

        return image, label
    
