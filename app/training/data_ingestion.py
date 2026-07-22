from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import os

class CroppedStanfordDogsDataset:
    def __init__(self, images_dir: str, annotations_dir: str, transform=None):
        """
        Args:
            images_dir: Path to the folder containing the stanford dogs images
            annotations_dir: Path to the folder containing the stanford dogs image annotations
            transform: Image transformations to be applied after cropping the image around the annotations
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.samples = self._build_dataset()

    def _build_dataset(self):
        """Builds a list mapping image paths to their corresponding XML paths and labels."""

        samples = []

        # Example folder format: 'n02085620-Chihuahua'
        breeds = sorted(os.listdir(self.images_dir))

        for label_idx, breed_folder in enumerate(breeds):
            img_folder_path = os.path.join(self.images_dir, breed_folder)
            xml_folder_path = os.path.join(self.annotations_dir, breed_folder)

            if not os.path.isdir(img_folder_path):
                continue

            for file_name in os.listdir(img_folder_path):
                if file_name.endswith('.jpg'):
                    # The XML file has the same name as the image, without the .jpg extension
                    file_base = os.path.splitext(file_name)[0]
                    img_path = os.path.join(img_folder_path, file_name)
                    xml_path = os.path.join(xml_folder_path, file_base)

                    if os.path.exists(xml_path):
                        samples.append({
                            'image_path': img_path,
                            'xml_path': xml_path,
                            'label': label_idx
                        })

        return samples

    def _get_bounding_box(self, xml_path):
        """Parses the XML file to find the bounding box coordinates."""

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Grab the first bounding box found in the XML
        bndbox = root.find('object').find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        return (xmin, ymin, xmax, ymax)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load the image using PIL
        with Image.open(sample['image_path']) as img:
            image = img.convert('RGB')

        # 2. Extract bounding box and crop
        box = self._get_bounding_box(sample['xml_path'])
        image = image.crop(box)

        # 3. Apply standard transforms (Resize, ToTensor, Normalize, Augmentations)
        if self.transform:
            image = self.transform(image)

        return image, sample['label'] 