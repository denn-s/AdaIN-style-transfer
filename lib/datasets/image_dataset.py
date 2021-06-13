import os

from PIL import Image
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, images_dir_path, transform=None):
        assert os.path.isdir(images_dir_path) is True

        self.images_dir_path = images_dir_path
        self.image_file_paths = [os.path.join(self.images_dir_path, image_file_name) for image_file_name in
                                 os.listdir(images_dir_path) if image_file_name.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.image_file_paths[idx]).convert('RGB')
        sample = self.transform(image)

        return sample
