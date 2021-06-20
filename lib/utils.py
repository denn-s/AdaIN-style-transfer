from typing import Dict, Any
import logging

import yaml
from PIL import Image
import torchvision.transforms as T


def load_image(image_file_path: str, device=None):
    img = Image.open(image_file_path)

    transforms = T.Compose([
        # T.Resize(512),
        # T.CenterCrop([256, 256]),
        T.ToTensor()
    ])

    tensor = transforms(img)
    tensor = tensor.unsqueeze_(0)

    if device:
        tensor = tensor.to(device)

    return tensor


def setup_logger(logger_name, log_file_path):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_config(config: Dict[str, Any], config_file_path: str):

    config_dict = vars(config)

    with open(config_file_path, 'w') as out_file:
        yaml.dump(config_dict, out_file, default_flow_style=False)
