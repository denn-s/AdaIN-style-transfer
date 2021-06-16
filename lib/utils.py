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
