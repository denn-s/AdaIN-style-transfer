import os
import argparse
import random

import torch
import torchvision.transforms as T

from lib.models.encoder_decoder_net import EncoderDecoderNet
from lib.utils import load_image


def test(args: argparse.Namespace):
    print('using content_image_file_path: {}'.format(args.content_image_file_path))
    print('using style_image_file_path: {}'.format(args.style_image_file_path))
    print('using model_file_path: {}'.format(args.model_file_path))

    print('using output_file_path: {}'.format(args.output_file_path))
    output_dir_path = os.path.dirname(args.output_file_path)
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device: {}'.format(device))

    content_image = load_image(args.content_image_file_path, device)
    print('loaded content image: {}'.format(content_image.shape))

    style_image = load_image(args.style_image_file_path, device)
    print('loaded style image: {}'.format(style_image.shape))

    encoder_decoder_net = EncoderDecoderNet()
    encoder_decoder_net.load_state_dict(torch.load(args.model_file_path))
    encoder_decoder_net.to(device)
    encoder_decoder_net.eval()
    print('prepared network')

    output_tensor, _ = encoder_decoder_net(content_image, style_image)
    print('calculated output: {}'.format(output_tensor.shape))

    output_data = output_tensor.cpu()[0]
    output_image = T.ToPILImage()(output_data)

    output_image.save(args.output_file_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-image-file-path', type=str, required=True)
    parser.add_argument('--style-image-file-path', type=str, required=True)
    parser.add_argument('--output-file-path', type=str, required=True)
    parser.add_argument('--model-file-path', type=str, required=True)

    args = parser.parse_args()

    # set random seed for reproducibility
    manual_seed = 42

    print("random seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    test(args)


if __name__ == '__main__':
    main()
