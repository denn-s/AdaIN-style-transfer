import os
import argparse

import torch
import torchvision.transforms as T

from lib.utils import load_image
from lib.models.encoder import Encoder
from lib.models.decoder import Decoder
from lib.models.encoder_decoder_net import EncoderDecoderNet


def test(content_image_file_path: str, style_image_file_path: str, output_image_file_path: str,
         encoder_model_file_path: str, decoder_model_file_path: str):
    print('using content_image_file_path: {}'.format(content_image_file_path))
    print('using style_image_file_path: {}'.format(style_image_file_path))

    print('using output_file_path: {}'.format(output_image_file_path))
    output_dir_path = os.path.dirname(output_image_file_path)
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device: {}'.format(device))

    content_image = load_image(content_image_file_path, device)
    print('loaded content image: {}'.format(content_image.shape))

    style_image = load_image(style_image_file_path, device)
    print('loaded style image: {}'.format(style_image.shape))

    print('using encoder_model_file_path: {}'.format(encoder_model_file_path))
    encoder = Encoder()
    encoder.features.load_state_dict(torch.load(encoder_model_file_path))
    print('loaded encoder network')

    print('using decoder_model_file_path: {}'.format(encoder_model_file_path))
    decoder = Decoder()
    decoder.load_state_dict(torch.load(decoder_model_file_path))
    print('loaded decoder network')

    encoder_num_layers = 30
    encoder_decoder_net = EncoderDecoderNet(encoder, encoder_num_layers, decoder)
    encoder_decoder_net.to(device)
    encoder_decoder_net.eval()
    print('prepared encoder-decoder network')

    output_tensor, _ = encoder_decoder_net(content_image, style_image)
    print('calculated output: {}'.format(output_tensor.shape))

    output_data = output_tensor.clip(0, 1).cpu()[0]
    output_image = T.ToPILImage()(output_data)

    output_image.save(output_image_file_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-image-file-path', type=str, required=True)
    parser.add_argument('--style-image-file-path', type=str, required=True)
    parser.add_argument('--output-image-file-path', type=str, required=True)
    parser.add_argument('--encoder-model-file-path', type=str, required=True)
    parser.add_argument('--decoder-model-file-path', type=str, required=True)

    args = parser.parse_args()

    test(args.content_image_file_path, args.style_image_file_path, args.output_image_file_path,
         args.encoder_model_file_path, args.decoder_model_file_path)


if __name__ == '__main__':
    main()
