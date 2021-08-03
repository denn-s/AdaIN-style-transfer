import os
import argparse

from tqdm import tqdm

from tools.test import test


def test_all(content_image_dir_path: str, style_image_dir_path: str, output_image_dir_path: str,
             encoder_model_file_path: str, decoder_model_file_path: str):

    if not os.path.exists(output_image_dir_path):
        os.makedirs(output_image_dir_path)

    # content and style image pairs as used by the paper authors
    content_style_pairs = [
        ['avril_cropped.jpg', 'impronte_d_artista_cropped.jpg'],
        ['chicago_cropped.jpg', 'ashville_cropped.jpg'],
        ['cornell_cropped.jpg', 'woman_with_hat_matisse_cropped.jpg'],
        ['lenna_cropped.jpg', 'en_campo_gris_cropped.jpg'],
        ['modern_cropped.jpg', 'goeritz_cropped.jpg'],
        ['sailboat_cropped.jpg', 'sketch_cropped.png']
    ]

    for content_file_name, style_file_name in tqdm(content_style_pairs):
        content_image_file_path = os.path.join(content_image_dir_path, content_file_name)
        style_image_file_path = os.path.join(style_image_dir_path, style_file_name)

        content_image_name, file_ext = os.path.splitext(content_file_name)
        style_image_name, file_ext = os.path.splitext(style_file_name)

        output_file_name = content_image_name + '_stylized_' + style_image_name + file_ext
        output_image_file_path = os.path.join(output_image_dir_path, output_file_name)

        test(content_image_file_path, style_image_file_path, output_image_file_path, encoder_model_file_path,
             decoder_model_file_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-image-dir-path', type=str, required=True)
    parser.add_argument('--style-image-dir-path', type=str, required=True)
    parser.add_argument('--output-image-dir-path', type=str, required=True)
    parser.add_argument('--encoder-model-file-path', type=str, required=True)
    parser.add_argument('--decoder-model-file-path', type=str, required=True)

    args = parser.parse_args()

    test_all(args.content_image_dir_path, args.style_image_dir_path, args.output_image_dir_path,
             args.encoder_model_file_path, args.decoder_model_file_path)


if __name__ == '__main__':
    main()
