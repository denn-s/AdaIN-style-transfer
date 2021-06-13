from typing import Dict

import os
import argparse
import random
from copy import deepcopy
import math
import datetime

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from lib.models.encoder import Encoder
from lib.models.decoder import Decoder
from lib.models.encoder_decoder_net import EncoderDecoderNet
from lib.datasets.image_dataset import ImageDataset
from lib.adain import AdaIN


def calc_content_loss(content_target, output_image):
    content_loss = torch.nn.MSELoss()
    return content_loss(output_image, content_target)


def calc_style_loss(style_target, style_output):
    # print('style_target keys: {}'.format(style_target.keys()))
    # print('style_ouput keys: {}'.format(style_output.keys()))

    style_loss = torch.nn.MSELoss()

    style_targets = list(style_target.values())
    style_outputs = list(style_output.values())

    mean_style_act, std_style_act = AdaIN.calc_mean_std(style_targets[0])
    mean_output_act, std_output_act = AdaIN.calc_mean_std(style_outputs[0])

    sum_mean = style_loss(mean_style_act, mean_output_act)
    sum_std = style_loss(std_style_act, std_output_act)

    for style_act, output_act in zip(style_targets[1:], style_outputs[1:]):
        # print('style_target: {}'.format(style_act.shape))
        # print('style_output: {}'.format(output_act.shape))

        mean_style_act, std_style_act = AdaIN.calc_mean_std(style_act)
        mean_output_act, std_output_act = AdaIN.calc_mean_std(output_act)
        # print('mean_style_act: {}'.format(mean_style_act.shape))
        # print('mean_output_act: {}'.format(mean_output_act.shape))
        # print('std_style_act: {}'.format(std_style_act.shape))
        # print('std_output_act: {}'.format(std_output_act.shape))

        sum_mean += style_loss(mean_style_act, mean_output_act)
        sum_std += style_loss(std_style_act, std_output_act)

    return sum_mean + sum_std


def train(args: argparse.Namespace):
    print('using output_dir_path: {}'.format(args.output_dir_path))
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path)

    dt_str = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
    checkpoint_dir_path = os.path.join(args.output_dir_path, 'train_' + dt_str)
    print('using checkpoint_dir_path: {}'.format(checkpoint_dir_path))
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)

    writer = SummaryWriter(log_dir=checkpoint_dir_path)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('using device: {}'.format(device))

    print('using content_images_dir_path: {}'.format(args.content_images_dir_path))
    if not os.path.isdir(args.content_images_dir_path):
        print('error: not a directroy: {}'.format(args.content_images_dir_path))

    print('using style_images_dir_path: {}'.format(args.style_images_dir_path))
    if not os.path.isdir(args.style_images_dir_path):
        print('error: not a directroy: {}'.format(args.style_images_dir_path))

    print('using encoder_model_file_path: {}'.format(args.encoder_model_file_path))
    encoder = Encoder()
    encoder.features.load_state_dict(torch.load(args.encoder_model_file_path))

    decoder = Decoder()
    if args.decoder_model_file_path is not None:
        decoder.load_state_dict(torch.load(args.decoder_model_file_path))

    encoder_num_layers = 30
    model = EncoderDecoderNet(encoder, encoder_num_layers, decoder)

    model.to(device)
    model.train()
    print('prepared network')

    transforms = T.Compose([
        T.Resize(512),
        T.RandomCrop([256, 256]),
        T.ToTensor()
    ])

    content_loader = DataLoader(ImageDataset(args.content_images_dir_path, transform=transforms),
                                batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)
    print('content_loader len: {}'.format(len(content_loader)))

    style_loader = DataLoader(ImageDataset(args.style_images_dir_path, transform=transforms),
                              batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    print('style_loader len: {}'.format(len(style_loader)))

    lr_scheduler_gamma = 0.1
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_scheduler_gamma)

    max_iterations = int(min([len(content_loader), len(style_loader)]))
    print('using max_iterations: {}'.format(max_iterations))

    encoder_activations = {}

    def get_activation(layer_num: int):
        def hook(model: torch.nn.Module, input: torch.Tensor, output: torch.Tensor):
            encoder_activations[str(layer_num)] = output.clone().detach()
            # print('activation_hook -> layer_num: {}'.format(layer_num))
        return hook

    # vgg19 relu1_1, relu2_1, relu3_1, relu4_1
    enc_style_layers = [3, 10, 17, 30]

    for layer_pos in enc_style_layers:
        model.encoder[layer_pos].register_forward_hook(get_activation(layer_pos))

    digits_epoch = int(math.log10(args.num_epochs)) + 1
    digits_iter = int(math.log10(max_iterations)) + 1
    print_str = 'epoch: [{:>' + str(digits_epoch) + '}|{}], iter: [{:>' + str(
        digits_iter) + '}|{}] -> content loss: {:>10.5f}, style loss: {:>10.5f}, total loss: {:>10.5f}'

    content_input = next(iter(content_loader)).to(device)
    style_input = next(iter(style_loader)).to(device)
    writer.add_graph(model, (content_input, style_input))

    encoder_activations = {}

    for epoch in range(args.num_epochs):

        content_iter = iter(content_loader)
        style_iter = iter(style_loader)

        for i in range(max_iterations):

            try:
                content_batch = next(content_iter).to(device)
                style_batch = next(style_iter).to(device)
            except Exception as e:
                print('exception occurred: {}'.format(repr(e)))
                continue

            output_image, target_fm = model(content_batch, style_batch)
            # print('train -> output_image: {}'.format(output_image.shape))

            target_activations = deepcopy(encoder_activations)
            # print('train -> encoder_activations kees: {}'.format(encoder_activations.keys()))
            encoder_activations = {}

            output_fm = model.encoder(output_image)
            # print('train -> output_fm: {}'.format(output_fm.shape))

            output_activations = deepcopy(encoder_activations)
            # print('train -> encoder_activations kees: {}'.format(encoder_activations.keys()))
            encoder_activations = {}

            content_loss = calc_content_loss(target_fm, output_fm)
            style_loss = args.style_weight * calc_style_loss(target_activations, output_activations)
            loss = content_loss + style_loss

            if i % args.log_n_iter == 0:
                print(
                    print_str.format(epoch + 1, args.num_epochs, i + 1, max_iterations, content_loss, style_loss, loss))

                global_step = (epoch * max_iterations) + i

                writer.add_scalar('loss_content', content_loss, global_step)
                writer.add_scalar('loss_style', style_loss, global_step)
                writer.add_scalar('total_loss', loss, global_step)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step)

                for name, weight in model.decoder.named_parameters():
                    writer.add_histogram(name, weight, global_step)

            if i % args.image_n_iter == 0:
                images_grid = torchvision.utils.make_grid(torch.cat((content_batch, style_batch, output_image), 0),
                                                          args.batch_size)
                writer.add_image('images', images_grid, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if epoch % args.save_n_epochs == 0 or epoch == args.num_epochs - 1:
            model_file_path = os.path.join(checkpoint_dir_path, 'epoch_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), model_file_path)

    writer.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content-images-dir-path', type=str, required=True)
    parser.add_argument('--style-images-dir-path', type=str, required=True)
    parser.add_argument('--output-dir-path', type=str, required=True)
    parser.add_argument('--encoder-model-file_path', type=str, required=True)
    parser.add_argument('--decoder-model-file-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--style-weight', type=float, default=10.0)
    parser.add_argument('--log-n-iter', type=int, default=100)
    parser.add_argument('--image-n-iter', type=int, default=500)
    parser.add_argument('--save-n-epochs', type=int, default=1)

    args = parser.parse_args()

    # set random seed for reproducibility
    manual_seed = 42

    print("random seed: ", manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    train(args)


if __name__ == '__main__':
    main()
