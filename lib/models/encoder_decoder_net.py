import torch
import torch.nn as nn

from lib.adain import AdaIN


class EncoderDecoderNet(nn.Module):

    def __init__(self, encoder_model: nn.Module, encoder_num_layers: int, decoder_model: nn.Module, ):
        super(EncoderDecoderNet, self).__init__()

        # first few layers from vgg19 up to relu4_1
        self.encoder = nn.Sequential(
            *list(encoder_model.features.children())[:encoder_num_layers + 1]
        )

        # freeze encoder weights
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.decoder = decoder_model

        # unfreeze encoder weights
        for parameter in self.decoder.parameters():
            parameter.requires_grad = True

        self.adain = AdaIN()

    def forward(self, input_content, input_style):

        # encode content and style images in feature space
        content_fm = self.encoder(input_content)
        # print('forward -> content_fm: {}'.format(content_fm.shape))

        style_fm = self.encoder(input_style)
        # print('forward -> style_fm: {}'.format(style_fm.shape))

        # feed both feature maps to AdaIN layer to produce target feature maps
        # equation 9
        target_fm = self.adain(content_fm, style_fm)
        # print('forward -> target_fm: {}'.format(target_fm.shape))

        # map target feature maps back to image space
        # generates stylized image
        # equation 10
        output_image = self.decoder(target_fm)
        # print('forward -> output_image: {}'.format(output_image.shape))

        return output_image, target_fm
