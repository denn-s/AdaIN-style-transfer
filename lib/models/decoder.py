import torch
import torch.nn as nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        # the decoder mostly mirrors the encoder, with all pooling layers replaced by nearest up-sampling to
        # reduce checkerboard effects
        self.features = nn.Sequential(

            nn.Conv2d(512, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, padding_mode='reflect'),
            # nn.ReLU()
        )

    def forward(self, input_fm: torch.Tensor):
        return self.features(input_fm)
