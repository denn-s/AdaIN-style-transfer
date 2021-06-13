import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        # modified VGG19 architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),  # relu1_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),  # relu2_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),  # relu3_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),  # relu4_1
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), ceil_mode=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, input_image):
        return self.features(input_image)
