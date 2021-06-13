import torch
import torch.nn as nn


class AdaIN(nn.Module):

    def __init__(self):
        super(AdaIN, self).__init__()

        self.epsilon = 0.00001

    @staticmethod
    def calc_mean_std(fm: torch.Tensor, epsilon=0.00001):
        # expected fm shape [N, C, H, W]
        # example: [4, 512, 32, 32]

        n = fm.shape[0]
        c = fm.shape[1]

        # print('fm: {}'.format(fm.shape))

        # equation 6
        fm_var = fm.view(n, c, -1).var(dim=2) + epsilon
        fm_std = fm_var.sqrt().view(n, c, 1, 1)

        # equation 5
        fm_mean = fm.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)

        return fm_mean, fm_std

    def forward(self, content_fm, style_fm):
        # equation 8
        mean_style_fm, std_style_fm = self.calc_mean_std(style_fm, self.epsilon)
        mean_content_fm, std_content_fm = self.calc_mean_std(content_fm, self.epsilon)

        # print('content_fm: {}'.format(content_fm.shape))
        # print('mean_content_fm: {}'.format(mean_content_fm.shape))
        # print('std_content_fm: {}'.format(std_content_fm.shape))

        mean_style_fm = mean_style_fm.expand(style_fm.shape)
        std_style_fm = std_style_fm.expand(style_fm.shape)

        # print('mean_content_fm: {}'.format(mean_content_fm.shape))
        # print('std_content_fm: {}'.format(std_content_fm.shape))

        # print('style_fm: {}'.format(style_fm.shape))
        # print('mean_style_fm: {}'.format(mean_style_fm.shape))
        # print('std_style_fm: {}'.format(std_style_fm.shape))

        mean_content_fm = mean_content_fm.expand(content_fm.shape)
        std_content_fm = std_content_fm.expand(content_fm.shape)

        # print('mean_style_fm: {}'.format(mean_style_fm.shape))
        # print('std_style_fm: {}'.format(std_style_fm.shape))

        target_fm = std_style_fm * ((content_fm - mean_content_fm) / std_content_fm) + mean_style_fm

        return target_fm
