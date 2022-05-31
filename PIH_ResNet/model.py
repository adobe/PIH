import numpy as np
import torch
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
from resnet import resnet18


class ColorStep1(torch.nn.Module):
    """
    Differentiable implementation of Color Transformation - Brightness, Contrast, Sturation.

    Implementation borrowed heavily from F.transform_tensor.
    """

    def __init__(self):
        super(ColorStep1, self).__init__()

    def _blend(self, img1, img2, ratio):

        bound = 1.0

        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound)

    def rgb_to_grayscale(self, img):
        r, g, b = img.unbind(dim=-3)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)

        return l_img

    def adjust_brightness(self, img, brightness_factor):

        return self._blend(img, torch.zeros_like(img), brightness_factor)

    def adjust_contrast(self, img, contrast_factor):

        mean = torch.mean(self.rgb_to_grayscale(img), dim=(-3, -2, -1), keepdim=True)

        return self._blend(img, mean, contrast_factor)

    def adjust_saturation(self, img, saturation_factor):

        return self._blend(img, self.rgb_to_grayscale(img), saturation_factor)

    def forward(self, image, feature, input_mask):
        """Change the luminosity of an input image.
        Args:
            image (Pytorch Tensor): Original input image.
            feature (Pyotrch Tensor):Output of the resnet - predicting the parameters.
            mask (Pytorch Tesnor): Alpha matte to compose image.
        Returns:
            input_composite: first composite results.
        """

        brightness = feature[0, 0]
        contrast = feature[0, 1]
        saturation = feature[0, 2]

        input_out = self.adjust_brightness(image, brightness)

        input_out = self.adjust_contrast(input_out, contrast)

        input_out = self.adjust_saturation(input_out, saturation)

        input_composite = input_out * input_mask + (1 - input_mask) * image

        return input_composite


class ColorStep2(torch.nn.Module):
    """
    Implementation of differentiable color transfor function.
    Cubic function for R,G,B spaces.
    """

    def __init__(self):
        super(ColorStep2, self).__init__()

    def forward(self, input_composite, output_color, input_mask, input_image):

        a_r = 0
        b_r = output_color[0, 1]
        c_r = output_color[0, 2]
        d_r = output_color[0, 3]

        a_g = 0
        b_g = output_color[0, 5]
        c_g = output_color[0, 6]
        d_g = output_color[0, 7]

        a_b = 0
        b_b = output_color[0, 9]
        c_b = output_color[0, 10]
        d_b = output_color[0, 11]

        # color_out = (input_composite * a + input_composite*input_composite *b).clamp(0,1)
        color_out_r = (
            input_composite[:, 0, ...]
            * input_composite[:, 0, ...]
            * input_composite[:, 0, ...]
            * d_r
            + input_composite[:, 0, ...] * input_composite[:, 0, ...] * c_r
            + input_composite[:, 0, ...] * b_r
            + torch.ones_like(input_composite[:, 0, ...]) * a_r
        ).clamp(0, 1)
        color_out_g = (
            input_composite[:, 1, ...]
            * input_composite[:, 1, ...]
            * input_composite[:, 1, ...]
            * d_g
            + input_composite[:, 1, ...] * input_composite[:, 1, ...] * c_g
            + input_composite[:, 1, ...] * b_g
            + torch.ones_like(input_composite[:, 1, ...]) * a_g
        ).clamp(0, 1)
        color_out_b = (
            input_composite[:, 2, ...]
            * input_composite[:, 2, ...]
            * input_composite[:, 2, ...]
            * d_b
            + input_composite[:, 2, ...] * input_composite[:, 2, ...] * c_b
            + input_composite[:, 2, ...] * b_b
            + torch.ones_like(input_composite[:, 2, ...]) * a_b
        ).clamp(0, 1)

        color_out = torch.cat(
            (
                color_out_r.unsqueeze(1),
                color_out_g.unsqueeze(1),
                color_out_b.unsqueeze(1),
            ),
            1,
        )

        output_composite = color_out * input_mask + (1 - input_mask) * input_image

        return output_composite


class Model(torch.nn.Module):
    def __init__(self, feature_dim=3, color_space=12, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.network = resnet18(num_classes=feature_dim).to(device)

        self.color = resnet18(num_classes=color_space, input_f=7).to(device)

        self.CT1 = ColorStep1()
        self.CT2 = ColorStep2()

    def initiate(self, input_image, input_mask, gt):
        self.input_image = input_image.to(self.device)
        self.input_mask = input_mask.to(self.device)
        self.gt = gt.to(self.device)

    def forward(self):
        input_all = torch.cat((self.input_image, self.input_mask), 1)

        embeddings = self.network(input_all)[0]

        self.brightness = embeddings[0, 0]
        self.contrast = embeddings[0, 1]
        self.saturation = embeddings[0, 2]

        input_composite = self.CT1(self.input_image, embeddings, self.input_mask)

        inputs_color = torch.cat(
            (self.input_image, input_composite, self.input_mask), 1
        )

        embeddings_2 = self.color(inputs_color)[0]

        self.b_r = embeddings_2[0, 1]
        self.b_g = embeddings_2[0, 5]
        self.b_b = embeddings_2[0, 9]

        output_composite = self.CT2(
            input_composite, embeddings_2, self.input_mask, self.input_image
        )
        # inter_features = self.network(images)[1]

        return input_composite, output_composite
