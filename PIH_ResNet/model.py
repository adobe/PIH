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

        return (
            ratio[:, None, None, None] * img1
            + (1.0 - ratio[:, None, None, None]) * img2
        ).clamp(0, bound)

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

        brightness = feature[:, 0]
        contrast = feature[:, 1]
        saturation = feature[:, 2]

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
        b_r = output_color[:, 1]
        c_r = output_color[:, 2]
        d_r = output_color[:, 3]

        a_g = 0
        b_g = output_color[:, 5]
        c_g = output_color[:, 6]
        d_g = output_color[:, 7]

        a_b = 0
        b_b = output_color[:, 9]
        c_b = output_color[:, 10]
        d_b = output_color[:, 11]

        # color_out = (input_composite * a + input_composite*input_composite *b).clamp(0,1)
        color_out_r = (
            input_composite[:, 0, ...]
            * input_composite[:, 0, ...]
            * input_composite[:, 0, ...]
            * d_r[:, None, None]
            + input_composite[:, 0, ...]
            * input_composite[:, 0, ...]
            * c_r[:, None, None]
            + input_composite[:, 0, ...] * b_r[:, None, None]
            + torch.ones_like(input_composite[:, 0, ...]) * a_r
        ).clamp(0, 1)
        color_out_g = (
            input_composite[:, 1, ...]
            * input_composite[:, 1, ...]
            * input_composite[:, 1, ...]
            * d_g[:, None, None]
            + input_composite[:, 1, ...]
            * input_composite[:, 1, ...]
            * c_g[:, None, None]
            + input_composite[:, 1, ...] * b_g[:, None, None]
            + torch.ones_like(input_composite[:, 1, ...]) * a_g
        ).clamp(0, 1)
        color_out_b = (
            input_composite[:, 2, ...]
            * input_composite[:, 2, ...]
            * input_composite[:, 2, ...]
            * d_b[:, None, None]
            + input_composite[:, 2, ...]
            * input_composite[:, 2, ...]
            * c_b[:, None, None]
            + input_composite[:, 2, ...] * b_b[:, None, None]
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
    def __init__(self, feature_dim=3, color_space=12):
        super().__init__()
        self.feature_dim = feature_dim
        self.network = resnet18(num_classes=feature_dim)

        self.color = resnet18(num_classes=color_space, input_f=7)

        self.CT1 = ColorStep1()
        self.CT2 = ColorStep2()

    def forward(self, input_image, input_mask):

        # On the device

        input_all = torch.cat((input_image, input_mask), 1)

        embeddings = self.network(input_all)[0]

        brightness = embeddings[0, 0]
        contrast = embeddings[0, 1]
        saturation = embeddings[0, 2]

        input_composite = self.CT1(input_image, embeddings, input_mask)

        inputs_color = torch.cat((input_image, input_composite, input_mask), 1)

        embeddings_2 = self.color(inputs_color)[0]

        b_r = embeddings_2[0, 1]
        b_g = embeddings_2[0, 5]
        b_b = embeddings_2[0, 9]

        output_composite = self.CT2(
            input_composite, embeddings_2, input_mask, input_image
        )
        # inter_features = self.network(images)[1]

        return (
            input_composite,
            output_composite,
            [brightness, contrast, saturation],
            [b_r, b_g, b_b],
        )
