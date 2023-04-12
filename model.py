import numpy as np
import torch
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as FN

from utils.resnet import resnet18, resnet34, PIHNet, EffNetV2, resnet50, resnet101, MobileNetV3
from utils.resnet_ibn import resnet50_ibn_b
from utils.unet.unet_model import UNet
from utils.unet_dis import UNet_mask, UNet_mask_light

########### ------------------ import Midas


from torchvision.transforms import Compose
# from midas.dpt_depth import DPTDepthModel
# from midas.midas_net import MidasNet
# from midas.midas_net_custom import MidasNet_small
# from midas.transforms import Resize, NormalizeImage, PrepareForNet


class LUT3D(torch.nn.Module):
    """A layer to apply an RGBTable."""

    def __init__(self):
        super(LUT3D, self).__init__()

    def forward(self, lut, img):
        """
        Args:
            lut: LUT to be applied, a 5-d tensor. First dimension is batch.
                For a pixel of value (R, G, B), it maps it to
                R' = lut[:, 0, B, G, R],
                G' = lut[:, 1, B, G, R],
                B' = lut[:, 2, B, G, R].
                The "BGR" order is because torch.grid_sample assumes guide_map of
                shape D x H x W, but the coord in grid is (x, y, z) , which is
                (R, G, B) in our case.
            img: input images, of shape B x C x H x W, in range [0, 1]
        Returns:
            out: output images, of shape B x C x H x W
        """
        guidemap = img * 2.0 - 1.0
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous().unsqueeze(1)
        out = FN.grid_sample(
            lut, guidemap, mode="bilinear", padding_mode="border", align_corners=False
        ).squeeze(2)
        return out


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


class Model_Composite(torch.nn.Module):
    def __init__(self, feature_dim=3, color_space=12, LUT=False, LUTdim=8, curve=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.network = resnet34(
            num_classes=feature_dim, input_f=7
        )  ## Background - composite

        self.color = resnet34(
            num_classes=color_space, input_f=10
        )  ## Background - composite - intermediate

        self.CT1 = ColorStep1()
        self.CT2 = ColorStep2()
        self.lut = LUT
        self.lutdim = LUTdim
        self.curve = curve
        print(self.curve)
        print("lutdim: %d" % (self.lutdim))
        if self.lut:
            self.lutnet = resnet34(
                num_classes=3 * LUTdim * LUTdim * LUTdim,
                input_f=7,
                sigmoid=True,
            )
            self.LUT3D = LUT3D()

    def forward(self, background, input_image, input_mask):

        # On the device

        # input_all = torch.cat((input_image, background, input_mask), 1)
        # # input_all = torch.cat((input_image, background), 1)

        # embeddings = self.network(input_all)[0]

        # brightness = embeddings[0, 0]
        # contrast = embeddings[0, 1]
        # saturation = embeddings[0, 2]

        # input_composite = self.CT1(input_image, embeddings, input_mask)

        input_composite = input_image.clone()

        if self.curve:
            inputs_color = torch.cat(
                (input_image, background, input_composite, input_mask), 1
            )
            # inputs_color = torch.cat((input_image, background, input_composite), 1)

            embeddings_2 = self.color(inputs_color)[0]

            b_r = embeddings_2[0, 1]
            b_g = embeddings_2[0, 5]
            b_b = embeddings_2[0, 9]

            output_composite = self.CT2(
                input_composite, embeddings_2, input_mask, input_image
            )
        else:
            output_composite = input_composite.clone()
            b_r = input_image[0, 0, 0, 0]
            b_g = input_image[0, 0, 0, 0]
            b_b = input_image[0, 0, 0, 0]

        # inter_features = self.network(images)[1]

        if self.lut:
            inputs_lut = torch.cat(
                (
                    input_image,
                    background,
                    input_mask,
                ),
                1,
            )
            # inputs_lut = torch.cat(
            #     (
            #         input_image,
            #         background,
            #         input_composite,
            #         output_composite,
            #     ),
            #     1,
            # )
            lut_table = self.lutnet(inputs_lut)[0]
            lut_table = torch.reshape(
                lut_table,
                (lut_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
            )
            lut_composite = (
                self.LUT3D(lut_table, output_composite) * input_mask
                + (1 - input_mask) * output_composite
            )
            return (
                input_composite,
                lut_composite,
                [b_r, b_g, b_b],
                [b_r, b_g, b_b],
            )

        else:
            return (
                input_composite,
                output_composite,
                [b_r, b_g, b_b],
                [b_r, b_g, b_b],
            )


class Model_Composite_PL(torch.nn.Module):
    def __init__(
        self,
        dim=32,
        sigmoid=True,
        scaling=False,
        masking=False,
        brush=False,
        nosig=False,
        onlyupsample=False,
        maskoffset=0.5,
        maskconvkernel=1,
        swap=False,
        lut=False,
        lutdim=16,
        joint=False,
        PIHNet_bool=False,
        Vit_bool=False,
        Eff_bool=False,
        aggupsample=False,
        depthmap=False,
        bgshadow=False,
        ibn=False,
        dual=False,
        lowres=False,
        high_res = False,
        lrdata=False,
        light=False,
    ):
        super().__init__()
        self.dim = dim
        self.scaling = scaling
        self.lut = lut
        self.lutdim = lutdim
        self.joint = joint
        self.depthmap = depthmap
        self.PIHNet_bool = PIHNet_bool
        self.Vit_bool = Vit_bool
        self.Eff_bool = Eff_bool
        self.ibn = ibn
        self.bgshadow = bgshadow
        self.dual = dual
        self.lowres = lowres
        self.high_res = high_res
        self.lrdata = lrdata
        if bgshadow:
            gainout = 2
        else:
            gainout = 1

        if self.PIHNet_bool:
            self.colornet = PIHNet
            print("Using PIHNet!")
        elif self.Vit_bool:
            self.colornet = VitNet
            print("Using ViT!")
        elif self.Eff_bool:
            self.colornet = MobileNetV3
            print("Using MobileNetV3!")

        elif self.ibn:
            print("Using ResNet 50 with ibn")
            self.colornet = resnet50_ibn_b
        else:
            self.colornet = resnet50
            
            print("Using ResNet! 50!")

        if self.lut:
            print("Using LUT")
            print("lutdim: %d" % (self.lutdim))

        if scaling:
            if self.lut:
                self.PL = resnet34(
                    num_classes=3 * self.lutdim * self.lutdim * self.lutdim,
                    input_f=8,
                    sigmoid=sigmoid,
                )

            else:

                self.PL = resnet34(
                    num_classes=self.dim * 3,
                    input_f=8,
                    sigmoid=sigmoid,
                )  ## Background - composite
        else:

            if self.lut:
                self.PL = self.colornet(
                    num_classes=3 * self.lutdim * self.lutdim * self.lutdim,
                    input_f=7,
                    sigmoid=sigmoid,
                )
            else:
                if self.depthmap:
                    input_d = 8
                else:
                    input_d = 7

                self.PL = self.colornet(
                    num_classes=self.dim * 3,
                    input_f=input_d,
                    sigmoid=sigmoid,
                )  ## Background - composite

        print("PLdim: %d" % (self.dim))
        self.PL3D = LUT3D()
        self.masking = masking
        self.maskoffset = maskoffset
        self.maskconvkernel = maskconvkernel
        if self.masking:
            if self.depthmap:
                input_unet_d = 11
            else:
                input_unet_d = 10
            if self.dual:
                print("Using Dual Gain Maps")
                self.gainnet = UNet_mask(
                    input_dim=input_unet_d,
                    Low_dim=True,
                    brush=brush,
                    nosig=nosig,
                    onlyupsample=onlyupsample,
                    maskoffset=maskoffset,
                    maskconvkernel=maskconvkernel,
                    swap=swap,
                    aggupsample=aggupsample,
                    outputdim=1,
                    lowres=self.lowres,
                )

                self.gainnet_out = UNet_mask(
                    input_dim=input_unet_d,
                    Low_dim=True,
                    brush=brush,
                    nosig=nosig,
                    onlyupsample=onlyupsample,
                    maskoffset=maskoffset,
                    maskconvkernel=maskconvkernel,
                    swap=swap,
                    aggupsample=aggupsample,
                    outputdim=1,
                    lowres=self.lowres,
                )
            else:
                if light:
                    self.gainnet = UNet_mask_light(
                        input_dim=input_unet_d,
                        Low_dim=True,
                        brush=brush,
                        nosig=nosig,
                        onlyupsample=onlyupsample,
                        maskoffset=maskoffset,
                        maskconvkernel=maskconvkernel,
                        swap=swap,
                        aggupsample=aggupsample,
                        outputdim=gainout,
                        lowres=self.lowres,
                    )
                else:
                     self.gainnet = UNet_mask(
                        input_dim=input_unet_d,
                        Low_dim=True,
                        brush=brush,
                        nosig=nosig,
                        onlyupsample=onlyupsample,
                        maskoffset=maskoffset,
                        maskconvkernel=maskconvkernel,
                        swap=swap,
                        aggupsample=aggupsample,
                        outputdim=gainout,
                        lowres=self.lowres,
                     )                   
        if self.depthmap:
            self.depthnet = MidasNet(
                "weights_midas/midas_v21-f6b98070.pt", non_negative=True
            )
            # net_w, net_h = 384, 384
            self.depthtransforms = torch.nn.Sequential(
                T.Resize((384, 384)),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def Resnet_no_grad(self):
        for param in self.PL.parameters():
            param.requires_grad = False

    def Resnet_with_grad(self):
        for param in self.PL.parameters():
            param.requires_grad = True

    def setscalor(self, scalor):
        """Using augmented recon weight

        Args:
            scalor (numpy float): generated scalor, from 0 to 5
        """

        self.scalor = scalor
    
    def forward_input(self,background, input_image, input_mask,curves,scale_sm=0.6):
        p_red = torch.from_numpy(curves[0]).float()
        p_green = torch.from_numpy(curves[1]).float()
        p_blue = torch.from_numpy(curves[2]).float()
        
        pl_table = torch.cat((p_red[None,None,:],p_green[None,None,:],p_blue[None,None,:]),1)
        pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            1, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            1, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            1, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )

        
        pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * background
            )
        
        output_results = (
                    pl_composite * ((self.output_final-0.6)*((1-scale_sm)/0.4)+(scale_sm)) * input_mask
                    + (1 - input_mask) * background
                )

        return (
            pl_composite,
            output_results,
            pl_table
        )     
        
    def forward(self, background, input_image, input_mask):
        """
        Args:
            lut: LUT to be applied, a 5-d tensor. First dimension is batch.
                For a pixel of value (R, G, B), it maps it to
                R' = lut[:, 0, B, G, R],
                G' = lut[:, 1, B, G, R],
                B' = lut[:, 2, B, G, R].
                The "BGR" order is because torch.grid_sample assumes guide_map of
                shape D x H x W, but the coord in grid is (x, y, z) , which is
                (R, G, B) in our case.
            img: input images, of shape B x C x H x W, in range [0, 1]
        Returns:
            out: output images, of shape B x C x H x W
        """
        # On the device
        
        
        if self.high_res:
            background_high = background.clone()
            input_image_high = input_image.clone()
            input_mask_high = input_mask.clone()
            ResizeT = T.Resize((512,512))
            background = ResizeT(background)
            input_image = ResizeT(input_image)
            
            input_mask = ResizeT(input_mask)
            
        
        if self.depthmap:
            input_image_depth = self.depthtransforms(input_image)
            with torch.no_grad():
                input_image_depthmap = self.depthnet(input_image_depth)
                input_image_depthmap = (
                    (
                        input_image_depthmap
                        - torch.amin(input_image_depthmap, dim=(1, 2))[:, None, None]
                    )
                ) / (
                    torch.amax(input_image_depthmap, dim=(1, 2))[:, None, None]
                    - torch.amin(input_image_depthmap, dim=(1, 2))[:, None, None]
                )
                if self.lowres:
                    input_image_depthmap = T.Resize((256, 256))(input_image_depthmap)[
                        :, None, ...
                    ]
                else:
                     input_image_depthmap = T.Resize((512, 512))(input_image_depthmap)[
                        :, None, ...
                    ]                   
                self.input_image_depthmap = input_image_depthmap
        if self.masking:
            if self.scaling:
                tensor_scalor = torch.ones_like(input_mask) * self.scalor
                input_all = torch.cat(
                    (input_image, background, input_mask, tensor_scalor), 1
                )
            else:
                if self.depthmap:
                    input_all = torch.cat(
                        (input_image, background, input_mask, input_image_depthmap), 1
                    )
                else:
                    input_all = torch.cat((input_image, background, input_mask), 1)
            # input_all = torch.cat((input_image, background), 1)
            if self.joint:

                pl_table = self.PL(input_all)[0]

            else:

                with torch.no_grad():
                    pl_table = self.PL(input_all)[0]

            b_dim = pl_table.shape[0]

            brightness = pl_table[0, 0]
            contrast = pl_table[0, 1]
            saturation = pl_table[0, 2]

            if self.lut:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
                )
            else:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.dim),
                )

                pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )
                
            self.pl_table = pl_table.clone()
            
            pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * background
            )
            
            if self.high_res:
                if self.lrdata:
                    pl_composite_large = (
                        self.PL3D(pl_table, input_image_high) * input_mask_high
                        + (1 - input_mask_high) * input_image_high
                    )
                else:
                    pl_composite_large = (
                        self.PL3D(pl_table, input_image_high) * input_mask_high
                        + (1 - input_mask_high) * background_high
                    )

            if self.depthmap:
                input_final = torch.cat(
                    (
                        input_image,
                        background,
                        input_mask,
                        input_image_depthmap,
                        pl_composite,
                    ),
                    1,
                )
            else:
                input_final = torch.cat(
                    (input_image, background, input_mask, pl_composite), 1
                )

            self.output_final_org = self.gainnet(input_final)



            if self.bgshadow:
                if self.dual:
                    self.output_final = self.output_final_org
                    self.output_bg_shadow = self.gainnet_out(input_final)
                    # print("whats up")
                    output_results = (
                        pl_composite * self.output_final * input_mask
                        + (1 - input_mask) * background * self.output_bg_shadow
                    )
                else:

                    self.output_final = self.output_final_org[:, 0, ...][:, None, ...]
                    self.gainmap = self.output_final.clone()
                    
                    self.output_bg_shadow = self.output_final_org[:, 1, ...][
                        :, None, ...
                    ]
                    # print("whats up")
                    output_results = (
                        pl_composite * self.output_final * input_mask
                        + (1 - input_mask) * background * self.output_bg_shadow
                    )

            else:
                self.output_final = self.output_final_org
                self.gainmap = self.output_final.clone()
                

                output_results = (
                    pl_composite * self.output_final * input_mask
                    + (1 - input_mask) * background
                )
                
                if self.high_res:
                    if self.lrdata:
                        pl_composite = pl_composite_large.clone()
                        self.output_final = T.Resize((pl_composite_large.shape[-2:]))(self.output_final)
                        self.gainmap = self.output_final.clone()
                        
                        output_results = (
                        pl_composite_large * self.output_final * input_mask_high
                        + (1 - input_mask_high) * input_image_high
                    )
                    else:
                        pl_composite = pl_composite_large.clone()
                        self.output_final = T.Resize((pl_composite_large.shape[-2:]))(self.output_final)
                        self.gainmap = self.output_final.clone()
                        
                        output_results = (
                        pl_composite_large * self.output_final * input_mask_high
                        + (1 - input_mask_high) * background_high
                    )

            return (
                pl_composite,
                output_results,
                [brightness, contrast, saturation],
                pl_table,
            )
        else:
            if self.scaling:
                tensor_scalor = torch.ones_like(input_mask) * self.scalor
                input_all = torch.cat(
                    (input_image, background, input_mask, tensor_scalor), 1
                )
            else:
                input_all = torch.cat((input_image, background, input_mask), 1)
            # input_all = torch.cat((input_image, background), 1)

            pl_table = self.PL(input_all)[0]

            b_dim = pl_table.shape[0]

            brightness = pl_table[0, 0]
            contrast = pl_table[0, 1]
            saturation = pl_table[0, 2]

            if self.lut:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
                )
            else:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.dim),
                )

                pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )

            pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * background
            )

            return (
                pl_composite,
                pl_composite,
                [brightness, contrast, saturation],
                pl_table,
            )


class Model_UNet(torch.nn.Module):
    def __init__(self, input=3, output=3):
        super().__init__()
        self.input = input
        self.network = UNet(
            n_channels=input, n_classes=output
        )  ## Background - composite - mask

    def forward(self, input_image, input_mask, input_bg, mask=True):

        if self.input == 4:
            input_image = torch.cat((input_image, input_mask), 1)

        elif self.input == 6:
            input_image = torch.cat((input_image, input_bg), 1)
        elif self.input == 7:
            input_image = torch.cat((input_image, input_bg, input_mask), 1)

        # On the device
        # input_all = torch.cat((input_image, input_mask), 1)

        # print(self.input)
        if mask:

            output_composite = self.network(input_image) * input_mask + input_image[
                :, :3, ...
            ] * (1 - input_mask)
        else:
            output_composite = self.network(input_image)

        # inter_features = self.network(images)[1]
        a = output_composite[0, 0, 0, 0]

        return (
            output_composite,
            output_composite,
            [a, a, a],
            [a, a, a],
        )


class Model_Composite_PL_NoBG(torch.nn.Module):
    def __init__(
        self,
        dim=32,
        sigmoid=True,
        scaling=False,
        masking=False,
        brush=False,
        nosig=False,
        onlyupsample=False,
        maskoffset=0.5,
        maskconvkernel=1,
        swap=False,
        lut=False,
        lutdim=16,
        joint=False,
        PIHNet_bool=False,
        Vit_bool=False,
        Eff_bool=False,
        aggupsample=False,
        lowres=False,
        high_res = False,
    ):
        super().__init__()
        self.dim = dim
        self.scaling = scaling
        self.lut = lut
        self.lutdim = lutdim
        self.joint = joint
        self.lowres = lowres

        self.PIHNet_bool = PIHNet_bool
        self.Vit_bool = Vit_bool
        self.Eff_bool = Eff_bool
        
        self.high_res = high_res

        if self.PIHNet_bool:
            self.colornet = PIHNet
            print("Using PIHNet!")
        elif self.Vit_bool:
            self.colornet = VitNet
            print("Using ViT!")
        elif self.Eff_bool:
            print("Using Eff!")
            
            self.colornet = EffNetV2

        else:
            self.colornet = resnet50
            print("Using ResNet! 50!")

        if self.lut:
            print("Using LUT")
            print("lutdim: %d" % (self.lutdim))

        if scaling:
            if self.lut:
                self.PL = resnet34(
                    num_classes=3 * self.lutdim * self.lutdim * self.lutdim,
                    input_f=8,
                    sigmoid=sigmoid,
                )

            else:

                self.PL = resnet34(
                    num_classes=self.dim * 3,
                    input_f=8,
                    sigmoid=sigmoid,
                )  ## Background - composite
        else:

            if self.lut:
                self.PL = self.colornet(
                    num_classes=3 * self.lutdim * self.lutdim * self.lutdim,
                    input_f=4,
                    sigmoid=sigmoid,
                )
            else:

                self.PL = self.colornet(
                    num_classes=self.dim * 3,
                    input_f=4,
                    sigmoid=sigmoid,
                )  ## Background - composite

        print("PLdim: %d" % (self.dim))
        self.PL3D = LUT3D()
        self.masking = masking
        self.maskoffset = maskoffset
        self.maskconvkernel = maskconvkernel
        if self.masking:
            self.gainnet = UNet_mask(
                input_dim=7,
                Low_dim=True,
                brush=brush,
                nosig=nosig,
                onlyupsample=onlyupsample,
                maskoffset=maskoffset,
                maskconvkernel=maskconvkernel,
                swap=swap,
                aggupsample=aggupsample,
                lowres=self.lowres,
            )

    def Resnet_no_grad(self):
        for param in self.PL.parameters():
            param.requires_grad = False

    def Resnet_with_grad(self):
        for param in self.PL.parameters():
            param.requires_grad = True

    def setscalor(self, scalor):
        """Using augmented recon weight

        Args:
            scalor (numpy float): generated scalor, from 0 to 5
        """

        self.scalor = scalor

    def forward(self, input_image, input_mask):
        """
        Args:
            lut: LUT to be applied, a 5-d tensor. First dimension is batch.
                For a pixel of value (R, G, B), it maps it to
                R' = lut[:, 0, B, G, R],
                G' = lut[:, 1, B, G, R],
                B' = lut[:, 2, B, G, R].
                The "BGR" order is because torch.grid_sample assumes guide_map of
                shape D x H x W, but the coord in grid is (x, y, z) , which is
                (R, G, B) in our case.
            img: input images, of shape B x C x H x W, in range [0, 1]
        Returns:
            out: output images, of shape B x C x H x W
        """
        # On the device
        
        if self.high_res:
            # background_high = background.clone()
            input_image_high = input_image.clone()
            input_mask_high = input_mask.clone()
            ResizeT = T.Resize((512,512))
            # background = ResizeT(background)
            input_image = ResizeT(input_image)
            
            input_mask = ResizeT(input_mask)
        
        
        
        if self.masking:
            if self.scaling:
                tensor_scalor = torch.ones_like(input_mask) * self.scalor
                input_all = torch.cat(
                    (input_image, background, input_mask, tensor_scalor), 1
                )
            else:
                input_all = torch.cat((input_image, input_mask), 1)
            # input_all = torch.cat((input_image, background), 1)
            if self.joint:

                pl_table = self.PL(input_all)[0]

            else:

                with torch.no_grad():
                    pl_table = self.PL(input_all)[0]

            b_dim = pl_table.shape[0]

            brightness = pl_table[0, 0]
            contrast = pl_table[0, 1]
            saturation = pl_table[0, 2]

            if self.lut:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
                )
            else:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.dim),
                )

                pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )

            pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * input_image
            )

            if self.high_res:
                pl_composite_large = (
                    self.PL3D(pl_table, input_image_high) * input_mask_high
                    + (1 - input_mask_high) * input_image_high
                )
            input_final = torch.cat((input_image, input_mask, pl_composite), 1)

            self.output_final = self.gainnet(input_final)
            output_results = (
                pl_composite * self.output_final * input_mask
                + (1 - input_mask) * input_image
            )

            if self.high_res:
                pl_composite = pl_composite_large.clone()
                self.output_final = T.Resize((pl_composite_large.shape[-2:]))(self.output_final)
                self.gainmap = self.output_final.clone()
                
                output_results = (
                    pl_composite_large * self.output_final * input_mask_high
                    + (1 - input_mask_high) * input_image_high
                )
            
            
            return (
                pl_composite,
                output_results,
                [brightness, contrast, saturation],
                pl_table,
            )
        else:
            if self.scaling:
                tensor_scalor = torch.ones_like(input_mask) * self.scalor
                input_all = torch.cat(
                    (input_image, background, input_mask, tensor_scalor), 1
                )
            else:
                input_all = torch.cat((input_image, input_mask), 1)
            # input_all = torch.cat((input_image, background), 1)

            pl_table = self.PL(input_all)[0]

            b_dim = pl_table.shape[0]

            brightness = pl_table[0, 0]
            contrast = pl_table[0, 1]
            saturation = pl_table[0, 2]

            if self.lut:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
                )
            else:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.dim),
                )

                pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )

            pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * input_image
            )

            return (
                pl_composite,
                pl_composite,
                [brightness, contrast, saturation],
                pl_table,
            )








class Model_Composite_PL_high_res(torch.nn.Module):
    def __init__(
        self,
        dim=32,
        sigmoid=True,
        scaling=False,
        masking=False,
        brush=False,
        nosig=False,
        onlyupsample=False,
        maskoffset=0.5,
        maskconvkernel=1,
        swap=False,
        lut=False,
        lutdim=16,
        joint=False,
        PIHNet_bool=False,
        Vit_bool=False,
        Eff_bool=False,
        aggupsample=False,
        depthmap=False,
        bgshadow=False,
        ibn=False,
        dual=False,
        lowres=False,
    ):
        super().__init__()
        self.dim = dim
        self.scaling = scaling
        self.lut = lut
        self.lutdim = lutdim
        self.joint = joint
        self.depthmap = depthmap
        self.PIHNet_bool = PIHNet_bool
        self.Vit_bool = Vit_bool
        self.Eff_bool = Eff_bool
        self.ibn = ibn
        self.bgshadow = bgshadow
        self.dual = dual
        self.lowres = lowres

        if bgshadow:
            gainout = 2
        else:
            gainout = 1

        if self.PIHNet_bool:
            self.colornet = PIHNet
            print("Using PIHNet!")
        elif self.Vit_bool:
            self.colornet = VitNet
            print("Using ViT!")
        elif self.Eff_bool:
            self.colornet = EffNetV2
            print("Using EffNet!")

        elif self.ibn:
            print("Using ResNet 50 with ibn")
            self.colornet = resnet50_ibn_b
        else:
            self.colornet = resnet50
            
            print("Using ResNet! 50!")

        if self.lut:
            print("Using LUT")
            print("lutdim: %d" % (self.lutdim))

        if scaling:
            if self.lut:
                self.PL = resnet34(
                    num_classes=3 * self.lutdim * self.lutdim * self.lutdim,
                    input_f=8,
                    sigmoid=sigmoid,
                )

            else:

                self.PL = resnet34(
                    num_classes=self.dim * 3,
                    input_f=8,
                    sigmoid=sigmoid,
                )  ## Background - composite
        else:

            if self.lut:
                self.PL = self.colornet(
                    num_classes=3 * self.lutdim * self.lutdim * self.lutdim,
                    input_f=7,
                    sigmoid=sigmoid,
                )
            else:
                if self.depthmap:
                    input_d = 8
                else:
                    input_d = 7

                self.PL = self.colornet(
                    num_classes=self.dim * 3,
                    input_f=input_d,
                    sigmoid=sigmoid,
                )  ## Background - composite

        print("PLdim: %d" % (self.dim))
        self.PL3D = LUT3D()
        self.masking = masking
        self.maskoffset = maskoffset
        self.maskconvkernel = maskconvkernel
        if self.masking:
            if self.depthmap:
                input_unet_d = 11
            else:
                input_unet_d = 10
            if self.dual:
                print("Using Dual Gain Maps")
                self.gainnet = UNet_mask(
                    input_dim=input_unet_d,
                    Low_dim=True,
                    brush=brush,
                    nosig=nosig,
                    onlyupsample=onlyupsample,
                    maskoffset=maskoffset,
                    maskconvkernel=maskconvkernel,
                    swap=swap,
                    aggupsample=aggupsample,
                    outputdim=1,
                    lowres=self.lowres,
                )

                self.gainnet_out = UNet_mask(
                    input_dim=input_unet_d,
                    Low_dim=True,
                    brush=brush,
                    nosig=nosig,
                    onlyupsample=onlyupsample,
                    maskoffset=maskoffset,
                    maskconvkernel=maskconvkernel,
                    swap=swap,
                    aggupsample=aggupsample,
                    outputdim=1,
                    lowres=self.lowres,
                )
            else:
                self.gainnet = UNet_mask(
                    input_dim=input_unet_d,
                    Low_dim=True,
                    brush=brush,
                    nosig=nosig,
                    onlyupsample=onlyupsample,
                    maskoffset=maskoffset,
                    maskconvkernel=maskconvkernel,
                    swap=swap,
                    aggupsample=aggupsample,
                    outputdim=gainout,
                    lowres=self.lowres,
                )
        if self.depthmap:
            self.depthnet = MidasNet(
                "weights_midas/midas_v21-f6b98070.pt", non_negative=True
            )
            # net_w, net_h = 384, 384
            self.depthtransforms = torch.nn.Sequential(
                T.Resize((384, 384)),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            )

    def Resnet_no_grad(self):
        for param in self.PL.parameters():
            param.requires_grad = False

    def Resnet_with_grad(self):
        for param in self.PL.parameters():
            param.requires_grad = True

    def setscalor(self, scalor):
        """Using augmented recon weight

        Args:
            scalor (numpy float): generated scalor, from 0 to 5
        """

        self.scalor = scalor

    def forward(self, background, input_image, input_mask):
        """
        Args:
            lut: LUT to be applied, a 5-d tensor. First dimension is batch.
                For a pixel of value (R, G, B), it maps it to
                R' = lut[:, 0, B, G, R],
                G' = lut[:, 1, B, G, R],
                B' = lut[:, 2, B, G, R].
                The "BGR" order is because torch.grid_sample assumes guide_map of
                shape D x H x W, but the coord in grid is (x, y, z) , which is
                (R, G, B) in our case.
            img: input images, of shape B x C x H x W, in range [0, 1]
        Returns:
            out: output images, of shape B x C x H x W
        """
        # On the device
        if self.depthmap:
            input_image_depth = self.depthtransforms(input_image)
            with torch.no_grad():
                input_image_depthmap = self.depthnet(input_image_depth)
                input_image_depthmap = (
                    (
                        input_image_depthmap
                        - torch.amin(input_image_depthmap, dim=(1, 2))[:, None, None]
                    )
                ) / (
                    torch.amax(input_image_depthmap, dim=(1, 2))[:, None, None]
                    - torch.amin(input_image_depthmap, dim=(1, 2))[:, None, None]
                )
                if self.lowres:
                    input_image_depthmap = T.Resize((256, 256))(input_image_depthmap)[
                        :, None, ...
                    ]
                else:
                     input_image_depthmap = T.Resize((512, 512))(input_image_depthmap)[
                        :, None, ...
                    ]                   
                self.input_image_depthmap = input_image_depthmap
        if self.masking:
            if self.scaling:
                tensor_scalor = torch.ones_like(input_mask) * self.scalor
                input_all = torch.cat(
                    (input_image, background, input_mask, tensor_scalor), 1
                )
            else:
                if self.depthmap:
                    input_all = torch.cat(
                        (input_image, background, input_mask, input_image_depthmap), 1
                    )
                else:
                    input_all = torch.cat((input_image, background, input_mask), 1)
            # input_all = torch.cat((input_image, background), 1)
            if self.joint:

                pl_table = self.PL(input_all)[0]

            else:

                with torch.no_grad():
                    pl_table = self.PL(input_all)[0]

            b_dim = pl_table.shape[0]

            brightness = pl_table[0, 0]
            contrast = pl_table[0, 1]
            saturation = pl_table[0, 2]

            if self.lut:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
                )
            else:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.dim),
                )

                pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )

            pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * background
            )
            
            if self.high_res:
                pl_composite_large = (
                    self.PL3D(pl_table, input_image_high) * input_mask_high
                    + (1 - input_mask_high) * background_high
                )

            if self.depthmap:
                input_final = torch.cat(
                    (
                        input_image,
                        background,
                        input_mask,
                        input_image_depthmap,
                        pl_composite,
                    ),
                    1,
                )
            else:
                input_final = torch.cat(
                    (input_image, background, input_mask, pl_composite), 1
                )

            self.output_final_org = self.gainnet(input_final)




            if self.bgshadow:
                if self.dual:
                    self.output_final = self.output_final_org
                    self.output_bg_shadow = self.gainnet_out(input_final)
                    # print("whats up")
                    output_results = (
                        pl_composite * self.output_final * input_mask
                        + (1 - input_mask) * background * self.output_bg_shadow
                    )
                else:

                    self.output_final = self.output_final_org[:, 0, ...][:, None, ...]
                    self.output_bg_shadow = self.output_final_org[:, 1, ...][
                        :, None, ...
                    ]
                    # print("whats up")
                    output_results = (
                        pl_composite * self.output_final * input_mask
                        + (1 - input_mask) * background * self.output_bg_shadow
                    )

            else:
                self.output_final = self.output_final_org

                output_results = (
                    pl_composite * self.output_final * input_mask
                    + (1 - input_mask) * background
                )

            return (
                pl_composite,
                output_results,
                [brightness, contrast, saturation],
                pl_table,
            )
        else:
            if self.scaling:
                tensor_scalor = torch.ones_like(input_mask) * self.scalor
                input_all = torch.cat(
                    (input_image, background, input_mask, tensor_scalor), 1
                )
            else:
                input_all = torch.cat((input_image, background, input_mask), 1)
            # input_all = torch.cat((input_image, background), 1)

            pl_table = self.PL(input_all)[0]

            b_dim = pl_table.shape[0]

            brightness = pl_table[0, 0]
            contrast = pl_table[0, 1]
            saturation = pl_table[0, 2]

            if self.lut:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.lutdim, self.lutdim, self.lutdim),
                )
            else:

                pl_table = torch.reshape(
                    pl_table,
                    (pl_table.shape[0], 3, self.dim),
                )

                pl_table = torch.cat(
                    (
                        pl_table[:, 0, None, None, :][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 1, None, :, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                        pl_table[:, 2, :, None, None][:, None, ...].expand(
                            b_dim, 1, self.dim, self.dim, self.dim
                        ),
                    ),
                    1,
                )

            pl_composite = (
                self.PL3D(pl_table, input_image) * input_mask
                + (1 - input_mask) * background
            )

            return (
                pl_composite,
                pl_composite,
                [brightness, contrast, saturation],
                pl_table,
            )
