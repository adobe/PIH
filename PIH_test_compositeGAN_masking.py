import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PIHData, PIHDataNGT, IhdDataset, PIHData_Composite
from model import (
    Model,
    Model_Composite,
    Model_UNet,
    Model_Composite_PL,
    Model_Composite_PL_NoBG,
)
from tqdm import tqdm
from torch import Tensor
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F


def get_args():
    parser = OptionParser()
    parser.add_option("--datadir", "--dd", help="Directory contains 2D images.")
    parser.add_option(
        "-g",
        "--gpu_id",
        dest="gpu_id",
        type="int",
        help="GPU number, default is None (-g 0 means use gpu 0)",
    )
    parser.add_option("--checkpoints", "--ld", help="Directory to checkpoints")
    parser.add_option("--tmp_results", "--tr", help="Results for temporary folder")

    parser.add_option(
        "-f",
        "--features",
        default=3,
        type="int",
        help="Dimension of the feature space.",
    )

    parser.add_option(
        "--num-testing",
        default=0,
        type="int",
        help="set the number for testing images, default is the entire dataset",
    )

    parser.add_option(
        "--batchsize",
        "--bs",
        dest="batchsize",
        default=4,
        type="int",
        help="batch size for training",
    )

    parser.add_option(
        "--ngt",
        action="store_true",
        help="If specified, inference without ground truth.",
    )

    parser.add_option(
        "--ihd",
        action="store_true",
        help="If specified, will use iHarmony dataset for the testing",
    )
    parser.add_option(
        "--composite",
        action="store_true",
        help="If specified, will use Composite dataset for the testing",
    )

    parser.add_option(
        "--unet",
        action="store_true",
        help="If specified, will use UNet for the testing",
    )
    parser.add_option(
        "--lut",
        action="store_true",
        help="If specified, will use lut as last step.",
    )
    parser.add_option(
        "--nocurve",
        action="store_true",
        help="If specified, will not use curve.",
    )
    parser.add_option(
        "--lut-dim",
        default=8,
        type="int",
        help="Dimension of the LUT.",
    )

    parser.add_option(
        "--piecewiselinear",
        action="store_true",
        help="If specified, will not piecewiselinear.",
    )

    parser.add_option(
        "--masking",
        action="store_true",
        help="If specified, will using masking.",
    )

    parser.add_option(
        "--brush",
        action="store_true",
        help="If specified, will using brush.",
    )
    parser.add_option(
        "--nosig",
        action="store_true",
        help="If specified, will using nosig.",
    )
    parser.add_option(
        "--onlyupsample",
        action="store_true",
        help="If specified, will only use upsampling.",
    )

    parser.add_option(
        "--aggupsample",
        action="store_true",
        help="If specified, will only use agg upsampling.",
    )

    parser.add_option(
        "--maskconvkernel",
        default=1,
        type="int",
        help="maskconvkernel.",
    )

    parser.add_option(
        "--dim",
        default=32,
        type="int",
        help="maskconvkernel.",
    )
    parser.add_option(
        "--maskoffset",
        default=0.5,
        type="float",
        help="maskoffset.",
    )
    parser.add_option(
        "--swap",
        action="store_true",
        help="If specified, will only use upsampling.",
    )
    parser.add_option(
        "--pihnetbool",
        action="store_true",
        help="If specified, will use pihnet.",
    )

    parser.add_option(
        "--vitbool",
        action="store_true",
        help="If specified, will use pihnet.",
    )

    parser.add_option(
        "--effbool",
        action="store_true",
        help="If specified, will use efficientnet v2 - s.",
    )

    parser.add_option(
        "--twoinputs",
        action="store_true",
        help="If specified, will use two inputs.",
    )
    parser.add_option(
        "--depthmap",
        action="store_true",
        help="If specified, will depthmap.",
    )
    parser.add_option(
        "--bgshadow",
        action="store_true",
        help="If specified, will use bgshadow.",
    )

    parser.add_option(
        "--ibn",
        action="store_true",
        help="If specified, will use ibn.",
    )
    parser.add_option(
        "--dual",
        action="store_true",
        help="If specified, will use ibn.",
    )
    parser.add_option(
        "--lowres",
        action="store_true",
        help="If specified, will lowres.",
    )

    (options, args) = parser.parse_args()
    return options


class Evaluater:
    def __init__(self):

        self.args = get_args()
        # self.device = torch.device(f"cuda:{self.args.gpu_id}")
        self.device = torch.device(f"cuda")
        self.num = False
        if self.args.num_testing > 0:
            self.num = True
            self.num_testing = self.args.num_testing
        print("Using device:", self.device)

        self.checkpoint_directory = self.args.checkpoints

        self.tmp = self.args.tmp_results

        os.makedirs(self.tmp, exist_ok=True)
        os.makedirs(self.tmp + "/mask/", exist_ok=True)
        os.makedirs(self.tmp + "/original/", exist_ok=True)
        os.makedirs(self.tmp + "/intermediate/", exist_ok=True)
        os.makedirs(self.tmp + "/results/", exist_ok=True)
        os.makedirs(self.tmp + "/bg/", exist_ok=True)
        if self.args.masking:
            os.makedirs(self.tmp + "/gainmap/", exist_ok=True)
            if self.args.bgshadow:
                os.makedirs(self.tmp + "/bgshadow/", exist_ok=True)
            if self.args.depthmap:
                os.makedirs(self.tmp + "/dp/", exist_ok=True)

        if self.args.piecewiselinear:
            os.makedirs(self.tmp + "/curves/", exist_ok=True)

        if not self.args.ngt:
            os.makedirs(self.tmp + "/real/", exist_ok=True)

        if self.args.ihd:
            self.args.train = False
            self.dataset = IhdDataset(self.args)

        else:
            if self.args.ngt:
                if self.args.composite:
                    self.dataset = PIHData_Composite(self.args.datadir)
                else:
                    self.dataset = PIHDataNGT(self.args.datadir, device=self.device)
            else:
                if self.args.composite:
                    self.dataset = PIHData_Composite(self.args.datadir, lowres=self.args.lowres)
                else:
                    self.dataset = PIHData(self.args.datadir, device=self.device)

        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            num_workers=8,
            prefetch_factor=4,
            drop_last=True,
        )

        self.data_length = len(self.dataset)
        if self.args.composite:

            if self.args.piecewiselinear:
                if self.args.twoinputs:
                    self.model = Model_Composite_PL_NoBG(
                        dim=self.args.dim,
                        sigmoid=True,
                        scaling=False,
                        masking=self.args.masking,
                        brush=self.args.brush,
                        nosig=self.args.nosig,
                        onlyupsample=self.args.onlyupsample,
                        maskoffset=self.args.maskoffset,
                        maskconvkernel=self.args.maskconvkernel,
                        swap=self.args.swap,
                        lut=self.args.lut,
                        lutdim=self.args.lut_dim,
                        PIHNet_bool=self.args.pihnetbool,
                        Vit_bool=self.args.vitbool,
                        Eff_bool=self.args.effbool,
                        aggupsample=self.args.aggupsample,
                        lowres=self.args.lowres,
                    )
                else:
                    self.model = Model_Composite_PL(
                        dim=self.args.dim,
                        sigmoid=True,
                        scaling=False,
                        masking=self.args.masking,
                        brush=self.args.brush,
                        nosig=self.args.nosig,
                        onlyupsample=self.args.onlyupsample,
                        maskoffset=self.args.maskoffset,
                        maskconvkernel=self.args.maskconvkernel,
                        swap=self.args.swap,
                        lut=self.args.lut,
                        PIHNet_bool=self.args.pihnetbool,
                        Vit_bool=self.args.vitbool,
                        Eff_bool=self.args.effbool,
                        aggupsample=self.args.aggupsample,
                        depthmap=self.args.depthmap,
                        bgshadow=self.args.bgshadow,
                        ibn=self.args.ibn,
                        dual=self.args.dual,
                        lowres=self.args.lowres,
                        lutdim=self.args.lut_dim,
                    )

            else:
                if self.args.unet:
                    self.model = Model_UNet(6)
                else:
                    if self.args.lut:
                        self.model = Model_Composite(
                            feature_dim=self.args.features,
                            LUT=True,
                            LUTdim=self.args.lut_dim,
                            curve=not self.args.nocurve,
                        )

                    else:
                        self.model = Model_Composite(feature_dim=self.args.features)
        else:
            self.model = Model(feature_dim=self.args.features)

        self.model.to(self.device)
        self.restore_model()

        self.criterion = torch.nn.L1Loss()

    def restore_model(self):
        """Restore latest model checkpoint (if any) and continue training from there."""

        checkpoint = torch.load(self.checkpoint_directory, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])

    def evaluate(self):
        """Evaluate the model!"""
        plt.ioff()
        self.model.eval()

        tqdm_bar = tqdm(enumerate(self.dataloader), "iteration")

        for index, (
            input_bg,
            input_composite,
            input_mask,
            input_real,
            names,
        ) in tqdm_bar:

            input_bg = input_bg.to(self.device)
            input_composite = input_composite.to(self.device)

            input_mask = input_mask.to(self.device)

            with torch.no_grad():
                if not self.args.ngt:
                    input_real = input_real.to(self.device)

                if self.args.unet:
                    inter_composite, output_composite, par1, par2 = self.model(
                        input_composite, input_mask, input_bg
                    )
                else:
                    if self.args.twoinputs:
                        inter_composite, output_composite, par1, par2 = self.model(
                            input_composite, input_mask
                        )
                    else:
                        inter_composite, output_composite, par1, par2 = self.model(
                            input_bg, input_composite, input_mask
                        )

            brightness, contrast, saturation = par1
            b_r, b_g, b_b = par1

            # if self.args.piecewiselinear:
            #     curves = par2.cpu().detach().numpy()
            #     red_curve = par2[0, 0, 0, 0, :]
            #     green_curve = par2[0, 0, 0, :, 0]
            #     blue_curve = par2[0, 0, :, 0, 0]

            #     plt.plot(red_curve, "r")
            #     plt.plot(green_curve, "g")
            #     plt.plot(blue_curve, "b")

            #     plt.savefig("books_read.png")
            # print(curves.shape)
            # if not self.args.ngt:

            #     loss_second = self.criterion(output_composite, gt)

            #     loss_first = self.criterion(input_composite, gt)
            #     loss = 1 * loss_second + 0 * loss_first
            # print(loss.item())

            for kk in range(self.args.batchsize):

                name_image = names[0].split("/")[-1].replace('jpg','png')

                if self.args.piecewiselinear:
                    curves = par2.cpu().detach().numpy()

                    red_curve = curves[0, 0, 0, 0, :]
                    green_curve = curves[0, 1, 0, :, 0]
                    blue_curve = curves[0, 2, :, 0, 0]

                    x = np.linspace(0,1,red_curve.shape[0])
                    plt.figure(figsize=(6,6))
                    # plt.rcParams["font.family"] = "Times New Roman"
                    ax = plt.subplot(111)
                    ax.plot(x,red_curve,color='r',lw=2)
                    ax.plot(x,green_curve,color='g',lw=2)
                    ax.plot(x,blue_curve,color='b',lw=2)
                    ax.spines.right.set_visible(False)
                    ax.spines.top.set_visible(False)
                    ax.set_xticks([0,1])
                    ax.set_xticklabels([0,1], fontsize=24 )
                    ax.set_yticks([0,1])
                    ax.set_yticklabels([0,1], fontsize=24 )
                    [x.set_linewidth(2) for x in ax.spines.values()]
                    leg = ax.legend(["Red","Green","Blue"], fontsize=24)
                    leg.get_frame().set_linewidth(0.0)
                    plt.savefig(self.tmp + "/curves/%s" % (name_image),dpi=300,bbox_inches='tight',transparent=True)

                    plt.close()

                image_all = T.ToPILImage()(output_composite[kk, ...].cpu())
                image_all.save(self.tmp + "/results/%s" % (name_image))

                image_mk = T.ToPILImage()(input_mask[kk, ...].cpu())
                image_mk.save(self.tmp + "/mask/%s" % (name_image))

                image_i = T.ToPILImage()(inter_composite[kk, ...].cpu())
                image_i.save(self.tmp + "/intermediate/%s" % (name_image))

                if not self.args.ngt:
                    image_gt = T.ToPILImage()(input_real[kk, ...].cpu())
                    image_gt.save(self.tmp + "/real/%s" % (name_image))

                image_og = T.ToPILImage()(input_composite[kk, ...].cpu())
                image_og.save(self.tmp + "/original/%s" % (name_image))

                image_bg = T.ToPILImage()(input_bg[kk, ...].cpu())
                image_bg.save(self.tmp + "/bg/%s" % (name_image))

                if self.args.masking:
                    image_gainmap = T.ToPILImage()(
                        self.model.output_final[kk, ...].cpu()
                        * input_mask[kk, ...].cpu()
                    )
                    image_gainmap.save(self.tmp + "/gainmap/%s" % (name_image))

                    if self.args.depthmap:
                        image_dp = T.ToPILImage()(
                            self.model.input_image_depthmap[kk, ...].cpu()
                        )

                        # print("max:", self.model.output_bg_shadow[kk, ...].cpu().max())
                        # print("min:", self.model.output_bg_shadow[kk, ...].cpu().min())

                        image_dp.save(self.tmp + "/dp/%s" % (name_image))

                    if self.args.bgshadow:
                        # output_bg_shadow

                        image_bgshadow = T.ToPILImage()(
                            self.model.output_bg_shadow[kk, ...].cpu()
                            * (1 - input_mask[kk, ...].cpu())
                        )

                        print("max:", self.model.output_bg_shadow[kk, ...].cpu().max())
                        print("min:", self.model.output_bg_shadow[kk, ...].cpu().min())

                        image_bgshadow.save(self.tmp + "/bgshadow/%s" % (name_image))

                # image_all = T.ToPILImage()(output_composite[kk, ...].cpu())
                # image_all.save(self.tmp + "/tmp%d_%d.jpg" % (index, kk))

                # image_i = T.ToPILImage()(input_composite[kk, ...].cpu())
                # image_i.save(self.tmp + "/tmp%d_%d_inter.jpg" % (index, kk))
                # if not self.args.ngt:
                #     image_gt = T.ToPILImage()(gt[kk, ...].cpu())
                #     image_gt.save(self.tmp + "/tmp%d_%d_gt.jpg" % (index, kk))

                # image_og = T.ToPILImage()(input_image[kk, ...].cpu())
                # image_og.save(self.tmp + "/tmp%d_%d_og.jpg" % (index, kk))

            if not self.args.ngt:
                tqdm_bar.set_description(
                    "I: {}. b: {:3f} c: {:3f} s: {:3f} br: {:3f} bg: {:3f} bb: {:3f}".format(
                        index,
                        brightness.item(),
                        contrast.item(),
                        saturation.item(),
                        b_r.item(),
                        b_g.item(),
                        b_b.item(),
                    )
                )
            else:
                tqdm_bar.set_description(
                    "I: {}. b: {:3f} c: {:3f} s: {:3f} br: {:3f} bg: {:3f} bb: {:3f}".format(
                        index,
                        brightness.item(),
                        contrast.item(),
                        saturation.item(),
                        b_r.item(),
                        b_g.item(),
                        b_b.item(),
                    )
                )
            if self.num and index > self.num_testing:
                break


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evaluater = Evaluater()
    evaluater.evaluate()
