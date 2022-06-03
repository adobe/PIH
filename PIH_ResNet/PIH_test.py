import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PIHData, PIHDataNGT
from model import Model
from tqdm import tqdm
from torch import Tensor

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
    (options, args) = parser.parse_args()
    return options


class Evaluater:
    def __init__(self):

        self.args = get_args()
        # self.device = torch.device(f"cuda:{self.args.gpu_id}")
        self.device = torch.device(f"cuda")

        print("Using device:", self.device)

        self.checkpoint_directory = self.args.checkpoints

        self.tmp = self.args.tmp_results

        os.makedirs(self.tmp, exist_ok=True)
        os.makedirs(self.tmp + "/original/", exist_ok=True)
        os.makedirs(self.tmp + "/intermediate/", exist_ok=True)
        os.makedirs(self.tmp + "/results/", exist_ok=True)
        if not self.args.ngt:
            os.makedirs(self.tmp + "/gt/", exist_ok=True)

        if self.args.ngt:
            self.dataset = PIHDataNGT(self.args.datadir, device=self.device)
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

        self.model.eval()

        tqdm_bar = tqdm(enumerate(self.dataloader), "iteration")

        for index, (input_image, input_mask, gt) in tqdm_bar:

            input_image = input_image.to(self.device)
            input_mask = input_mask.to(self.device)
            if not self.args.ngt:
                gt = gt.to(self.device)

            input_composite, output_composite, par1, par2 = self.model(
                input_image, input_mask
            )

            brightness, contrast, saturation = par1
            b_r, b_g, b_b = par2

            if not self.args.ngt:

                loss_second = self.criterion(output_composite, gt)

                loss_first = self.criterion(input_composite, gt)
                loss = 1 * loss_second + 0 * loss_first
            # print(loss.item())

            for kk in range(self.args.batchsize):

                # image_all = T.ToPILImage()(output_composite[kk, ...].cpu())
                # image_all.save(self.tmp + "/results/tmp%d_%d.jpg" % (index, kk))

                # image_i = T.ToPILImage()(input_composite[kk, ...].cpu())
                # image_i.save(self.tmp + "/intermediate/tmp%d_%d.jpg" % (index, kk))

                # image_gt = T.ToPILImage()(gt[kk, ...].cpu())
                # image_gt.save(self.tmp + "/gt/tmp%d_%d.jpg" % (index, kk))

                image_og = T.ToPILImage()(input_image[kk, ...].cpu())
                image_og.save(self.tmp + "/original/tmp%d_%d.jpg" % (index, kk))

                image_all = T.ToPILImage()(output_composite[kk, ...].cpu())
                image_all.save(self.tmp + "/tmp%d_%d.jpg" % (index, kk))

                image_i = T.ToPILImage()(input_composite[kk, ...].cpu())
                image_i.save(self.tmp + "/tmp%d_%d_inter.jpg" % (index, kk))
                if not self.args.ngt:
                    image_gt = T.ToPILImage()(gt[kk, ...].cpu())
                    image_gt.save(self.tmp + "/tmp%d_%d_gt.jpg" % (index, kk))

                image_og = T.ToPILImage()(input_image[kk, ...].cpu())
                image_og.save(self.tmp + "/tmp%d_%d_og.jpg" % (index, kk))

            if not self.args.ngt:
                tqdm_bar.set_description(
                    "I: {}. L: {:3f} b: {:3f} c: {:3f} s: {:3f} br: {:3f} bg: {:3f} bb: {:3f}".format(
                        index,
                        loss_second.item(),
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


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evaluater = Evaluater()
    evaluater.evaluate()
