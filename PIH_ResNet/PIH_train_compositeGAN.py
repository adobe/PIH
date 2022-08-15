import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DataCompositeGAN
from model import Model, Model_Composite, Model_UNet, Model_Composite_PL
from tqdm import tqdm
from torch import Tensor
import networks
from unet_dis import UNetDiscriminatorSN
import random
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
    parser.add_option(
        "--logdir", "--ld", help="Directory for saving logs and checkpoints"
    )
    parser.add_option(
        "-f",
        "--features",
        default=3,
        type="int",
        help="Dimension of the feature space.",
    )

    parser.add_option(
        "--frequency",
        default=1,
        type="int",
        help="frequency to update discriminator",
    )

    parser.add_option(
        "--learning-rate",
        "--lr",
        default=1e-5,
        type="float",
        help="learning rate for the model",
    )

    parser.add_option(
        "--learning-rate-d",
        "--lrd",
        default=1e-5,
        type="float",
        help="learning rate for the discriminator model",
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
        "--workers",
        default=16,
        type="int",
        help="Dimension of the feature space.",
    )

    parser.add_option(
        "-e", "--epochs", default=20000, type="int", help="Number of epochs to train"
    )
    parser.add_option(
        "--force_train_from_scratch",
        "--overwrite",
        action="store_true",
        help="If specified, training will start from scratch."
        " Otherwise, latest checkpoint (if any) will be used",
    )

    parser.add_option(
        "--multi_GPU",
        "--distribuited",
        action="store_true",
        help="If specified, training will use multiple GPU.",
    )

    parser.add_option(
        "--unet",
        action="store_true",
        help="If specified, training will use UNet.",
    )

    parser.add_option(
        "--unetmask",
        action="store_true",
        help="If specified, training will use unet mask.",
    )

    parser.add_option(
        "--reconloss",
        action="store_true",
        help="If specified, training will use reconloss on the mask.",
    )

    # parser.add_option(
    #     "--reconweight",
    #     default=1,
    #     type="float",
    #     help="Recon weight",
    # )
    parser.add_option(
        "--inputdim",
        default=3,
        type="int",
        help="Dimension of the input image.",
    )
    parser.add_option(
        "--sgd",
        action="store_true",
        help="If specified, training will use SGD Optimizer.",
    )
    parser.add_option(
        "--pixel",
        action="store_true",
        help="If specified, using pixel discrinimator.",
    )
    parser.add_option(
        "--unetd",
        action="store_true",
        help="If specified, using unet discrinimator.",
    )

    parser.add_option(
        "--unetdnoskip",
        action="store_true",
        help="If specified, not using skip connection for unet discrinimator.",
    )
    parser.add_option(
        "--tempdir",
        "--tp",
        default="tmp",
        help="temp dir for saving intermediate results during the training.",
    )

    parser.add_option(
        "--trainingratio",
        default=1,
        type="float",
        help="Ratio for the training data. (e.g., 0.1 indicates using 10 percent of the data for training)",
    )
    parser.add_option(
        "--ganlossmask",
        action="store_true",
        help="If specified, will use gan loss for mask.",
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
        "--piecewiselinear",
        action="store_true",
        help="If specified, will not piecewiselinear.",
    )

    parser.add_option(
        "--pairaugment",
        action="store_true",
        help="If specified, will use paired augmentation.",
    )
    parser.add_option(
        "--purepairaugment",
        action="store_true",
        help="If specified, will use paired augmentation.",
    )

    parser.add_option(
        "--lowdim",
        action="store_true",
        help="If specified, will use low dim dis.",
    )

    parser.add_option(
        "--nosigmoid",
        action="store_true",
        help="If specified, will not use sigmoid.",
    )

    parser.add_option(
        "--inputdimD",
        default=3,
        type="int",
        help="Dimension of the input image for D.",
    )
    parser.add_option(
        "--lut-dim",
        default=8,
        type="int",
        help="Dimension of the LUT.",
    )

    parser.add_option(
        "--warmup",
        default=0,
        type="int",
        help="Warmup to initialize.",
    )

    parser.add_option(
        "--reconratio",
        default=0,
        type="float",
        help="Ratio for self reconstruction. (e.g., 0.1 indicates using 10 percent of the data for self reconsruction)",
    )

    parser.add_option(
        "--reconweight",
        default=1,
        type="float",
        help="wight for self reconstruction.",
    )

    parser.add_option(
        "--reconwithgan",
        action="store_true",
        help="If specified, will add adversarial loss for the recon training.",
    )

    parser.add_option(
        "--augreconweight",
        action="store_true",
        help="If specified, will add augmentation on the recon weight.",
    )

    parser.add_option(
        "--losstype",
        default=0,
        type="int",
        help="Loss function type, with the argument augreconweight. 0: lambda*gan 1:lamda*gan+(1-lambda)*l1, scale dis 2:lamda*gan + (1-lamda)*l1, not scale dis",
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
        "--onlyupsample",
        action="store_true",
        help="If specified, will only use upsampling.",
    )
    parser.add_option(
        "--nosig",
        action="store_true",
        help="If specified, will using nosig.",
    )
    parser.add_option(
        "--maskconvkernel",
        default=1,
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
        help="If specified, will using nosig.",
    )
    parser.add_option(
        "--colorjitter",
        action="store_true",
        help="If specified, will use colorjitter.",
    )
    parser.add_option(
        "--joint",
        action="store_true",
        help="If specified, will use joint-training.",
    )
    parser.add_option(
        "--pihnetbool",
        action="store_true",
        help="If specified, will use pihnet.",
    )

    parser.add_option("--maskingcp", help="Directory for masking checkpoint")
    (options, args) = parser.parse_args()
    return options


class Trainer:
    def __init__(self):

        self.args = get_args()
        # self.device = torch.device(f"cuda:{self.args.gpu_id}")
        self.device = torch.device(f"cuda")

        print("Using device:", self.device)

        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.dataset = DataCompositeGAN(
            self.args.datadir,
            self.args.trainingratio,
            augment=self.args.pairaugment,
            colorjitter=self.args.colorjitter,
        )

        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            num_workers=self.args.workers,
            prefetch_factor=8,
            drop_last=True,
        )

        self.data_length = len(self.dataset)

        if self.args.unet:
            self.model = Model_UNet(input=self.args.inputdim)
        else:
            if self.args.piecewiselinear:
                self.model = Model_Composite_PL(
                    dim=32,
                    sigmoid=(not self.args.nosigmoid),
                    scaling=self.args.augreconweight,
                    masking=self.args.masking,
                    brush=self.args.brush,
                    nosig=self.args.nosig,
                    onlyupsample=self.args.onlyupsample,
                    maskoffset=self.args.maskoffset,
                    maskconvkernel=self.args.maskconvkernel,
                    swap=self.args.swap,
                    lut=self.args.lut,
                    lutdim=self.args.lut_dim,
                    joint=self.args.joint,
                    PIHNet_bool=self.args.pihnetbool,
                )
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

        if self.args.pixel:
            self.model_D = networks.define_D(3, 64, "pixel")
        else:
            if self.args.unetd:
                print("Input dim for discriminator: %d" % (self.args.inputdimD))
                if self.args.unetdnoskip:
                    print("No Skip connection!")
                    self.model_D = UNetDiscriminatorSN(
                        input_dim=self.args.inputdimD,
                        skip_connection=False,
                        Low_dim=self.args.lowdim,
                    )
                else:
                    print("With Skip connection!")

                    self.model_D = UNetDiscriminatorSN(
                        input_dim=self.args.inputdimD,
                        Low_dim=self.args.lowdim,
                    )
            else:
                self.model_D = networks.define_D(3, 64, "n_layers", 3)

        if torch.cuda.device_count() > 1 and self.args.multi_GPU:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)

            self.model_D = torch.nn.DataParallel(self.model_D)

        self.model.to(self.device)

        self.model_D.to(self.device)

        self.criterion_GAN = networks.GANLoss(
            "vanilla", gan_loss_mask=self.args.ganlossmask
        ).to(self.device)
        if self.args.ganlossmask:
            print("Using GAN Loss Mask")
        else:
            print("Not using GAN Loss Mask")

        if self.args.reconwithgan:
            print(
                "Using l1+gan for pair recon! l1 weight = %f" % (self.args.reconweight)
            )
        else:
            print("Using gan for pair recon!")
        print("recon ratio: %f" % (self.args.reconratio))

        if self.args.augreconweight:
            print(
                "Using augmented recon weight, reconweight*ganloss + L1 loss, ranging from 0 - 1"
            )
            print("Using loss type:%d" % (self.args.losstype))
        # if self.args.reconloss:
        self.reconloss = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        if self.args.sgd:
            print("Using SGD")
            self.optimizer_D = torch.optim.SGD(
                self.model_D.parameters(), lr=self.args.learning_rate_d, momentum=0.9
            )
        else:
            print("Using Adam")

            self.optimizer_D = torch.optim.Adam(
                self.model_D.parameters(), lr=self.args.learning_rate_d
            )

        self.start_epoch = 1
        if not self.args.force_train_from_scratch:
            self.restore_model()
        else:
            input("Training from scratch. Are you sure? (Ctrl+C to kill):")

        if self.args.masking:
            if self.args.maskingcp:
                self.restore_mask_model()
            else:
                print("Using Joint training")

        os.makedirs(
            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s"
            % (self.args.tempdir),
            exist_ok=True,
        )

    def load_matched_state_dict(self, model, state_dict, print_stats=True):
        """
        Only loads weights that matched in key and shape. Ignore other weights.
        """

        num_matched, num_total = 0, 0
        curr_state_dict = model.state_dict()
        for key in curr_state_dict.keys():
            num_total += 1
            if (
                key in state_dict
                and curr_state_dict[key].shape == state_dict[key].shape
            ):
                curr_state_dict[key] = state_dict[key]
                num_matched += 1
        model.load_state_dict(curr_state_dict)
        if print_stats:
            print(f"Loaded state_dict: {num_matched}/{num_total} matched")

    def restore_mask_model(self):
        """Restore latest model checkpoint (if any) and continue training from there."""

        checkpoint_path = self.args.maskingcp

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.load_matched_state_dict(self.model, checkpoint["state_dict"])
        self.model.Resnet_no_grad()

    def restore_model(self):
        """Restore latest model checkpoint (if any) and continue training from there."""

        checkpoint_path = sorted(
            glob(os.path.join(self.checkpoint_directory, "*")),
            key=lambda x: int(re.match(".*[a-z]+(\d+).pth", x).group(1)),
        )
        if checkpoint_path:
            checkpoint_path = checkpoint_path[-1]
            print(f"Found saved model at: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_dict"])
            self.start_epoch = (
                checkpoint["epoch"] + 1
            )  # Start at next epoch of saved model

            self.model_D.load_state_dict(checkpoint["state_dict_D"])
            self.optimizer_D.load_state_dict(checkpoint["optimizer_dict_D"])

            print(f"Finish restoring model. Resuming at epoch {self.start_epoch}")

        else:
            print("No saved model found. Training from scratch.")

    def save_model(self, epoch):
        """Save model checkpoint.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        torch.save(
            {
                "epoch": epoch,  # Epoch we just finished
                "state_dict": self.model.state_dict(),
                "optimizer_dict": self.optimizer.state_dict(),
                "state_dict_D": self.model_D.state_dict(),
                "optimizer_dict_D": self.optimizer_D.state_dict(),
                "args": self.args,
            },
            os.path.join(self.checkpoint_directory, "ckpt{}.pth".format(epoch)),
        )

    def train(self):
        """Train the model!"""

        # tqdm_bar = tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch")
        #         sys.exit()
        losses_l1_all = []
        losses_G_all = []
        losses_D_all = []
        losses_G_all_com = []
        losses_D_all_com = []

        for epoch in range(self.start_epoch, self.args.epochs + 1):

            self.model.train()
            tqdm_bar = tqdm(enumerate(self.dataloader), "Index")

            for index, (
                image_bg_bg,
                im_composite,
                mask,
                im_real,
                mask_bg,
                im_real_augment,
                fname,
                bname,
            ) in tqdm_bar:

                image_bg_bg = image_bg_bg.to(self.device)
                im_composite = im_composite.to(self.device)
                mask = mask.to(self.device)
                im_real = im_real.to(self.device)
                im_real_augment = im_real_augment.to(self.device)

                mask_bg = mask_bg.to(self.device)

                if np.random.rand() < self.args.reconratio or epoch <= self.args.warmup:

                    if self.args.unet:
                        # print("Using UNet")
                        input_composite, output_composite, par1, par2 = self.model(
                            im_real, mask_bg, image_bg_bg, mask=self.args.unetmask
                        )
                        # print(output_composite.max())
                    else:

                        if self.args.augreconweight:
                            self.model.setscalor(random.uniform(0, 1))

                        input_composite, output_composite, par1, par2 = self.model(
                            image_bg_bg, im_real, mask_bg
                        )

                        (
                            input_composite_aug,
                            output_composite_aug,
                            par1_aug,
                            par2_aug,
                        ) = self.model(image_bg_bg, im_real_augment, mask_bg)

                        # print(par2_aug.shape)
                    if self.args.reconwithgan:

                        ## Update D

                        if index % self.args.frequency == 0:

                            for param in self.model_D.parameters():
                                param.requires_grad = True

                            self.optimizer_D.zero_grad()

                            if self.args.inputdimD == 3:
                                # fake_AB = torch.cat((mask, output_composite), 1)

                                fake_AB = output_composite_aug.clone()
                            elif self.args.inputdimD == 4:
                                fake_AB = torch.cat((mask_bg, output_composite_aug), 1)

                            elif self.args.inputdimD == 6:
                                fake_AB = torch.cat(
                                    (image_bg_bg, output_composite_aug), 1
                                )

                            elif self.args.inputdimD == 7:
                                fake_AB = torch.cat(
                                    (image_bg_bg, mask_bg, output_composite_aug), 1
                                )

                            else:
                                print(
                                    "Using a wrong input dimension for discriminator, supporting 3, 6, 7"
                                )
                            pred_fake = self.model_D(fake_AB.detach())
                            # print(pred_fake.shape)
                            if self.args.ganlossmask:
                                loss_D_fake = self.criterion_GAN(
                                    pred_fake, False, mask=mask_bg
                                )
                            else:
                                loss_D_fake = self.criterion_GAN(pred_fake, False)

                            # real_AB = torch.cat((mask_bg, im_real), 1)
                            if self.args.inputdimD == 3:

                                real_AB = im_real.clone()

                            elif self.args.inputdimD == 4:
                                real_AB = torch.cat((mask_bg, im_real), 1)

                            elif self.args.inputdimD == 6:
                                real_AB = torch.cat((image_bg_bg, im_real), 1)

                            elif self.args.inputdimD == 7:

                                real_AB = torch.cat((image_bg_bg, mask_bg, im_real), 1)

                            pred_real = self.model_D(real_AB)

                            # print("Real_label mean: %f  Fake label mean: %f"%(pred_real.mean(),pred_fake.mean()))
                            if self.args.ganlossmask:
                                loss_D_real = self.criterion_GAN(
                                    pred_real, True, mask=mask
                                )
                            else:
                                loss_D_real = self.criterion_GAN(pred_real, True)

                            loss_D = 0.5 * (loss_D_fake + loss_D_real)

                            if self.args.augreconweight:
                                if self.args.losstype == 0:
                                    loss_D = loss_D * self.model.scalor
                                if self.args.losstype == 1:
                                    loss_D = loss_D * self.model.scalor
                                if self.args.losstype == 2:
                                    pass

                            loss_D.backward()

                            self.optimizer_D.step()

                            losses_D_all.append(loss_D.item())
                        ## Update G

                        for param in self.model_D.parameters():
                            param.requires_grad = False

                        self.optimizer.zero_grad()

                        if self.args.inputdimD == 3:
                            # fake_AB = torch.cat((mask, output_composite), 1)

                            fake_AB = output_composite_aug.clone()
                        elif self.args.inputdimD == 4:
                            fake_AB = torch.cat((mask_bg, output_composite_aug), 1)

                        elif self.args.inputdimD == 6:
                            fake_AB = torch.cat((image_bg_bg, output_composite_aug), 1)

                        elif self.args.inputdimD == 7:
                            fake_AB = torch.cat(
                                (image_bg_bg, mask_bg, output_composite_aug), 1
                            )

                        else:
                            print(
                                "Using a wrong input dimension for discriminator, supporting 3, 6, 7"
                            )

                        pred_fake = self.model_D(fake_AB)
                        if self.args.ganlossmask:

                            loss_G_adv = self.criterion_GAN(pred_fake, True, mask=mask)
                        else:
                            loss_G_adv = self.criterion_GAN(pred_fake, True)

                        if self.args.purepairaugment:
                            loss_l1 = 0 * self.reconloss(
                                output_composite, im_real
                            ) + self.reconloss(output_composite_aug, im_real)
                        else:
                            loss_l1 = 1 * self.reconloss(
                                output_composite, im_real
                            ) + self.reconloss(output_composite_aug, im_real)

                        if self.args.augreconweight:
                            if self.args.losstype == 0:
                                # print("love 000")
                                loss_G_all = self.model.scalor * loss_G_adv + loss_l1
                            if self.args.losstype == 1:
                                # print("love 001")

                                loss_G_all = (
                                    self.model.scalor * loss_G_adv
                                    + (1 - self.model.scalor) * loss_l1
                                )
                            if self.args.losstype == 2:
                                # print("love 002")

                                loss_G_all = (
                                    self.model.scalor * loss_G_adv
                                    + (1 - self.model.scalor) * loss_l1
                                )

                        else:
                            loss_G_all = loss_G_adv + self.args.reconweight * loss_l1

                        loss_G_all.backward()

                        self.optimizer.step()

                        if self.args.augreconweight:
                            tqdm_bar.set_description(
                                "E: {}. L_1: {:3f} L_1_raw: {:3f} L_G: {:3f} L_D: {:3f} L_all: {:3f} Scalor: {:3f}".format(
                                    epoch,
                                    loss_l1.item() / self.model.scalor,
                                    1 * loss_l1.item(),
                                    self.model.scalor * loss_G_adv.item(),
                                    loss_D.item(),
                                    loss_G_all.item(),
                                    self.model.scalor,
                                )
                            )
                        else:
                            tqdm_bar.set_description(
                                "E: {}. L_1: {:3f} L_1_raw: {:3f} L_G: {:3f} L_D: {:3f} L_all: {:3f}".format(
                                    epoch,
                                    self.args.reconweight * loss_l1.item(),
                                    1 * loss_l1.item(),
                                    loss_G_adv.item(),
                                    loss_D.item(),
                                    loss_G_all.item(),
                                )
                            )

                        losses_G_all.append(loss_G_adv.item())
                        if self.args.augreconweight:
                            losses_l1_all.append(loss_l1.item())
                        else:
                            losses_l1_all.append(self.args.reconweight * loss_l1.item())

                        if epoch % 1 == 0 and index < 20:
                            # self.save_model(epoch)

                            for kk in range(self.args.batchsize):
                                # print("Red: ", par2[kk, 0, 0, 0, :])
                                # print("Red: ", par2[kk, 0, 3, 4, :])

                                name = (
                                    fname[kk].split("/")[-1].split(".")[0]
                                    + "_"
                                    + bname[kk].split("/")[-1].split(".")[0]
                                )
                                name = "%d_%d" % (index, kk)

                                if self.args.masking:
                                    image_gainmap = T.ToPILImage()(
                                        self.model.output_final[kk, ...]
                                        .cpu()
                                        .clamp(0, 1)
                                    )
                                    image_gainmap.save(
                                        "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_L1_aug_gain.jpg"
                                        % (self.args.tempdir, name)
                                    )

                                image_aug = T.ToPILImage()(
                                    output_composite_aug[kk, ...].cpu().clamp(0, 1)
                                )

                                image_aug.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_L1_aug.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_aug_inter = T.ToPILImage()(
                                    input_composite_aug[kk, ...].cpu().clamp(0, 1)
                                )

                                image_aug_inter.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_L1_aug_inter.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_all = T.ToPILImage()(
                                    output_composite[kk, ...].cpu().clamp(0, 1)
                                )

                                image_all.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_real = T.ToPILImage()(
                                    im_real[kk, ...].cpu().clamp(0, 1)
                                )
                                image_real.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_real_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_real_aug = T.ToPILImage()(
                                    im_real_augment[kk, ...].cpu().clamp(0, 1)
                                )
                                image_real_aug.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_real_aug_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_og = T.ToPILImage()(
                                    im_real_augment[kk, ...].cpu().clamp(0, 1)
                                )
                                image_og.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_composite_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_dis = T.ToPILImage()(
                                    pred_fake[kk, ...].cpu().clamp(0, 1)
                                )
                                image_dis.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_dis_score_fake_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_dis_true = T.ToPILImage()(
                                    pred_real[kk, ...].cpu().clamp(0, 1)
                                )
                                image_dis_true.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_dis_score_real_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                    else:

                        for param in self.model_D.parameters():
                            param.requires_grad = False

                        self.optimizer.zero_grad()

                        if self.args.purepairaugment:
                            loss_l1 = 0 * self.reconloss(
                                output_composite, im_real
                            ) + self.reconloss(output_composite_aug, im_real)
                        else:
                            loss_l1 = 1 * self.reconloss(
                                output_composite, im_real
                            ) + self.reconloss(output_composite_aug, im_real)

                        loss_all = self.args.reconweight * loss_l1
                        loss_all.backward()

                        self.optimizer.step()

                        tqdm_bar.set_description(
                            "E: {}. L_1: {:3f}".format(
                                epoch,
                                self.args.reconweight * loss_l1.item(),
                            )
                        )
                        losses_l1_all.append(self.args.reconweight * loss_l1)
                        if epoch % 1 == 0 and index < 20:
                            # self.save_model(epoch)

                            for kk in range(self.args.batchsize):
                                # print("Red: ", par2[kk, 0, 0, 0, :])
                                # print("Red: ", par2[kk, 0, 3, 4, :])

                                name = (
                                    fname[kk].split("/")[-1].split(".")[0]
                                    + "_"
                                    + bname[kk].split("/")[-1].split(".")[0]
                                )
                                name = "%d_%d" % (index, kk)

                                image_aug = T.ToPILImage()(
                                    output_composite_aug[kk, ...].cpu().clamp(0, 1)
                                )

                                image_aug.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_L1_aug.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_all = T.ToPILImage()(
                                    output_composite[kk, ...].cpu().clamp(0, 1)
                                )

                                image_all.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_real = T.ToPILImage()(
                                    im_real[kk, ...].cpu().clamp(0, 1)
                                )
                                image_real.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_real_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_real_aug = T.ToPILImage()(
                                    im_real_augment[kk, ...].cpu().clamp(0, 1)
                                )
                                image_real_aug.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_real_aug_L1.jpg"
                                    % (self.args.tempdir, name)
                                )

                                image_og = T.ToPILImage()(
                                    im_real[kk, ...].cpu().clamp(0, 1)
                                )
                                image_og.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_composite_L1.jpg"
                                    % (self.args.tempdir, name)
                                )
                else:

                    if self.args.unet:
                        # print("Using UNet")
                        input_composite, output_composite, par1, par2 = self.model(
                            im_composite, mask, image_bg_bg, mask=self.args.unetmask
                        )
                        # print(output_composite.max())
                    else:

                        if self.args.augreconweight:
                            self.model.setscalor(random.uniform(0, 1))

                        input_composite, output_composite, par1, par2 = self.model(
                            image_bg_bg, im_composite, mask
                        )

                    brightness, contrast, saturation = par1

                    ## Update D

                    if index % self.args.frequency == 0:

                        for param in self.model_D.parameters():
                            param.requires_grad = True

                        self.optimizer_D.zero_grad()

                        if self.args.inputdimD == 3:
                            # fake_AB = torch.cat((mask, output_composite), 1)

                            fake_AB = output_composite.clone()
                        elif self.args.inputdimD == 4:
                            fake_AB = torch.cat((mask, output_composite), 1)

                        elif self.args.inputdimD == 6:
                            fake_AB = torch.cat((image_bg_bg, output_composite), 1)

                        elif self.args.inputdimD == 7:
                            fake_AB = torch.cat(
                                (image_bg_bg, mask, output_composite), 1
                            )

                        else:
                            print(
                                "Using a wrong input dimension for discriminator, supporting 3, 6, 7"
                            )
                        pred_fake = self.model_D(fake_AB.detach())
                        # print(pred_fake.shape)
                        if self.args.ganlossmask:
                            loss_D_fake = self.criterion_GAN(
                                pred_fake, False, mask=mask
                            )
                        else:
                            loss_D_fake = self.criterion_GAN(pred_fake, False)

                        # real_AB = torch.cat((mask_bg, im_real), 1)
                        if self.args.inputdimD == 3:

                            real_AB = im_real.clone()

                        elif self.args.inputdimD == 4:
                            real_AB = torch.cat((mask_bg, im_real), 1)

                        elif self.args.inputdimD == 6:
                            real_AB = torch.cat((image_bg_bg, im_real), 1)

                        elif self.args.inputdimD == 7:

                            real_AB = torch.cat((image_bg_bg, mask_bg, im_real), 1)

                        pred_real = self.model_D(real_AB)

                        # print("Real_label mean: %f  Fake label mean: %f"%(pred_real.mean(),pred_fake.mean()))
                        if self.args.ganlossmask:
                            loss_D_real = self.criterion_GAN(pred_real, True, mask=mask)
                        else:
                            loss_D_real = self.criterion_GAN(pred_real, True)

                        loss_D = 0.5 * (loss_D_fake + loss_D_real)
                        if self.args.augreconweight:
                            if self.args.losstype == 0:
                                loss_D = self.model.scalor * loss_D
                            if self.args.losstype == 1:
                                loss_D = self.model.scalor * loss_D
                            if self.args.losstype == 2:
                                pass

                        loss_D.backward()

                        self.optimizer_D.step()

                        losses_D_all_com.append(loss_D.item())

                    ## Update G

                    for param in self.model_D.parameters():
                        param.requires_grad = False

                    self.optimizer.zero_grad()

                    if self.args.inputdimD == 3:
                        # fake_AB = torch.cat((mask, output_composite), 1)

                        fake_AB = output_composite.clone()
                    elif self.args.inputdimD == 4:
                        fake_AB = torch.cat((mask, output_composite), 1)

                    elif self.args.inputdimD == 6:
                        fake_AB = torch.cat((image_bg_bg, output_composite), 1)

                    elif self.args.inputdimD == 7:
                        fake_AB = torch.cat((image_bg_bg, mask, output_composite), 1)

                    else:
                        print(
                            "Using a wrong input dimension for discriminator, supporting 3, 6, 7"
                        )

                    pred_fake = self.model_D(fake_AB)
                    if self.args.ganlossmask:

                        loss_G_adv = self.criterion_GAN(pred_fake, True, mask=mask)
                    else:
                        loss_G_adv = self.criterion_GAN(pred_fake, True)

                    if self.args.augreconweight:
                        if self.args.losstype == 0:
                            loss_G_adv = self.model.scalor * loss_G_adv
                        if self.args.losstype == 1:
                            loss_G_adv = self.model.scalor * loss_G_adv
                        else:
                            pass

                    loss_G_adv.backward()

                    self.optimizer.step()

                    losses_D_all_com.append(loss_G_adv.item())
                    if self.args.augreconweight:

                        tqdm_bar.set_description(
                            "E: {}. L_G: {:3f} L_D: {:3f} Scalor: {:3f} ".format(
                                epoch,
                                loss_G_adv.item(),
                                loss_D.item(),
                                self.model.scalor,
                            )
                        )

                    else:
                        tqdm_bar.set_description(
                            "E: {}. L_G: {:3f} L_D: {:3f}".format(
                                epoch,
                                loss_G_adv.item(),
                                loss_D.item(),
                            )
                        )
                    if epoch % 1 == 0 and index < 20:
                        # self.save_model(epoch)

                        for kk in range(self.args.batchsize):

                            # print("Red: ", par2[0, 0, 0, 0, :])

                            name = (
                                fname[kk].split("/")[-1].split(".")[0]
                                + "_"
                                + bname[kk].split("/")[-1].split(".")[0]
                            )
                            name = "%d_%d" % (index, kk)
                            image_all = T.ToPILImage()(
                                output_composite[kk, ...].cpu().clamp(0, 1)
                            )
                            if self.args.masking:
                                image_gainmap = T.ToPILImage()(
                                    self.model.output_final[kk, ...].cpu().clamp(0, 1)
                                )
                                image_gainmap.save(
                                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out_gain.jpg"
                                    % (self.args.tempdir, name)
                                )
                            image_dis = T.ToPILImage()(
                                pred_fake[kk, ...].cpu().clamp(0, 1)
                            )
                            image_dis.save(
                                "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_dis_score_fake.jpg"
                                % (self.args.tempdir, name)
                            )

                            image_dis_true = T.ToPILImage()(
                                pred_real[kk, ...].cpu().clamp(0, 1)
                            )
                            image_dis_true.save(
                                "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_dis_score_real.jpg"
                                % (self.args.tempdir, name)
                            )

                            image_all.save(
                                "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out.jpg"
                                % (self.args.tempdir, name)
                            )

                            image_i = T.ToPILImage()(
                                input_composite[kk, ...].cpu().clamp(0, 1)
                            )
                            image_i.save(
                                "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_inter.jpg"
                                % (self.args.tempdir, name)
                            )

                            image_real = T.ToPILImage()(
                                im_real[kk, ...].cpu().clamp(0, 1)
                            )
                            image_real.save(
                                "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_real.jpg"
                                % (self.args.tempdir, name)
                            )

                            image_og = T.ToPILImage()(
                                im_composite[kk, ...].cpu().clamp(0, 1)
                            )
                            image_og.save(
                                "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_composite.jpg"
                                % (self.args.tempdir, name)
                            )

            # print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n brightness {brightness} contrast {contrast} saturation {saturation} hue {hue}")
            # np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 1 == 0:
                self.save_model(epoch)

                np.save(
                    os.path.join(
                        self.checkpoint_directory, "loss_l1_{}.pth".format(epoch)
                    ),
                    np.array(losses_l1_all),
                )
                np.save(
                    os.path.join(
                        self.checkpoint_directory, "loss_G_{}.pth".format(epoch)
                    ),
                    np.array(losses_G_all),
                )
                np.save(
                    os.path.join(
                        self.checkpoint_directory, "loss_D_{}.pth".format(epoch)
                    ),
                    np.array(losses_D_all),
                )
                np.save(
                    os.path.join(
                        self.checkpoint_directory, "loss_Gcom_{}.pth".format(epoch)
                    ),
                    np.array(losses_G_all_com),
                )
                np.save(
                    os.path.join(
                        self.checkpoint_directory, "loss_Dcom_{}.pth".format(epoch)
                    ),
                    np.array(losses_D_all_com),
                )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
