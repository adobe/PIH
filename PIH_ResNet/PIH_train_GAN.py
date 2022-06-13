import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PIHData, PIHDataRandom, IhdDataset
from model import Model
from tqdm import tqdm
from torch import Tensor
import networks


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
        "--random_aug",
        action="store_true",
        help="If specified, training will modify the color on the fly.",
    )

    parser.add_option(
        "--ihd",
        action="store_true",
        help="If specified, will use iHarmony dataset for the training",
    )
    parser.add_option(
        "--train",
        action="store_true",
        help="If specified, will start training (currently only for iHarmony dataset)",
    )

    parser.add_option(
        "--gan-weight",
        default=0,
        type="float",
        help="GAN weight, default without using GAN",
    )

    parser.add_option(
        "--noreconloss",
        action="store_true",
        help="If specified, will not using recon loss)",
    )
    parser.add_option(
        "--conditional",
        action="store_true",
        help="If specified, will use conditional gan loss",
    )

    parser.add_option(
        "--tempdir",
        "--tp",
        default="tmp",
        help="temp dir for saving intermediate results during the training.",
    )
    (options, args) = parser.parse_args()
    return options


class Trainer:
    def __init__(self):

        self.args = get_args()
        # self.device = torch.device(f"cuda:{self.args.gpu_id}")
        self.device = torch.device(f"cuda")

        print("Using device:", self.device)

        if self.args.gan_weight > 0:
            self.gan = True
            self.gan_weight = self.args.gan_weight
            self.norecon = self.args.noreconloss
            self.conditional = self.args.conditional
        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        if self.args.ihd:
            self.args.train = True
            self.dataset = IhdDataset(self.args)
        else:
            if self.args.random_aug:
                self.dataset = PIHDataRandom(self.args.datadir, device=self.device)
            else:
                self.dataset = PIHData(self.args.datadir, device=self.device)

        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            num_workers=self.args.workers,
            prefetch_factor=8,
            drop_last=True,
        )

        self.data_length = len(self.dataset)
        self.model = Model(feature_dim=self.args.features)

        if self.gan:
            if self.conditional:
                print("Using Conditional GAN!")
                self.model_D = networks.define_D(7, 64, "n_layers", 6)
            else:
                print("Using Non-conditional GAN!")
                self.model_D = networks.define_D(4, 64, "n_layers", 6)

        if torch.cuda.device_count() > 1 and self.args.multi_GPU:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)
            if self.gan:
                self.model_D = torch.nn.DataParallel(self.model_D)

        self.model.to(self.device)

        if self.gan:
            self.model_D.to(self.device)
        # self.model(command="per_gpu_initialize")
        self.criterion = torch.nn.L1Loss()

        if self.gan:
            self.criterion_GAN = networks.GANLoss("vanilla").to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        if self.gan:
            self.optimizer_D = torch.optim.Adam(
                self.model_D.parameters(), lr=self.args.learning_rate_d
            )

        self.start_epoch = 1
        if not self.args.force_train_from_scratch:
            self.restore_model()
        else:
            input("Training from scratch. Are you sure? (Ctrl+C to kill):")

        os.makedirs(
            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s"
            % (self.args.tempdir),
            exist_ok=True,
        )

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
            if self.gan:
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
        if self.gan:
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
        else:
            torch.save(
                {
                    "epoch": epoch,  # Epoch we just finished
                    "state_dict": self.model.state_dict(),
                    "optimizer_dict": self.optimizer.state_dict(),
                    "args": self.args,
                },
                os.path.join(self.checkpoint_directory, "ckpt{}.pth".format(epoch)),
            )

    def train(self):
        """Train the model!"""

        losses = []
        par = torch.tensor([0.0, 1.0])
        par.requires_grad = True

        # tqdm_bar = tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch")
        #         sys.exit()
        for epoch in range(self.start_epoch, self.args.epochs + 1):

            self.model.train()
            tqdm_bar = tqdm(enumerate(self.dataloader), "Index")

            for index, (input_image, input_mask, gt, names) in tqdm_bar:

                input_image = input_image.to(self.device)
                input_mask = input_mask.to(self.device)
                gt = gt.to(self.device)

                input_composite, output_composite, par1, par2 = self.model(
                    input_image, input_mask
                )

                brightness, contrast, saturation = par1
                b_r, b_g, b_b = par2

                loss_second = self.criterion(output_composite, gt)

                loss_first = self.criterion(input_composite, gt)

                loss = 2 * loss_second + 1 * loss_first

                if self.gan:

                    ## Update D

                    for param in self.model_D.parameters():
                        param.requires_grad = True

                    self.optimizer_D.zero_grad()

                    if self.conditional:
                        fake_AB = torch.cat(
                            (input_image, input_mask, output_composite), 1
                        )
                    else:
                        fake_AB = torch.cat((input_mask, output_composite), 1)

                    pred_fake = self.model_D(fake_AB.detach())
                    loss_D_fake = self.criterion_GAN(pred_fake, False)

                    if self.conditional:
                        real_AB = torch.cat((input_image, input_mask, gt), 1)
                    else:
                        real_AB = torch.cat((input_mask, gt), 1)

                    pred_real = self.model_D(real_AB)
                    loss_D_real = self.criterion_GAN(pred_real, True)

                    loss_D = 0.5 * (loss_D_fake + loss_D_real)

                    loss_D.backward()

                    self.optimizer_D.step()

                    ## Update G

                    for param in self.model_D.parameters():
                        param.requires_grad = False

                    self.optimizer.zero_grad()
                    if self.conditional:
                        fake_AB = torch.cat(
                            (input_image, input_mask, output_composite), 1
                        )
                    else:
                        fake_AB = torch.cat((input_mask, output_composite), 1)

                    pred_fake = self.model_D(fake_AB)
                    loss_G_adv = self.criterion_GAN(pred_fake, True)

                    if self.norecon:
                        loss_G = loss_G_adv

                    else:
                        loss_G = loss + 3 * (self.gan_weight) * loss_G_adv

                    loss_G.backward()
                    self.optimizer.step()

                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # print(loss.item())
                losses.append(loss.item())

                if epoch % 2 == 0 and index < 20:
                    # self.save_model(epoch)

                    for kk in range(self.args.batchsize):

                        name = names[kk].split("/")[-1].split(".")[0]
                        image_all = T.ToPILImage()(output_composite[kk, ...].cpu())
                        image_all.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_out.jpg"
                            % (self.args.tempdir, name)
                        )

                        image_i = T.ToPILImage()(input_composite[kk, ...].cpu())
                        image_i.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_inter.jpg"
                            % (self.args.tempdir, name)
                        )

                        image_gt = T.ToPILImage()(gt[kk, ...].cpu())
                        image_gt.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_gt.jpg"
                            % (self.args.tempdir, name)
                        )

                        image_og = T.ToPILImage()(input_image[kk, ...].cpu())
                        image_og.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_og.jpg"
                            % (self.args.tempdir, name)
                        )
                if self.gan:
                    tqdm_bar.set_description(
                        "E: {}. L: {:3f} L_G: {:3f} L_D: {:3f}".format(
                            epoch,
                            loss_second.item(),
                            loss_G.item(),
                            loss_D.item(),
                        )
                    )

                else:
                    if self.args.multi_GPU:
                        tqdm_bar.set_description(
                            "E: {}. L: {:3f} b: {:3f} c: {:3f} s: {:3f} br: {:3f} bg: {:3f} bb: {:3f}".format(
                                epoch,
                                loss_second.item(),
                                brightness[0].item(),
                                contrast[0].item(),
                                saturation[0].item(),
                                b_r[0].item(),
                                b_g[0].item(),
                                b_b[0].item(),
                            )
                        )
                    else:
                        tqdm_bar.set_description(
                            "E: {}. L: {:3f} b: {:3f} c: {:3f} s: {:3f} br: {:3f} bg: {:3f} bb: {:3f}".format(
                                epoch,
                                loss_second.item(),
                                brightness.item(),
                                contrast.item(),
                                saturation.item(),
                                b_r.item(),
                                b_g.item(),
                                b_b.item(),
                            )
                        )
            # print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n brightness {brightness} contrast {contrast} saturation {saturation} hue {hue}")
            # np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 10 == 0:
                self.save_model(epoch)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
