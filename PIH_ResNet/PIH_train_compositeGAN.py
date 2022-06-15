import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import DataCompositeGAN
from model import Model, Model_Composite
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
        "--frequency",
        default=3,
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

        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.dataset = DataCompositeGAN(self.args.datadir)

        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            num_workers=self.args.workers,
            prefetch_factor=8,
            drop_last=True,
        )

        self.data_length = len(self.dataset)
        self.model = Model_Composite(feature_dim=self.args.features)

        self.model_D = networks.define_D(4, 64, "n_layers", 6)

        if torch.cuda.device_count() > 1 and self.args.multi_GPU:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)

            self.model_D = torch.nn.DataParallel(self.model_D)

        self.model.to(self.device)

        self.model_D.to(self.device)

        self.criterion_GAN = networks.GANLoss("vanilla").to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        self.optimizer_D = torch.optim.SGD(
            self.model_D.parameters(), lr=self.args.learning_rate_d, momentum=0.9
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
        for epoch in range(self.start_epoch, self.args.epochs + 1):

            self.model.train()
            tqdm_bar = tqdm(enumerate(self.dataloader), "Index")

            for index, (
                image_bg_bg,
                im_composite,
                mask,
                im_real,
                mask_bg,
                fname,
                bname,
            ) in tqdm_bar:

                image_bg_bg = image_bg_bg.to(self.device)
                im_composite = im_composite.to(self.device)
                mask = mask.to(self.device)
                im_real = im_real.to(self.device)
                mask_bg = mask_bg.to(self.device)

                input_composite, output_composite, par1, par2 = self.model(
                    image_bg_bg, im_composite, mask
                )

                brightness, contrast, saturation = par1
                b_r, b_g, b_b = par2

                ## Update D

                if index % self.args.frequency == 0:

                    for param in self.model_D.parameters():
                        param.requires_grad = True

                    self.optimizer_D.zero_grad()

                    fake_AB = torch.cat((mask, output_composite), 1)

                    pred_fake = self.model_D(fake_AB.detach())
                    loss_D_fake = self.criterion_GAN(pred_fake, False)

                    real_AB = torch.cat((mask_bg, im_real), 1)

                    pred_real = self.model_D(real_AB)
                    loss_D_real = self.criterion_GAN(pred_real, True)

                    loss_D = 0.5 * (loss_D_fake + loss_D_real)

                    loss_D.backward()

                    self.optimizer_D.step()

                ## Update G

                for param in self.model_D.parameters():
                    param.requires_grad = False

                self.optimizer.zero_grad()

                fake_AB = torch.cat((mask, output_composite), 1)

                pred_fake = self.model_D(fake_AB)
                loss_G_adv = self.criterion_GAN(pred_fake, True)

                loss_G_adv.backward()

                self.optimizer.step()

                if epoch % 1 == 0 and index < 20:
                    # self.save_model(epoch)

                    for kk in range(self.args.batchsize):

                        name = (
                            fname[kk].split("/")[-1].split(".")[0]
                            + "_"
                            + bname[kk].split("/")[-1].split(".")[0]
                        )
                        name = "%d_%d" % (index, kk)
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

                        image_real = T.ToPILImage()(im_real[kk, ...].cpu())
                        image_real.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_real.jpg"
                            % (self.args.tempdir, name)
                        )

                        image_og = T.ToPILImage()(im_composite[kk, ...].cpu())
                        image_og.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_training/%s/%s_composite.jpg"
                            % (self.args.tempdir, name)
                        )

                tqdm_bar.set_description(
                    "E: {}. L_G: {:3f} L_D: {:3f}".format(
                        epoch,
                        loss_G_adv.item(),
                        loss_D.item(),
                    )
                )
            # print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n brightness {brightness} contrast {contrast} saturation {saturation} hue {hue}")
            # np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 2 == 0:
                self.save_model(epoch)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
