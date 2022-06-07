import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PIHData, PIHDataRandom
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
        "--batchsize",
        "--bs",
        dest="batchsize",
        default=4,
        type="int",
        help="batch size for training",
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

        if self.args.random_aug:
            print("haha")
            self.dataset = PIHDataRandom(self.args.datadir, device=self.device)
        else:
            self.dataset = PIHData(self.args.datadir, device=self.device)

        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            num_workers=8,
            prefetch_factor=4,
            drop_last=True,
        )

        self.data_length = len(self.dataset)
        self.model = Model(feature_dim=self.args.features)

        if torch.cuda.device_count() > 1 and self.args.multi_GPU:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)
        # self.model(command="per_gpu_initialize")
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )

        self.start_epoch = 1
        if not self.args.force_train_from_scratch:
            self.restore_model()
        else:
            input("Training from scratch. Are you sure? (Ctrl+C to kill):")

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
            },
            os.path.join(self.checkpoint_directory, "ckpt{}.pth".format(epoch)),
        )

    def train(self):
        """Train the model!"""

        losses = []
        par = torch.tensor([0.0, 1.0])
        par.requires_grad = True

        tqdm_bar = tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch")
        #         sys.exit()
        for epoch in range(self.start_epoch, self.args.epochs + 1):

            self.model.train()
            tqdm_bar = tqdm(enumerate(self.dataloader), "Index")

            for index, (input_image, input_mask, gt) in tqdm_bar:

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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(loss.item())
                losses.append(loss.item())

                if epoch % 10 == 0:
                    # self.save_model(epoch)

                    for kk in range(self.args.batchsize):

                        image_all = T.ToPILImage()(output_composite[kk, ...].cpu())
                        image_all.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_results/tmp%d_%d.jpg"
                            % (index, kk)
                        )

                        image_i = T.ToPILImage()(input_composite[kk, ...].cpu())
                        image_i.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_results/tmp%d_%d_inter.jpg"
                            % (index, kk)
                        )

                        image_gt = T.ToPILImage()(gt[kk, ...].cpu())
                        image_gt.save(
                            "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/temp_results/tmp%d_%d__gt.jpg"
                            % (index, kk)
                        )

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
            np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 10 == 0:
                self.save_model(epoch)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
