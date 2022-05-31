import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PIHData
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
        default=1,
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
    (options, args) = parser.parse_args()
    return options


class Trainer:
    def __init__(self):

        self.args = get_args()
        self.device = torch.device(f"cuda:{self.args.gpu_id}")
        print("Using device:", self.device)

        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.dataset = PIHData(self.args.datadir, device=self.device)
        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            num_workers=1,
        )

        self.data_length = len(self.dataset)
        self.model = Model(feature_dim=self.args.features, device=self.device)
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
        for epoch in tqdm_bar:

            self.model.train()
            for index, (input_image, input_mask, gt) in enumerate(self.dataloader):

                self.model.initiate(input_image, input_mask, gt)

                input_composite, output_composite = self.model()

                loss = 2 * self.criterion(
                    output_composite, self.model.gt
                ) + 0 * self.criterion(input_composite, self.model.gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(loss.item())
                losses.append(loss.item())

                if epoch % 100 == 0:
                    # self.save_model(epoch)
                    image_all = T.ToPILImage()(output_composite[0, ...].cpu())
                    image_all.save(
                        "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp%d.jpg"
                        % (index)
                    )

                    image_i = T.ToPILImage()(input_composite[0, ...].cpu())
                    image_i.save(
                        "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp%d_inter.jpg"
                        % (index)
                    )

            tqdm_bar.set_description(
                "E: {}. L: {:3f} b: {:3f} c: {:3f} s: {:3f} br: {:3f} bg: {:3f} bb: {:3f}".format(
                    epoch,
                    loss.item(),
                    self.model.brightness,
                    self.model.contrast,
                    self.model.saturation,
                    self.model.b_r,
                    self.model.b_g,
                    self.model.b_b,
                )
            )
            # print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n brightness {brightness} contrast {contrast} saturation {saturation} hue {hue}")
            np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 100 == 0:
                self.save_model(epoch)
                image_all = T.ToPILImage()(output_composite[0, ...].cpu())
                image_all.save(
                    "/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp1.jpg"
                )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
