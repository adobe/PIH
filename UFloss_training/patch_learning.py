import os
import re
from glob import glob
from optparse import OptionParser
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import UFData
from model import Model
from tqdm import tqdm
from resnet import resnet18,resnet18_m
from network import SimpleNet

## Just for plotting
import matplotlib

matplotlib.use("TkAgg")
import sigpy.plot as pl

# TODO: Tensorboard
# TODO: Learning rate decay
# TODO: Tune temperature (~0.07?)
# TODO: Maybe sample from memory bank too?
# Done: Sample from encodings?


def get_args():
    parser = OptionParser()
    parser.add_option("--datadir", "--dd", help="Directory contains 2D patches.")
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
        default=128,
        type="int",
        help="Dimension of the feature space.",
    )
    parser.add_option(
        "--learning-rate",
        "--lr",
        default=1e-4,
        type="float",
        help="learning rate for the model",
    )
    parser.add_option(
        "--temperature",
        "--temp",
        default=1.00,
        type=float,
        help="temperature parameter default: 1",
    )
    parser.add_option(
        "--batchsize",
        "--bs",
        dest="batchsize",
        default=32,
        type="int",
        help="batch size for training",
    )
    parser.add_option(
        "-e", "--epochs", default=200, type="int", help="Number of epochs to train"
    )
    # parser.add_option('-m', '--model', dest='model',
    #                   default=False, help='load checkpoints')
    parser.add_option(
        "--use_magnitude",
        action="store_true",
        default=False,
        help="If specified, use image magnitude.",
    )
    parser.add_option(
        "--use_phase_augmentation",
        action="store_true",
        default=False,
        help="If specified, use phase augmentation.",
    )
    parser.add_option(
        "--use_mag_augmentation",
        action="store_true",
        default=False,
        help="If specified, use mag augmentation. (randomly scale it by a factor between 0.9 and 1.1)",
    )
    # parser.add_option('-x', '--sx', dest='sx',
    #                   default=256, type='int', help='image dim: x')
    # parser.add_option('-y', '--sy', dest='sy',
    #                   default=320, type='int', help='image dim: y')
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
        print("Using magnitude:", self.args.use_magnitude)

        self.checkpoint_directory = os.path.join(f"{self.args.logdir}", "checkpoints")
        os.makedirs(self.checkpoint_directory, exist_ok=True)

        self.dataset = UFData(
            self.args.datadir,
            magnitude=bool(self.args.use_magnitude),
            device=self.device,
            phase_aug=self.args.use_phase_augmentation,
            mag_aug=self.args.use_mag_augmentation,
        )
        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            drop_last=True,
            num_workers=24,
            prefetch_factor=4,
        )

        self.data_length = len(self.dataset)
        if bool(self.args.use_magnitude):
            self.model = Model(
                resnet18_m,
                temperature=self.args.temperature,
                feature_dim=self.args.features,
                device=self.device,
                data_length=self.data_length,
            )
        else:
            self.model = Model(
                resnet18,
                temperature=self.args.temperature,
                feature_dim=self.args.features,
                device=self.device,
                data_length=self.data_length,
            )
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
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

        # Initiate the memory bank
        self.model.eval()
        for index, (indices, images) in enumerate(tqdm(self.dataloader, "Step")):

            images = images.to(self.device)
            embeddings = self.model(images)
            with torch.no_grad():
                self.model.update_memory_bank(embeddings, indices)

        #         sys.exit()
        for epoch in tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch"):

            self.model.train()
            for index, (indices, images) in enumerate(tqdm(self.dataloader, "Step")):

                images = images.to(self.device)
                embeddings = self.model(images)
                #
                logits = self.model.get_logits_labels(embeddings)

                loss = self.criterion(logits, indices.to(self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.model.update_memory_bank(embeddings, indices)

                if index % 400 == 0:
                    print(loss.item())
                    losses.append(loss.item())

            print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n")
            np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 10 == 0:
                self.save_model(epoch)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
