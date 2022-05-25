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
from resnet import resnet18,resnet18_m
from network import SimpleNet


import torchvision.transforms as T
import torchvision.transforms.functional as F
# TODO: Tensorboard
# TODO: Learning rate decay
# TODO: Tune temperature (~0.07?)
# TODO: Maybe sample from memory bank too?
# Done: Sample from encodings?


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
        default=2,
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

        self.dataset = PIHData(
            self.args.datadir,
            device=self.device
        )
        self.dataloader = DataLoader(
            self.dataset,
            self.args.batchsize,
            shuffle=True,
            num_workers=1,
            # prefetch_factor=2,
        )

        self.data_length = len(self.dataset)
        self.model = Model(
            resnet18,
            feature_dim=self.args.features,
            device=self.device
        )
        self.criterion = torch.nn.MSELoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.learning_rate,momentum=0.9
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
        par = torch.tensor([0.0,1.0])
        par.requires_grad = True
        self.optimizer2 = torch.optim.SGD(
            [par], lr=self.args.learning_rate,momentum=0.9
        )

        #         sys.exit()
        for epoch in tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch"):

            self.model.train()
            for index, (input_image, input_mask, gt) in enumerate(tqdm(self.dataloader, "Step")):


                
                input_image = input_image.to(self.device)
                input_mask = input_mask.to(self.device)
                gt = gt.to(self.device)
                
                input_all = torch.cat((input_image,input_mask),1)
                
                embeddings = self.model(input_all)[0,...]
                #

                brightness = abs(embeddings[0])
                saturation = abs(embeddings[1])
                # brightness = par[0]
                # saturation = par[1]
                # contrast = embeddings[2]
                # hue = embeddings[3]
                
                # input_out = torch.clip(input_image*brightness,min=0,max=1)
                
                input_out = input_image.clone()
                r, g, b = input_out.unbind(dim=-3)
  
                l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(input_out.dtype)
                l_img = l_img.unsqueeze(dim=-3)
                
                mean = torch.mean(l_img, dim=(-3, -2, -1), keepdim=True)
                
                input_out = (saturation * input_out + (1.0 - saturation) * mean).clamp(0, 1)
                input_out = torch.clip(input_out*brightness,min=0,max=1)
                
                # input_out = T.functional.adjust_brightness(input_image,brightness)
                # input_out = T.functional.adjust_saturation(input_out,saturation)
                
                input_composite = input_out * input_mask + (1-input_mask)*input_image
                
                loss = self.criterion(input_composite, gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print(loss.item())
                losses.append(loss.item())
                
                
                if epoch % 100 == 0:
                    # self.save_model(epoch)
                    image_all = T.ToPILImage()(input_composite[0,...].cpu())
                    image_all.save("/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp%d.jpg"%(index))

            print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n brightness {brightness} saturation {saturation}")
            np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 100 == 0:
                self.save_model(epoch)
                image_all = T.ToPILImage()(input_composite[0,...].cpu())
                image_all.save("/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp1.jpg")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
