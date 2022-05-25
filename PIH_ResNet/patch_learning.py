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
from torch import Tensor

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
        default=3,
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
        self.criterion = torch.nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate,
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


    def _blend(self,img1, img2, ratio):
        # ratio = float(ratio)
        bound = 1.0
    
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound)
    
    
    def rgb_to_grayscale(self,img):
        r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
        # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
    
        return l_img
    
    
    def adjust_brightness(self, img, brightness_factor):


        return self._blend(img, torch.zeros_like(img), brightness_factor)
    
    
    
    def adjust_contrast(self,img, contrast_factor):

        mean = torch.mean(self.rgb_to_grayscale(img), dim=(-3, -2, -1), keepdim=True)


        return self._blend(img, mean, contrast_factor)
    
    def adjust_saturation(self,img, saturation_factor):
        
                        # l_img = self.rgb_to_grayscale(input_out)
                # input_out = (saturation * input_out + (1.0 - saturation) * l_img).clamp(0, 1)
        
        
        return self._blend(img, self.rgb_to_grayscale(img), saturation_factor)
    
    
    
    def _rgb2hsv(self,img):
        r, g, b = img.unbind(dim=-3)

        # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
        # src/libImaging/Convert.c#L330
        maxc = torch.max(img, dim=-3).values
        minc = torch.min(img, dim=-3).values

        # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
        # from happening in the results, because
        #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
        #   + H channel has division by `(maxc - minc)`.
        #
        # Instead of overwriting NaN afterwards, we just prevent it from occuring so
        # we don't need to deal with it in case we save the NaN in a buffer in
        # backprop, if it is ever supported, but it doesn't hurt to do so.
        eqc = maxc == minc

        cr = maxc - minc
        # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
        ones = torch.ones_like(maxc)
        s = cr / torch.where(eqc, ones, maxc)
        # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
        # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
        # would not matter what values `rc`, `gc`, and `bc` have here, and thus
        # replacing denominator with 1 when `eqc` is fine.
        cr_divisor = torch.where(eqc, ones, cr)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = torch.fmod((h / 6.0 + 1.0), 1.0)
        return torch.stack((h, s, maxc), dim=-3)


    def _hsv2rgb(self,img):
        h, s, v = img.unbind(dim=-3)
        i = torch.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.to(dtype=torch.int32)

        p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
        q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
        t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6

        mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)

        a1 = torch.stack((v, q, p, p, t, v), dim=-3)
        a2 = torch.stack((t, v, v, q, p, p), dim=-3)
        a3 = torch.stack((p, p, t, v, v, q), dim=-3)
        a4 = torch.stack((a1, a2, a3), dim=-4)

        return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)

    
    
    def adjust_hue(self,img, hue_factor):

        img = self._rgb2hsv(img)
        h, s, v = img.unbind(dim=-3)
        h = (h + hue_factor) % 1.0
        img = torch.stack((h, s, v), dim=-3)
        img_hue_adj =self._hsv2rgb(img)

        return img_hue_adj
    
    
    
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

        tqdm_bar = tqdm(range(self.start_epoch, self.args.epochs + 1), "Epoch")
        #         sys.exit()
        for epoch in tqdm_bar:

            self.model.train()
            for index, (input_image, input_mask, gt) in enumerate(self.dataloader):


                
                input_image = input_image.to(self.device)
                input_mask = input_mask.to(self.device)
                gt = gt.to(self.device)
                
                input_all = torch.cat((input_image,input_mask),1)
                
                embeddings = self.model(input_all)[0,...]
                #

                brightness = abs(embeddings[0])
                contrast = abs(embeddings[1])
                saturation = abs(embeddings[2])
                hue = 0
                
                # brightness = par[0]
                # saturation = par[1]
                # contrast = embeddings[2]
                # hue = embeddings[3]
                
                input_out = input_image.clone()
                
                

                
                input_out = self.adjust_brightness(input_out, brightness)
                    
                
                input_out = self.adjust_contrast(input_out, contrast)
                
                
                input_out = self.adjust_saturation(input_out, saturation)
                

                
                
                
                # input_out = self.adjust_hue(input_out,hue)
                
                
                
                
                
                # input_out = torch.clip(input_out*brightness,min=0,max=1)
                
                # input_out = T.functional.adjust_brightness(input_image,brightness)
                # input_out = T.functional.adjust_saturation(input_out,saturation)
                
                
                
                
                
                input_composite = input_out * input_mask + (1-input_mask)*input_image
                
            
                
                loss = self.criterion(input_composite, gt)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print(loss.item())
                losses.append(loss.item())
                
                
                if epoch % 100 == 0:
                    # self.save_model(epoch)
                    image_all = T.ToPILImage()(input_composite[0,...].cpu())
                    image_all.save("/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp%d.jpg"%(index))

            tqdm_bar.set_description("E: {}. L: {:3f} b: {:3f} c: {:3f} s: {:3f}".format(epoch,loss.item(),brightness,contrast,saturation))
            # print(f"\n\n\tEpoch {epoch}. Loss {loss.item()}\n brightness {brightness} contrast {contrast} saturation {saturation} hue {hue}")
            np.save(os.path.join(self.args.logdir, "loss_all.npy"), np.array(losses))

            if epoch % 100 == 0:
                self.save_model(epoch)
                image_all = T.ToPILImage()(input_composite[0,...].cpu())
                image_all.save("/home/kewang/sensei-fs-symlink/users/kewang/projects/data_processing/tmp1.jpg")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    trainer = Trainer()
    trainer.train()
