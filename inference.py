#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import PIL
from PIL import Image
import cv2
from model import Model_Composite_PL, Model_Composite
from optparse import OptionParser
import os
import time


transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
transform = T.Compose([T.ToTensor()])
resize = T.Resize((512, 512))


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst




def get_args():
    parser = OptionParser()
    parser.add_option("--bg", help="Directory to the background image.")
    parser.add_option("--fg", help="Directory to the foreground image.")
    
    parser.add_option("--checkpoints", "--ld", help="Directory to checkpoints, default is model/ckpt_g39.pth")
    
    parser.add_option(
        "--gpu",
        action="store_true",
        help="If specified, will use GPU",
    )
    
    parser.add_option(
        "--light",
        action="store_true",
        help="If specified, will use light model",
    )
    
    (options, args) = parser.parse_args()
    return options





class Evaluater:
    def __init__(self):

        self.args = get_args()
        
        self.name_cat = self.args.bg.split('/')[-1].split('.')[0]+'_'+self.args.fg.split('/')[-1].split('.')[0]
        
        
        self.fg = Image.open(self.args.fg)
        
        self.mask = self.fg.split()[-1]
        
        self.background = Image.open(self.args.bg).resize(self.fg.size)
        
        self.img_composite = Image.composite(self.fg, self.background, self.mask)
        
        


        
        if self.args.gpu:
            device = "cuda"
        else:
            device = "cpu"
            
        self.Model = Model_Composite_PL(
                            dim=32,
                            masking=True,
                            brush=True,
                            maskoffset=0.6,
                            swap=True,
                            Vit_bool=False,
                            onlyupsample=True,
                            aggupsample=True,
                            light=self.args.light,
                            Eff_bool=self.args.light,
                        ).to(device)
        
        if self.args.checkpoints is not None:
            model_path = self.args.checkpoints
        else:
            model_path = os.getcwd() + '/pretrained/ckpt_g39.pth'
        
        checkpoint = torch.load(model_path, map_location=device)
        self.Model.load_state_dict(checkpoint["state_dict"])
        
        self.Model.eval()
        self.bg_low= resize(self.background)
        self.composite_low= resize(self.img_composite)
        self.mask_low = resize(self.mask)

        # Load image


        self.torch_bg = transform(self.background).to(device)
        self.torch_composite = transform(self.img_composite).to(device)
        self.torch_mask = transforms_mask(self.mask).to(device)
        
        self.torch_bg_low = transform(self.bg_low).to(device)
        self.torch_composite_low = transform(self.composite_low).to(device)
        self.torch_mask_low = transforms_mask(self.mask_low).to(device)
        
    def evaluate(self):
                
        with torch.no_grad():
            inter_composite, output_composite, par1, par2 = self.Model(
                self.torch_bg_low[None, ...],
                self.torch_composite_low[None, ...],
                self.torch_mask_low[None, ...],
            )

        
        hr_intermediate = (
                self.Model.PL3D(self.Model.pl_table, self.torch_composite[None,...]) * self.torch_mask
                + (1 - self.torch_mask) * self.torch_bg
            )
        
        Gainmap_Resize = T.Resize(self.torch_bg.shape[-2:])
        # print(Gain_map)
        
        output_results = (
                    hr_intermediate * Gainmap_Resize(self.Model.gainmap) * self.torch_mask
                    + (1 - self.torch_mask) * self.torch_bg
                )
        
        output_lr = T.ToPILImage()(output_results[0,...])
        output_lr.save('results/%s_final.png'%(self.name_cat))
        
        output_gm = T.ToPILImage()( (Gainmap_Resize(self.Model.gainmap) * self.torch_mask)[0,...])
        
        output_gm.save('results/%s_gainmap.png'%(self.name_cat))
        
        
        
        
        
        #### Save Fig
        
        curves = par2.cpu().detach().numpy()

        red_curve = curves[0, 0, 0, 0, :]
        green_curve = curves[0, 1, 0, :, 0]
        blue_curve = curves[0, 2, :, 0, 0]

        plt.figure()
        plt.plot(np.linspace(0, 1, 32), red_curve, "r")
        plt.plot(np.linspace(0, 1, 32), green_curve, "g")
        plt.plot(np.linspace(0, 1, 32), blue_curve, "b")
        plt.ylim(0, 1)
        plt.legend(["Reg", "Green", "Blue"])
        plt.title("Learned Color Curves")

        plt.savefig("results/%s_color.jpg"%(self.name_cat))
        
        
        
        plt.close()
        
        im_final = get_concat_h( self.img_composite,get_concat_h(self.mask,output_lr))
        
        im_final.save('results/%s_results_summary.png'%(self.name_cat))
        

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    evaluater = Evaluater()
    evaluater.evaluate()