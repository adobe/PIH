# Copyright 2023 Adobe. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


from glob import glob
import os
import numpy as np
import torch

from PIL import Image

import torchvision.transforms as T
import torchvision.transforms.functional as F


from torch.utils.data import Dataset
import random
import sys


class PIHData(Dataset):
    def __init__(self, data_directory, device=torch.device("cpu")):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training image data.
        max_offset : tuple
            The maximum offset to crop an image to.
        magnitude : bool
            If True, train using magnitude image as input. Otherwise, use real and imaginary image in separate channels.
        device : torch.device
            The device to load the data to.
        complex : bool
            If True, return images as complex data. Otherwise check for magnitude return or for real and imaginary
            channels. This is needed when training, since post processing is done in the model (adds phase augmentation
            and converts to magnitude or channels). Magnitude and channels are implemented for evaluation.
        """

        self.image_paths = glob(f"{data_directory}/*_gt.jpg")
        print(
            f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths."
        )
        self.device = device
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        patch: torch.Tensor

        """

        image_path = self.image_paths[index]
        ground_truth = Image.open(image_path)
        input_image = Image.open(image_path[: image_path.rindex("_")] + ".jpg")
        input_mask = Image.open(image_path[: image_path.rindex("_")] + "_mask.jpg")

        # original_image = np.load(self.image_paths[index])[None].astype(np.complex64)

        return (
            self.transforms(input_image),
            self.transforms_mask(input_mask),
            self.transforms(ground_truth),
            image_path,
        )


class PIHDataRandom(Dataset):
    def __init__(self, data_directory, device=torch.device("cpu")):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training image data.
        max_offset : tuple
            The maximum offset to crop an image to.
        magnitude : bool
            If True, train using magnitude image as input. Otherwise, use real and imaginary image in separate channels.
        device : torch.device
            The device to load the data to.
        complex : bool
            If True, return images as complex data. Otherwise check for magnitude return or for real and imaginary
            channels. This is needed when training, since post processing is done in the model (adds phase augmentation
            and converts to magnitude or channels). Magnitude and channels are implemented for evaluation.
        """

        self.image_paths = glob(f"{data_directory}/*_gt.jpg")
        print(
            f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths."
        )
        self.device = device
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        patch: torch.Tensor

        """

        image_path = self.image_paths[index]

        ground_truth = self.transforms(Image.open(image_path))
        mask_torch = self.transforms(
            Image.open(image_path[: image_path.rindex("_")] + "_mask.jpg")
        )

        imag_torch = T.functional.adjust_contrast(
            ground_truth, (np.random.rand() * 0.4 + 0.8)
        )

        imag_torch = T.functional.adjust_brightness(
            imag_torch, (np.random.rand() * 0.4 + 0.8)
        )

        imag_torch = T.functional.adjust_saturation(
            imag_torch, (np.random.rand() * 0.4 + 0.8)
        )

        # Read functions for color transform: Cross - chaneel - YCC
        imag_torch[0, ...] = (
            imag_torch[0, ...] * (np.random.rand() * 0.3 + 0.70)
            + imag_torch[0, ...] * imag_torch[0, ...] * (np.random.rand() - 0.5) * 0.1
            + imag_torch[0, ...]
            * imag_torch[0, ...]
            * imag_torch[0, ...]
            * (np.random.rand() - 0.5)
            * 0.05
        )

        imag_torch[1, ...] = (
            imag_torch[1, ...] * (np.random.rand() * 0.3 + 0.70)
            + +imag_torch[1, ...] * imag_torch[1, ...] * (np.random.rand() - 0.5) * 0.1
            + imag_torch[1, ...]
            * imag_torch[1, ...]
            * imag_torch[1, ...]
            * (np.random.rand() - 0.5)
            * 0.05
        )

        imag_torch[2, ...] = (
            imag_torch[2, ...] * (np.random.rand() * 0.3 + 0.70)
            + imag_torch[2, ...] * imag_torch[2, ...] * (np.random.rand() - 0.5) * 0.1
            + imag_torch[2, ...]
            * imag_torch[2, ...]
            * imag_torch[2, ...]
            * (np.random.rand() - 0.5)
            * 0.05
        )

        imag_composite = ground_truth * (1 - mask_torch) + imag_torch * mask_torch
        # original_image = np.load(self.image_paths[index])[None].astype(np.complex64)

        return (
            imag_composite,
            mask_torch,
            ground_truth,
            image_path,
        )


class PIHDataNGT(Dataset):
    def __init__(self, data_directory, device=torch.device("cpu")):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training image data.
        max_offset : tuple
            The maximum offset to crop an image to.
        magnitude : bool
            If True, train using magnitude image as input. Otherwise, use real and imaginary image in separate channels.
        device : torch.device
            The device to load the data to.
        complex : bool
            If True, return images as complex data. Otherwise check for magnitude return or for real and imaginary
            channels. This is needed when training, since post processing is done in the model (adds phase augmentation
            and converts to magnitude or channels). Magnitude and channels are implemented for evaluation.
        """

        self.image_paths = glob(f"{data_directory}/*_mask.jpg")
        print(
            f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths."
        )
        self.device = device
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        patch: torch.Tensor

        """

        image_path = self.image_paths[index]
        # ground_truth = Image.open(image_path)
        input_image = Image.open(image_path[: image_path.rindex("_")] + ".jpg")
        input_mask = Image.open(image_path[: image_path.rindex("_")] + "_mask.jpg")

        # original_image = np.load(self.image_paths[index])[None].astype(np.complex64)

        return (
            self.transforms(input_image),
            self.transforms_mask(input_mask),
            self.transforms_mask(input_mask),
            image_path,
        )


class IhdDataset(Dataset):
    def __init__(self, opt):
        self.image_paths = []
        self.isTrain = opt.train
        if opt.train == True:
            print("loading training file")
            self.trainfile = opt.datadir + "IHD_train.txt"
            with open(self.trainfile, "r") as f:
                for line in f.readlines():
                    self.image_paths.append(
                        os.path.join(opt.datadir, "", line.rstrip())
                    )
        else:
            print("loading test file")
            self.trainfile = opt.datadir + "IHD_test.txt"
            with open(self.trainfile, "r") as f:
                for line in f.readlines():
                    self.image_paths.append(
                        os.path.join(opt.datadir, "", line.rstrip())
                    )
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
        self.image_size = 512

        print(
            f"Using data from: {opt.datadir}\nFound {len(self.image_paths)} image paths."
        )

    def __getitem__(self, index):

        path = self.image_paths[index]
        name_parts = path.split("_")
        mask_path = self.image_paths[index].replace("composite_images", "masks")
        mask_path = mask_path.replace(("_" + name_parts[-1]), ".png")
        target_path = self.image_paths[index].replace("composite_images", "real_images")
        target_path = target_path.replace(
            ("_" + name_parts[-2] + "_" + name_parts[-1]), ".jpg"
        )

        comp = Image.open(path).convert("RGB")
        real = Image.open(target_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if np.random.rand() > 0.5 and self.isTrain:
            comp, mask, real = F.hflip(comp), F.hflip(mask), F.hflip(real)

        if not (comp.size[0] == self.image_size and comp.size[1] == self.image_size):
            # assert 0
            comp = F.resize(comp, [self.image_size, self.image_size])
            mask = F.resize(mask, [self.image_size, self.image_size])
            real = F.resize(real, [self.image_size, self.image_size])

        comp = self.transforms(comp)
        mask = self.transforms_mask(mask)

        real = self.transforms(real)

        return (comp, mask, real, path)

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)


class DataCompositeGAN(Dataset):
    def __init__(self, data_directory, ratio=1, augment=False, colorjitter=True, lowres=False,return_raw=False, ratio_constrain=False):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training image data.
        """
        self.lowres = lowres
        self.image_paths = glob(f"{data_directory}/masks/*_mask.png")

        self.image_paths = self.image_paths[0 : int(len(self.image_paths) * ratio)]

        self.length = len(self.image_paths)
        print(
            f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths."
        )
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
        self.colorjitter = colorjitter
        if self.colorjitter:
            self.transform_color = T.ColorJitter(
                brightness=[0.65, 1.35], contrast=0.2, saturation=0, hue=0
            ) ## 0.3 0.7
        self.augment = augment
        self.returnraw = return_raw
        self.ratio_constrain = ratio_constrain
        if ratio_constrain:
            print("Using Constrained Ratio")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        patch: torch.Tensor

        Foreground
        """

        path_bg = self.image_paths[index]  # ForeGround

        path_fg = self.image_paths[np.random.randint(0, self.length)]

        ### fore-ground image loading

        path_fg_image = path_fg.replace("masks/", "real_images/")
        path_fg_image = path_fg_image.replace("_mask.png", ".png")

        path_fg_bg = path_fg.replace("masks/", "bg/")

        mask_fg = Image.open(path_fg)

        image_fg = Image.open(path_fg_image)

        image_fg_bg = Image.open(path_fg_bg)

            

        ### back-ground image loading

        path_bg_image = path_bg.replace("masks/", "real_images/")
        path_bg_image = path_bg_image.replace("_mask.png", ".png")

        path_bg_bg = path_bg.replace("masks/", "bg/")

        mask_bg = Image.open(path_bg)

        image_bg = Image.open(path_bg_image)
        
            

        image_bg_augment = image_bg

        if self.augment:
            if "before" in path_bg_image:
                path_bg_image_augment = path_bg_image.replace("before", "after")
            elif "after" in path_bg_image:
                path_bg_image_augment = path_bg_image.replace("after", "before")
                
            
            # image_bg_augment = Image.open(path_bg.replace("masks/",'composite/'))
            image_bg_augment = Image.open(path_bg_image_augment)
            

        image_bg_bg = Image.open(path_bg_bg)
        if self.lowres:
            mask_fg = mask_fg.resize((256,256))
            image_fg = image_fg.resize((256,256))
            image_fg_bg = image_fg_bg.resize((256,256))
            mask_bg = mask_bg.resize((256,256))
            image_bg = image_bg.resize((256,256))
            image_bg_augment = image_bg_augment.resize((256,256))
            image_bg_bg = image_bg_bg.resize((256,256))
            



        mask_bg_bbox = mask_bg.getbbox()
        mask_fg_bbox = mask_fg.getbbox()
        
        

        ## Target
        x_1_1, y_1_1, x_1_2, y_1_2 = mask_bg_bbox
        center_1_x = (x_1_1 + x_1_2) / 2
        center_1_y = (y_1_1 + y_1_2) / 2

        ##
        x_2_1, y_2_1, x_2_2, y_2_2 = mask_fg_bbox
        ration_x = (x_1_2 - x_1_1) / (x_2_2 - x_2_1) if x_2_2 != x_2_1 else 1
        ration_y = (y_1_2 - y_1_1) / (y_2_2 - y_2_1) if y_2_2 != y_2_1 else 1

        ## Scaling
        
        if not self.ratio_constrain:
        
            mask_fg_aff = F.affine(
                mask_fg, angle=0, translate=[0, 0], scale=min(ration_y, ration_x), shear=0
            )
            image_fg_aff = F.affine(
                image_fg, angle=0, translate=[0, 0], scale=min(ration_y, ration_x), shear=0
            )
        else:
            
            length_box = max(y_1_2-y_1_1,x_1_2-x_1_1)
        
            if length_box < 100:
                ration_x = (100) / (x_2_2 - x_2_1) if x_2_2 != x_2_1 else 1
                ration_y = (100) / (y_2_2 - y_2_1) if y_2_2 != y_2_1 else 1
            
            mask_fg_aff = F.affine(
            mask_fg, angle=0, translate=[0, 0], scale=min(ration_y, ration_x), shear=0)
            image_fg_aff = F.affine(
            image_fg, angle=0, translate=[0, 0], scale=min(ration_y, ration_x), shear=0
            )
        
        
        
        if mask_fg_aff.getbbox() == None:
            mask_fg_aff = F.affine(mask_fg, angle=0, translate=[0, 0], scale=1, shear=0)

        x_2_1_a, y_2_1_a, x_2_2_a, y_2_2_a = mask_fg_aff.getbbox()
        center_2_x_a = (x_2_1_a + x_2_2_a) / 2
        center_2_y_a = (y_2_1_a + y_2_2_a) / 2

        shift_fg_x = np.random.randint(-10, 10)
        shift_fg_y = np.random.randint(-10, 10)

        mask_fg_aff_all = F.affine(
            mask_fg_aff,
            angle=0,
            translate=[
                center_1_x - center_2_x_a + shift_fg_x,
                center_1_y - center_2_y_a + shift_fg_y,
            ],
            scale=1,
            shear=0,
        )
        image_fg_aff_all = F.affine(
            image_fg_aff,
            angle=0,
            translate=[
                center_1_x - center_2_x_a + shift_fg_x,
                center_1_y - center_2_y_a + shift_fg_y,
            ],
            scale=1,
            shear=0,
        )

        if self.colorjitter:
            if np.random.rand() < 1:
                # print("i love you one")
                image_fg_aff_all = self.transform_color(image_fg_aff_all)

        im_composite = Image.composite(image_fg_aff_all, image_bg_bg, mask_fg_aff_all)

        ## What we want to output? Background, im_composite, mask_fg_aff_all, real_image
        
        if self.returnraw:
            if self.colorjitter:
                if np.random.rand() < 1:
                    # print("i love you two")

                    image_bg_augment_f = self.transform_color(image_bg_augment)
                    image_bg_augment = Image.composite(image_bg_augment_f, image_bg, mask_bg)
            else:
                image_bg_augment = Image.composite(image_bg_augment, image_bg, mask_bg)
                
            
            return (
                self.transforms(image_bg_bg),
                self.transforms(im_composite),
                self.transforms_mask(mask_fg_aff_all),
                self.transforms(image_bg),
                self.transforms_mask(mask_bg),
                self.transforms(image_bg_augment),
                path_fg,
                path_bg,
        )
        
        else:

            shift_bg_x = np.random.randint(-10, 10)
            shift_bg_y = np.random.randint(-10, 10)

            mask_bg_shift = F.affine(
                mask_bg,
                angle=0,
                translate=[
                    shift_bg_x,
                    shift_bg_y,
                ],
                scale=1,
                shear=0,
            )

            image_bg_shift = F.affine(
                image_bg,
                angle=0,
                translate=[
                    shift_bg_x,
                    shift_bg_y,
                ],
                scale=1,
                shear=0,
            )

            image_bg_augment_shift = F.affine(
                image_bg_augment,
                angle=0,
                translate=[
                    shift_bg_x,
                    shift_bg_y,
                ],
                scale=1,
                shear=0,
            )

            im_real = Image.composite(image_bg_shift, image_bg_bg, mask_bg_shift)

            if self.colorjitter:
                if np.random.rand() < 1:
                    # print("i love you two")

                    image_bg_augment_shift = self.transform_color(image_bg_augment_shift)

            im_real_augment = Image.composite(
                image_bg_augment_shift, image_bg_bg, mask_bg_shift
            )

            # Dataset output orders: 1. Background (inpainted) 2. Image Composite 3. Mask 4. Real Image
            return (
                self.transforms(image_bg_bg),
                self.transforms(im_composite),
                self.transforms_mask(mask_fg_aff_all),
                self.transforms(im_real),
                self.transforms_mask(mask_bg_shift),
                self.transforms(im_real_augment),
                path_fg,
                path_bg,
            )


class PIHData_Composite(Dataset):
    def __init__(self, data_directory,lowres,original=False):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training image data.
        max_offset : tuple
            The maximum offset to crop an image to.
        magnitude : bool
            If True, train using magnitude image as input. Otherwise, use real and imaginary image in separate channels.
        device : torch.device
            The device to load the data to.
        complex : bool
            If True, return images as complex data. Otherwise check for magnitude return or for real and imaginary
            channels. This is needed when training, since post processing is done in the model (adds phase augmentation
            and converts to magnitude or channels). Magnitude and channels are implemented for evaluation.
        """

        self.image_paths = glob(f"{data_directory}/*_bg.jpg")
        print(
            f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths."
        )
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
        self.lowres = lowres
        self.original = original
        if lowres:
            self.res = 256
        else:
            self.res = 512

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        patch: torch.Tensor

        """

        image_path = self.image_paths[index]
        
        if self.original:
            input_bg = Image.open(image_path)
            
            input_composite = Image.open(image_path.replace("bg", "composite"))
            input_mask = Image.open(image_path.replace("bg", "mask"))
            if os.path.exists(image_path.replace("bg", "real")):
                input_real = Image.open(image_path.replace("bg", "real"))
            else:
                input_real = Image.open(image_path.replace("bg", "gt"))
        
        else:
            input_bg = Image.open(image_path).resize((self.res, self.res))
            
            input_composite = Image.open(image_path.replace("bg", "composite")).resize(
                (self.res, self.res)
            )
            input_mask = Image.open(image_path.replace("bg", "mask")).resize((self.res, self.res))
            if os.path.exists(image_path.replace("bg", "real")):
                input_real = Image.open(image_path.replace("bg", "real")).resize((self.res, self.res))
            else:
                input_real = Image.open(image_path.replace("bg", "gt")).resize((self.res, self.res))

        # original_image = np.load(self.image_paths[index])[None].astype(np.complex64)

        return (
            self.transforms(input_bg),
            self.transforms(input_composite),
            self.transforms_mask(input_mask),
            self.transforms(input_real),
            image_path,
        )


class DataCompositeGAN_iharmony(Dataset):
    def __init__(
        self, data_directory, ratio=1, augment=False, colorjitter=True, return_raw=False,lowres=False
    ):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training image data.
        """

        self.image_paths = glob(f"{data_directory}/masks/*_mask.png")

        self.image_paths = self.image_paths[0 : int(len(self.image_paths) * ratio)]

        self.length = len(self.image_paths)
        print(
            f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths."
        )
        self.transforms = T.Compose([T.ToTensor()])
        self.transforms_mask = T.Compose([T.Grayscale(), T.ToTensor()])
        self.colorjitter = colorjitter
        self.lowres = lowres
        if self.colorjitter:
            self.transform_color = T.ColorJitter(
                brightness=0.3, contrast=0.1, saturation=0.0, hue=0.0
            )
        self.augment = augment
        self.return_raw = return_raw

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Get image at the specified index.

        Parameters
        ----------
        index : int
            The image index.

        Returns
        -------
        patch: torch.Tensor

        Foreground
        """

        path_bg = self.image_paths[index]  # ForeGround

        path_fg = self.image_paths[np.random.randint(0, self.length)]

        ### fore-ground image loading

        path_fg_image = path_fg.replace("masks/", "real_images/")
        path_fg_image = path_fg_image.replace("_mask.png", ".jpg")

        path_fg_bg = path_fg.replace("masks/", "bg/")

        mask_fg = Image.open(path_fg)

        image_fg = Image.open(path_fg_image)

        image_fg_bg = Image.open(path_fg_bg)

        ### back-ground image loading

        path_bg_image = path_bg.replace("masks/", "real_images/")
        path_bg_image = path_bg_image.replace("_mask.png", ".jpg")

        path_bg_bg = path_bg.replace("masks/", "bg/")

        mask_bg = Image.open(path_bg)

        image_bg = Image.open(path_bg_image)

        image_bg_augment = image_bg

        if self.augment:
            path_bg_image_augment = path_bg_image.replace(
                "real_images", "composite"
            ).replace(".jpg", "_composite.jpg")
            # if "before" in path_bg_image:
            #     path_bg_image_augment = path_bg_image.replace("before", "after")
            # elif "after" in path_bg_image:
            #     path_bg_image_augment = path_bg_image.replace("after", "before")
            image_bg_augment = Image.open(path_bg_image_augment)

        image_bg_bg = Image.open(path_bg_bg)
        
        
        if self.lowres:
            mask_fg = mask_fg.resize((256,256))
            image_fg = image_fg.resize((256,256))
            image_fg_bg = image_fg_bg.resize((256,256))
            mask_bg = mask_bg.resize((256,256))
            image_bg = image_bg.resize((256,256))
            image_bg_augment = image_bg_augment.resize((256,256))
            image_bg_bg = image_bg_bg.resize((256,256))

        mask_bg_bbox = mask_bg.getbbox()
        mask_fg_bbox = mask_fg.getbbox()

        ## Target
        x_1_1, y_1_1, x_1_2, y_1_2 = mask_bg_bbox
        center_1_x = (x_1_1 + x_1_2) / 2
        center_1_y = (y_1_1 + y_1_2) / 2

        ##
        x_2_1, y_2_1, x_2_2, y_2_2 = mask_fg_bbox
        ration_x = (x_1_2 - x_1_1) / (x_2_2 - x_2_1) if x_2_2 != x_2_1 else 1
        ration_y = (y_1_2 - y_1_1) / (y_2_2 - y_2_1) if y_2_2 != y_2_1 else 1

        ## Scaling
        mask_fg_aff = F.affine(
            mask_fg, angle=0, translate=[0, 0], scale=min(ration_y, ration_x), shear=0
        )
        image_fg_aff = F.affine(
            image_fg, angle=0, translate=[0, 0], scale=min(ration_y, ration_x), shear=0
        )
        if mask_fg_aff.getbbox() == None:
            mask_fg_aff = F.affine(mask_fg, angle=0, translate=[0, 0], scale=1, shear=0)

        x_2_1_a, y_2_1_a, x_2_2_a, y_2_2_a = mask_fg_aff.getbbox()
        center_2_x_a = (x_2_1_a + x_2_2_a) / 2
        center_2_y_a = (y_2_1_a + y_2_2_a) / 2

        shift_fg_x = np.random.randint(-10, 10)
        shift_fg_y = np.random.randint(-10, 10)

        mask_fg_aff_all = F.affine(
            mask_fg_aff,
            angle=0,
            translate=[
                center_1_x - center_2_x_a + shift_fg_x,
                center_1_y - center_2_y_a + shift_fg_y,
            ],
            scale=1,
            shear=0,
        )
        image_fg_aff_all = F.affine(
            image_fg_aff,
            angle=0,
            translate=[
                center_1_x - center_2_x_a + shift_fg_x,
                center_1_y - center_2_y_a + shift_fg_y,
            ],
            scale=1,
            shear=0,
        )

        if self.colorjitter:
            if np.random.rand() < 1:
                # print("i love you one")
                image_fg_aff_all = self.transform_color(image_fg_aff_all)

        im_composite = Image.composite(image_fg_aff_all, image_bg_bg, mask_fg_aff_all)

        ## What we want to output? Background, im_composite, mask_fg_aff_all, real_image

        if self.return_raw:
            
            if self.colorjitter:
                if np.random.rand() < 1:
                    # print("i love you two")

                    image_bg_augment_f = self.transform_color(image_bg_augment)
                    image_bg_augment = Image.composite(image_bg_augment_f, image_bg_augment, mask_bg)
            # else:
            #     image_bg_augment = Image.composite(image_bg_augment, image_bg, mask_bg)
            
            
            
            return (
                self.transforms(image_bg_bg),
                self.transforms(im_composite),
                self.transforms_mask(mask_fg_aff_all),
                self.transforms(image_bg),
                self.transforms_mask(mask_bg),
                self.transforms(image_bg_augment),
                path_fg,
                path_bg,
            )
        else:

            shift_bg_x = np.random.randint(-10, 10)
            shift_bg_y = np.random.randint(-10, 10)

            mask_bg_shift = F.affine(
                mask_bg,
                angle=0,
                translate=[
                    shift_bg_x,
                    shift_bg_y,
                ],
                scale=1,
                shear=0,
            )

            image_bg_shift = F.affine(
                image_bg,
                angle=0,
                translate=[
                    shift_bg_x,
                    shift_bg_y,
                ],
                scale=1,
                shear=0,
            )

            image_bg_augment_shift = F.affine(
                image_bg_augment,
                angle=0,
                translate=[
                    shift_bg_x,
                    shift_bg_y,
                ],
                scale=1,
                shear=0,
            )

            im_real = Image.composite(image_bg_shift, image_bg_bg, mask_bg_shift)

            if self.colorjitter:
                if np.random.rand() < 1:
                    # print("i love you two")

                    image_bg_augment_shift = self.transform_color(
                        image_bg_augment_shift
                    )

            im_real_augment = Image.composite(
                image_bg_augment_shift, image_bg_bg, mask_bg_shift
            )

            # Dataset output orders: 1. Background (inpainted) 2. Image Composite 3. Mask 4. Real Image
            return (
                self.transforms(image_bg_bg),
                self.transforms(im_composite),
                self.transforms_mask(mask_fg_aff_all),
                self.transforms(im_real),
                self.transforms_mask(mask_bg_shift),
                self.transforms(im_real_augment),
                path_fg,
                path_bg,
            )
