from glob import glob

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
        )
