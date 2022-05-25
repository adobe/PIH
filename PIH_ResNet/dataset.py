from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import sys
class UFData(Dataset):

    def __init__(self, data_directory, max_offset=None, magnitude=False, device=torch.device('cpu'), phase_aug=False,mag_aug=False):
        """

        Parameters
        ----------
        data_directory : str
            The directory containing the training npy data.
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
        
        self.phase_aug = phase_aug
        self.mag_aug = mag_aug
        self.image_paths = glob(f"{data_directory}/*.npy")
        print(f"Using data from: {data_directory}\nFound {len(self.image_paths)} image paths.")
        self.device = device
        self.magnitude = magnitude

       

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
        original_image = np.load(self.image_paths[index])[None].astype(np.complex64)
        
        if self.mag_aug:
            original_image = self.augment_mag(original_image)
            
            
        if self.phase_aug:
            original_image = self.augment(original_image)
#             sys.exit()
        if self.magnitude:
            return index,torch.tensor(np.abs(original_image))
        else:
            return index,torch.tensor(np.concatenate((original_image.real, original_image.imag), axis=0))

    def augment(self,image):
        c = random.uniform(0, 1)
        c = np.exp(1j*2*np.pi*c)
        return image*c
    def augment_mag(self,image):
        c = random.uniform(0.9, 1.1)
        return image*c
        