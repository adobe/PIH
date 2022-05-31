import numpy as np
import torch
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
from resnet import resnet18




class ColorStep1(torch.nn.Module):
    """
    Differentiable implementation of Color Transformation - Brightness, Contrast, Sturation.

    Implementation borrowed heavily from F.transform_tensor.
    """
    
    
    def __init__(self):
        super(ColorStep1, self).__init__()
                
        
    def _blend(self,img1, img2, ratio):

        bound = 1.0
    
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound)
    
    
    def rgb_to_grayscale(self,img):
        r, g, b = img.unbind(dim=-3)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        l_img = l_img.unsqueeze(dim=-3)
    
        return l_img
    
    
    def adjust_brightness(self, img, brightness_factor):


        return self._blend(img, torch.zeros_like(img), brightness_factor)
    
    
    
    def adjust_contrast(self,img, contrast_factor):

        mean = torch.mean(self.rgb_to_grayscale(img), dim=(-3, -2, -1), keepdim=True)


        return self._blend(img, mean, contrast_factor)
    
    def adjust_saturation(self,img, saturation_factor):
                
        return self._blend(img, self.rgb_to_grayscale(img), saturation_factor)
    
    def forward(self,image,feature,input_mask):
        """Change the color of an input image.
        Args:
            image (Pytorch Tensor): Original input image.
            feature (Pyotrch Tensor):Output of the resnet - predicting the parameters.
            mask (Pytorch Tesnor): Alpha matte to compose image.
        Returns:
            input_composite: first composite results.
        """
        
        
        brightness = feature[0]
        contrast = feature[1]
        saturation = feature[2]
        
                    
        input_out = self.adjust_brightness(image, brightness)
        
        input_out = self.adjust_contrast(input_out, contrast)
        
        
        input_out = self.adjust_saturation(input_out, saturation)
        
        
        
        input_composite = input_out * input_mask + (1-input_mask)*image
        
        return input_composite
    




class Model(torch.nn.Module):
    def __init__(
        self,
        feature_dim=128,
        color_space = 12,
        device=torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.network = resnet18(num_classes=feature_dim).to(device)

        self.color = resnet18(num_classes=color_space,input_f=7).to(device)
    
    def forward(self, images):
        embeddings = self.network(images)[0]  # B, H, W, F (channels last)

        # inter_features = self.network(images)[1]
        
        return embeddings
    
    def forward_color(self,images):
        
        color = self.color(images)[0]
        
        return color
