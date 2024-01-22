import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import input_mapping, get_B_gauss


class FourierFeaturesPixelDataset(Dataset):

    def __init__(self, img, B, train=True):
        """Dataset of Fourier features for a single image
        Parameters
        ----------
        img : np.ndarray
            Image.
        B : torch.Tensor
            Matrix of random parameters.
        train : bool, optional
            If True, the dataset is for training (1/4 pixels of the image), otherwise it is for validation (all pixels of the image). 
            The default is True.
        """
        super(FourierFeaturesPixelDataset, self).__init__()
        self.img = img
        self.train = train
        self.B_gauss = B
        
        self.data = self.extract_features()

        self.features = input_mapping(self.data[0], self.B_gauss)
        self.target = self.data[1]



    def extract_features(self):
        """Extract the features from the image
        Returns
        -------
        list
            List of the features and the target.
        """
        coords = np.linspace(0, 1, self.img.shape[0], endpoint=False)
        pixel_grid = np.stack(np.meshgrid(coords, coords), -1)
        if self.train == True:
            data = [torch.Tensor(pixel_grid[::2, ::2]).view(-1, 2), torch.Tensor(self.img[::2, ::2]).view(-1, self.img.shape[-1])]
        else: 
            data = [torch.Tensor(pixel_grid).view(-1, 2), torch.Tensor(self.img).view(-1, self.img.shape[-1])]
        return data
    


    def __len__(self):
        """Get the length of the dataset"""
        return len(self.features)



    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        data_sample = self.features[idx]
        target_sample = self.target[idx]
        return data_sample, target_sample



    

