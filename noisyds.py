import os
import torch

from skimage import io
from skimage.util import img_as_float32
from torch.utils.data import Dataset

class NoisyDataset(Dataset):
    def __init__(self, imDir) -> None:
        self.imDir = imDir
        self.imNames = os.listdir(os.path.join(self.imDir,"imp"))

    def __len__(self):
        return len(self.imNames)
    
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        noisyImName = os.path.join(self.imDir, "imp",
                                    self.imNames[i])
        truthImName = os.path.join(self.imDir, "org",
                                    self.imNames[i])

        noisyIm = img_as_float32(io.imread(noisyImName))
        truthIm = img_as_float32(io.imread(truthImName))

        sample = (noisyIm, truthIm)

        return sample