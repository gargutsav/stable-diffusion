import os
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

class LGS(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 phase="train",
                 center_crop=False,
                 ):

        self.data_root = data_root
        self.metadata = pd.read_csv(f'{self.data_root}/{phase}_subset.csv')

        self.image_paths = [os.path.join(self.data_root, 'imgs_256_04_27', f_path) for f_path in self.metadata['images'].values]
        self.captions = [caption for caption in self.metadata['caption'].values]

        self._length = len(self.image_paths)

        self.size = size
        self.center_crop = center_crop
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=(flip_p if phase == 'train' else 0))

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        example["caption"] = self.captions[i]
        image = Image.open(self.image_paths[i])

        if not image.mode == "RGB":
            image = image.convert("RGB")
            
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example