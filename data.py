from PIL import Image
import os

from torch.utils.data import Dataset


class LoadData(Dataset):
    def __init__(self, dataset_path, phase, transform = None):
        if phase == 'train':
            label_path = os.path.join(dataset_path, 'labels', 'train.txt')
        elif phase == 'test':
            label_path = os.path.join(dataset_path, 'labels', 'val.txt')
        self.label_path = label_path

        self.data =  {}
        with open(label_path, 'r') as f:
            for x in f:
                x = x.rstrip()
                x = x.split()
                im_path = os.path.join(dataset_path, 'images', x[0])
                self.data[im_path] = int(x[1])

        self.image_names = list(self.data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        im_path = self.image_names[idx]
        label = self.data[im_path]

        im = Image.open(im_path)
        im = im.convert(mode='RGB')
        if self.transform:
            im = self.transform(im)

        return im, label
