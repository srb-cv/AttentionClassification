from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import glob


class Tobacco(Dataset):
    def __init__(self, root, num_train=100, train_val_ratio=0.8, num_splits=10, channels=1, preprocess=None,
                 random_state=1337):
        assert (num_train in range(10, 101, 10))
        assert (channels in [1, 3])
        self.root = root
        self.num_train = num_train
        self.train_val_ratio = train_val_ratio
        self.num_splits = num_splits
        self.channels = channels
        self.preprocess = preprocess
        self.random_state = random_state
        self.splits = self._create_splits()
        self.current_index = 0
        self.current_mode = "train"
        self.samples = []
        self.load_split(self.current_mode, self.current_index)

    def __getitem__(self, index):
        sample = self.samples[index]
        image, gt = self._load_sample(*sample)
        if self.preprocess is not None:
            for processor in self.preprocess:
                image, gt = processor(image, gt)
        return image, gt

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Number of splits: {}\n'.format(len(self.splits))
        fmt_str += '    Current split: {}\n'.format(self.current_index)
        fmt_str += '    Current mode: {}\n'.format(self.current_mode)
        fmt_str += '    Number of training images in current split: {}\n'.format(
            len(self.splits[self.current_index]["train"]))
        fmt_str += '    Number of validation images in current split: {}\n'.format(
            len(self.splits[self.current_index]["val"]))
        fmt_str += '    Number of test images in current split: {}\n'.format(
            len(self.splits[self.current_index]["test"]))
        return fmt_str

    def _create_splits(self):
        splits = []
        for i in range(self.num_splits):
            classes = sorted(
                [class_ for class_ in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, class_))])
            split = {"train": [], "val": [], "test": []}
            for j in range(len(classes)):
                samples = glob.glob(os.path.join(self.root, classes[j], "*.tif"))
                samples = sorted(samples)
                np.random.seed(self.random_state + i)
                np.random.shuffle(samples)
                for sample in samples[:int(self.num_train * self.train_val_ratio)]:
                    split["train"].append((sample, j))
                for sample in samples[int(self.num_train * self.train_val_ratio):self.num_train]:
                    split["val"].append((sample, j))
                for sample in samples[self.num_train:]:
                    split["test"].append((sample, j))
            np.random.seed(self.random_state + i)
            np.random.shuffle(split["train"])
            np.random.shuffle(split["val"])
            np.random.shuffle(split["test"])
            splits.append(split)
        np.random.seed()
        return splits

    def load_split(self, mode=None, index=None):
        self.current_mode = self.current_mode if mode is None else mode
        self.current_index = self.current_index if index is None else index
        self.samples = self.splits[self.current_index][self.current_mode]

    def _load_sample(self, path, gt):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = image[:, :, np.newaxis]
        image = np.tile(image, self.channels)
        return image, gt
