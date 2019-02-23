from torch.utils.data import Dataset
import os
import numpy as np
import cv2


class CDIP(Dataset):
    def __init__(self, root, mode='train', channels=1, include_ocr=False, exclude_tobacco=False, preprocess=None,
                 random_state=1337):
        assert (mode in ['train', 'val', 'test'])
        assert (channels in [1, 3])
        self.root = root
        self.mode = mode
        self.channels = channels
        self.include_ocr = include_ocr
        self.exclude_tobacco = exclude_tobacco
        self.preprocess = preprocess
        self.random_state = random_state
        self.duplicates = open(os.path.join(root, "duplicates.txt")).read().split()
        self.samples = self._find_samples()

    def __getitem__(self, index):
        sample, gt = self.samples[index].split()
        gt = int(gt)
        image = self._load_image(sample)
        if self.include_ocr:
            ocr = self._load_ocr(sample)
            if self.preprocess is not None:
                for processor in self.preprocess:
                    image, gt, ocr = processor(image, gt, ocr)
            return image, gt, ocr
        else:
            if self.preprocess is not None:
                for processor in self.preprocess:
                    image, gt = processor(image, gt)
            return image, gt

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Dataset mode: {}\n'.format(self.mode)
        fmt_str += '    Number of images: {}\n'.format(self.__len__())
        return fmt_str

    def _find_samples(self):
        samples = os.path.join(self.root, "labels", self.mode + ".txt")
        samples = open(samples).readlines()
        if self.exclude_tobacco:
            for line in samples[:]:
                if os.path.basename(line.split()[0]) in self.duplicates:
                    samples.remove(line)
        np.random.seed(self.random_state)
        np.random.shuffle(samples)
        np.random.seed()
        return samples

    def _load_image(self, path):
        image = cv2.imread(os.path.join(self.root, "images", path), cv2.IMREAD_GRAYSCALE)
        image = image[:, :, np.newaxis]
        image = np.tile(image, self.channels)
        return image

    def _load_ocr(self, path):
        ocr = open(os.path.join(self.root, "ocr", path[:-4] + ".txt")).read()
        return ocr
