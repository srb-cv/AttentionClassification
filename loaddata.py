from PIL import Image
import os

from torch.utils.data import Dataset

class LoadData(Dataset):
    def __init__(self, header_path, footer_path, left_path, right_path, transform=None):
        self.header_path = header_path
        self.footer_path = footer_path
        self.left_path = left_path
        self.right_path = right_path

        self.image_names, self.class_name = self.get_file_names(header_path)
        self.get_class_encoding(header_path)

        self.ctoi = {}
        self.itoc = {}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        label = self.class_names[idx]

        header = os.path.join(self.header_path, label, self.image_names[idx])
        footer = os.path.join(self.footer_path, label, self.image_names[idx])
        left = os.path.join(self.left_path, label, self.image_names[idx])
        right = os.path.join(self.right_path, label, self.image_names[idx])

        header = Image.open(header)
        footer = Image.open(footer)
        left = Image.open(left)
        right = Image.open(right)

        label = self.ctoi[label]

        return header, footer, left, right, label

    def get_file_names(self, path):
        total_files, class_name = [], []
        for root, dir, files in os.walk(path):
            for f in files:
                if '.tff' in f:
                    total_files.append(f)
                    class_name.append(root.split('/')[-1])
        return total_files, class_name

    def get_class_encoding(self, path):
        class_names = []
        for root, dir, files in os.walk(path):
            for d in dir:
                class_names.append(d)

        class_names = sorted(class_names)
        self.ctoi = {c:i for i,c in enumerate(class_names)}
        self.itoc = {i:c for i,c in enumerate(class_names)}
