import numpy as np
import torch


class ToTensor:
    def __call__(self, *sample):
        out_sample = []
        for element in sample:
            if isinstance(element, np.ndarray):
                tensor = torch.from_numpy(element)
                out_sample.append(tensor)
            elif isinstance(element, int):
                element = np.array([element], dtype=np.long)
                tensor = torch.from_numpy(element)
                out_sample.append(tensor)
            else:
                raise TypeError("Cannot understand datatype", type(element))
        return out_sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToFloat:
    def __call__(self, *sample):
        out_sample = []
        for element in sample:
            if isinstance(element, np.ndarray) and element.ndim == 3 and element.dtype == np.uint8:
                element = element.astype(np.float32) / 255
            out_sample.append(element)
        return out_sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToInteger:
    def __call__(self, *sample):
        out_sample = []
        for element in sample:
            if isinstance(element, np.ndarray) and element.ndim == 3 and element.dtype == np.float32:
                element = (element * 255).astype(np.uint8)
            out_sample.append(element)
        return out_sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TransposeImage:
    def __call__(self, *sample):
        out_sample = []
        for element in sample:
            if isinstance(element, np.ndarray) and element.ndim == 3:
                element = element.transpose((2, 0, 1))
            out_sample.append(element)
        return out_sample

    def __repr__(self):
        return self.__class__.__name__ + '()'
