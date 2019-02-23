import cv2
import numpy as np


class DownScale:
    def __init__(self, target_resolution=(240, 320)):
        super().__init__()
        self.target_resolution = target_resolution

    def __call__(self, image, *args):
        image_low = cv2.resize(image, dsize=self.target_resolution, interpolation=cv2.INTER_NEAREST)
        if image_low.ndim == 2:
            image_low = image_low[:, :, np.newaxis]
        result = [image_low]

        if args is None:
            return result
        elif len(args) == 1:
            result.append((args[0]))
        else:
            for arg in args:
                result.append(arg)

        return tuple(result)
