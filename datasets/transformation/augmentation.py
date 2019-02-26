import cv2
import numpy as np
import random
import math
import numbers

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



class RandomResizedCrop:
    def __init__(self, target_resolution=(240, 320), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation=cv2.INTER_LINEAR):
        super().__init__()
        self.target_resolution = target_resolution
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5 and min(ratio) <= (h / w) <= max(ratio):
                w, h = h, w

            if w <= img.shape[0] and h <= img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[1] - w) // 2
        j = (img.shape[0] - w) // 2
        return i, j, w, w

    def __call__(self, image, *args):

        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped_image = image[j:j+w, i:i+h]
        image_low = cv2.resize(cropped_image, dsize=self.target_resolution, interpolation=self.interpolation)
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


class RandomRotation:
    def __init__(self, degrees=(0, 90, -90), resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) < 2 :
                raise ValueError("If degrees is a sequence, it must be of len > 2.")
            self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation. Rotation will be 0 or 90 , with given Probabilities

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.choice(degrees, p = [0.5, 0.25, 0.25])

        return angle

    def __call__(self, image, *args):

        angle = self.get_params(self.degrees)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        if rotated.ndim == 2:
            rotated = rotated[:, :, np.newaxis]
        result = [rotated]

        if args is None:
            return result
        elif len(args) == 1:
            result.append((args[0]))
        else:
            for arg in args:
                result.append(arg)

        return tuple(result)


