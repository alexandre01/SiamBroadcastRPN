import cv2
import numpy as np
from utils.bbox_utils import format_from_to, to_absolute_coords, to_percentage_coords


MEAN = (104, 117, 123)


def base_transform(image, size):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x = x.astype(np.float32)
    return x


def jitter_transform(bbox):
    bbox = format_from_to(bbox.copy(), "x1y1x2y2", "x1y1wh")
    bbox[:2] += 0.25 * bbox[2:] * np.random.randn(2)
    bbox[2:] += 0.25 * bbox[2:] * np.random.randn(2)

    return format_from_to(bbox, "x1y1wh", "x1y1x2y2")


def get_image_size(image):
    """
    Utility that outputs the size (w, h) of a OpenCV2 image.
    """
    return tuple(image.shape[1::-1])


def crop(image, rect, fill):
    """
    rect: in absolute coordinates, x1y1wh format.
    """

    pads = np.concatenate((-rect[:2], rect[2:] - get_image_size(image)))
    padding = max(0, int(pads.max()))

    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=fill)
    rect = padding + rect
    image = image[rect[1]:rect[3], rect[0]:rect[2]]

    return image


def adjust_bbox(bbox, rect):
    bbox = bbox.copy()

    bbox[:2] = np.maximum(bbox[:2], rect[:2])
    bbox[:2] -= rect[:2]
    bbox[2:] = np.minimum(bbox[2:], rect[2:])
    bbox[2:] -= rect[:2]

    return bbox


class ConvertFromInts(object):
    def __call__(self, image, bbox, prev_bbox=None):
        return image.astype(np.float32), bbox, prev_bbox


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox, prev_bbox=None):
        for t in self.transforms:
            if t is not False:
                image, bbox, prev_bbox = t(image, bbox, prev_bbox)
        return image, bbox, prev_bbox


class ToAbsoluteCoords(object):
    def __call__(self, image, bbox, prev_bbox=None):
        bbox = to_absolute_coords(bbox.copy(), image.shape)

        if prev_bbox is not None:
            prev_bbox = to_absolute_coords(prev_bbox.copy(), image.shape)

        return image, bbox, prev_bbox


class ToPercentCoords(object):
    def __call__(self, image, bbox, prev_bbox=None):

        bbox = to_percentage_coords(bbox.copy(), image.shape)

        if prev_bbox is not None:
            prev_bbox = to_percentage_coords(prev_bbox.copy(), image.shape)

        return image, bbox, prev_bbox


class Crop(object):
    def __init__(self, mean=MEAN, context_amount=0.5,
                 random_translate=False, random_translate_range=0.3,
                 random_resize=False, random_resize_scale_min=0.35, random_resize_scale_max=1.5,
                 return_rect=False, center_at_pred=False, make_square=False):
        self.mean = mean
        self.context_amount = context_amount

        self.random_translate = random_translate
        self.random_translate_range = random_translate_range

        self.random_resize = random_resize
        self.random_resize_scale_min = random_resize_scale_min
        self.random_resize_scale_max = random_resize_scale_max

        self.return_rect = return_rect
        self.center_at_pred = center_at_pred

        self.make_square = make_square

    def __call__(self, image, bbox, prev_bbox=None):

        # If prev_bbox is provided, use this rectangle as cropping area
        if self.center_at_pred and prev_bbox is not None:
            rect = prev_bbox.copy()
        else:
            rect = bbox.copy()

        # Convert to cxcywh
        rect = format_from_to(rect, "x1y1x2y2", "cxcywh")

        # Add context to the cropping area
        if not self.make_square:
            context = self.context_amount * rect[2:].sum()
            rect[2:] = np.sqrt((rect[2:] + context).prod())
        else:
            rect[2:] += 2 * self.context_amount * rect[2:]

        if self.random_resize:
            rect[2:] *= np.random.uniform(self.random_resize_scale_min, self.random_resize_scale_max)

        if self.random_translate:
            displacement = np.random.uniform(-1, 1, 2) * self.random_translate_range * rect[2:]
            rect[:2] -= displacement

        # Convert back to x1y1x2y2 format
        rect = format_from_to(rect, "cxcywh", "x1y1x2y2").astype(int)

        # Crop the image
        image = crop(image, rect, self.mean)

        # Adjust bounding box coordinates
        bbox = adjust_bbox(bbox, rect)

        if prev_bbox is not None:
            prev_bbox = adjust_bbox(prev_bbox, rect)

        if not self.return_rect:
            return image, bbox, prev_bbox
        else:
            return image, bbox, prev_bbox, rect


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox, prev_bbox=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, bbox, prev_bbox


class RandomMirror(object):
    def __call__(self, image, bbox, prev_bbox=None):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]

            bbox[0::2] = width - bbox.copy()[2::-2]

            if prev_bbox is not None:
                prev_bbox[0::2] = width - prev_bbox.copy()[2::-2]

        return image, bbox, prev_bbox


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, bbox, prev_bbox=None):
        image = image.copy()
        image, bbox, prev_bbox = self.rand_brightness(image, bbox, prev_bbox)

        """
        # Do not distort hue and saturation
        
        if np.random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        image, bbox, prev_bbox = distort(image, bbox, prev_bbox)
        """

        return image, bbox, prev_bbox


class RandomSaturation(object):
    def __init__(self, lower=0.8, upper=1.2):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, bbox, prev_bbox=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        return image, bbox, prev_bbox


class RandomHue(object):
    def __init__(self, delta=15.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, bbox, prev_bbox=None):
        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, bbox, prev_bbox


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, bbox, prev_bbox=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, bbox, prev_bbox


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, bbox, prev_bbox=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, bbox, prev_bbox


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, bbox, prev_bbox=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
        return image, bbox, prev_bbox


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, bbox, prev_bbox=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
        return image, bbox, prev_bbox


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """

        image = image[:, :, self.swaps]
        return image


class MotionBlur(object):
    def __init__(self):
        """
        Add motion blur to every second image.
        """
        kernel_size = 9
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))

        if np.random.randint(2):
            # Horizontal blur
            kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        else:
            # Vertical blur
            kernel_motion_blur[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)

        self.kernel_motion_blur = kernel_motion_blur / kernel_size

    def __call__(self, image, bbox, prev_bbox=None):
        if np.random.randint(2):
            image = cv2.filter2D(image, -1, self.kernel_motion_blur)

        return image, bbox, prev_bbox


class Transform(object):
    def __init__(self, context_amount=0.5,
                 random_translate=False, random_translate_range=0.3,
                 random_resize=False, random_resize_scale_min=0.35, random_resize_scale_max=1.5,
                 size=300, mean=MEAN,
                 motion_blur=False, make_square=False):

        self.transform = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Crop(mean=mean, context_amount=context_amount,
                 random_translate=random_translate, random_translate_range=random_translate_range,
                 random_resize=random_resize, random_resize_scale_min=random_resize_scale_min,
                 random_resize_scale_max=random_resize_scale_max, make_square=make_square),
            ToPercentCoords(),
            motion_blur and MotionBlur(),
            Resize(size),
        ])

    def __call__(self, image, bbox, prev_bbox=None):
        image, bbox, prev_bbox = self.transform(image, bbox, prev_bbox)

        if prev_bbox is not None:
            prev_bbox = jitter_transform(prev_bbox)

        return image, bbox, prev_bbox
