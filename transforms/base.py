import cv2
import numpy as np
from utils.bbox_utils import format_from_to, to_absolute_coords, to_percentage_coords


def base_transform(image, size):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x = x.astype(np.float32)
    return x


def jitter_transform(bbox):
    bbox = format_from_to(bbox.copy(), "x1y1x2y2", "x1y1wh")
    bbox[:2] += 0.1 * bbox[2:] * np.random.randn(2)
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


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bbox, prev_bbox=None):
        for t in self.transforms:
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
    def __init__(self, mean, context_amount):
        self.mean = mean
        self.context_amount = context_amount

    def __call__(self, image, bbox, prev_bbox=None):

        # Convert to cxcywh
        rect = format_from_to(bbox.copy(), "x1y1x2y2", "cxcywh")

        # Add context to the cropping area
        context = self.context_amount * rect[2:].sum()
        rect[2:] = np.sqrt((rect[2:] + context).prod())

        # Convert back to x1y1x2y2 format
        rect = format_from_to(rect, "cxcywh", "x1y1x2y2").astype(int)

        # Crop the image
        image = crop(image, rect, self.mean)

        # Adjust bounding box coordinates
        bbox = adjust_bbox(bbox, rect)

        if prev_bbox is not None:
            prev_bbox = adjust_bbox(prev_bbox, rect)

        return image, bbox, prev_bbox


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, bbox, prev_bbox=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, bbox, prev_bbox


class Transform(object):
    def __init__(self, context_amount=0.5, size=300, mean=(104, 117, 123)):

        self.transform = Compose([
            ToAbsoluteCoords(),
            Crop(mean, context_amount),
            ToPercentCoords(),
            Resize(size),
        ])

    def __call__(self, image, bbox, prev_bbox=None):
        image, bbox, prev_bbox = self.transform(image, bbox, prev_bbox)

        if prev_bbox is not None:
            prev_bbox = jitter_transform(prev_bbox)

        return image, bbox, prev_bbox


class BaseTransform(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, bbox, prev_bbox=None):
        image = base_transform(image, self.size)

        if prev_bbox is not None:
            prev_bbox = jitter_transform(prev_bbox)

        return image, bbox, prev_bbox
