import torch
import cv2
from torch.utils.data import Dataset
import os
from utils import IoU
from transforms.transforms import *
from pycocotools.coco import COCO


class CocoDetection(Dataset):
    def __init__(self, root, annFile):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())

    def getFromCategory(self, category_id):
        imgIds = self.coco.getImgIds(catIds=category_id);
        img_id = imgIds[np.random.randint(len(imgIds))]

        return self.getFromImageId(img_id, category_id=category_id)

    def getFromImageId(self, img_id, category_id=None):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_file = os.path.join(self.root, img_name)

        if category_id is not None:
            anns = list(filter(lambda x: x["category_id"] == category_id, anns))

        # Deal with the case where no annotation exists in the image
        if len(anns) == 0:
            return self.getRandom()

        anno = anns[np.random.randint(len(anns))]

        # Deal with the case of too small objects
        if anno["area"] < 500:
            return self.getRandom()

        catgory_id = anno["category_id"]
        bbox = np.array(anno["bbox"])
        mask = self.coco.annToMask(anno)

        # Convert bounding box to x1y1x2y2 format.
        bbox = format_from_to(bbox, "x1y1wh", "x1y1x2y2")

        return img_file, bbox, mask, catgory_id

    def __getitem__(self, index):
        img_id = self.ids[index]

        return self.getFromImageId(img_id)

    def getRandom(self):
        return self[np.random.randint(len(self))]

    def __len__(self):
        return len(self.ids)


class COCODistractor(Dataset):
    def __init__(self, cocodataset, size):
        self.dataset = cocodataset
        self.size = size
        self.transform = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            Crop(center_at_pred=True, context_amount=1.5),
            ToPercentCoords(),
            Resize(300),
        ])

    def __getitem__(self, index):
        img_file, bbox, _, catgory_id = self.dataset.getRandom()

        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        bbox = to_percentage_coords(bbox, img.shape)
        img_z, bbox_z, _ = self.transform(img, bbox)

        bbox_xprev = jitter_transform(bbox)
        img_x, bbox_x, bbox_xprev = self.transform(img, bbox, bbox_xprev)

        H, W, _ = img_x.shape
        abs_bbox = to_absolute_coords(bbox_x, img_x.shape)
        w, h = abs_bbox[2:] - abs_bbox[:2]

        # Import a distractor (an instance from the same category)
        img_file2, bbox2, mask2, _ = self.dataset.getFromCategory(catgory_id)
        img2 = cv2.imread(img_file2, cv2.IMREAD_COLOR)
        bbox2 = bbox2.astype(int)
        w2, h2 = bbox2[2:] - bbox2[:2]

        cropped_img = img2[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]
        cropped_mask = mask2[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]

        # Scale the distractor image so that it has the same size as the first one.
        ratio = np.sqrt(w * h) / np.sqrt(w2 * h2)

        if np.isinf(ratio) or np.isnan(ratio):
            return self[np.random.randint(len(self))]

        w2, h2 = min(int(w2 * ratio), W - 1), min(int(h2 * ratio), H - 1)
        cropped_img = cv2.resize(cropped_img, (w2, h2))
        cropped_mask = cv2.resize(cropped_mask, (w2, h2)).astype(np.bool)

        # max trails (10)
        for _ in range(10):
            x = np.random.randint(W - w2)
            y = np.random.randint(H - h2)
            bbox2 = to_percentage_coords(np.array([x, y, x + w2, y + h2]), img_x.shape)

            # Avoid too difficult cases where the distractor completely occludes the main instance
            if IoU(bbox_x, bbox2) < 0.30:
                break

        img_x[y:y + h2, x:x + w2][cropped_mask] = cropped_img[cropped_mask]

        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB) / 255.
        img_z = torch.from_numpy(img_z).permute(2, 0, 1).float()
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) / 255.
        img_x = torch.from_numpy(img_x).permute(2, 0, 1).float()

        bbox_z, bbox_x = torch.from_numpy(bbox_z).float(), torch.from_numpy(bbox_x).float()
        bbox_xprev = torch.from_numpy(bbox_xprev).float()

        return img_z, img_x, bbox_z, bbox_x, bbox_xprev

    def __len__(self):
        return self.size


class COCONegativePair(Dataset):
    def __init__(self, cocodataset, size, cfg, transform=None):
        self.dataset = cocodataset
        self.size = size
        self.cfg = cfg

        self.transform = transform
        if transform is not None and not isinstance(transform, list):
            self.transform = [transform, transform]

    def __getitem__(self, index):
        img_file_z, bbox_z, _, category_id = self.dataset.getRandom()
        img_file_x, bbox_x, _, _ = self.dataset.getFromCategory(category_id)

        img_z = cv2.imread(img_file_z, cv2.IMREAD_COLOR)
        img_x = cv2.imread(img_file_x, cv2.IMREAD_COLOR)

        bbox_xprev = bbox_x

        # Convert to percentage coordinates.
        bbox_z = to_percentage_coords(bbox_z, img_z.shape)
        bbox_x = to_percentage_coords(bbox_x, img_x.shape)
        bbox_xprev = to_percentage_coords(bbox_xprev, img_x.shape)

        if self.transform is not None:
            img_z, bbox_z, _ = self.transform[0](img_z, bbox_z)
            img_x, bbox_x, bbox_xprev = self.transform[1](img_x, bbox_x, bbox_xprev)

        # Convert to RBG image, and scale values to [0, 1].
        img_z = self.cfg.MODEL.INPUT_RANGE * cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB) / 255.
        img_x = self.cfg.MODEL.INPUT_RANGE * cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) / 255.

        # Convert to PyTorch Tensors (in particular for images, (w, h, c) is transformed to (c, w, h)).
        img_z = torch.from_numpy(img_z).permute(2, 0, 1).float()
        img_x = torch.from_numpy(img_x).permute(2, 0, 1).float()

        bbox_z = torch.from_numpy(bbox_z).float()

        # The search image doesn't contain the exemplar, there is no groundtruth bounding box.
        bbox_x = torch.zeros(4)
        bbox_xprev = torch.zeros(4)

        return img_z, img_x, bbox_z, bbox_x, bbox_xprev

    def __len__(self):
        return self.size


class COCOPositivePair(Dataset):
    def __init__(self, cocodataset, size, cfg, transform=None):
        self.dataset = cocodataset
        self.size = size
        self.cfg = cfg

        self.transform = transform
        if transform is not None and not isinstance(transform, list):
            self.transform = [transform, transform]

    def __getitem__(self, index):
        img_file, bbox, _, _ = self.dataset.getRandom()

        img = cv2.imread(img_file, cv2.IMREAD_COLOR)

        # Convert to percentage coordinates.
        bbox = to_percentage_coords(bbox, img.shape)

        if self.transform is not None:
            img_z, bbox_z, _ = self.transform[0](img, bbox)
            img_x, bbox_x, bbox_xprev = self.transform[1](img, bbox, bbox)

        # Convert to RBG image, and scale values to [0, 1].
        img_z = self.cfg.MODEL.INPUT_RANGE * cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB) / 255.
        img_x = self.cfg.MODEL.INPUT_RANGE * cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB) / 255.

        # Convert to PyTorch Tensors (in particular for images, (w, h, c) is transformed to (c, w, h)).
        img_z = torch.from_numpy(img_z).permute(2, 0, 1).float()
        img_x = torch.from_numpy(img_x).permute(2, 0, 1).float()

        bbox_z = torch.from_numpy(bbox_z).float()
        bbox_x = torch.from_numpy(bbox_x).float()
        bbox_xprev = torch.from_numpy(bbox_xprev).float()

        return img_z, img_x, bbox_z, bbox_x, bbox_xprev

    def __len__(self):
        return self.size
