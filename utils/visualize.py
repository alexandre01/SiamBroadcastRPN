import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .bbox_utils import to_absolute_coords, format_from_to
from itertools import cycle
import io
from PIL import Image
from torchvision import transforms


def draw_rectangle(bbox, color="r"):
    """
    bbox: x1y1x2y2 format.
    """

    x1, y1, w, h = format_from_to(bbox, "x1y1x2y2", "x1y1wh")
    rectangle = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, fill=False)
    plt.gca().add_patch(rectangle)


def plot_sample(img, bbox, title=None, gt_box=None, anchor=None, prev_bbox=None, anchor_id=None):
    """
    img: (Tensor)
    bbox: x1y1x2y2 format, percentage coordinates.
    """

    plt.imshow(img.permute(1, 2, 0))
    draw_rectangle(to_absolute_coords(bbox, img.shape))

    if gt_box is not None:
        draw_rectangle(to_absolute_coords(gt_box, img.shape), "b")

    if anchor is not None:
        draw_rectangle(to_absolute_coords(anchor, img.shape), "y")

    if prev_bbox is not None:
        draw_rectangle(to_absolute_coords(prev_bbox, img.shape), "g")

    if anchor_id is not None:
        plt.gca().text(0.95, 0.95, "anchor_id: {}".format(anchor_id), transform = plt.gca().transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

    if title:
        plt.gca().set_title(title)

    plt.axis("off")


def plot_pair(exemplar, search, title=None, gt_box=None, prev_bbox=None, anchor=None, anchor_id=None):
    """Plots a pair of samples (exemplar/search)."""

    plt.tight_layout()

    if title:
        plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plot_sample(*exemplar, title="Exemplar")
    plt.subplot(1, 2, 2)
    plot_sample(*search, title="Search", gt_box=gt_box, prev_bbox=prev_bbox, anchor=anchor, anchor_id=anchor_id)


def plot_bboxes(anchors, format="x1y1wh", title=None, random_color=True):
    """
    Plots a list of bounding boxes.
    """
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.gca().set_aspect('equal', adjustable='box')

    cycol = cycle('bgrcmk')

    n = len(anchors)
    for i in range(n):
        color = next(cycol) if random_color else "r"
        draw_rectangle(format_from_to(anchors[i], format, "x1y1x2y2"), color=color)

    if title:
        plt.gca().set_title(title)


def plot_to_tensor():
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image).unsqueeze(0)
    buf.close()

    return image
