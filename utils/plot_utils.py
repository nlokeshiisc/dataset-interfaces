import cv2
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def load_color_img(img_path, plot_fig=False):
    assert img_path is not None, "Cannot load none image"
    image = cv2.imread(f"{img_path}", cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if plot_fig == True:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    return image


def draw_bounding_box(img, box, color):
    x_min, y_min, x_max, y_max = box
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 3)


def annotate_image_preds(
    img_path=None, image=None, gt_boxes=None, pred_boxes=None, plot_fig=True, text=None
):
    """Annotates image with ground truth and predicted boxes
    This API disrupts the passed image. So, if you want to use the original image, pass a copy of it.

    Args:
        img_path:
        gt_boxes:
        pred_boxes:
        plot_image (bool, optional): . Defaults to True.
        text : . Defaults to None.

    Returns:
        _type_:
    """
    if image is None:
        image = load_color_img(img_path, plot_fig=False)

    if gt_boxes is not None:
        for gt_box in gt_boxes:
            color = (0, 255, 0)
            draw_bounding_box(image, list(gt_box), color)

    if pred_boxes is not None:
        for pred_box in pred_boxes:
            color = (255, 0, 0)
            draw_bounding_box(image, list(pred_box), color)

    if text is not None:
        image = cv2.putText(
            image,
            text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
            cv2.LINE_AA,
        )

    if plot_fig == True:
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    return image


def plot_hist(data, title="title", num_bins=10, plot_fig=True):
    """
    Plots the histogram of the passed array
    """

    density, bins, _ = plt.hist(data, density=True, bins=num_bins)
    count, _ = np.histogram(data, bins)
    for x, y, num in zip(bins, density, count):
        if num != 0:
            plt.text(x, y + 0.05, num, fontsize=10, rotation=-90)  # x,y,str
    plt.title(title)
    if plot_fig == True:
        plt.show()


def view_image_grid(
    *,
    image_files=None,
    images=None,
    gt_boxes=None,
    pred_boxes=None,
    title="",
    image_titles=None,
    nrows=None,
    ncols=None,
    plot_fig=False,
    save_fig=False,
):
    """_summary_
    pass either a list of image files or a list of images. But not both.
    """
    assert not (
        image_files is None and images is None
    ), "pass either the images or the image files"

    plt.clf()
    if images is None:
        images = [load_color_img(file, plot_fig=False) for file in image_files]

    if nrows is None and ncols is None:
        nrows = 4
        ncols = len(images) // nrows + 1
    elif nrows is not None and ncols is None:
        ncols = len(images) // nrows + 1
    elif ncols is not None and nrows is None:
        nrows = len(images) // ncols + 1

    if gt_boxes is not None and pred_boxes is not None:
        images = [
            annotate_image_preds(image=i, gt_boxes=g, pred_boxes=p, plot_fig=False)
            for i, g, p in zip(images, gt_boxes, pred_boxes)
        ]
    elif gt_boxes is not None:
        images = [
            annotate_image_preds(image=i, gt_boxes=g, plot_fig=False)
            for i, g in zip(images, gt_boxes)
        ]
    elif pred_boxes is not None:
        images = [
            annotate_image_preds(image=i, pred_boxes=p, plot_fig=False)
            for i, p in zip(images, pred_boxes)
        ]

    images = [cv2.resize(image, (400, 300)) for image in images]

    fig = plt.figure()
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,
    )
    for _, (ax, im) in enumerate(zip(grid, images)):
        ax.imshow(im)
        ax.axis("off")
        if image_titles is not None:
            ax.set_title(image_titles[_])

    print(title)
    plt.title(title)
    if save_fig == True:
        plt.savefig(f"temp.png", bbox_inches="tight", dpi=300)
    if plot_fig == True:
        plt.show()
