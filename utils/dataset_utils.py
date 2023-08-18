import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from imagecorruptions import corrupt
import constants as constants


def load_gray_image(img_path, plot_fig=False):
    im_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if plot_fig:
        plt.gca()
        plt.imshow(im_gray)
    return im_gray


def load_color_image(img_path, plot_fig=False) -> np.ndarray:
    if isinstance(img_path, Path):
        img_path = str(img_path.absolute())
    im_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)
    if plot_fig:
        plt.gca()
        plt.imshow(im_color)
    return im_color


def save_color_image(img_path, img):
    """Saves the color image

    Args:
        img_path (_type_): _description_
        img (_type_): _description_
    """
    img_path = Path(img_path)
    if not img_path.parent.exists():
        img_path.parent.mkdir(parents=True)
    img_path = str(img_path.absolute())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename=img_path, img=img)


def illuminate(image, gamma=1.0):
    """Apply gamma correction to an image."""

    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def blur_image(image, blur_kernel):
    """
    Blurs the image.
    arg1 : image
    arg2 : blurriness. Higher the value, more the blurriness.
    returns : blurred image.
    """
    return cv2.blur(image, (blur_kernel, blur_kernel), cv2.BORDER_DEFAULT)


def corrupt_image(
    *,
    corruption_name=None,
    image=None,
    image_file=None,
    severity=5,
    save_fig=False,
    plot_fig=False,
    save_fig_path=None
):
    """This will corrupt the image.

    Args:
        corruption_name (_type_): _description_
        image (_type_, optional): _description_. Defaults to None.
        image_file (_type_, optional): _description_. Defaults to None.
        severity (int, optional): _description_. Defaults to 5.
    """
    assert (image is not None) or (
        image_file is not None
    ), "Either image or image_file must be provided"

    if corruption_name is None:
        corruption_name = np.random.choice(constants.CORRUPTIONS)

    if save_fig is True:
        assert save_fig_path is not None, "save_fig_path must be provided"

    if image is None:
        image = load_color_image(image_file)

    corrupted_image = corrupt(
        image=image, corruption_name=corruption_name, severity=severity
    )

    if save_fig:
        save_color_image(img_path=save_fig_path, img=corrupted_image)
    if plot_fig:
        plt.gca()
        plt.imshow(corrupted_image)

    return corrupted_image
