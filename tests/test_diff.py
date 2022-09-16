import math

import cv2

from tests.test_utils import image_file_path as img_path
from visual_label_compare.diff import diff_images
from visual_label_compare.image_util import apply_mask


def test_diff_label_with_itself():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))

    # WHEN
    mse, ssim = diff_images(label1a, "Label1a", label1a, "Label1a")\

    # THEN
    assert mse == 0


def test_diff_very_similar_labels():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))
    label1b = cv2.imread(img_path("label1b.png"))

    # WHEN
    mse, ssim = diff_images(label1a, "Label1a", label1b, "Label1b")

    # THEN
    assert math.isclose(mse, 140, abs_tol=1)


def test_diff_very_similar_labels_with_mask():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))
    label1b = cv2.imread(img_path("label1b.png"))
    mask = cv2.imread(img_path("label-mask.png"), cv2.IMREAD_GRAYSCALE)
    masked1a = apply_mask(label1a, mask)
    masked1b = apply_mask(label1b, mask)

    # WHEN
    mse, ssim = diff_images(masked1a, "MaskedLabel1a", masked1b, "MaskedLabel1b")

    # THEN
    assert mse == 0


def test_diff_somewhat_similar_labels():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))
    label2 = cv2.imread(img_path("label2.png"))

    # WHEN
    mse, ssim = diff_images(label1a, "Label1a", label2, "Label2")

    # THEN
    assert math.isclose(mse, 601, abs_tol=1)


def test_diff_somewhat_similar_labels_with_mask():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))
    label2 = cv2.imread(img_path("label2.png"))
    mask = cv2.imread(img_path("label-mask.png"), cv2.IMREAD_GRAYSCALE)
    masked1a = apply_mask(label1a, mask)
    masked2 = apply_mask(label2, mask)

    # WHEN
    mse, ssim = diff_images(masked1a, "MaskedLabel1a", masked2, "MaskedLabel2")

    # THEN
    assert math.isclose(mse, 461, abs_tol=1)
