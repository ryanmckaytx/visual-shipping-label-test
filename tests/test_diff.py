import cv2

from visual_label_compare.diff import diff_images, apply_mask


def test_diff_label_with_itself():
    label1a = cv2.imread("images/label1a.png")

    assert diff_images(label1a, "Label1a", label1a, "Label1a") == 1


def test_diff_very_similar_labels():
    label1a = cv2.imread("images/label1a.png")
    label1b = cv2.imread("images/label1b.png")

    ssim = diff_images(label1a, "Label1a", label1b, "Label1b")
    assert ssim >= 0.99
    assert ssim < 1


def test_diff_very_similar_labels_with_mask():
    label1a = cv2.imread("images/label1a.png")
    label1b = cv2.imread("images/label1b.png")
    mask = cv2.imread("images/label-mask.png", cv2.IMREAD_GRAYSCALE)
    masked1a = apply_mask(label1a, mask)
    masked1b = apply_mask(label1b, mask)

    assert diff_images(masked1a, "MaskedLabel1a", masked1b, "MaskedLabel1b") == 1


def test_diff_somewhat_similar_labels():
    label1a = cv2.imread("images/label1a.png")
    label2 = cv2.imread("images/label2.png")

    ssim = diff_images(label1a, "Label1a", label2, "Label2")
    assert ssim >= 0.97
    assert ssim < 0.99
