import cv2

from visual_label_compare.diff import diff_images


def test_diff_label_with_itself():
    label1 = cv2.imread("images/label1.png")

    assert diff_images(label1, label1, "Label1 vs Label1") == 1


def test_diff_similar_labels():
    label1 = cv2.imread("images/label1.png")
    label2 = cv2.imread("images/label2.png")

    assert diff_images(label1, label2, "Label1 vs Label2") > 0.97
