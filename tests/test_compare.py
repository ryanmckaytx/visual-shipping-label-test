import cv2

from visual_label_compare.compare import compare_images


def test_compare_label_to_itself():
    label1 = cv2.imread("images/label1.png")

    # convert the image to grayscale
    label1 = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)

    assert compare_images(label1, label1, "Label1 vs Label1") == 1


def test_compare_similar_labels():
    label1 = cv2.imread("images/label1.png")
    label2 = cv2.imread("images/label2.png")

    # convert the image to grayscale
    label1 = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
    label2 = cv2.cvtColor(label2, cv2.COLOR_BGR2GRAY)

    assert compare_images(label1, label2, "Label1 vs Label2") > 0.97
