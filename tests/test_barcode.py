import cv2

from tests.test_utils import image_file_path as img_path
from visual_label_compare.image_util import apply_mask
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt


def test_barcode():
    # GIVEN
    label1a = cv2.imread(img_path("label1a.png"))
    # inverted mask
    mask = cv2.bitwise_not(cv2.imread(img_path("label-mask.png"), cv2.IMREAD_GRAYSCALE))
    # get barcode area
    masked1a = apply_mask(label1a, mask)

    # WHEN
    detected_barcodes = decode(masked1a)

    # THEN
    assert len(detected_barcodes) == 1
    assert str(detected_barcodes[0].data.decode("utf-8")).endswith("92055901755477000000000015")
    display_detected_barcode(masked1a, detected_barcodes[0])


def display_detected_barcode(src_image, barcode):
    (x, y, w, h) = barcode.rect

    img = src_image.copy()
    # draw the barcode bounding box
    cv2.rectangle(img, (x - 10, y - 10),
                  (x + w + 10, y + h + 10),
                  (0, 220, 0), 4)

    # setup the figure
    fig = plt.figure(figsize=(5, 4), dpi=300)
    plt.suptitle("Detected Barcode")
    plt.imshow(img)
    plt.axis("off")
    plt.show()
