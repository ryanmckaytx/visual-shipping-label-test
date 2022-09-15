import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def image_file_path(filename):
    return f"{TEST_DIR}/images/{filename}"
