# pylint: disable=redefined-outer-name, unused-argument
import shutil
from typing import Any
from pathlib import Path

import pytest

from color2gray.methods import all_methods, BaseMethod
from color2gray.tools import load_image, save_image

# Define the test directory and output directory
TEST_DIR = Path(__file__).parent
INPUT_IMAGES_DIR = TEST_DIR / "input_images"
OUTPUT_DIR = TEST_DIR / "output_images"
PARAMS: dict[str, list[dict[str, Any]]] = {}


@pytest.fixture
def get_images() -> list[Path]:
    """Setup the output directory before running tests."""

    assert (
        INPUT_IMAGES_DIR.exists()
    ), f"Input images directory {INPUT_IMAGES_DIR} does not exist."

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    images = []

    for ext in ["*.jpg", "*.png"]:
        for image_file in INPUT_IMAGES_DIR.glob(ext):
            images.append(image_file.absolute())

    return images


def test_all_methods(get_images):
    """Test all methods with all images."""
    for method_name, Method in all_methods.items():
        for i, param in enumerate(PARAMS.get(method_name, [{}])):
            method = Method(**param)
            method_output_dir = OUTPUT_DIR / method_name / f"test_{i}"
            method_output_dir.mkdir(parents=True, exist_ok=True)
            _test_method(method, get_images, method_output_dir)


def _test_method(method: BaseMethod, images: list[Path], output_dir: Path):
    """Test a specific method with all images."""

    for image_path in images:
        image = load_image(image_path)
        gray_image = method(image)

        # plt.imshow(image)
        # plt.imshow(gray_image, cmap="gray")
        # plt.axis("off")
        # plt.show()

        output_path = output_dir / image_path.name
        save_image(gray_image, output_path)
        assert output_path.exists(), f"Output image {output_path} was not created."
        # print(
        #     f"Processed {image_path} with {method.__class__.__name__} and saved to {output_path}"
        # )
