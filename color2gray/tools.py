from pathlib import Path

import numpy as np
from PIL import Image


def convert_image(image_path: Path | str) -> np.ndarray:
    """Convert an image to grayscale using the specified method.

    Args:
        image_path (Path | str): The path to the input image.
        method (str): The conversion method to use ('average' or 'average_luminous').

    Returns:
        np.ndarray: The converted grayscale image.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    np_image = np.array(image, dtype=np.uint8)

    return np_image

def save_image(image: np.ndarray, output_path: Path | str) -> None:
    """Save a grayscale image to the specified path.

    Args:
        image (np.ndarray): The grayscale image to save.
        output_path (Path | str): The path where the image will be saved.
    """
    # Convert the image to uint8
    image = image.astype(np.uint8)
    # Create a PIL Image from the numpy array
    pil_image = Image.fromarray(image.squeeze(), mode='L')
    # Save the image
    pil_image.save(output_path)
