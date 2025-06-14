from typing import Literal

import numpy as np

from .base_method import BaseMethod

# Based on the luminosity factors of the National Television System Committee
RGB_FACTOR = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class AverageLuminanceMethod(BaseMethod):
    def _convert(
        self, image: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.int8]], **_
    ) -> np.ndarray:
        """Convert an image to grayscale using the average luminous method.
        This method calculates the average of the RGB channels for each pixel
        and uses it as the grayscale value, weighted by the luminous factors.

        Args:
            image (np.ndarray | None, optional): The input image in RGB format.
            If None, takes the image from the instance.
        Returns:
            np.ndarray: The grayscale image.
        """
        # Calculate the grayscale image using the luminous factors
        gray_image = np.sum(image * RGB_FACTOR, axis=2, keepdims=True)

        return gray_image
