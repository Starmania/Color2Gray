from typing import Literal

import numpy as np

from .base_method import BaseMethod


class AverageMethod(BaseMethod):
    def _convert(
        self, image: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.int8]], **_
    ) -> np.ndarray:
        """Convert an image to grayscale using the average method.
        This method calculates the average of the RGB channels for each pixel
        and uses it as the grayscale value.

        Args:
            image (np.ndarray | None, optional): The input image in RGB format.\
            If None, takes the image from the instance.
        Returns:
            np.ndarray: The grayscale image.
        """
        # Calculate the average across the color channels (axis 2)
        gray_image = np.average(image, axis=2, keepdims=True).squeeze()

        return gray_image
