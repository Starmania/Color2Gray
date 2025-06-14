"""NECG (Neighborhood Effect Color to Grayscale)

Lim, W. H., & Isa, N. A. M. (2011). \
Color to grayscale conversion based on neighborhood pixels effect approach for \
digital image. In Proc. int. conf. on electrical and electronics engineering \
(pp. 157-161).

Summary:
The NECG method computes the grayscale value by:
1. Summing the intensity of each RGB channel across the image.
2. Calculating the weight of each channel as its sum divided by the total sum of all channels.
3. Producing the grayscale image as a weighted sum of the RGB channels using these weights.
"""

from typing import Literal

import numpy as np

from .base_method import BaseMethod


class NECGMethod(BaseMethod):
    def _convert(
        self, image: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.int8]], **_
    ) -> np.ndarray:
        """Convert an image to grayscale using the NECG method.

        Args:
            image (np.ndarray): The input RGB image with shape (H, W, 3)

        Returns:
            np.ndarray: The grayscale image.
        """
        # Extract RGB channels
        channel_red = image[:, :, 0]
        channel_green = image[:, :, 1]
        channel_blue = image[:, :, 2]

        # Calculate total intensity levels for each channel
        sum_red = np.sum(channel_red)
        sum_green = np.sum(channel_green)
        sum_blue = np.sum(channel_blue)

        # Calculate weights
        total = sum_red + sum_green + sum_blue
        weight_red = sum_red / total
        weight_green = sum_green / total
        weight_blue = sum_blue / total

        # Convert to grayscale using the weights
        gray_image = (
            weight_red * channel_red
            + weight_green * channel_green
            + weight_blue * channel_blue
        )

        return gray_image
