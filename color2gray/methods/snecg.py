"""SNECG (Stretched Neighborhood Effect Color to Grayscale)

Lim, W. H., & Isa, N. A. M. (2011). \
Color to grayscale conversion based on neighborhood pixels effect approach for \
digital image. In Proc. int. conf. on electrical and electronics engineering \
(pp. 157-161).

Summary:
The SNECG method computes the grayscale value by:
1. Summing the intensity of each RGB channel across the image.
2. Calculating the weight of each channel as its sum divided by the total sum of all channels.
3. Finding the minimum and maximum values for each channel.
4. Stretched the values of each channel to the range [0, 255].
5. Producing the grayscale image as a weighted sum of the stretched RGB channels using these weights.


It differ from NECG by stretching the RGB values to the range [0, 255] \
before combining them, which helps to enhance the contrast in the grayscale image.
"""

from typing import Literal

import numpy as np

from .base_method import BaseMethod


class SNECGMethod(BaseMethod):
    def _convert(
        self, image: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.int8]], **_
    ) -> np.ndarray:
        """
        Convert the input RGB image to grayscale using the \
            SNECG (Stretched Neighborhood Effect Color to Grayscale) method.

        Args:
            image (np.ndarray): The input RGB image with shape (H, W, 3)

        Returns:
            np.ndarray: Grayscale image with shape (H, W)
        """
        # Split the image into R, G, B channels
        channel_red = image[:, :, 0]
        channel_green = image[:, :, 1]
        channel_blue = image[:, :, 2]

        # Calculate total intensity levels for each channel (equations 2-4)
        sum_red = np.sum(channel_red)
        sum_green = np.sum(channel_green)
        sum_blue = np.sum(channel_blue)

        # Calculate weight contributions (equations 5-7)
        total = sum_red + sum_green + sum_blue
        weight_red = sum_red / total
        weight_green = sum_green / total
        weight_blue = sum_blue / total

        # Find min and max values for each channel (equations 9-14)
        min_red, max_red = np.min(channel_red), np.max(channel_red)
        min_green, max_greeen = np.min(channel_green), np.max(channel_green)
        min_blue, max_blue = np.min(channel_blue), np.max(channel_blue)

        # Avoid division by zero in case all values are the same
        range_red = max_red - min_red if max_red != min_red else 1
        range_green = max_greeen - min_green if max_greeen != min_green else 1
        range_blue = max_blue - min_blue if max_blue != min_blue else 1

        # Calculate stretched values (equations 16-18)
        R_stretched = 255 * ((channel_red - min_red) / range_red)
        G_stretched = 255 * ((channel_green - min_green) / range_green)
        B_stretched = 255 * ((channel_blue - min_blue) / range_blue)

        # Combine channels with weights (equation 15)
        grayscale = (
            weight_red * R_stretched
            + weight_green * G_stretched
            + weight_blue * B_stretched
        )

        # Ensure the grayscale values are in the range [0, 255]
        assert (
            np.clip(grayscale, 0, 255).all() == grayscale.all()
        ), "Grayscale values are out of bounds."

        return grayscale
