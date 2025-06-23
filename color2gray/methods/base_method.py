from abc import abstractmethod
from typing import Any, Dict, Literal, Optional

import numpy as np


class BaseMethod:

    def __init__(self, **params: Optional[Dict[str, Any]]):
        """
        Initialize the BaseMethod with optional parameters.
        :param params: Optional parameters for the conversion method.
        """
        # self.params = params if params is not None else {}
        if params is None:
            params = {}
        self.params: Dict[str, Any] = params

    @staticmethod
    def _validate_image(image: np.ndarray):
        """
        Validate the input image.
        If no image is provided, use the initialized image.
        :param image: Optional input image to validate.
        :return: Validated image as a numpy array.
        """
        assert isinstance(image, np.ndarray), "Image must be a numpy array."

        # Check that the image is RGB or RGBA. If it's grayscale, it's an error.
        # Convert RGBA to RGB if necessary.
        if image.ndim == 2:
            raise ValueError("Input image must be RGB, not grayscale.")
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB.")

        if image.shape[2] == 4:
            print(
                "Warning: Input image has an alpha channel. This can affect the conversion."
            )

        if image.max() > 255 or image.max() <= 1.0:
            raise ValueError(
                "Input image values should be in the range [0, 255]. "
                "If the input image is normalized, please multiply it by 255."
            )

    def convert(
        self, image: np.ndarray, **params: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Convert the input image to grayscale.

        Args:
            image (np.ndarray): The input image in RGB format.
            **params: Additional parameters for the conversion method.
        Returns:
            np.ndarray: The grayscale image.
        Raises:
            ValueError: If no image is provided or if the image is not in the correct format.


        """
        if image is None:
            raise ValueError("No image provided for conversion.")

        image = image[:, :, :3]  # Convert RGBA to RGB by removing the alpha channel

        self._validate_image(image)

        if params is None:
            params = self.params

        grayscale_image = self._convert(image, **params)

        if grayscale_image.max() <= 1.0:
            raise ValueError(
                f"{self.__class__.__name__} : "
                "The grayscale image values should be in the range [0, 255]. "
                "If the input image is normalized, please multiply it by 255."
            )
        if grayscale_image.max() > 255:
            raise ValueError(
                f"{self.__class__.__name__} : "
                "The grayscale image values should be in the range [0, 255]. "
                "If the input image is not normalized, please divide it by 255."
            )

        return grayscale_image.astype(np.uint8)

    @abstractmethod
    def _convert(
        self,
        image: np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.int8]],
        **params,
    ) -> np.ndarray:
        """
        Abstract method to convert the input image to grayscale.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __call__(self, image: np.ndarray, **params) -> np.ndarray:
        return self.convert(image, **params)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
