from typing import Type

from .average import AverageMethod
from .average_luminous import AverageLuminanceMethod
from .base_method import BaseMethod
from .necg import NECGMethod
from .snecg import SNECGMethod

__all__ = [
    "AverageMethod",
    "AverageLuminanceMethod",
    "NECGMethod",
    "SNECGMethod",
]

all_methods: dict[str, Type[BaseMethod]] = {
    "average": AverageMethod,
    "average_luminance": AverageLuminanceMethod,
    "necg": NECGMethod,
    "snecg": SNECGMethod,
}
