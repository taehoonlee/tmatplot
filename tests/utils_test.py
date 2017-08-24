import numpy as np
import pytest

from tmatplot.utils import getRanges
from tmatplot.utils import makeKwargs


def test_getRanges():
    assert getRanges([1, 2, 3, 5, 6, 8]) == [(1, 3), (5, 6), (8, 8)]


def test_makeKwargs():
    assert makeKwargs() == {}
    assert makeKwargs(bins=5) == {'bins': 5}
