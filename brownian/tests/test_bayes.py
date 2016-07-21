import unittest
from nose.tools import assert_raises, assert_almost_equal
from numpy.testing import assert_array_almost_equal

from brownian import bayes

class EmptyTest(unittest.TestCase):
    def test_empty(self):
        pass