"""
End-to-end tests of the mama2 software.  This should be run via pytest. TODO(jonbjala)
"""

import numpy as np
import os

import pytest
import tempfile

import mama2.mama2 as mama2

test_directory = os.path.dirname(__file__)

@pytest.fixture(scope="module")
def temp_test_dir():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as t:
        yield t

#===================================================================================================

def test_1(temp_test_dir, request):
    """
    TODO(jonbjala)
    """
    pass
