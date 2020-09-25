"""
End-to-end tests of the mama2 software.  This should be run via pytest. TODO(jonbjala)
"""

import numpy as np
import os

import pytest
import tempfile

import mama2.mama2 as mama2

test_directory = os.path.dirname(__file__)
data_directory = os.path.join(test_directory, 'data')

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


#===================================================================================================
# read_and_qc_sumstats
# TODO(jonbjala) Like unit tests, wrap in a class?

def test__read_and_qc_sumstats__invalid_rows__expected_drops():
    """
    Test the reading in and QCing of summary stats.  Uses precanned fake summary stats with
    various rows containing conditions that should be filtered out.  Uses default values for
    all optional parameters.
    """
    ss_file = "sumstats_filtertest.txt"
    ss_full_filepath = os.path.join(data_directory, ss_file)

    df = mama2.read_and_qc_sumstats(ss_full_filepath)

    res_col = next(c for c in df.columns if "keep" in c)
    num_keeps = int(next(token for token in res_col.split("_") if token.isnumeric()))

    assert len(df) == num_keeps
    assert all(~df[res_col].str.contains("Drop"))
    assert all(df[res_col].str.contains("Keep"))


def test__read_and_qc_sumstats__various_file_seps__same_results():
    """
    TODO(jonbjala)
    """
    pass


def test__read_and_qc_sumstats__alt_filters__expected_results():
    """
    TODO(jonbjala)
    """
    pass


def test__read_and_qc_sumstats__alt_colnames_filters__expected_results():
    """
    TODO(jonbjala)
    """
    pass


def test__read_and_qc_sumstats__specify_col_map__expected_results():
    """
    TODO(jonbjala)
    """
    pass