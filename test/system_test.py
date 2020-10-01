"""
End-to-end tests of the mama2 software.  This should be run via pytest. TODO(jonbjala)
"""


import os
import tempfile

import numpy as np
import pandas as pd
import pytest

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



#===================================================================================================
# harmonize_all

class TestHarmonizeAll:

    _LD_SNPS = ['rs100', 'rs150', 'rs200', 'rs250', 'rs300', 'rs350', 'rs400']
    _LD_DATA = {
        'dummy' : [1, 2, 3, 4, 5, 6, 7],
    }

    _POP1_SNPS = ['rs100', 'rs125', 'rs200', 'rs225', 'rs300', 'rs325', 'rs400', 'rs425']
    _POP1_DATA = {
        mama2.A1_COL : ['C', 'T', 'G', 'A', 'C', 'T', 'G', 'A'],
        mama2.A2_COL : ['A', 'C', 'T', 'G', 'A', 'C', 'T', 'G'],
        mama2.BETA_COL : [1.0, 0.2, 1.0, -0.3, 1.0, 0.5, 1.0, -5.0],
        mama2.FREQ_COL : [0.1, 0.9, 0.5, 0.8, 0.2, 0.75, 0.2, 0.7]
    }

    _POP2_SNPS = ['rs050', 'rs100', 'rs200', 'rs300', 'rs325', 'rs350']
    _POP2_DATA = {
        mama2.A1_COL : ['C', 'A', 'G', 'A', 'T', 'A'],
        mama2.A2_COL : ['A', 'C', 'T', 'C', 'C', 'G'],
        mama2.BETA_COL : [0.5, -1.0, 1.0, -1.0, -0.2, 0.5],
        mama2.FREQ_COL : [0.8, 0.8, 0.4, 0.5, 0.9, 0.75]
    }

    _POP3_SNPS = ['rs100', 'rs200', 'rs250', 'rs300', 'rs325', 'rs350', 'rs425', 'rs500']
    _POP3_DATA = {
        mama2.A1_COL : ['C', 'G', 'A', 'A', 'T', 'G', 'G', 'A'],
        mama2.A2_COL : ['A', 'T', 'G', 'C', 'C', 'A', 'A', 'G'],
        mama2.BETA_COL : [1.0, 1.0, 0.4, -1.0, 0.7, 0.5, 0.3, -5.0],
        mama2.FREQ_COL : [0.1, 0.3, 0.5, 0.8, 0.2, 0.75, 0.2, 0.7]
    }


    #########
    def test__varying_snps__expected_results(self):

        # Intersection should be "rs100", "rs200", and "rs300"
        df_ld = pd.DataFrame(TestHarmonizeAll._LD_DATA,
                           index=TestHarmonizeAll._LD_SNPS)
        df_1 = pd.DataFrame(TestHarmonizeAll._POP1_DATA,
                           index=TestHarmonizeAll._POP1_SNPS)
        df_2 = pd.DataFrame(TestHarmonizeAll._POP2_DATA,
                           index=TestHarmonizeAll._POP2_SNPS)
        df_3 = pd.DataFrame(TestHarmonizeAll._POP3_DATA,
                           index=TestHarmonizeAll._POP3_SNPS)
        pops = {1 : df_1.copy(), 2 : df_2.copy(), 3 : df_3.copy()}

        df_ld_copy = df_ld.copy()
        mama2.harmonize_all(pops, df_ld_copy)

        # Make sure the number of populations stays the same
        assert len(pops) == 3
        assert 1 in pops
        assert 2 in pops
        assert 3 in pops

        # Make sure the populations are the expected length and contain the expected SNPs
        for df in pops.values():
            assert len(df) == 3
            assert all(df[mama2.BETA_COL] == 1.0)
            assert all(df[mama2.FREQ_COL] <= 0.5)
            assert all(x in df.index for x in ['rs100', 'rs200', 'rs300'])

        # Make sure the LD score DF is the expected length and contains the expected SNPs
        assert len(df_ld_copy) == 3
        assert all(x in df_ld_copy.index for x in ['rs100', 'rs200', 'rs300'])
