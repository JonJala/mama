"""
System tests for util/sumstats.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import tempfile

import pytest

import mama.mama as mama
import mama.mama_pipeline as mp
import mama.util.sumstats as ss



@pytest.fixture(scope="module")
def temp_test_dir():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as t:
        yield t


#===================================================================================================
class TestProcessSumstats:

    def test__invalid_rows__expected_drops(self):
        """
        Test the reading in and QCing of summary stats.  Uses precanned fake summary stats with
        various rows containing conditions that should be filtered out.  Uses default values for
        all optional parameters.
        """
        ss_file = "sumstats_filter/data.txt"
        ss_full_filepath = os.path.join(data_directory, ss_file)
        orig_df = mp.obtain_df(ss_full_filepath, "SS")

        df = ss.process_sumstats(orig_df, mp.MAMA_RE_EXPR_MAP, mp.MAMA_REQ_STD_COLS,
                                 mp.MAMA_STD_FILTERS)

        res_col = next(c for c in df.columns if "keep" in c)
        num_keeps = int(next(token for token in res_col.split("_") if token.isnumeric()))

        assert len(df) == num_keeps
        assert all(~df[res_col].str.contains("Drop"))
        assert all(df[res_col].str.contains("Keep"))


    def test__alt_filters__expected_results(self):
        """
        TODO(jonbjala)
        """
        pass


    def test__alt_colnames_filters__expected_results(self):
        """
        TODO(jonbjala)
        """
        pass


    def test__specify_col_map__expected_results(self):
        """
        TODO(jonbjala)
        """
        pass