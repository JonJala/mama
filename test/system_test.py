"""
End-to-end tests of the mama2 software.  This should be run via pytest. TODO(jonbjala)
"""

import copy
import itertools
import os
import tempfile

import argparse as argp
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

class TestProcessSumstats:

    def test__invalid_rows__expected_drops(self):
        """
        Test the reading in and QCing of summary stats.  Uses precanned fake summary stats with
        various rows containing conditions that should be filtered out.  Uses default values for
        all optional parameters.
        """
        ss_file = "sumstats_filter/data.txt"
        ss_full_filepath = os.path.join(data_directory, ss_file)

        orig_df = mama2.obtain_df(ss_full_filepath, "SS")

        df = mama2.process_sumstats(orig_df)

        res_col = next(c for c in df.columns if "keep" in c)
        num_keeps = int(next(token for token in res_col.split("_") if token.isnumeric()))

        assert len(df) == num_keeps
        assert all(~df[res_col].str.contains("Drop"))
        assert all(df[res_col].str.contains("Keep"))


    def test__various_file_seps__same_results(self):
        """
        TODO(jonbjala)
        """
        pass


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

#===================================================================================================
# validate_inputs

@pytest.fixture(scope="module")
def valid_basic_pargs(temp_test_dir):

    # Test parameters
    num_anc = 3
    num_phen = 3

    # The resulting namespace that will be returned
    pargs = argp.Namespace()

    # Make sure we have an absolute path to the temporary directory
    tmp_dir_path = os.path.abspath(temp_test_dir)

    # Construct ancestry and phenotype names
    ancestries = ["ANC" + str(a) for a in range(num_anc)]
    phenotypes = ["PHEN" + str(p) for p in range(num_phen)]

    # Create the (empty except for column names) LD Scores file
    ldscores_filename = os.path.join(tmp_dir_path, "ldscores.txt")
    ld_anc_cols = ["%s_%s" % anc_tuple
                   for anc_tuple in itertools.combinations_with_replacement(ancestries, 2)]
    ld_cols = ld_anc_cols + [mama2.SNP_COL]
    ld_df = pd.DataFrame(columns=ld_cols)
    ld_df.to_csv(ldscores_filename, sep="\t", index=False)

    # Create the (empty except for column names) sumstats files
    ss_files = {(anc, phen) : os.path.join(tmp_dir_path,
                                           '%s_%s_ss.txt' % (anc.lower(), phen.lower()))
                for anc in ancestries for phen in phenotypes}
    ss_cols = mama2.MAMA_REQ_STD_COLS
    ss_df = pd.DataFrame(columns=ss_cols)
    for ss_file in ss_files.values():
        ss_df.to_csv(ss_file, sep="\t", index=False)


    # Set namespace attributes
    pargs.out = os.path.join(tmp_dir_path, 'test_prefix')
    pargs.freq_bounds = [0.0, 1.0]
    pargs.sumstats = [(f,a,p) for ((a,p), f) in ss_files.items()]
    pargs.ld_scores = ldscores_filename

    return pargs


class TestValidateInputs:

    def test__happy_path__expected_results(self, valid_basic_pargs, temp_test_dir):
        result = mama2.validate_inputs(valid_basic_pargs, dict())

        num_ancestries = len(result[mama2.ANCESTRIES])

        assert result[mama2.OUT_DIR] == temp_test_dir
        assert result[mama2.OUT_PREFIX] == 'test_prefix'
        assert num_ancestries > 0
        assert result[mama2.RE_MAP] == mama2.MAMA_RE_EXPR_MAP
        assert result[mama2.FILTER_MAP] == mama2.MAMA_STD_FILTERS

        for attr in vars(valid_basic_pargs):
            assert getattr(valid_basic_pargs, attr) == result[attr]


    def test__existing_directory_conflicts_with_out_prefix__throws_error(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Make sure there is a collision between the specified out prefix and the
        # existing temporary directory
        n.out = os.path.dirname(n.out)
        prefix = os.path.basename(n.out)

        with pytest.raises(RuntimeError) as ex_info:
            mama2.validate_inputs(n, dict())
        assert prefix in str(ex_info.value)


    def test__missing_out_directory__throws_error(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Insert bogus directory into out prefix path
        dir_name = os.path.dirname(n.out)
        prefix_name = os.path.basename(n.out)
        invalid_subdir = "random_directory_abcxyz"
        n.out = os.path.join(dir_name, "random_directory_abcxyz", prefix_name)

        with pytest.raises(FileNotFoundError) as ex_info:
            mama2.validate_inputs(n, dict())
        assert invalid_subdir in str(ex_info.value)


    def test__invalid_frequency_filter_range__throws_error(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set min frequency > max frequency
        n.freq_bounds = [1.0, 0.0]

        with pytest.raises(RuntimeError) as ex_info:
            mama2.validate_inputs(n, dict())
        assert str(n.freq_bounds[0]) in str(ex_info.value)
        assert str(n.freq_bounds[1]) in str(ex_info.value)


    def test__missing_ld_pair_col__throws_error(self, valid_basic_pargs, temp_test_dir):

        n = copy.copy(valid_basic_pargs)

        # Remove column from LD scores file
        ld_df = pd.read_csv(n.ld_scores, sep=None, engine='python', nrows=1, comment="#")
        ld_cols = ld_df.columns
        ld_pair_cols = [col for col in ld_cols if "_" in col]

        dropped_col = ld_pair_cols[0]
        dropped_anc1, dropped_anc2 = dropped_col.split("_")
        ld_df.drop([dropped_col], axis=1, inplace=True)
        bad_ldscores_file = os.path.join(temp_test_dir, 'missing_pair_col_ldscores.txt')
        ld_df.to_csv(bad_ldscores_file, sep="\t", index=False)
        n.ld_scores = bad_ldscores_file

        with pytest.raises(RuntimeError) as ex_info:
            mama2.validate_inputs(n, dict())
        assert dropped_anc1 in str(ex_info.value)
        assert dropped_anc2 in str(ex_info.value)


    def test__missing_ld_snp_col__throws_error(self, valid_basic_pargs, temp_test_dir):

        n = copy.copy(valid_basic_pargs)

        # Remove column from LD scores file
        ld_df = pd.read_csv(n.ld_scores, sep=None, engine='python', nrows=1, comment="#")
        ld_cols = ld_df.columns
        ld_pair_cols = [col for col in ld_cols if "_" in col]

        dropped_col = mama2.SNP_COL
        bad_ldscores_file = os.path.join(temp_test_dir, 'missing_snp_col_ldscores.txt')
        ld_df.drop([dropped_col], axis=1, inplace=True)
        ld_df.to_csv(bad_ldscores_file, sep="\t", index=False)
        n.ld_scores = bad_ldscores_file

        with pytest.raises(RuntimeError) as ex_info:
            mama2.validate_inputs(n, dict())
        assert dropped_col in str(ex_info.value)


    def test__add_and_replace_re__expected_results(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set some regular expressions (at least one add and at least one replace)
        add_col = mama2.CHR_COL
        add_re = 'XYZ'

        replace_col = mama2.A1_COL
        replace_re = '.*A1.*'

        setattr(n, mama2.to_arg(mama2.MAMA_RE_ADD_FLAGS[add_col]), add_re)
        setattr(n, mama2.to_arg(mama2.MAMA_RE_REPLACE_FLAGS[replace_col]), replace_re)

        result = mama2.validate_inputs(n, dict())
        assert result[mama2.RE_MAP][add_col] == (mama2.MAMA_RE_EXPR_MAP[add_col] + "|" + add_re)
        assert result[mama2.RE_MAP][replace_col] == replace_re


    @pytest.mark.parametrize("farg, filter_name",
        [
            ("allow_non_rs", mama2.SNP_PREFIX_FILTER),
            ("allow_non_1_22_chr", mama2.CHR_FILTER),
            ("allow_palindromic_snps", mama2.SNP_PALIN_FILT)
        ])
    def test__filter_removal_flags__expected_results(self, farg, filter_name, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Remove the indicated filter
        setattr(n, farg, True)

        result = mama2.validate_inputs(n, dict())
        assert filter_name not in result[mama2.FILTER_MAP]


    @pytest.mark.parametrize("min_f, max_f",
        [
            (0.0, 0.4),
            (0.6, 1.0),
            (0.1, 0.5),
            (-0.1, 0.5),
            (0.2, 0.8),
            (0.1, 1.5)
        ])
    def test__freq_filter_flag__expected_results(self, min_f, max_f, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set the frequency filter flag
        setattr(n, "freq_bounds", [min_f, max_f])

        freq_data = [min_f - 0.00001, min_f, min_f + 0.00001, 0.5 * (min_f + max_f),
                     max_f - 0.00001, max_f, max_f + 0.00001]
        df = pd.DataFrame({mama2.FREQ_COL : freq_data})

        result = mama2.validate_inputs(n, dict())

        assert mama2.FREQ_FILTER in result[mama2.FILTER_MAP]
        freq_filter_func, freq_filter_desc = result[mama2.FILTER_MAP][mama2.FREQ_FILTER]

        filt_results = freq_filter_func(df)

        assert filt_results.sum() == 2
        assert filt_results[0] == True
        assert filt_results[len(filt_results) - 1] == True
        assert str(min_f) in freq_filter_desc
        assert str(max_f) in freq_filter_desc

    # TODO(jonbjala) Test column mapping (SS file with missing columns)
