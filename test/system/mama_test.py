"""
System tests for mama.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import argparse as argp
import copy
import itertools
import tempfile

import numpy as np
import pandas as pd
import pytest

import mama2.mama as mama
import mama2.mama_pipeline as mp
import mama2.util.sumstats as ss


test_directory = os.path.dirname(__file__)
data_directory = os.path.abspath(os.path.join(test_directory, '../data'))

@pytest.fixture(scope="module")
def temp_test_dir():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as t:
        yield t


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
    ld_cols = ld_anc_cols + [ss.SNP_COL]
    ld_df = pd.DataFrame(columns=ld_cols)
    ld_df.to_csv(ldscores_filename, sep="\t", index=False)

    # Create the (empty except for column names) sumstats files
    ss_files = {(anc, phen) : os.path.join(tmp_dir_path,
                                           '%s_%s_ss.txt' % (anc.lower(), phen.lower()))
                for anc in ancestries for phen in phenotypes}
    ss_cols = mama.MAMA_REQ_STD_COLS
    ss_df = pd.DataFrame(columns=ss_cols)
    for ss_file in ss_files.values():
        ss_df.to_csv(ss_file, sep="\t", index=False)


    # Set namespace attributes
    pargs.out = os.path.join(tmp_dir_path, 'test_prefix')
    pargs.sumstats = [(f,a,p) for ((a,p), f) in ss_files.items()]
    pargs.ld_scores = ldscores_filename

    return pargs


#===================================================================================================
class TestValidateInputs:

    def test__happy_path__expected_results(self, valid_basic_pargs, temp_test_dir):
        result = mama.validate_inputs(valid_basic_pargs, dict())

        num_ancestries = len(result[mama.ANCESTRIES])

        assert result[mama.OUT_DIR] == temp_test_dir
        assert result[mama.OUT_PREFIX] == 'test_prefix'
        assert num_ancestries > 0
        assert result[mama.RE_MAP] == mama.MAMA_RE_EXPR_MAP
        assert result[mama.FILTER_MAP] == mama.MAMA_STD_FILTERS
        assert len(result[mama.SUMSTATS_MAP]) == len(valid_basic_pargs.sumstats)
        assert not result[mama.HARM_FILENAME_FSTR]
        assert not result[mama.REG_FILENAME_FSTR]
        assert result[mama.REG_LD_COEF_OPT] == mama.MAMA_REG_OPT_ALL_FREE
        assert result[mama.REG_SE2_COEF_OPT] == mama.MAMA_REG_OPT_ALL_FREE
        assert result[mama.REG_INT_COEF_OPT] == mama.MAMA_REG_OPT_ALL_FREE

        for attr in vars(valid_basic_pargs):
            assert getattr(valid_basic_pargs, attr) == result[attr]

    #########

    def test__invalid_frequency_filter_range__throws_error(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set min frequency > max frequency
        n.freq_bounds = [1.0, 0.0]

        with pytest.raises(RuntimeError) as ex_info:
            mama.validate_inputs(n, dict())
        assert str(n.freq_bounds[0]) in str(ex_info.value)
        assert str(n.freq_bounds[1]) in str(ex_info.value)

    #########

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
            mama.validate_inputs(n, dict())
        assert dropped_anc1 in str(ex_info.value)
        assert dropped_anc2 in str(ex_info.value)

    #########

    def test__missing_ld_snp_col__throws_error(self, valid_basic_pargs, temp_test_dir):

        n = copy.copy(valid_basic_pargs)

        # Remove column from LD scores file
        ld_df = pd.read_csv(n.ld_scores, sep=None, engine='python', nrows=1, comment="#")
        ld_cols = ld_df.columns
        ld_pair_cols = [col for col in ld_cols if "_" in col]

        dropped_col = ss.SNP_COL
        bad_ldscores_file = os.path.join(temp_test_dir, 'missing_snp_col_ldscores.txt')
        ld_df.drop([dropped_col], axis=1, inplace=True)
        ld_df.to_csv(bad_ldscores_file, sep="\t", index=False)
        n.ld_scores = bad_ldscores_file

        with pytest.raises(RuntimeError) as ex_info:
            mama.validate_inputs(n, dict())
        assert dropped_col in str(ex_info.value)

    #########

    def test__add_and_replace_re__expected_results(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set some regular expressions (at least one add and at least one replace)
        add_col = ss.CHR_COL
        add_re = 'XYZ'

        replace_col = ss.A1_COL
        replace_re = '.*A1.*'

        setattr(n, mama.to_arg(mama.MAMA_RE_ADD_FLAGS[add_col]), add_re)
        setattr(n, mama.to_arg(mama.MAMA_RE_REPLACE_FLAGS[replace_col]), replace_re)

        result = mama.validate_inputs(n, dict())
        assert result[mama.RE_MAP][add_col] == (mama.MAMA_RE_EXPR_MAP[add_col] + "|" + add_re)
        assert result[mama.RE_MAP][replace_col] == replace_re

    #########

    @pytest.mark.parametrize("farg, filter_name",
        [
            ("allow_palindromic_snps", mp.SNP_PALIN_FILT)
        ])
    def test__filter_removal_flags__expected_results(self, farg, filter_name, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Remove the indicated filter
        setattr(n, farg, True)

        result = mama.validate_inputs(n, dict())
        assert filter_name not in result[mama.FILTER_MAP]


    #########

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
        df = pd.DataFrame({ss.FREQ_COL : freq_data})

        result = mama.validate_inputs(n, dict())

        assert mama.FREQ_FILTER in result[mama.FILTER_MAP]
        freq_filter_func, freq_filter_desc = result[mama.FILTER_MAP][mama.FREQ_FILTER]

        filt_results = freq_filter_func(df)

        assert filt_results.sum() == 2
        assert filt_results[0] == True
        assert filt_results[len(filt_results) - 1] == True
        assert str(min_f) in freq_filter_desc
        assert str(max_f) in freq_filter_desc

    # TODO(jonbjala) Test other filters?  (like CHR)
    #########

    LD_OPT_TUPLES = [
        ("dummy_attr", mama.MAMA_REG_OPT_ALL_FREE),
        ("reg_ld_perf_corr", mama.MAMA_REG_OPT_PERF_CORR)
    ]

    SE2_OPT_TUPLES = [
        ("dummy_attr", mama.MAMA_REG_OPT_ALL_FREE),
        ("reg_se2_zero", mama.MAMA_REG_OPT_ALL_ZERO),
        ("reg_se2_ident", mama.MAMA_REG_OPT_IDENT),
        ("reg_se2_diag", mama.MAMA_REG_OPT_OFFDIAG_ZERO),
    ]

    INT_OPT_TUPLES = [
        ("dummy_attr", mama.MAMA_REG_OPT_ALL_FREE),
        ("reg_int_zero", mama.MAMA_REG_OPT_ALL_ZERO),
        ("reg_int_diag", mama.MAMA_REG_OPT_OFFDIAG_ZERO),
    ]
    @pytest.mark.parametrize("ld_se2_int_tuples",
                             list(itertools.product(LD_OPT_TUPLES, SE2_OPT_TUPLES, INT_OPT_TUPLES)))
    def test__reg_coef_nonfile_opts__expected_results(self, ld_se2_int_tuples, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set flag values
        for t in ld_se2_int_tuples:
            setattr(n, t[0], True)

        result = mama.validate_inputs(n, dict())

        assert result[mama.REG_LD_COEF_OPT] == ld_se2_int_tuples[0][1]
        assert result[mama.REG_SE2_COEF_OPT] == ld_se2_int_tuples[1][1]
        assert result[mama.REG_INT_COEF_OPT] == ld_se2_int_tuples[2][1]

    #########

    def test__reg_coef_file_opts__expected_results(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        arr = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        filename = os.path.abspath(os.path.join(data_directory, "coef_mat/sym_mat_1.coef"))
        n.reg_ld_coef = mama.input_np_matrix(filename)

        result = mama.validate_inputs(n, dict())

        assert isinstance(result[mama.REG_LD_COEF_OPT], np.ndarray)
        assert result[mama.REG_LD_COEF_OPT].shape == (9, 9)
        assert np.allclose(result[mama.REG_LD_COEF_OPT], np.outer(arr, arr))
        assert result[mama.REG_SE2_COEF_OPT] == mama.MAMA_REG_OPT_ALL_FREE
        assert result[mama.REG_INT_COEF_OPT] == mama.MAMA_REG_OPT_ALL_FREE

    #########

    def test__specify_harm_out__expected_results(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set harmonized output flag
        n.out_harmonized = True

        result = mama.validate_inputs(n, dict())
        assert result[mama.HARM_FILENAME_FSTR]
        assert n.out in result[mama.HARM_FILENAME_FSTR]
        assert mama.HARMONIZED_SUFFIX in result[mama.HARM_FILENAME_FSTR]

    #########

    def test__specify_reg_out__expected_results(self, valid_basic_pargs):

        n = copy.copy(valid_basic_pargs)

        # Set harmonized output flag
        n.out_reg_coef = True

        result = mama.validate_inputs(n, dict())
        assert result[mama.REG_FILENAME_FSTR]
        assert n.out in result[mama.REG_FILENAME_FSTR]
        assert mama.LD_COEF_SUFFIX in result[mama.REG_FILENAME_FSTR]


    # TODO(jonbjala) Test column mapping (SS file with missing columns)
