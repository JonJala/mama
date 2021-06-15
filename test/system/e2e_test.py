"""
System tests for mama_pipeline.py.  This should be run via pytest.
"""

import os
import sys
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_dir = os.path.abspath(os.path.join(main_dir, 'test'))
data_dir = os.path.abspath(os.path.join(test_dir, 'data'))
sys.path.append(main_dir)

import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

import mama.mama_pipeline as mp
import mama.util.sumstats as ss

@pytest.fixture(scope="function")
def temp_test_dir():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as t:
        yield t


TWO_POP_DIR = os.path.abspath(os.path.join(data_dir, 'two_pop'))
POP1_PHENO1_FILE = os.path.abspath(os.path.join(TWO_POP_DIR, 'pop1_pheno1_sumstats.txt'))
POP2_PHENO1_FILE = os.path.abspath(os.path.join(TWO_POP_DIR, 'pop2_pheno1_sumstats.txt'))
POP1_POP2_LDSCORE_FILE = os.path.abspath(os.path.join(TWO_POP_DIR, 'pop1_pop2_chr1.l2.ldscore'))


#===================================================================================================
class TestEndToEnd:


    #########
    def test__two_pops_default_logging__expected_results(self, temp_test_dir):

        mama_py = os.path.abspath(os.path.join(main_dir, 'mama.py'))

        test_prefix = 'test'
        full_test_prefix = os.path.join(temp_test_dir, test_prefix)
        logfile = full_test_prefix + ".log"

        harmfile1 = full_test_prefix + "_POP1_PHENO1.hrm"
        harmfile2 = full_test_prefix + "_POP2_PHENO1.hrm"
        harmfile1_expected = os.path.join(TWO_POP_DIR, "POP1_PHENO1.hrm_expected")
        harmfile2_expected = os.path.join(TWO_POP_DIR, "POP2_PHENO1.hrm_expected")

        reg_ld_file = full_test_prefix + "_ld_reg.cf"
        reg_int_file = full_test_prefix + "_int_reg.cf"
        reg_se2_file = full_test_prefix + "_se2_reg.cf"
        reg_ld_file_expected = os.path.join(TWO_POP_DIR, "ld_reg.cf_expected")
        reg_int_file_expected = os.path.join(TWO_POP_DIR, "int_reg.cf_expected")
        reg_se2_file_expected = os.path.join(TWO_POP_DIR, "se2_reg.cf_expected")

        resultfile1 = full_test_prefix + "_POP1_PHENO1.res"
        resultfile2 = full_test_prefix + "_POP2_PHENO1.res"
        resultfile1_expected = os.path.join(TWO_POP_DIR, "POP1_PHENO1.res_expected")
        resultfile2_expected = os.path.join(TWO_POP_DIR, "POP2_PHENO1.res_expected")


        # Determine the command that will be run
        run_cmd = [
            mama_py,
            '--sumstats', '%s,POP1,PHENO1' % POP1_PHENO1_FILE, '%s,POP2,PHENO1' % POP2_PHENO1_FILE,
            '--ld-scores', POP1_POP2_LDSCORE_FILE,
            '--out', full_test_prefix,
            '--out-harmonized',
            '--out-reg-coef',
        ]

        # Run the process, make sure output is captured and process exit code is good
        cp = subprocess.run(run_cmd, capture_output=True, check=True)
        sout = cp.stdout.decode("utf-8")
        serr = cp.stderr.decode("utf-8")

        # Make sure various statements are in the log and were sent to stdout
        assert os.path.exists(logfile)
        with open(logfile) as logf:
            logtext = logf.read()

        assert "POP1,PHENO1" in logtext
        assert "POP1,PHENO1" in sout
        assert "POP2,PHENO1" in logtext
        assert "POP2,PHENO1" in sout

        assert "Number of SNPS in initial intersection of all sources: 32" in logtext
        assert "Number of SNPS in initial intersection of all sources: 32" in sout

        assert "Writing harmonized summary statistics to disk" in logtext
        assert "Writing harmonized summary statistics to disk" in sout

        assert "Harmonized POP1 PHENO1 mean chi squared:" not in logtext
        assert "Harmonized POP1 PHENO1 mean chi squared:" not in sout
        assert "Harmonized POP2 PHENO1 mean chi squared:" not in logtext
        assert "Harmonized POP2 PHENO1 mean chi squared:" not in sout

        assert "SNPs to make omega positive semi-definite" in logtext
        assert "SNPs to make omega positive semi-definite" in sout

        assert "Skipping positive-semi-definiteness check of Omega" not in logtext
        assert "Skipping positive-semi-definiteness check of Omega" not in sout

        assert "SNPs due to non-positive-definiteness of sigma" in logtext
        assert "SNPs due to non-positive-definiteness of sigma" in sout

        assert "total SNPs due to non-positive-(semi)-definiteness of omega / sigma" in logtext
        assert "total SNPs due to non-positive-(semi)-definiteness of omega / sigma" in sout

        assert "Mean Chi^2 for (\'POP1\', \'PHENO1\') = 3.4" in logtext
        assert "Mean Chi^2 for (\'POP1\', \'PHENO1\') = 3.4" in sout

        assert "%s_POP1_PHENO1.res" % full_test_prefix in logtext
        assert "%s_POP1_PHENO1.res" % full_test_prefix in sout
        assert "%s_POP2_PHENO1.res" % full_test_prefix in logtext
        assert "%s_POP2_PHENO1.res" % full_test_prefix in sout


        # Make sure the harmonized files exist and contain the right contents
        assert os.path.exists(harmfile1)
        assert os.path.exists(harmfile2)

        harm1_df = pd.read_csv(harmfile1, sep=None, comment="#", engine="python")
        harm2_df = pd.read_csv(harmfile2, sep=None, comment="#", engine="python")

        harm1_expected_df = pd.read_csv(harmfile1_expected, sep=None, comment="#", engine="python")
        harm2_expected_df = pd.read_csv(harmfile2_expected, sep=None, comment="#", engine="python")

        pd.testing.assert_frame_equal(harm1_df, harm1_expected_df, check_exact=False)
        pd.testing.assert_frame_equal(harm2_df, harm2_expected_df, check_exact=False)


        # Make sure the regression coefficient files exist and contain the right contents
        assert os.path.exists(reg_ld_file)
        assert os.path.exists(reg_int_file)
        assert os.path.exists(reg_se2_file)

        ld_coef_matrix = np.fromfile(reg_ld_file, sep='\t')
        int_coef_matrix = np.fromfile(reg_int_file, sep='\t')
        se2_coef_matrix = np.fromfile(reg_se2_file, sep='\t')

        ld_coef_matrix_expected = np.fromfile(reg_ld_file_expected, sep='\t')
        int_coef_matrix_expected = np.fromfile(reg_int_file_expected, sep='\t')
        se2_coef_matrix_expected = np.fromfile(reg_se2_file_expected, sep='\t')

        assert np.allclose(ld_coef_matrix, ld_coef_matrix_expected)
        assert np.allclose(int_coef_matrix, int_coef_matrix_expected)
        assert np.allclose(se2_coef_matrix, se2_coef_matrix_expected)


        # Make sure the result files exist and contain the right contents
        assert os.path.exists(resultfile1)
        assert os.path.exists(resultfile2)

        res1_df = pd.read_csv(resultfile1, sep=None, comment="#", engine="python")
        res2_df = pd.read_csv(resultfile2, sep=None, comment="#", engine="python")

        res1_df_expected = pd.read_csv(resultfile1_expected, sep=None, comment="#", engine="python")
        res2_df_expected = pd.read_csv(resultfile2_expected, sep=None, comment="#", engine="python")

        pd.testing.assert_frame_equal(res1_df, res1_df_expected, check_exact=False)
        pd.testing.assert_frame_equal(res2_df, res2_df_expected, check_exact=False)


    #########
    def test__one_pop_default_logging__expected_results(self, temp_test_dir):

        mama_py = os.path.abspath(os.path.join(main_dir, 'mama.py'))

        test_prefix = 'test'
        full_test_prefix = os.path.join(temp_test_dir, test_prefix)
        logfile = full_test_prefix + ".log"
        harmfile1 = full_test_prefix + "_POP1_PHENO1.hrm"
        harmfile2 = full_test_prefix + "_POP2_PHENO1.hrm"

        reg_ld_file = full_test_prefix + "_ld_reg.cf"
        reg_int_file = full_test_prefix + "_int_reg.cf"
        reg_se2_file = full_test_prefix + "_se2_reg.cf"
        reg_ld_file_expected = os.path.join(TWO_POP_DIR, "ld_reg.cf_expected")
        reg_int_file_expected = os.path.join(TWO_POP_DIR, "int_reg.cf_expected")
        reg_se2_file_expected = os.path.join(TWO_POP_DIR, "se2_reg.cf_expected")

        resultfile1 = full_test_prefix + "_POP1_PHENO1.res"
        resultfile2 = full_test_prefix + "_POP2_PHENO1.res"
        reg_ld_file_expected = os.path.join(TWO_POP_DIR, "ld_reg.cf_expected")
        reg_int_file_expected = os.path.join(TWO_POP_DIR, "int_reg.cf_expected")
        reg_se2_file_expected = os.path.join(TWO_POP_DIR, "se2_reg.cf_expected")

        # Determine the command that will be run
        run_cmd = [
            mama_py,
            '--sumstats', '%s,POP1,PHENO1' % POP1_PHENO1_FILE,
            '--ld-scores', POP1_POP2_LDSCORE_FILE,
            '--out', full_test_prefix,
            '--out-harmonized',
            '--out-reg-coef',
        ]

        # Run the process, make sure output is captured and process exit code is good
        cp = subprocess.run(run_cmd, capture_output=True, check=True)
        sout = cp.stdout.decode("utf-8")
        serr = cp.stderr.decode("utf-8")

        # Make sure various statements are in the log and were sent to stdout
        assert os.path.exists(logfile)
        with open(logfile) as logf:
            logtext = logf.read()

        assert "POP1,PHENO1" in logtext
        assert "POP1,PHENO1" in sout
        assert "POP2,PHENO1" not in logtext
        assert "POP2,PHENO1" not in sout

        assert "Number of SNPS in initial intersection of all sources: 32" in logtext
        assert "Number of SNPS in initial intersection of all sources: 32" in sout

        assert "Writing harmonized summary statistics to disk" in logtext
        assert "Writing harmonized summary statistics to disk" in sout

        assert "Harmonized POP1 PHENO1 mean chi squared:" not in logtext
        assert "Harmonized POP1 PHENO1 mean chi squared:" not in sout

        assert "SNPs to make omega positive semi-definite" not in logtext
        assert "SNPs to make omega positive semi-definite" not in sout

        assert "Skipping positive-semi-definiteness check of Omega" in logtext
        assert "Skipping positive-semi-definiteness check of Omega" in sout

        assert "SNPs due to non-positive-definiteness of sigma" in logtext
        assert "SNPs due to non-positive-definiteness of sigma" in sout

        assert "total SNPs due to non-positive-(semi)-definiteness of omega / sigma" in logtext
        assert "total SNPs due to non-positive-(semi)-definiteness of omega / sigma" in sout

        assert "Mean Chi^2 for (\'POP1\', \'PHENO1\')" in logtext
        assert "Mean Chi^2 for (\'POP1\', \'PHENO1\')" in sout

        assert "%s_POP1_PHENO1.res" % full_test_prefix in logtext
        assert "%s_POP1_PHENO1.res" % full_test_prefix in sout
        assert "%s_POP2_PHENO1.res" % full_test_prefix not in logtext
        assert "%s_POP2_PHENO1.res" % full_test_prefix not in sout


        # Make sure the harmonized files exist and contain the right contents
        assert os.path.exists(harmfile1)
        assert not os.path.exists(harmfile2)

        harm1_df = pd.read_csv(harmfile1, sep=None, comment="#", engine="python")
        harm1_expected_df = pd.read_csv(POP1_PHENO1_FILE, sep=None, comment="#", engine="python")


        # Make sure the regression coefficient files exist and contain the right contents
        assert os.path.exists(reg_ld_file)
        assert os.path.exists(reg_int_file)
        assert os.path.exists(reg_se2_file)

        ld_coef_matrix = np.fromfile(reg_ld_file, sep='\t')
        int_coef_matrix = np.fromfile(reg_int_file, sep='\t')
        se2_coef_matrix = np.fromfile(reg_se2_file, sep='\t')

        ld_coef_matrix_expected = np.fromfile(reg_ld_file_expected, sep='\t')
        int_coef_matrix_expected = np.fromfile(reg_int_file_expected, sep='\t')
        se2_coef_matrix_expected = np.fromfile(reg_se2_file_expected, sep='\t')

        assert len(ld_coef_matrix) == 1
        assert len(int_coef_matrix) == 1
        assert len(se2_coef_matrix) == 1

        assert np.allclose(ld_coef_matrix, ld_coef_matrix_expected[0])
        assert np.allclose(int_coef_matrix, int_coef_matrix_expected[0])
        assert np.allclose(se2_coef_matrix, se2_coef_matrix_expected[0])

        # Make sure the result files exist and contain the right contents
        assert os.path.exists(resultfile1)
        assert not os.path.exists(resultfile2)

        res1_df = pd.read_csv(resultfile1, sep=None, comment="#", engine="python")
        res1_df_expected = pd.read_csv(POP1_PHENO1_FILE, sep=None, comment="#", engine="python")

        pd.testing.assert_series_equal(res1_df[ss.SNP_COL], res1_df_expected[ss.SNP_COL])
        pd.testing.assert_series_equal(res1_df[ss.CHR_COL], res1_df_expected[ss.CHR_COL])
        pd.testing.assert_series_equal(res1_df[ss.BP_COL], res1_df_expected[ss.BP_COL])
        pd.testing.assert_series_equal(res1_df[ss.A1_COL], res1_df_expected[ss.A1_COL])
        pd.testing.assert_series_equal(res1_df[ss.A2_COL], res1_df_expected[ss.A2_COL])
        pd.testing.assert_series_equal(res1_df[ss.FREQ_COL], res1_df_expected['FRQ'], check_names=False)
        pd.testing.assert_series_equal(res1_df[ss.BETA_COL], res1_df_expected[ss.BETA_COL], check_dtype=False)
        pd.testing.assert_series_equal(res1_df[mp.ORIGINAL_N_COL_RENAME], res1_df_expected[ss.N_COL], check_names=False)
        se_ratios = res1_df[ss.SE_COL] / res1_df_expected[ss.SE_COL]
        assert np.isclose(se_ratios.min(), se_ratios.max())
