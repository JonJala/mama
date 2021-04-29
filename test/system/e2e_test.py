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


POP1_PHENO1_FILE = os.path.abspath(os.path.join(data_dir, 'two_pop/pop1_pheno1_sumstats.txt'))
POP2_PHENO1_FILE = os.path.abspath(os.path.join(data_dir, 'two_pop/pop2_pheno1_sumstats.txt'))
POP1_POP2_LDSCORE_FILE = os.path.abspath(os.path.join(data_dir,
                             'two_pop/pop1_pop2_chr1.l2.ldscore'))

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
        reg_ld_file = full_test_prefix + "_ld_reg.cf"
        reg_int_file = full_test_prefix + "_int_reg.cf"
        reg_se2_file = full_test_prefix + "_se2_reg.cf"
        resultfile1 = full_test_prefix + "_POP1_PHENO1.res"
        resultfile2 = full_test_prefix + "_POP2_PHENO1.res"

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

        # TODO(jonbjala) Include checks against file contents


        # Make sure the regression coefficient files exist and contain the right contents
        assert os.path.exists(reg_ld_file)
        assert os.path.exists(reg_int_file)
        assert os.path.exists(reg_se2_file)

        # TODO(jonbjala) Include checks against file contents


        # Make sure the result files exist and contain the right contents
        assert os.path.exists(resultfile1)
        assert os.path.exists(resultfile2)

        # TODO(jonbjala) Include checks against file contents



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
        resultfile1 = full_test_prefix + "_POP1_PHENO1.res"
        resultfile2 = full_test_prefix + "_POP2_PHENO1.res"

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

        # TODO(jonbjala) Include checks against file contents


        # Make sure the regression coefficient files exist and contain the right contents
        assert os.path.exists(reg_ld_file)
        assert os.path.exists(reg_int_file)
        assert os.path.exists(reg_se2_file)

        # TODO(jonbjala) Include checks against file contents


        # Make sure the result files exist and contain the right contents
        assert os.path.exists(resultfile1)
        assert not os.path.exists(resultfile2)

        # TODO(jonbjala) Include checks against file contents
