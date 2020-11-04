"""
System tests for mama_pipeline.py.  This should be run via pytest.
"""

import os
import sys
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_dir = os.path.abspath(os.path.join(main_dir, 'test'))
data_dir = os.path.abspath(os.path.join(test_dir, 'data'))
sys.path.append(main_dir)

import logging

import numpy as np
import pandas as pd
import pytest

import mama2.mama_pipeline as mp
import mama2.reg_mama as mr
import mama2.util.sumstats as ss


#===================================================================================================
class TestRunLdScoreRegressions:

    LD_POP1 = np.array([1.0, 2.0, 3.0, 4.0])
    LD_POP2 = np.array([1.0, 1.0, 2.0, 3.0])
    LD_POP3 = np.array([1.0, 1.0, 1.0, 2.0])
    LDS = np.column_stack((LD_POP1, LD_POP2, LD_POP3))

    LD_SCORES = np.sqrt(np.apply_along_axis(lambda x: np.outer(x, x), axis=1, arr=LDS))

    SE_POP1 = np.sqrt(np.array([1, 2, 2, 2]))
    SE_POP2 = np.sqrt(np.array([1, 2, 2, 2]))
    SE_POP3 = np.sqrt(np.array([1, 2, 2, 2]))
    SES = np.column_stack((SE_POP1, SE_POP2, SE_POP3))


    BETA_POP1 = np.sqrt(np.array([1, 2, 2, 2]))
    BETA_POP2 = np.sqrt(np.array([1, 2, 2, 2]))
    BETA_POP3 = np.sqrt(np.array([1, 2, 2, 2]))
    BETAS = np.sqrt(LDS + np.square(SES) + 1)

    #########
    def test__testdata1__expected_results(self):

        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(
            TestRunLdScoreRegressions.BETAS, TestRunLdScoreRegressions.SES,
            TestRunLdScoreRegressions.LD_SCORES)
        print("JJ: ld_coef\n", ld_coef)
        print("JJ: int_coef\n", int_coef)
        print("JJ: se2_coef\n", se2_coef)
        print("\n\n\n")
        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(
            TestRunLdScoreRegressions.BETAS, TestRunLdScoreRegressions.SES,
            TestRunLdScoreRegressions.LD_SCORES, se_prod_fixed_opt=mr.MAMA_REG_OPT_ALL_ZERO)
        print("JJ: zero SE2 ld_coef\n", ld_coef)
        print("JJ: zero SE2 int_coef\n", int_coef)
        print("JJ: zero SE2 se2_coef\n", se2_coef)
        print("\n\n\n")
        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(
            TestRunLdScoreRegressions.BETAS, TestRunLdScoreRegressions.SES,
            TestRunLdScoreRegressions.LD_SCORES, int_fixed_opt=mr.MAMA_REG_OPT_ALL_ZERO)
        print("JJ: zero intercept ld_coef\n", ld_coef)
        print("JJ: zero intercept int_coef\n", int_coef)
        print("JJ: zero intercept se2_coef\n", se2_coef)
        print("\n\n\n")
        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(
            TestRunLdScoreRegressions.BETAS, TestRunLdScoreRegressions.SES,
            TestRunLdScoreRegressions.LD_SCORES, se_prod_fixed_opt=mr.MAMA_REG_OPT_IDENT)
        print("JJ: identity se2 ld_coef\n", ld_coef)
        print("JJ: identity se2 int_coef\n", int_coef)
        print("JJ: identity se2 se2_coef\n", se2_coef)
        print("\n\n\n")
        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(
            TestRunLdScoreRegressions.BETAS, TestRunLdScoreRegressions.SES,
            TestRunLdScoreRegressions.LD_SCORES, ld_fixed_opt=mr.MAMA_REG_OPT_PERF_CORR)
        print("JJ: perf corr ld_coef\n", ld_coef)
        print("JJ: perf corr int_coef\n", int_coef)
        print("JJ: perf corr se2_coef\n", se2_coef)
        # TODO(jonbjala) Finish / actually write this test
        assert True
