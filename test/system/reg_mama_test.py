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


PRE_SPECIFIED = "prespec"

LD_COEF_OPTS = [mr.MAMA_REG_OPT_ALL_FREE, mr.MAMA_REG_OPT_SET_CORR, PRE_SPECIFIED]
@pytest.fixture(scope="function", params=LD_COEF_OPTS)
def ldc_opt(request):
    return request.param


INT_COEF_OPTS = [mr.MAMA_REG_OPT_ALL_FREE, mr.MAMA_REG_OPT_ALL_ZERO, mr.MAMA_REG_OPT_OFFDIAG_ZERO,
                 PRE_SPECIFIED]
@pytest.fixture(scope="function", params=INT_COEF_OPTS)
def intc_opt(request):
    return request.param


SE2_COEF_OPTS = [mr.MAMA_REG_OPT_ALL_FREE, mr.MAMA_REG_OPT_ALL_ZERO, mr.MAMA_REG_OPT_IDENT,
                 mr.MAMA_REG_OPT_OFFDIAG_ZERO, PRE_SPECIFIED]
@pytest.fixture(scope="function", params=SE2_COEF_OPTS)
def se2c_opt(request):
    return request.param


#===================================================================================================
class TestRunLdScoreRegressions:

    PS_BASE = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    PS_LD = 1.0 * PS_BASE
    PS_INT = 2.0 * PS_BASE
    PS_SE2 = 3.0 * PS_BASE

    ########
    def test__3_pops_varying_opts__expected_results(self, ldc_opt, intc_opt, se2c_opt):

        # Create betas, standard errors, and ld scores
        betas = 100.0 * np.sqrt(TestRunLdScoreRegressions.PS_BASE) + 10.0
        ses = 0.1 * TestRunLdScoreRegressions.PS_BASE + 0.05
        ld_scores = np.array([1.00 * TestRunLdScoreRegressions.PS_BASE,
                              1.10 * TestRunLdScoreRegressions.PS_BASE,
                              1.15 * TestRunLdScoreRegressions.PS_BASE])


        # If any precanned coefficient matrices are to be specified, do that
        ld_opt = TestRunLdScoreRegressions.PS_LD if ldc_opt == PRE_SPECIFIED else ldc_opt
        int_opt = TestRunLdScoreRegressions.PS_INT if intc_opt == PRE_SPECIFIED else intc_opt
        se2_opt = TestRunLdScoreRegressions.PS_SE2 if se2c_opt == PRE_SPECIFIED else se2c_opt
        ld_scale = 0.7

        # Call the regression function
        f_opts = {mr.REG_LD_OPT_NAME : ld_opt,
                      mr.REG_INT_OPT_NAME : int_opt,
                      mr.REG_SE_OPT_NAME : se2_opt,
                      mr.REG_LD_SCALE_FACTOR_NAME : ld_scale}
        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(betas, ses, ld_scores, **f_opts)

        # Check to make sure coefficient matrices satisfy the selected options
        if ldc_opt == PRE_SPECIFIED:
            assert np.allclose(ld_coef, ld_opt)
        elif ldc_opt == mr.MAMA_REG_OPT_SET_CORR:
            for i in range(1, 3):
                for j in range(i+1, 3):
                    assert np.isclose(ld_coef[i, j],
                                      ld_scale * np.sqrt(ld_coef[i, i] * ld_coef[j, j]))

        if intc_opt == PRE_SPECIFIED:
            assert np.allclose(int_coef, int_opt)
        elif intc_opt == mr.MAMA_REG_OPT_ALL_ZERO:
            assert np.allclose(int_coef, 0.0)
        elif intc_opt == mr.MAMA_REG_OPT_OFFDIAG_ZERO:
            assert np.allclose(np.diag(np.diag(int_coef)), int_coef)

        if se2c_opt == PRE_SPECIFIED:
            assert np.allclose(se2_coef, se2_opt)
        elif se2c_opt == mr.MAMA_REG_OPT_ALL_ZERO:
            assert np.allclose(se2_coef, 0.0)
        elif se2c_opt == mr.MAMA_REG_OPT_IDENT:
            assert np.allclose(se2_coef, np.identity(3))
        elif se2c_opt == mr.MAMA_REG_OPT_OFFDIAG_ZERO:
            assert np.allclose(np.diag(np.diag(se2_coef)), se2_coef)


        # Check to make sure the coefficient matrices are symmetric
        assert np.allclose(ld_coef, ld_coef.T)
        assert np.allclose(int_coef, int_coef.T)
        assert np.allclose(se2_coef, se2_coef.T)

        # Check regression is satisfied in the case when all matrices are free / unconstrained
        if all([ldc_opt == mr.MAMA_REG_OPT_ALL_FREE,
                intc_opt == mr.MAMA_REG_OPT_ALL_FREE,
                se2c_opt == mr.MAMA_REG_OPT_ALL_FREE]):
            for i in range(3):
                for j in range(i, 3):
                    for snp in range(3):
                        assert np.isclose(ld_coef[i, j] * ld_scores[snp, i, j] +
                                          int_coef[i, j] * 1.0 +
                                          se2_coef[i, j] * ses[snp, i] * ses[snp, j],
                                          betas[snp, i] * betas[snp, j])


    #########
    def test__one_pop_precanned_data__expected_results(self):
        # Run for one population (mostly tests weighting)
        ld_scores = np.array([1.0, 4.0, 9.0]).reshape((3,1,1))
        ses = np.sqrt(np.array([2.0, 4.0, 6.0])).reshape((3,1))
        betas = np.array([1.0, 2.0, 3.0]).reshape((3,1))

        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(betas, ses, ld_scores)

        assert np.isclose(ld_coef, 1.0)
        assert np.isclose(int_coef, 0.0)
        assert np.isclose(se2_coef, 0.0)


    #########
    def test__two_pop_precanned_data__expected_results(self):
        # Run for two populations (further weighting tests)
        ld_scores = np.array([[[1.0, 0.0], [0.0, 4.0]],
                              [[9.0, 1.0], [1.0, 16.0]],
                              [[25.0, 100.0], [100.0, 36.0]]])
        ses = np.array([[4.0, 8.0],
                        [7.0, 4.0],
                        [10.0, 0.0]])
        betas = np.array([[1.0, 1.0],
                          [0.0, 2.0],
                          [1.0, 3.0]])

        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(betas, ses, ld_scores)

        expected_ld_coef = np.array([[0.7, 5.0 / 46.0], [5.0 / 46.0, 0.25]])
        expected_int_coef = np.array([[3.5, -181.0 / 23.0], [-181.0 / 23.0, 0.0]])
        expected_se2_coef = np.array([[-0.2, 51.0 / 184.0], [51.0 / 184.0, 0.0]])

        assert np.allclose(ld_coef, expected_ld_coef)
        assert np.allclose(int_coef, expected_int_coef)
        assert np.allclose(se2_coef, expected_se2_coef)

   ########
    @pytest.mark.parametrize("ld_scale_factor",
                             [-1.0, -0.5, -0.3, 0.0, 0.3, 0.5, 1.0, None, None])
    def test__3_pops_set_corr__expected_results(self, ld_scale_factor):

        scale_factor = ld_scale_factor if ld_scale_factor else np.random.rand()

        # Create betas, standard errors, and ld scores
        betas = 100.0 * np.sqrt(TestRunLdScoreRegressions.PS_BASE) + 10.0
        ses = 0.1 * TestRunLdScoreRegressions.PS_BASE + 0.05
        ld_scores = np.array([1.00 * TestRunLdScoreRegressions.PS_BASE,
                              1.10 * TestRunLdScoreRegressions.PS_BASE,
                              1.15 * TestRunLdScoreRegressions.PS_BASE])

        f_opts = {mr.REG_LD_OPT_NAME : mr.MAMA_REG_OPT_SET_CORR,
                  mr.REG_INT_OPT_NAME : mr.MAMA_REG_OPT_ALL_FREE,
                  mr.REG_SE_OPT_NAME : mr.MAMA_REG_OPT_ALL_FREE,
                  mr.REG_LD_SCALE_FACTOR_NAME : scale_factor}
        ld_coef, int_coef, se2_coef = mp.run_ldscore_regressions(betas, ses, ld_scores, **f_opts)


        for i in range(1, 3):
            for j in range(i+1, 3):
                assert np.isclose(ld_coef[i, j],
                                  scale_factor * np.sqrt(ld_coef[i, i] * ld_coef[j, j]))
