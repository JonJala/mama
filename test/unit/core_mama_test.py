"""
Unit tests for core_mama.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import itertools

import numpy as np
import pandas as pd
import pytest

import mama2.core_mama as core_mama


M_VALUES = [1, 3, 500]
P_VALUES = [1, 2, 5]


###########################################

class TestCreateSigmaMatrix:

    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__all_zero_matrices__zero_result(self, M, P):
        input_ses = np.zeros(shape=(M, P))
        input_se2_coefs = np.zeros(shape=(P, P))
        input_const_coefs = np.zeros(shape=(P, P))

        res = core_mama.create_sigma_matrix(input_ses, input_se2_coefs, input_const_coefs)

        assert res.shape == (M, P, P)
        assert np.allclose(res, 0.0)


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__zero_sematrices__result_is_constcoef(self, M, P):

        rand_mat = np.random.rand(P, P)

        input_ses = np.zeros(shape=(M, P))
        input_se2_coefs = np.zeros(shape=(P, P))
        input_const_coefs = 0.5 * (rand_mat + rand_mat.T)

        res = core_mama.create_sigma_matrix(input_ses, input_se2_coefs, input_const_coefs)

        assert res.shape == (M, P, P)
        assert all([np.allclose(res[slicenum], input_const_coefs) for slicenum in range(M)])


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__zero_constcoef_ses_ones__resultdiag_is_se2coefdiag(self, M, P):

        rand_mat = np.random.rand(P, P)

        input_ses = np.ones(shape=(M, P))
        input_se2_coefs = 0.5 * (rand_mat + rand_mat.T)
        input_const_coefs = np.zeros(shape=(P, P))

        res = core_mama.create_sigma_matrix(input_ses, input_se2_coefs, input_const_coefs)

        se2_diag = np.diag(input_se2_coefs)
        assert res.shape == (M, P, P)
        assert all([np.allclose(np.diag(res[slicenum]), se2_diag) for slicenum in range(M)])


    #########
    @pytest.mark.parametrize("ses, se2_coefs, const_coefs, expected_sigma",
        [
        (
            np.array([[1, -1, 0.5], [0, 0.5, 1], [0.5, 1, -1], [0, 0, 0]], dtype=np.float64),
            np.array([[1, 1, 0], [1, -1, 1], [0, 1, 0.5]], dtype=np.float64),
            np.array([[1, -1, 0], [-1, 1, -1], [0, -1, 1]], dtype=np.float64),
            np.array([[[2, -1, 0], [-1, 0, -1], [0, -1, 1.125]],
                      [[1, -1, 0], [-1, 0.75, -1], [0, -1, 1.5]],
                      [[1.25, -1, 0], [-1, 0, -1], [0, -1, 1.5]],
                      [[1, -1, 0], [-1, 1, -1], [0, -1, 1]]], dtype=np.float64)
        )
        ]
    )
    def test__set_small_inputs__expected_result(self, ses, se2_coefs,
        const_coefs, expected_sigma):
        M = ses.shape[0]
        P = ses.shape[1]

        res = core_mama.create_sigma_matrix(ses, se2_coefs, const_coefs)

        assert res.shape == (M, P, P)
        assert np.allclose(res, expected_sigma)


###########################################

class TestCreateOmegaMatrix:

    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__all_zero_matrices__zero_result(self, M, P):
        input_ldscores = np.zeros(shape=(M, P, P))
        input_ldscore_coefs = np.zeros(shape=(P, P))

        res = core_mama.create_omega_matrix(input_ldscores, input_ldscore_coefs)

        assert res.shape == (M, P, P)
        assert np.allclose(res, 0.0)


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__ldscore_coefs_identity_matrix__result_diag_equals_ldscores_diag(self, M, P):

        rand_mat = np.random.rand(M, P, P)

        input_ldscores = np.zeros(shape=(M, P, P))
        for slicenum in range(M):
            input_ldscores[slicenum] += 0.5 * (rand_mat[slicenum] + rand_mat[slicenum].T)
        input_ldscore_coefs = np.identity(P)

        res = core_mama.create_omega_matrix(input_ldscores, input_ldscore_coefs)

        assert res.shape == input_ldscores.shape
        for slicenum in range(M):
            for i in range(P):
                for j in range(P):
                    expected = input_ldscores[slicenum, i, j] if i == j else 0.0 
                    assert np.isclose(res[slicenum, i, j], expected)


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__all_ldscore_slices_identity_matrices__result_slice_diags_are_coef_diags(self, M, P):

        rand_mat = np.random.rand(P, P)

        input_ldscores = np.full(shape=(M, P, P), fill_value=np.identity(P))
        input_ldscore_coefs = 0.5 * (rand_mat + rand_mat.T)

        res = core_mama.create_omega_matrix(input_ldscores, input_ldscore_coefs)

        assert res.shape == input_ldscores.shape
        for slicenum in range(M):
            for i in range(P):
                for j in range(P):
                    expected = input_ldscore_coefs[i, j] if i == j else 0.0 
                    assert np.isclose(res[slicenum, i, j], expected)


    #########
    @pytest.mark.parametrize("ld_scores, ld_score_coefs, expected_omega",
        [
        (
            np.array([[[1, 1, 1], [1, 0, -1], [1, -1, -1]],
                      [[2, 2, 2], [2, 0, -2], [2, -2, -2]],
                      [[3, 3, 3], [3, 0, -3], [3, -3, -3]],
                      [[1, 1, 1], [-1, -1, -1], [1, 1, 1]]], dtype=np.float64),
            np.array([[1, 2, 3], [2, -1, 2], [3, 2, 0]], dtype=np.float64),
            np.array([[[1, 2, 3], [2, 0, -2], [3, -2, 0]],
                      [[2, 4, 6], [4, 0, -4], [6, -4, 0]],
                      [[3, 6, 9], [6, 0, -6], [9, -6, 0]],
                      [[1, 2, 3], [-2, 1, -2], [3, 2, 0]]], dtype=np.float64)
        )
        ]
    )
    def test__set_small_inputs__expected_result(self, ld_scores, ld_score_coefs, expected_omega):

        res = core_mama.create_omega_matrix(ld_scores, ld_score_coefs)

        assert res.shape == ld_scores.shape
        assert np.allclose(res, expected_omega)


###########################################

class TestRunMamaMethod:

    # TODO(jonbjala) Test this!!
    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__omega_with_some_negative_or_zero_diag__drop_corresponding_snps(self, M, P):
        assert True


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__happy_path_precanned_data__expected_results(self, M, P):
        assert True

    # def test_1(self):

    #     omega = np.ones((2,3,3)) # 2 SNPs, 3 populations
    #     sigma = np.zeros((2,3,3)) # 2 SNPs, 3 populations
    #     sigma[0] = np.identity(3)
    #     sigma[1] = np.identity(3)
    #     betas = np.array([[0, 1, 2], [-1, 0, 1]])
    #     mama2.run_mama_method(betas, omega, sigma)
    #     assert True

    # def test_2(self):

    #     omega = np.array([[[2.0, np.sqrt(2)], [np.sqrt(2), 1]]]) # 1 SNPs, 2 populations
    #     sigma = np.array([[[4, 0], [0, 1]]]) # 1 SNPs, 2 populations
    #     betas = np.array([[2, 1]])
    #     mama2.run_mama_method(betas, omega, sigma)
    #     assert True


    # def test_3(self):
    #     M = 3
    #     P = 4
    #     d_indices = np.arange(P)

    #     omega = np.zeros((M, P, P))
    #     sigma = np.zeros((M, P, P))

    #     rand_mat_1 = np.random.rand(M, P)
    #     rand_mat_2 = np.random.rand(M, P)

    #     omega[:, d_indices, d_indices] = np.random.rand(M, P)
    #     sigma[:, d_indices, d_indices] = np.random.rand(M, P)

    #     betas = np.ones((M,P))
    #     mama2.run_mama_method(betas, omega, sigma)
    #     assert True

    # def test_4(self):
    #     # 4 SNPs, 2 populations
    #     betas = np.array([[1,2], [3,4], [5,6], [7,8]])
    #     ses = np.array([[1,-1], [-2,2], [4,0], [0.5, 1]])
    #     ldscores = np.array([[1,-1], [-2,2], [4,0], [1,1]])
    #     mama2.run_ldscore_regressions(betas, ses, ldscores)
    #     assert False


###########################################

class TestQcSigma:

    #########
    @pytest.mark.parametrize("sigma, expected",
        [
            (
                [[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[3, 0], [0, 3]]],
                [True, True, True]
            ),
            (
                [[[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
                 [[-2, -1, 0], [-1, -2, -1], [0, -1, -2]],
                 [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                [True, False, True]
            )
        ])
    # TODO(jonbjala) Add a few more cases?
    def test__varying_sigma_slices__return_expected(self, sigma, expected):
        sigma_arr = np.array(sigma)
        expected_arr = np.array(expected)

        result_arr = core_mama.qc_sigma(sigma_arr)

        assert np.array_equal(result_arr, expected_arr)


###########################################

class TestQcOmega:

    #########
    @pytest.mark.parametrize("omega, exp_psd, exp_tweaks",
        [
            (
                [[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[3, 0], [0, 3]]],
                [True, True, True],
                [False, False, False]
            ),
            (
                [[[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
                 [[-2, -1, 0], [-1, -2, -1], [0, -1, -2]],
                 [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[2, 2, 2], [2, 1, 3], [2, 3, 2]]],
                [True, False, True, True],
                [False, False, False, True]
            )
        ])
    # TODO(jonbjala) Add a few more cases?
    def test__varying_omega_slices__return_expected(self, omega, exp_psd, exp_tweaks):
        omega_arr = np.array(omega)
        expected_pos_semi_def_arr = np.array(exp_psd)
        expected_tweaked = np.array(exp_tweaks)

        pos_semi_def_arr, tweaked_arr = core_mama.qc_omega(omega_arr)

        assert np.array_equal(pos_semi_def_arr, expected_pos_semi_def_arr)
        assert np.array_equal(tweaked_arr, expected_tweaked)


###########################################

class TestTweakOmega:

    #########
    @pytest.mark.parametrize("omega",
        [
            [[2, 2, 2], [2, 1, 3], [2, 3, 2]]
        ])
    # TODO(jonbjala) Add timeout to this test?  Also, add more test cases / parameters
    def test__varying_omega__return_expected(self, omega):
        omega_matrix = np.array(omega)
        result = core_mama.tweak_omega(omega_matrix)
        assert np.all(np.linalg.eigvalsh(result) >= 0.0)
