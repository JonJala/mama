"""
Unit tests of the mama2 software.  This should be run via pytest. TODO(jonbjala) - Need to break this up eventually
"""

import itertools
import os

import numpy as np
import pytest

import mama2.mama2 as mama2


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

        res = mama2.create_sigma_matrix(input_ses, input_se2_coefs, input_const_coefs)

        assert res.shape == (M, P, P)
        assert np.allclose(res, 0.0)


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__zero_sematrices__result_is_constcoef(self, M, P):

        rand_mat = np.random.rand(P, P)

        input_ses = np.zeros(shape=(M, P))
        input_se2_coefs = np.zeros(shape=(P, P))
        input_const_coefs = 0.5 * (rand_mat + rand_mat.T)

        res = mama2.create_sigma_matrix(input_ses, input_se2_coefs, input_const_coefs)

        assert res.shape == (M, P, P)
        assert all([np.allclose(res[slicenum], input_const_coefs) for slicenum in range(M)])


    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__zero_constcoef_ses_ones__resultdiag_is_se2coefdiag(self, M, P):

        rand_mat = np.random.rand(P, P)

        input_ses = np.ones(shape=(M, P))
        input_se2_coefs = 0.5 * (rand_mat + rand_mat.T)
        input_const_coefs = np.zeros(shape=(P, P))

        res = mama2.create_sigma_matrix(input_ses, input_se2_coefs, input_const_coefs)

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

        res = mama2.create_sigma_matrix(ses, se2_coefs, const_coefs)

        assert res.shape == (M, P, P)
        assert np.allclose(res, expected_sigma)


###########################################

class TestCreateOmegaMatrix:

    #########
    @pytest.mark.parametrize("M, P", itertools.product(M_VALUES, P_VALUES))
    def test__all_zero_matrices__zero_result(self, M, P):
        input_ldscores = np.zeros(shape=(M, P, P))
        input_ldscore_coefs = np.zeros(shape=(P, P))

        res = mama2.create_omega_matrix(input_ldscores, input_ldscore_coefs)

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

        res = mama2.create_omega_matrix(input_ldscores, input_ldscore_coefs)

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

        res = mama2.create_omega_matrix(input_ldscores, input_ldscore_coefs)

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

        res = mama2.create_omega_matrix(ld_scores, ld_score_coefs)

        assert res.shape == ld_scores.shape
        assert np.allclose(res, expected_omega)
