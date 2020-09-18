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

N_VAR_VALUES = [3, 4, 10]  # Some tests rely on these numbers being >= 2
N_PTS_VALUES = [10, 100, 1000]
N_TUPLES = [(i,j) for (i, j) in itertools.product(N_VAR_VALUES, N_PTS_VALUES) if i <= j]

###########################################

class TestRunRegression:

    #########
    @pytest.mark.parametrize("N_vars, N_pts", N_TUPLES)
    def test__linear_function_no_const_no_fixed__return_exact_coefficients(self, N_vars, N_pts):

        # Randomize the independent values and the expected coefficients
        indep_var_values = np.random.rand(N_pts, N_vars)
        coefs = np.random.rand(N_vars)

        # Calculate the dependent values
        dep_var = np.dot(indep_var_values, coefs)

        # Call the regression function
        result1 = mama2.run_regression(dep_var, indep_var_values)
        result2 = mama2.run_regression(dep_var, indep_var_values, np.full(N_vars, np.NaN))

        # Assert that the solution values match the expected / original coefficients
        assert len(result1) == N_vars
        assert np.allclose(result1, coefs)
        assert len(result2) == N_vars
        assert np.allclose(result2, coefs)


    #########
    @pytest.mark.parametrize("N_vars, N_pts", N_TUPLES)
    def test__linear_function_with_const_no_fixed__return_exact_coefficients(self, N_vars, N_pts):

        # Randomize the independent values and the expected coefficients, make first column constant
        indep_var_values = np.random.rand(N_pts, N_vars)
        indep_var_values[:, 0] = np.ones(N_pts)
        coefs = np.random.rand(N_vars)

        # Calculate the dependent values
        dep_var = np.dot(indep_var_values, coefs)

        # Call the regression function
        result1 = mama2.run_regression(dep_var, indep_var_values)
        result2 = mama2.run_regression(dep_var, indep_var_values, np.full(N_vars, np.NaN))

        # Assert that the solution values match the expected / original coefficients
        assert len(result1) == N_vars
        assert np.allclose(result1, coefs)
        assert len(result2) == N_vars
        assert np.allclose(result2, coefs)


    #########
    @pytest.mark.parametrize("N_vars, N_pts", N_TUPLES)
    def test__linear_function_no_const_with_fixed__return_exact_coefficients(self, N_vars, N_pts):

        # Randomize the independent values and the expected coefficients
        indep_var_values = np.random.rand(N_pts, N_vars)
        coefs = np.random.rand(N_vars)
        fixed_coefs = np.full(N_vars, np.NaN)

        # Fix every other coefficient
        fixed_count = 0
        for i in range(0, N_vars, 2):
            coefs[i] = 10.0 * float(i)
            fixed_coefs[i] = coefs[i]
            fixed_count += 1

        # Calculate the dependent values
        dep_var = np.dot(indep_var_values, coefs)

        # Call the regression function
        result = mama2.run_regression(dep_var, indep_var_values, fixed_coefs)

        # Assert that the solution values match the expected / original coefficients
        assert len(result) == N_vars
        assert np.allclose(result, coefs)


    #########
    @pytest.mark.parametrize("N_vars", N_VAR_VALUES)
    def test__linear_function_with_jitter__return_expected(self, N_vars):

        # Randomize the independent values and the expected coefficients
        N_pts = 10 ** 6 # Use a large value here to limit correlation between noise and jitter
        indep_var_values = np.random.rand(N_pts, N_vars)
        coefs = np.random.rand(N_vars)

        # Calculate the dependent values and jitter them
        dep_var = np.dot(indep_var_values, coefs)
        dep_var[::2] += 0.5
        dep_var[1::2] -= 0.5

        # Call the regression function
        result1 = mama2.run_regression(dep_var, indep_var_values)
        result2 = mama2.run_regression(dep_var, indep_var_values, np.full(N_vars, np.NaN))

        # Assert that the solution values match the expected / original coefficients
        assert len(result1) == N_vars
        assert np.allclose(result1, coefs, atol=0.02, rtol=0.05)
        assert len(result2) == N_vars
        assert np.allclose(result2, coefs, atol=0.02, rtol=0.05)


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


###########################################

class TestRunMamaMethod:

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
