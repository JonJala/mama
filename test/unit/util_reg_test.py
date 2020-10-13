"""
Unit tests util/reg.py.  This should be run via pytest.
"""

import itertools
import os
import tempfile

import numpy as np
import pytest

import mama2.util.reg as reg


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
        result1 = reg.run_regression(dep_var, indep_var_values)
        result2 = reg.run_regression(dep_var, indep_var_values, np.full(N_vars, np.NaN))

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
        result1 = reg.run_regression(dep_var, indep_var_values)
        result2 = reg.run_regression(dep_var, indep_var_values, np.full(N_vars, np.NaN))

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
        result = reg.run_regression(dep_var, indep_var_values, fixed_coefs)

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
        result1 = reg.run_regression(dep_var, indep_var_values)
        result2 = reg.run_regression(dep_var, indep_var_values, np.full(N_vars, np.NaN))

        # Assert that the solution values match the expected / original coefficients
        assert len(result1) == N_vars
        assert np.allclose(result1, coefs, atol=0.02, rtol=0.05)
        assert len(result2) == N_vars
        assert np.allclose(result2, coefs, atol=0.02, rtol=0.05)
