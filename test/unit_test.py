"""
Unit tests of the mama2 software.  This should be run via pytest. TODO(jonbjala) - Need to break this up eventually
"""

import itertools
import os

import numpy as np
import pandas as pd
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

class TestRunLDScoreRegressions:

    #########
    def test__preset_inputs__expected_results(self):
        # TODO(jonbjala) Need some small testcases with precanned data (at least one > 2 pops)
        assert True


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


###########################################

# TODO(jonbjala) Create fixture to generate dataframes filled with arange (parameterized)

class TestFilterSumstats:

    __DATAFRAME_LENGTHS = [2, 3, 6, 10, 50]

    __DATAFRAME_SHAPES = list(itertools.permutations(__DATAFRAME_LENGTHS, 2))

    # TODO(jonbjala) Test filter that examines a column that's not present! / a filter that throws

    #########
    @pytest.mark.parametrize("df_shape", __DATAFRAME_SHAPES)
    def test__single_filt__return_expected_results(self, df_shape):

        # Fill DF with numbers 0..N-1 where there are N cells
        num_rows = df_shape[0]
        num_cols = df_shape[1]
        num_cells = num_rows * num_cols
        df = pd.DataFrame(np.arange(num_cells).reshape(df_shape))

        # Identify value whose row to filter out and the expected filtering indices
        target_int = num_cells // 2
        expected_indices = (df == target_int).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once and has the correct number of rows
        assert expected_indices.sum() == 1
        assert len(df) == num_rows

        # Filter the dataframe
        func_name = "Hello, world"
        result_indices, filt_map = mama2.filter_sumstats(df,
            {func_name : lambda df: (df == target_int).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 1
        assert func_name in filt_map

        #   Confirm dataframe does not contain the target int and has one fewer row
        assert len(df) == num_rows - 1
        assert (df == target_int).any(axis=None) == False
        assert all(result_indices == filt_map[func_name])
        assert all(result_indices == expected_indices)




    #########
    @pytest.mark.parametrize("df_shape", __DATAFRAME_SHAPES)
    def test__two_filt_no_overlap__return_expected_results(self, df_shape):

        # Fill DF with numbers 0..N-1 where there are N cells
        num_rows = df_shape[0]
        num_cols = df_shape[1]
        num_cells = num_rows * num_cols
        df = pd.DataFrame(np.arange(num_cells).reshape(df_shape))

        # Identify value whose row to filter out and the expected filtering indices
        target_int1 = 0
        target_int2 = num_cells - 1
        expected_indices1 = (df == target_int1).any(axis='columns')
        expected_indices2 = (df == target_int2).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once and has the correct number of rows
        assert expected_indices1.sum() == 1
        assert expected_indices2.sum() == 1
        assert len(df) == num_rows

        # Filter the dataframe
        func_name1 = "Hello"
        func_name2 = "World"
        result_indices, filt_map = mama2.filter_sumstats(df,
            {func_name1 : lambda df: (df == target_int1).any(axis='columns'),
             func_name2 : lambda df: (df == target_int2).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 2
        assert func_name1 in filt_map
        assert func_name2 in filt_map

        #   Confirm dataframe does not contain the target int and has one fewer row
        assert len(df) == num_rows - 2
        assert (df == target_int1).any(axis=None) == False
        assert (df == target_int1).any(axis=None) == False
        assert all(result_indices == filt_map[func_name1] | filt_map[func_name2])
        assert result_indices.sum() == 2


    #########
    @pytest.mark.parametrize("df_shape", __DATAFRAME_SHAPES)
    def test__one_filt_duplicated__same_as_just_once(self, df_shape):

        # Fill DF with numbers 0..N-1 where there are N cells
        num_rows = df_shape[0]
        num_cols = df_shape[1]
        num_cells = num_rows * num_cols
        df = pd.DataFrame(np.arange(num_cells).reshape(df_shape))

        # Identify value whose row to filter out and the expected filtering indices
        target_int = num_cells // 2
        expected_indices = (df == target_int).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once and has the correct number of rows
        assert expected_indices.sum() == 1
        assert len(df) == num_rows

        # Filter the dataframe
        func_name1 = "Hello, world"
        func_name2 = "Goodbye, everyone"
        result_indices, filt_map = mama2.filter_sumstats(df,
            {func_name1 : lambda df: (df == target_int).any(axis='columns'),
             func_name2 : lambda df: (df == target_int).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 2
        assert func_name1 in filt_map
        assert func_name2 in filt_map

        #   Confirm dataframe does not contain the target int and has one fewer row
        assert len(df) == num_rows - 1
        assert (df == target_int).any(axis=None) == False
        assert all(result_indices == filt_map[func_name1])
        assert all(result_indices == filt_map[func_name2])
        assert all(result_indices == expected_indices)


    #########
    @pytest.mark.parametrize("df_shape", __DATAFRAME_SHAPES)
    def test__np_filt_or_useless_filt__no_change(self, df_shape):

        # Fill DF with numbers 0..N-1 where there are N cells
        num_rows = df_shape[0]
        num_cols = df_shape[1]
        num_cells = num_rows * num_cols
        df = pd.DataFrame(np.arange(num_cells).reshape(df_shape))

        # BEFORE: Confirm dataframe copy contains the correct number of rows
        df_copy1 = df.copy()
        assert len(df_copy1) == num_rows

        # Filter the copy of the dataframe with no filter
        result_indices, filt_map = mama2.filter_sumstats(df_copy1, {})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 0

        #   Confirm dataframe is unchanged
        assert len(df_copy1) == num_rows
        assert df_copy1.equals(df)

        ######

        # BEFORE: Confirm dataframe copy contains the correct number of rows
        df_copy2 = df.copy()
        assert len(df_copy2) == num_rows

        # Filter the copy of the dataframe with useless filter
        func_name = "useless"
        result_indices, filt_map = mama2.filter_sumstats(df_copy1,
            {func_name : lambda df: (df == -1).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 1
        assert func_name in filt_map

        #   Confirm dataframe is unchanged
        assert len(df_copy2) == num_rows
        assert df_copy2.equals(df)
        assert not any(filt_map[func_name])
        assert not any(result_indices)


    #########
    def test__empty_dataframe__no_change(self):
        # Fill DF with numbers 0..N-1 where there are N cells
        df = pd.DataFrame()

        # BEFORE: Confirm dataframe copy contains the correct number of rows
        assert len(df) == 0

        # Filter the dataframe with useless filter
        func_name = "useless"
        result_indices, filt_map = mama2.filter_sumstats(df,
            {func_name : lambda df: (df == -1).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 1
        assert func_name in filt_map

        #   Confirm dataframe is unchanged
        assert len(df) == 0
        assert not any(filt_map[func_name])
        assert not any(result_indices)


###########################################


@pytest.fixture(scope="function")
def rename_test_df():

    # Control parameters
    upper_col_bound = 10  # Make sure this is at least 3, controls number of columns
    num_rows = 5

    # Derived parameters
    cols = [0] + list(range(upper_col_bound))
    num_cols = len(cols)
    num_cells = num_rows * num_cols

    # Create and return DataFrame
    return pd.DataFrame(np.arange(num_cells).reshape(num_rows, num_cols), columns=cols)


class TestRenameSumstatsCols:

    # Mappings for various conditions
    _EMPTY_MAP = {}  # Empty renaming
    _ONE_COL_MAP = {1 : -1}  # Map one column to new value
    _TWO_COL_MAP = {1 : -1, 2 : -2}  # Map two columns to new values
    _SWAP_MAP = {1 : 2, 2 : 1}  # Map two columns to each other

    _MISSING_KEY_MAP1 = {-1 : -2}  # One missing key
    _MISSING_KEY_MAP2 = {-1 : -2, -10 : -11}  # Two missing keys
    _MISSING_KEY_MAP3 = {1 : -1, -2 : -3}  # Mix missing and present keys

    _COLLISION_UNCHANGED_MAP = {1 : 0}  # Map to same as an unmapped column
    _COLLISION_RENAMED_MAP = {1 : -1, 2 : -1}  # Map two columns to same value


    #########
    @pytest.mark.parametrize("col_map", [_EMPTY_MAP, _ONE_COL_MAP, _TWO_COL_MAP, _SWAP_MAP])
    def test__happy_path__rename_columns__expected_results(self, col_map, rename_test_df):

        # Calculate expected results
        exp_cols = [col_map.get(col, col) for col in rename_test_df.columns.to_list()]

        # Rename the columns of the dataframe
        mama2.rename_sumstats_cols(rename_test_df, col_map)

        # Check the columns of the transformed dataframe
        assert all(c_act == c_exp for (c_act, c_exp) in
            zip(rename_test_df.columns.to_list(), exp_cols))


    #########
    @pytest.mark.parametrize("col_map", [_MISSING_KEY_MAP1, _MISSING_KEY_MAP2, _MISSING_KEY_MAP3])
    def test__missing_columns__throws_error(self, col_map, rename_test_df):

        # Rename the columns of the dataframe
        with pytest.raises(RuntimeError):
            mama2.rename_sumstats_cols(rename_test_df, col_map)


    #########
    @pytest.mark.parametrize("col_map", [_COLLISION_UNCHANGED_MAP, _COLLISION_RENAMED_MAP])
    def test__rename_collisions__throws_error(self, col_map, rename_test_df):

        # Rename the columns of the dataframe
        with pytest.raises(RuntimeError):
            mama2.rename_sumstats_cols(rename_test_df, col_map)

###########################################

class TestDetermineColumnMapping:

    _ORIG_COLS = ['Alice', 'Bob', 'Carla', 'Duke', 'Eve', 'Fred', 'Ginger']

    _RE_MAP_HAPPY_1 = {'Red' : r'al.+|XXX', 'Blue' : r'.o.', 'Green' : r'Hello|World|gINger'}
    _RE_MAP_HAPPY_2 = {'Red' : r'ABC.[0-9]+|Al.*|XXX\d', 'Blue' : r'.*b', 'Green' : r'Du.*'}

    _RE_MAP_1_TO_2 = {'Red' : r'Eve|Ginger', 'Blue' : r'.*la', 'Green' : r'Alice'}
    _RE_MAP_2_TO_1 = {'Red' : r'Car.*', 'Blue' : r'.*la', 'Green' : r'Alice'}

    _RE_MAP_MISSING = {'Red' : r'Al.*', 'Blue' : r'Bob', 'Green' : r'YYY'}

    #########
    @pytest.mark.parametrize("re_map", [_RE_MAP_HAPPY_1, _RE_MAP_HAPPY_2])
    def test__happy_path__expected_results(self, re_map):
        req_cols = list(re_map.keys())
        num_map_cols = len(req_cols)

        # All columns should match, so specifying varying required cols should all work
        for i in range(num_map_cols + 1):
            res = mama2.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS, re_map,
                                                 req_cols[:i])
            print("JJ: ", res)
            assert len(res) == num_map_cols
            res_vals = set(res.values())
            assert len(res.values()) == num_map_cols


    #########
    def test__map_one_col_to_two_std_cols__throw_error(self):
        with pytest.raises(RuntimeError):
            mama2.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
                TestDetermineColumnMapping._RE_MAP_1_TO_2)


    #########
    def test__map_two_cols_to_same_std_col__throw_error(self):
        with pytest.raises(RuntimeError):
            mama2.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
                TestDetermineColumnMapping._RE_MAP_2_TO_1)


    #########
    def test__req_col_not_matched__throw_error(self):
        # Without required columns, this should succeed and match everything but 1
        res = mama2.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
            TestDetermineColumnMapping._RE_MAP_MISSING)
        assert len(res) == len(TestDetermineColumnMapping._RE_MAP_MISSING) - 1

        # After requiring all columns in the map to be matched, should throw an error
        with pytest.raises(RuntimeError):
            mama2.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
                TestDetermineColumnMapping._RE_MAP_MISSING,
                list(TestDetermineColumnMapping._RE_MAP_MISSING.keys()))

    # TODO(jonbjala) Check for case when req_cols specifies std cols not in re_expr_map.keys()?


# TODO(jonbjala) Test qc_sumstats()

# TODO(jonbjala) Test filter functions at some point
