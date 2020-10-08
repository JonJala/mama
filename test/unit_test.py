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

# TODO(jonbjala) Check for contents of exception messages when possible: https://docs.pytest.org/en/stable/assert.html

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

_FILTER_TEST_DATAFRAME_LENGTHS = [2, 3, 6, 10, 50]
_FILTER_TEST_DATAFRAME_SHAPES = list(itertools.permutations(_FILTER_TEST_DATAFRAME_LENGTHS, 2))
@pytest.fixture(scope="function", params=_FILTER_TEST_DATAFRAME_SHAPES)
def runfilter_test_df(request):

    # Fill DF with numbers 0..N-1 where there are N cells
    num_rows = request.param[0]
    num_cols = request.param[1]
    num_cells = num_rows * num_cols

    return pd.DataFrame(np.arange(num_cells).reshape((num_rows, num_cols)))


class TestRunFilters:

    # TODO(jonbjala) Test filter that examines a column that's not present! / a filter that throws

    #########
    def test__single_filt__return_expected_results(self, runfilter_test_df):

        # Identify value whose row to filter out and the expected filtering indices
        target_int = runfilter_test_df.size // 2
        expected_indices = (runfilter_test_df == target_int).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once and has the correct number of rows
        assert expected_indices.sum() == 1

        # Filter the dataframe
        func_name = "Hello, world"
        result_indices, filt_map = mama2.run_filters(runfilter_test_df,
            {func_name : lambda df: (df == target_int).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 1
        assert func_name in filt_map

        #   Confirm result indices correct
        assert all(result_indices == filt_map[func_name])
        assert all(result_indices == expected_indices)




    #########
    def test__two_filt_no_overlap__return_expected_results(self, runfilter_test_df):

        # Identify value whose row to filter out and the expected filtering indices
        target_int1 = 0
        target_int2 = runfilter_test_df.size - 1
        expected_indices1 = (runfilter_test_df == target_int1).any(axis='columns')
        expected_indices2 = (runfilter_test_df == target_int2).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once
        assert expected_indices1.sum() == 1
        assert expected_indices2.sum() == 1

        # Filter the dataframe
        func_name1 = "Hello"
        func_name2 = "World"
        result_indices, filt_map = mama2.run_filters(runfilter_test_df,
            {func_name1 : lambda df: (df == target_int1).any(axis='columns'),
             func_name2 : lambda df: (df == target_int2).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 2
        assert func_name1 in filt_map
        assert func_name2 in filt_map

        #   Confirm result indices correct
        assert all(result_indices == filt_map[func_name1] | filt_map[func_name2])
        assert result_indices.sum() == 2


    #########
    def test__one_filt_duplicated__same_as_just_once(self, runfilter_test_df):

        # Identify value whose row to filter out and the expected filtering indices
        target_int = runfilter_test_df.size // 2
        expected_indices = (runfilter_test_df == target_int).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once
        assert expected_indices.sum() == 1

        # Filter the dataframe
        func_name1 = "Hello, world"
        func_name2 = "Goodbye, everyone"
        result_indices, filt_map = mama2.run_filters(runfilter_test_df,
            {func_name1 : lambda df: (df == target_int).any(axis='columns'),
             func_name2 : lambda df: (df == target_int).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 2
        assert func_name1 in filt_map
        assert func_name2 in filt_map

        #   Confirm result indices correct
        assert all(result_indices == filt_map[func_name1])
        assert all(result_indices == filt_map[func_name2])
        assert all(result_indices == expected_indices)


    #########
    def test__np_filt_or_useless_filt__no_change(self, runfilter_test_df):

        # BEFORE: Copy dataframe
        df_copy1 = runfilter_test_df.copy()

        # Filter the copy of the dataframe with no filter
        result_indices, filt_map = mama2.run_filters(df_copy1, {})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 0

        #   Confirm dataframe is unchanged
        assert df_copy1.equals(runfilter_test_df)

        ######

        # BEFORE: Copy dataframes
        df_copy2 = runfilter_test_df.copy()

        # Filter the copy of the dataframe with useless filter
        func_name = "useless"
        result_indices, filt_map = mama2.run_filters(df_copy1,
            {func_name : lambda df: (df == -1).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 1
        assert func_name in filt_map

        #   Confirm dataframe is unchanged
        assert df_copy2.equals(runfilter_test_df)
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
        result_indices, filt_map = mama2.run_filters(df,
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
                                                 set(req_cols[:i]))
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
                set(TestDetermineColumnMapping._RE_MAP_MISSING.keys()))

    # TODO(jonbjala) Check for case when req_cols specifies std cols not in re_expr_map.keys()?


# TODO(jonbjala) Test qc_sumstats()

# TODO(jonbjala) Test filter functions at some point


###########################################

class TestIntersectSnpList:

    #########
    @pytest.mark.parametrize("mults, expected_intersection_size",
    [
            ([2, 3, 5], 7),
            ([2, 4, 6, 8], 5),
            ([1, 1, 1], 100),
            ([], 50),
            ([5], 30)
    ])
    def test__varying_snps__expected_intersection(self, mults, expected_intersection_size):

        # Derived parameters
        lcm = np.lcm.reduce(mults) if mults else 1
        n_max = lcm * expected_intersection_size

        # Function to return a dataframe with SNP IDs spaced out by n_mult
        def get_df(n_mult):
            snp_list = [str(n_mult * snp_num) for snp_num in range(1, n_max + 1)]
            return pd.DataFrame(data={"dummy" : [1.0] * len(snp_list)}, index=snp_list)

        # Determine LD score and sumstat parameters
        ldscores = get_df(1)
        sumstats = {i : get_df(i) for i in mults}

        intersection = mama2.intersect_snp_lists(sumstats, ldscores)
        assert len(intersection) == expected_intersection_size


    def test__no_intersection__returns_empty_index(self):

        # Determine LD score and sumstat parameters
        ld_snps = [str(snp_num) for snp_num in range(1, 10)]
        ldscores = pd.DataFrame(data={"dummy" : [1.0] * len(ld_snps)}, index=ld_snps)

        # Ensure the sumstats all have SNPs that are outside of the LD score SNP list
        sumstat_snps = ld_snps = [str(snp_num) for snp_num in
                                  range(len(ld_snps) + 1, len(ld_snps) + 10)]
        sumstats = {i : pd.DataFrame(data={"dummy" : [1.0] * len(sumstat_snps)}, index=sumstat_snps)
                    for i in range(1, 5)}

        intersection = mama2.intersect_snp_lists(sumstats, ldscores)
        assert len(intersection) == 0


###########################################

class TestFlipAlleles:

    #########
    @pytest.mark.parametrize("flips",
    [
            ([True, True, True, False, False, True]),
            ([False, True, False, True, False, True]),
            ([False, False, True, False, True, False]),
            ([True, True, True, True, True, True]),
            ([False, False, False, False, False, False]),
            ([False, False, True, False, False, False])
    ])
    def test__varying_flips__expected_results(self, flips):
        snps = ['rs01', 'rs02', 'rs03', 'rs04', 'rs05', 'rs06']
        a1 = ['G', 'A', 'C', 'T', 'G', 'A']
        a2 = ['T', 'G', 'A', 'C', 'T', 'G']
        betas = [1.0, -2.0, 3.0, -4.0, 5.0, 0.0]
        freqs = [1.0, 0.5, 0.6, 0.4, 0.1, 0.0]
        data = {mama2.FREQ_COL:freqs, mama2.A1_COL:a1, mama2.A2_COL:a2, mama2.BETA_COL:betas}
        df = pd.DataFrame(data, index=snps)

        # Flip
        df_copy = df.copy()
        indices = pd.Series(flips, index=snps)
        mama2.flip_alleles(df_copy, indices)

        # Check to make sure the results are as expected
        freq_check = (indices & (df_copy[mama2.FREQ_COL] == 1.0 - df[mama2.FREQ_COL])) | \
                     (~indices & (df_copy[mama2.FREQ_COL] == df[mama2.FREQ_COL]))
        beta_check = (indices & (df_copy[mama2.BETA_COL] == -df[mama2.BETA_COL])) | \
                     (~indices & (df_copy[mama2.BETA_COL] == df[mama2.BETA_COL]))
        a1_check = (indices & (df_copy[mama2.A1_COL] == df[mama2.A2_COL])) | \
                   (~indices & (df_copy[mama2.A1_COL] == df[mama2.A1_COL]))
        a2_check = (indices & (df_copy[mama2.A2_COL] == df[mama2.A1_COL])) | \
                   (~indices & (df_copy[mama2.A2_COL] == df[mama2.A2_COL]))

        assert all(freq_check)
        assert all(beta_check)
        assert all(a1_check)
        assert all(a2_check)


###########################################

class TestStandardizeAllSumstats:


    _SNPS = ['rs01', 'rs02', 'rs03', 'rs04', 'rs05', 'rs06']
    _BETAS = [1.0, -2.0, 3.0, -4.0, 5.0, 0.0]
    _FREQS = [1.0, 0.5, 0.6, 0.4, 0.1, 0.0]

    _DATA_1 = {
        mama2.A1_COL : ['G', 'A', 'C', 'T', 'G', 'A'],
        mama2.A2_COL : ['T', 'G', 'A', 'C', 'T', 'G'],
        mama2.BETA_COL : _BETAS,
        mama2.FREQ_COL : _FREQS
    }

    _DATA_2 = {
        # Both A1 and A2 are complements of DATA_1
        mama2.A1_COL : ['C', 'T', 'G', 'A', 'C', 'T'],
        mama2.A2_COL : ['A', 'C', 'T', 'G', 'A', 'C'],
        mama2.BETA_COL : _BETAS,
        mama2.FREQ_COL : _FREQS
    }

    _DATA_3 = {
        mama2.A1_COL : ['G', 'A', 'C', 'T', 'G', 'A'],
        mama2.A2_COL : ['T', 'C', 'A', 'C', 'T', 'G'],  # Differs in 2nd spot from DATA_1
        mama2.BETA_COL : _BETAS,
        mama2.FREQ_COL : _FREQS
    }

    _DATA_4 = {
        mama2.A1_COL : ['G', 'T', 'G', 'T', 'G', 'A'],  # Differs in 2nd and 3rd spots from DATA_1
        mama2.A2_COL : ['T', 'G', 'A', 'C', 'T', 'G'],
        mama2.BETA_COL : _BETAS,
        mama2.FREQ_COL : _FREQS
    }

    _DATA_5 = {
        # The first is a reference allele flip, and the second is a complement allele flip
        mama2.A1_COL : ['T', 'C', 'C', 'T', 'G', 'A'],
        mama2.A2_COL : ['G', 'T', 'A', 'C', 'T', 'G'],
        mama2.BETA_COL : _BETAS,
        mama2.FREQ_COL : _FREQS
    }

    #########
    def test__complement_snps__all_match(self):

        # Tests two populations with major/minor alleles that are complements of each other
        # (no dropping or flipping needs to occur)

        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df2 = pd.DataFrame(TestStandardizeAllSumstats._DATA_2,
                             index=TestStandardizeAllSumstats._SNPS)
        orig_pops = {1 : df1, 2 : df2}

        # Test each population as the reference, including not specifying (should default to 1)
        dfs = [(1, df1), (2, df2), ()]
        for ref_tuple in dfs:
            df1_copy = df1.copy()
            df2_copy = df2.copy()
            pop_dict = {1 : df1_copy, 2 : df2_copy}

            ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                mama2.standardize_all_sumstats(pop_dict, ref_tuple)

            # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
            assert ref_popname == ref_tuple[0] if ref_tuple else 1
            assert ref_popname in pop_dict
            assert ref_popname in orig_pops
            ref_df = orig_pops[ref_popname]

            # No drops should be needed
            assert ~any(cumulative_drop_indices)

            # Make sure both pops are accounted for and both indicate no drops
            assert len(drop_dict) == 2
            assert ~any(drop_dict[1])
            assert ~any(drop_dict[2])

            # Make sure both pops are accounted for and both indicate no flips
            assert len(ref_flip_dict) == 2
            assert ~any(ref_flip_dict[1])
            assert ~any(ref_flip_dict[2])

            # Make sure columns still agree (aside from complements)
            assert all((df1_copy[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df1_copy[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))
            assert all((df2_copy[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df2_copy[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))


    #########
    def test__allele_flip_snps__expected_flips(self):

        # Tests two populations with major/minor alleles that are complements of each other
        # (no dropping or flipping needs to occur)

        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df5 = pd.DataFrame(TestStandardizeAllSumstats._DATA_5,
                             index=TestStandardizeAllSumstats._SNPS)
        orig_pops = {1 : df1, 5 : df5}

        # Test each population as the reference, including not specifying (should default to 1)
        dfs = [(1, df1), (5, df5), ()]
        for ref_tuple in dfs:
            df1_copy = df1.copy()
            df5_copy = df5.copy()
            pop_dict = {1 : df1_copy, 5 : df5_copy}

            ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                mama2.standardize_all_sumstats(pop_dict, ref_tuple)

            # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
            assert ref_popname == ref_tuple[0] if ref_tuple else 1
            assert ref_popname in pop_dict
            assert ref_popname in orig_pops
            ref_df = orig_pops[ref_popname]

            # No drops should be needed
            assert ~any(cumulative_drop_indices)

            # Make sure both pops are accounted for and both indicate no drops
            assert len(drop_dict) == 2
            assert ~any(drop_dict[1])
            assert ~any(drop_dict[5])

            # Make sure both pops are accounted for and one indicates two flips
            assert len(ref_flip_dict) == 2
            assert (((ref_flip_dict[1].sum() == 0) and (ref_flip_dict[5].sum() == 2)) or
                   ((ref_flip_dict[1].sum() == 2) and (ref_flip_dict[5].sum() == 0)))
            assert (ref_flip_dict[1] | ref_flip_dict[5]).sum() == 2

            # Make sure columns now agree (aside from complements)
            assert all((df1_copy[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df1_copy[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))
            assert all((df5_copy[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df5_copy[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))


    #########
    def test__mismatched_snp__expected_drop(self):

        # Tests two populations with a mismatched major/minor allele (unfixable)
        # so there should be one drop recommended

        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df3 = pd.DataFrame(TestStandardizeAllSumstats._DATA_3,
                             index=TestStandardizeAllSumstats._SNPS)
        orig_pops = {1 : df1, 3 : df3}

        # Test each population as the reference, including not specifying (should default to 1)
        dfs = [(1, df1), (3, df3), ()]
        for ref_tuple in dfs:
            df1_copy = df1.copy()
            df3_copy = df3.copy()
            pop_dict = {1 : df1_copy, 3 : df3_copy}

            ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                mama2.standardize_all_sumstats(pop_dict, ref_tuple)

            # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
            assert ref_popname == ref_tuple[0] if ref_tuple else 1
            assert ref_popname in pop_dict
            assert ref_popname in orig_pops
            ref_df = orig_pops[ref_popname]

            # No drops should be needed
            assert cumulative_drop_indices.sum() == 1
            assert cumulative_drop_indices[cumulative_drop_indices.index[1]]

            # Make sure both pops are accounted for and one drop is indicated
            assert len(drop_dict) == 2
            assert ((drop_dict[1].sum() == 0 and drop_dict[3].sum() == 1) or
                    (drop_dict[1].sum() == 1 and drop_dict[3].sum() == 0))

            # Make sure both pops are accounted for and no flips are recorded
            assert len(ref_flip_dict) == 2
            assert ~any(ref_flip_dict[1])
            assert ~any(ref_flip_dict[3])

            # Make sure columns now agree (aside from complements)
            assert all((df1_copy[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df1_copy[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))
            assert all((df3_copy[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df3_copy[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))


    #########
    def test__varying_snps__expected_results(self):
        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df2 = pd.DataFrame(TestStandardizeAllSumstats._DATA_2,
                           index=TestStandardizeAllSumstats._SNPS)
        df3 = pd.DataFrame(TestStandardizeAllSumstats._DATA_3,
                           index=TestStandardizeAllSumstats._SNPS)
        df4 = pd.DataFrame(TestStandardizeAllSumstats._DATA_4,
                           index=TestStandardizeAllSumstats._SNPS)
        df5 = pd.DataFrame(TestStandardizeAllSumstats._DATA_5,
                             index=TestStandardizeAllSumstats._SNPS)

        ref_df = df1.copy()

        pop_dict = {1 : df1, 2 : df2, 3 : df3, 4 : df4, 5 : df5}

        ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                mama2.standardize_all_sumstats(pop_dict)

        # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
        assert ref_popname == 1

        # Two drops should be indicated (in the second and third spots)
        assert cumulative_drop_indices.sum() == 2
        assert all(cumulative_drop_indices[cumulative_drop_indices.index[1:3]])

        # Make sure the drop breakdown is correct
        assert len(drop_dict) == 5
        assert ~any(drop_dict[1])
        assert ~any(drop_dict[2])
        assert drop_dict[3].sum() == 1
        assert drop_dict[4].sum() == 2
        assert (drop_dict[3] | drop_dict[4]).sum() == 2
        assert ~any(drop_dict[5])

        # Make sure reference allele flips were recorded correctly
        assert len(ref_flip_dict) == 5
        assert ~any(ref_flip_dict[1])
        assert ~any(ref_flip_dict[2])
        assert ~any(ref_flip_dict[3])
        assert ~any(ref_flip_dict[4])
        assert ref_flip_dict[5].sum() == 2

        for df in pop_dict.values():
            assert all((df[mama2.A1_COL] == ref_df[mama2.A1_COL]) |
                       (df[mama2.A1_COL].replace(mama2.COMPLEMENT) == ref_df[mama2.A1_COL]))


###########################################

@pytest.fixture(scope="function", params=[1, 2, 3, 10])
def collate_df_values_test_df(request):

    num_snps = request.param
    snps = [f'rs{s:03d}' for s in range(1, num_snps + 1)]

    return pd.DataFrame(index=snps, columns=[mama2.BETA_COL, mama2.SE_COL])

class TestCollateDfValues:

    #########
    @pytest.mark.parametrize("num_pops", [1, 2, 3, 5])
    def test__diff_ancestry_same_pheno__return_expected(self, collate_df_values_test_df, num_pops):

        num_snps = len(collate_df_values_test_df)
        pheno = "dummy_phen"

        # Create population data frames
        sumstats = {(i, pheno) : collate_df_values_test_df.copy() for i in range(num_pops)}
        for pop_df in sumstats.values():
            pop_df[mama2.BETA_COL] = np.random.rand(num_snps)
            pop_df[mama2.SE_COL] = np.random.rand(num_snps)
        pop_ids = list(sumstats.keys())

        # Create LD score data frames
        ld_cols = ["%s_%s" % (i[0],j[0]) for i in pop_ids for j in pop_ids if i[0] >= j[0]]
        ldscores = pd.DataFrame(index=collate_df_values_test_df.index, columns=ld_cols)
        for col in ld_cols:
            ldscores[col] = np.random.rand(num_snps)

        # Use default order, then specify (should be same as default), then reverse order
        beta_arr_1, se_arr_1, ld_arr_1 = mama2.collate_df_values(sumstats, ldscores)
        beta_arr_2, se_arr_2, ld_arr_2 = mama2.collate_df_values(sumstats, ldscores, pop_ids)
        beta_arr_3, se_arr_3, ld_arr_3 = mama2.collate_df_values(sumstats, ldscores, pop_ids[::-1])

        # Check the first return element-wise
        for pop_num in range(num_pops):
            df = sumstats[(pop_num, pheno)]
            assert np.allclose(df[mama2.BETA_COL], beta_arr_1[:, pop_num])
            assert np.allclose(df[mama2.SE_COL], se_arr_1[:, pop_num])
        for col in ldscores.columns:
            p_nums  = col.split("_")
            p1, p2 =  (int(p_nums[0]), int(p_nums[1]))
            assert np.allclose(ldscores[col], ld_arr_1[:, p1, p2])
            assert np.allclose(ldscores[col], ld_arr_1[:, p2, p1])

        # Now check the other two calls against the first (should be same, and reversed)
        assert np.allclose(beta_arr_1, beta_arr_2)
        assert np.allclose(se_arr_1, se_arr_2)
        assert np.allclose(ld_arr_1, ld_arr_2)

        assert np.allclose(beta_arr_1, beta_arr_3[:, ::-1])
        assert np.allclose(se_arr_1, se_arr_3[:, ::-1])
        assert np.allclose(ld_arr_1, ld_arr_3[:, ::-1, ::-1])


    #########
    @pytest.mark.parametrize("num_phen", [1, 2, 3, 5])
    def test__same_ancestry_diff_pheno__return_expected(self, collate_df_values_test_df, num_phen):

        num_snps = len(collate_df_values_test_df)
        phens = ["phen_%s" % i for i in range(num_phen)]

        # Create population data frames
        sumstats = {(0, pheno) : collate_df_values_test_df.copy() for pheno in phens}
        for pop_df in sumstats.values():
            pop_df[mama2.BETA_COL] = np.random.rand(num_snps)
            pop_df[mama2.SE_COL] = np.random.rand(num_snps)
        pop_ids = list(sumstats.keys())

        # Create LD score data frames
        ld_cols = ["0_0"]
        ldscores = pd.DataFrame(index=collate_df_values_test_df.index, columns=ld_cols)
        for col in ld_cols:
            ldscores[col] = np.random.rand(num_snps)

        # Use default order, then specify (should be same as default), then reverse order
        beta_arr_1, se_arr_1, ld_arr_1 = mama2.collate_df_values(sumstats, ldscores)
        beta_arr_2, se_arr_2, ld_arr_2 = mama2.collate_df_values(sumstats, ldscores, pop_ids)
        beta_arr_3, se_arr_3, ld_arr_3 = mama2.collate_df_values(sumstats, ldscores, pop_ids[::-1])

        # Check the first return element-wise
        for phen_num in range(num_phen):
            df = sumstats[(0, phens[phen_num])]
            assert np.allclose(df[mama2.BETA_COL], beta_arr_1[:, phen_num])
            assert np.allclose(df[mama2.SE_COL], se_arr_1[:, phen_num])
        for col in ldscores.columns:
            assert np.allclose(ldscores[col], ld_arr_1[:, 0, 0])
            assert np.allclose(ldscores[col], ld_arr_1[:, 0, 0])

        # Now check the other two calls against the first (should be same, and reversed)
        assert np.allclose(beta_arr_1, beta_arr_2)
        assert np.allclose(se_arr_1, se_arr_2)
        assert np.allclose(ld_arr_1, ld_arr_2)

        assert np.allclose(beta_arr_1, beta_arr_3[:, ::-1])
        assert np.allclose(se_arr_1, se_arr_3[:, ::-1])
        assert np.allclose(ld_arr_1, ld_arr_3[:, ::-1, ::-1])

    # TODO(jonbjala) Need to test a "mixed" case where there are multiple ancestries and phenotypes


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

        result_arr = mama2.qc_sigma(sigma_arr)

        assert np.array_equal(result_arr, expected_arr)


###########################################

class TestQcOmega:

    #########
    @pytest.mark.parametrize("omega, expected",
        [
            (
                [[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[3, 0], [0, 3]]],
                [True, True, True]
            ),
            (
                [[[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
                 [[-2, -1, 0], [-1, -2, -1], [0, -1, -2]],
                 [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                 [[2, 2, 2], [2, 1, 3], [2, 3, 2]]],
                [True, False, True, True]
            )
        ])
    # TODO(jonbjala) Add a few more cases?
    def test__varying_omega_slices__return_expected(self, omega, expected):
        omega_arr = np.array(omega)
        expected_arr = np.array(expected)

        result_arr = mama2.qc_omega(omega_arr)

        assert np.array_equal(result_arr, expected_arr)


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
        result = mama2.tweak_omega(omega_matrix)
        assert np.all(np.linalg.eigvalsh(result) >= 0.0)



###########################################

class TestSSInputTuple:

    #########
    @pytest.mark.parametrize("input_string, expected",
        [
            ("A/File,ANC1,Pheno1", ("A/File", "ANC1", "Pheno1")),
            ("Another/File,ANC2,P2", ("Another/File", "ANC2", "P2"))
        ])
    def test__happy_path__expected_results(self, input_string, expected):
        assert mama2.ss_input_tuple(input_string) == expected

    #########
    @pytest.mark.parametrize("input_string",
        [
            "A/FileANC1,Pheno1",
            "Another/FileANC2P2"
        ])
    def test__too_few_components__throw_error(self, input_string):
        with pytest.raises(RuntimeError):
            mama2.ss_input_tuple(input_string)
