"""
Unit tests for util/df.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import itertools
import tempfile

import numpy as np
import pandas as pd
import pytest

import mama2.util.df as df_util


_FILTER_TEST_DATAFRAME_LENGTHS = [2, 3, 6, 10, 50]
_FILTER_TEST_DATAFRAME_SHAPES = list(itertools.permutations(_FILTER_TEST_DATAFRAME_LENGTHS, 2))
@pytest.fixture(scope="function", params=_FILTER_TEST_DATAFRAME_SHAPES)
def runfilter_test_df(request):

    # Fill DF with numbers 0..N-1 where there are N cells
    num_rows = request.param[0]
    num_cols = request.param[1]
    num_cells = num_rows * num_cols

    return pd.DataFrame(np.arange(num_cells).reshape((num_rows, num_cols)))


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


###########################################

class TestRunFilters:

    #########
    def test__single_filt__return_expected_results(self, runfilter_test_df):

        # Identify value whose row to filter out and the expected filtering indices
        target_int = runfilter_test_df.size // 2
        expected_indices = (runfilter_test_df == target_int).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once and has the correct number of rows
        assert expected_indices.sum() == 1

        # Filter the dataframe
        func_name = "Hello, world"
        result_indices, filt_map = df_util.run_filters(runfilter_test_df,
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
        result_indices, filt_map = df_util.run_filters(runfilter_test_df,
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
        result_indices, filt_map = df_util.run_filters(runfilter_test_df,
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
    def test__no_filt_or_useless_filt__no_change(self, runfilter_test_df):

        # BEFORE: Copy dataframe
        df_copy1 = runfilter_test_df.copy()

        # Filter the copy of the dataframe with no filter
        result_indices, filt_map = df_util.run_filters(df_copy1, {})

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
        result_indices, filt_map = df_util.run_filters(df_copy1,
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
        result_indices, filt_map = df_util.run_filters(df,
            {func_name : lambda df: (df == -1).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 1
        assert func_name in filt_map

        #   Confirm dataframe is unchanged
        assert len(df) == 0
        assert not any(filt_map[func_name])
        assert not any(result_indices)


    #########
    def test__legit_optional_filt__return_expected_results(self, runfilter_test_df):

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
        result_indices, filt_map = df_util.run_filters(runfilter_test_df,
            {func_name1 : lambda df: (df == target_int1).any(axis='columns')},
            {func_name2 : lambda df: (df == target_int2).any(axis='columns')})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 2
        assert func_name1 in filt_map
        assert func_name2 in filt_map

        #   Confirm result indices correct
        assert filt_map[func_name2] is not None
        assert all(result_indices == filt_map[func_name1] | filt_map[func_name2])
        assert result_indices.sum() == 2

    #########
    def test__failed_optional_filt__return_expected_results(self, runfilter_test_df):

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
        result_indices, filt_map = df_util.run_filters(runfilter_test_df,
            {func_name1 : lambda df: (df == target_int1).any(axis='columns')},
            {func_name2 : lambda df: x / 0})

        # AFTER:
        #   Confirm filt_map correct
        assert len(filt_map) == 2
        assert func_name1 in filt_map
        assert func_name2 in filt_map

        #   Confirm result indices correct
        assert filt_map[func_name2] is None
        assert all(result_indices == filt_map[func_name1])
        assert result_indices.sum() == 1


    #########
    def test__failed_req_filt__throws_error(self, runfilter_test_df):

        # Identify value whose row to filter out and the expected filtering indices
        target_int = runfilter_test_df.size // 2
        expected_indices = (runfilter_test_df == target_int).any(axis='columns')

        # BEFORE: Confirm dataframe contains the target int once and has the correct number of rows
        assert expected_indices.sum() == 1

        # Filter the dataframe
        func_name = "Hello, world"
        with pytest.raises(ZeroDivisionError):
            result_indices, filt_map = df_util.run_filters(runfilter_test_df,
                                                           {func_name : lambda df: 1 / 0})

###########################################

class TestRenameDataframeCols:

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
        df_util.rename_dataframe_cols(rename_test_df, col_map)

        # Check the columns of the transformed dataframe
        assert all(c_act == c_exp for (c_act, c_exp) in
            zip(rename_test_df.columns.to_list(), exp_cols))


    #########
    @pytest.mark.parametrize("col_map", [_MISSING_KEY_MAP1, _MISSING_KEY_MAP2, _MISSING_KEY_MAP3])
    def test__missing_columns__throws_error(self, col_map, rename_test_df):

        # Rename the columns of the dataframe
        with pytest.raises(RuntimeError):
            df_util.rename_dataframe_cols(rename_test_df, col_map)


    #########
    @pytest.mark.parametrize("col_map", [_COLLISION_UNCHANGED_MAP, _COLLISION_RENAMED_MAP])
    def test__rename_collisions__throws_error(self, col_map, rename_test_df):

        # Rename the columns of the dataframe
        with pytest.raises(RuntimeError):
            df_util.rename_dataframe_cols(rename_test_df, col_map)

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
            res = df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS, re_map,
                                                 set(req_cols[:i]))
            assert len(res) == num_map_cols
            res_vals = set(res.values())
            assert len(res.values()) == num_map_cols


    #########
    @pytest.mark.parametrize("re_map", [_RE_MAP_HAPPY_1, _RE_MAP_HAPPY_2])
    def test__re_map_keys_missing_req_col__throw_error(self, re_map):
        extra_req_col = "EXTRA_COL"
        req_cols = set(re_map.keys()) | {extra_req_col}

        with pytest.raises(RuntimeError) as ex_info:
            res = df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS, re_map,
                                                   req_cols)
        assert extra_req_col in str(ex_info.value)


    #########
    def test__map_one_col_to_two_std_cols__throw_error(self):
        with pytest.raises(RuntimeError):
            df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
                TestDetermineColumnMapping._RE_MAP_1_TO_2)


    #########
    def test__map_two_cols_to_same_std_col__throw_error(self):
        with pytest.raises(RuntimeError):
            df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
                TestDetermineColumnMapping._RE_MAP_2_TO_1)


    #########
    def test__req_col_not_matched__throw_error(self):
        # Without required columns, this should succeed and match everything but 1
        # (that column is, in effect, optional)
        res = df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
            TestDetermineColumnMapping._RE_MAP_MISSING)
        assert len(res) == len(TestDetermineColumnMapping._RE_MAP_MISSING) - 1

        # After requiring all columns in the map to be matched, should throw an error
        with pytest.raises(RuntimeError):
            df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
                TestDetermineColumnMapping._RE_MAP_MISSING,
                set(TestDetermineColumnMapping._RE_MAP_MISSING.keys()))

    #########
    def test__optional_col_matches__is_in_result_map(self):
        # Remove one required element (it will still match, just not be required)
        req_cols = set(list(TestDetermineColumnMapping._RE_MAP_HAPPY_1.keys())[1:])

        res = df_util.determine_column_mapping(TestDetermineColumnMapping._ORIG_COLS,
            TestDetermineColumnMapping._RE_MAP_HAPPY_1, req_cols)
        assert len(res) == len(TestDetermineColumnMapping._RE_MAP_HAPPY_1)


###########################################

class TestIntersectIndices:

    #########
    @pytest.mark.parametrize("mults, expected_intersection_size",
    [
            ([2, 3, 5], 7),
            ([2, 4, 6, 8], 5),
            ([1, 1, 1], 100),
            ([], 50),
            ([5], 30)
    ])
    def test__varying_indices__expected_intersection(self, mults, expected_intersection_size):

        # Derived parameters
        lcm = np.lcm.reduce(mults) if mults else 1
        n_max = lcm * expected_intersection_size

        # Function to return a dataframe with indices spaced out by n_mult
        def get_df(n_mult):
            i_list = [str(n_mult * i_num) for i_num in range(1, n_max + 1)]
            return pd.DataFrame(data={"dummy" : [1.0] * len(i_list)}, index=i_list)

        # Determine init and dfs parameters
        init = get_df(1)
        dfs = {i : get_df(i) for i in mults}

        intersection = df_util.intersect_indices(list(dfs.values()), init)
        assert len(intersection) == expected_intersection_size


    def test__no_intersection__returns_empty_index(self):

        # Determine init parameter
        init_indices = [str(i_num) for i_num in range(1, 10)]
        init = pd.DataFrame(data={"dummy" : [1.0] * len(init_indices)}, index=init_indices)

        # Ensure the dataframes all have indices that are outside of the init index list
        df_indices = [str(i_num) for i_num in range(len(init_indices) + 1, len(init_indices) + 10)]
        dfs = {i : pd.DataFrame(data={"dummy" : [1.0] * len(df_indices)}, index=df_indices)
               for i in range(1, 5)}

        intersection = df_util.intersect_indices(list(dfs.values()), init)
        assert len(intersection) == 0


    @pytest.mark.parametrize("mults, max_int",
    [
            ([2, 3, 5], 7),
            ([2, 4, 6, 8], 5),
            ([1, 1, 1], 100),
            ([], 50),
            ([5], 30)
    ])
    def test__vary_whether_init_is_specified__expected_result(self, mults, max_int):

        # Derived parameters
        lcm = np.lcm.reduce(mults) if mults else 1
        n_max = lcm * max_int

        # Function to return a dataframe with indices spaced out by n_mult
        def get_df(n_mult):
            i_list = [str(n_mult * i_num) for i_num in range(1, n_max + 1)]
            return pd.DataFrame(data={"dummy" : [1.0] * len(i_list)}, index=i_list)

        # Determine init and dfs parameters
        dfs = {i : get_df(i) for i in mults}

        for i in mults:
            dfs_copy = dfs.copy()
            df_i = dfs_copy.pop(i)

            intersection1 = df_util.intersect_indices(list(dfs_copy.values()), df_i)
            intersection2 = df_util.intersect_indices(list(dfs.values()))
            intersection3 = df_util.intersect_indices(list(dfs.values()), df_i)

            assert len(intersection1) == len(intersection2)
            assert len(intersection1) == len(intersection3)
            assert all(intersection1 == intersection2)
            assert all(intersection1 == intersection3)
