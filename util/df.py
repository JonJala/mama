#!/usr/bin/env python3

"""
Python functions to process maps and dataframes
"""

import collections
import functools
import re
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# TODO(jonbjala) Currently exceptions thrown by filters are not caught.  Need to decide whether this
#                is correct longterm (i.e. filters should never throw or should halt the
#                whole program if they do), though it seems like SOME kind of error reporting
#                mechanism would be useful


# Constants / Parameters / Types #############

Filter = Callable[[pd.DataFrame], pd.Series]
FilterMap = Dict[str, Filter]


# Functions ##################################

#################################
def run_filters(df: pd.DataFrame, filters: FilterMap) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Runs a list of filters on the input dataframe, returning a dictionary of Boolean Series
    indicating which rows were caught by the filters and the Boolean Series corresponding
    to the union of all filtering

    :param sumstats_df: Dataframe holding data
    :param filters: Dictionary of filter name -> filter function

    :return: Tuple containing:
             1) The indices of the union of rows being dropped, and
             2) A dictionary mapping filter name (same as the key in "filters" input parameter)
                to an ordered collection (pd.Series) of booleans indicating which rows were caught
                by the associated filter
    """
    # Run the individual filters
    filt_results = {filter_name : filter_func(df) for filter_name, filter_func in filters.items()}

    # Figure out the indices of the union of SNPs caught by all the filters
    all_false = pd.Series(data=np.full(len(df), False), index=df.index)
    cumulative_indices = functools.reduce(lambda s1, s2: s1 | s2, filt_results.values(), all_false)

    return cumulative_indices, filt_results


#################################
def rename_dataframe_cols(input_df: pd.DataFrame, column_map: Dict[str, str]):
    """
    Standardizes column names in the input dataframe.  Modifications are done IN PLACE to
    the dataframe (i.e. it is altered)!

    :param input_df: Input dataframe
    :param column_map: Dictionary of column names of input_df mapped to standard column name

    :raises RuntimeError: If a column in the keys of the renaming map is missing from the DataFrame
    :raises RuntimeError: If any renamed column will have the same name as another after mapping
    """

    # Get current column list (before mapping)
    df_col_list_before = input_df.columns.to_list()

    # Check to make sure no column in the mapping keys is missing from the df column list
    missing_cols = {col for col in column_map.keys() if col not in df_col_list_before}
    if missing_cols:
        raise RuntimeError("Columns %s to be mapped are not present in DataFrame columns %s" %
            (missing_cols, df_col_list_before))

    # Get column list after mapping
    df_col_list_after = [column_map.get(col, col) for col in df_col_list_before]

    # Check to make sure no column mapping collides with any other "after mapping" column
    col_count = collections.Counter(df_col_list_after)
    colliding_cols = {old_col for old_col, new_col in column_map.items() if col_count[new_col] > 1}
    if colliding_cols:
        collisions = {column_map[col] for col in colliding_cols}
        raise RuntimeError("Columns %s map to columns (%s) that will create duplicates in the "
                           " final result: %s" % (colliding_cols, collisions, df_col_list_after))

    # Perform the renaming
    input_df.rename(columns=column_map, inplace=True)


#################################
# TODO(jonbjala) Add support for (or maybe require?) compiled RE objects as values of re_expr_map?
def determine_column_mapping(orig_col_list: List[str], re_expr_map: Dict[str, str],
                             req_cols: Set[str] = set()) -> Dict[str, str]:
    """
    Given a list of column names (orig_col_list) and a map of standard names to regular expressions,
    determine a mapping between elements of orig_col_list and the standard names.  The optional
    parameter req_cols checks for required standard columns that must be found / mapped to.

    The result mapping must be one-to-one (no collisions in domain or co-domain).  Case is ignored
    when searching for matches.

    Note: Regular expressions must match the full column (see fullmatch() in re module docs)

    :param orig_col_list: List of column names that need to be standardized
    :param re_expr_map: Map of standard names to regular expressions used for matching
    :param req_cols: If specified, used to check to make sure certain standard columns are "found"

    :raises RuntimeError: If the resulting mapping would not be one-to-one (collision found)
    :raises RuntimeError: If one or more elements in req_cols are not in the result mapping values

    :return: Dictionary mapping column names in a summary stat file to standard names
    """

    # Confirm there is no column specified in req_cols that is missing from the re_expr_map keys
    # (this makes sure that every required column can be "found")
    missing_map_cols = req_cols - set(re_expr_map.keys()) 
    if missing_map_cols:
        raise RuntimeError("Required columns %s are missing from the input regex map %s." %
                           (missing_map_cols, re_expr_map))

    # Map input columns to set of possible standardized column matches
    initial_mapping = {orig_col : set(filter(lambda m: m if re.fullmatch(re_expr_map[m], orig_col,
                       flags=re.IGNORECASE) else None, re_expr_map.keys()))
                       for orig_col in orig_col_list}

    # Check to make sure some columns don't map to more than one standard column
    # Note: Need to use list for multiple_matches instead of a set since a set must hold
    #       hashable / immutable objects
    multiple_matches = [(orig_col, std_col_set) for orig_col, std_col_set
                        in initial_mapping.items() if len(std_col_set) > 1]
    if multiple_matches:
        raise RuntimeError("The following ambiguous column matches were found: %s" %
                           multiple_matches)

    # Construct candidate result mapping (dispense with sets in the values)
    result_map = {orig_col : std_cols.pop() for orig_col, std_cols in initial_mapping.items()
                  if std_cols}

    # Check to make sure multiple columns don't match to the same standard column
    # Note: See note a few lines up w.r.t multiple_matches for list vs set discussion
    reverse_map = {std_col : set(filter(lambda m:
        result_map[m] == std_col, result_map.keys())) for std_col in result_map.values()}
    multiple_reverse_matches = [(std_col, orig_col_set) for std_col, orig_col_set
                                in reverse_map.items() if len(orig_col_set) > 1]
    if multiple_reverse_matches:
        raise RuntimeError("The following ambiguous column reverse matches were found: %s" %
                           multiple_reverse_matches)

    # Lastly, if req_cols is specified, check to make sure all are present
    missing_std_cols = req_cols - set(result_map.values())
    if missing_std_cols:
        raise RuntimeError("No matches for the following columns were found: %s" % missing_std_cols)

    return result_map


#################################
def intersect_indices(dfs: Sequence[pd.DataFrame], init: pd.DataFrame = None) -> pd.Index:
    """
    Returns a pandas Index that contains the intersection of all indices across the indicated
    dataframes (accessed by dfs sequence), with an optional addfitional parameter for the initial
    indices

    :param dfs: Sequence of DataFrame objects whose indices will be intersected
    :param init: DataFrame to use for the initial list of indices

    :return: A pandas Index containing the intersection of indices from input sources
    """
    # Make sure we have an initial DataFrame
    if init is None:
        init = dfs[0]
        dfs = dfs[1:]

    # Run intersection() on all sets of indices
    return functools.reduce(lambda c_ind, df_p: c_ind.intersection(df_p.index), dfs, init.index)