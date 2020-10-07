#!/usr/bin/env python3

"""
Python tool for multi-ancestry, multi-trait analysis
"""

import argparse as argp
import collections
import functools
import gc
import logging
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd


# Constants / parameters #############

# Software version
__version__ = '1.0.0'

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL1 = "grantgoldman0@gmail.com"
SOFTWARE_CORRESPONDENCE_EMAIL2 = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org"

# Standard column names
SNP_COL = 'SNP'
BP_COL = 'BP'
CHR_COL = 'CHR'
BETA_COL = 'BETA'
FREQ_COL = 'FREQ'
SE_COL = 'SE'
A1_COL = 'A1'
A2_COL = 'A2'

# Map of default regular expressions used to convert summary stat column names to standardized names
# TODO(jonbjala) Refine these more, just use these values are placeholders for now
MAMA_RE_EXPR_MAP = {
    SNP_COL : '.*SNP.*',
    BP_COL : '.*BP.*',
    CHR_COL : '.*CHR.*',
    BETA_COL : '.*BETA.*',
    FREQ_COL : '.*FREQ.*|.*FRQ.*',
    SE_COL : '.*SE.*',
    A1_COL : '.*A1.*',
    A2_COL : '.*A2.*',
}

# Columns that MAMA requires
MAMA_REQ_STD_COLS = {SNP_COL, CHR_COL, BETA_COL, FREQ_COL, SE_COL, A1_COL, A2_COL}



# Construct useful base-pair related constant sets
COMPLEMENT = {
"A" : "T",
"T" : "A",
"C" : "G",
"G" : "C",
}
BASES = set(COMPLEMENT.keys())

# Standard filter function description format string
DEFAULT_FILTER_FUNC_DESC = "Filters out SNPs %s"

# Standard filter functions used for SNPs for MAMA
MAMA_STD_FILTER_FUNCS = {
    'NO NAN' :
        {
        'func' : lambda df: df.isnull().any(axis=1),
        'description' : DEFAULT_FILTER_FUNC_DESC % "with any NaN values"
        },
    'FREQ BOUNDS' :
        {
        'func' : lambda df: ~df[FREQ_COL].between(0.0, 1.0),
        'description' : DEFAULT_FILTER_FUNC_DESC % "with FREQ values outside of [0.0, 1.0]"
        },
    'SE BOUNDS' :
        {
        'func' : lambda df: df[SE_COL].lt(0.0),
        'description' : DEFAULT_FILTER_FUNC_DESC % "with negative SE values"
        },
    'SNP PREFIX' :
        {
        'func' : lambda df: ~df[SNP_COL].str.startswith('rs'),
        'description' : DEFAULT_FILTER_FUNC_DESC % "whose IDs do not begin with \"rs\""
        },
    'CHR BOUNDS' :
        {
        'func' : lambda df: ~df[CHR_COL].between(1, 22),
        'description' : DEFAULT_FILTER_FUNC_DESC % "with listed chromosomes not in the range 1-22"
        },
    'INVALID SNPS' :
        {
        'func' : lambda df: df[A1_COL] == df[A2_COL],
        'description' : DEFAULT_FILTER_FUNC_DESC % "with major allele = minor allele"
        },
    'PALINDROMIC SNPS' :
        {
        'func' : lambda df: df[A1_COL].replace(COMPLEMENT) == df[A2_COL],
        'description' : DEFAULT_FILTER_FUNC_DESC % "where major and minor alleles are a base pair" # TODO(jonbjala) Is this description ok?
        },
    }


####################################################################################################

# Derived constants #############

# Logging banner to use at the top of the log file
HEADER = """
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<>
<> MAMA: Multi-Ancestry Meta-Analysis
<> Version: %s
<> (C) 2020 Social Science Genetic Association Consortium (SSGAC)
<> MIT License
<>
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<> Software-related correspondence: %s or %s
<> All other correspondence: %s
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
""" % (__version__, SOFTWARE_CORRESPONDENCE_EMAIL1, SOFTWARE_CORRESPONDENCE_EMAIL2,
       OTHER_CORRESPONDENCE_EMAIL)

# Filter function dictionaries (name to function mapping or description) for MAMA
MAMA_STD_FILTERS = {fname : finfo['func'] for fname, finfo in MAMA_STD_FILTER_FUNCS.items()}
MAMA_STD_FILT_DESC = {fname : finfo['description'] for fname, finfo
                      in MAMA_STD_FILTER_FUNCS.items()}


####################################################################################################

# Functions and Classes #############

AncestryId = Any
PhenotypeId = Any
PopulationId = Tuple[AncestryId, PhenotypeId]

def get_mama_parser(progname: str) -> argp.ArgumentParser:
    """
    Return a parser configured for this command line utility

    :param prog: Value to pass to ArgumentParser for prog (should generally be sys.argv[0])

    :return: argparse ArgumentParser
    """
    parser = argp.ArgumentParser(prog=progname)

    # LD Score Regression Options
    ld_reg = parser.add_argument_group(title="LD Score Regression Specifications",
                                       description="Options for LD Score Regression")
    ld_reg.add_argument("--reg-files", metavar="FILE_PATH_LIST", type=str, required=True, nargs="+",
                        help="TODO(jonbjala)")

    # Core MAMA Method Options
    core_mama = parser.add_argument_group(title="Core MAMA Method Specifications",
                                          description="Options for Core MAMA Method")
    core_mama.add_argument("--drop-non-posdef-snps", action="store_true",
                           help="TODO(jonbjala)")

    # TODO(jonbjala)
    """
    Options / Flags: (ignore LD score creation options for now)

    1) Internal processing using standardized units or not (inputs and outputs always in allele count)

    2) Inputs = gwas files, ancestry and phenotype names (each gwas needs to be associated with one of each of those),
                LD score file, (LD score regression coefs?)

    3) RE to either add or replace column matching with (use RE?  or just wildcarding?)

    4) options dealing with intermediate outputs (harmonized GWAS??  LD regression coefficients?)

    5) Out prefix

    6) log level

    7) LD score regression options (???)

    8) Turn on/off sumstat QC filters??

    Outputs:

    1) How to report betas and SEs

    2) Log file

    3) Need to report reference allele information?

    4) What should go to terminal / log file in terms of actual results?  (averages of omega / sigma / etc?)

    """


    return parser


def to_flag(arg_str: str) -> str:
    """
    Utility method to convert from the name of an argparse Namespace attribute / variable
    (which often is adopted elsewhere in this code, as well) to the corresponding flag

    :param arg_str: Name of the arg

    :return: The name of the flag (sans "--")
    """

    return arg_str.replace("_", "-")


def to_arg(flag_str: str) -> str:
    """
    Utility method to convert from an argparse flag name to the name of the corresponding attribute
    in the argparse Namespace (which often is adopted elsewhere in this code, as well)

    :param flag_str: Name of the flag (sans "--")

    :return: The name of the argparse attribute/var
    """

    return flag_str.replace("-", "_")


def set_up_logger() -> str:
    """
    Set up the logger for this utility.

    :return: Returns the full path to the log output file
    """

    # TODO(jonbjala)

    return ""


def get_user_inputs(argv: List[str], parsed_args: argp.Namespace) -> str:
    """
    Create dictionary of user-specified options/flags and their values.  Leverages the argparse
    parsing output to glean the actual value, but checks for actual user-set flags in the input

    :param argv: Tokenized list of inputs (meant to be sys.argv in most cases)
    :param parsed_args: Result of argparse parsing the user input

    :return: Dictionary containing user-set args keyed to their values
    """

    # Search for everything beginning with "--" (flag names), strip off the --, take everything
    # before any "=", and convert - to _
    user_set_args = {to_arg(token[2:].split("=")[0]) for token in argv if token.startswith("--")}

    # Since any flag actually specified by the user shouldn't have been replaced by a default
    # value, one can grab the actual value from argparse without having to parse again
    return {user_arg:getattr(parsed_args, user_arg) for user_arg in user_set_args}


def validate_inputs(pargs: argp.Namespace, user_args: Dict[str, Any]):
    """
    Responsible for coordinating whatever initial validation of inputs can be done

    :param pargs: Result of argparse parsing user command / flags
    :param user_args: Flags explicitly set by the user along with their values

    :return: Dictionary that contains flags and parameters needed by this program.  It contains
             user-input flags along with defaults set through argparse, and any additional flags
             added as calculations proceed
    """

    # TODO(jonbjala)

    return dict()


def format_terminal_call(cmd: List[str]) -> str:
    """
    Format commands to/from the terminal for readability

    :param cmd: List of strings much like sys.argv

    :return: Formatted string used for display purposes
    """

    return ' '.join(cmd).replace("--", " \\ \n\t--")



def intersect_snp_lists(sumstats: Dict[str, pd.DataFrame], ldscores: pd.DataFrame) -> pd.Index:
    """
    Returns a pandas Index that contains the intersection of all SNPs across the summary statistics
    and the LD scores

    :param sumstats: Dictionary mapping a population name to a DataFrame holding the summary
                     stat information.  The DFs should all have been QCed already.
    :param ldscores: DataFrame of LD score information

    :return pd.Index: A pandas Index containing the intersection of SNPs from input sources
    """

    # Run intersection() on all sets of indices (using the LD score SNP list as the initializer)
    return functools.reduce(lambda c_ind, df_p: c_ind.intersection(df_p.index),
                            sumstats.values(), ldscores.index)


# TODO(jonbjala) Currently exceptions thrown by filters are not caught.  Need to decide whether this
#                is correct longterm (i.e. filters should never throw or should halt the
#                whole program if they do), though it seems like SOME kind of error reporting
#                mechanism would be useful
SumstatFilter = Callable[[pd.DataFrame], pd.Series]
def run_filters(df: pd.DataFrame, filters: Dict[str, SumstatFilter]) -> Dict[str, pd.Series]:
    """
    Runs a list of filters on the input dataframe, returning a dictionary of Boolean Series
    indicating which rows / SNPs were caught by the filters and the Boolean Series corresponding
    to the union of all filtering

    :param sumstats_df: Dataframe holding summary stats
    :param filters: Dictionary of filter name -> filter function

    :return: Tuple containing:
             1) The indices of the union of SNPs being dropped, and
             2) A dictionary mapping filter name (same as the key in "filters" input parameter)
                to an ordered collection (pd.Series) of booleans indicating which SNPs were caught
                by the associated filter
    """
    # Run the individual filters
    filt_results = {filter_name : filter_func(df) for filter_name, filter_func in filters.items()}

    # Figure out the indices of the union of SNPs caught by all the filters
    all_false = pd.Series(data=np.full(len(df), False), index=df.index)
    cumulative_indices = functools.reduce(lambda s1, s2: s1 | s2, filt_results.values(), all_false)

    return cumulative_indices, filt_results


def flip_alleles(df: pd.DataFrame, flip_indices: pd.Series):
    """
    Given an Series of booleans (corresponding to rows/indices in the df input parameter), flips
    the major and minor alleles in the df (in place) for rows indicated by True.  Assumes that
    flip_indices is the correct length for df (they each contain the same number of rows)

    :param df: QC'ed DataFrame holding the summary stat information
    :param flip_indices: Series holding True/False (True for rows that should be flipped)
    """

    # Replace freq with 1.0 - freq, beta with -beta, and swap major and minor alleles
    # (but only do so for the indices where flip_indices = True)
    df[FREQ_COL].mask(flip_indices, 1.0 - df[FREQ_COL], inplace=True)
    df[BETA_COL].mask(flip_indices, -df[BETA_COL], inplace=True)
    df.loc[flip_indices, [A1_COL, A2_COL]] = df.loc[flip_indices, [A2_COL, A1_COL]].values


def standardize_all_sumstats(sumstats: Dict[PopulationId, pd.DataFrame],
                             ref: Tuple[Any, pd.DataFrame]=()
                             ) -> Tuple[Any, pd.Series, pd.Series, Dict[Any, pd.Series]]:
    """
    Takes a set of summary statistics and standardizes them according to a reference set.  This will
    involve keeping any that match reference alleles (or strand-swapped versions), adjusting any
    that are reference allele-flipped, and keep track of the rest.  If a reference isn't included,
    one of the populations from the sumstats input parameter will be chosen.

    :param sumstats: Dictionary mapping a population id to a DataFrame holding the summary
                     stat information.  The DFs should all have been QCed already and should all
                     match SNP lists exactly.
    :param ref: Reference DataFrame (used as a source of ground truth for major/minor alleles)
                Assumed to be QCed and match SNP lists with DFs in sumstats input parameter

    :return: A tuple containing:
             1) The identifier of the reference population
             2) The indices that should be dropped due to a mismatch of at least one GWAS
             3) The recommended drops (as indices) broken down by population
             4) The indices of SNPs that needed adjusting due to major/minor allele swaps
                in at least one GWAS) along with a dictionary that contains the drops broken down by
                population name
    """

    # Get list of name / DataFrame pairs by population
    ss_pops = list(sumstats.items())

    # Determine reference population name and DataFrame (if not supplied just designate one)
    ref = ref if ref else ss_pops[0]
    ref_popid = ref[0]
    ref_df = ref[1]


    # Define filter functions / filter function dictionary with respect to the reference population
    ref_a1 = ref_df[A1_COL]
    ref_a2 = ref_df[A2_COL]
    ref_a1_comp = ref_df[A1_COL].replace(COMPLEMENT)
    ref_a2_comp = ref_df[A2_COL].replace(COMPLEMENT)

    def allele_match(df: pd.DataFrame):
        exact_match = (df[A1_COL] == ref_a1) & (df[A2_COL] == ref_a2)
        sflip_match = (df[A1_COL] == ref_a1_comp) & (df[A2_COL] == ref_a2_comp)
        return exact_match | sflip_match

    def allele_ref_swap(df: pd.DataFrame):
        exact_swap = (df[A1_COL] == ref_a2) & (df[A2_COL] == ref_a1)
        sflip_swap = (df[A1_COL] == ref_a2_comp) & (df[A2_COL] == ref_a1_comp)
        return exact_swap | sflip_swap

    allele_filts = {"allele_match" : allele_match, "allele_ref_swap" : allele_ref_swap}


    # Run filters on all populations (filters are checking for allele matches, so any that aren't
    # caught should be dropped)
    drop_dict = {}
    ref_flip_dict = {}
    cumulative_drop_indices = pd.Series(data=np.full(len(ref_df), False), index=ref_df.index)
    for pop_id, pop_df in ss_pops:
        # Run the filters to determine which SNPs match (and which match aside from ref allele flip)
        keep_indices, filt_indices = run_filters(pop_df, allele_filts)

        # Determine which SNPs should be dropped and which should be flipped (reference alleles)
        drop_indices = ~keep_indices
        flip_indices = filt_indices["allele_ref_swap"]

        # Save off the required information
        drop_dict[pop_id] = drop_indices
        ref_flip_dict[pop_id] = flip_indices
        cumulative_drop_indices |= drop_indices

        # Flip the SNPs that need to be flipped
        flip_alleles(pop_df, flip_indices)


    return ref_popid, cumulative_drop_indices, drop_dict, ref_flip_dict


def harmonize_all(sumstats: Dict[PopulationId, pd.DataFrame], ldscores: pd.DataFrame):
    """
    Does the harmonization between the QC'ed input summary statistics and the LD scores.  The
    DataFrames are all modified in place (SNPs/rows dropped and reference alleles transformed
    as needed), and all inputs are expected to have indices = SNP ID (beginning with "rs")

    :param sumstats: Dictionary mapping a population id to a DataFrame holding the summary
                     stat information.  The DFs should all have been QCed already.
    :param ldscores: DataFrame of LD score information
    """

    # Intersect all the SNP lists to get the SNPs all data sources have in common
    snp_intersection = intersect_snp_lists(sumstats, ldscores)

    # Reduce each DF down to the SNP intersection TODO(jonbjala) Add logging / reporting, tally of drops, etc
    for pop_id, pop_df in sumstats.items():
        snps_to_drop = pop_df.index.difference(snp_intersection)
        pop_df.drop(snps_to_drop, inplace=True)
    snps_to_drop = ldscores.index.difference(snp_intersection)
    ldscores.drop(snps_to_drop, inplace=True)

    # Standardize alleles in the summary statistics
    ref_popid, drop_indices, drop_dict, ref_flip_dict = standardize_all_sumstats(sumstats)

    # Drop SNPs where there was an unfixable major/minor allele mismatch
    for pop_id, pop_df in sumstats.items():
        pop_df.drop(pop_df.index[drop_indices], inplace=True)

    # TODO(jonbjala) Log dropped SNPs (at least a total)

def rename_sumstats_cols(sumstats_df: pd.DataFrame, column_map: Dict[str, str]):
    """
    Standardizes column names in the input dataframe.  Modifications are done IN PLACE to
    the dataframe (i.e. it is altered)!

    :param sumstats_df: Dataframe holding
    :param column_map: Dictionary of column in the sumstats_df mapped to standard column name

    :raises RuntimeError: If a column included in the renaming map is missing from the DataFrame
    :raises RuntimeError: If any renamed column will have the same name as another after mapping
    """

    # Get current column list (before mapping)
    df_col_list_before = sumstats_df.columns.to_list()

    # Check to make sure no column in the mapping is missing
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
    sumstats_df.rename(columns=column_map, inplace=True)


def qc_sumstats(sumstats_df: pd.DataFrame, filters: Dict[str, SumstatFilter],
                column_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series,
                                                     Dict[str, pd.Series], List]:
    """
    Runs QC steps like renaming columns and dropping rows based on filters

    :param sumstats_df: Dataframe holding
    :param column_map: Dictionary of column in the sumstats_df mapped to standard column name

    :return: Tuple containing:
             1) A modified copy of the input data frame (SNPs dropped, columns renamed, etc.)
             2) The indices of the union of SNPs being dropped, and
             3) A dictionary mapping filter name to an ordered collection (pd.Series) of
                booleans indicating which SNPs to drop for that filter
             4) A series containing the rsIDs of SNPs that are dropped due to being duplicates
    """

    # Make copy of the dataframe (this copy will be modified)
    df = sumstats_df.copy()

    # Rename columns to standardized names
    rename_sumstats_cols(df, column_map)

    # Make sure SNP IDs are lower case ("rs..." rather than "RS...")
    df[SNP_COL] = df[SNP_COL].str.lower()

    # Run filters and drop rows
    cumulative_drop_indices, filt_drop_indices = run_filters(df, filters)
    df.drop(df.index[cumulative_drop_indices], inplace=True)

    # Drop duplicate SNP rows, set the SNP column to be the index, and sort by index
    dup_snp_indices = df[SNP_COL].duplicated()
    dup_snps = df[SNP_COL][dup_snp_indices]
    df.drop(df.index[dup_snp_indices], inplace=True)
    df.set_index(SNP_COL, inplace=True)
    df.sort_index(inplace=True)


    return df, cumulative_drop_indices, filt_drop_indices, dup_snps.to_list()


# TODO(jonbjala) Add support for (or maybe require?) compiled RE objects as values of re_expr_map?
def determine_column_mapping(orig_col_list: List[str], re_expr_map: Dict[str, str],
                             req_cols: Set[str] = []) -> Dict[str, str]:
    """
    Given a list of column names (orig_col_list) and a map of standard names to regular expressions,
    determine a mapping between elements of orig_col_list and the standard names.  The optional
    parameter req_cols checks for required standard columns that must be found / present.

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

    # TODO(jonbjala) The code for this method might be dense and not as clear as is desirable?

    # TODO(jonbjala) Check for case when req_cols specifies std cols not in re_expr_map.keys()?

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
    if req_cols:
        missing_std_cols = req_cols - set(result_map.values())
        if missing_std_cols:
            raise RuntimeError("No matches for the following columns were found: %s" %
                               missing_std_cols)

    return result_map


def process_sumstats(initial_df: pd.DataFrame,
                     column_map: Dict[str, str] = None,
                     re_expr_map: Dict[str, str] = MAMA_RE_EXPR_MAP,
                     req_std_cols: Set[str] = MAMA_REQ_STD_COLS,
                     filters: Dict[str, SumstatFilter] = MAMA_STD_FILTERS,
                     filt_descriptions: Dict[str, str] = MAMA_STD_FILT_DESC) -> pd.DataFrame:
    """
    Read the specified summary statistics file into a Pandas DataFrame, and run QC steps on it,
    the most important and notable being standardizing column names and running filters to drop
    SNPs (e.g. where MAF < 0)

    :param initial_df: DataFrame as initially read in
    :param column_map: Map used to rename columns to standard strings.  If not passed in, then
                       re_expr_map is used to calculate it.
    :param re_expr_map: Map of standard column names to regular expressions used for matching
                        against summary stat file column names.
    :param req_std_cols: Required standard columns in the resulting DataFrame
    :param filters: Map of filter functions used to drop undesired SNPs
    :param filt_descriptions: Map of filter function descriptions

    :raises RuntimeError: If req_std_cols contains columns not in column_map.keys()

    :return: A modified dataframe with renamed columns (and the SNP ID column as the index)
             minus undesired SNPs / rows.
    """

    # Log the top portion of the dataframe at debug level
    logging.debug("First set of rows from initial reading of summary stats:\n%s", initial_df.head())

    # Generate column map if necessary and then validate
    if not column_map:
        column_map = determine_column_mapping(initial_df.columns.to_list(), re_expr_map)
    missing_req_cols = req_std_cols - set(column_map.values())
    if missing_req_cols:
        raise RuntimeError("Required columns (%s) missing from column mapping: %s" %
                               (missing_req_cols, column_map))


    # Run QC on the df
    qc_df, drop_indices, per_filt_drop_map, dups = qc_sumstats(initial_df, filters, column_map)

    # Log SNP drop info
    for filt_name, filt_drops in per_filt_drop_map.items():
        logging.info("Filtered out  %d SNPs with \"%s\" (%s)", filt_drops.sum(), filt_name,
            filt_descriptions.get(filt_name, "No description available"))
        logging.debug("RS IDs = %s\n", initial_df[filt_drops.to_list()])
    logging.info("\nFiltered out %d SNPs in total (as the union of drops, this may be "
                 "less than the total of all the per-filter drops)", drop_indices.sum())
    logging.info("\nAdditionally dropped %d duplicate SNPs", len(dups))
    logging.debug("RS IDs = %s\n", dups)

    return qc_df


def run_regression(dep_var: np.ndarray, indep_vars: np.ndarray,
                   fixed_coefs: np.ndarray = None) -> np.ndarray:
    """
    Regress dependent variable on the N_var independent variables indicated in indep_vars.

    If fixed_coefs is specified, it must be of length N_var, with NaNs indicating unconstrained
    variables and other numbers indicating the value to which to fix the corresponding coefficient.

    :param dep_var: 1-D (length = N_pts) ndarray for the dependent variable
    :param indep_vars: N_pts x N_var ndarray describing the independent variables
    :param fixed_coefs: 1-D (length = N_vars) ndarray describing fixed coefficients.
                        If None, all variables are unconstrained.

    :return: 1-D (length = N_vars) ndarray containing the regression coefficient values
             in the same order as listed in indep_vars
    """

    # Determine number of independent variables (including constrained / fixed coefficient ones)
    N_var = indep_vars.shape[1]

    # Create empty solution vector
    result = np.zeros(N_var)

    # Process any fixed-coefficient variables
    if fixed_coefs is not None:  # Check explicitly against None since ndarray is not True or False
        # Make copy of dep_var since this will be modified
        dep_var_vect = np.copy(dep_var)

        # Get the indices of the fixed coefficients
        unconstrained_var_indices = np.isnan(fixed_coefs)
        constrained_var_indices = np.logical_not(unconstrained_var_indices)

        # Adjust the dependent variable accordingly
        dep_var_vect -= np.sum(indep_vars[:, constrained_var_indices] *
                               fixed_coefs[constrained_var_indices], axis=1)

        # Set the corresponding elements in the solution vector
        result[constrained_var_indices] = fixed_coefs[constrained_var_indices]
    else:
        # All variables are unconstrained and dependent variable is read-only
        dep_var_vect = dep_var
        unconstrained_var_indices = np.full(N_var, True)

    # Run the regression on the (remaining) unconstrained variables
    # It returns a tuple, but we only care about the first element
    result[unconstrained_var_indices] = np.linalg.lstsq(
        indep_vars[:, unconstrained_var_indices], dep_var_vect, rcond=None)[0]

    return result


def run_ldscore_regressions(harm_betas, harm_ses,
                            ldscores) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the LD score and beta SE regression.  Assumes the PxP submatrices in the ldscores and the
    P columns of harmonized summary stat data have the same ordering of corresponding ancestries.

    :param harm_betas: MxP matrix (M SNPs by P populations) of betas / effect sizes
    :param harm_ses: MxP matrix (M SNPs by P populations) of beta standard errors
    :param ldscores: (Mx)PxP symmetric matrices containing LD scores (PxP per SNP)

    :return: A tuple holding regression coefficient matrices (ldscore, constant, and se^2),
             each one a PxP ndarray
    """

    # Useful constants
    LD_SCORE_COEF = 0
    CONST_COEF = 1
    SE_PROD_COEF = 2
    N_VARS = 3  # There are 3 coefficient matrices being determined, see lines immediately above

    # Determine some ndarray / matrix dimension lengths
    M = harm_betas.shape[0]
    P = harm_betas.shape[1]

    # Allocate space for the regression matrix, order will be ld scores, constant, and se product
    # (will be partially overwritten at each iteration but no need to reallocate each time)
    reg_matrix = np.zeros((M, N_VARS))
    reg_matrix[:, CONST_COEF] = np.ones(M)

    # Allocate coefs matrix (3 x P x P, slices are LD score, constant, se^2 in that order)
    result_coefs = np.zeros((N_VARS, P, P))

    # Allocate fixed_coefs vector (length 3, order will be ld scores, constant, and se product)
    fixed_coefs = np.full(N_VARS, np.NaN)

    # Calculate each element (and therefore its symmetric opposite, as well)
    for p1 in range(P):
        for p2 in range(p1, P):
            # Set the needed columns in the regression matrix
            reg_matrix[:, LD_SCORE_COEF] = ldscores[:, p1, p2] # LD Score column
            reg_matrix[:, SE_PROD_COEF] = np.multiply(harm_ses[:, p1], harm_ses[:, p2]) # SE product

            # TODO(jonbjala) Need various options to control what to fix things to
            fixed_coefs[SE_PROD_COEF] = np.NaN if p1 == p2 else 0.0 # Only use for diagonals

            # Run the regression and set opposing matrix entry to make coef matrix symmetric
            result_coefs[:, p1, p2] = run_regression(
                np.multiply(harm_betas[:, p1], harm_betas[:, p2]), reg_matrix, fixed_coefs)
            result_coefs[:, p2, p1] = result_coefs[:, p1, p2]


    return result_coefs[LD_SCORE_COEF], result_coefs[CONST_COEF], result_coefs[SE_PROD_COEF]


def create_omega_matrix(ldscores: np.ndarray, reg_ldscore_coefs: np.ndarray) -> np.ndarray:
    """
    Creates the omega matrix for each SNP.  Assumes the PxP submatrices in the ldscores and the
    PxP matrix of LD regression coefficients have the same ordering of corresponding ancestries.

    :param ldscores: (Mx)PxP symmetric matrices containing LD scores (PxP per SNP)
    :param reg_ldscore_coefs: PxP symmetric matrix containing LD score regression coefficients

    :return: The Omega matrices as indicated in the MAMA paper (PxP per SNP) = (Mx)PxP
    """

    # Multiply PxP slices of LD scores with the regression coefficients component-wise
    return reg_ldscore_coefs * ldscores


def create_sigma_matrix(sumstat_ses, reg_se2_coefs, reg_const_coefs):
    """
    Creates the sigma matrix for each SNP.  Assumes the PxP submatrices in the ldscores and the
    PxP matrix of LD regression coefficients have the same ordering of corresponding ancestries.

    :param sumstat_se: Standard errors for the SNPs for each population (M x P matrix)
    :param reg_se2_coefs: PxP symmetric matrix containing SE^2 regression coefficients
    :param reg_const_coefs: PxP symmetric matrix containing Constant term regression coefficients

    :return: The Sigma matrices as indicated in the MAMA paper (PxP per SNP) = (Mx)PxP
    """

    # Get values for M and P (used to keep track of slices / indices / broadcasting)
    M = sumstat_ses.shape[0]
    P = sumstat_ses.shape[1]

    # Create an MxPxP matrix with each PxP slice initially equal to reg_const_coefs
    result_matrix = np.full(shape=(M, P, P), fill_value=reg_const_coefs)

    # Create an M X P matrix, whose rows of length P will need to be added to the diagonals
    # of the PxP slices in the final result
    se_diags_as_matrix = sumstat_ses * sumstat_ses * np.diag(reg_se2_coefs)

    # Broadcast-add the rows of the SE term matrix to the diagonals of slices of the result matrix
    d_indices = np.arange(P)
    result_matrix[:, d_indices, d_indices] += se_diags_as_matrix

    return result_matrix


def run_mama_method(betas, omega, sigma):
    """
    Runs the core MAMA method to combine results and generate final, combined summary statistics

    :param harm_sumstats: TODO(jonbjala)
    :param omega: TODO(jonbjala)
    :param sigma: TODO(jonbjala)

    :return Tuple[np.ndarray, np.ndarray]: Resulting betas and standard errors # TODO(jonbjala) need to include drops
    """

    # TODO(jonbjala) Remove printouts, add positive (semi-)def checks and needed processing, dropping some SNPs

    # Get values for M and P (used to keep track of slices / indices / broadcasting)
    M = omega.shape[0]
    P = omega.shape[1]

    # Create a 3D matrix, M rows of Px1 column vectors with shape (M, P, 1)
    d_indices = np.arange(P)
    omega_diag = omega[:, d_indices, d_indices][:, :, np.newaxis]
    omega_pp_scaled = np.divide(omega, omega_diag)  # Slice rows are Omega'_pjj / omega_pp,j

    # Produce center matrix in steps (product of omega terms, add omega and sigma, then invert)
    center_matrix_inv = -omega_pp_scaled[:, :, :, np.newaxis] * omega[:, :, np.newaxis, :]
    center_matrix_inv += omega[:, np.newaxis, :, :] + sigma[:, np.newaxis, :, :] # Broadcast add
    center_matrix = np.linalg.inv(center_matrix_inv) # Inverts each slice separately
    del center_matrix_inv; gc.collect() # Clean up the inverse matrix to free space

    # Calculate (Omega'_p,j/omega_pp,j) * center_matrix
    left_product = np.matmul(omega_pp_scaled[:, :, np.newaxis, :], center_matrix)
    del center_matrix; gc.collect() # Clean up the center matrix to free space

    # Calculate denominator (M x P x 1 x 1)
    denom = np.matmul(left_product,
                      np.transpose(omega_pp_scaled[:, :, np.newaxis, :], (0, 1, 3, 2)))
    denom_recip = np.reciprocal(denom)
    denom_recip_view = denom_recip.view()
    denom_recip_view.shape = (M, P)

    # Calculate numerator (M x P x 1 x 1))
    left_product_view = left_product.view()
    left_product_view.shape = (M, P, P)
    numer = np.matmul(left_product_view, betas[:,:,np.newaxis])
    numer_view = numer.view()
    numer_view.shape = (M, P)

    # Calculate result betas and standard errors
    new_betas = denom_recip_view * numer_view
    new_beta_ses = np.sqrt(denom_recip_view)

    return new_betas, new_beta_ses


def collate_df_values(sumstats: Dict[PopulationId, pd.DataFrame], ldscores: pd.DataFrame,
                      ordering: List[PopulationId] = None) -> pd.DataFrame:
    """
    Function that gathers data from DataFrames (betas, ses, etc.) into ndarrays for use in
    vectorized processing

    :param sumstats: Dictionary of population identifier -> DataFrame
    :param ldscores: DataFrame for the LD scores
    :param ordering: Optional parameter indicating the order in which populations should be arranged
                     (if not specified, the ordering of the sumstats dictionary keys will be used)

    :return Tuple[np.ndArray, np.ndArray, np.ndArray]: Betas (MxP), SEs (MxP), and LD scores (MxPxP)
    """

    # Make sure ordering is specified
    if not ordering:
        ordering = list(sumstats.keys())

    # Move summary statistic data into arrays to allow for vectorized operations
    #   1) Gather important numbers to use for shapes and dimensions
    num_pops = len(sumstats)
    num_snps = len(ldscores)
    #   2) Create empty arrays so the memory can be allocated all at once
    beta_arr = np.zeros((num_snps, num_pops))
    se_arr = np.zeros((num_snps, num_pops))
    ldscore_arr = np.zeros((num_snps, num_pops, num_pops))
    #   3) Copy data into place
    for pop_num, pop_id in enumerate(ordering):
        pop_df = sumstats[pop_id]
        ancestry_id = pop_id[0]

        beta_arr[:, pop_num] = pop_df[BETA_COL].to_numpy()
        se_arr[:, pop_num] = pop_df[SE_COL].to_numpy()

        # For LD scores, need to iterate through populations a second time to process pairs
        for second_pop_num, second_pop_id in enumerate(ordering):
            second_ancestry_id = second_pop_id[0]

            ldscore_col_name = "%s_%s" % (ancestry_id, second_ancestry_id)
            if ldscore_col_name not in ldscores.columns:
                ldscore_col_name = "%s_%s" % (second_ancestry_id, ancestry_id)

            ldscore_arr[:, pop_num, second_pop_num] = ldscores[ldscore_col_name].to_numpy()

    return beta_arr, se_arr, ldscore_arr


def obtain_df(possible_df: Any, id_val: Any) -> pd.DataFrame:
    """
    Small helper function that handles functionality related to reading in a DataFrame

    :param possible_df: Should either be a string indicating the full path to a file to be
                        read into a DataFrame or the DataFrame itself.  All other possibilities will
                        result in this function raising an error
    :param id_str: Used for logging / error-reporting to identify the data being read / checked

    :raises RuntimeError: If possible_df is not a string or pd.DataFrame

    :return pd.DataFrame: Returns a DataFrame
    """

    # If this is (presumably) a filename, read in the file
    if isinstance(possible_df, str):
        logging.info("Reading in %s file: %s", id_val, possible_df)
        possible_df = pd.read_csv(possible_df, sep=None, engine='python', comment='#')
    # If neither a string (presumed to be a filename) nor DataFrame are passed in, throw error
    elif not isinstance(sumstats[pop_name], pd.DataFrame):
        raise RuntimeError("ERROR: Either pass in filename or DataFrame for %s rather than [%s]" %
                           (id_val, type(possible_df)))

    return possible_df


def qc_ldscores(ldscores_df: pd.DataFrame):
    """
    Runs QC steps on LD scores.  This will be much lighter-weight than what is done on summary
    statistics, as it assumes that the LD score file was generated using this software.

    :param ldscores_df: Dataframe holding ldscores

    :return pd.DataFrame: DataFrame containing the QC'ed LD scores
    """
    # Make copy of the dataframe (this copy will be modified)
    df = ldscores_df.copy()

    # Make sure SNP IDs are lower case ("rs..." rather than "RS...")
    df[SNP_COL] = df[SNP_COL].str.lower()

    # Set SNP column to be the index and sort
    df.set_index(SNP_COL, inplace=True)
    df.sort_index(inplace=True)

    return df


def qc_sigma(sigma: np.ndarray) -> np.ndarray:
    """
    Runs checks over the sigma matrices for positive-definiteness.  Returns an array of length M
    (where M = number of SNPs) along the SNP axis (the first dimension of the MxPxP sigma)
    where True indicates positive definiteness and False indicates non-positive definiteness

    :param sigma: MxPxP matrix for Sigma values

    :return np.ndarray: Array of length M where True indicates positive definiteness and False
                        indicates non-positive definiteness
    """

    # Create result vector of length M, all values defaulting to False
    M = sigma.shape[0]
    result = np.full(M, False)

    # Iterate over the M PxP matrices of sigma
    for i in range(M):
        sigma_slice = sigma[i, :, :]
        try:
            np.linalg.cholesky(sigma_slice)
            result[i] = True
        except np.linalg.LinAlgError as e:
            # If not positive definite, then the Cholesky decomposition raises a LinAlgError
            pass

    return result


def tweak_omega(omega_slice: np.ndarray) -> np.ndarray:
    """
    Tweaks the off-diagonal elements of a non positive semi-definite omega matrix to make it
    positive semi-definite.  This assumes that necessary checks are done ahead of time to ensure
    this method will converge (e.g. all diagonal elements must be positive)

    :param omega_slice: PxP symmetric Omega matrix

    :return np.ndarray: A modified omega that is now positive semi-definite
    """

    # First get the component-wise square root of the diagonal
    omega_diag = np.diag(omega_slice).copy()
    omega_sqrt_diag = np.sqrt(omega_diag)

    # Clamp off diagonal elements to values based on product of the corresponding diagonal entries
    omega_slice = np.minimum(np.outer(omega_sqrt_diag, omega_sqrt_diag), omega_slice)

    # Then, scale down off-diagonal elements until positive semi-definite
    d_indices = np.diag_indices_from(omega_slice)
    while np.any(np.linalg.eigvalsh(omega_slice) < 0.0):
        omega_slice *= 0.99
        omega_slice[d_indices] = omega_diag

    return omega_slice


def qc_omega(omega: np.ndarray) -> np.ndarray:
    """
    Runs checks over the omega matrices for positive-semi-definiteness.  Tweaks omega where possible
    to correct for non-positive-semi-definiteness and returns an array of length M
    (where M = number of SNPs) along the SNP axis (the first dimension of the MxPxP omega)
    where True indicates positive semi-definiteness and False indicates
    non-positive semi-definiteness

    :param omega: MxPxP matrix for Omega values

    :return np.ndarray: Array of length M where True indicates positive semi-definiteness and False
                        indicates non-positive semi-definiteness
    """

    # Create result vector of length M, all values defaulting to False
    M = omega.shape[0]
    result = np.full(M, False)

    # Iterate over the M PxP matrices of sigma
    for i in range(M):
        omega_slice = omega[i, :, :]

        # Check for positive semi-definiteness (if PSD, set to True and move on)
        if np.all(np.linalg.eigvalsh(omega_slice) >= 0.0):
            result[i] = True
            continue

        # If diagonal entries aren't positive, move on
        if np.any(np.diag(omega_slice) <= 0.0):
            continue

        # We can try to tweak ths slice of omega to become positive semi-definite
        omega_slice = tweak_omega(omega_slice)
        result[i] = True

    return result


# TODO(jonbjala) Allowing specifying population order?
def mama_pipeline(sumstats: Dict[PopulationId, pd.DataFrame], ldscores: Any,
                  column_maps: Dict[PopulationId, Dict[str, str]] = {},
                  re_expr_map: Dict[str, str] = MAMA_RE_EXPR_MAP) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the steps in the overall MAMA pipeline

    :param sumstats: Dictionary of population identifier -> filename or DataFrame
    :param ldscores: Filename or DataFrame for the LD scores
    :param column_maps: Dictionary containing any column mappings indexed by population identifier
                        (same as used for sumstats parameter).  If none exists, the re_expr_map
                        will be used to determine column mappings
    :param re_expr_map: Regular expressions used to map column names to standard columns

    :return Tuple[np.ndarray, np.ndarray]: Tuple containing
                                               1) Result betas
                                               2) Result standard errors on the betas
                                               3) Array indicating True where SNPs were effectively
                                                  dropped due to invalid sigma / omega matrices
    """

    # Initial validation
    # TODO(jonbjala)

    # Check / read in LD scores and then QC
    ldscores = obtain_df(ldscores, "LD scores")
    ldscores = qc_ldscores(ldscores)
    # TODO(jonbjala) QC LD scores (should probably use sumstats.keys() minimally)

    # Check / read in summary stats and then QC
    for pop_id in sumstats.keys():
        # Read in if necessary (and update dictionary)
        sumstats[pop_id] = obtain_df(sumstats[pop_id], str(pop_id) + " sumstats")

        # QC summary stats (along with some light validation and some logging of drops)
        pop_df = sumstats[pop_id]
        col_map = column_maps.get(pop_id, None)  # If a column map exists for this pop, use that
        sumstats[pop_id] = process_sumstats(pop_df, col_map, re_expr_map, MAMA_REQ_STD_COLS,
                                            MAMA_STD_FILTERS, MAMA_STD_FILT_DESC)

    # Harmonize the summary stats and LD scores
    harmonize_all(sumstats, ldscores)

    # Copy values to numpy ndarrays to use in vectorized processing
    beta_arr, se_arr, ldscore_arr = collate_df_values(sumstats, ldscores)
    # TODO(jonbjala) Maybe delete DataFrames at this point since they are no longer needed?, might need to save off pop_ids

    # Run LD score regressions
    ld_coef, const_coef, se2_coef = run_ldscore_regressions(beta_arr, se_arr, ldscore_arr)

    # TODO(jonbjala) Log

    # Calculate Omegas and Sigmas
    omega = create_omega_matrix(ldscore_arr, ld_coef)
    sigma = create_sigma_matrix(se_arr, se2_coef, const_coef)

    # Check omega and sigma for validity based on positive (semi-)definiteness
    omega_valid = qc_omega(omega).reshape((omega.shape[0], 1, 1))
    sigma_valid = qc_sigma(sigma).reshape((sigma.shape[0], 1, 1))
    omega_sigma_drops = np.logical_not(np.logical_and(omega_valid, sigma_valid))

    # Run the MAMA method
    # Use identity matrices for "bad" SNPs to allow vectorized operations without having to copy
    new_betas, new_beta_ses = run_mama_method(beta_arr,
                              np.where(omega_sigma_drops, np.identity(omega.shape[1]), omega),
                              np.where(omega_sigma_drops, np.identity(sigma.shape[1]), sigma))

    return new_betas, new_beta_ses, omega_sigma_drops


ParserFunc = Callable[[str], argp.ArgumentParser]
def setup_func(argv: List[str], get_parser: ParserFunc) -> Tuple[argp.Namespace, Dict[str, Any]]:
    """
    Function to handle argument parsing, logging setup, and header printing

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    :param get_parser: Function to call to get argument parser, given a program name
    """

    # Parse the input flags using argparse
    parser = get_parser(argv[0])
    parsed_args = parser.parse_args(argv[1:])

    # Break down inputs to keep track of arguments and values specified directly by the user
    user_args = get_user_inputs(argv, parsed_args)

    # Set up the logger
    full_logfile_path = set_up_logger()

    # Log header and other information (include parsed, user-specified args at debug level)
    logging.info(HEADER)
    logging.info("See full log at: %s\n", full_logfile_path)
    logging.info("Program executed via:\n%s", format_terminal_call(argv))
    logging.debug("\nProgram was called with the following arguments:\n%s", user_args)

    return parsed_args, user_args


def main_func(argv: List[str]):
    """
    Main function that should handle all the top-level processing for this program

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    """

    # Perform argument parsing and program setup
    parsed_args, user_args = setup_func(argv, get_mama_parser)

    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        iargs = validate_inputs(parsed_args, user_args)

        # Run the MAMA pipeline
        mama_pipeline(iargs)

        # Log any remaining information (like timing info?) TODO(jonbjala)

    except Exception as ex:
        logging.exception(ex)
        sys.exit(1)


if __name__ == "__main__":

    # Call the main function
    main_func(sys.argv)
