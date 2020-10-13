#!/usr/bin/env python3

"""
Python functions to process summary statistics
"""

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from util.df import Filter, FilterMap, rename_dataframe_cols, run_filters


# Constants / Parameters / Types #############

# Standard column names
SNP_COL = 'SNP'
BP_COL = 'BP'
CHR_COL = 'CHR'
BETA_COL = 'BETA'
FREQ_COL = 'FREQ'
SE_COL = 'SE'
A1_COL = 'A1'
A2_COL = 'A2'


# Construct useful base-pair related constant sets / maps
COMPLEMENT = {
    "A" : "T",
    "T" : "A",
    "C" : "G",
    "G" : "C",
}
BASES = set(COMPLEMENT.keys())


# Functions ##################################

#################################
def create_freq_filter(min_freq: float, max_freq: float) -> Filter:
    return lambda df: ~df[FREQ_COL].between(min_freq, max_freq)


#################################
def qc_sumstats(sumstats_df: pd.DataFrame, filters: FilterMap,
                column_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.Series,
                                                     Dict[str, pd.Series], List]:
    """
    Runs QC steps like renaming columns and dropping rows based on filters

    :param sumstats_df: Dataframe holding summary statistics data
    :param column_map: Dictionary of column name in the sumstats_df mapped to standard column name

    :return: Tuple containing:
             1) A modified copy of the input data frame (SNPs dropped, columns renamed, etc.)
             2) The indices of the union of SNPs being dropped, and
             3) A dictionary mapping filter name to an ordered collection (pd.Series) of
                booleans indicating which SNPs to drop for that filter
             4) A list containing the rsIDs of SNPs that are dropped due to being duplicates
    """

    # Make copy of the dataframe (this copy will be modified)
    df = sumstats_df.copy()

    # Rename columns to standardized names
    rename_dataframe_cols(df, column_map)

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


#################################
def flip_alleles(df: pd.DataFrame, flip_indices: pd.Series):
    """
    Given an Series of Booleans (corresponding to rows/indices in the df input parameter), flips
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


#################################
def standardize_all_sumstats(sumstats: Dict[Any, pd.DataFrame],
                             ref: Tuple[Any, pd.DataFrame]=()
                             ) -> Tuple[Any, pd.Series, pd.Series, Dict[Any, pd.Series]]:
    """
    Takes a collection of summary statistics DataFrames and standardizes them according to a
    designated reference summary statistics DataFrame.  If a reference is not specified, the first
    summary statistics DataFrame from the inputs will be chosen as such.

    Standardization involves:
    	1) Identifying (and adjusting) SNPs that are reference allele-flipped
    	2) Identifying SNPs that should be dropped based on a total reference allele mismatch
    	   in at least one summary statistics DataFrame compared to the reference DataFrame

    :param sumstats: Dictionary mapping some kind of ID to a DataFrame holding summary
                     stat information.  The DFs should all have been QCed already and should all
                     match SNP lists exactly.
    :param ref: Reference DataFrame (used as a source of ground truth for major/minor alleles)
                Assumed to be QCed and match SNP lists with DFs in sumstats input parameter

    :return: A tuple containing:
             1) The ID of the reference population
             2) The indices that should be dropped due to a mismatch of at least one GWAS
             3) The recommended drops (as indices) broken down by population
             4) A mapping of ID to indices where a reference allele swap took place
    """

    # Get list of name / DataFrame pairs by population
    ss_pops = list(sumstats.items())

    # Determine reference population name and DataFrame (if not supplied just designate one)
    ref = ref if ref else ss_pops[0]
    ref_id = ref[0]
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

    return ref_id, cumulative_drop_indices, drop_dict, ref_flip_dict


#################################
def write_sumstats_to_file(filename: str, df: pd.DataFrame):
    """
    Helper function that writes a summary statistics DataFrame to disk

    :param filename: Full path to output file
    :param df: DataFrame holding the summary statistics
    """
    df.to_csv(filename, sep="\t", index_label=SNP_COL)


#################################
def process_sumstats(initial_df: pd.DataFrame, re_expr_map: Dict[str, str],
                     req_std_cols: Set[str], filters: Dict[str, Tuple[Filter, str]],
                     column_map: Dict[str, str] = None) -> pd.DataFrame:
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
    :param filters: Map of filter functions (and descriptions) used to drop undesired SNPs

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
    filter_map = {f_name : f_func for (f_name, (f_func, f_desc)) in filters.items()}
    qc_df, drop_indices, per_filt_drop_map, dups = qc_sumstats(initial_df, filter_map, column_map)

    # Log SNP drop info
    for filt_name, filt_drops in per_filt_drop_map.items():
        logging.info("Filtered out %d SNPs with \"%s\" (%s)", filt_drops.sum(), filt_name,
            filters.get(filt_name, "No description available")[1])
        if logging.root.level <= logging.DEBUG:
            logging.debug("\tRS IDs = %s\n", initial_df.index[filt_drops].to_list())
    logging.info("\nFiltered out %d SNPs in total (as the union of drops, this may be "
                 "less than the total of all the per-filter drops)", drop_indices.sum())
    logging.info("Additionally dropped %d duplicate SNPs", len(dups))
    if logging.root.level <= logging.DEBUG:
        logging.debug("\tRS IDs = %s\n", dups)

    return qc_df