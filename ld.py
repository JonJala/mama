#!/usr/bin/env python3

import gc
import itertools as it
import logging
import math
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import time

from util.bim import (read_bim_file, BIM_COLUMNS, BIM_CHR_COL, BIM_RSID_COL, BIM_CM_COL,
                      BIM_BP_COL, BIM_A1_COL, BIM_A2_COL, BIM_SEPARATOR)
from util.df import run_filters


# BIM QC filters
# TODO(jonbjala) Include filter for palindromic SNPs?  A1 == A2? Any other filters?
DUP_RSID_FILTER_NAME = "dup_rsids"
DUP_BP_FILTER_NAME = "dup_bp"
NULL_VAL_FILTER_NAME = "null_values"
BIM_FILTERS = {DUP_RSID_FILTER_NAME : lambda df : df.duplicated(subset=[BIM_RSID_COL], keep=False),
               DUP_BP_FILTER_NAME : lambda df : df.duplicated(subset=[BIM_BP_COL], keep=False),
               NULL_VAL_FILTER_NAME : lambda df : df.isnull().any(axis=1)}


# TODO(jonbjala) Set some function attributes and/or use constants
def get_population_indices(df1: pd.DataFrame, df2: pd.DataFrame = None):

    if df2 is None:
        return df1['index'].to_list()

    merged_df = df1.merge(df2, how='inner', on=[BIM_RSID_COL], suffixes=['_1', '_2'], sort=True)
    # TODO(jonbjala) Run checks / filters to make sure RSIDs and/or alleles match?
    return merged_df["index_1"].to_list(), merged_df["index_2"].to_list()


# TODO(jonbjala) Add checks for valid inputs?
def calculate_lower_extents(values, window_size):
    M = len(values)
    lower_extent = np.arange(M, 0, -1, dtype=int)
    current_start = 0
    for current_stop in range(M):
        while values[current_stop] - values[current_start] > window_size:
            lower_extent[current_start] = current_stop - current_start
            current_start += 1

    return lower_extent


def qc_bim_df(bim_df: pd.DataFrame, drop: bool=True):
    # QC the bim dataframe
    cumulative_drops, drop_dict = run_filters(bim_df, BIM_FILTERS)
    if drop:
        bim_df.drop(bim_df.index[cumulative_drops], inplace=True)
        for filt_name, drop_indices in drop_dict.items():
            print("\tDropped %s SNPs using %s filter: %s" %
                (np.count_nonzero(drop_indices), filt_name, bim_df[BIM_RSID_COL][drop_indices].to_list()))

        # TODO(jonbjala) Log information about df filtering / QC?

        # Make the index a separate column to allow for filtering at the .bed processing step
        bim_df.reset_index(inplace=True)

    # TODO(jonbjala) Need for something like bim_df.sort_values(by=['bp'])?
    return bim_df, cumulative_drops, drop_dict


def read_and_qc_bim_file(bim_filename: str):

    # TODO(jonbjala) Replace with logging
    print("Processing bim file: [%s]" % bim_filename)

    # Read in the file from disk
    bim_df = read_bim_file(bim_filename)
    orig_M = len(bim_df)
    # TODO(jonbjala) Log information about raw df?

    # QC the bim dataframe
    return orig_M, qc_bim_df(bim_df)[0]


def calculate_Nrecip_and_R_with_nan(G, max_lower_extent, lower_extents):
    M, N = G.shape

    scratchpad = np.zeros((max_lower_extent, N))

    banded_R = np.zeros((max_lower_extent, M))
    banded_N = np.zeros((max_lower_extent, M))

    # Iterate through the SNPs and calculate the R and N results for each
    for snp_num in range(M):

        # Create the slices needed for indexing into the various matrices
        scratch_slice = slice(lower_extents[snp_num])
        g_slice = slice(snp_num, snp_num + lower_extents[snp_num])
        result_slice = scratch_slice

        # Calculate the intermediate values needed to calculate R and N
        scratchpad[scratch_slice] = G[snp_num] * G[g_slice]

        # Calculate the results based on the intermediate values
        banded_R[result_slice, snp_num] = np.nanmean(scratchpad[scratch_slice], axis=1)
        banded_N[result_slice, snp_num] = np.reciprocal(
            np.count_nonzero(~np.isnan(scratchpad[scratch_slice]), axis=1) - 2.0)

    return banded_R, banded_N


def calculate_R_without_nan(G, lower_extents, step_size=100, mmap_prefix='./'):
    M, N = G.shape
    max_lower_extent = max(lower_extents)
    logging.info("Calculating R")
    scratchpad = np.zeros((max_lower_extent + step_size, step_size), dtype=np.float32)

    try:
        banded_R = np.zeros((max_lower_extent, M), dtype=np.float32)
    except MemoryError:
        logging.info("Insufficient memory for R matrix, using memmap.  Expect slower performance.")
        banded_R = np.memmap(f"{mmap_prefix}rband_mmap.dat", dtype=np.float32,
                             mode="w+", shape=(max_lower_extent, M))
    
    for left_snp in range(0, M, step_size):
        next_left_snp = min(left_snp + step_size, M)
        block_width = next_left_snp - left_snp
        rect_length = block_width + lower_extents[next_left_snp - 1] - 1

        np.matmul(G[left_snp:left_snp+rect_length, :],
                  G[left_snp:next_left_snp, :].T, out=scratchpad[0:rect_length,0:block_width])

        for snp_num in range(left_snp, next_left_snp):
            offset = snp_num - left_snp
            banded_R[0:lower_extents[snp_num], snp_num] =\
                scratchpad[offset:offset + lower_extents[snp_num], offset]

    banded_R *= np.reciprocal(float(N), dtype=np.float32)
    if type(banded_R) == np.memmap:
        banded_R.flush()
    logging.info("Done calculating R")
    return banded_R


# TODO(jonbjala) N is really only used for single ancestry
# Assumes matrices are filtered to common SNPs before being passed in
def calculate_ld_scores(banded_r: Tuple[np.ndarray], N: float = 3.0, lower_extents: np.array = None):

    # TODO(jonbjala) Include more input checks
    one_anc = len(banded_r) == 1

    # Input parameter checking plus creating some aliases for readability
    r_1 = banded_r[0]
    r_2 = r_1 if one_anc else banded_r[1]

    extent_1, M_1 = r_1.shape
    extent_2, M_2 = r_2.shape
    if (M_1 != M_2):
        pass  # TODO(jonbjala) Throw error

    M = M_1
    joint_extent = min(extent_1, extent_2)
    
    # Start with product of R matrices, divided through by the R diagonal (which is a row here)
    logging.debug("Calculating correlation product / squared correlation...")
    r_prod = np.multiply(r_1[0:joint_extent], r_2[0:joint_extent])
    final_divisor = r_prod[0].copy()
    
    # Sum up the R-product entries as a start to the ld_scores (needs correction if single ancestry)
    ld_scores = np.sum(r_prod, axis=0)
    for offset in range(1, joint_extent):
        ld_scores[offset:M] += r_prod[offset, 0:M-offset]

    # If single ancestry, need to correct the values
    if one_anc:
        logging.debug("Calculating correction...")
        diag_row = r_1[0]

        ld_scores *= (N - 1.0)

        # Reuse r_prod matrix (since it isn't needed anymore) to reduce memory usage
        final_divisor *= (N - 2.0)
        r_prod.fill(0.0)
        diag_element_products = r_prod
        for snp_num in range(M):
            end_offset = min(lower_extents[snp_num], M - snp_num)
            diag_element_products[:end_offset, snp_num] =\
                diag_row[snp_num] * diag_row[snp_num:snp_num+end_offset]

        # Calculate correction using diagonal plus bottom half of matrix bands
        correction = np.sum(diag_element_products, axis=0)

        # Now include the terms from the top half (but not the diagonal, which was already included)
        for offset in range(1, joint_extent):
            correction[offset:M] += diag_element_products[offset, 0:M-offset]
        ld_scores -= correction

        del diag_element_products
        del correction

    del r_prod
    gc.collect()

    # Divide through by the R product diagonal
    logging.debug("Dividing through by product diagonal")

    ld_scores /= final_divisor
    logging.debug("Complete")
    return ld_scores
