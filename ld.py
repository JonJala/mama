#!/usr/bin/env python3

import itertools as it
import math
import sys
from typing import Optional
import timeit

from memory_profiler import profile
import numpy as np
import pandas as pd
import time

from util.df import run_filters

KBP_WINDOW = 2
CM_WINDOW = 1

DIR_PREFIX = "/var/genetics/data/1000G/public/20130502/processed/gen/"
EUR_BED_FILENAME = DIR_PREFIX + "EUR/EUR_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.bed"
EUR_BIM_FILENAME = DIR_PREFIX + "EUR/EUR_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.bim"
EUR_FAM_FILENAME = DIR_PREFIX + "EUR/EUR_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.fam"

EAS_BED_FILENAME = DIR_PREFIX + "EAS/EAS_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.bed"
EAS_BIM_FILENAME = DIR_PREFIX + "EAS/EAS_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.bim"
EAS_FAM_FILENAME = DIR_PREFIX + "EAS/EAS_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.fam"

AFR_BED_FILENAME = DIR_PREFIX + "AFR/AFR_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.bed"
AFR_BIM_FILENAME = DIR_PREFIX + "AFR/AFR_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.bim"
AFR_FAM_FILENAME = DIR_PREFIX + "AFR/AFR_chr22_phase3_shapeit2_mvncall_integrated_v5a.20130502.fam"

within_ancestry = True
dist_col = 'BP'


# Derived constants
BP_WINDOW = KBP_WINDOW * 1000
BED_FILENAME = EUR_BED_FILENAME
BIM_FILENAME = EUR_BIM_FILENAME
FAM_FILENAME = EUR_FAM_FILENAME

# Useful universal constants
BITS_PER_BYTE = 8


# BIM columns
BIM_CHR_COL = 'CHR'
BIM_RSID_COL = 'RSID'
BIM_CM_COL = 'CM'
BIM_BP_COL = 'BP'
BIM_A1_COL = 'A1'
BIM_A2_COL = 'A2'
BIM_COLUMNS = (BIM_CHR_COL, BIM_RSID_COL, BIM_CM_COL, BIM_BP_COL, BIM_A1_COL, BIM_A2_COL)

# BIM separator
BIM_SEPARATOR = "\t"

# BIM index column number
BIM_INDEX_COL_NUM = BIM_COLUMNS.index(BIM_RSID_COL)

# BIM QC filters
# TODO(jonbjala) Include filter for palindromic SNPs?  A1 == A2? Any other filters?
BIM_FILTERS = {"dup_rsids" : lambda df : df.duplicated(subset=[BIM_RSID_COL], keep=False),
               "dup_bp" : lambda df : df.duplicated(subset=[BIM_BP_COL], keep=False),
               "null_values" : lambda df : df.isnull().any(axis=1)}


def get_sample_size_from_fam_file(fam_filename: str):
    with open(fam_filename) as f:
        N = sum(1 for line in f)    

    return N


def read_bim_file(bim_filename: str):
    bim_df = pd.read_csv(bim_filename, sep=BIM_SEPARATOR, names=BIM_COLUMNS)
    return bim_df



# The first two bytes of a .bed file should be this
BED_FILE_PREFIX_MAGIC_HEX = '6c1b'
BED_FILE_PREFIX_MAGIC_BYTEARRAY = bytearray.fromhex(BED_FILE_PREFIX_MAGIC_HEX)

# The third byte of a .bed file indicating SNP- or individual-major (should be SNP)
BED_FILE_PREFIX_SNP_MAJOR_MAGIC_HEX = '01'
BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY = bytearray.fromhex(BED_FILE_PREFIX_SNP_MAJOR_MAGIC_HEX)

# Some .bed file-specific constants
BED_BITS_PER_SAMPLE = 2  # TODO(jonbjala) Calculate from BED_BINARY_TO_VALUE_MAP keys?
BED_SAMPLES_PER_BYTE = BITS_PER_BYTE // BED_BITS_PER_SAMPLE

def read_bed_file(bed_filename: str, M: int, N: int, indices: Optional[list] = None):

    # Amount of bytes to read in for each SNP
    bed_block_size_in_bytes = math.ceil(N/BED_SAMPLES_PER_BYTE)

    print("Num individuals = ", N)
    print("Num SNPs = ", M)
    print("Block size in bytes = ", bed_block_size_in_bytes)

    G = np.zeros((M, N), dtype=np.float32)
    
    with open(bed_filename, 'rb') as bed_file:
        initial_bytes = bed_file.read(3)
        if not(initial_bytes[0:2] == BED_FILE_PREFIX_MAGIC_BYTEARRAY):
            raise RuntimeError("Error: Initial bytes of bed file [0x%s] are not expected [%s].",
                initial_bytes[0:2].hex(), BED_FILE_PREFIX_MAGIC_HEX)

        if not(initial_bytes[2] == BED_FILE_PREFIX_SNP_MAJOR_MAGIC_BYTEARRAY[0]):
            raise RuntimeError("Error: BED file not in SNP major order, third byte = %s" % 
                hex(initial_bytes[2]))

        raw_bed_file_contents = bed_file.read()
        if len(raw_bed_file_contents) != M * bed_block_size_in_bytes:
            # TODO(jonbjala) Log warning?
            print("Bed file %s, which is supposed to have %s SNPs for %s individuals should "
                  "contain %s bytes of information, but contains %s." %
                  (bed_filename, M, N, M * bed_block_size_in_bytes, len(raw_bed_file_contents)))

        for i in range(0, M):
            start_byte_pos = i * bed_block_size_in_bytes
            # print(list(raw_bed_file_contents[
            #     start_byte_pos : start_byte_pos + bed_block_size_in_bytes]))
            # print("raw_bed_file_contents length = %s, start_byte_pos, start_byte_pos + bed_block_size_in_bytes = " % len(raw_bed_file_contents), (start_byte_pos, start_byte_pos + bed_block_size_in_bytes))
            G[i] = read_bed_file.BED_MAP_ARRAY[list(raw_bed_file_contents[
                start_byte_pos : start_byte_pos + bed_block_size_in_bytes])].ravel()[0:N]

    return G


# See https://www.cog-genomics.org/plink/1.9/formats#bed
read_bed_file.BED_BINARY_TO_VALUE_MAP = {
    '00' : 0.0,
    '01' : np.nan,
    '10' : 1.0,
    '11' : 2.0
}

# A bit complicated, but this is to map a byte's worth of data from a .bed file to a Numpy array
# that contains the genotype information for the entries that comprise the byte
read_bed_file.BED_BYTE_TO_VALARR_MAP = {
    int("%s%s%s%s" % tup, base=2) :
        np.array([read_bed_file.BED_BINARY_TO_VALUE_MAP[element] for element in reversed(tup)],
                 dtype=np.float32) for tup in it.product(
                     read_bed_file.BED_BINARY_TO_VALUE_MAP.keys(),
                     repeat=BED_SAMPLES_PER_BYTE)
}

read_bed_file.BED_MAP_ARRAY = np.array([read_bed_file.BED_BYTE_TO_VALARR_MAP[x]
                                        for x in sorted(
                                            read_bed_file.BED_BYTE_TO_VALARR_MAP.keys())])


# TODO(jonbjala) This needs to be written better
def write_bed_file(bed_filename: str, G: np.ndarray):
    # G is shape (M, N)
    M, N = G.shape

    # Amount of bytes to associate each SNP
    bed_block_size_in_bytes = math.ceil(N/BED_SAMPLES_PER_BYTE)
    pad_amount = bed_block_size_in_bytes * BED_SAMPLES_PER_BYTE - N

    print("Num individuals = ", N)
    print("Num SNPs = ", M)
    print("Block size in bytes = ", bed_block_size_in_bytes)
    print("Pad amount = ", pad_amount)

    padded_G = np.pad(np.nan_to_num(G, nan=write_bed_file.NAN_REPLACEMENT),
        pad_width=((0,0), (0, pad_amount)))

    temp_bytes = bytearray(bed_block_size_in_bytes)
    print("Block size in bytes = ")
    with open(bed_filename, 'wb') as bed_file:
        bed_file.write(BED_FILE_PREFIX_MAGIC_NUMBER)
        bed_file.write(BED_FILE_PREFIX_SNP_MAJOR_MAGIC_NUMBER)

        for snp in range(0, M):
            count = 0
            for cluster_start in range(0, N, BED_SAMPLES_PER_BYTE):
                map_key = tuple(padded_G[snp, cluster_start:cluster_start+BED_SAMPLES_PER_BYTE])
                temp_bytes[count] = write_bed_file.BED_VALARR_TO_BYTE_MAP[map_key]
                count += 1

            bed_file.write(temp_bytes)

# See https://www.cog-genomics.org/plink/1.9/formats#bed
write_bed_file.NAN_REPLACEMENT = -1.0
write_bed_file.BED_VALUE_TO_BINARY_MAP = {
    0.0 : '00',
    write_bed_file.NAN_REPLACEMENT : '01',
    1.0 : '10',
    2.0 : '11'
}
write_bed_file.BED_VALARR_TO_BYTE_MAP = {
    tup : int("%s%s%s%s" % tuple(write_bed_file.BED_VALUE_TO_BINARY_MAP[element] 
        for element in reversed(tup)), base=2)
            for tup in it.product(write_bed_file.BED_VALUE_TO_BINARY_MAP.keys(),
                                  repeat=BED_SAMPLES_PER_BYTE)}


# TODO(jonbjala) Set some function attributes and/or use constants
def get_population_indices(df1: pd.DataFrame, df2: pd.DataFrame = None):

    if df2 is None:
        return df1['index'].to_list()

    merged_df = df1.merge(df2, how='inner', on=[BIM_RSID_COL], suffixes=['_1', '_2'], sort=True)
    # TODO(jonbjala) Run checks / filters to make sure RSIDs and/or alleles match?
    return merged_df["index_1"].to_list(), merged_df["index_2"].to_list()



# TODO(jonbjala) Add checks for valid inputs?
def calculate_bim_window_indices(values, window_size):
    M = len(values)
    window_start = np.zeros(M, dtype=int)
    window_stop = np.full_like(window_start, M-1, dtype=int)

    current_start = 0
    for current_stop in range(M):
        while values[current_stop] - values[current_start] > window_size:
            window_stop[current_start] = current_stop - 1
            current_start += 1
        window_start[current_stop] = current_start    

    return window_start, window_stop


def get_snp_half_windows(values, window_size):
    M = len(values)
    half_window_up = np.zeros(M, dtype=int)
    half_window_down = np.zeros(M, dtype=int)

    cur_window_start = 0
    cur_window_end = 0
    most_recent_valid_index = 0
    for snp_num in range(M):

        if pd.isnull(values[snp_num]):
            continue

        threshold = window_size + values[snp_num]
        while cur_window_end < M and not values[cur_window_end] > threshold:
            if not pd.isnull(values[cur_window_end]):
                most_recent_valid_index = cur_window_end
                half_window_up[most_recent_valid_index] = most_recent_valid_index - snp_num + 1

            cur_window_end += 1

        half_window_down[snp_num] = most_recent_valid_index - snp_num + 1

    return half_window_up, half_window_down



def calculate_N_and_R_bands(G, band_size, banded_start, banded_stop):
    M, N = G.shape
    offset = (band_size - 1) // 2

    scratchpad_for_G = np.zeros((band_size, N))

    banded_R = np.zeros((M, band_size))
    banded_N = np.zeros((M, band_size))
    for i in range(M):
        if i % 100 == 0:
            print("Processing SNP", i)
        start = banded_start[i]
        stop = banded_stop[i]

        abs_start = banded_start[i] - offset + i
        abs_stop = banded_stop[i] - offset + i

        scratchpad_for_G[start : stop] = G[i] * G[abs_start : abs_stop]
        banded_R[i][start : stop] = np.nanmean(scratchpad_for_G[start : stop], axis=1)
        banded_N[i][start : stop] = np.reciprocal(
            np.count_nonzero(~np.isnan(scratchpad_for_G[start : stop]), axis=1) - 2.0)

    return banded_R.T, banded_N.T


def main_func():
    #files = {'EUR' : {'bed' : EUR_BED_FILENAME, 'bim' : EUR_BIM_FILENAME, 'fam' : EUR_FAM_FILENAME},
             # 'EAS' : {'bed' : EAS_BED_FILENAME, 'bim' : EAS_BIM_FILENAME, 'fam' : EAS_FAM_FILENAME},
             # 'AFR' : {'bed' : AFR_BED_FILENAME, 'bim' : AFR_BIM_FILENAME, 'fam' : AFR_FAM_FILENAME},
    #    }
    # files = {'EAS' : {'bed' : '/var/genetics/data/1000G/public/latest/processed/misc/legacy/geno_ancestry/chr1_EAS_mind02_geno02_maf01.bed', 'bim' : '/var/genetics/data/1000G/public/latest/processed/misc/legacy/geno_ancestry/chr1_EAS_mind02_geno02_maf01.bim', 'fam' : '/var/genetics/data/1000G/public/latest/processed/misc/legacy/geno_ancestry/chr1_EAS_mind02_geno02_maf01.fam'}}
    files = {'DUMMY_POP' : {'bed' : './dummy2.bed', 'bim' : './dummy2.bim', 'fam' : './dummy2.fam'}}
    
    populations = list(files.keys())

    # Get sample sizes for each population
    N_total = {p : get_sample_size_from_fam_file(files[p]['fam']) for p in populations}
    N_max = max(N_total.values())



    # Merge the bim files into one dataframe using the default outer join
    bim_dfs = {p : read_bim_file(files[p]['bim']) for p in populations}
    M_total = {p : len(bim_dfs[p]) for p in populations}
    # bim_union_df = pd.concat(list(bim_dfs.values()), axis=1, keys=populations)
    # print("bim_df length = ", len(bim_union_df))
    # Get the total number of SNPS in the union of populations
    # M_union = len(bim_union_df)

    # TODO(jonbjala) Potentially check to make sure CHR and BP values agree across populations...what to do about A1/A1 mismatches?


    # Generate windows for all populations and get the max half-window size over all SNPs and pops
    # window_size = CM_WINDOW
    # dist_col = 'CM'
    # window_size = BP_WINDOW
    # dist_col = 'BP'
    window_size = 50
    dist_col = 'CM'


    half_window_sizes = {p : get_snp_half_windows(bim_dfs[p][dist_col], window_size) for p in populations}
    print("half_window_sizes = \n", half_window_sizes)
    max_half_window = {p : max(w.max() for w in half_window_sizes[p]) for p in populations}
    print("max_half_window = \n", max_half_window)
    band_size = {p : 2 * max_half_window[p] - 1 for p in populations}
    print("band_size = \n", band_size)
    banded_start = {p : max_half_window[p] - half_window_sizes[p][0] for p in populations}
    banded_stop = {p : max_half_window[p] - 1 + half_window_sizes[p][1] for p in populations}

    # TODO(jonbjala) Delete some of the dfs?
    # del bim_union_df
    for p in populations:
        del bim_dfs[p]
    del bim_dfs

    # Generate pared down (half-window) N and R matrices (for all populations)
    banded_R = dict()
    banded_N = dict()
    for p in populations:
        print("Calculating R and N for", p)
        G_for_pop = read_bed_file(files[p]['bed'], M_total[p], N_total[p])
        print("Original G = \n", G_for_pop)
        G_for_pop = G_for_pop - np.nanmean(G_for_pop, axis=1)[:, np.newaxis]
        print("Demeaned G = \n", G_for_pop)
        # G_for_pop = G_for_pop / np.nanstd(G_for_pop, axis=1)[:, np.newaxis]
        # print("Standardized G = \n", G_for_pop)
        banded_R[p], banded_N[p] = calculate_N_and_R_bands(G_for_pop, band_size[p],
                                                           banded_start[p], banded_stop[p])
        print("banded_R = \n", banded_R[p])
        print("banded_N = \n", banded_N[p])
    del G_for_pop
    print("Done calculating R and N! ")
    # Calculate LD scores
    ld_scores = {p : np.zeros(M_total[p]) for p in populations}
    for p in populations:
        diag_row = max_half_window[p] - 1
        print("diag_row = ", diag_row)
        print("Num SNPs in %s:" % p, M_total[p])
        print("max_half_window in %s:" % p, max_half_window[p])
        print()
        scores = ld_scores[p]
        r_band_square = np.square(banded_R[p])
        n_band = banded_N[p]
        
        diag = banded_R[p][diag_row]
        print("diag = \n", diag)
        print("diag shape = \n", diag.shape)
        start_index = banded_start[p]
        stop_index = banded_stop[p]

        correction = np.zeros_like(n_band)
        for snp_num in range(M_total[p]):
            print("Processing SNP ", snp_num)
            print("start_index = ", start_index[snp_num])
            print("stop_index = ", stop_index[snp_num])
            print("Shape of X = ", correction[start_index[snp_num]:stop_index[snp_num], snp_num].shape)
            print("Shape of Y = ", diag[snp_num].shape)
            print("Z = ", diag[0:stop_index[snp_num]-start_index[snp_num]])
            print("Shape of Z = ", diag[0:stop_index[snp_num]-start_index[snp_num]].shape)
            correction[start_index[snp_num]:stop_index[snp_num], snp_num] = diag[0:stop_index[snp_num]-start_index[snp_num]] / diag[snp_num]
        print("intermediate correction = \n", correction)
        print("r square = \n", r_band_square)
        r_band_square /= np.square(diag) # Becomes rhat_band_square
        print("r hat square = \n", r_band_square)
        correction -= r_band_square
        correction *= n_band

        print("correction = \n", correction)
        r_band_square -= correction


        print("corrected r_hat square = \n", r_band_square)
        scores = np.sum(r_band_square, axis=0)

    print()
    print(scores)


# Fill with hardcoded G matrix (testcase without nan's)
#
# 4 SNPs, 3 PEOPLE
#
#  0 1 2
#  1 1 2
#  1 0 0 
#  2 2 1
#
def fill_bed_file_non_nan(file_name: str):
    with open(file_name, 'wb') as bed_file:
        bed_file.write(bytearray.fromhex('6c1b01'))

        bed_file.write(bytearray.fromhex('38'))
        bed_file.write(bytearray.fromhex('3a'))
        bed_file.write(bytearray.fromhex('02'))
        bed_file.write(bytearray.fromhex('2f')) 



def qc_bim_df(bim_df: pd.DataFrame):
    # QC the bim dataframe
    cumulative_drops, drop_dict = run_filters(bim_df, BIM_FILTERS)
    bim_df.drop(bim_df.index[cumulative_drops], inplace=True)

    # TODO(jonbjala) Log information about df filtering / QC?

    # Make the index a separate column to allow for filtering at the .bed processing step
    bim_df.reset_index(inplace=True)

    # TODO(jonbjala) Need for something like bim_df.sort_values(by=['bp'])?
    return bim_df


def read_and_qc_bim_file(pop_name: str, bim_filename: str):

    # TODO(jonbjala) Replace with logging
    print("Processing bim for population %s" % pop_name)

    # Read in the file from disk
    bim_df = read_bim_file(bim_filename)
    orig_M = len(bim_df)
    # TODO(jonbjala) Log information about raw df?

    # QC the bim dataframe
    return orig_M, qc_bim_df(bim_df)


def calculate_Nrecip_and_R_with_nan(G, band_size, lower_portion):
    M, N = G.shape

    mid_index_of_result = (band_size - 1) // 2
    max_dist_from_diag = (band_size + 1) // 2

    scratchpad = np.zeros((max_dist_from_diag, N))

    banded_R = np.zeros((band_size, M))
    banded_N = np.zeros((band_size, M))

    print("M = ", M)
    print("N = ", N)
    print("band_size = ", band_size)
    print("mid_index_of_result = ", mid_index_of_result)
    print("max_dist_from_diag = ", max_dist_from_diag)
    print("lower_portion = ", lower_portion)

    # Create indices used to copy some entries in result to make use of symmetry
    result_diag_row_indices = np.arange(mid_index_of_result, -1, -1)
    result_diag_col_indices = np.arange(0, mid_index_of_result + 1)

    # Iterate through the SNPs and calculate the R and N results for each
    for snp_num in range(M):

        if snp_num % 100 == 0:
            print("Processing SNP", snp_num)

        # Create the slices needed for indexing into the various matrices
        scratch_slice = slice(lower_portion[snp_num])
        g_slice = slice(snp_num, snp_num + lower_portion[snp_num])
        result_slice = slice(mid_index_of_result,
            min(mid_index_of_result+lower_portion[snp_num], band_size - 1))

        # Calculate the intermediate values needed to calculate R and N
        scratchpad[scratch_slice] = G[snp_num] * G[g_slice]

        # Calculate the results based on the intermediate values
        banded_R[result_slice, snp_num] = np.nanmean(scratchpad[scratch_slice], axis=1)
        banded_N[result_slice, snp_num] = np.reciprocal(
            np.count_nonzero(~np.isnan(scratchpad[scratch_slice]), axis=1) - 2.0)

        # Copy the results to the portion of the result that are the same due to symmetry
        banded_R[result_diag_row_indices[scratch_slice], result_diag_col_indices[scratch_slice]] =\
            banded_R[result_slice, snp_num]
        banded_N[result_diag_row_indices[scratch_slice], result_diag_col_indices[scratch_slice]] =\
            banded_N[result_slice, snp_num]

        # Update the column indices of the diagonal portion that needs to be copied
        result_diag_col_indices += 1

    return banded_R, banded_N


# TODO(jonbjala) Need to determine whether matrices are filtered to intersection of SNPs before
#                calling this function or not.  For now, assume they are.
# TODO(jonbjala) N_recip is really only used for single ancestry
def calculate_ld_scores(r_1: np.ndarray, n_recip_1: np.ndarray,
                        r_2: np.ndarray = None, n_recip_2: np.ndarray = None):

    one_anc = r_2 is None
    if one_anc:
        r_2 = r_1
        n_recip_2 = n_recip_1

    bandwidth_1, M = r_1.shape
    bandwidth_2 = r_2.shape[0]
    min_bandwidth = min(bandwidth_1, bandwidth_2)

    diag_row_1 = (bandwidth_1 - 1) // 2
    diag_row_2 = (bandwidth_2 - 1) // 2
    min_diag_row = (min_bandwidth - 1) // 2

    # Calculate the slices used for indexing into rows of the input matrices
    vslice_1 = slice(diag_row_1 - min_diag_row, diag_row_1 + min_diag_row + 1)
    vslice_2 = vslice_1 if one_anc \
        else slice(diag_row_2 - min_diag_row, diag_row_2 + min_diag_row + 1)

    # Create scratchpad matrix that will eventually produce the ld scores result
    # Start with product of R matrices, divided through by the R diagonal (which is a row here)
    scratch = np.multiply(r_1[vslice_1], r_2[vslice_2])
    np.divide(scratch, scratch[min_diag_row], out=scratch)
    # TODO(jonbjala) Replace nan or inf by 0?  Actually, just using nansum below for now

    # If single ancestry, need to correct the values
    if one_anc:
        correction = np.zeros_like(scratch)
        r_diag = r_1[min_diag_row]
        print("r_diag = \n", r_diag)
        # TODO(jonbjala) Need to handle case where M - min_diag_row < min_diag_row?
        for snp_num in range(min_diag_row):            
            offset = min_diag_row - snp_num
            np.divide(r_diag[0 : min_bandwidth - offset], r_diag[snp_num],
                out=correction[offset:, snp_num])
            correction[offset:, snp_num] -= scratch[offset:, snp_num]
            correction[offset:, snp_num] *= n_recip_1[offset:, snp_num]

        for snp_num in range(min_diag_row, M - min_diag_row):
            np.divide(r_diag[snp_num - min_diag_row :
                snp_num + min_diag_row + 1], r_diag[snp_num], out=correction[:, snp_num])
            correction[:, snp_num] -= scratch[:, snp_num]
            correction[:, snp_num] *= n_recip_1[:, snp_num]

        for snp_num in range(M - min_diag_row, M):
            offset = M - snp_num
            np.divide(r_diag[snp_num : M], r_diag[snp_num],
                out=correction[:offset, snp_num])
            correction[:offset, snp_num] -= scratch[:offset, snp_num]
            correction[:offset, snp_num] *= n_recip_1[:offset, snp_num]

        scratch -= correction

    ld_scores = np.nansum(scratch, axis=0)
    return ld_scores


def main_func_2():
    # files = {'DUMMY_POP' : {'bed' : './dummy2.bed', 'bim' : './dummy2.bim', 'fam' : './dummy2.fam'}}
    files = {'EUR' : {'bed' : EUR_BED_FILENAME, 'bim' : EUR_BIM_FILENAME, 'fam' : EUR_FAM_FILENAME},
             'EAS' : {'bed' : EAS_BED_FILENAME, 'bim' : EAS_BIM_FILENAME, 'fam' : EAS_FAM_FILENAME},
             'AFR' : {'bed' : AFR_BED_FILENAME, 'bim' : AFR_BIM_FILENAME, 'fam' : AFR_FAM_FILENAME},
            }
    # files = {'POP1' : {'bed' : "./POP1.bed", 'bim' : "./POP1.bim", 'fam' : "./POP1.fam"},
    #          'POP2' : {'bed' : "./POP2.bed", 'bim' : "./POP2.bim", 'fam' : "./POP2.fam"},
    #          'POP3' : {'bed' : "./POP3.bed", 'bim' : "./POP3.bim", 'fam' : "./POP3.fam"},
    #         }
    populations = tuple(files.keys())

    # Get sample sizes for each population
    N = {p : get_sample_size_from_fam_file(files[p]['fam']) for p in populations}

    # TODO(jonbjala) Need to add chromosome processing / support (for now, assume 1 chromosome / chromosome numbers irrelevant)

    # Read in bim files and do some initial QC / filtering
    bim_dfs = dict()
    orig_M = dict()
    for p in populations:
        orig_M[p], bim_dfs[p] = read_and_qc_bim_file(p, files[p]['bim'])

    # Save off indices to allow for filtering / slicing of .bed
    bed_filtering_indices = {p : bim_dfs[p]['index'].to_list() for p in populations}

    # Get the number of SNPs for each population
    M = {p : len(bim_dfs[p]) for p in populations}

    # Create cross-population indices (indices to use to process the same SNP in each)
    for p in populations:
        bim_dfs[p]['index'] = range(M[p])
    # TODO(jonbjala) Can maybe also use: map(lambda df: df['index'] = range(len(df)))
    # TODO(jonbjala) Not sure if the p1 = p2 is really needed
    crosspop_indices = {p : dict() for p in populations}
    for p1 in populations:
        for p2 in populations:
            if p1 == p2:
                crosspop_indices[p1][p2] = get_population_indices(bim_dfs[p1])
            else:
                crosspop_indices[p1][p2], crosspop_indices[p2][p1] =\
                    get_population_indices(bim_dfs[p1], bim_dfs[p2])


    # Generate band sizes
    dist_col = 'BP' # TODO(jonbjala) Set this correctly
    win_size = 100 # TODO(jonbjala) Set this correctly
    start_indices = dict()
    stop_indices = dict()
    max_half_window = dict()
    bandwidth = dict()
    upper_portion = dict()
    lower_portion = dict()
    for p in populations:
        # TODO(jonbjala) This can maybe be cleaned up a bit / streamlined
        start_indices[p], stop_indices[p] = calculate_bim_window_indices(
            bim_dfs[p][dist_col].astype(int), win_size)

        diag_indices = np.arange(len(start_indices[p]), dtype=int)
        lower_portion[p] = stop_indices[p] - diag_indices
        upper_portion[p] = diag_indices - start_indices[p]
        max_half_window[p] = max(upper_portion[p].max(), lower_portion[p].max())
        bandwidth[p] = 2 * max_half_window[p] + 1

    
    # Clear dictionary / free up memory
    # TODO(jonbjala): Does this need to be done here / do we need to save anything off?
    bim_dfs.clear()
    del bim_dfs


    n_recip_banded = dict()
    r_banded = dict()
    for p in populations:
        print("Calculating and writing R and N matrices for population %s" % p)
        # print("\tStart indices %s" % start_indices[p])
        # print("\tStop indices %s" % stop_indices[p])
        # print("\tMax half window %s" % max_half_window[p])
        # print("\tBandwidth %s" % bandwidth[p])
        # print()
        # print("\tBed filtering indices %s" % bed_filtering_indices[p])
        # print("\tCross-pop indices:")
        # for q in populations:
        #     print("\t\t%s: %s" % (q, crosspop_indices[p][q]))
        # orig_G = read_bed_file(files[p]['bed'], orig_M[p], N[p])
        G_for_pop = read_bed_file(files[p]['bed'], orig_M[p], N[p])[bed_filtering_indices[p], :]
        mean_per_snp = np.nanmean(G_for_pop, axis=1)
        G_for_pop -= mean_per_snp[:, np.newaxis] # Need newaxis to allow for broadcast

        # TODO(jonbjala) Allow for standardizing

        # print("Orig M = ", orig_M[p])
        # print("Num bed filtering indices = ", len(bed_filtering_indices[p]))
        # print("M = ", M[p])
        # print("Unfiltered G =\n", orig_G)
        # print("Filtered G = \n", G_for_pop)
        r_banded[p], n_recip_banded[p] = calculate_Nrecip_and_R_with_nan(G_for_pop, bandwidth[p],
                                                                         lower_portion[p])
        # TODO(jonbjala) Perhaps store N and R separately to avoid having to load both in always?
        np.savez("./%s_RN" % p, R=r_banded[p], allow_pickle=False, N_RECIP=n_recip_banded[p])
    del G_for_pop
    del mean_per_snp

    start_indices.clear()
    del start_indices

    stop_indices.clear()
    del stop_indices

    max_half_window.clear()
    del max_half_window

    bandwidth.clear()
    del bandwidth

    upper_portion.clear()
    del upper_portion

    lower_portion.clear()
    del lower_portion

    r_banded.clear()
    del r_banded

    n_recip_banded.clear()
    del n_recip_banded


    # TODO(jonbjala) Could do a fancier ordering of loading in the data, slightly reduce redundancy
    for p1_index in range(len(populations)):
        p1 = populations[p1_index]
        with np.load("./%s_RN.npz" % p1, allow_pickle=False) as p1_data:
            p1_r = p1_data['R']  # TODO(jonbjala) Use constants here and previously
            p1_n_recip = p1_data['N_RECIP']

        # Perform single population LD score calculations
        ld_scores = calculate_ld_scores(p1_r, p1_n_recip)
        print("ld_scores for population %s = \n" % p1, ld_scores)
        del p1_n_recip

        # Perform cross-population LD score calculations
        for p2_index in range(p1_index + 1, len(populations)):
            p2 = populations[p2_index]
            with np.load("./%s_RN.npz" % p2, allow_pickle=False) as p2_data:
                p2_r = p2_data['R']  # TODO(jonbjala) Use constants here and previously
            ld_scores = calculate_ld_scores(p1_r, None, p2_r) # TODO(jonbjala) Fix this call structure
            print("ld_scores for populations %s and %s = \n" % (p1, p2), ld_scores)




# for p in populations:
#         print("Calculating R and N for", p)
#         G_for_pop = read_bed_file(files[p]['bed'], M_total[p], N_total[p])
#         print("Original G = \n", G_for_pop)
#         G_for_pop = G_for_pop - np.nanmean(G_for_pop, axis=1)[:, np.newaxis]
#         print("Demeaned G = \n", G_for_pop)
#         # G_for_pop = G_for_pop / np.nanstd(G_for_pop, axis=1)[:, np.newaxis]
#         # print("Standardized G = \n", G_for_pop)
#         banded_R[p], banded_N[p] = calculate_N_and_R_bands(G_for_pop, band_size[p],
#                                                            banded_start[p], banded_stop[p])
#         print("banded_R = \n", banded_R[p])
#         print("banded_N = \n", banded_N[p])
#     del G_for_pop
#     print("Done calculating R and N! ")

if __name__ == '__main__':



    # G = read_bed_file("./dummy2.bed", 4, 3)  # Currently dummy2 is the non-nan testcase
    # print(G)


    # exit()
    # M = len(read_bim_file("./dummy.bim"))
    # N = get_sample_size_from_fam_file("./dummy.fam")
    # print("M, N for dummy = ", M, N)


    # G = read_bed_file("./dummy.bed", M, N)
    # print("G = \n", G)

    # write_bed_file("./dummy_test.bed", G)
    # G_test = read_bed_file("./dummy_test.bed", M, N)
    # print("G_test = \n", G)

    # G_POP1 = np.zeros((11, 5))
    # G_POP2 = np.zeros((9, 5))
    # G_POP3 = np.zeros((9, 5))


    # G_POP1[0, :] = [0.0, np.nan, 2.0, 1.0, 1.0]
    # G_POP1[1, :] = [1.0, np.nan, 2.0, 1.0, 1.0]
    # G_POP1[2, :] = [2.0, np.nan, 2.0, 1.0, 1.0]
    # G_POP1[4, :] = [np.nan, np.nan, 2.0, 1.0, 1.0]
    # G_POP1[5, :] = [0.0, np.nan, 2.0, 1.0, 1.0]
    # G_POP1[6, :] = [1.0, np.nan, 2.0, 1.0, 1.0]

    # G_POP2[0, :] = [1.0, np.nan, 2.0, 1.0, 0.0]
    # G_POP2[2, :] = [2.0, np.nan, 2.0, 1.0, 1.0]
    # G_POP2[3, :] = [1.0, np.nan, 2.0, 1.0, 0.0]
    # G_POP2[4, :] = [2.0, np.nan, 2.0, 1.0, 1.0]
    # G_POP2[5, :] = [1.0, np.nan, 2.0, 1.0, 0.0]
    # G_POP2[6, :] = [2.0, np.nan, 2.0, 1.0, 1.0]

    # G_POP3[1, :] = [0.0, np.nan, 2.0, 1.0, 2.0]
    # G_POP3[2, :] = [0.0, np.nan, 2.0, 1.0, 0.0]
    # G_POP3[3, :] = [0.0, np.nan, 2.0, 1.0, 0.0]
    # G_POP3[4, :] = [0.0, np.nan, 2.0, 1.0, 2.0]
    # G_POP3[5, :] = [0.0, np.nan, 2.0, 1.0, 2.0]
    # G_POP3[6, :] = [0.0, np.nan, 2.0, 1.0, np.nan]

    # # print("G_POP1 = \n", G_POP1)
    # # print("G_POP2 = \n", G_POP2)
    # # print("G_POP3 = \n", G_POP3)


    # write_bed_file("./POP1.bed", G_POP1)
    # write_bed_file("./POP2.bed", G_POP2)
    # write_bed_file("./POP3.bed", G_POP3)


    main_func_2()
    exit()



