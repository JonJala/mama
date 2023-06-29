import logging

import numpy as np
import pandas as pd

from ld import (calculate_R_without_nan, calculate_ld_scores, calculate_lower_extents,
                get_population_indices, read_bim_file, qc_bim_df, read_and_qc_bim_file)
from util.bed import read_bed_file
from util.bim import BIM_RSID_COL
from util.df import run_filters
from util.fam import get_sample_size_from_fam_file

# TODO(jonbjala) Allow possibility of caching the R matrix?



class PopInfo:

    # Drop SNPs with variance below this number (.bed processing)
    SNP_VAR_THRESHOLD = 1.0e-8

    # Display up to this many SNPs when debug logging
    MAX_DISPLAY_COUNT = 10

    def __init__(self, pop_id: str, bedbimfam_prefix: str, dist_col: str, win_size, # TODO(jonbjala) How to handle general numeric typing?
                 standardize: bool, r_band_filename: str = ""): # TODO(jonbjala) Handle r filename and caching differently, maybe with kwargs?

        # Save off the population ID
        self.id = pop_id
        logging.info("\nRunning pre-processing on population %s", self.id)


        # Save off other info
        self.dist_col = dist_col
        self.win_size = win_size
        logging.debug("Using column %s for distance with a threshold of %s",
                      self.dist_col, self.win_size)


        # Read the sample size from the fam file
        self.N = get_sample_size_from_fam_file("%s.fam" % bedbimfam_prefix)
        logging.debug("N = %s", self.N)


        # Create dictionary for indices needed to run calculations with this population and others
        self.cross_indices = dict()
        self.swap_indices = dict()


        # Read in bim file
        self.bim_df = read_bim_file(f"{bedbimfam_prefix}.bim")
        self.orig_M = len(self.bim_df)
        logging.debug("Starting value of M = %s", self.orig_M)


        # Read in bed file
        G = read_bed_file(f"{bedbimfam_prefix}.bed", self.orig_M, self.N)


        # Run filters on bim and bed values        
        _, bim_drop_indices, bim_drop_dict = qc_bim_df(self.bim_df, drop=False)

        bed_drop_indices = np.nanvar(G, axis=1) < PopInfo.SNP_VAR_THRESHOLD        
        self.total_drop_indices = bim_drop_indices | bed_drop_indices
        total_drop_count = np.count_nonzero(self.total_drop_indices)
        self.M = self.orig_M - total_drop_count

        logging.info("Filtering out a total of %s SNPs.  %s remain", total_drop_count, self.M)
        if logging.root.level <= logging.DEBUG:
            for filt_name, drop_indices in bim_drop_dict.items():
                drop_count = np.count_nonzero(drop_indices)
                full_rsid_list = self.bim_df[BIM_RSID_COL][drop_indices].to_list()
                logging.debug("\tCaught %s SNPs using %s filter: %s", drop_count, filt_name,
                              full_rsid_list if drop_count < PopInfo.MAX_DISPLAY_COUNT else 
                              full_rsid_list + ["..."])
            drop_count = np.count_nonzero(bed_drop_indices)
            full_rsid_list = self.bim_df[BIM_RSID_COL][bed_drop_indices].to_list()
            logging.debug("\tCaught %s SNPs with variance < %s: %s", drop_count,
                          PopInfo.SNP_VAR_THRESHOLD,
                          full_rsid_list if drop_count < PopInfo.MAX_DISPLAY_COUNT else 
                          full_rsid_list + ["..."])


        # Drop elements from G and the bim dataframe
        G = np.delete(G, self.total_drop_indices, axis=0)
        self.bim_df.drop(self.bim_df.index[self.total_drop_indices], inplace=True)

        # Add index column to allow for calculation of cross-population indices
        self.bim_df.reset_index(drop=True, inplace=True)
        self.bim_df.reset_index(inplace=True)

        # Calculate windows / extents using the bim dataframe
        self.lower_extents = calculate_lower_extents(self.bim_df[self.dist_col], self.win_size)
        self.max_lower_extent = max(self.lower_extents)
        logging.debug("Max lower extent = %s", self.max_lower_extent)

        # Report NaN info from G
        if logging.root.level <= logging.DEBUG:
            nans = np.isnan(G)
            if np.any(nans):
                logging.debug("NaNs detected. Total: %s, Max per snp: %s, Max per person: %s",
                              np.count_nonzero(nans), np.count_nonzero(nans, axis=1),
                              np.count_nonzero(nans, axis=0))
            del nans


        # Calculate frequencies for each SNP
        mean_per_snp = np.nanmean(G, axis=1)
        self.snp_freq = 0.5 * mean_per_snp


        # Demean and possibly standardize G, then replace NaNs with 0.0
        G -= mean_per_snp[:, np.newaxis] # Need newaxis to allow for broadcast      
        if standardize:
            logging.info("Standardizing the G matrix...")
            G /= np.nanstd(G, axis=1)[:, np.newaxis] # Need newaxis to allow for broadcast
        np.nan_to_num(G, copy=False)


        # Calculate R matrix
        r_band_mat = calculate_R_without_nan(G, self.lower_extents)


        # Double-check for 0 entries in the R diagonal
        if np.any(r_band_mat[0] == 0.0):
            logging.info("Diagonal elements of R are 0.")
            logging.info("\t Zeros at: %s", np.argwhere(r_band_mat[0] == 0.0))


        # Save the matrix to disk if a filename is given
        self.r_band_filename = r_band_filename
        logging.debug("Saving R matrix for population %s to disk...", self.id)
        np.save(self.r_band_filename, r_band_mat, allow_pickle=False)

        logging.info("Finished pre-processing on population %s\n", self.id)


    def get_banded_R(self, use_mmap: bool = True):
        # Retrieve the matrix from disk
        logging.debug("Retrieving R matrix for population %s from disk...", self.id)
        mmap_mode = 'r' if use_mmap else None
        try:
            matrix = np.load(self.r_band_filename, allow_pickle=False, mmap_mode=mmap_mode)
        except MemoryError as exc:
            if not use_mmap:
                logging.info("Cannot load R matrix for population %s, using mmap", self.id)
                matrix = np.load(self.r_band_filename, allow_pickle=False, mmap_mode='r')
            else:
                raise exc
        return matrix


    def calc_cross_pop_indices(self, other: 'PopInfo'):
        logging.debug("Calculating cross-population indices for populations %s and %s",
                      self.id, other.id)
        self.cross_indices[other.id], other.cross_indices[self.id],\
            self.swap_indices[other.id], other.swap_indices[self.id] =\
            get_population_indices(self.bim_df, other.bim_df)
        invalid_swaps_self = [i for i in self.swap_indices[other.id] if i > self.M]



    def calc_ldscores(self, other: 'PopInfo', self_mat: np.ndarray=None,
        other_mat: np.ndarray = None):

        if self.id == other.id:
            r = self.get_banded_R() if self_mat is None else self_mat
            logging.info("Calculating LD scores for population %s", self.id)
            ldscores = pd.Series(calculate_ld_scores((r,), self.N, self.lower_extents),
                                 index=self.bim_df[BIM_RSID_COL],
                                 name="%s_%s" % (self.id, self.id))
        else:
            logging.debug("In calc_ldscores() for %s and %s", self.id, other.id)

            joint_M = len(self.cross_indices[other.id])
            joint_extent = min(self.max_lower_extent, other.max_lower_extent)

            c1 = self.cross_indices[other.id]
            c2 = other.cross_indices[self.id]

            r1 = np.zeros((joint_extent, joint_M))
            r2 = np.zeros((joint_extent, joint_M))

            b1 = np.full(self.M, False)
            b1[c1] = True

            b2 = np.full(other.M, False)
            b2[c2] = True

            swap_mask = np.full(self.M, False)
            swap_mask[self.swap_indices[other.id]] = True

            # r1_full = (self.get_banded_R()
            #     if self_mat is None else self_mat)[:, self.cross_indices[other.id]]
            # r2_full = (other.get_banded_R()
            #     if other_mat is None else other_mat)[:, other.cross_indices[self.id]]
            r1_full = self.get_banded_R() if self_mat is None else self_mat
            r2_full = other.get_banded_R() if other_mat is None else other_mat

            joint_extent_after_processing = 0
            for i in range(joint_M):
                b1_window = b1[c1[i]:min(self.M, c1[i]+joint_extent)]
                b1_sum = sum(b1_window)
                r1[0:b1_sum, i] = r1_full[0:len(b1_window), c1[i]][b1_window]

                b2_window = b2[c2[i]:min(other.M, c2[i]+joint_extent)]
                b2_sum = sum(b2_window)
                r2[0:b2_sum, i] = r2_full[0:len(b2_window), c2[i]][b2_window]

                joint_extent_after_processing = max(joint_extent_after_processing, b1_sum, b2_sum)


            # Create series with RSIDs as the index
            logging.info("Calculating cross-population LD scores for populations %s and %s",
                         self.id, other.id)
            ldscores = pd.Series(calculate_ld_scores((r1[0:joint_extent_after_processing],
                                                      r2[0:joint_extent_after_processing]),
                                                     swaps=swap_mask[self.cross_indices[other.id]]),
                                 index=self.bim_df[BIM_RSID_COL].iloc[self.cross_indices[other.id]],
                                 name="%s_%s" % (self.id, other.id)) # TODO(jonbjala) The formatted string should be a constant somewhere
        return ldscores
