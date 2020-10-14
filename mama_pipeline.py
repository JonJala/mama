#!/usr/bin/env python3

"""
Python functions that implement the core MAMA processing
"""

import gc
import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from core_mama import (create_omega_matrix, create_sigma_matrix, run_mama_method, qc_omega,
                       qc_sigma)
from reg_mama import (MAMA_REG_OPT_ALL_FREE, MAMA_REG_OPT_ALL_ZERO, MAMA_REG_OPT_OFFDIAG_ZERO,
                      MAMA_REG_OPT_IDENT, MAMA_REG_OPT_PERF_CORR, run_ldscore_regressions)
from util.df import Filter, intersect_indices
from util.sumstats import (SNP_COL, BP_COL, CHR_COL, BETA_COL, FREQ_COL, SE_COL, A1_COL,
                           A2_COL, COMPLEMENT, BASES, create_freq_filter, create_chr_filter,
                           standardize_all_sumstats, process_sumstats)


# Constants / Parameters / Types #############

AncestryId = Any
PhenotypeId = Any
PopulationId = Tuple[AncestryId, PhenotypeId]

# Columns that MAMA requires
MAMA_REQ_STD_COLS = {SNP_COL, CHR_COL, BETA_COL, FREQ_COL, SE_COL, A1_COL, A2_COL}

# Map of default regular expressions used to convert summary stat column names to standardized names
# TODO(jonbjala) Refine these more, just use these values are placeholders for now
MAMA_RE_EXPR_MAP = {
    SNP_COL : '.*SNP.*|.*RS.*',
    BP_COL : '.*BP.*',
    CHR_COL : '.*CHR.*',
    BETA_COL : '.*BETA.*',
    FREQ_COL : '.*FREQ.*|.*FRQ.*|AF.*',
    SE_COL : '.*SE.*',
    A1_COL : '.*A1.*|.*MAJOR.*|.*EFFECT.*ALL.*|REF.*',
    A2_COL : '.*A2.*|.*MINOR.*|.*OTHER.*ALL.*|ALT.*',
}

# Frequency filtering defaults
DEFAULT_MAF_MIN = 0.01
DEFAULT_MAF_MAX = 0.99

# Chromosome filtering defaults
DEFAULT_CHR_LIST = [str(cnum) for cnum in range(1, 23)] + ['X', 'Y']

# Filter-related materials
NAN_FILTER = 'NO NAN'
FREQ_FILTER = 'FREQ BOUNDS'
SE_FILTER = 'SE BOUNDS'
CHR_FILTER = 'CHR VALUES'
SNP_DUP_ALL_FILTER = 'DUPLICATE ALLELE SNPS'
SNP_PALIN_FILT = 'PALINDROMIC SNPS'
SNP_INVALID_ALLELES_FILTER = 'INVALID ALLELES'
MAMA_STD_FILTER_FUNCS = {
    NAN_FILTER :
        {
            'func' : lambda df: df[list(MAMA_REQ_STD_COLS)].isnull().any(axis=1),
            'description' : "Filters out SNPs with any NaN values in required "
                            "columns %s" % MAMA_REQ_STD_COLS
        },
    FREQ_FILTER :
        {
            'func' : create_freq_filter(DEFAULT_MAF_MIN, DEFAULT_MAF_MAX),
            'description' : "Filters out SNPs with FREQ values outside of "
                            "[%s, %s]" % (DEFAULT_MAF_MIN, DEFAULT_MAF_MAX)
        },
    SE_FILTER :
        {
            'func' : lambda df: df[SE_COL].lt(0.0),
            'description' : "Filters out SNPs with negative SE values"
        },
    CHR_FILTER :
        {
            'func' : create_chr_filter(DEFAULT_CHR_LIST),
            'description' : "Filters out SNPs with listed chromosomes not in %s" % DEFAULT_CHR_LIST
        },
    SNP_DUP_ALL_FILTER :
        {
            'func' : lambda df: df[A1_COL] == df[A2_COL],
            'description' : "Filters out SNPs with major allele = minor allele"
        },
    SNP_PALIN_FILT :
        {
            'func' : lambda df: df[A1_COL].replace(COMPLEMENT) == df[A2_COL],
            'description' : "Filters out SNPs where major / minor alleles are a base pair"
        },
    SNP_INVALID_ALLELES_FILTER :
        {
            'func' : lambda df: ~df[A1_COL].isin(BASES) | ~df[A2_COL].isin(BASES),
            'description' : "Filters out SNPs with alleles not in %s" % BASES
        },
    }


# Derived Constants###########################

# Filter function dictionaries (name to function mapping or description) for MAMA
MAMA_STD_FILTERS = {fname : (finfo['func'], finfo['description'])
                    for fname, finfo in MAMA_STD_FILTER_FUNCS.items()}


# Functions ##################################

#################################
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


#################################
def qc_ldscores(ldscores_df: pd.DataFrame):
    """
    Runs QC steps on LD scores.  This will be much lighter-weight than what is done on summary
    statistics, as it assumes that the LD score file was generated using this software.

    :param ldscores_df: Dataframe holding ldscores

    :return pd.DataFrame: DataFrame containing the QC'ed LD scores
    """
    # Make copy of the dataframe (this copy will be modified)
    df = ldscores_df.copy()

    # Drop any lines with NaN
    nan_drops = df.isnull().any(axis=1)
    df.drop(df.index[nan_drops], inplace=True)

    # Make sure SNP IDs are lower case ("rs..." rather than "RS...")
    df[SNP_COL] = df[SNP_COL].str.lower()

    # Set SNP column to be the index and sort
    df.set_index(SNP_COL, inplace=True)
    df.sort_index(inplace=True)

    return df


#################################
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
    snp_intersection = intersect_indices(sumstats.values(), ldscores)
    logging.info("\nNumber of SNPS in initial intersection of all sources: %s",
                 len(snp_intersection))

    # Reduce each DF down to the SNP intersection
    for pop_id, pop_df in sumstats.items():
        snps_to_drop = pop_df.index.difference(snp_intersection)
        pop_df.drop(snps_to_drop, inplace=True)
    snps_to_drop = ldscores.index.difference(snp_intersection)
    ldscores.drop(snps_to_drop, inplace=True)

    # Standardize alleles in the summary statistics
    logging.info("\nStandardizing reference alleles in summary statistics.")
    ref_popid, drop_indices, drop_dict, ref_flip_dict = standardize_all_sumstats(sumstats)
    logging.info("Standardized to population: %s", ref_popid)
    logging.info("Dropped %s SNPs during reference allele standardization." % drop_indices.sum())
    if logging.root.level <= logging.DEBUG:
        logging.debug("RS IDs of drops during standardization: %s",
                      sumstats[ref_popid].index[drop_indices].to_list())

    # Drop SNPs as a result of standardization of reference alleles
    for pop_id, pop_df in sumstats.items():
        pop_df.drop(pop_df.index[drop_indices], inplace=True)


#################################
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


#################################
# TODO(jonbjala) Allowing specifying population order?  For now go with order in sumstats dictionary
def mama_pipeline(sumstats: Dict[PopulationId, Any], ldscores: Any,
                  column_maps: Dict[PopulationId, Dict[str, str]] = {},
                  re_expr_map: Dict[str, str] = MAMA_RE_EXPR_MAP,
                  filters: Dict[str, Tuple[Filter, str]] = MAMA_STD_FILTERS,
                  ld_opt: Any = MAMA_REG_OPT_ALL_FREE,
                  se_prod_opt: Any = MAMA_REG_OPT_ALL_FREE,
                  int_opt: Any = MAMA_REG_OPT_ALL_FREE,
                  harmonized_file_fstr: str = "",
                  reg_coef_fstr: str = "") -> Dict[PopulationId, pd.DataFrame]:
    """
    Runs the steps in the overall MAMA pipeline

    :param sumstats: Dictionary of population identifier -> filename or DataFrame
    :param ldscores: Filename or DataFrame for the LD scores
    :param column_maps: Dictionary containing any column mappings indexed by population identifier
                        (same as used for sumstats parameter).  If none exists, the re_expr_map
                        will be used to determine column mappings
    :param re_expr_map: Regular expressions used to map column names to standard columns
    :param filters: Map of filter name to a (function, description) tuple, used to filter
                    summary statistics
    # TODO(jonbjala) Switch to kwargs?

    :return Dict[PopulationId, pd.DataFrame]: Result summary statistics dictionary (reference to
                                              the same dictionary passed in, but with updated
                                              summary statistics)
    """
    # TODO(jonbjala) Move reading files back out to mama.py?
    # Check / read in LD scores and then QC
    logging.info("\nReading in and running QC on LD Scores.")
    ldscores = obtain_df(ldscores, "LD scores")
    ldscores = qc_ldscores(ldscores)

    # Check / read in summary stats and then QC
    logging.info("\nReading in summary statistics.\n")
    for pop_id in sumstats.keys():
        # Read in if necessary (and update dictionary)
        sumstats[pop_id] = obtain_df(sumstats[pop_id], str(pop_id) + " sumstats")

        # QC summary stats (along with some light validation and some logging of drops)
        pop_df = sumstats[pop_id]
        col_map = column_maps.get(pop_id, None)  # If a column map exists for this pop, use that
        logging.info("\nRunning QC on %s summary statistics", pop_id)
        logging.debug("\tColumn mapping = %s\n", col_map)
        sumstats[pop_id] = process_sumstats(pop_df, re_expr_map, MAMA_REQ_STD_COLS,
                                            filters, col_map)

    # Harmonize the summary stats and LD scores (write to disk if requested)
    harmonize_all(sumstats, ldscores)
    if harmonized_file_fstr:
        logging.info("\nWriting harmonized summary statistics to disk.\n")
        for (ancestry, phenotype), harm_ss_df in sumstats.items():
            filename = harmonized_file_fstr % (ancestry, phenotype)
            logging.debug("\t%s", filename)
            write_sumstats_to_file(filename, harm_ss_df)

    # Copy values to numpy ndarrays to use in vectorized processing
    beta_arr, se_arr, ldscore_arr = collate_df_values(sumstats, ldscores)

    # Run LD score regressions
    logging.info("\nRunning LD Score regression.")
    ld_coef, const_coef, se2_coef = run_ldscore_regressions(beta_arr, se_arr, ldscore_arr,
                                                            ld_fixed_opt=ld_opt,
                                                            se_prod_fixed_opt=se_prod_opt,
                                                            int_fixed_opt=int_opt)

    # Log coefficients at debug level
    logging.debug("Regression coefficients (LD):\n%s", ld_coef)
    logging.debug("Regression coefficients (Intercept):\n%s", const_coef)
    logging.debug("Regression coefficients (SE^2):\n%s", se2_coef)
    if reg_coef_fstr:
        logging.info("\nWriting regression coefficients to disk.\n")
        ld_coef.tofile(reg_coef_fstr % "ld", sep='\t')
        const_coef.tofile(reg_coef_fstr % "int", sep='\t')
        se2_coef.tofile(reg_coef_fstr % "se2", sep='\t')

    # Calculate Omegas and Sigmas
    logging.info("\nCreating omega and sigma matrices.")
    omega = create_omega_matrix(ldscore_arr, ld_coef)
    sigma = create_sigma_matrix(se_arr, se2_coef, const_coef)

    # Check omega and sigma for validity based on positive (semi-)definiteness
    # Create drop arrays shapes that allow for broadcasting and comparison later
    omega_valid = qc_omega(omega).reshape((omega.shape[0], 1, 1))
    sigma_valid = qc_sigma(sigma).reshape((sigma.shape[0], 1, 1))
    omega_sigma_drops = np.logical_not(np.logical_and(omega_valid, sigma_valid))
    omega_sigma_1d_drops = omega_sigma_drops.ravel()  # Need a 1-D array for DataFrame drops later
    logging.info("Dropped %s SNPs due to non-positive-(semi)-definiteness of omega / sigma.",
                 omega_sigma_1d_drops.sum())
    if logging.root.level <= logging.DEBUG:
        logging.debug("\tRS IDs = %s", ldscores.index[omega_sigma_1d_drops].to_list())

    # Run the MAMA method
    # Use identity matrices for "bad" SNPs to allow vectorized operations without having to copy
    logging.info("\nRunning main MAMA method.")
    new_betas, new_beta_ses = run_mama_method(beta_arr,
                              np.where(omega_sigma_drops, np.identity(omega.shape[1]), omega),
                              np.where(omega_sigma_drops, np.identity(sigma.shape[1]), sigma))

    # Copy values back to the summary statistics DataFrames (and make omega / sigma - related drops)
    logging.info("\nPreparing results for output.")
    for pop_num, ((ancestry, phenotype), ss_df) in enumerate(sumstats.items()):
        ss_df[BETA_COL] = new_betas[:, pop_num]
        ss_df[SE_COL] = new_beta_ses[:, pop_num]
        ss_df.drop(ss_df.index[omega_sigma_1d_drops], inplace=True)
        # TODO(jonbjala) Effective N

    return sumstats
