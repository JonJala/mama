#!/usr/bin/env python3

"""
Python functions that implement the core MAMA processing
"""

import gc
import logging
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm

from core_mama import (create_omega_matrix, create_sigma_matrix, run_mama_method, qc_omega,
                       qc_sigma)
from reg_mama import (MAMA_REG_OPT_ALL_FREE, MAMA_REG_OPT_ALL_ZERO, MAMA_REG_OPT_OFFDIAG_ZERO,
                      MAMA_REG_OPT_IDENT, MAMA_REG_OPT_PERF_CORR, run_ldscore_regressions)
from util.df import Filter, intersect_indices
from util.sumstats import (SNP_COL, BP_COL, CHR_COL, BETA_COL, FREQ_COL, SE_COL, A1_COL,
                           A2_COL, P_COL, INFO_COL, N_COL, Z_COL, COMPLEMENT, BASES,
                           MAX_RSID_LOGGING, create_freq_filter, create_chr_filter,
                           standardize_all_sumstats, process_sumstats)


# Constants / Parameters / Types #############

AncestryId = Any
PhenotypeId = Any
PopulationId = Tuple[AncestryId, PhenotypeId]

# Columns that MAMA requires
MAMA_REQ_STD_COLS = {SNP_COL, CHR_COL, BETA_COL, FREQ_COL, SE_COL, A1_COL, A2_COL, BP_COL, P_COL}

# Map of default regular expressions used to convert summary stat column names to standardized names
# TODO(jonbjala) Refine these more, just use these values are placeholders for now
MAMA_RE_EXPR_MAP = {
    SNP_COL : '.*SNP.*|.*RS.*',
    BP_COL : '.*BP.*|.*POS.*',
    CHR_COL : '.*CHR.*',
    BETA_COL : '.*BETA.*',
    FREQ_COL : '.*FREQ.*|.*FRQ.*|.*AF',
    SE_COL : '.*SE.*',
    A1_COL : '.*A1.*|.*MAJOR.*|.*EFFECT.*ALL.*|REF.*',
    A2_COL : '.*A2.*|.*MINOR.*|.*OTHER.*ALL.*|ALT.*',
    P_COL : 'P|P.*VAL.*',
    INFO_COL : 'INFO',
    N_COL : 'N',
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
SNP_NEGATIVE_P_FILTER = 'NEGATIVE GWAS P'
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
            'func' : lambda df: df[SE_COL].le(0.0),
            'description' : "Filters out SNPs with non-positive SE values"
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
    SNP_NEGATIVE_P_FILTER :
        {
            'func' : lambda df: df[P_COL].lt(0.0),
            'description' : "Filters out SNPs with negative GWAS P values"
        },
    }

# Column name to rename N column to if it exists (to help resolve ambiguity since there should be
# an effective N column added)
ORIGINAL_N_COL_RENAME = "N_ORIG"

# Column name to rename N column to if it exists (to help resolve ambiguity since there should be
# an effective N column added)
N_EFF_COL = "N_EFF"


# Derived Constants###########################

# Filter function dictionaries (name to function mapping or description) for MAMA
MAMA_STD_FILTERS = {fname : (finfo['func'], finfo['description'])
                    for fname, finfo in MAMA_STD_FILTER_FUNCS.items()}

# Calculate constants used in determination of P values for MAMA
ln = np.log
LN_2 = ln(2.0)
RECIP_LN_10 = np.reciprocal(ln(10.0))

# Functions ##################################

#################################
def obtain_df(possible_df: Union[str, pd.DataFrame], id_val: Any) -> pd.DataFrame:
    """
    Small helper function that handles functionality related to reading in a DataFrame

    :param possible_df: Should either be a string indicating the full path to a file to be
                        read into a DataFrame or the DataFrame itself.  All other possibilities will
                        result in this function raising an error
    :param id_val: Used for logging / error-reporting to identify the data being read / checked

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
def harmonize_all(sumstats: Dict[PopulationId, pd.DataFrame], ldscores: pd.DataFrame,
                  snp_list: pd.Index = None):
    """
    Does the harmonization between the QC'ed input summary statistics and the LD scores.  The
    DataFrames are all modified in place (SNPs/rows dropped and reference alleles transformed
    as needed), and all inputs are expected to have indices = SNP ID (beginning with "rs")

    :param sumstats: Dictionary mapping a population id to a DataFrame holding the summary
                     stat information.  The DFs should all have been QCed already.
    :param ldscores: DataFrame of LD score information
    :param snp_list: If specified, a Series containing rsIDs to which to restrict analysis
    """

    # Intersect all the SNP lists to get the SNPs all data sources have in common
    snp_intersection = intersect_indices(sumstats.values(), ldscores)
    if snp_list is not None:
        logging.info("Restricting to user-supplied SNP list (%s SNPs)...", len(snp_list))
        snp_intersection = snp_intersection.intersection(snp_list)
    logging.info("\n\nNumber of SNPS in initial intersection of all sources: %s",
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
def calculate_n_eff(pop: int, n_orig: np.ndarray, sigma: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Function that calculates effective N

    :param pop: Number of the population (used to index into sigma)
    :param n_orig: Array of original per-SNP N values for this population
    :param sigma: MxPxP matrix of Sigma values
    :param se: Array of length M of standard errors

    :return: The array of per-SNP effective N's
    """
    return n_orig * sigma[:, pop, pop] * np.reciprocal(np.square(se))


#################################
def calculate_p(z: np.array) -> np.array:
    """
    Function that calculates P for the MAMA results

    :param z: Z scores

    :return: P values for MAMA
             (as strings, to allow for very large negative exponents)
    """
    # Since P = 2 * normal_cdf(-|Z|), P = e ^ (log_normal_cdf(-|Z|) + ln 2)
    # This can be changed to base 10 as P = 10 ^ ((log_normal_cdf(-|Z|) + ln 2) / ln 10)
    log_10_p = RECIP_LN_10 * (norm.logcdf(-np.abs(z)) + LN_2)

    # Break up the log based 10 of P values into the integer and fractional part
    frac_part, int_part = np.modf(log_10_p)

    # Construct strings for the P values
    # 1) Add one to the fractional part to ensure that the result mantissa is between 1 and 10
    # 2) Subtract one from the integer part to compensate and keep the overall value correct
    result = np.char.add(np.char.add(np.power(10.0, (frac_part + 1.0)).astype(str), 'e'),
                         (int_part - 1).astype(int).astype(str))

    return result


#################################
# TODO(jonbjala) Allowing specifying population order?  For now go with order in sumstats dictionary
def mama_pipeline(sumstats: Dict[PopulationId, Any], ldscore_list: List[Any], snp_list: str = None,
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
    :param ldscore_list: List of filenames and/or DataFrames of LD scores (will be concatenated)
    :param snp_list: Path to file containing list of rsIDs to which to restrict analysis
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

    # Check / read in LD scores and then QC
    logging.info("\n\nReading in and running QC on LD Scores")
    logging.debug("\nList of files: %s", ldscore_list)
    ldscores = pd.concat((obtain_df(f, "LD Scores") for f in ldscore_list), ignore_index=True)
    ldscores = qc_ldscores(ldscores)

    # Check / read in summary stats and then QC
    logging.info("\n\nReading in summary statistics.")
    for pop_id in sumstats.keys():
        # Read in if necessary (and update dictionary)
        logging.info("\n\n")
        sumstats[pop_id] = obtain_df(sumstats[pop_id], str(pop_id) + " sumstats")

        # QC summary stats (along with some light validation and some logging of drops)
        pop_df = sumstats[pop_id]
        col_map = column_maps.get(pop_id, None)  # If a column map exists for this pop, use that
        logging.info("\nRunning QC on %s summary statistics", pop_id)
        logging.debug("\tColumn mapping = %s\n", col_map)
        sumstats[pop_id] = process_sumstats(pop_df, re_expr_map, MAMA_REQ_STD_COLS,
                                            filters, col_map)

    # If a SNP list is given, read that in
    if snp_list:
        snp_list = pd.read_csv(snp_list, sep='\n', engine='python', comment='#',
                               dtype=str, names=[SNP_COL], squeeze=True)
        # snp_list.set_index(SNP_COL, inplace=True)
        snp_list = pd.Index(data=snp_list, dtype=str)

    # Harmonize the summary stats and LD scores (write to disk if requested)
    harmonize_all(sumstats, ldscores, snp_list)
    if harmonized_file_fstr:
        logging.info("\n\nWriting harmonized summary statistics to disk.\n")
        for (ancestry, phenotype), harm_ss_df in sumstats.items():
            filename = harmonized_file_fstr % (ancestry, phenotype)
            logging.debug("\t%s", filename)
            write_sumstats_to_file(filename, harm_ss_df)

    # Copy values to numpy ndarrays to use in vectorized processing
    beta_arr, se_arr, ldscore_arr = collate_df_values(sumstats, ldscores)

    # Run LD score regressions
    logging.info("\n\nRunning LD Score regression.")
    ld_coef, const_coef, se2_coef = run_ldscore_regressions(beta_arr, se_arr, ldscore_arr,
                                                            ld_fixed_opt=ld_opt,
                                                            se_prod_fixed_opt=se_prod_opt,
                                                            int_fixed_opt=int_opt)

    # Log coefficients at debug level (and write coefficients to disk if option is selected)
    logging.info("Regression coefficients (LD):\n%s", ld_coef)
    logging.info("Regression coefficients (Intercept):\n%s", const_coef)
    logging.info("Regression coefficients (SE^2):\n%s", se2_coef)
    if reg_coef_fstr:
        logging.info("\nWriting regression coefficients to disk.\n")
        ld_coef.tofile(reg_coef_fstr % "ld", sep='\t')
        const_coef.tofile(reg_coef_fstr % "int", sep='\t')
        se2_coef.tofile(reg_coef_fstr % "se2", sep='\t')

    # Calculate Omegas and Sigmas
    logging.info("\n\nCreating omega and sigma matrices.")
    omega = create_omega_matrix(ldscore_arr, ld_coef)
    sigma = create_sigma_matrix(se_arr, se2_coef, const_coef)

    # Check omega and sigma for validity based on positive (semi-)definiteness
    # Create drop arrays shapes that allow for broadcasting and comparison later
    omega_drops = np.logical_not(qc_omega(omega))
    sigma_drops = np.logical_not(qc_sigma(sigma))
    omega_sigma_drops = np.logical_or(omega_drops, sigma_drops)

    logging.info("Dropped %s SNPs due to non-positive-semi-definiteness of omega.",
                 omega_drops.sum())
    if logging.root.level <= logging.DEBUG:
        omega_drop_rsids = ldscores.index[omega_drops].to_list()
        logging.debug("\tRS IDs = %s", omega_drop_rsids[:MAX_RSID_LOGGING] + ["..."]
                      if len(omega_drop_rsids) > MAX_RSID_LOGGING else omega_drop_rsids)
    logging.info("Dropped %s SNPs due to non-positive-definiteness of sigma.",
                 sigma_drops.sum())
    if logging.root.level <= logging.DEBUG:
        sigma_drop_rsids = ldscores.index[sigma_drops].to_list()
        logging.debug("\tRS IDs = %s", sigma_drop_rsids[:MAX_RSID_LOGGING] + ["..."]
                      if len(sigma_drop_rsids) > MAX_RSID_LOGGING else sigma_drop_rsids)
    logging.info("Dropped %s total SNPs due to non-positive-(semi)-definiteness of omega / sigma.",
                 omega_sigma_drops.sum())
    if logging.root.level <= logging.DEBUG:
        os_drop_rsids = ldscores.index[omega_sigma_drops].to_list()
        logging.debug("\tRS IDs = %s", os_drop_rsids[:MAX_RSID_LOGGING] + ["..."]
                      if len(os_drop_rsids) > MAX_RSID_LOGGING else os_drop_rsids)
    os_drops_reshaped = omega_sigma_drops.reshape((ldscore_arr.shape[0], 1, 1))

    # Run the MAMA method
    # Use identity matrices for "bad" SNPs to allow vectorized operations without having to copy
    logging.info("\n\nRunning main MAMA method.")
    new_betas, new_beta_ses = run_mama_method(beta_arr,
                              np.where(os_drops_reshaped, np.identity(omega.shape[1]), omega),
                              np.where(os_drops_reshaped, np.identity(sigma.shape[1]), sigma))

    # Copy values back to the summary statistics DataFrames
    # Also, perform any remaining calculations / formatting
    logging.info("\nPreparing results for output.")
    final_snp_count = 0
    for pop_num, ((ancestry, phenotype), ss_df) in enumerate(sumstats.items()):
        logging.info("\tPopulation %s: %s", pop_num, (ancestry, phenotype))
        new_df_data_list = [(CHR_COL, ss_df[CHR_COL]),
                            (BP_COL, ss_df[BP_COL]), (A1_COL, ss_df[A1_COL]),
                            (A2_COL, ss_df[A2_COL]), (FREQ_COL, ss_df[FREQ_COL])]

        # Update the betas and standard errors
        new_df_data_list.append((BETA_COL, new_betas[:, pop_num]))
        new_df_data_list.append((SE_COL, new_beta_ses[:, pop_num]))

        # Calculate Z score
        z_scores = ss_df[BETA_COL] / ss_df[SE_COL]
        mean_chi_2 = np.square(z_scores).mean()
        logging.info("\t\tMean Chi^2 for %s = %s", (ancestry, phenotype), mean_chi_2)
        new_df_data_list.append((Z_COL, z_scores))

        # Calculate P column
        new_df_data_list.append((P_COL, calculate_p(z_scores.to_numpy())))

        # Calculate effective N
        if N_COL in ss_df.columns:
            new_df_data_list.append((N_EFF_COL, calculate_n_eff(pop_num, ss_df[N_COL].to_numpy(),
                                                                sigma, new_beta_ses[:, pop_num])))
            new_df_data_list.append((ORIGINAL_N_COL_RENAME, ss_df[N_COL]))

        # Construct the output dataframe
        new_df = pd.DataFrame(data=dict(new_df_data_list), index=ss_df.index)

        # Drop SNPs due to omega / sigma, and sort by BP and CHR
        new_df.drop(new_df.index[omega_sigma_drops], inplace=True)
        new_df.sort_values(by=[CHR_COL, BP_COL], inplace=True)
        final_snp_count = len(new_df)

        # Replace the old dataframe with the new one
        sumstats[(ancestry, phenotype)] = new_df

    logging.info("\nFinal SNP count = %s", final_snp_count)

    return sumstats
