#!/usr/bin/env python3

"""
Python functions that implement the core MAMA processing
"""

import logging
from typing import Any, Dict, List, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from core_mama import (create_omega_matrix, create_sigma_matrix, run_mama_method, qc_omega,
                       qc_sigma)
from reg_mama import (REG_INT_OPT_NAME, REG_LD_OPT_NAME, REG_LD_SCALE_FACTOR_NAME, REG_SE_OPT_NAME,
                      MAMA_REG_OPT_ALL_FREE, run_ldscore_regressions)
from util.df import Filter, intersect_indices
from util.sumstats import (SNP_COL, BP_COL, CHR_COL, BETA_COL, FREQ_COL, SE_COL, A1_COL,
                           A2_COL, P_COL, INFO_COL, N_COL, Z_COL, COMPLEMENT, BASES,
                           MAX_RSID_LOGGING, create_freq_filter, create_chr_filter,
                           standardize_all_sumstats, process_sumstats)


# Constants / Parameters / Types #############

# Pylint upper-case errors disabled here to adhere to Python typing module conventions
AncestryId = Any  # pylint: disable=invalid-name
PhenotypeId = Any  # pylint: disable=invalid-name
PopulationId = Tuple[AncestryId, PhenotypeId]

# Columns that MAMA requires
MAMA_REQ_COLS_MAP = {
    SNP_COL : str,
    CHR_COL : str,
    BETA_COL : float,
    FREQ_COL : float,
    SE_COL : float,
    A1_COL : str,
    A2_COL : str,
    BP_COL : int,
    P_COL : float
}
MAMA_REQ_STD_COLS = set(MAMA_REQ_COLS_MAP.keys())

# Map of default regular expressions used to convert summary stat column names to standardized names
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
                            f"columns {MAMA_REQ_STD_COLS}"
        },
    FREQ_FILTER :
        {
            'func' : create_freq_filter(DEFAULT_MAF_MIN, DEFAULT_MAF_MAX),
            'description' : "Filters out SNPs with FREQ values outside of "
                            f"[{DEFAULT_MAF_MIN}, {DEFAULT_MAF_MAX}]"
        },
    SE_FILTER :
        {
            'func' : lambda df: df[SE_COL].le(0.0),
            'description' : "Filters out SNPs with non-positive SE values"
        },
    CHR_FILTER :
        {
            'func' : create_chr_filter(DEFAULT_CHR_LIST),
            'description' : f"Filters out SNPs with listed chromosomes not in {DEFAULT_CHR_LIST}"
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
            'description' : f"Filters out SNPs with alleles not in {BASES}"
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


# Regular expression indicating columns to keep in harmonized summary statistics
MAMA_HARM_COLS_RE = '|'.join(MAMA_REQ_STD_COLS)

# Calculate constants used in determination of P values for MAMA
ln = np.log  # pylint: disable=invalid-name
LN_2 = ln(2.0)
RECIP_LN_10 = np.reciprocal(ln(10.0))

# Functions ##################################

#################################
def obtain_df(possible_df: Union[str, pd.DataFrame], id_val: Any, sep_arg: Union[None, str] = None
    ) -> pd.DataFrame:
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

        # Catch ParserWarning that warns of switch to Python engine if that happens
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.ParserWarning,
                                    message="Falling back to the \'python\' engine")
            possible_df = pd.read_csv(possible_df, sep=sep_arg, comment='#',
                                      dtype=MAMA_REQ_COLS_MAP)
    # If neither a string (presumed to be a filename) nor DataFrame are passed in, throw error
    elif not isinstance(possible_df, pd.DataFrame):
        raise RuntimeError(f"ERROR: Either pass in filename or DataFrame for {id_val} "
                           f"rather than [{type(possible_df)}]")

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

    # Reduce each DF down to the SNP intersection (and drop extraneous columns, too)
    for pop_df in sumstats.values():
        snps_to_drop = pop_df.index.difference(snp_intersection)
        pop_df.drop(snps_to_drop, inplace=True)
        pop_df.drop(pop_df.columns.difference(list(MAMA_RE_EXPR_MAP.keys())), axis=1, inplace=True)
    snps_to_drop = ldscores.index.difference(snp_intersection)
    ldscores.drop(snps_to_drop, inplace=True)

    # Standardize alleles in the summary statistics
    logging.info("\nStandardizing reference alleles in summary statistics.")
    ref_popid, tot_drop_indices, drop_dict, ref_flip_dict = standardize_all_sumstats(sumstats)  # pylint: disable=unused-variable,line-too-long
    logging.info("Standardized to population: %s", ref_popid)
    logging.info("Dropped %s SNPs during reference allele standardization.", tot_drop_indices.sum())
    if logging.root.level <= logging.DEBUG:
        logging.debug("RS IDs of drops during standardization: %s",
                      sumstats[ref_popid].index[tot_drop_indices].to_list())

    # Drop SNPs as a result of standardization of reference alleles
    for pop_df in sumstats.values():
        pop_df.drop(pop_df.index[tot_drop_indices], inplace=True)
    ldscores.drop(ldscores.index[tot_drop_indices], inplace=True)


#################################
def write_sumstats_to_file(filename: str, df: pd.DataFrame):
    """
    Helper function that writes a summary statistics DataFrame to disk

    :param filename: Full path to output file
    :param df: DataFrame holding the summary statistics
    """
    df.to_csv(filename, sep="\t", index_label=SNP_COL, na_rep="NaN")


#################################
def collate_df_values(sumstats: Dict[PopulationId, pd.DataFrame], ldscores: pd.DataFrame,
                      ordering: List[PopulationId] = None) -> Tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray]:
    """
    Function that gathers data from DataFrames (betas, ses, etc.) into ndarrays for use in
    vectorized processing

    :param sumstats: Dictionary of population identifier -> DataFrame
    :param ldscores: DataFrame for the LD scores
    :param ordering: Optional parameter indicating the order in which populations should be arranged
                     (if not specified, the ordering of the sumstats dictionary keys will be used)

    :return: Betas (MxP), SEs (MxP), and LD scores (MxPxP)
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

            ldscore_col_name = "_".join((str(ancestry_id), str(second_ancestry_id)))
            if ldscore_col_name not in ldscores.columns:
                ldscore_col_name = "_".join((str(second_ancestry_id), str(ancestry_id)))

            ldscore_arr[:, pop_num, second_pop_num] = ldscores[ldscore_col_name].to_numpy()

    return beta_arr, se_arr, ldscore_arr


#################################
def calculate_z(betas: np.ndarray, ses: np.ndarray) -> np.ndarray:
    """
    Function that calculates Z scores from betas and standard errors.  Shape
    of betas and ses must be the same / broadcastable.

    :param pop: Array of beta values
    :param ses: Array of standard errors

    :return: The array of Z scores
    """
    return np.divide(betas, ses)


#################################
def calculate_mean_chi_sq(z_scores: np.ndarray) -> float:
    """
    Function that calculates a mean chi squared statistic from
    Z scores

    :param z_scores: Array of Z scores

    :return: The mean chi squared statistic
    """
    return np.square(z_scores).mean()


#################################
def calculate_n_eff(pop: int, n_orig: np.ndarray, sigma: np.ndarray, ses: np.ndarray) -> np.ndarray:
    """
    Function that calculates effective N

    :param pop: Number of the population (used to index into sigma)
    :param n_orig: Array of original per-SNP N values for this population
    :param sigma: MxPxP matrix of Sigma values
    :param ses: Array of length M of standard errors

    :return: The array of per-SNP effective N's
    """
    return n_orig * sigma[:, pop, pop] * np.reciprocal(np.square(ses))


#################################
def calculate_p(z_scores: np.array) -> np.array:
    """
    Function that calculates P for the MAMA results

    :param z_scores: Z scores

    :return: P values for MAMA
             (as strings, to allow for very large negative exponents)
    """
    # Since P = 2 * normal_cdf(-|Z|), P = e ^ (log_normal_cdf(-|Z|) + ln 2)
    # This can be changed to base 10 as P = 10 ^ ((log_normal_cdf(-|Z|) + ln 2) / ln 10)
    log_10_p = RECIP_LN_10 * (norm.logcdf(-np.abs(z_scores)) + LN_2)

    # Break up the log based 10 of P values into the integer and fractional part
    # To handle the case of Z = 0 (and not result in "10e-1"), set initial values to (-1.0, 1.0)
    frac_part, int_part = np.full_like(z_scores, -1.0), np.full_like(z_scores, 1.0)
    np.modf(log_10_p, out=(frac_part, int_part), where=(z_scores != 0.0))

    # Construct strings for the P values
    # 1) Add one to the fractional part to ensure that the result mantissa is between 1 and 10
    # 2) Subtract one from the integer part to compensate and keep the overall value correct
    result = np.char.add(np.char.add(np.power(10.0, (frac_part + 1.0)).astype(str), 'e'),
                         (int_part - 1).astype(int).astype(str))

    return result


#################################
def mama_pipeline(sumstats: Dict[PopulationId, Any], ldscore_list: List[Any], snp_list: str = None,
                  column_maps: Dict[PopulationId, Dict[str, str]] = None,
                  re_expr_map: Dict[str, str] = None,
                  filters: Dict[str, Tuple[Filter, str]] = MAMA_STD_FILTERS,
                  ld_opt: Any = MAMA_REG_OPT_ALL_FREE,
                  se_prod_opt: Any = MAMA_REG_OPT_ALL_FREE,
                  int_opt: Any = MAMA_REG_OPT_ALL_FREE,
                  ld_corr_scale_factor = 1.0,
                  std_units: bool = False,
                  harmonized_file_fstr: str = "",
                  reg_coef_fstr: str = "",
                  sep: Union[None, str] = None) -> Dict[PopulationId, pd.DataFrame]:
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

    # Get number of populations
    num_pops = len(sumstats)

    # If column maps are not specified, set to empty dictionary
    if not column_maps:
        column_maps = {}

    # If regular expression map isn't specified, use MAMA default
    if not re_expr_map:
        re_expr_map = MAMA_RE_EXPR_MAP.copy()

    # Check / read in LD scores and then QC
    logging.info("\n\nReading in and running QC on LD Scores\n")
    logging.debug("\nList of files: %s", ldscore_list)
    ldscores = pd.concat((obtain_df(f, "LD Scores", sep) for f in ldscore_list), ignore_index=True)
    ldscores = qc_ldscores(ldscores)

    # Check / read in summary stats and then QC
    logging.info("\n\nReading in summary statistics.")
    for pop_id in sumstats.keys():
        # Read in if necessary (and update dictionary)
        logging.info("\n\n")
        sumstats[pop_id] = obtain_df(sumstats[pop_id], str(pop_id) + " sumstats", sep)

        # QC summary stats (along with some light validation and some logging of drops)
        pop_df = sumstats[pop_id]
        col_map = column_maps.get(pop_id, None)  # If a column map exists for this pop, use that
        logging.info("\nRunning QC on %s summary statistics", pop_id)
        logging.debug("\tColumn mapping = %s\n", col_map)
        sumstats[pop_id] = process_sumstats(pop_df, re_expr_map, MAMA_REQ_STD_COLS,
                                            filters, col_map)

    # If a SNP list is given, read that in
    if snp_list:
        snp_list = pd.read_csv(snp_list, sep='\n', engine='c', comment='#',
                               dtype=str, names=[SNP_COL], squeeze=True)
        snp_list = pd.Index(data=snp_list, dtype=str)

    # Harmonize the summary stats and LD scores (write to disk if requested)
    harmonize_all(sumstats, ldscores, snp_list)
    if harmonized_file_fstr:
        logging.info("\n\nWriting harmonized summary statistics to disk.\n")
        for (ancestry, phenotype), harm_ss_df in sumstats.items():
            filename = harmonized_file_fstr % (ancestry, phenotype)
            logging.debug("\t%s", filename)
            write_sumstats_to_file(filename, harm_ss_df)

    # Log mean chi squared statistics for each population if needed
    if logging.root.level <= logging.DEBUG:
        logging.debug("\n")
        for (ancestry, phenotype), harm_ss_df in sumstats.items():
            mean_chi_sq = calculate_mean_chi_sq(calculate_z(harm_ss_df[BETA_COL].to_numpy(),
                                                            harm_ss_df[SE_COL].to_numpy()))
            logging.debug("Harmonized %s %s mean chi squared: %s",
                ancestry, phenotype, mean_chi_sq)
        logging.debug("\n")

    # If using a standardized units model, convert to stdized units here (convert back later)
    if std_units:
        for pop_id, pop_df in sumstats.items():
            logging.debug("Converting %s to standardized units", pop_id)
            conversion_factor_col = np.sqrt(2.0 * pop_df[FREQ_COL] * (1.0 - pop_df[FREQ_COL]))
            pop_df[BETA_COL] = pop_df[BETA_COL] * conversion_factor_col
            pop_df[SE_COL] = pop_df[SE_COL] * conversion_factor_col

    # Copy values to numpy ndarrays to use in vectorized processing
    beta_arr, se_arr, ldscore_arr = collate_df_values(sumstats, ldscores)

    # Run LD score regressions
    logging.info("\n\nRunning LD Score regression.")
    fixed_opts = {REG_LD_OPT_NAME : ld_opt,
                  REG_SE_OPT_NAME : se_prod_opt,
                  REG_INT_OPT_NAME : int_opt,
                  REG_LD_SCALE_FACTOR_NAME : ld_corr_scale_factor}
    logging.debug("\tOptions = %s", fixed_opts)
    ld_coef, const_coef, se2_coef = run_ldscore_regressions(beta_arr, se_arr,
                                                            ldscore_arr, **fixed_opts)

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
    omega_pos_semi_def, tweaked_omegas = qc_omega(omega) \
        if num_pops > 1 else (np.full(omega.shape[0], True), np.full(omega.shape[0], False))
    omega_drops = np.logical_not(omega_pos_semi_def)
    sigma_drops = np.logical_not(qc_sigma(sigma))
    omega_sigma_drops = np.logical_or(omega_drops, sigma_drops)
    logging.info("Average Omega (including dropped slices) =\n%s", omega.mean(axis=0))
    logging.info("Average Sigma (including dropped slices) =\n%s", sigma.mean(axis=0))

    if num_pops > 1:
        logging.info("\nAdjusted %s SNPs to make omega positive semi-definite.",
                     tweaked_omegas.sum())
        if logging.root.level <= logging.DEBUG:
            omega_tweaked_rsids = ldscores.index[tweaked_omegas].to_list()
            logging.debug("\tRS IDs = %s", omega_tweaked_rsids[:MAX_RSID_LOGGING] + ["..."]
                          if len(omega_tweaked_rsids) > MAX_RSID_LOGGING else omega_tweaked_rsids)

        logging.info("\nDropped %s SNPs due to non-positive-semi-definiteness of omega.",
                     omega_drops.sum())
        if logging.root.level <= logging.DEBUG:
            omega_drop_rsids = ldscores.index[omega_drops].to_list()
            logging.debug("\tRS IDs = %s", omega_drop_rsids[:MAX_RSID_LOGGING] + ["..."]
                          if len(omega_drop_rsids) > MAX_RSID_LOGGING else omega_drop_rsids)
    else:
        logging.info("\nSkipping positive-semi-definiteness check of Omega due to the "
                     "presence of only one population.\n")

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

    num_snps = ldscore_arr.shape[0]  # pylint: disable=unsubscriptable-object
    os_drops_reshaped = omega_sigma_drops.reshape((num_snps, 1, 1))

    # Run the MAMA method
    # Use identity matrices for "bad" SNPs to allow vectorized operations without having to copy
    logging.info("\n\nRunning main MAMA method.")
    new_betas, new_beta_ses = run_mama_method(beta_arr,
                                              np.where(os_drops_reshaped,
                                                       np.identity(omega.shape[1]), omega),
                                              np.where(os_drops_reshaped,
                                                       np.identity(sigma.shape[1]), sigma))

    # Copy values back to the summary statistics DataFrames
    # Also, perform any remaining calculations / formatting
    logging.info("\nPreparing results for output.\n")
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
        z_scores = calculate_z(new_betas[:, pop_num], new_beta_ses[:, pop_num])
        new_df_data_list.append((Z_COL, z_scores))

        # Calculate P column
        new_df_data_list.append((P_COL, calculate_p(z_scores)))

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

        # Report mean chi squared
        mean_chi_2 = calculate_mean_chi_sq(new_df[Z_COL].to_numpy())
        logging.info("\t\tMean Chi^2 for %s = %s", (ancestry, phenotype), mean_chi_2)

        # If using a standardized units model, convert back from stdized units here
        if std_units:
            logging.debug("\t\tConverting %s from standardized units", (ancestry, phenotype))
            conversion_factor_col = np.reciprocal(np.sqrt(
                2.0 * new_df[FREQ_COL] * (1.0 - new_df[FREQ_COL])))
            new_df[BETA_COL] = new_df[BETA_COL] * conversion_factor_col
            new_df[SE_COL] = new_df[SE_COL] * conversion_factor_col

    logging.info("\nFinal SNP count = %s", final_snp_count)

    return sumstats
