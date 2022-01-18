#!/usr/bin/env python3

"""
Python functions for QC and harmonization of summary statistics and LD scores
"""

from typing import Any, Dict, Tuple

import numpy as np

from util.reg import run_regression


# Constants / Parameters / Types #############

REG_LD_OPT_NAME = "ld_fixed_opt"
REG_INT_OPT_NAME = "int_fixed_opt"
REG_SE_OPT_NAME = "se_fixed_opt"
REG_LD_SCALE_FACTOR_NAME = "ld_scale_factor"

MAMA_REG_OPT_ALL_FREE = "all_unconstrained"
MAMA_REG_OPT_ALL_ZERO = "all_zero"
MAMA_REG_OPT_OFFDIAG_ZERO = "offdiag_zero"
MAMA_REG_OPT_IDENT = "identity"
MAMA_REG_OPT_SET_CORR = "set_corr"

LD_SCORE_COEF = 0
CONST_COEF = 1
SE_PROD_COEF = 2
N_VARS = 3  # There are 3 coefficient matrices being determined, see lines immediately above

# Functions ##################################

#################################
def fixed_option_helper(num_pops: int, opt_str: Any = MAMA_REG_OPT_ALL_FREE) -> np.ndarray:
    """
    Determines a fixed coefficient matrix for use in the MAMA regressions based on whichever
    option is passed in (default is totally unconstrained)

    :param num_pops: The number of populations involved (if this is P, the return will be PxP)
    :param opt_str: Option describing the constraints.  Expected to be a string, but if this
                    is an ndarray, this will be passed back as the return value

    :return: An ndarray of constraints / fixed coefficients corresponding to the given option
             and number of populations (PxP matrix if num_pops = P)
    """

    if isinstance(opt_str, np.ndarray):
        m_size = len(opt_str)
        if m_size != num_pops:
            raise RuntimeError(f"Regression coefficient matrix size ({m_size}x{m_size}) "
                               f"does not match number of populations {num_pops}")
        result = opt_str
    elif opt_str == MAMA_REG_OPT_ALL_FREE:
        result = np.full((num_pops, num_pops), np.NaN)
    elif opt_str == MAMA_REG_OPT_ALL_ZERO:
        result = np.zeros((num_pops, num_pops))
    elif opt_str == MAMA_REG_OPT_OFFDIAG_ZERO:
        result = np.zeros((num_pops, num_pops))
        d_indices = np.diag_indices_from(result)
        result[d_indices] = np.NaN
    elif opt_str == MAMA_REG_OPT_IDENT:
        result = np.identity(num_pops)
    elif opt_str == MAMA_REG_OPT_SET_CORR:
        # MAMA_REG_OPT_SET_CORR must be handled elsewhere (a constant matrix does not suffice)
        result = np.full((num_pops, num_pops), np.NaN)
    else:
        raise RuntimeError(f"Invalid type ({type(opt_str)}) or value ({opt_str}%s) for opt_str")

    return result


#################################
def run_ldscore_regressions(harm_betas: np.ndarray, harm_ses: np.ndarray, ldscores: np.ndarray,
                            **kwargs: Dict[Any, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs the LD score and beta SE regression.  Assumes the PxP submatrices in the ldscores and the
    P columns of harmonized summary stat data have the same ordering of corresponding ancestries.

    :param harm_betas: MxP matrix (M SNPs by P populations) of betas / effect sizes
    :param harm_ses: MxP matrix (M SNPs by P populations) of beta standard errors
    :param ldscores: (Mx)PxP symmetric matrices containing LD scores (PxP per SNP)
    :param **kwargs: Keyword arguments such as ld_fixed_opt, se_prod_fixed_opt, and int_fixed_opt,
                     which can be used to control fixed options

    :return: A tuple holding regression coefficient matrices (ldscore, constant, and se^2),
             each one a PxP ndarray
    """

    # Determine some ndarray / matrix dimension lengths
    M = harm_betas.shape[0]
    P = harm_betas.shape[1]

    # Determine fixed options
    ld_fixed_opt = kwargs.get(REG_LD_OPT_NAME, MAMA_REG_OPT_ALL_FREE)
    int_fixed_opt = kwargs.get(REG_INT_OPT_NAME, MAMA_REG_OPT_ALL_FREE)
    se_prod_fixed_opt = kwargs.get(REG_SE_OPT_NAME, MAMA_REG_OPT_ALL_FREE)
    ld_corr_scale_factor = kwargs.get(REG_LD_SCALE_FACTOR_NAME, 1.0)

    # Allocate space for the regression matrix, order will be ld scores, constant, and se product
    # (will be partially overwritten at each iteration but no need to reallocate each time)
    reg_matrix = np.zeros((M, N_VARS))
    reg_matrix[:, CONST_COEF] = np.ones(M)

    # Allocate coefs matrix (3 x P x P, slices are LD score, constant, se^2 in that order)
    result_coefs = np.zeros((N_VARS, P, P))

    # Allocate fixed_coefs matrix (3xPxP, order will be ld scores, constant, and se product)
    fixed_coefs = np.full((N_VARS, P, P), np.NaN)
    fixed_opts = (ld_fixed_opt, int_fixed_opt, se_prod_fixed_opt)  # Same order as *_COEF values
    for i, opt in enumerate(fixed_opts):
        fixed_coefs[i] = fixed_option_helper(P, opt)

    # Allocate population weights
    pop_weights = np.zeros((P, M))

    # Calculate diagonal coefficients first (off-diagonal cases can depend on values from here)
    for p in range(P):
        # Set the needed columns in the regression matrix
        reg_matrix[:, LD_SCORE_COEF] = ldscores[:, p, p] # LD Score column
        reg_matrix[:, SE_PROD_COEF] = np.multiply(harm_ses[:, p], harm_ses[:, p]) # SE product

        # Determine weight factor for this population
        pop_weights[p] = np.where(ldscores[:, p, p] > 0.0, np.reciprocal(ldscores[:, p, p]), 0.0)

        # Run the regression
        result_coefs[:, p, p] = run_regression(
            np.multiply(harm_betas[:, p], harm_betas[:, p]), reg_matrix,
            np.ravel(fixed_coefs[:, p, p]), pop_weights[p])

    # Handle the case where MAMA_REG_OPT_PERF_CORR was set (if P == 1, though we can skip this)
    if P > 1 and not isinstance(ld_fixed_opt, np.ndarray) \
             and ld_fixed_opt == MAMA_REG_OPT_SET_CORR:

        # Calculate the sqrt(diagonal) and create the outer product multiplied by the scale factor
        ld_sqrt_diag = np.sqrt(np.diag(result_coefs[LD_SCORE_COEF, :, :]))
        fixed_coefs[LD_SCORE_COEF] = ld_corr_scale_factor * np.outer(ld_sqrt_diag, ld_sqrt_diag)

        # Need to reset the diagonal elements (don't want the scale factor there)
        np.fill_diagonal(fixed_coefs[LD_SCORE_COEF], np.diag(result_coefs[LD_SCORE_COEF, :, :]))


    # Calculate each off-diagonal element (and therefore its symmetric opposite, as well)
    for p1 in range(P):
        for p2 in range(p1 + 1, P):
            # Set the needed columns in the regression matrix
            reg_matrix[:, LD_SCORE_COEF] = ldscores[:, p1, p2] # LD Score column
            reg_matrix[:, SE_PROD_COEF] = np.multiply(harm_ses[:, p1], harm_ses[:, p2]) # SE product

            # Determine weight factor for this population pair (1/abs(cross-pop LD score))
            cross_pop_weights = np.where(ldscores[:, p1, p2] != 0.0,
                                         np.abs(np.reciprocal(ldscores[:, p1, p2])), 0.0)

            # Run the regression and set opposing matrix entry to make coef matrix symmetric
            result_coefs[:, p1, p2] = run_regression(
                np.multiply(harm_betas[:, p1], harm_betas[:, p2]), reg_matrix,
                np.ravel(fixed_coefs[:, p1, p2]), cross_pop_weights)
            result_coefs[:, p2, p1] = result_coefs[:, p1, p2]


    return result_coefs[LD_SCORE_COEF], result_coefs[CONST_COEF], result_coefs[SE_PROD_COEF]
