#!/usr/bin/env python3

"""
Python tool for multi-ancestry, multi-trait analysis
"""

import argparse as argp
import functools
import gc
import logging
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


# Constants / parameters #############

# Software version
__version__ = '1.0.0'

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL1 = "grantgoldman0@gmail.com"
SOFTWARE_CORRESPONDENCE_EMAIL2 = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org"

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

####################################################################################################

# Functions and Classes #############

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


def harmonize_gwas_with_ldscores(sumstats, ldscores):
    """
    Does the harmonization between the QC'ed input summary statistics and the LD scores

    :param sumstats: TODO(jonbjala)
    :param ldscores: TODO(jonbjala)

    :return: TODO(jonbjala)
    """


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

    :return: A tuple holding regression coefficient matrices (ldscore, se^2, and constant),
             all MxPxP (PxP slices for each of M SNPs)
    """
    # M = harm_betas.shape[0]
    # P = harm_betas.shape[1]

    # ld_score_coefs = np.zeros((3, P, P))
    # const_coefs = np.zeros((3, P, P))
    # se2_coefs = np.zeros((3, P, P))


    # for i in range(P):
    #     for j in range(i,P):

    #         indep_vars = {}

    #         ld_score_coefs[i,i], const_coefs[i,i], se2_coefs[i,i] = run_regression(
    #             harm_betas[:,i], harm_betas[:,i]
    #         )
    #         ld_score_coefs[i,j], const_coefs[i,j], se2_coefs[i,j] = run_regression(
    #         )
    #         ld_score_coefs[j,i] = ld_score_coefs[i,j]
    #         const_coefs[j,i] = const_coefs[i,j]
    #         se2_coefs[j,i] = se2_coefs[i,j]

    # # Stack and rearrange ldscores, constant values, and squared SEs
    # # (into PxMx3 matrix with column order = ldscore, const, se^2)
    # # stacked_reg_vals = np.concatenate((ldscores[np.newaxis, :, :], np.ones((1, M, P)),
    # #    np.square(harm_ses)[np.newaxis, :, :]), axis=0)
    # #stacked_reg_vals = np.transpose(stacked_reg_vals, axes=(2,1,0))

    # print("\nJJ:\n", stacked_reg_vals)

    # return ld_score_coefs, const_coefs, se2_coefs


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



def run_mama_method(harm_betas, omega, sigma):
    """
    Runs the core MAMA method to combine results and generate final, combined summary statistics

    :param harm_sumstats: TODO(jonbjala)
    :param omega: TODO(jonbjala)
    :param sigma: TODO(jonbjala)

    :return: TODO(jonbjala)
    """

    # TODO(jonbjala) Remove printouts, add positive (semi-)def checks and needed processing, dropping some SNPs

    # Get values for M and P (used to keep track of slices / indices / broadcasting)
    M = omega.shape[0]
    P = omega.shape[1]
    print("\nJJ: omega\n", omega)
    print("JJ: \n", omega.shape)
    # Create a 3D matrix, M rows of Px1 column vectors with shape (M, P, 1)
    d_indices = np.arange(P)
    omega_diag = omega[:, d_indices, d_indices][:, :, np.newaxis]
    print("\nJJ: omega_diag\n", omega_diag)
    print("JJ:\n", omega_diag.shape)
    omega_pp_scaled = np.divide(omega, omega_diag)  # Slice rows are Omega'_pjj / omega_pp,j
    print("\nJJ: omega_pp_scaled\n", omega_pp_scaled)
    print("JJ: \n", omega_pp_scaled.shape)

    # Produce center matrix in steps (product of omega terms, add omega and sigma, then invert)
    center_matrix_inv = -omega_pp_scaled[:, :, :, np.newaxis] * omega[:, :, np.newaxis, :]
    print("\nJJ: omega_outer_prod\n", center_matrix_inv)
    print("JJ: \n", center_matrix_inv.shape)
    center_matrix_inv += omega[:,np.newaxis,:,:] + sigma[:,np.newaxis,:,:] # Broadcast add
    print("\nJJ: omega_outer_prod+omega+sigma\n", center_matrix_inv)
    print("JJ: \n", center_matrix_inv.shape)
    center_matrix = np.linalg.inv(center_matrix_inv) # Inverts each slice separately
    del center_matrix_inv; gc.collect() # Clean up the inverse matrix to free space
    print("\nJJ: center_matrix\n", center_matrix)
    print("JJ: \n", center_matrix.shape)

    # Calculate (Omega'_p,j/omega_pp,j) * center_matrix
    left_product = np.matmul(omega_pp_scaled[:, :, np.newaxis, :], center_matrix)
    del center_matrix; gc.collect() # Clean up the center matrix to free space
    print("\nJJ: left_product\n", left_product)
    print("JJ: \n", left_product.shape)

    # Calculate denominator (M x P x 1 x 1)
    denom = np.matmul(left_product,
                      np.transpose(omega_pp_scaled[:, :, np.newaxis, :], (0, 1, 3, 2)))
    # print("\nJJ: denom prod\n", denom)
    # print("JJ:\n", denom.shape)
    denom_recip = np.reciprocal(denom)
    # print("\nJJ: denom \n", denom)
    # print("JJ:\n", denom.shape)
    denom_recip_view = denom_recip.view()
    denom_recip_view.shape = (M, P)
    print("\nJJ: denom_recip_view \n", denom_recip_view)
    print("JJ:\n", denom_recip_view.shape)

    # Calculate numerator (M x P x 1 x 1))
    left_product_view = left_product.view()
    left_product_view.shape = (M, P, P)
    print("\nJJ: left_product_view\n", left_product_view)
    print("JJ: \n", left_product_view.shape)
    harm_betas_t = harm_betas[:,:,np.newaxis]
    print("\nJJ: harm_betas \n", harm_betas)
    print("JJ:\n", harm_betas.shape)
    numer = np.matmul(left_product_view, harm_betas[:,:,np.newaxis])
    print("\nJJ: numer\n", numer)
    print("JJ:\n", numer.shape)
    numer_view = numer.view()
    numer_view.shape = (M, P)
    print("\nJJ: numer_view\n", numer_view)
    print("JJ:\n", numer_view.shape)

    new_betas = denom_recip_view * numer_view
    new_beta_ses = np.sqrt(denom_recip_view)

    print("\nJJ: new_betas\n", new_betas)
    print("JJ:\n", new_betas.shape)
    print("\nJJ: new_beta_ses\n", new_beta_ses)
    print("JJ:\n", new_beta_ses.shape)


def mama_pipeline(iargs):
    """
    Runs the steps in the overall MAMA pipeline

    :param iargs: Internal namespace object that holds both parsed values of input arguments and
                  derived / intermediate values for this program
    """


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
