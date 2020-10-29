#!/usr/bin/env python3

"""
Python functions that implement the core MAMA processing
"""

import gc
from typing import Tuple

import numpy as np


# Functions ##################################

#################################
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


#################################
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


#################################
def qc_omega(omega: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs checks over the omega matrices for positive-semi-definiteness.  Tweaks omega where possible
    to correct for non-positive-semi-definiteness and returns an array of length M
    (where M = number of SNPs) along the SNP axis (the first dimension of the MxPxP omega)
    where True indicates positive semi-definiteness and False indicates
    non-positive semi-definiteness

    :param omega: MxPxP matrix for Omega values

    :return: Tuple containing:
                1) Array of length M where True indicates positive semi-definiteness and False
                   indicates non-positive semi-definiteness
                2) Array of length M where True indicates the omega was tweaked to make it
                   positive semi-definite (False otherwise)
    """

    # Create result vectors of length M, all values defaulting to False
    M = omega.shape[0]
    pos_semi_def_indices = np.full(M, False)
    tweaked_omega_indices = np.full(M, False)

    # Iterate over the M PxP matrices of sigma
    for i in range(M):
        omega_slice = omega[i, :, :]

        # Check for positive semi-definiteness (if PSD, set to True and move on)
        if np.all(np.linalg.eigvalsh(omega_slice) >= 0.0):
            pos_semi_def_indices[i] = True
            continue

        # If diagonal entries aren't positive, move on
        if np.any(np.diag(omega_slice) <= 0.0):
            continue

        # We can try to tweak ths slice of omega to become positive semi-definite
        omega[i, :, :] = tweak_omega(omega_slice)
        pos_semi_def_indices[i] = True
        tweaked_omega_indices[i] = True

    return pos_semi_def_indices, tweaked_omega_indices


#################################
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


#################################
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
        except np.linalg.LinAlgError:
            # If not positive definite, then the Cholesky decomposition raises a LinAlgError
            pass

    return result


#################################
def run_mama_method(betas, omega, sigma):
    """
    Runs the core MAMA method to combine results and generate final, combined summary statistics

    :param harm_sumstats: TODO(jonbjala)
    :param omega: TODO(jonbjala)
    :param sigma: TODO(jonbjala)

    :return: Tuple containing:
                 1) Result ndarray of betas (MxP) where M = SNPs and P = populations
                 2) Result ndarray of beta standard errors (MxP) where M = SNPs and P = populations
    """

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
    del center_matrix_inv  # Clean up the inverse matrix to free space
    gc.collect()

    # Calculate (Omega'_p,j/omega_pp,j) * center_matrix
    left_product = np.matmul(omega_pp_scaled[:, :, np.newaxis, :], center_matrix)
    del center_matrix  # Clean up the center matrix to free space
    gc.collect()

    # Calculate denominator (M x P x 1 x 1)
    denom = np.matmul(left_product,
                      np.transpose(omega_pp_scaled[:, :, np.newaxis, :], (0, 1, 3, 2)))
    denom_recip = np.reciprocal(denom)
    denom_recip_view = denom_recip.view()
    denom_recip_view.shape = (M, P)

    # Calculate numerator (M x P x 1 x 1))
    left_product_view = left_product.view()
    left_product_view.shape = (M, P, P)
    numer = np.matmul(left_product_view, betas[:, :, np.newaxis])
    numer_view = numer.view()
    numer_view.shape = (M, P)

    # Calculate result betas and standard errors
    new_betas = denom_recip_view * numer_view
    new_beta_ses = np.sqrt(denom_recip_view)

    return new_betas, new_beta_ses
