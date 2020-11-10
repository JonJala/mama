#!/usr/bin/env python3

"""
Python functions related to running regressions
"""

import numpy as np


# Functions ##################################

#################################
def run_regression(dep_var: np.ndarray, indep_vars: np.ndarray,
                   fixed_coefs: np.ndarray = None, weights: np.ndarray = None) -> np.ndarray:
    """
    Regress dependent variable on the N_var independent variables indicated in indep_vars.

    If fixed_coefs is specified, it must be of length N_var, with NaNs indicating unconstrained
    variables and other numbers indicating the value to which to fix the corresponding coefficient.

    :param dep_var: 1-D (length = N_pts) ndarray for the dependent variable
    :param indep_vars: N_pts x N_var ndarray describing the independent variables
    :param fixed_coefs: 1-D (length = N_vars) ndarray describing fixed coefficients.
                        If None, all variables are unconstrained.
    :param weights: 1-D (length = N_pts) ndarray describing weights.
                        (variables are premultiplied by component-wise sqrt of weights)
                        If None, this an unweighted regression (weights are all effectively 1.0)

    :return: 1-D (length = N_vars) ndarray containing the regression coefficient values
             in the same order as listed in indep_vars
    """

    # Determine number of independent variables (including constrained / fixed coefficient ones)
    n_var = indep_vars.shape[1]

    # Create empty solution vector
    result = np.zeros(n_var)

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
        unconstrained_var_indices = np.full(n_var, True)

    # If weights are specified, make adjustments as required
    if weights is not None: # Check explicitly against None since ndarray is not True or False
        if np.any(weights < 0.0):
            raise RuntimeError("Regression weights cannot be negative!  Weights = %s" % weights)
        wt_factor = np.sqrt(weights)
    else:
        wt_factor = np.ones_like(dep_var_vect)

    # Run the regression on the (remaining) unconstrained variables
    # It returns a tuple, but we only care about the first element
    result[unconstrained_var_indices] = np.linalg.lstsq(
        wt_factor[:, np.newaxis] * indep_vars[:, unconstrained_var_indices],
        wt_factor * dep_var_vect, rcond=None)[0]

    return result
