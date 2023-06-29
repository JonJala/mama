#!/usr/bin/env python3

"""
Wrapper / proxy to call old legacy code to create LD Scores file for MAMA.  This might
eventually be replaced with newly written code.
"""

import argparse as argp
import contextlib
import io
from itertools import combinations, combinations_with_replacement
import logging
from operator import itemgetter
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from mama import (get_user_inputs, numpy_err_handler, output_prefix, ParserFunc, setup_func,
                  set_up_logger, to_arg, to_flag)
from pop_info import PopInfo
from util.bim import BIM_RSID_COL, BIM_CM_COL, BIM_BP_COL, BIM_COL_TYPES
from util.sumstats import SNP_COL

# Software version
__version__ = '1.0.0'

# Separator used to pass in bed/bim/fam prefix and ancestry ID
# Note: Do not make this whitespace!  (it will negatively affect parsing)
INPUT_SEP = ","

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org"

# Logging banner to use at the top of the log file
HEADER = f"""
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<>
<> MAMA LD SCORE GENERATION: Multi-Ancestry Meta-Analysis LD Scores
<> Version: {__version__}
<> (C) 2020 Social Science Genetic Association Consortium (SSGAC)
<> MIT License
<>
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<> Software-related correspondence: {SOFTWARE_CORRESPONDENCE_EMAIL}
<> All other correspondence: {OTHER_CORRESPONDENCE_EMAIL}
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
"""

# The default short file prefix to use for output and logs
# TODO(jonbjala) Factor this out into util (along with copy in mama.py)
DEFAULT_SHORT_PREFIX = "mama_ldscore"

# Default prefix to use for output when not specified
# TODO(jonbjala) Factor this out into util (along with copy in mama.py)
DEFAULT_FULL_OUT_PREFIX = os.path.join(os.getcwd(), DEFAULT_SHORT_PREFIX)

# Default base pair windowing distance
DEFAULT_BP_THRESHOLD = BIM_COL_TYPES[BIM_BP_COL](10**6)

# Internal dictionary names
POPS = 'pop_tuple'
NUM_POPS = 'num_pops'
GENDATA_PATHS = 'gendata_paths'
STANDARDIZE = 'standardize'
WINDOW_COL = 'window_col'
WINDOW_THRESHOLD = 'window_threshold'
OUT_PREFIX = 'out_prefix'


# Returns argparse type validator to make sure a numeric type is non-negative
def non_negative(numeric_type):
    def check_non_negative(val_to_be_checked: numeric_type):
        number_as_type = numeric_type(val_to_be_checked)
        if number_as_type < 0:
            raise argp.ArgumentTypeError("Value specified (%s) must be >= 0" % number_as_type)
        return number_as_type
    return check_non_negative


# TODO(jonbjala) Can maybe make use of input_file?
def bedbimfam_prefix(prefix: str):
    prefix = prefix.strip()

    # TODO(jonbjala) These suffixes should maybe be in the bed.py, bim.py, and fam.py util files?
    for suffix in ('.bed', '.bim', '.fam'):
        if not os.path.exists(prefix + suffix):
                raise argp.ArgumentTypeError("The input file [%s%s] does not exist." %
                                             (prefix, suffix))
    return prefix


def gendata_pair(s_input: str) -> Tuple[str, str]:
    """
    Used for parsing some inputs to this program, namely the pairs used to identify bed/bim/fam
    files along with ancestries.  Whitespace is removed, but no case-changing
    occurs.

    :param s_input: String passed in by argparse

    :return: Tuple (all strings) containing:
                 1) bed/bim/fam path prefix
                 2) ancestry
    """
    try:
        file_prefix, ancestry = map(lambda x: x.strip(), s_input.split(INPUT_SEP))
    except Exception as exc:
        raise argp.ArgumentTypeError("Error parsing %s into file prefix and ancestry" %
                                     s_input) from exc

    return ancestry, bedbimfam_prefix(file_prefix)


#################################
# TODO(jonbjala) Make use of "dest" parameter so that the args namespace attributes can easily
# be referred to by constants?
def get_ldscore_parser(progname: str) -> argp.ArgumentParser:
    """
    Return a parser configured for this command line utility

    :param prog: Value to pass to ArgumentParser for prog (should generally be sys.argv[0])

    :return: argparse ArgumentParser
    """

    # Create the initally blank parser
    parser = argp.ArgumentParser(prog=progname)

    # Now, add argument groups and options:

    # Input Options
    in_opt = parser.add_argument_group(title="Main Input Specifications")
    in_opt.add_argument("--gendata", type=gendata_pair, nargs="+", required=True,
                        metavar="FILE_PREFIX%sANCESTRY" % INPUT_SEP,
                        help="List of pairs FP%sA where FP is path prefix to a set of bed/bim/fam"
                             "files and A is the name of an ancestry.  The ancestry is used for "
                             "to generate LD scores labeled by ANC1_ANC2 in the resulting LD Score "
                             "file where ANC1 and ANC2 are ancestries.  Currently, the bed/bim/fam "
                             "files are assumed to be single-chromosome and the ancestries are "
                             "case sensitive." % INPUT_SEP)

    # Output Options
    out_opt = parser.add_argument_group(title="Output Specifications")
    out_opt.add_argument("--out", metavar="FILE_PREFIX", type=output_prefix,
                         default=DEFAULT_FULL_OUT_PREFIX,
                         help="Full prefix of output files (logs, LD score file, etc.).  If not "
                              "set, [current working directory]/%s = \"%s\" will be used.  "
                              "Note: The containing directory specified must already exist." %
                              (DEFAULT_SHORT_PREFIX, DEFAULT_FULL_OUT_PREFIX))
    # TODO(jonbjala) Allow for output of R matrices and other intermediate info?

    # General Options
    gen_opt = parser.add_argument_group(title="General Options")
    gen_opt.add_argument("--use-standardized-units", default=False, action="store_true",
                         help="This option should be specified to cause the processing done in "
                              "MAMA to be done in standardized units.")

    #   Logging options (subgroup)
    log_opt = gen_opt.add_mutually_exclusive_group()
    log_opt.add_argument("--quiet", action="store_true",
                         help="This option will cause the program to limit logging and terminal "
                              "output to warnings and errors, reducing output compared to "
                              "the default/standard logging mode.  It is mutually "
                              "exclusive with the --verbose/--debug option.")
    log_opt.add_argument("--verbose", action="store_true",
                         help="This option will greatly increase the logging and terminal output "
                              "of the program compared to the default/standard logging mode.  "
                              "This is useful for debugging and greater visibility into the "
                              "processing that is occurring.  It is mutually exclusive with the "
                              "--quiet option.")

    # Regression Options
    reg_opt = parser.add_argument_group(title="Window Specifications",
                                        description="Determining the bandwidth of the covariances")
    #   LD score coefficient options (subgroup)
    reg_ld_opt = reg_opt.add_mutually_exclusive_group()
    reg_ld_opt.add_argument("--window-bp", type=non_negative(BIM_COL_TYPES[BIM_BP_COL]),
                            metavar="NUM_BASEPAIRS",
                            help="Specifies that the windowing around a given SNP is in number of "
                                 "base pairs (inclusive).  Value must be non-negative.  "
                                 "This is mutually exclusive with other --window-* options")
    reg_ld_opt.add_argument("--window-cm", type=non_negative(BIM_COL_TYPES[BIM_CM_COL]),
                            metavar="NUM_CM",
                            help="Specifies that the windowing around a given SNP is in number of "
                                 "centimorgans (inclusive).  Value must be non-negative.  "
                                 "This is mutually exclusive with other --window-* options")


    # Summary Statistics Filtering Options
    snp_filt_opt = parser.add_argument_group(title="SNP Filtering Options",
                                             description="Options for filtering/processing "
                                                         "summary stats")
    # TODO(jonbjala) Allow for SNP filtering options
    # snp_filt_opt.add_argument("--freq-bounds", nargs=2, metavar=("MIN", "MAX"), type=float,
    #                           help="This option adjusts the filtering of summary statistics.  "
    #                               "Specify minimum frequency first, then maximum.  "
    #                               "Defaults to minimum of %s and maximum of %s." %
    #                                (DEFAULT_MAF_MIN, DEFAULT_MAF_MAX))



    return parser



#################################
def validate_inputs(pargs: argp.Namespace, user_args: Dict[str, Any]):
    """
    Responsible for coordinating whatever initial validation of inputs can be done

    :param pargs: Result of argparse parsing user command / flags
    :param user_args: Flags explicitly set by the user along with their values

    :return: Dictionary that contains flags and parameters needed by this program.  It contains
             user-input flags along with defaults set through argparse, and any additional flags
             added as calculations proceed
    """

    # Log user-specified arguments
    logging.debug("\nProgram was called with the following arguments:\n%s", user_args)

    # Prepare dictionary that will hold internal values for this program
    internal_values = dict()

    internal_values[GENDATA_PATHS] = dict(pargs.gendata)
    internal_values[POPS] = tuple(internal_values[GENDATA_PATHS].keys())
    internal_values[NUM_POPS] = len(internal_values[POPS])

    internal_values[STANDARDIZE] = pargs.use_standardized_units

    if pargs.window_bp:
        internal_values[WINDOW_COL] = BIM_BP_COL
        internal_values[WINDOW_THRESHOLD] = pargs.window_bp
    elif pargs.window_cm:
        internal_values[WINDOW_COL] = BIM_CM_COL
        internal_values[WINDOW_THRESHOLD] = pargs.window_cm
    else:
        internal_values[WINDOW_COL] = BIM_BP_COL
        internal_values[WINDOW_THRESHOLD] = DEFAULT_BP_THRESHOLD

    internal_values[OUT_PREFIX] = pargs.out


    return internal_values


#################################
def main_func(argv: List[str]):
    """
    Main function that should handle all the top-level processing for this program

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    """

    # Perform argument parsing and program setup
    parsed_args, user_args = setup_func(argv, get_ldscore_parser, HEADER)

    # Set Numpy error handling to shunt error messages to a logging function
    np.seterr(all='call')
    np.seterrcall(numpy_err_handler)

    # Attempt to print package version info (pandas has a nice version info summary)
    if logging.root.level <= logging.DEBUG:
        logging.debug("\nPrinting Pandas' version summary:")
        with contextlib.redirect_stdout(io.StringIO()) as f:
            pd.show_versions()
        logging.debug("%s\n", f.getvalue())

    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        iargs = validate_inputs(parsed_args, user_args)

        # Read in population info
        # TODO(jonbjala) Need to either use temp dir or allow for user specified dir for R matrix files
        logging.info("\nPerforming per-population calculations (filtering and correlations)...")
        popinfo = {p : PopInfo(pop_id=p, bedbimfam_prefix=iargs[GENDATA_PATHS][p],
                               dist_col=iargs[WINDOW_COL], win_size=iargs[WINDOW_THRESHOLD],
                               standardize=iargs[STANDARDIZE],
                               r_band_filename="%s_%s_R.npy" % (iargs[OUT_PREFIX], p))
                   for p in iargs[POPS]}

        # Calculate cross-population indices and store them in the popinfo objects
        logging.info("\nCalculating cross-population indices...")
        for p1info, p2info in combinations(popinfo.values(), 2):
            p1info.calc_cross_pop_indices(p2info)

        # Calculate LD scores
        logging.info("\nCalculating LD scores...")
        # ldscores_list = [p1info.calc_ldscores(p2info) for p1info, p2info in
        #                     combinations_with_replacement(popinfo.values(), 2)]
        # This approach is longer than a one-liner that uses a list comprehension, but does
        # allow for reverse ordering by R matrix size and making use of some caching for greater
        # IO efficiency
        sorted_pop_tuple = sorted((pop_obj for pop_obj in popinfo.values()),
                                  key=lambda p: p.M * p.max_lower_extent, reverse=True)
        ldscores_list = []
        for p1_index in range(iargs[NUM_POPS]):
            p1_info = sorted_pop_tuple[p1_index]
            p1_r = p1_info.get_banded_R()
            for p2_index in range(p1_index, iargs[NUM_POPS]):
                p2_info = sorted_pop_tuple[p2_index]
                ldscores_list.append(p1_info.calc_ldscores(p2_info, self_mat=p1_r))
        del p1_r

        # Collate all the LD scores
        logging.info("\nCollating LD scores...")
        result_df = pd.concat(ldscores_list, axis=1, join="outer")

        # Save scores to disk
        # TODO(jonbjala) Maybe make this a function, at least use constants for sep and na_rep
        ld_score_filename = f"{iargs[OUT_PREFIX]}_ldscores.txt"
        logging.info("\nSaving LD scores to [%s]...", ld_score_filename)
        result_df.to_csv(ld_score_filename, sep="\t", na_rep="NaN", index_label=SNP_COL)

        # Log any remaining information TODO(jonbjala) Timing info?
        logging.info("\nExecution complete.\n")

    # Disable pylint error since we do actually want to capture all exceptions here
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(exc)
        sys.exit(1)



if __name__ == '__main__':
    main_func(sys.argv)
