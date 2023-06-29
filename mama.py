#!/usr/bin/env python3

"""
Python tool for multi-ancestry, multi-trait analysis
"""

import argparse as argp
import contextlib
import glob
import io
import itertools
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from mama_pipeline import (MAMA_REQ_STD_COLS, MAMA_RE_EXPR_MAP, MAMA_STD_FILTERS,
                           DEFAULT_MAF_MIN, DEFAULT_MAF_MAX, FREQ_FILTER, CHR_FILTER,
                           SNP_PALIN_FILT, DEFAULT_CHR_LIST, mama_pipeline, PopulationId,
                           write_sumstats_to_file)
from reg_mama import (MAMA_REG_OPT_ALL_FREE, MAMA_REG_OPT_ALL_ZERO, MAMA_REG_OPT_OFFDIAG_ZERO,
                      MAMA_REG_OPT_IDENT, MAMA_REG_OPT_SET_CORR)
from util.df import determine_column_mapping, Filter
from util.sumstats import SNP_COL, create_chr_filter, create_freq_filter


# Constants / Parameters / Types #############

ParserFunc = Callable[[str], argp.ArgumentParser]

# Software version
__version__ = '1.0.0'

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org"

# The default short file prefix to use for output and logs
DEFAULT_SHORT_PREFIX = "mama"

# Constants used for labeling output files
RESULTS_SUFFIX = ".res"
HARMONIZED_SUFFIX = ".hrm"
LD_COEF_SUFFIX = "reg.cf"

# Separator used to pass in triples of summary stats file, ancestry, and phenotype
# Note: Do not make this whitespace!  (it will negatively affect parsing)
INPUT_TRIPLE_SEP = ","

# Dictionary keys for internal usage
OUT_DIR = "output_directory"
OUT_PREFIX = "output_prefix"
ANCESTRIES = "ancestries"
RE_MAP = "re_map"
COL_MAP = "col_map"
FILTER_MAP = "filter_map"
SUMSTATS_MAP = "sumstats_map"
REG_LD_COEF_OPT = "regression_ld_option"
REG_SE2_COEF_OPT = "regression_se2_option"
REG_INT_COEF_OPT = "regression_intercept_option"
REG_LD_COEF_SCALE_COEF = "regression_ld_scale_factor"
HARM_FILENAME_FSTR = "harmonized_sumstats_filename_format_str"
REG_FILENAME_FSTR = "regression_coef_filename_format_str"

# Correlation scaling factor min and max
CORR_MIN_SCALING = -1.0
CORR_MAX_SCALING = 1.0

# Derived Constants###########################

# Dictionaries that create and map argparse flags to the corresponding column affected
MAMA_RE_REPLACE_FLAGS = {col : f"replace-{col.lower()}-col-match" for col in MAMA_RE_EXPR_MAP}
MAMA_RE_ADD_FLAGS = {col : f"add-{col.lower()}-col-match" for col in MAMA_RE_EXPR_MAP}

# Default prefix to use for output when not specified
DEFAULT_FULL_OUT_PREFIX = os.path.join(os.getcwd(), DEFAULT_SHORT_PREFIX)

# Logging banner to use at the top of the log file
HEADER = f"""
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<>
<> MAMA: Multi-Ancestry Meta-Analysis
<> Version: {__version__}
<> (C) 2020 Social Science Genetic Association Consortium (SSGAC)
<> MIT License
<>
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
<> Software-related correspondence: {SOFTWARE_CORRESPONDENCE_EMAIL}
<> All other correspondence: {OTHER_CORRESPONDENCE_EMAIL}
<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
"""


# Functions ##################################

#################################
def numpy_err_handler(err: str, flag: bytes):
    """
    Function that numpy should call when an error occurs.  This is used to ensure that any errors
    are also logged, as opposed to just going to stderr and not being collected in the log

    :param err: String describing the error
    :param flag: A byte describing the error (see numpy.seterrcall() docs)
    """
    logging.error("Received Numpy error: %s (%s)", err, flag)


#################################
def reg_ex(s_input: str) -> str:
    """
    Used for parsing some inputs to this program, namely regular expressions given as input.
    Whitespace is removed, but no case-changing occurs.

    :param s_input: String passed in by argparse

    :return str: The regular expression
    """
    stripped_regex = s_input.strip()
    try:
        re.compile(stripped_regex)
    except re.error as exc:
        raise RuntimeError(f"Invalid regular expression \"{stripped_regex}\" "
                           f"supplied: {exc}") from exc

    return stripped_regex


def input_file(s_input: str) -> str:
    """
    Used for parsing some inputs to this program, namely filenames given as input.
    Whitespace is removed, but no case-changing occurs.  Existence of the file is verified.

    :param s_input: String passed in by argparse

    :return str: The filename
    """
    stripped_file = s_input.strip()
    if not os.path.exists(stripped_file):
        raise FileNotFoundError(f"The input file [{stripped_file}] does not exist.")

    return os.path.abspath(stripped_file)


def output_prefix(s_input: str) -> str:
    """
    Used for parsing some inputs to this program, namely full file prefixes used for output.
    Whitespace is removed, but no case-changing occurs.

    :param s_input: String passed in by argparse

    :return str: The prefix
    """

    stripped_p = s_input.strip()

    # Validate existence of output directory (and that no conflicts exist)
    if os.path.exists(stripped_p):
        raise RuntimeError(f"The designated output prefix \"{stripped_p}\" conflicts with "
                           f"an existing file or directory")

    s_dir = os.path.dirname(stripped_p)
    if not os.path.exists(s_dir):
        raise FileNotFoundError(f"The designated output directory [{s_dir}] does not exist.")

    return stripped_p


def ss_input_tuple(s_input: str) -> Tuple[str, str, str]:
    """
    Used for parsing some inputs to this program, namely the triples used to identify summary
    statistics files, ancestries, and phenotypes.  Whitespace is removed, but no case-changing
    occurs.

    :param s_input: String passed in by argparse

    :return: Tuple (all strings) containing:
                 1) summary statistics file path
                 2) ancestry
                 3) phenotype
    """

    try:
        ss_file, ancestry, phenotype = map(lambda x: x.strip(), s_input.split(INPUT_TRIPLE_SEP))
    except Exception as exc:
        raise RuntimeError(f"Error parsing {s_input} into GWAS file, ancestry, "
                           f"and phenotype") from exc

    return input_file(ss_file), ancestry.strip(), phenotype.strip()


def input_np_matrix(s_input: str) -> np.ndarray:
    """
    Used for parsing some inputs to this program, namely Numpy ndarrays (such as regression
    coefficient matrices).

    :param s_input: String passed in by argparse

    :return: ndarray containing the matrix in the file indicated
    """
    filename = input_file(s_input)
    return np.fromfile(filename, sep='\t')


def glob_path(s_input: str) -> List[str]:
    """
    Used for parsing some inputs to this program, namely glob paths (see Python glob module docs).

    :param s_input: String passed in by argparse

    :return: List of file paths
    """
    file_path_list = glob.glob(s_input)
    if not file_path_list:
        raise RuntimeError(f"Glob string \"{s_input}\" matches with no files.")
    return [os.path.abspath(f) for f in file_path_list]


def corr_coef(s_input: str) -> float:
    """
    Used for parsing some inputs to this program, namely input correlation coefficients

    :param s_input: String passed in by argparse

    :return: Float value of correlation coefficient
    """

    c = float(s_input)
    if c < CORR_MIN_SCALING or c > CORR_MAX_SCALING:
        raise ValueError(f"Value given for correlation coefficient ({s_input}) must be between "
                         f"{CORR_MIN_SCALING} and {CORR_MAX_SCALING}.")
    return c


#################################
def get_mama_parser(progname: str) -> argp.ArgumentParser:
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
    in_opt.add_argument("--sumstats", type=ss_input_tuple, nargs="+", required=True,
                        metavar=f"FILE{INPUT_TRIPLE_SEP}ANCESTRY{INPUT_TRIPLE_SEP}PHENOTYPE",
                        help=f"List of triples F{INPUT_TRIPLE_SEP}A{INPUT_TRIPLE_SEP}P "
                             "where F is path to a summary statistics file, "
                             "A is the name of an ancestry, and P is the name of a "
                             "phenotype.  The ancestry is used for column lookup in the "
                             "LD Score file (columns are expected to be of the form ANC1_ANC2, "
                             "where ANC1 and ANC2 are ancestries.  The ancestry and phenotype "
                             "for a given summary statistics file are used in combination as a "
                             "unique identifier.  Currently, these are all case sensitive.")
    in_opt.add_argument("--ld-scores", type=glob_path, required=True, metavar="GLOB_PATH",
                        help="Path to LD scores file(s).  See python glob module for documentation "
                             "on the string to be provided here (full path with support for \"*\", "
                             "\"?\", and \"[]\").  This string should be encased in quotes.  "
                             "Note: File columns are assumed to be of the form "
                             "ANC1_ANC2, where ANC1 and ANC2 are ancestries.  Matching is case "
                             "sensitive, so these should match exactly to the ancestries passed "
                             "in via the --sumstats flag.")
    in_opt.add_argument("--snp-list", type=input_file, required=False, metavar="FILE",
                        help="Path to optional SNP list file (one rsID per line).  "
                             "If specified, this list will be used to restrict the final list "
                             "of SNPs reported (anything outside of this list will be dropped)")

    # Output Options
    out_opt = parser.add_argument_group(title="Output Specifications")
    out_opt.add_argument("--out", metavar="FILE_PREFIX", type=output_prefix,
                         default=DEFAULT_FULL_OUT_PREFIX,
                         help="Full prefix of output files (logs, sumstats results, etc.).  "
                              f"If not set, [current working directory]/{DEFAULT_SHORT_PREFIX} = "
                              f"\"{DEFAULT_FULL_OUT_PREFIX}\" will be used.  "
                              "Note: The containing directory specified must already exist.")
    out_opt.add_argument("--out-reg-coef", action="store_true",
                         help="If specified, MAMA will output the LD regression coefficients "
                              "to disk.  This is useful for reference, but also in the case "
                              "where it is desired to edit the matrices and then pass back into "
                              "MAMA with the --reg-*-coef options below to enable more complex  "
                              "constraints than are allowed for in the existing precanned "
                              "options.  The mechanism used is Numpy's tofile() method with a "
                              "tab separator (\\t) specified, which produces ASCII files with "
                              "the elements of the matrices listed in row-major order.")
    out_opt.add_argument("--out-harmonized", action="store_true",
                         help="If specified, MAMA will output harmonized summary statistics "
                              "to disk.  This can be useful for reference and (potentially) "
                              "debugging / troubleshooting.  This will take place after "
                              "harmonizing all input GWAS files with each other and the LD scores.")

    # General Options
    gen_opt = parser.add_argument_group(title="General Options")
    gen_opt.add_argument("--use-standardized-units", default=False, action="store_true",
                         help="This option should be specified to cause the processing done in "
                              "MAMA to be done in standardized units.  Inputs and outputs are "
                              "always in allele count, but internal processing can be done in "
                              "standardized units by selecting this option (units will be "
                              "converted to standard units before processing, and then back to "
                              "allele count before final results are reported)")
    gen_opt.add_argument("--input-sep", default=None, type=str,
                         help="This option is what is passed via the \"sep\" argument in Pandas' "
                              "read_csv() function when reading in summary statistics and LD score "
                              "files.  This defaults to None, meaning the delimiter "
                              "will be inferred and the Python parsing engine will be used, which  "
                              "yields maximum flexibility, but slower performance.  Specifying a "
                              "value for this flag will cause MAMA to try to use the C parsing "
                              "engine and can significantly speed up execution, but all input file "
                              "reads will share this parameter, so it must work for all inputs.  "
                              "See Pandas' read_csv \"sep\" parameter documentation for more "
                              "details and information."
                            )
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
    reg_opt = parser.add_argument_group(title="Regression Specifications",
                                        description="Optional regression inputs / constraints")
    #   LD score coefficient options (subgroup)
    reg_ld_opt = reg_opt.add_mutually_exclusive_group()
    reg_ld_opt.add_argument("--reg-ld-coef", type=input_np_matrix, metavar="FILE",
                            help="Optional argument indicating the ASCII file containing the "
                                 "regression coefficients for the LD scores.  If this is "
                                 "specified, this will override calculation of LD score "
                                 "coefficients.  The mechanism used is Numpy's fromfile() method "
                                 "with a tab separator (\\t) specified.  For a CxC matrix, the "
                                 "file is C^2 numbers in row-major order separated by tabs.  "
                                 "This is mutually exclusive with other --reg-ld-* options")
    reg_ld_opt.add_argument("--reg-ld-set-corr", type=corr_coef, metavar="CORR_COEF",
                            help="Optional argument indicating that off-diagonal elements in the "
                                 "LD score coefficients matrix should be set to be equal to the "
                                 "square root of the product of the associated diagonal entries  "
                                 "multiplied by the given scaling factor in range "
                                 f"[{CORR_MIN_SCALING}, {CORR_MAX_SCALING}].  "
                                 "This is mutually exclusive with other --reg-ld-* options")
    reg_ld_opt.add_argument("--reg-ld-unc", action="store_true",
                            help="Optional argument indicating that the LD score regression "
                                 "coefficients are unconstrained.  This is the default option.")
    #   SE^2 coefficient options (subgroup)
    reg_se2_opt = reg_opt.add_mutually_exclusive_group()
    reg_se2_opt.add_argument("--reg-se2-coef", type=input_np_matrix, metavar="FILE",
                             help="Optional argument indicating the file containing the regression "
                                  "coefficients for SE^2.  If this is specified, this will "
                                  "override calculation of SE^2 coefficients.  "
                                  "This is mutually exclusive with other --reg-se2-* options")
    reg_se2_opt.add_argument("--reg-se2-zero", action="store_true",
                             help="Optional argument indicating that the SE^2 coefficients matrix "
                                  "should be set to be all zeroes.  "
                                  "This is mutually exclusive with other --reg-se2-* options")
    reg_se2_opt.add_argument("--reg-se2-ident", action="store_true",
                             help="Optional argument indicating that the SE^2 coefficients matrix "
                                  "should be set to be the identity matrix.  "
                                  "This is mutually exclusive with other --reg-se2-* options")
    reg_se2_opt.add_argument("--reg-se2-diag", action="store_true",
                             help="Optional argument indicating that the SE^2 coefficients matrix "
                                  "should have off-diagonal elements set to zero.  This is the "
                                  "default option.  "
                                  "This is mutually exclusive with other --reg-se2-* options")
    reg_se2_opt.add_argument("--reg-se2-unc", action="store_true",
                             help="Optional argument indicating that the SE^2 regression "
                                  "coefficients are unconstrained.")
    #   Intercept coefficient options (subgroup)
    reg_int_opt = reg_opt.add_mutually_exclusive_group()
    reg_int_opt.add_argument("--reg-int-coef", type=input_np_matrix, metavar="FILE",
                             help="Optional argument indicating the file containing the regression "
                                  "coefficients for the intercept.  If this is specified, this "
                                  "will override calculation of intercept coefficients.  "
                                  "This is mutually exclusive with other --reg-int-* options")
    reg_int_opt.add_argument("--reg-int-zero", action="store_true",
                             help="Optional argument indicating that the intercept coefficients "
                                  "matrix should be set to be all zeroes.  "
                                  "This is mutually exclusive with other --reg-int-* options")
    reg_int_opt.add_argument("--reg-int-diag", action="store_true",
                             help="Optional argument indicating that the intercept coefficients "
                                  "matrix should have off-diagonal elements set to zero.  "
                                  "This is mutually exclusive with other --reg-int-* options")
    reg_int_opt.add_argument("--reg-int-unc", action="store_true",
                             help="Optional argument indicating that the intercept regression "
                                  "coefficients are unconstrained.  This is the default option.")

    # Summary Statistics Filtering Options
    ss_filt_opt = parser.add_argument_group(title="Summary Statistics Filtering Options",
                                            description="Options for filtering/processing "
                                                        "summary stats")
    ss_filt_opt.add_argument("--freq-bounds", nargs=2, metavar=("MIN", "MAX"), type=float,
                             help="This option adjusts the filtering of summary statistics.  "
                                  "Specify minimum frequency first, then maximum.  "
                                  f"Defaults to minimum of {DEFAULT_MAF_MIN} and "
                                  f"maximum of {DEFAULT_MAF_MAX}.")
    ss_filt_opt.add_argument("--allowed-chr-values", type=str.upper, nargs="+",
                             help="This option allows specification of allowed values for the "
                                  "chromosome field in summary statistics.  Case is converted to "
                                  "upper here and in the resulting data.  "
                                  f"Default is {DEFAULT_CHR_LIST}.")
    ss_filt_opt.add_argument("--allow-palindromic-snps", action="store_true",
                             help="This option removes the filter that drops SNPs whose major "
                                  "and minor alleles form a base pair (e.g. Major allele = \'G\' "
                                  "and Minor allele = \'C\')")

    # Summary Statistics Column Options
    ss_col_opt = parser.add_argument_group(title="Summary Statistics Column Options",
                                           description="Options for parsing summary stats columns")
    for col, default_regex in MAMA_RE_EXPR_MAP.items():
        col_opt_group = ss_col_opt.add_mutually_exclusive_group()
        col_opt_group.add_argument("--" + MAMA_RE_ADD_FLAGS[col], metavar="REGEX", type=reg_ex,
                                   help="This option adds to the default (case-insensitive) "
                                        f"regular expression \"{default_regex}\" used for "
                                        f"matching / identifying the {col} column.  "
                                        "Use any valid Python re module string.  "
                                        "Mutually exclusive with other "
                                        f"--*-{col.lower()}-col-match options ")
        col_opt_group.add_argument("--" + MAMA_RE_REPLACE_FLAGS[col], metavar="REGEX", type=reg_ex,
                                   help="This option replaces the default (case-insensitive) "
                                        f"regular expression \"{default_regex}\" used for "
                                        f"matching / identifying the {col} column.  "
                                        "Use any valid Python re module string.  "
                                        "Mutually exclusive with other "
                                        f"--*-{col.lower()}-col-match options ")

    return parser


#################################
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


#################################
def format_terminal_call(cmd: List[str]) -> str:
    """
    Format commands to/from the terminal for readability

    :param cmd: List of strings much like sys.argv

    :return: Formatted string used for display purposes
    """

    return ' '.join(cmd).replace("--", " \\ \n\t--")


#################################
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
    return {user_arg : getattr(parsed_args, user_arg) for user_arg in user_set_args}


#################################
def set_up_logger(log_file: str, log_level: int):
    """
    Set up the logger for this utility.

    :param log_file: Full path to the file used to store logs
    :param log_level: Level used for logging
    """

    log_handlers = []

    # Create the stderr handler
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_formatter = logging.Formatter('%(levelname)s: %(message)s')
    stderr_handler.setFormatter(stderr_formatter)
    log_handlers.append(stderr_handler)

    # Create the stdout handler (if, based on log level, it could possibly have messages to log)
    if log_level <= logging.INFO:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(log_level)
        stdout_formatter = logging.Formatter('%(message)s')
        stdout_handler.setFormatter(stdout_formatter)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        log_handlers.append(stdout_handler)

    # Create the file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(file_formatter)
    log_handlers.append(file_handler)

    # Set logging handlers and level for root logger
    logging.basicConfig(handlers=log_handlers, level=log_level, datefmt='%I:%M:%S %p')


#################################
def setup_func(argv: List[str], get_parser: ParserFunc,
               header: str = HEADER) -> Tuple[argp.Namespace, Dict[str, Any]]:
    """
    Function to handle argument parsing, logging setup, and header printing

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    :param get_parser: Function to call to get argument parser, given a program name

    :return: Tuple of:
               1) Argparse Namespace of parsed arguments
               2) Dictionary of user-specified arguments
    """

    # Parse the input flags using argparse
    parser = get_parser(argv[0])
    parsed_args = parser.parse_args(argv[1:])

    # Break down inputs to keep track of arguments and values specified directly by the user
    user_args = get_user_inputs(argv, parsed_args)

    # Set up the logger
    log_file = parsed_args.out + ".log"
    if parsed_args.quiet:
        log_level = logging.WARN
    elif parsed_args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    set_up_logger(log_file, log_level)

    # Log header and other information
    logging.info(header)
    logging.info("See full log at: %s\n", os.path.abspath(log_file))
    logging.info("\nProgram executed via:\n%s\n", format_terminal_call(argv))

    return parsed_args, user_args


#################################
# Disable pylint branch warning because we actually need all these checks
def validate_reg_options(pargs: argp.Namespace, internal_values: Dict[str, Any]): # pylint: disable=too-many-branches
    """
    Responsible for validating regression-related inputs and populating the internal map with
    values related to this processing

    :param pargs: Result of argparse parsing user command / flags
    :param internal_values: Dictionary containing internal values that might be updated
                            in this function
    """

    # Process regression coefficient options
    num_pops = len(pargs.sumstats)
    num_pops_sq = num_pops * num_pops
    #   1) LD coefs
    ld_coef_matrix = getattr(pargs, "reg_ld_coef", None)
    if ld_coef_matrix is not None:
        if len(ld_coef_matrix) != num_pops_sq:
            raise RuntimeError(f"Expected a matrix with {num_pops_sq} elements for "
                               f"regression coefficients (LD) but got {len(ld_coef_matrix)}.")
        internal_values[REG_LD_COEF_OPT] = ld_coef_matrix.reshape((num_pops, num_pops))
        internal_values[REG_LD_COEF_SCALE_COEF] = None
    elif getattr(pargs, "reg_ld_set_corr", None):
        internal_values[REG_LD_COEF_OPT] = MAMA_REG_OPT_SET_CORR
        internal_values[REG_LD_COEF_SCALE_COEF] = getattr(pargs, "reg_ld_set_corr")
    else:
        internal_values[REG_LD_COEF_OPT] = MAMA_REG_OPT_ALL_FREE
        internal_values[REG_LD_COEF_SCALE_COEF] = None
    logging.debug("Regression coeffient option (LD) = %s", internal_values[REG_LD_COEF_OPT])
    logging.debug("Regression coeffient option (LD Scale) = %s",
                  internal_values[REG_LD_COEF_SCALE_COEF])
    #   2) Intercept coefs
    int_coef_matrix = getattr(pargs, "reg_int_coef", None)
    if int_coef_matrix is not None:
        if len(int_coef_matrix) != num_pops * num_pops:
            raise RuntimeError(f"Expected a matrix with {num_pops_sq} elements for "
                               "regression coefficients (intercept) but got "
                               f"{len(int_coef_matrix)}.")
        internal_values[REG_INT_COEF_OPT] = int_coef_matrix.reshape((num_pops, num_pops))
    elif getattr(pargs, "reg_int_zero", None):
        internal_values[REG_INT_COEF_OPT] = MAMA_REG_OPT_ALL_ZERO
    elif getattr(pargs, "reg_int_diag", None):
        internal_values[REG_INT_COEF_OPT] = MAMA_REG_OPT_OFFDIAG_ZERO
    else:
        internal_values[REG_INT_COEF_OPT] = MAMA_REG_OPT_ALL_FREE
    logging.debug("Regression coeffient option (Intercept) = %s", internal_values[REG_INT_COEF_OPT])
    #   3) SE^2 coefs
    se2_coef_matrix = getattr(pargs, "reg_se2_coef", None)
    if se2_coef_matrix is not None:
        if len(se2_coef_matrix) != num_pops * num_pops:
            raise RuntimeError(f"Expected a matrix with {num_pops_sq} elements for regression "
                               f"coefficients (SE^2) but got {len(se2_coef_matrix)}.")
        internal_values[REG_SE2_COEF_OPT] = se2_coef_matrix.reshape((num_pops, num_pops))
    elif getattr(pargs, "reg_se2_zero", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_ALL_ZERO
    elif getattr(pargs, "reg_se2_ident", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_IDENT
    elif getattr(pargs, "reg_se2_diag", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_OFFDIAG_ZERO
    elif getattr(pargs, "reg_se2_unc", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_ALL_FREE
    else:
        # The default option for SE^2 should be off-diagonal elements set to 0
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_OFFDIAG_ZERO
    logging.debug("Regression coeffient option (SE^2) = %s", internal_values[REG_SE2_COEF_OPT])


#################################
def construct_re_map(pargs: argp.Namespace) -> Dict[str, str]:
    """
    Responsible for constructing the regular expressions map for column matching used by this
    execution of the MAMA program.  It begins with the default map and then adjusts it based on
    user argparse inputs.

    :param pargs: Result of argparse parsing user command / flags

    :return: Dictionary that maps regular expressions to standard column names
             (used for column matching used by this execution of the MAMA program)
    """
    re_map = MAMA_RE_EXPR_MAP.copy()
    for req_col in MAMA_REQ_STD_COLS:
        additional_re = getattr(pargs, to_arg(MAMA_RE_ADD_FLAGS[req_col]), None)
        replacement_re = getattr(pargs, to_arg(MAMA_RE_REPLACE_FLAGS[req_col]), None)
        if additional_re:
            re_map[req_col] = f"{re_map[req_col]}|{additional_re}"
        elif replacement_re:
            re_map[req_col] = replacement_re

    logging.debug("\nRegex map = %s", re_map)
    return re_map


#################################
def construct_filter_map(pargs: argp.Namespace) -> Dict[str, Tuple[Filter, str]]:
    """
    Responsible for constructing the sumstats filter map for QC of GWAS used by this
    execution of the MAMA program.  It begins with the default map and then adjusts it based on
    user argparse inputs.

    :param pargs: Result of argparse parsing user command / flags

    :return: Dictionary that maps names of filters to the function and description of the filter
             (used for GWAS QC by this execution of the MAMA program)
    """

    filt_map = MAMA_STD_FILTERS.copy()
    if getattr(pargs, "freq_bounds", None):
        if pargs.freq_bounds[0] > pargs.freq_bounds[1]:
            raise RuntimeError(f"Minimum MAF ({pargs.freq_bounds[0]}) must be less than "
                               f"maximum MAF ({pargs.freq_bounds[1]})")
        filt_map[FREQ_FILTER] = (create_freq_filter(pargs.freq_bounds[0], pargs.freq_bounds[1]),
                                 "Filters out SNPs with FREQ values outside of "
                                 f"[{pargs.freq_bounds[0]}, {pargs.freq_bounds[1]}]")
    if getattr(pargs, "allowed_chr_values", None):
        filt_map[CHR_FILTER] = (create_chr_filter(pargs.allowed_chr_values),
                                "Filters out SNPs with listed chromosomes not in "
                                f"{pargs.allowed_chr_values}")
    if getattr(pargs, "allow_palindromic_snps", None):
        del filt_map[SNP_PALIN_FILT]

    logging.debug("\nFilter map = %s\n", filt_map)
    return filt_map


#################################
def construct_ss_and_col_maps(pargs: argp.Namespace, re_map: Dict[str, str])\
-> Tuple[Dict[str, Dict[str, str]], Dict[PopulationId, str]]:
    """
    Responsible for constructing:
        1) the map between population ID and column map used for mapping sumstats columns
        2) the map between population ID (ancestry + phenotype) and summary stats filename

    :param pargs: Result of argparse parsing user command / flags

    :return: Tuple containing:
                1) the map between population ID and column map used for mapping sumstats columns
                2) the map between population ID (ancestry + phenotype) and summary stats filename
    """

    col_map = {}
    ss_map = {}
    for ss_file, anc, phen in pargs.sumstats:
        cols = list(pd.read_csv(ss_file, sep=None, engine='python', nrows=1, comment="#").columns)
        ss_map[(anc, phen)] = ss_file
        try:
            col_map[(anc, phen)] = determine_column_mapping(cols, re_map, MAMA_REQ_STD_COLS)
        except RuntimeError as exc:
            raise RuntimeError(f"Column mapping error for summary statistics file {ss_file} "
                               f"(ancestry = {anc} and phenotype = {phen}): {exc}") from exc

    return col_map, ss_map


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
    internal_values = {}

    # Get output directory
    internal_values[OUT_DIR] = os.path.dirname(pargs.out)

    # Validate columns of the LD scores file(s)
    for ld_score_file in pargs.ld_scores:
        ld_cols = set(
            pd.read_csv(ld_score_file, sep=None, engine='python', nrows=1, comment="#").columns)
        ancestries = {a for ss_file, a, p in pargs.sumstats}
        anc_tuples = itertools.combinations_with_replacement(ancestries, 2)
        missing_ld_pair_cols = {anc_tuple for anc_tuple in anc_tuples
                                if not('_'.join(anc_tuple) in ld_cols or
                                       '_'.join(anc_tuple[::-1]) in ld_cols)}
        if missing_ld_pair_cols:
            raise RuntimeError(f"The LD scores file {ld_score_file} is missing columns "
                               f"for the following ancestry pairs: {missing_ld_pair_cols}")
        if SNP_COL not in ld_cols:
            raise RuntimeError(f"The LD scores file {ld_score_file} is missing "
                               f"SNP column \"{SNP_COL}\"")

    # Construct RE map for sumstats column matching (must be done before verifying sumstats columns)
    internal_values[RE_MAP] = construct_re_map(pargs)

    # Construct maps of pop ID to sumstats file and to column mappings (validate along the way)
    internal_values[COL_MAP], internal_values[SUMSTATS_MAP] =\
        construct_ss_and_col_maps(pargs, internal_values[RE_MAP])

    # Create filter map to use for summary statistics
    internal_values[FILTER_MAP] = construct_filter_map(pargs)

    # Validate and process regression options
    validate_reg_options(pargs, internal_values)

    # If harmonized summary statistics should be written to disk, determine filename format string
    internal_values[HARM_FILENAME_FSTR] = pargs.out + "_%s_%s" + HARMONIZED_SUFFIX \
        if getattr(pargs, "out_harmonized", None) else ""

    # If regression coefficients should be written to disk, determine filename format string
    internal_values[REG_FILENAME_FSTR] = pargs.out + "_%s_" + LD_COEF_SUFFIX \
        if getattr(pargs, "out_reg_coef", None) else ""

    # Copy attributes to the internal dictionary from parsed args
    for attr in vars(pargs):
        internal_values[attr] = getattr(pargs, attr)

    # Set some extra values based on parsed arg values
    internal_values[OUT_PREFIX] = os.path.basename(pargs.out)
    internal_values[ANCESTRIES] = ancestries

    return internal_values


#################################
def main_func(argv: List[str]):
    """
    Main function that should handle all the top-level processing for this program

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    """

    # Perform argument parsing and program setup
    parsed_args, user_args = setup_func(argv, get_mama_parser)

    # Set Numpy error handling to shunt error messages to a logging function
    np.seterr(all='call')
    np.seterrcall(numpy_err_handler)

    # Attempt to print package version info (pandas has a nice version info summary)
    if logging.root.level <= logging.DEBUG:
        logging.debug("Printing Pandas' version summary:")
        with contextlib.redirect_stdout(io.StringIO()) as f:
            pd.show_versions()
        logging.debug("%s\n", f.getvalue())

    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        iargs = validate_inputs(parsed_args, user_args)

        # Run the MAMA pipeline
        result_sumstats = mama_pipeline(iargs[SUMSTATS_MAP], iargs['ld_scores'], iargs['snp_list'],
                                        iargs[COL_MAP], iargs[RE_MAP], iargs[FILTER_MAP],
                                        iargs[REG_LD_COEF_OPT], iargs[REG_SE2_COEF_OPT],
                                        iargs[REG_INT_COEF_OPT], iargs[REG_LD_COEF_SCALE_COEF],
                                        iargs['use_standardized_units'], iargs[HARM_FILENAME_FSTR],
                                        iargs[REG_FILENAME_FSTR], iargs['input_sep'])

        # Write out the summary statistics to disk
        logging.info("Writing results to disk.")
        for (ancestry, phenotype), ss_df in result_sumstats.items():
            filename = f"{iargs['out']}_{ancestry}_{phenotype}{RESULTS_SUFFIX}"
            logging.info("\t%s", filename)
            write_sumstats_to_file(filename, ss_df)

        # Log any remaining information TODO(jonbjala) Timing info?
        logging.info("\nExecution complete.\n")

    # Disable pylint error since we do actually want to capture all exceptions here
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(exc)
        sys.exit(1)


#################################
if __name__ == "__main__":

    # Call the main function
    main_func(sys.argv)
