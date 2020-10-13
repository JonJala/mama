#!/usr/bin/env python3

"""
Python tool for multi-ancestry, multi-trait analysis
"""

import argparse as argp
import itertools
import logging
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import pandas as pd

from mama_pipeline import (MAMA_REQ_STD_COLS, MAMA_RE_EXPR_MAP, MAMA_STD_FILTERS,
                           DEFAULT_MAF_MIN, DEFAULT_MAF_MAX, mama_pipeline)
from reg_mama import (MAMA_REG_OPT_ALL_FREE, MAMA_REG_OPT_ALL_ZERO, MAMA_REG_OPT_OFFDIAG_ZERO,
                      MAMA_REG_OPT_IDENT, MAMA_REG_OPT_PERF_CORR)
from sumstats import SNP_COL
from util.df import determine_column_mapping

# Constants / Parameters / Types #############

ParserFunc = Callable[[str], argp.ArgumentParser]

# Software version
__version__ = '1.0.0'

# Email addresses to use in header banner to denote contacts
SOFTWARE_CORRESPONDENCE_EMAIL1 = "grantgoldman0@gmail.com"
SOFTWARE_CORRESPONDENCE_EMAIL2 = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org"

# The default short file prefix to use for output and logs
DEFAULT_SHORT_PREFIX = "mama"

# Constants used for labeling output files
RESULTS_SUFFIX = ".res"
HARMONIZED_SUFFIX = ".hrm"
LD_COEF_SUFFIX = "reg.coef"

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
HARM_FILENAME_FSTR = "harmonized_sumstats_filename_format_str"


# Derived Constants###########################

# Dictionaries that create and map argparse flags to the corresponding column affected
MAMA_RE_REPLACE_FLAGS = {col : "replace-%s-col-match" % col.lower() for col in MAMA_REQ_STD_COLS}
MAMA_RE_ADD_FLAGS = {col : "add-%s-col-match" % col.lower() for col in MAMA_REQ_STD_COLS}

# Default prefix to use for output when not specified
DEFAULT_FULL_OUT_PREFIX = "%s/%s" % (os.getcwd(), DEFAULT_SHORT_PREFIX)

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


# Functions ##################################

#################################
def reg_ex(s:str) -> str:
    """
    Used for parsing some inputs to this program, namely regular expressions given as input.
    Whitespace is removed, but no case-changing occurs.

    :param s: String passed in by argparse

    :return str: The regular expression
    """
    stripped_regex = s.strip()
    try:
        re.compile(stripped_regex)
    except re.error as ex:
        raise RuntimeError("Invalid regular expression \"%s\" supplied: %s" %
                           (stripped_regex, ex))

    return stripped_regex


def input_file(s:str) -> str:
    """
    Used for parsing some inputs to this program, namely filenames given as input.
    Whitespace is removed, but no case-changing occurs.  Existence of the file is verified.

    :param s: String passed in by argparse

    :return str: The filename
    """
    stripped_file = s.strip()
    if not os.path.exists(stripped_file):
        raise FileNotFoundError("The input file [%s] does not exist." % stripped_file)

    return os.path.abspath(stripped_file)


def input_prefix(s:str) -> str:
    """
    Used for parsing some inputs to this program, namely full file prefixes given as input.
    Whitespace is removed, but no case-changing occurs.

    :param s: String passed in by argparse

    :return str: The prefix
    """

    stripped_p = s.strip()

    # Validate existence of output directory (and that no conflicts exist)
    if os.path.exists(stripped_p):
        raise RuntimeError("The designated output prefix \"%s\" conflicts with an existing "
                           "file or directory" % stripped_p)

    s_dir = os.path.dirname(stripped_p)
    if not os.path.exists(s_dir):
        raise FileNotFoundError("The designated output directory [%s] does not exist." % s_dir)

    return stripped_p


def ss_input_tuple(s:str) -> Tuple[str, str, str]:
    """
    Used for parsing some inputs to this program, namely the triples used to identify summary
    statistics files, ancestries, and phenotypes.  Whitespace is removed, but no case-changing
    occurs.

    :param s: String passed in by argparse

    :return: Tuple (all strings) containing:
                 1) summary statistics file path
                 2) ancestry
                 3) phenotype
    """
    try:
        ss_file, ancestry, phenotype = map(lambda x: x.strip(), s.split(INPUT_TRIPLE_SEP))
    except:
        raise RuntimeError("Error parsing %s into GWAS file, ancestry, and phenotype" % s)

    return input_file(ss_file), ancestry.strip(), phenotype.strip()


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
                        metavar="FILE%sANCESTRY%sPHENOTYPE" % (INPUT_TRIPLE_SEP, INPUT_TRIPLE_SEP),
                        help="List of triples F%sA%sP where F is path to a summary statistics "
                             "file, A is the name of an ancestry, and P is the name of a "
                             "phenotype.  The ancestry is used for column lookup in the "
                             "LD Score file (columns are expected to be of the form ANC1_ANC2, "
                             "where ANC1 and ANC2 are ancestries.  The ancestry and phenotype "
                             "for a given summary statistics file are used in combination as a "
                             "unique identifier.  Currently, these are all case sensitive." %
                             (INPUT_TRIPLE_SEP, INPUT_TRIPLE_SEP))
    in_opt.add_argument("--ld-scores", type=str, required=True, metavar="FILE",
                        help="Path to LD scores file.  Columns are assumed to be of the form "
                             "ANC1_ANC2, where ANC1 and ANC2 are ancestries.  Matching is case "
                             "sensitive, so these should match exactly to the ancestries passed "
                             "in via the --sumstats flag.")

    # Output Options
    out_opt = parser.add_argument_group(title="Output Specifications")
    out_opt.add_argument("--out", metavar="FILE_PREFIX", type=input_prefix,
                         default=DEFAULT_FULL_OUT_PREFIX,
                         help="Full prefix of output files (logs, sumstats results, etc.).  If not "
                              "set, [current working directory]/%s = \"%s\" will be used.  "
                              "Note: The containing directory specified must already exist." %
                              (DEFAULT_SHORT_PREFIX, DEFAULT_FULL_OUT_PREFIX))
    out_opt.add_argument("--out-ld-coef", action="store_true",
                         help="If specified, MAMA will output the LD regression coefficients "
                              "to disk.  This is useful for reference, but also in the case "
                              "where it is desired to edit the matrices and then pass back into "
                              "MAMA with the --reg-*-coef options below to enable more complex  "
                              "constraints than are allowed for in the existing precanned options.")
    out_opt.add_argument("--out-harmonized", action="store_true",
                         help="If specified, MAMA will output harmonized summary statistics "
                              "to disk.  This can be useful for reference and (potentially) "
                              "debugging / troubleshooting.  This will take place after "
                              "harmonizing all input GWAS files with each other and the LD scores.")

    # General Options
    gen_opt = parser.add_argument_group(title="General Options")
    gen_opt.add_argument("--use-standardized-units", action="store_true",
                         help="This option should be specified to cause the processing done in "
                              "MAMA to be done in standardized units.  Inputs and outputs are "
                              "always in allele count, but internal processing can be done in "
                              "standardized units by selecting this option (units will be "
                              "converted to standard units before processing, and then back to "
                              "allele count before final results are reported)")
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
    reg_ld_opt.add_argument("--reg-ld-coef", type=input_file, metavar="FILE",
                            help="Optional argument indicating the file containing the regression "
                                 "coefficients for the LD scores.  If this is specified, this will "
                                 "override calculation of LD score coefficients.  "
                                 "This is mutually exclusive with other --reg-ld-* options")
    reg_ld_opt.add_argument("--reg-ld-perf-corr", action="store_true",
                            help="Optional argument indicating that off-diagonal elements in the "
                                 "LD score coefficients matrix should be set to be equal to the "
                                 "square root of the product of the associated diagonal entries.  "
                                 "This is mutually exclusive with other --reg-ld-* options")
    #   SE^2 coefficient options (subgroup)
    reg_se2_opt = reg_opt.add_mutually_exclusive_group()
    reg_se2_opt.add_argument("--reg-se2-coef", type=input_file, metavar="FILE",
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
                                  "should have off-diagonal elements set to zero.  "
                                  "This is mutually exclusive with other --reg-se2-* options")
    #   Intercept coefficient options (subgroup)
    reg_int_opt = reg_opt.add_mutually_exclusive_group()
    reg_int_opt.add_argument("--reg-int-coef", type=input_file, metavar="FILE",
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

    # Summary Statistics Filtering Options
    ss_filt_opt = parser.add_argument_group(title="Summary Statistics Filtering Options",
                                            description="Options for filtering/processing "
                                                        "summary stats")
    # TODO(jonbjala) Decide if this should be defaulted or just processed if specified
    ss_filt_opt.add_argument("--freq-bounds", default=[DEFAULT_MAF_MIN, DEFAULT_MAF_MAX],
                             nargs=2, metavar=("MIN", "MAX"), type=float,
                             help="This option adjusts the filtering of summary statistics.  "
                                  "Specify minimum frequency first, then maximum.  "
                                  "Defaults to minimum of %s and maximum of %s." %
                                  (DEFAULT_MAF_MIN, DEFAULT_MAF_MAX))
    ss_filt_opt.add_argument("--allow-non-rs", action="store_true",
                             help="This option removes the filter that drops SNPs whose IDs do not "
                                  "begin with \"rs\" (case-insensitive)")
    ss_filt_opt.add_argument("--allow-non-1-22-chr", action="store_true",
                             help="This option removes the filter that drops SNPs whose chromosome "
                                  "number is not in the range 1-22")
    ss_filt_opt.add_argument("--allow-palindromic-snps", action="store_true",
                             help="This option removes the filter that drops SNPs whose major "
                                  "and minor alleles form a base pair (e.g. Major allele = \'G\' "
                                  "and Minor allele = \'C\')")
    # TODO(jonbjala) HWE option?  (need to add that filter first)

    # Summary Statistics Column Options
    ss_col_opt = parser.add_argument_group(title="Summary Statistics Column Options",
                                           description="Options for parsing summary stats columns")
    for col in MAMA_REQ_STD_COLS:
        col_opt_group = ss_col_opt.add_mutually_exclusive_group()
        col_opt_group.add_argument("--" + MAMA_RE_ADD_FLAGS[col], metavar="REGEX", type=reg_ex,
                                   help="This option adds to the default (case-insenstive) "
                                        "regular expression \"%s\" used for "
                                        "matching / identifying the %s column.  "
                                        "Use any valid Python re module string."
                                        "Mutually exclusive with other --*-%s-col-match options " %
                                        (MAMA_RE_EXPR_MAP[col], col, col.lower()))
        col_opt_group.add_argument("--" + MAMA_RE_REPLACE_FLAGS[col], metavar="REGEX", type=reg_ex,
                                   help="This option replaces the default (case-insenstive) "
                                        "regular expression \"%s\" used for "
                                        "matching / identifying the %s column.  "
                                        "Use any valid Python re module string.  "
                                        "Mutually exclusive with other --*-%s-col-match options " %
                                        (MAMA_RE_EXPR_MAP[col], col, col.lower()))

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
    log_file = parsed_args.out + ".log"
    if parsed_args.quiet:
        log_level = logging.WARN
    elif parsed_args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    full_logfile_path = set_up_logger(log_file, log_level)

    # Log header and other information (include parsed, user-specified args at debug level)
    # Note: This is done at WARNING level to ensure it's displayed
    logging.info(HEADER)
    logging.info("See full log at: %s\n", log_file)
    logging.info("Program executed via:\n%s\n", format_terminal_call(argv))
    logging.debug("Program was called with the following arguments:\n%s", user_args)

    return parsed_args, user_args, parser


#################################
def write_sumstats_to_file(filename: str, df: pd.DataFrame):
    """
    Helper function that writes a summary statistics DataFrame to disk

    :param filename: Full path to output file
    :param df: DataFrame holding the summary statistics
    """
    df.to_csv(filename, sep="\t", index_label=SNP_COL)


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

    # Prepare dictionary that will hold internal values for this program
    internal_values = dict()

    # TODO(jonbjala) Verify directory permissions for output?
    out_dir = os.path.dirname(pargs.out)

    # Validate frequency filter bounds
    if pargs.freq_bounds[0] > pargs.freq_bounds[1]:
        raise RuntimeError("Minimum MAF (%s) must be less than maximum MAF (%s) " %
                           (pargs.freq_bounds[0], pargs.freq_bounds[1]))

    # Validate columns of the LD scores file
    ld_cols = set(
        pd.read_csv(pargs.ld_scores, sep=None, engine='python', nrows=1, comment="#").columns)
    ancestries = {a for ss_file, a, p in pargs.sumstats}
    anc_tuples = itertools.combinations_with_replacement(ancestries, 2)
    missing_ld_pair_cols = {anc_tuple for anc_tuple in anc_tuples
        if not("%s_%s" % anc_tuple in ld_cols or "%s_%s" % anc_tuple[::-1] in ld_cols)}
    if missing_ld_pair_cols:
        raise RuntimeError("The LD scores file %s is missing columns for the following "
                           "ancestry pairs: %s" % (pargs.ld_scores, missing_ld_pair_cols))
    if SNP_COL not in ld_cols:
        raise RuntimeError("The LD scores file %s is missing SNP column \"%s\"" %
                           (pargs.ld_scores, SNP_COL))

    # Construct RE map for sumstats column matching (must be done before verifying sumstats columns)
    re_map = MAMA_RE_EXPR_MAP.copy()
    for req_col in MAMA_REQ_STD_COLS:
        additional_re = getattr(pargs, to_arg(MAMA_RE_ADD_FLAGS[req_col]), None)
        replacement_re = getattr(pargs, to_arg(MAMA_RE_REPLACE_FLAGS[req_col]), None)
        if additional_re:
            re_map[req_col] = "%s|%s" % (re_map[req_col], additional_re)
        elif replacement_re:
            re_map[req_col] = replacement_re
    logging.debug("\nRegex map = %s", re_map)

    # Validate columns of all the sumstats files (via trying to map them to standard column names)
    col_map = dict()
    ss_map = dict()
    for ss_file, a, p in pargs.sumstats:
        cols = list(pd.read_csv(ss_file, sep=None, engine='python', nrows=1, comment="#").columns)
        ss_map[(a, p)] = ss_file
        try:
            col_map[(a, p)] = determine_column_mapping(cols, re_map, MAMA_REQ_STD_COLS)
        except RuntimeError as ex:
            raise RuntimeError("Column mapping error for summary statistics file %s (ancestry = "
                               "%s and phenotype = %s): %s" % (ss_file, a, p, ex))

    # Create filter map to use for summary statistics
    filt_map = MAMA_STD_FILTERS.copy()
    if pargs.freq_bounds != [DEFAULT_MAF_MIN, DEFAULT_MAF_MAX]:
        filt_map[FREQ_FILTER] = (create_freq_filter(pargs.freq_bounds[0], pargs.freq_bounds[1]),
                                 DEFAULT_FILTER_FUNC_DESC % "with FREQ values outside of [%s, %s]" %
                                 (pargs.freq_bounds[0], pargs.freq_bounds[1]))
    if getattr(pargs, "allow_non_rs", None):
        del filt_map[SNP_PREFIX_FILTER]
    if getattr(pargs, "allow_non_1_22_chr", None):
        del filt_map[CHR_FILTER]
    if getattr(pargs, "allow_palindromic_snps", None):
        del filt_map[SNP_PALIN_FILT]
    logging.debug("\nFilter map = %s\n", filt_map)

    # Process regression coefficient options
    #   1) LD coefs
    ld_coef_file = getattr(pargs, "reg_ld_coef", None)
    if ld_coef_file:
        # TODO(jonbjala) Check file!
        internal_values[REG_LD_COEF_OPT] = ld_coef_file
    elif getattr(pargs, "reg_ld_perf_corr", None):
        internal_values[REG_LD_COEF_OPT] = MAMA_REG_OPT_PERF_CORR
    else:
        internal_values[REG_LD_COEF_OPT] = MAMA_REG_OPT_ALL_FREE
    logging.debug("Regression coeffient option (LD) = %s", internal_values[REG_LD_COEF_OPT])
    #   2) SE^2 coefs
    se2_coef_file = getattr(pargs, "reg_se2_coef", None)
    if se2_coef_file:
        # TODO(jonbjala) Check file?
        internal_values[REG_SE2_COEF_OPT] = se2_coef_file
    elif getattr(pargs, "reg_se2_zero", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_ALL_ZERO
    elif getattr(pargs, "reg_se2_ident", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_IDENT
    elif getattr(pargs, "reg_se2_diag", None):
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_OFFDIAG_ZERO
    else:
        internal_values[REG_SE2_COEF_OPT] = MAMA_REG_OPT_ALL_FREE
    logging.debug("Regression coeffient option (SE^2) = %s", internal_values[REG_SE2_COEF_OPT])
    #   3) Intercept coefs
    int_coef_file = getattr(pargs, "reg_int_coef", None)
    if int_coef_file:
        # TODO(jonbjala) Check file?
        internal_values[REG_INT_COEF_OPT] = int_coef_file
    elif getattr(pargs, "reg_int_zero", None):
        internal_values[REG_INT_COEF_OPT] = MAMA_REG_OPT_ALL_ZERO
    elif getattr(pargs, "reg_int_diag", None):
        internal_values[REG_INT_COEF_OPT] = MAMA_REG_OPT_OFFDIAG_ZERO
    else:
        internal_values[REG_INT_COEF_OPT] = MAMA_REG_OPT_ALL_FREE
    logging.debug("Regression coeffient option (Intercept) = %s", internal_values[REG_INT_COEF_OPT])

    # If harmonized summary statistics should be written to disk, determine filename format string
    internal_values[HARM_FILENAME_FSTR] = pargs.out + "_%s_%s_" + HARMONIZED_SUFFIX \
        if getattr(pargs, "out_harmonized", None) else ""

    # Copy attributes to the internal dictionary from parsed args
    for attr in vars(pargs):
        internal_values[attr] = getattr(pargs, attr)

    # Set some extra values based on parsed arg values
    internal_values[OUT_DIR] = out_dir
    internal_values[OUT_PREFIX] = os.path.basename(pargs.out)
    internal_values[ANCESTRIES] = ancestries
    internal_values[RE_MAP] = re_map
    internal_values[COL_MAP] = col_map
    internal_values[FILTER_MAP] = filt_map
    internal_values[SUMSTATS_MAP] = ss_map

    return internal_values


#################################
def main_func(argv: List[str]):
    """
    Main function that should handle all the top-level processing for this program

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    """

    # Perform argument parsing and program setup
    parsed_args, user_args, parser = setup_func(argv, get_mama_parser)

    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        iargs = validate_inputs(parsed_args, user_args)

        # Run the MAMA pipeline
        result_sumstats = mama_pipeline(iargs[SUMSTATS_MAP], iargs['ld_scores'], iargs[COL_MAP],
                                        iargs[RE_MAP], iargs[FILTER_MAP], iargs[REG_LD_COEF_OPT],
                                        iargs[REG_SE2_COEF_OPT], iargs[REG_INT_COEF_OPT],
                                        iargs[HARM_FILENAME_FSTR])

        # Write out the summary statistics to disk
        logging.info("Writing results to disk.")
        for (ancestry, phenotype), ss_df in result_sumstats.items():
            filename = "%s_%s_%s%s" % (iargs["out"], ancestry, phenotype, RESULTS_SUFFIX)
            logging.info("\t%s", filename)
            write_sumstats_to_file(filename, ss_df)

        # Log any remaining information TODO(jonbjala) Timing info?
        logging.info("\nExecution complete.\n")

    except Exception as ex:
        logging.exception(ex)
        sys.exit(1)


#################################
if __name__ == "__main__":

    # Call the main function
    main_func(sys.argv)
