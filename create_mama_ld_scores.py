#!/usr/bin/env python3

"""
Python helper tool to create LD scores for running MAMA
"""

import argparse as argp
import logging
import sys
from typing import Any, Dict, List

from .mama2 import setup_func

# TODO(jonbjala) Can use function to limit argparse values some: https://docs.python.org/3/library/argparse.html#type  (like making sure int is non-neg)


# Constants / parameters #############

# Default minimum MAF for processing in LD calculations (input filter)
DEFAULT_MIN_IN_MAF = 0.5

# Default window size (units = # of base pairs in each direction)
DEFAULT_WINDOW_SIZE = 100

# Default minimum MAF for producing an LD score (output filter)
DEFAULT_MIN_OUT_MAF = 0.5

####################################################################################################

# Derived constants #############

####################################################################################################

# Functions and Classes #############

def get_ld_parser(progname: str) -> argp.ArgumentParser:
    """
    Return a parser configured for this command line utility

    :param prog: Value to pass to ArgumentParser for prog (should generally be sys.argv[0])

    :return: argparse ArgumentParser
    """

    parser = argp.ArgumentParser(prog=progname)

    # Input Options
    ld_in_opts = parser.add_argument_group(title="LD Score Input Specifications",
                                           description="Options for LD Score Input")
    ld_in_opts.add_argument("--min_in_maf", type=float, default=DEFAULT_MIN_OUT_MAF,
                            help="TODO(jonbjala)")

    # LD Score Creation Options
    ld_opts = parser.add_argument_group(title="LD Score Creation Specifications",
                                        description="Options for LD Score Creation")
    ld_opts.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                         help="TODO(jonbjala)")

    # Output Filtering Options
    ld_out_opts = parser.add_argument_group(title="LD Score Output Specifications",
                                            description="Options for LD Score Output")
    ld_out_opts.add_argument("--min_out_maf", type=float, default=DEFAULT_MIN_OUT_MAF,
                             help="TODO(jonbjala)")
    ld_out_opts.add_argument("--standardize-output", action="store_true",
                             help="If set, output will be in standardized units, rather than "
                             "allele count, which is the default behavior.")
    # TODO(jonbjala)

    return parser


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


def create_ld_scores(iargs: Dict[str, Any]):
    """
    Function that creates LD scores

    :param iargs: Dictionary holding internal values for this program
    """


def main_func(argv: List[str]):
    """
    Main function that should handle all the top-level processing for this program

    :param argv: List of arguments passed to the program (meant to be sys.argv)
    """

    # Perform argument parsing and program setup
    parsed_args, user_args = setup_func(argv, get_ld_parser)


    # Execute the rest of the program, but catch and log exceptions before failing
    try:

        # Validate user inputs and create internal dictionary
        iargs = validate_inputs(parsed_args, user_args)

        # Create LD scores
        create_ld_scores(iargs)

        # Log any remaining information (like timing info?) TODO(jonbjala)

    except Exception as ex:
        logging.exception(ex)
        sys.exit(1)


if __name__ == "__main__":

    # Call the main function
    main_func(sys.argv)
