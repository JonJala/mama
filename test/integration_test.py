"""
End-to-end tests of the mama2 software.  This should be run via pytest. TODO(jonbjala)
"""


import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import mama2.mama2 as mama2

test_directory = os.path.dirname(__file__)
data_directory = os.path.join(test_directory, 'data')

#===================================================================================================

def test__two_pops_diff_ancestries_same_pheno__expected_results(request):
    """
    TODO(jonbjala)
    """
    ss1_filename = "two_pop/pop1_pheno1_sumstats.txt"
    ss1_full_filepath = os.path.join(data_directory, ss1_filename)

    ss2_filename = "two_pop/pop2_pheno1_sumstats.txt"
    ss2_full_filepath = os.path.join(data_directory, ss2_filename)

    pheno_name = "dummy_phenotype"
    sumstats = {("POP1", pheno_name) : ss1_full_filepath, ("POP2", pheno_name) : ss2_full_filepath}

    ldscores_filename = "two_pop/pop1_pop2_chr1.l2.ldscore"
    ldscores_full_filepath = os.path.join(data_directory, ldscores_filename)    

    result_sumstats = mama2.mama_pipeline(sumstats, ldscores_full_filepath)

    # print("JJ: DF 1\n", result_sumstats[("POP1", pheno_name)])
    # print("JJ: DF 2\n", result_sumstats[("POP2", pheno_name)])
    assert True

    # TODO(jonbjala) Test existence of harmonized outputs in at least one case
