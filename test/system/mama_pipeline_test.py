"""
System tests for mama_pipeline.py.  This should be run via pytest.
"""

import os
import sys
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_dir = os.path.abspath(os.path.join(main_dir, 'test'))
data_dir = os.path.abspath(os.path.join(test_dir, 'data'))
sys.path.append(main_dir)

import logging

import numpy as np
import pandas as pd
import pytest

import mama2.mama_pipeline as mp
import mama2.reg_mama as mr
import mama2.util.sumstats as ss


#===================================================================================================
class TestHarmonizeAll:

    _LD_SNPS = ['rs100', 'rs150', 'rs200', 'rs250', 'rs300', 'rs350', 'rs400']
    _LD_DATA = {
        'dummy' : [1, 2, 3, 4, 5, 6, 7],
    }

    _POP1_SNPS = ['rs100', 'rs125', 'rs200', 'rs225', 'rs300', 'rs325', 'rs400', 'rs425']
    _POP1_DATA = {
        ss.A1_COL : ['C', 'T', 'G', 'A', 'C', 'T', 'G', 'A'],
        ss.A2_COL : ['A', 'C', 'T', 'G', 'A', 'C', 'T', 'G'],
        ss.BETA_COL : [1.0, 0.2, 1.0, -0.3, 1.0, 0.5, 1.0, -5.0],
        ss.FREQ_COL : [0.1, 0.9, 0.5, 0.8, 0.2, 0.75, 0.2, 0.7]
    }

    _POP2_SNPS = ['rs050', 'rs100', 'rs200', 'rs300', 'rs325', 'rs350']
    _POP2_DATA = {
        ss.A1_COL : ['C', 'A', 'G', 'A', 'T', 'A'],
        ss.A2_COL : ['A', 'C', 'T', 'C', 'C', 'G'],
        ss.BETA_COL : [0.5, -1.0, 1.0, -1.0, -0.2, 0.5],
        ss.FREQ_COL : [0.8, 0.8, 0.4, 0.5, 0.9, 0.75]
    }

    _POP3_SNPS = ['rs100', 'rs200', 'rs250', 'rs300', 'rs325', 'rs350', 'rs425', 'rs500']
    _POP3_DATA = {
        ss.A1_COL : ['C', 'G', 'A', 'A', 'T', 'G', 'G', 'A'],
        ss.A2_COL : ['A', 'T', 'G', 'C', 'C', 'A', 'A', 'G'],
        ss.BETA_COL : [1.0, 1.0, 0.4, -1.0, 0.7, 0.5, 0.3, -5.0],
        ss.FREQ_COL : [0.1, 0.3, 0.5, 0.8, 0.2, 0.75, 0.2, 0.7]
    }


    #########
    def test__varying_snps__expected_results(self):

        # Intersection should be "rs100", "rs200", and "rs300"
        df_ld = pd.DataFrame(TestHarmonizeAll._LD_DATA,
                           index=TestHarmonizeAll._LD_SNPS)
        df_1 = pd.DataFrame(TestHarmonizeAll._POP1_DATA,
                           index=TestHarmonizeAll._POP1_SNPS)
        df_2 = pd.DataFrame(TestHarmonizeAll._POP2_DATA,
                           index=TestHarmonizeAll._POP2_SNPS)
        df_3 = pd.DataFrame(TestHarmonizeAll._POP3_DATA,
                           index=TestHarmonizeAll._POP3_SNPS)
        pops = {1 : df_1.copy(), 2 : df_2.copy(), 3 : df_3.copy()}

        df_ld_copy = df_ld.copy()
        mp.harmonize_all(pops, df_ld_copy)

        # Make sure the number of populations stays the same
        assert len(pops) == 3
        assert 1 in pops
        assert 2 in pops
        assert 3 in pops

        # Make sure the populations are the expected length and contain the expected SNPs
        for df in pops.values():
            assert len(df) == 3
            assert all(df[ss.BETA_COL] == 1.0)
            assert all(df[ss.FREQ_COL] <= 0.5)
            assert all(x in df.index for x in ['rs100', 'rs200', 'rs300'])

        # Make sure the LD score DF is the expected length and contains the expected SNPs
        assert len(df_ld_copy) == 3
        assert all(x in df_ld_copy.index for x in ['rs100', 'rs200', 'rs300'])


#===================================================================================================
class TestMamaPipeline:

    #########
    def test__single_pop_const_ses__same_betas_scaled_ses(self, caplog):
        caplog.set_level(logging.INFO)

        pop_id = ('POP1', 'PHENO1')
        pop1_ssfile = os.path.abspath(os.path.join(data_dir, 'two_pop/pop1_pheno1_sumstats.txt'))
        orig_df = mp.obtain_df(pop1_ssfile, pop_id)
        orig_df[ss.SE_COL] = 0.5

        ldscore_file = os.path.abspath(os.path.join(data_dir, 'two_pop/pop1_pop2_chr1.l2.ldscore'))

        results = mp.mama_pipeline({pop_id : orig_df.copy()}, [ldscore_file])
        result_df = results[pop_id]

        orig_df.set_index(ss.SNP_COL, inplace=True)

        # Make sure we only compare using SNPs that are in the result
        result_snps = result_df.index
        orig_df = orig_df.loc[result_snps]

        assert np.allclose(result_df[ss.BETA_COL], orig_df[ss.BETA_COL])
        assert np.allclose(1.0, result_df[ss.SE_COL])
