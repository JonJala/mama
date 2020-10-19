"""
Unit tests for mama_pipeline.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import numpy as np
import pandas as pd
import pytest

import mama2.mama_pipeline as mp
import mama2.util.sumstats as ss

@pytest.fixture(scope="function", params=[1, 2, 3, 10])
def collate_df_values_test_df(request):

    num_snps = request.param
    snps = [f'rs{s:03d}' for s in range(1, num_snps + 1)]

    return pd.DataFrame(index=snps, columns=[ss.BETA_COL, ss.SE_COL])


###########################################

class TestCollateDfValues:

    #########
    @pytest.mark.parametrize("num_pops", [1, 2, 3, 5])
    def test__diff_ancestry_same_pheno__return_expected(self, collate_df_values_test_df, num_pops):

        num_snps = len(collate_df_values_test_df)
        pheno = "dummy_phen"

        # Create population data frames
        sumstats = {(i, pheno) : collate_df_values_test_df.copy() for i in range(num_pops)}
        for pop_df in sumstats.values():
            pop_df[ss.BETA_COL] = np.random.rand(num_snps)
            pop_df[ss.SE_COL] = np.random.rand(num_snps)
        pop_ids = list(sumstats.keys())

        # Create LD score data frames
        ld_cols = ["%s_%s" % (i[0],j[0]) for i in pop_ids for j in pop_ids if i[0] >= j[0]]
        ldscores = pd.DataFrame(index=collate_df_values_test_df.index, columns=ld_cols)
        for col in ld_cols:
            ldscores[col] = np.random.rand(num_snps)

        # Use default order, then specify (should be same as default), then reverse order
        beta_arr_1, se_arr_1, ld_arr_1 = mp.collate_df_values(sumstats, ldscores)
        beta_arr_2, se_arr_2, ld_arr_2 = mp.collate_df_values(sumstats, ldscores, pop_ids)
        beta_arr_3, se_arr_3, ld_arr_3 = mp.collate_df_values(sumstats, ldscores, pop_ids[::-1])

        # Check the first return element-wise
        for pop_num in range(num_pops):
            df = sumstats[(pop_num, pheno)]
            assert np.allclose(df[ss.BETA_COL], beta_arr_1[:, pop_num])
            assert np.allclose(df[ss.SE_COL], se_arr_1[:, pop_num])
        for col in ldscores.columns:
            p_nums  = col.split("_")
            p1, p2 =  (int(p_nums[0]), int(p_nums[1]))
            assert np.allclose(ldscores[col], ld_arr_1[:, p1, p2])
            assert np.allclose(ldscores[col], ld_arr_1[:, p2, p1])

        # Now check the other two calls against the first (should be same, and reversed)
        assert np.allclose(beta_arr_1, beta_arr_2)
        assert np.allclose(se_arr_1, se_arr_2)
        assert np.allclose(ld_arr_1, ld_arr_2)

        assert np.allclose(beta_arr_1, beta_arr_3[:, ::-1])
        assert np.allclose(se_arr_1, se_arr_3[:, ::-1])
        assert np.allclose(ld_arr_1, ld_arr_3[:, ::-1, ::-1])


    #########
    @pytest.mark.parametrize("num_phen", [1, 2, 3, 5])
    def test__same_ancestry_diff_pheno__return_expected(self, collate_df_values_test_df, num_phen):

        num_snps = len(collate_df_values_test_df)
        phens = ["phen_%s" % i for i in range(num_phen)]

        # Create population data frames
        sumstats = {(0, pheno) : collate_df_values_test_df.copy() for pheno in phens}
        for pop_df in sumstats.values():
            pop_df[ss.BETA_COL] = np.random.rand(num_snps)
            pop_df[ss.SE_COL] = np.random.rand(num_snps)
        pop_ids = list(sumstats.keys())

        # Create LD score data frames
        ld_cols = ["0_0"]
        ldscores = pd.DataFrame(index=collate_df_values_test_df.index, columns=ld_cols)
        for col in ld_cols:
            ldscores[col] = np.random.rand(num_snps)

        # Use default order, then specify (should be same as default), then reverse order
        beta_arr_1, se_arr_1, ld_arr_1 = mp.collate_df_values(sumstats, ldscores)
        beta_arr_2, se_arr_2, ld_arr_2 = mp.collate_df_values(sumstats, ldscores, pop_ids)
        beta_arr_3, se_arr_3, ld_arr_3 = mp.collate_df_values(sumstats, ldscores, pop_ids[::-1])

        # Check the first return element-wise
        for phen_num in range(num_phen):
            df = sumstats[(0, phens[phen_num])]
            assert np.allclose(df[ss.BETA_COL], beta_arr_1[:, phen_num])
            assert np.allclose(df[ss.SE_COL], se_arr_1[:, phen_num])
        for col in ldscores.columns:
            assert np.allclose(ldscores[col], ld_arr_1[:, 0, 0])
            assert np.allclose(ldscores[col], ld_arr_1[:, 0, 0])

        # Now check the other two calls against the first (should be same, and reversed)
        assert np.allclose(beta_arr_1, beta_arr_2)
        assert np.allclose(se_arr_1, se_arr_2)
        assert np.allclose(ld_arr_1, ld_arr_2)

        assert np.allclose(beta_arr_1, beta_arr_3[:, ::-1])
        assert np.allclose(se_arr_1, se_arr_3[:, ::-1])
        assert np.allclose(ld_arr_1, ld_arr_3[:, ::-1, ::-1])

    # TODO(jonbjala) Need to test a "mixed" case where there are multiple ancestries and phenotypes

# TODO(jonbjala) Test qc_ldscores? obtain_df()?

###########################################

class TestCalculateNEff:

    #########
    # TODO(jonbjala) Need a few more cases
    @pytest.mark.parametrize("pop_num, n_orig, sigma, se, expected",
        [
            (0, np.array([100, 1000, 5000]), np.full((3, 2, 2), 0.5 * np.identity(2)),
                np.array([1.0, 0.5, 0.25]), np.array([50, 2000, 40000]))
        ])
    def test__varying_inputs__return_expected(self, pop_num, n_orig, sigma, se, expected):
        assert np.allclose(mp.calculate_n_eff(pop_num, n_orig, sigma, se), expected)
