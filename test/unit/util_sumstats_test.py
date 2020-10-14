"""
Unit tests of util/sumstats.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import tempfile

import numpy as np
import pandas as pd
import pytest

import mama2.util.sumstats as ss


###########################################

class TestFlipAlleles:

    #########
    @pytest.mark.parametrize("flips",
    [
            ([True, True, True, False, False, True]),
            ([False, True, False, True, False, True]),
            ([False, False, True, False, True, False]),
            ([True, True, True, True, True, True]),
            ([False, False, False, False, False, False]),
            ([False, False, True, False, False, False])
    ])
    def test__varying_flips__expected_results(self, flips):
        snps = ['rs01', 'rs02', 'rs03', 'rs04', 'rs05', 'rs06']
        a1 = ['G', 'A', 'C', 'T', 'G', 'A']
        a2 = ['T', 'G', 'A', 'C', 'T', 'G']
        betas = [1.0, -2.0, 3.0, -4.0, 5.0, 0.0]
        freqs = [1.0, 0.5, 0.6, 0.4, 0.1, 0.0]
        data = {ss.FREQ_COL:freqs, ss.A1_COL:a1, ss.A2_COL:a2, ss.BETA_COL:betas}
        df = pd.DataFrame(data, index=snps)

        # Flip
        df_copy = df.copy()
        indices = pd.Series(flips, index=snps)
        ss.flip_alleles(df_copy, indices)

        # Check to make sure the results are as expected
        freq_check = (indices & (df_copy[ss.FREQ_COL] == 1.0 - df[ss.FREQ_COL])) | \
                     (~indices & (df_copy[ss.FREQ_COL] == df[ss.FREQ_COL]))
        beta_check = (indices & (df_copy[ss.BETA_COL] == -df[ss.BETA_COL])) | \
                     (~indices & (df_copy[ss.BETA_COL] == df[ss.BETA_COL]))
        a1_check = (indices & (df_copy[ss.A1_COL] == df[ss.A2_COL])) | \
                   (~indices & (df_copy[ss.A1_COL] == df[ss.A1_COL]))
        a2_check = (indices & (df_copy[ss.A2_COL] == df[ss.A1_COL])) | \
                   (~indices & (df_copy[ss.A2_COL] == df[ss.A2_COL]))

        assert all(freq_check)
        assert all(beta_check)
        assert all(a1_check)
        assert all(a2_check)


###########################################

class TestStandardizeAllSumstats:


    _SNPS = ['rs01', 'rs02', 'rs03', 'rs04', 'rs05', 'rs06']
    _BETAS = [1.0, -2.0, 3.0, -4.0, 5.0, 0.0]
    _FREQS = [1.0, 0.5, 0.6, 0.4, 0.1, 0.0]

    _DATA_1 = {
        ss.A1_COL : ['G', 'A', 'C', 'T', 'G', 'A'],
        ss.A2_COL : ['T', 'G', 'A', 'C', 'T', 'G'],
        ss.BETA_COL : _BETAS,
        ss.FREQ_COL : _FREQS
    }

    _DATA_2 = {
        # Both A1 and A2 are complements of DATA_1
        ss.A1_COL : ['C', 'T', 'G', 'A', 'C', 'T'],
        ss.A2_COL : ['A', 'C', 'T', 'G', 'A', 'C'],
        ss.BETA_COL : _BETAS,
        ss.FREQ_COL : _FREQS
    }

    _DATA_3 = {
        ss.A1_COL : ['G', 'A', 'C', 'T', 'G', 'A'],
        ss.A2_COL : ['T', 'C', 'A', 'C', 'T', 'G'],  # Differs in 2nd spot from DATA_1
        ss.BETA_COL : _BETAS,
        ss.FREQ_COL : _FREQS
    }

    _DATA_4 = {
        ss.A1_COL : ['G', 'T', 'G', 'T', 'G', 'A'],  # Differs in 2nd and 3rd spots from DATA_1
        ss.A2_COL : ['T', 'G', 'A', 'C', 'T', 'G'],
        ss.BETA_COL : _BETAS,
        ss.FREQ_COL : _FREQS
    }

    _DATA_5 = {
        # The first is a reference allele flip, and the second is a complement allele flip
        ss.A1_COL : ['T', 'C', 'C', 'T', 'G', 'A'],
        ss.A2_COL : ['G', 'T', 'A', 'C', 'T', 'G'],
        ss.BETA_COL : _BETAS,
        ss.FREQ_COL : _FREQS
    }

    #########
    def test__complement_snps__all_match(self):

        # Tests two populations with major/minor alleles that are complements of each other
        # (no dropping or flipping needs to occur)

        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df2 = pd.DataFrame(TestStandardizeAllSumstats._DATA_2,
                             index=TestStandardizeAllSumstats._SNPS)
        orig_pops = {1 : df1, 2 : df2}

        # Test each population as the reference, including not specifying (should default to 1)
        dfs = [(1, df1), (2, df2), ()]
        for ref_tuple in dfs:
            df1_copy = df1.copy()
            df2_copy = df2.copy()
            pop_dict = {1 : df1_copy, 2 : df2_copy}

            ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                ss.standardize_all_sumstats(pop_dict, ref_tuple)

            # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
            assert ref_popname == ref_tuple[0] if ref_tuple else 1
            assert ref_popname in pop_dict
            assert ref_popname in orig_pops
            ref_df = orig_pops[ref_popname]

            # No drops should be needed
            assert ~any(cumulative_drop_indices)

            # Make sure both pops are accounted for and both indicate no drops
            assert len(drop_dict) == 2
            assert ~any(drop_dict[1])
            assert ~any(drop_dict[2])

            # Make sure both pops are accounted for and both indicate no flips
            assert len(ref_flip_dict) == 2
            assert ~any(ref_flip_dict[1])
            assert ~any(ref_flip_dict[2])

            # Make sure columns still agree (aside from complements)
            assert all((df1_copy[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df1_copy[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))
            assert all((df2_copy[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df2_copy[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))


    #########
    def test__allele_flip_snps__expected_flips(self):

        # Tests two populations with major/minor alleles that are complements of each other
        # (no dropping or flipping needs to occur)

        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df5 = pd.DataFrame(TestStandardizeAllSumstats._DATA_5,
                             index=TestStandardizeAllSumstats._SNPS)
        orig_pops = {1 : df1, 5 : df5}

        # Test each population as the reference, including not specifying (should default to 1)
        dfs = [(1, df1), (5, df5), ()]
        for ref_tuple in dfs:
            df1_copy = df1.copy()
            df5_copy = df5.copy()
            pop_dict = {1 : df1_copy, 5 : df5_copy}

            ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                ss.standardize_all_sumstats(pop_dict, ref_tuple)

            # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
            assert ref_popname == ref_tuple[0] if ref_tuple else 1
            assert ref_popname in pop_dict
            assert ref_popname in orig_pops
            ref_df = orig_pops[ref_popname]

            # No drops should be needed
            assert ~any(cumulative_drop_indices)

            # Make sure both pops are accounted for and both indicate no drops
            assert len(drop_dict) == 2
            assert ~any(drop_dict[1])
            assert ~any(drop_dict[5])

            # Make sure both pops are accounted for and one indicates two flips
            assert len(ref_flip_dict) == 2
            assert (((ref_flip_dict[1].sum() == 0) and (ref_flip_dict[5].sum() == 2)) or
                   ((ref_flip_dict[1].sum() == 2) and (ref_flip_dict[5].sum() == 0)))
            assert (ref_flip_dict[1] | ref_flip_dict[5]).sum() == 2

            # Make sure columns now agree (aside from complements)
            assert all((df1_copy[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df1_copy[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))
            assert all((df5_copy[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df5_copy[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))


    #########
    def test__mismatched_snp__expected_drop(self):

        # Tests two populations with a mismatched major/minor allele (unfixable)
        # so there should be one drop recommended

        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df3 = pd.DataFrame(TestStandardizeAllSumstats._DATA_3,
                             index=TestStandardizeAllSumstats._SNPS)
        orig_pops = {1 : df1, 3 : df3}

        # Test each population as the reference, including not specifying (should default to 1)
        dfs = [(1, df1), (3, df3), ()]
        for ref_tuple in dfs:
            df1_copy = df1.copy()
            df3_copy = df3.copy()
            pop_dict = {1 : df1_copy, 3 : df3_copy}

            ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                ss.standardize_all_sumstats(pop_dict, ref_tuple)

            # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
            assert ref_popname == ref_tuple[0] if ref_tuple else 1
            assert ref_popname in pop_dict
            assert ref_popname in orig_pops
            ref_df = orig_pops[ref_popname]

            # No drops should be needed
            assert cumulative_drop_indices.sum() == 1
            assert cumulative_drop_indices[cumulative_drop_indices.index[1]]

            # Make sure both pops are accounted for and one drop is indicated
            assert len(drop_dict) == 2
            assert ((drop_dict[1].sum() == 0 and drop_dict[3].sum() == 1) or
                    (drop_dict[1].sum() == 1 and drop_dict[3].sum() == 0))

            # Make sure both pops are accounted for and no flips are recorded
            assert len(ref_flip_dict) == 2
            assert ~any(ref_flip_dict[1])
            assert ~any(ref_flip_dict[3])

            # Make sure columns now agree (aside from complements)
            assert all((df1_copy[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df1_copy[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))
            assert all((df3_copy[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df3_copy[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))


    #########
    def test__varying_snps__expected_results(self):
        df1 = pd.DataFrame(TestStandardizeAllSumstats._DATA_1,
                           index=TestStandardizeAllSumstats._SNPS)
        df2 = pd.DataFrame(TestStandardizeAllSumstats._DATA_2,
                           index=TestStandardizeAllSumstats._SNPS)
        df3 = pd.DataFrame(TestStandardizeAllSumstats._DATA_3,
                           index=TestStandardizeAllSumstats._SNPS)
        df4 = pd.DataFrame(TestStandardizeAllSumstats._DATA_4,
                           index=TestStandardizeAllSumstats._SNPS)
        df5 = pd.DataFrame(TestStandardizeAllSumstats._DATA_5,
                             index=TestStandardizeAllSumstats._SNPS)

        ref_df = df1.copy()

        pop_dict = {1 : df1, 2 : df2, 3 : df3, 4 : df4, 5 : df5}

        ref_popname, cumulative_drop_indices, drop_dict, ref_flip_dict = \
                ss.standardize_all_sumstats(pop_dict)

        # In python 3.7+, dictionary keys are ordered (so should default to 1 here)
        assert ref_popname == 1

        # Two drops should be indicated (in the second and third spots)
        assert cumulative_drop_indices.sum() == 2
        assert all(cumulative_drop_indices[cumulative_drop_indices.index[1:3]])

        # Make sure the drop breakdown is correct
        assert len(drop_dict) == 5
        assert ~any(drop_dict[1])
        assert ~any(drop_dict[2])
        assert drop_dict[3].sum() == 1
        assert drop_dict[4].sum() == 2
        assert (drop_dict[3] | drop_dict[4]).sum() == 2
        assert ~any(drop_dict[5])

        # Make sure reference allele flips were recorded correctly
        assert len(ref_flip_dict) == 5
        assert ~any(ref_flip_dict[1])
        assert ~any(ref_flip_dict[2])
        assert ~any(ref_flip_dict[3])
        assert ~any(ref_flip_dict[4])
        assert ref_flip_dict[5].sum() == 2

        for df in pop_dict.values():
            assert all((df[ss.A1_COL] == ref_df[ss.A1_COL]) |
                       (df[ss.A1_COL].replace(ss.COMPLEMENT) == ref_df[ss.A1_COL]))

# TODO(jonbjala) Test qc_sumstats() and process_sumstats()

###########################################

class TestCreateFilters:

    #########
    @pytest.mark.parametrize("min_freq, max_freq",
    [
            (0.0, 1.0),
            (-1.0, 2.0),
            (0.5, 0.6)
    ])
    def test__create_freq_filter__expected_results(self, min_freq, max_freq):
        tol = 0.001
        filt_func = ss.create_freq_filter(min_freq, max_freq)
        f_list = [min_freq - tol, min_freq, min_freq + tol, 0.5 * (min_freq + max_freq),
                  max_freq - tol, max_freq, max_freq + tol]
        expected = [True, False, False, False, False, False, True]

        df = pd.DataFrame(data={ss.FREQ_COL : f_list, 'dummy' : [1.0 for f in f_list]})
        assert all(filt_func(df) == expected)

    #########
    @pytest.mark.parametrize("chr_list",
    [
            ['1', '2', '3', 'X', 'Y'],
            [],
            ['5', '10']
    ])
    def test__create_chr_filter__expected_results(self, chr_list):
        filt_func = ss.create_chr_filter(chr_list)
        extra_c = ['Dummy1', 'Dummy2', '-100000', '99999']
        c_list = chr_list + extra_c
        expected = [False] * len(chr_list) + [True] * len(extra_c)

        df = pd.DataFrame(data={ss.CHR_COL : c_list, 'dummy' : [1.0 for f in c_list]})
        assert all(filt_func(df) == expected)