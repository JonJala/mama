"""
Unit tests for ld.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import numpy as np
import pytest

import mama.ld as ld


###########################################

def convert_Rfull_to_Rabridged(R_full, extents):
    M = R_full.shape[0]
    max_extent = max(extents)
    R_abridged = np.zeros((max_extent, M))
    for i in range(M):
        R_abridged[0:extents[i], i] = R_full[i:i+extents[i], i]
    return R_abridged


# Also replaces NaN
def demeaned_G(G, standardize=False, replace_nan=True):
    G_demeaned = G - np.nanmean(G, axis=1)[:, np.newaxis]
    if standardize:
        G_demeaned /= np.nanstd(G_demeaned, axis=1)[:, np.newaxis]
    if replace_nan:
        np.nan_to_num(G_demeaned, copy=False)
    return G_demeaned


SEED_LIST = [1, 10, 22, 38, 777, 1001, 2036, 4096]

class TestCalculateR:

    G_LIST = [
        np.array([[0.0, 1.0, 2.0, 2.0],
                  [1.0, 0.0, 0.0, 2.0],
                  [2.0, 1.0, 0.0, 0.0],
                  [1.0, 0.0, 1.0, 0.0],
                  [2.0, 2.0, 1.0, 1.0]]),
        np.array([[1.0, 2.0],
                  [0.0, 1.0],
                  [0.0, 2.0]]),
        np.array([[0.0, 1.0],
                  [2.0, 1.0]]),
        np.array([[2.0]]),
        np.array([[0.0, 2.0, 2.0],
                  [1.0, 1.0, 2.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [2.0, 0.0, 0.0],
                  [2.0, 0.0, 0.0],
                  [1.0, 0.0, 1.0]]),
        np.array([[1.0, 1.0],
                  [1.0, 0.0],
                  [2.0, 1.0]])
        ]

    STANDARDIZE = [True, False]


    # M x N
    G_SHAPE_LIST = [(1, 1000), (10, 1000), (100, 1000), (1000, 1000), (475, 33)]
    G_SHAPE_LIST = [(10, 1000)]
   

    #########
    @pytest.mark.parametrize("G_orig", G_LIST)
    def test__precannedsmallG__maxextents__return_expected(self, G_orig):
        G = demeaned_G(G_orig)
        M, N = G.shape
        extents = np.array(list(range(M, 0, -1)))

        R_actual = ld.calculate_R_without_nan(G, extents)

        R_full_expected = np.cov(G, ddof=0).reshape((M, M))      
        R_expected = convert_Rfull_to_Rabridged(R_full_expected, extents)

        assert np.allclose(R_actual, R_expected)


    #########
    @pytest.mark.parametrize("G_orig", G_LIST)
    @pytest.mark.parametrize("r_seed", SEED_LIST)
    def test__precannedsmallG__variousextents__return_expected(self, G_orig, r_seed):
        G = demeaned_G(G_orig) 
        M, N = G.shape
        rng = np.random.default_rng(r_seed)
        extents = [rng.integers(low=1, high=M-i, endpoint=True) for i in range(M)]

        R_actual = ld.calculate_R_without_nan(G, extents)

        R_full_expected = np.cov(G, ddof=0).reshape((M, M))      
        R_expected = convert_Rfull_to_Rabridged(R_full_expected, extents)

        assert np.allclose(R_actual, R_expected)


    #########
    @pytest.mark.parametrize("G_orig", G_LIST)
    def test__precannedsmallG__varioussteps__return_sameresult(self, G_orig):
        G = demeaned_G(G_orig)
        M, N = G.shape
        extents = np.array(list(range(M, 0, -1)))

        R_default = ld.calculate_R_without_nan(G, extents)

        R_1 = ld.calculate_R_without_nan(G, extents, step_size=1)
        R_2 = ld.calculate_R_without_nan(G, extents, step_size=2)
        R_half_minus = ld.calculate_R_without_nan(G, extents,
                                                  step_size=max(M // 2 - 1, 1))
        R_half = ld.calculate_R_without_nan(G, extents,
                                            step_size=max(M // 2, 1))
        R_half_plus = ld.calculate_R_without_nan(G, extents,
                                                 step_size=M // 2 + 1)
        R_max_minus = ld.calculate_R_without_nan(G, extents,
                                                  step_size=max(M - 1, 1))


        assert np.allclose(R_default, R_1)
        assert np.allclose(R_default, R_2)
        assert np.allclose(R_default, R_half_minus)
        assert np.allclose(R_default, R_half)
        assert np.allclose(R_default, R_half_plus)
        assert np.allclose(R_default, R_max_minus)



    #########
    @pytest.mark.parametrize("G_shape", G_SHAPE_LIST)
    @pytest.mark.parametrize("standardize", STANDARDIZE)
    @pytest.mark.parametrize("r_seed", SEED_LIST)
    def test__uniformrandomG__noextents__return_expected(self, G_shape, standardize, r_seed):
        rng = np.random.default_rng(r_seed)
        G_orig = rng.integers(low=0, high=2, size=G_shape, endpoint=True).astype(np.float32)
        G = demeaned_G(G_orig, standardize)
        M, N = G.shape
        extents = np.array(list(range(M, 0, -1)))

        R_actual = ld.calculate_R_without_nan(G, extents)

        R_full_expected = np.cov(G, ddof=0).reshape((M, M))      
        R_expected = convert_Rfull_to_Rabridged(R_full_expected, extents)

        assert np.allclose(R_actual, R_expected, rtol=1e-04, atol=1e-04)

    #########
    @pytest.mark.parametrize("G_shape", G_SHAPE_LIST)
    @pytest.mark.parametrize("standardize", STANDARDIZE)
    @pytest.mark.parametrize("r_seed", SEED_LIST)
    def test__uniformrandomG__variousextents__return_expected(self, G_shape, standardize, r_seed):
        rng = np.random.default_rng(r_seed)
        G_orig = rng.integers(low=0, high=2, size=G_shape, endpoint=True).astype(np.float32)
        G = demeaned_G(G_orig, standardize)
        M, N = G.shape
        extents = [rng.integers(low=1, high=M-i, endpoint=True) for i in range(M)]

        R_actual = ld.calculate_R_without_nan(G, extents)

        R_full_expected = np.cov(G, ddof=0).reshape((M, M))      
        R_expected = convert_Rfull_to_Rabridged(R_full_expected, extents)

        assert np.allclose(R_actual, R_expected, rtol=1e-04, atol=1e-04)

    #########
    @pytest.mark.parametrize("G_shape", G_SHAPE_LIST)
    @pytest.mark.parametrize("standardize", STANDARDIZE)
    @pytest.mark.parametrize("r_seed", SEED_LIST)
    def test__uniformrandomG__varioussteps__return_expected(self, G_shape, standardize, r_seed):
        rng = np.random.default_rng(r_seed)
        G_orig = rng.integers(low=0, high=2, size=G_shape, endpoint=True).astype(np.float32)
        G = demeaned_G(G_orig, standardize)
        M, N = G.shape
        extents = np.array(list(range(M, 0, -1)))

        R_default = ld.calculate_R_without_nan(G, extents)

        R_1 = ld.calculate_R_without_nan(G, extents, step_size=1)
        R_2 = ld.calculate_R_without_nan(G, extents, step_size=2)
        R_half_minus = ld.calculate_R_without_nan(G, extents,
                                                  step_size=max(M // 2 - 1, 1))
        R_half = ld.calculate_R_without_nan(G, extents,
                                            step_size=max(M // 2, 1))
        R_half_plus = ld.calculate_R_without_nan(G, extents,
                                                 step_size=M // 2 + 1)
        R_max_minus = ld.calculate_R_without_nan(G, extents,
                                                  step_size=max(M - 1, 1))


        assert np.allclose(R_default, R_1, rtol=1e-04, atol=1e-04)
        assert np.allclose(R_default, R_2, rtol=1e-04, atol=1e-04)
        assert np.allclose(R_default, R_half_minus, rtol=1e-04, atol=1e-04)
        assert np.allclose(R_default, R_half, rtol=1e-04, atol=1e-04)
        assert np.allclose(R_default, R_half_plus, rtol=1e-04, atol=1e-04)
        assert np.allclose(R_default, R_max_minus, rtol=1e-04, atol=1e-04)


    #########
    @pytest.mark.skip(reason="Need to see if this test is feasible to write")
    def test__toolargeR__return_mmap(self):
        # TODO(jonbjala) Need to try to write this
        assert True


M_LIST = [1, 10, 100, 1000]
GEO_P_LIST = [0.1, 0.3, 0.5, 0.75, 0.9]

class TestCalculateLowerExtents:

    #########
    @pytest.mark.parametrize("M", M_LIST)
    def test__allextentsone__return_expected(self, M):
        window_size = 10
        values = [i * (window_size + 1) for i in range(M)]

        actual_lower_extents = ld.calculate_lower_extents(values, window_size)

        assert all(actual_lower_extents == 1.0)


    #########
    @pytest.mark.parametrize("M", M_LIST)
    def test__allextentsmax__return_expected(self, M):
        snp_dist = 10
        window_size = 2 * M * snp_dist
        values = [i * snp_dist for i in range(M)]

        actual_lower_extents = ld.calculate_lower_extents(values, window_size)
        expected_lower_extents = np.arange(M, 0, -1)

        assert all(np.equal(expected_lower_extents, actual_lower_extents))

    #########    
    def test__increasingwindow__return_expected(self):
        M = 100
        snp_dist = 10
        values = [i * snp_dist for i in range(M)]
        decreasing_range = np.arange(M, 0, -1)
        for multiplier in range(M):
            window_size = multiplier * snp_dist
            actual_lower_extents = ld.calculate_lower_extents(values, window_size)
            expected_lower_extents = np.minimum(decreasing_range, multiplier + 1)
            assert all(np.equal(expected_lower_extents, actual_lower_extents))


    #########
    # TODO(jonbjala) Come back to these tests
    # @pytest.mark.parametrize("M", M_LIST)
    # @pytest.mark.parametrize("r_seed", SEED_LIST)
    # @pytest.mark.parametrize("geo_p", GEO_P_LIST)
    # def test__randomextents__return_expected(self, M, r_seed, geo_p):
    #     rng = np.random.default_rng(r_seed)

    #     # TODO(jonbjala) Can maybe do this more cleanly
    #     arange_M = np.arange(M, dtype=float)
    #     initial_endpts = rng.geometric(geo_p, size=M) + arange_M - 1.0
    #     endpts = np.minimum(M-1, np.maximum.accumulate(initial_endpts), dtype=float)
    #     extents = endpts - arange_M + 1.0
        
    #     window_size = np.lcm.reduce(extents)
    #     distances = np.zeros(M)
    #     curpt = 0
    #     for i in range(M):
    #         if distances[i] == 0:
    #         else:


    #         # if endpts[i] > most_recent_endpt:
    #         #     for j in range(most_recent_endpt + 2, endpts[i] + 2):





    #     values = [i * snp_dist for i in range(M)]

    #     actual_lower_extents = ld.calculate_lower_extents(values, window_size)
    #     expected_lower_extents = np.arange(M, 0, -1)

    #     assert all(np.equal(expected_lower_extents, actual_lower_extents))


class TestCalculateLdScores:


    #########    
    def test__twopop__precanned__return_expected(self):
        r_1 = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        r_2 = np.array([[3.0, 1.0, 3.0], [1.0, 3.0, 1.0], [3.0, 1.0, 3.0]])
        
        expected_ld_scores = np.array([14.0/3.0, 19.0/12.0, 16.0 / 9.0])
        r_prod = np.multiply(r_1, r_2)
        assert np.array_equal(expected_ld_scores, np.sum(r_prod, axis=0) / np.diag(r_prod))

        extents = np.array([3, 2, 1])
        banded_r_1 = convert_Rfull_to_Rabridged(r_1, extents)
        banded_r_2 = convert_Rfull_to_Rabridged(r_2, extents)


        assert np.array_equal(banded_r_1, np.array([[1.0, 4.0, 6.0], [2.0, 5.0, 0.0], [3.0, 0.0, 0.0]]))
        assert np.array_equal(banded_r_2, np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 0.0], [3.0, 0.0, 0.0]]))

        actual_ld_scores = ld.calculate_ld_scores((banded_r_1, banded_r_2))

        assert all(np.equal(expected_ld_scores, actual_ld_scores))


    #########
    @pytest.mark.parametrize("M", M_LIST)
    @pytest.mark.parametrize("r_seed", SEED_LIST)
    def test__twopop__random__maxextents__return_expected(self, M, r_seed):
        rng = np.random.default_rng(r_seed)
        r_1_pre = rng.random(size=(M,M))
        r_2_pre = rng.random(size=(M,M))

        r_1 = 0.5 * (r_1_pre + r_1_pre.T)
        r_2 = 0.5 * (r_2_pre + r_2_pre.T)  
        r_prod = np.multiply(r_1, r_2)

        expected_ld_scores = np.sum(r_prod, axis=0) / np.diag(r_prod)

        extents = np.array(list(range(M, 0, -1)))
        banded_r = (convert_Rfull_to_Rabridged(r_1, extents),
                    convert_Rfull_to_Rabridged(r_2, extents))

        actual_ld_scores = ld.calculate_ld_scores(banded_r)

        assert np.allclose(expected_ld_scores, actual_ld_scores)


    #########    
    def test__onepop__precanned__return_expected(self):        
        r = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
        r2 = np.square(r)
        expected_correction = np.array([-3.0, -1.0/16.0, -1.0/9.0])
        expected_ld_scores = np.sum(r2, axis=0) / np.diag(r2) - expected_correction


        extents = np.array([3, 2, 1])
        banded_r = convert_Rfull_to_Rabridged(r, extents)

        assert np.array_equal(banded_r, np.array([[1.0, 4.0, 6.0], [2.0, 5.0, 0.0], [3.0, 0.0, 0.0]]))
        
        actual_ld_scores = ld.calculate_ld_scores((banded_r,), N=3.0)

        assert np.array_equal(expected_ld_scores, actual_ld_scores)


    #########
    @pytest.mark.parametrize("M", M_LIST)
    @pytest.mark.parametrize("r_seed", SEED_LIST)
    def test__onepop__largeN__ldscores_close(self, M, r_seed):
        rng = np.random.default_rng(r_seed)
        r_pre = rng.random(size=(M,M))
        
        r = 0.5 * (r_pre + r_pre.T)
        extents = np.array(list(range(M, 0, -1)))        
        banded_r = convert_Rfull_to_Rabridged(r, extents)

        N1 = 10**8
        N2 = N1 + 1
        actual_ld_scores_N1 = ld.calculate_ld_scores((banded_r,), N1)
        actual_ld_scores_N2 = ld.calculate_ld_scores((banded_r,), N2)

        assert np.allclose(actual_ld_scores_N1, actual_ld_scores_N2)


    #########
    @pytest.mark.parametrize("M", M_LIST)
    def test__onepop__allones_largeN__return_expected(self, M):
        r = np.ones((M,M))
        extents = np.array(list(range(M, 0, -1)))        
        banded_r = convert_Rfull_to_Rabridged(r, extents)

        approx_expected_ld_scores = np.full(M, M)
        N = 10**8
        actual_ld_scores = ld.calculate_ld_scores((banded_r,), N)
        
        assert np.allclose(approx_expected_ld_scores, actual_ld_scores)


    #########
    @pytest.mark.parametrize("M", M_LIST)
    def test__onepop__allones_smallN__return_expected(self, M):
        r = np.ones((M,M))
        extents = np.array(list(range(M, 0, -1)))        
        banded_r = convert_Rfull_to_Rabridged(r, extents)

        actual_ld_scores_3 = ld.calculate_ld_scores((banded_r,), 3)
        actual_ld_scores_4 = ld.calculate_ld_scores((banded_r,), 4)

        correction_terms = -2.0 * (actual_ld_scores_3 - actual_ld_scores_4)

        expected_ld_scores_5 = actual_ld_scores_3 + correction_terms - (3.0 * correction_terms)
        actual_ld_scores_5 = ld.calculate_ld_scores((banded_r,), 5)
        
        assert np.allclose(expected_ld_scores_5, actual_ld_scores_5)