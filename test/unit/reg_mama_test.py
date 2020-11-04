"""
Unit tests for reg_mama.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import numpy as np
import pytest

import mama2.reg_mama as reg_mama


###########################################

class TestFixedOptionHelper:

    SIZE_LIST = [1, 2, 3, 4]

    #########
    @pytest.mark.parametrize("size", SIZE_LIST)
    def test__all_free__return_expected(self, size):

        result1 = reg_mama.fixed_option_helper(size, reg_mama.MAMA_REG_OPT_ALL_FREE)
        result2 = reg_mama.fixed_option_helper(size)

        nan_result1 = np.isnan(result1)
        nan_result2 = np.isnan(result2)

        assert np.all(nan_result1)
        assert np.all(nan_result2)

    #########
    @pytest.mark.parametrize("size", SIZE_LIST)
    def test__all_zero__return_expected(self, size):

        result = reg_mama.fixed_option_helper(size, reg_mama.MAMA_REG_OPT_ALL_ZERO)

        assert np.all(np.where(result == 0.0, True, False))

    #########
    @pytest.mark.parametrize("size", SIZE_LIST)
    def test__offdiag_zero__return_expected(self, size):

        result = reg_mama.fixed_option_helper(size, reg_mama.MAMA_REG_OPT_OFFDIAG_ZERO)
        nan_result = np.isnan(result)
        assert np.all(np.diag(nan_result))
        assert np.all(nan_result.sum(axis=0) == 1)
        assert np.all(nan_result.sum(axis=1) == 1)

    #########
    @pytest.mark.parametrize("size", SIZE_LIST)
    def test__identity__return_expected(self, size):

        result = reg_mama.fixed_option_helper(size, reg_mama.MAMA_REG_OPT_IDENT)

        assert np.all(np.diag(result) == 1.0)
        assert np.all(np.where(result == 1.0, True, False).sum(axis=0) == 1)
        assert np.all(np.where(result == 1.0, True, False).sum(axis=1) == 1)

    #########
    @pytest.mark.parametrize("size", SIZE_LIST)
    def test__valid_matrix_input__return_expected(self, size):

        M = np.random.rand(size, size)
        result = reg_mama.fixed_option_helper(size, M)

        assert np.array_equal(result, M)

    #########
    @pytest.mark.parametrize("size", SIZE_LIST)
    def test__invalid_matrix_input__return_expected(self, size):

        M = np.random.rand(size + 1, size + 1)
        with pytest.raises(RuntimeError) as ex_info:
            reg_mama.fixed_option_helper(size, M)

        assert str(size) in str(ex_info.value)
        assert str(size + 1) in str(ex_info.value)

    #########
    def test__invalid_opt_str_type__return_expected(self):

        val = 1.5

        with pytest.raises(RuntimeError) as ex_info:
            reg_mama.fixed_option_helper(5, float(val))

        assert "float" in str(ex_info.value)
        assert str(val) in str(ex_info.value)

    #########
    def test__invalid_opt_str_value__return_expected(self):

        val = "INVALID_OPTION_ABC_XYZ"

        with pytest.raises(RuntimeError) as ex_info:
            reg_mama.fixed_option_helper(5, val)

        assert "str" in str(ex_info.value)
        assert str(val) in str(ex_info.value)
