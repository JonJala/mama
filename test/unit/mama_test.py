"""
Unit tests for mama.py.  This should be run via pytest.
"""

import os
import sys
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
test_directory = os.path.abspath(os.path.join(main_directory, 'test'))
data_directory = os.path.abspath(os.path.join(test_directory, 'data'))
sys.path.append(main_directory)

import tempfile

import pytest

import mama.mama as mama


@pytest.fixture(scope="module")
def temp_test_dir():
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as t:
        yield t


###########################################

class TestSSInputTuple:

    #########
    def test__happy_path__expected_results(self, temp_test_dir):

        ancestry = "dummy_anc"
        phenotype = "dummy_phen"
        filename = os.path.join(temp_test_dir, "test_ssinputtuple_file.txt")
        with open(filename, 'w'):
            pass

        input_string = ",".join([filename, ancestry, phenotype])
        expected = (filename, ancestry, phenotype)
        assert mama.ss_input_tuple(input_string) == expected

    #########
    def test__too_few_components__throw_error(self, temp_test_dir):

        ancestry = "dummy_anc"
        filename = os.path.join(temp_test_dir, "test_ssinputtuple_file.txt")
        with open(filename, 'w'):
            pass

        input_string = ",".join([filename, ancestry])

        with pytest.raises(RuntimeError):
            mama.ss_input_tuple(input_string)

    #########
    def test__file_nonexistent__throw_error(self, temp_test_dir):

        ancestry = "dummy_anc"
        phenotype = "dummy_phen"
        filename = os.path.join(temp_test_dir, "test_missing_file.txt")

        input_string = ",".join([filename, ancestry, phenotype])

        with pytest.raises(FileNotFoundError):
            mama.ss_input_tuple(input_string)


###########################################

class TestCorrCoef:

    @pytest.mark.parametrize("invalid_ld_corr_scale_factor",
                             [os.open, None, [], (), {}])
    def test__invalid_reg_coef_ld_scale_factor_type___throws_type_error(self,
        invalid_ld_corr_scale_factor):

        with pytest.raises(TypeError) as ex_info:
            mama.corr_coef(invalid_ld_corr_scale_factor)

    @pytest.mark.parametrize("invalid_ld_corr_scale_factor",
                             ["abc", "", "[]", "-1.1", "1.5", "3", "[-0.2, 0.2]"])
    def test__invalid_reg_coef_ld_scale_factor_value___throws_value_error(self,
        invalid_ld_corr_scale_factor):

        with pytest.raises(ValueError) as ex_info:
            mama.corr_coef(invalid_ld_corr_scale_factor)
        assert str(invalid_ld_corr_scale_factor) in str(ex_info.value)

    #########


# TODO(jonbjala) Test input_file(), output_prefix(), reg_ex(), input_np_matrix(), and glob_path()
# TODO(jonbjala) Test to_flag(), to_arg(), format_terminal_call(), and get_user_inputs()
