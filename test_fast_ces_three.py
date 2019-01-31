import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from fast_log_CES_function import fast_log_ces


@pytest.fixture
def setup_log_ces():
    out = {} 
    out['sigma_points'] = np.array([[1, 2, 3, 4, 5]])
    out['coeffs'] = np.array([0.4, 0.3, 0.3, 0.5])
    out['included_positions'] = np.array([0, 1, 2])

    return out


@pytest.fixture
def expected_log_ces():
    out = {}
    out['result'] = np.array([2.07310478])
    
    return out
 

def test_log_ces(setup_log_ces, expected_log_ces):
    calc_log_ces = fast_log_ces(
            setup_log_ces['sigma_points'],
            setup_log_ces['coeffs'],
            setup_log_ces['included_positions']
            )
    assert_array_almost_equal(calc_log_ces, expected_log_ces['result'])
