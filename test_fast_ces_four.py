# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 00:02:39 2019

@author: Laura
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from fast_log_CES_function import fast_log_ces


@pytest.fixture
def setup_log_ces():
    out = {} 
    out['sigma_points'] = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    out['coeffs'] = np.array([0.25, 0.25, 0.25, 0.25, 0.5])
    out['included_positions'] = np.array([0, 1, 2, 3])

    return out


@pytest.fixture
def expected_log_ces():
    out = {}
    out['result'] = np.array([2.80208862])
    
    return out
 

def test_log_ces(setup_log_ces, expected_log_ces):
    calc_log_ces = fast_log_ces(
            setup_log_ces['sigma_points'],
            setup_log_ces['coeffs'],
            setup_log_ces['included_positions']
            )
    assert_array_almost_equal(calc_log_ces, expected_log_ces['result'])