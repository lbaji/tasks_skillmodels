import numpy as np
import numba
import cmath
import math

sigma_points = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10]])
included_positions = np.array([0, 1, 2])
coeffs = np.array([0.4, 0.3, 0.3, 0.5])

@numba.jit(nopython=True)
def fast_log_ces_janos(sigma_points, coeffs, included_positions):
    """Calculates the next period's predicted state of the latent factors' states.  
    
    Args:
        * sigma_points: 2d array of sigma_points or states being transformed
        * coeffs: 1d array with coefficients specific to this transition function
          If the coeffs include an intercept term (e.g. the log of a TFP term),
          this has to be the FIRST or LAST element of coeffs.
        * included_positions: 1d array with the positions of the factors that are
          included in the transition equation 

    Returns
        * 1d array
    
    """
    nres = sigma_points.shape[0]
    phi = coeffs[-1]
    
    result = np.empty(nres)
    for i in range(nres):
        res = 0
        for pos in included_positions:
            res += coeffs[pos] * math.exp(sigma_points[i, pos] * phi)
            #res += coeffs[pos] * np.exp(sigma_points[i, pos] * phi)
        res = math.log(res) / phi
        #res = np.log(res) / phi
        result[i] = res

    return result
    
result = fast_log_ces_janos(sigma_points, coeffs, included_positions)
print(result)
