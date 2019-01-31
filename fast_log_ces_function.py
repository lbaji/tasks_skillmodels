# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 14:29:11 2019

@author: Laura
"""
import numpy as np
import numba

sigma_points = np.array([[1, 2, 3, 4, 5], 
                         [6, 7, 8, 9, 10]])
included_positions = np.array([0, 1, 2])
coeffs = np.array([0.4, 0.3, 0.3, 0.5])

def fast_log_ces(sigma_points, coeffs, included_positions):
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
    nfac = sigma_points.shape[1]
    nres = sigma_points.shape[0]
    phi = coeffs[-1]
    gammas = np.zeros(nfac)
    for p, pos in enumerate(included_positions):
        gammas[pos] = coeffs[p]
    
    if included_positions.shape[0] != sigma_points.shape[1]:
        sigma_points = np.delete(sigma_points, np.where(gammas == 0), 1)
        gammas = gammas[gammas != 0]
    exponents = sigma_points * phi
    x = np.exp(exponents)
    #x = multiplication(x, gammas, nres)
    x = np.dot(x, gammas)
    scaling_factor = 1 / phi
    result = scaling_factor * np.log(x)

    return result
    
@numba.jit(nopython=True)
def multiplication(x, gammas, nres):
    mult = np.zeros(nres)
    for i in range(nres):
        mult[i]= np.sum(x[i, :]*gammas)
    return mult

result = fast_log_ces(sigma_points, coeffs, included_positions)
print(result)
