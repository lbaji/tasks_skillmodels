import numpy as np
from time import time
from fast_log_CES_function_Janos import fast_log_ces_janos
from fast_log_CES_function import fast_log_ces
from transition_functions import log_ces

# load and prepare data
sigma_points_1 = np.random.rand(10000, 5)
included_positions_1 = np.array([0, 1, 2])
coeffs_1 = np.array([0.4, 0.3, 0.3, 0.5])

sigma_points_2 = np.random.rand(30000, 5)
included_positions_2 = np.array([0, 1, 2, 3, 4])
coeffs_2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.5])

sigma_points_3 = np.random.rand(40000, 8)
included_positions_3 = np.array([0, 1, 2, 3])
coeffs_3 = np.array([0.25, 0.25, 0.25, 0.25, 0.5])

# time the function with (10000, 5)
runtimes_ces_1 = []
runtimes_fast_ces_1 = []
runtimes_janos_1 = []

for i in range(2):
    start = time()
    log_ces(
            sigma_points = sigma_points_1, 
            coeffs = coeffs_1, 
            included_positions = included_positions_1
            )
    stop = time()
    runtimes_ces_1.append(stop - start)
    
for l in range(10):
    start = time()
    fast_log_ces(
            sigma_points = sigma_points_1, 
            coeffs = coeffs_1, 
            included_positions = included_positions_1
            )
    stop = time()
    runtimes_fast_ces_1.append(stop - start)

for j in range(10):
    start = time()
    fast_log_ces_janos(
            sigma_points = sigma_points_1, 
            coeffs = coeffs_1, 
            included_positions = included_positions_1
            )
    stop = time()
    runtimes_janos_1.append(stop - start)
    
# time the function with (30000, 5)
runtimes_ces_2 = []
runtimes_fast_ces_2 = []
runtimes_janos_2 = []

for i in range(2):
    start = time()
    log_ces(
            sigma_points = sigma_points_2, 
            coeffs = coeffs_2, 
            included_positions = included_positions_2
            )
    stop = time()
    runtimes_ces_2.append(stop - start)
    
for l in range(10):
    start = time()
    fast_log_ces(
            sigma_points = sigma_points_2, 
            coeffs = coeffs_2, 
            included_positions = included_positions_2
            )
    stop = time()
    runtimes_fast_ces_2.append(stop - start)
    
for j in range(10):
    start = time()
    fast_log_ces_janos(
            sigma_points = sigma_points_2, 
            coeffs = coeffs_2, 
            included_positions = included_positions_2
            )
    stop = time()
    runtimes_janos_2.append(stop - start)
    
# time the function with (40000, 4)
runtimes_ces_3 = []
runtimes_fast_ces_3 = []
runtimes_janos_3 = []

for i in range(2):
    start = time()
    log_ces(
            sigma_points = sigma_points_3, 
            coeffs = coeffs_3, 
            included_positions = included_positions_3
            )
    stop = time()
    runtimes_ces_3.append(stop - start)
    
for l in range(10):
    start = time()
    fast_log_ces(
            sigma_points = sigma_points_3, 
            coeffs = coeffs_3, 
            included_positions = included_positions_3
            )
    stop = time()
    runtimes_fast_ces_3.append(stop - start)
    
for j in range(10):
    start = time()
    fast_log_ces_janos(
            sigma_points = sigma_points_3, 
            coeffs = coeffs_3, 
            included_positions = included_positions_3
            )
    stop = time()
    runtimes_janos_3.append(stop - start)

print("baseline_ces with (10000, 3) took {} seconds.".format(
        np.mean(runtimes_ces_1)))
print("fast_log_ces with (10000, 3) took {} seconds.".format(
        np.mean(runtimes_fast_ces_1[2:])))
print("fast_log_ces_janos with (10000, 3) took {} seconds.".format(
        np.mean(runtimes_janos_1[2:])))

print("baseline_ces with (30000, 5) took {} seconds.".format(
        np.mean(runtimes_ces_2)))
print("fast_log_ces with (30000, 5) took {} seconds.".format(
        np.mean(runtimes_fast_ces_2[2:])))
print("fast_log_ces_janos with (30000, 5) took {} seconds.".format(
        np.mean(runtimes_janos_2[2:])))

print("baseline_ces with (40000, 4) took {} seconds.".format(
        np.mean(runtimes_ces_3)))
print("fast_log_ces with (40000, 4) took {} seconds.".format(
        np.mean(runtimes_fast_ces_3[2:])))
print("fast_log_ces_janos with (40000, 4) took {} seconds.".format(
        np.mean(runtimes_janos_3[2:])))
