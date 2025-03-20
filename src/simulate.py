import numpy as np
import scipy.stats as stats
import sys

def generate_true_parameters(): # used ChatGPT to help outline this file 
    a = np.random.uniform(0.5, 2.0)
    v = np.random.uniform(0.5, 2.0)
    t = np.random.uniform(0.1, 0.5)
    return a, v, t

def forward_equations(a, v, t):
    y = np.exp(-a * v)
    R_pred = 1 / (1 + y)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v ** 3)) * ((1 - 2 * a * v * y - y**2) / (1 + y)**2)
    return R_pred, M_pred, V_pred

def sample_observed_statistics(R_pred, M_pred, V_pred, N):
    R_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))
    return R_obs, M_obs, V_obs

def simulate_and_print(N, iterations=1000):
    for _ in range(iterations):
        a, v, t = generate_true_parameters()
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        R_obs, M_obs, V_obs = sample_observed_statistics(R_pred, M_pred, V_pred, N)
        print(f"{a},{v},{t},{R_obs},{M_obs},{V_obs}") 
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 src/simulate.py <N> <iterations>")
        sys.exit(1)

    N = int(sys.argv[1])
    iterations = int(sys.argv[2])
    simulate_and_print(N, iterations)
