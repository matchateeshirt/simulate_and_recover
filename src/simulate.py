import numpy as np
import scipy.stats as stats

def generate_true_parameters():
    """Randomly generates true parameters for the diffusion model."""
    a = np.random.uniform(0.5, 2.0)  # Boundary separation
    v = np.random.uniform(0.5, 2.0)  # Drift rate
    t = np.random.uniform(0.1, 0.5)  # Non-decision time
    return a, v, t

def forward_equations(a, v, t):
    """Compute predicted summary statistics using EZ diffusion forward equations."""
    y = np.exp(-a * v)
    R_pred = 1 / (1 + y)
    M_pred = t + (a / (2 * v)) * ((1 - y) / (1 + y))
    V_pred = (a / (2 * v ** 3)) * ((1 - 2 * a * v * y - y ** 2) / (y + 1) ** 2)
    return R_pred, M_pred, V_pred

def sample_observed_statistics(R_pred, M_pred, V_pred, N):
    """Sample observed statistics using the given distributions."""
    R_obs = np.random.binomial(N, R_pred) / N
    M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))
    return R_obs, M_obs, V_obs

def inverse_equations(R_obs, M_obs, V_obs):
    """Recover parameters using inverse EZ equations."""
    L = np.log(R_obs / (1 - R_obs))
    v_est = np.sign(R_obs - 0.5) * 4 * np.sqrt(L * (R_obs ** 2 * L - R_obs * L + R_obs - 0.5) / V_obs)
    a_est = L / v_est
    t_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
    return a_est, v_est, t_est

def simulate_and_recover(N, iterations=1000):
    """Runs the full simulate-and-recover process."""
    biases = []
    squared_errors = []
    for _ in range(iterations):
        a, v, t = generate_true_parameters()
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        R_obs, M_obs, V_obs = sample_observed_statistics(R_pred, M_pred, V_pred, N)
        a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
        bias = np.array([a - a_est, v - v_est, t - t_est])
        biases.append(bias)
        squared_errors.append(bias ** 2)
    return np.mean(biases, axis=0), np.mean(squared_errors, axis=0)

def run_full_experiment():
    """Runs the experiment for N = 10, 40, 4000 and saves results."""
    results = []
    for N in [10, 40, 4000]:
        bias, sq_error = simulate_and_recover(N)
        results.append((N, bias, sq_error))
        print(f"N = {N}: Bias = {bias}, Squared Error = {sq_error}")
    return results

if __name__ == "__main__":
    run_full_experiment()
