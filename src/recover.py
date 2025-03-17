import numpy as np

def inverse_equations(R_obs, M_obs, V_obs):
    """Recover parameters using inverse EZ equations."""
    if not (0 < R_obs < 1):  # Prevent log(0) errors
        raise ValueError("R_obs must be between 0 and 1 (exclusive).")

    L = np.log(R_obs / (1 - R_obs))  # Log odds of accuracy
    
    if V_obs <= 0:
        raise ValueError("V_obs must be positive.")

    try:
        # Compute v_est using the corrected formula
        term = (L * (R_obs**2 * L - R_obs * L + R_obs - 0.5)) / V_obs
        v_est = np.sign(R_obs - 0.5) * np.sqrt(max(0, term))  # Ensure positivity

        # Compute a_est
        a_est = abs(L / max(1e-6, v_est))  # Ensure no division by zero

        # Compute t_est
        exp_term = np.exp(-v_est * a_est)
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - exp_term) / (1 + exp_term))

        # Debugging Output
        print(f"DEBUG: R_obs={R_obs}, M_obs={M_obs}, V_obs={V_obs}")
        print(f"DEBUG: L={L}, v_est={v_est}, a_est={a_est}, t_est={t_est}")

        return a_est, v_est, t_est
    except ValueError:
        return np.nan, np.nan, np.nan  # Handle invalid calculations gracefully
