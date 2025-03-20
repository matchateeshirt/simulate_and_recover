import numpy as np
import sys

def inverse_equations(R_obs, M_obs, V_obs): # used ChatGPT to format + debug 
    """Recover parameters using inverse EZ equations."""
    if not (0 < R_obs < 1):  
        print(f"Warning: R_obs out of bounds: {R_obs}")  
        return np.nan, np.nan, np.nan  

    L = np.log(R_obs / (1 - R_obs))  

    if V_obs <= 0:
        print(f"Warning: V_obs is non-positive: {V_obs}")  
        return np.nan, np.nan, np.nan  

    try:
        term = L * (R_obs**2 * L - R_obs * L + R_obs - 0.5) / max(1e-6, V_obs)
        v_est = np.sign(R_obs - 0.5) * np.sqrt(max(0, term))
        a_est = abs(L) / max(1e-6, v_est)

        exp_term = np.exp(-v_est * a_est)
        t_est = M_obs - (a_est / (2 * v_est)) * ((1 - exp_term) / (1 + exp_term))

        return a_est, v_est, t_est
    except ValueError:
        return np.nan, np.nan, np.nan

def recover_parameters_from_file(input_file, output_file):
    """Reads a results file and writes recovered parameters to a file."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = line.strip().split(',')
            if len(data) < 6:
                print(f"Skipping malformed line: {line.strip()}")  
                continue  

            R_obs = float(data[3])  
            M_obs = float(data[4])  
            V_obs = float(data[5])  

            a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
            if not np.isnan(a_est):
                f_out.write(f"{a_est},{v_est},{t_est}\n")  
                print(f"DEBUG: {a_est},{v_est},{t_est}")  

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/recover.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace("results_", "recovered_results_")
    recover_parameters_from_file(input_file, output_file)
