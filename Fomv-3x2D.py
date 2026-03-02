"""
FOMV: Field Operator for Measured Viability - Final Version
Author: Osvaldo Morales
License: AGPL-3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Parameters (Table 1)
# -----------------------------------------------------------------------------
params = {
    'alpha1': 0.1, 'alpha2': 0.2, 'delta': 0.05, 'beta1': 0.3,
    'gamma1': 0.2, 'gamma2': 0.1, 'gamma3': 0.1, 'phi1': 0.3,
    'phi2': 0.2, 'psi1': 0.2, 'psi2': 0.2, 'kappa1': 0.2, 'kappa2': 0.1,
    'Ec': 0.1, 'Er': 0.5, 'Lc': 1.5, 'Lr': 0.8,
}

sim_params = {
    'sigma': 0.05,
    'Tmax': 50,
    'R': 500,
    'Bgrid': 20,
    'Mgrid': 20,
    'B_range': [0, 1.2],
    'M_range': [0, 0.8],
    'bootstrap_reps': 100,
    'alpha': 0.05,
    'fast_samples': 20,
    'n_cores': mp.cpu_count()
}

# -----------------------------------------------------------------------------
# Core dynamics (vectorised)
# -----------------------------------------------------------------------------
def sigmoid(x): return 1 / (1 + np.exp(-x))

def hard_nonlinear_dynamics_vectorized(x, theta, eta):
    B, M, E, G, T, C = x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5]
    B_star = B + theta['alpha1']*(1 - E) - theta['alpha2']*G
    M_star = (1 - theta['delta'])*M + theta['beta1'] * sigmoid(B - T)
    E_star = E + theta['gamma1']*G - theta['gamma2']*B - theta['gamma3']*M
    G_star = G + theta['phi1']*E - theta['phi2']*(B + M)*(1 - T)
    T_star = T - theta['psi1']*M*(1 - G) + theta['psi2']*G
    C_star = C + theta['kappa1']*T - theta['kappa2']*B
    x_star = np.column_stack([B_star, M_star, E_star, G_star, T_star, C_star]) + eta
    return np.clip(x_star, 0, 1)

def generate_noise_vectorized(sigma, n):
    n_needed = n
    collected = []
    total = 0
    while total < n_needed:
        batch_size = min(10000, (n_needed - total) * 2)
        u = np.random.uniform(-1, 1, size=(batch_size, 6))
        prob = np.prod(0.75 * (1 - u**2), axis=1)
        accept = np.random.rand(batch_size) < prob
        accepted = u[accept]
        if len(accepted) > 0:
            collected.append(accepted)
            total += len(accepted)
    noise = np.vstack(collected)[:n_needed]
    return sigma * noise

# -----------------------------------------------------------------------------
# Absorption conditions (vectorised)
# -----------------------------------------------------------------------------
def is_collapsed_vectorized(x, theta):
    B, M, E = x[:,0], x[:,1], x[:,2]
    return (E <= theta['Ec']) | (B + M >= theta['Lc'])

def is_recovered_vectorized(x, theta):
    B, M, E = x[:,0], x[:,1], x[:,2]
    return (E >= theta['Er']) & (B + M <= theta['Lr'])

# -----------------------------------------------------------------------------
# Multi‑trajectory simulation (vectorised)
# -----------------------------------------------------------------------------
def simulate_trajectories_vectorized(x0, theta, sigma, Tmax):
    n = x0.shape[0]
    x = x0.copy()
    absorptions = np.full(n, None, dtype=object)
    times = np.full(n, Tmax, dtype=int)
    for t in range(Tmax):
        active = (absorptions == None)
        if not np.any(active): break
        collapsed = is_collapsed_vectorized(x[active], theta)
        recovered = is_recovered_vectorized(x[active], theta)
        idx_collapsed = np.where(active)[0][collapsed]
        absorptions[idx_collapsed] = 'C'
        times[idx_collapsed] = t
        idx_recovered = np.where(active)[0][recovered]
        absorptions[idx_recovered] = 'R'
        times[idx_recovered] = t
        still_active = active & (absorptions == None)
        if not np.any(still_active): break
        eta = generate_noise_vectorized(sigma, np.sum(still_active))
        x[still_active] = hard_nonlinear_dynamics_vectorized(x[still_active], theta, eta)
    return absorptions, times

# -----------------------------------------------------------------------------
# Fast variable sampling (returns averages of E,G,T,C)
# -----------------------------------------------------------------------------
def generate_fast_samples(B, M, theta, sigma, n_samples, burnin=500):
    samples = []
    fast = np.random.uniform(0, 1, size=4)   # (E,G,T,C)
    total_steps = n_samples + burnin
    for i in range(total_steps):
        x = np.array([B, M, fast[0], fast[1], fast[2], fast[3]])
        eta = generate_noise_vectorized(sigma, 1)[0]
        x_new = hard_nonlinear_dynamics_vectorized(x.reshape(1,-1), theta, eta.reshape(1,-1))
        fast = x_new[0, 2:]
        if i >= burnin:
            samples.append(fast.copy())
    samples = np.array(samples)
    E_avg = np.mean(samples[:,0])
    G_avg = np.mean(samples[:,1])
    T_avg = np.mean(samples[:,2])
    C_avg = np.mean(samples[:,3])
    return samples, (E_avg, G_avg, T_avg, C_avg)

# -----------------------------------------------------------------------------
# Single grid point computation
# -----------------------------------------------------------------------------
def compute_point(BM, theta, sigma, Tmax, R, fast_samples):
    B, M = BM
    try:
        fast_arr, (Ea, Ga, Ta, Ca) = generate_fast_samples(B, M, theta, sigma, fast_samples)
        all_times_C = []
        q_sum = 0.0
        total_traj = 0
        for fast in fast_arr:
            x0 = np.tile(np.array([B, M, fast[0], fast[1], fast[2], fast[3]]), (R, 1))
            absorptions, times = simulate_trajectories_vectorized(x0, theta, sigma, Tmax)
            q_sum += np.sum(absorptions == 'R')
            total_traj += R
            all_times_C.extend(times[absorptions == 'C'])
        q_hat = q_sum / total_traj if total_traj > 0 else np.nan
        mfpt_hat = np.mean(all_times_C) if all_times_C else np.nan
        return (B, M, q_hat, mfpt_hat, all_times_C, Ea, Ga, Ta, Ca)
    except Exception as e:
        print(f"Error at (B={B:.3f}, M={M:.3f}): {e}")
        raise

# -----------------------------------------------------------------------------
# Parallel grid scan
# -----------------------------------------------------------------------------
def estimate_on_grid_parallel(B_grid, M_grid, theta, sigma, Tmax, R,
                              fast_samples, n_cores):
    points = [(B, M) for B in B_grid for M in M_grid]
    func = partial(compute_point, theta=theta, sigma=sigma,
                   Tmax=Tmax, R=R, fast_samples=fast_samples)
    with mp.Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap(func, points), total=len(points), desc="Grid points"))

    nB, nM = len(B_grid), len(M_grid)
    Q = np.full((nB, nM), np.nan)
    MFPT = np.full((nB, nM), np.nan)
    E_avg = np.full((nB, nM), np.nan)
    G_avg = np.full((nB, nM), np.nan)
    T_avg = np.full((nB, nM), np.nan)
    C_avg = np.full((nB, nM), np.nan)
    times_data = {}
    idx = 0
    for i, B in enumerate(B_grid):
        for j, M in enumerate(M_grid):
            (_, _, q, mfpt, times, Ea, Ga, Ta, Ca) = results[idx]
            Q[i,j] = q
            MFPT[i,j] = mfpt
            E_avg[i,j] = Ea
            G_avg[i,j] = Ga
            T_avg[i,j] = Ta
            C_avg[i,j] = Ca
            times_data[(i,j)] = times
            idx += 1
    return Q, MFPT, times_data, E_avg, G_avg, T_avg, C_avg

# -----------------------------------------------------------------------------
# Bootstrap confidence bands (resampling)
# -----------------------------------------------------------------------------
def bootstrap_bands_from_times(times_data, B_grid, M_grid, bootstrap_reps, alpha=0.05):
    nB, nM = len(B_grid), len(M_grid)
    MFPT_hat = np.full((nB, nM), np.nan)
    MFPT_lower = np.full((nB, nM), np.nan)
    MFPT_upper = np.full((nB, nM), np.nan)
    for i in range(nB):
        for j in range(nM):
            times = times_data.get((i,j), [])
            if len(times) == 0: continue
            mfpt_hat = np.mean(times)
            MFPT_hat[i,j] = mfpt_hat
            boot_means = [np.mean(np.random.choice(times, size=len(times), replace=True))
                          for _ in range(bootstrap_reps)]
            MFPT_lower[i,j] = np.percentile(boot_means, 100 * alpha / 2)
            MFPT_upper[i,j] = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return MFPT_hat, MFPT_lower, MFPT_upper

# -----------------------------------------------------------------------------
# Basic 2D plot
# -----------------------------------------------------------------------------
def plot_mfpt_2d(B_grid, M_grid, MFPT, title="MFPT in (B, M)"):
    B_mesh, M_mesh = np.meshgrid(B_grid, M_grid, indexing='ij')
    plt.figure(figsize=(8,6))
    contour = plt.contourf(B_mesh, M_mesh, MFPT, levels=20, cmap='viridis')
    plt.colorbar(contour, label='MFPT')
    plt.xlabel('Backlog B'); plt.ylabel('Memory M'); plt.title(title)
    plt.show()

# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("FOMV: High‑resolution simulation")
    print("="*60)

    B_grid = np.linspace(sim_params['B_range'][0], sim_params['B_range'][1], sim_params['Bgrid'])
    M_grid = np.linspace(sim_params['M_range'][0], sim_params['M_range'][1], sim_params['Mgrid'])

    print(f"\nUsing {sim_params['n_cores']} cores. Grid: {sim_params['Bgrid']}×{sim_params['Mgrid']} points.")
    print("Estimating (may take several minutes)...")

    Q, MFPT, times_data, E_avg, G_avg, T_avg, C_avg = estimate_on_grid_parallel(
        B_grid, M_grid, params, sim_params['sigma'],
        sim_params['Tmax'], sim_params['R'],
        sim_params['fast_samples'], sim_params['n_cores']
    )

    MFPT_hat, MFPT_lower, MFPT_upper = bootstrap_bands_from_times(
        times_data, B_grid, M_grid, sim_params['bootstrap_reps'], sim_params['alpha']
    )

    plot_mfpt_2d(B_grid, M_grid, MFPT_hat, title="MFPT (B,M) – high resolution")

    # -------------------------------------------------------------------------
    # Build DataFrame with all variables
    # -------------------------------------------------------------------------
    import pandas as pd
    B_vals = np.repeat(B_grid, len(M_grid))
    M_vals = np.tile(M_grid, len(B_grid))
    data = {
        'B': B_vals,
        'M': M_vals,
        'E': E_avg.flatten(),
        'G': G_avg.flatten(),
        'T': T_avg.flatten(),
        'C': C_avg.flatten(),
        'MFPT': MFPT_hat.flatten(),
        'Q': Q.flatten()
    }
    df = pd.DataFrame(data).dropna()
    df['logMFPT'] = np.log(df['MFPT'])
    print(f"\nDataFrame with {len(df)} valid points. Columns: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # Interactive 3D visualisation (Plotly)
    # -------------------------------------------------------------------------
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import ipywidgets as widgets
        from IPython.display import display

        var_options = ['B', 'M', 'E', 'G', 'T', 'C', 'MFPT', 'Q', 'logMFPT']

        x_w = widgets.Dropdown(options=var_options, value='B', description='X:')
        y_w = widgets.Dropdown(options=var_options, value='M', description='Y:')
        z_w = widgets.Dropdown(options=var_options, value='T', description='Z:')
        col_w = widgets.Dropdown(options=var_options, value='logMFPT', description='Color:')
        size_w = widgets.FloatSlider(value=3, min=1, max=10, step=0.5, description='Size:')
        button = widgets.Button(description='Update 3D')
        out = widgets.Output()

        def update_3d(b):
            with out:
                out.clear_output()
                fig = px.scatter_3d(df, x=x_w.value, y=y_w.value, z=z_w.value,
                                    color=col_w.value, color_continuous_scale='viridis',
                                    title=f'Cube: {x_w.value} vs {y_w.value} vs {z_w.value}',
                                    opacity=0.8)
                fig.update_traces(marker=dict(size=size_w.value))
                fig.show()
        button.on_click(update_3d)

        display(widgets.VBox([x_w, y_w, z_w, col_w, size_w, button, out]))

        # ---------------------------------------------------------------------
        # 2D slices with slider
        # ---------------------------------------------------------------------
        print("\n--- 2D slices ---")
        fixed_var = widgets.Dropdown(options=var_options, value='M', description='Fix:')
        fixed_val = widgets.FloatSlider(min=df['M'].min(), max=df['M'].max(), step=0.05, description='Value:')
        x2d = widgets.Dropdown(options=var_options, value='B', description='X:')
        y2d = widgets.Dropdown(options=var_options, value='T', description='Y:')
        col2d = widgets.Dropdown(options=var_options, value='logMFPT', description='Color:')

        def update_2d(fixed_var, fixed_val, x2d, y2d, col2d):
            tol = 0.05 * (df[fixed_var].max() - df[fixed_var].min())
            subset = df[np.abs(df[fixed_var] - fixed_val) < tol]
            if len(subset) == 0:
                print("No data in this slice.")
                return
            plt.figure(figsize=(8,6))
            sc = plt.scatter(subset[x2d], subset[y2d], c=subset[col2d],
                             cmap='viridis', s=50, edgecolor='k')
            plt.colorbar(sc, label=col2d)
            plt.xlabel(x2d); plt.ylabel(y2d)
            plt.title(f'{fixed_var} ≈ {fixed_val:.2f}')
            plt.grid(True)
            plt.show()

        interact_2d = widgets.interactive(update_2d,
                                          fixed_var=fixed_var,
                                          fixed_val=fixed_val,
                                          x2d=x2d,
                                          y2d=y2d,
                                          col2d=col2d)
        display(interact_2d)

    except ImportError as e:
        print("\nInstall plotly, ipywidgets and pandas for interactive visualisation:")
        print("pip install plotly ipywidgets pandas")

    print("\nSimulation completed.")
