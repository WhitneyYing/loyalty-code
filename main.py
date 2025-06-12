#!/usr/bin/env python3
"""
main.py – Dynamic-programming grid search for optimal
          two-tier loyalty bonuses 
"""
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 0) PARAMETERS
beta, alpha, theta0, tau1 = 0.95, 0.20, 1.0, 20.0
C_vals = np.arange(0, 60, 20)
B_vals = np.arange(0, 60, 20)
R_menu = np.array([0., 10.])
S_step = 5.0
U = lambda s: np.sqrt(s)           # concave utility

# -------------------------------------------------------------------
# 1)  VALUE ITERATION (single sweep helper)
def _bellman_sweep(V, phi_l, phi_h):
    V_new = np.empty_like(V)
    for iC, C in enumerate(C_vals):
        for iB, B in enumerate(B_vals):
            phi = phi_l if C < tau1 else phi_h
            best = -1e9
            for R in R_menu:
                theta = theta0 + phi * R
                for S in np.arange(0, B + R * theta + 1e-9, S_step):
                    C1, B1 = C + R, B + R * theta - S
                    jC = min(int(round(C1 / 20)), len(C_vals) - 1)
                    jB = min(int(round(B1 / 20)), len(B_vals) - 1)
                    best = max(best, U(S) - alpha * R + beta * V[jC, jB])
            V_new[iC, iB] = best
    return V_new

def value_iteration(phi_l, phi_h, tol=1e-4, max_iter=200, return_err=False):
    V = np.zeros((len(C_vals), len(B_vals)))
    errs = []
    for _ in range(max_iter):
        V_new = _bellman_sweep(V, phi_l, phi_h)
        err   = np.abs(V_new - V).max()
        errs.append(err)
        if err < tol:
            break
        V = V_new
    return (V, errs) if return_err else V

# -------------------------------------------------------------------
# 2)  GRID SEARCH + HEATMAP
def sweep():
    phi_grid = np.linspace(0, 0.5, 11)
    heat     = np.full((len(phi_grid), len(phi_grid)), np.nan)
    for i, pl in enumerate(phi_grid):
        for j, ph in enumerate(phi_grid):
            if ph <= pl:
                continue
            heat[i, j] = value_iteration(pl, ph)[0, 0]
    return phi_grid, heat

# -------------------------------------------------------------------
# 3)  MAIN ENTRY POINT
if __name__ == "__main__":
    # --- grid search ---
    phi, heat = sweep()
    i_star, j_star = divmod(np.nanargmax(heat), heat.shape[1])
    pl_star, ph_star = phi[i_star], phi[j_star]
    print(f"Optimum (phi_l, phi_h)=({pl_star:.2f},{ph_star:.2f}), "
          f"W(0,0)={heat[i_star, j_star]:.2f}")

    # --- Figure 1: heat-map ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(heat, origin='lower',
                    extent=[phi[0], phi[-1], phi[0], phi[-1]], aspect='auto')
    ax1.scatter(ph_star, pl_star, s=120, c='r', marker='*', label='optimum')
    ax1.set_xlabel(r'$\varphi_h$')
    ax1.set_ylabel(r'$\varphi_\ell$')
    ax1.set_title('Platform value $W(0,0)$')
    fig1.colorbar(im, ax=ax1, label='value')
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("heatmap_star.png", dpi=300)

    # --- Figure 2: convergence curve ---
    _, err = value_iteration(pl_star, ph_star, return_err=True)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.semilogy(range(1, len(err) + 1), err, marker='o')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Bellman error')
    ax2.set_title('Value-iteration convergence')
    fig2.tight_layout()
    fig2.savefig("bellman_conv.png", dpi=300)

