#!/usr/bin/env python3
"""
main.py  –  Dynamic‐programming grid search for optimal
            two-tier loyalty bonuses.
"""
import numpy as np                # numerical core
import matplotlib.pyplot as plt   # plotting
from itertools import product
# -------------------------------------------------------------------
# 0)  PARAMETERS
beta, alpha, theta0, tau1 = 0.95, 0.20, 1.0, 20.0
C_vals = np.arange(0, 60, 20)
B_vals = np.arange(0, 60, 20)
R_menu = np.array([0., 10.])
S_step = 5.0
U = lambda s: np.sqrt(s)          # concave utility
# -------------------------------------------------------------------
# 1)  VALUE ITERATION
def value_iteration(phi_l, phi_h, tol=1e-4, iters=200):
    V = np.zeros((len(C_vals), len(B_vals)))
    for _ in range(iters):
        V_new = V.copy()
        for iC, C in enumerate(C_vals):
            for iB, B in enumerate(B_vals):
                phi = phi_l if C < tau1 else phi_h
                best = -1e9
                for R in R_menu:
                    theta = theta0 + phi * R
                    for S in np.arange(0, B+R*theta+1e-9, S_step):
                        C1, B1 = C+R, B+R*theta - S
                        jC = min(int(round(C1 / 20)), len(C_vals)-1)
                        jB = min(int(round(B1 / 20)), len(B_vals)-1)
                        best = max(best, U(S)-alpha*R + beta*V[jC, jB])
                V_new[iC, iB] = best
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    return V
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
    phi, heat = sweep()
    i, j = np.nanargmax(heat)//heat.shape[1], np.nanargmax(heat)%heat.shape[1]
    pl_star, ph_star = phi[i], phi[j]
    print(f"Optimum (phi_l, phi_h)=({pl_star:.2f},{ph_star:.2f})",
          f"W(0,0)={heat[i,j]:.2f}")
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(heat, origin='lower',
                   extent=[phi[0], phi[-1], phi[0], phi[-1]], aspect='auto')
    ax.scatter(ph_star, pl_star, s=120, c='r', marker='*')
    ax.set_xlabel(r'$\varphi_h$'); ax.set_ylabel(r'$\varphi_\ell$')
    ax.set_title('Platform value $W(0,0)$')
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig("heatmap_star.png", dpi=300)


