# main.py  –  MIT Licence
# Reproduces DP grid search, heat-map (Fig 2) and Bellman-error plot (Fig 3)

import numpy as np, matplotlib.pyplot as plt
from itertools import product

# --- model primitives -------------------------------------------------
beta, alpha, theta0, tau1 = 0.95, 0.20, 1.0, 20.0
U = lambda s: np.sqrt(s)              # CRRA utility (ρ=0.5)

# coarse state / action grids — edit for high-res
C_vals = np.arange(0, 60, 20)         # cumulative spend
B_vals = np.arange(0, 60, 20)         # balance
R_menu = np.array([0., 10.])          # recharge choices
S_step = 5.0                          # spending grid step

def value_iteration(phi_l, phi_h, tol=1e-4, max_iter=200):
    """Inner DP: returns value matrix V indexed by (C_idx, B_idx)."""
    V = np.zeros((len(C_vals), len(B_vals)))
    for _ in range(max_iter):
        V_new = np.empty_like(V)
        for iC, C in enumerate(C_vals):
            for iB, B in enumerate(B_vals):
                phi = phi_l if C < tau1 else phi_h
                best = -1e9
                for R in R_menu:
                    theta = theta0 + phi * R
                    for S in np.arange(0, B + R*theta + 1e-9, S_step):
                        C1, B1 = C + R, B + R*theta - S
                        jC = min(int(round(C1/20)), len(C_vals)-1)
                        jB = min(int(round(B1/20)), len(B_vals)-1)
                        best = max(best, U(S) - alpha*R + beta*V[jC, jB])
                V_new[iC, iB] = best
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    return V

# --- outer grid search ------------------------------------------------
phi_grid = np.linspace(0, 0.5, 11)    # 0, 0.05, … , 0.50
heat = np.full((len(phi_grid), len(phi_grid)), np.nan)

for i, phi_l in enumerate(phi_grid):
    for j, phi_h in enumerate(phi_grid):
        if phi_h <= phi_l:             # enforce φ_h > φ_l
            continue
        heat[i,j] = value_iteration(phi_l, phi_h)[0,0]

# locate optimum
opt_idx = np.nanargmax(heat)
i_star, j_star = np.unravel_index(opt_idx, heat.shape)
phi_l_star, phi_h_star = phi_grid[i_star], phi_grid[j_star]
print(f"Optimal (phi_l, phi_h)=({phi_l_star:.2f},{phi_h_star:.2f}),  W(0,0)={heat[i_star,j_star]:.2f}")

# --- Figure 2: heat-map -----------------------------------------------
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(heat, origin='lower',
               extent=[phi_grid[0], phi_grid[-1], phi_grid[0], phi_grid[-1]],
               aspect='auto')
ax.scatter(phi_h_star, phi_l_star, s=120, c='r', marker='*', label='optimum')
ax.set_xlabel(r'$\varphi_h$'); ax.set_ylabel(r'$\varphi_\ell$')
ax.set_title('Platform value $W(0,0)$')
fig.colorbar(im, ax=ax, label='value'); ax.legend()
fig.tight_layout(); fig.savefig('heatmap_star.png', dpi=300)

# --- Figure 3: Bellman convergence at optimum -------------------------
def bellman_error(phi_l, phi_h):
    V, errs = np.zeros_like(heat[0:len(C_vals),0:len(B_vals)]), []
    for _ in range(200):
        V_new = value_iteration(phi_l, phi_h, max_iter=1)   # one sweep
        errs.append(np.abs(V_new - V).max())
        if errs[-1] < 1e-7: break
        V = V_new
    return errs

err = bellman_error(phi_l_star, phi_h_star)
plt.figure(figsize=(4,3))
plt.semilogy(err); plt.xlabel('iteration'); plt.ylabel(r'Bellman error')
plt.title('Value-iteration convergence'); plt.tight_layout()
plt.savefig('bellman_conv.png', dpi=300)

