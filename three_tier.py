#!/usr/bin/env python3
"""
three_tier.py – dynamic-programming grid search for optimal
                3-tier loyalty bonuses (fixed thresholds 20 & 40)
"""
import numpy as np, itertools as it, matplotlib.pyplot as plt

# ——————————————————— 0)  primitives ——————————————————————
beta, alpha, theta0 = 0.95, 0.20, 1.0
U = lambda s: np.sqrt(s)                     # concave utility
C_vals = np.arange(0, 60, 20)               # {0,20,40}
B_vals = np.arange(0, 60, 20)               # {0,20,40}
R_menu, S_step = np.array([0., 10.]), 5.0

# fixed thresholds for 3 tiers: [0,20), [20,40), [40,∞)
tau = np.array([20., 40.])

# ——————————————————— 1)  helper ————————————————————————
def theta(C, R, phi):
    tier = np.searchsorted(tau, C, side="right") - 1
    return theta0 + phi[tier] * R

def bellman_sweep(V, phi):
    Vn = np.empty_like(V)
    for iC, C in enumerate(C_vals):
        for iB, B in enumerate(B_vals):
            best = -1e9
            for R in R_menu:
                th = theta(C, R, phi)
                for S in np.arange(0, B + R*th + 1e-9, S_step):
                    C1, B1 = C + R, B + R*th - S
                    jC = min(int(round(C1/20)), len(C_vals)-1)
                    jB = min(int(round(B1/20)), len(B_vals)-1)
                    best = max(best, U(S) - alpha*R + beta*V[jC, jB])
            Vn[iC, iB] = best
    return Vn

def value_iteration(phi, tol=1e-4, max_iter=200):
    V = np.zeros((len(C_vals), len(B_vals)))
    for _ in range(max_iter):
        V_new = bellman_sweep(V, phi)
        if np.abs(V_new - V).max() < tol:
            break
        V = V_new
    return V

# ——————————————————— 2)  grid search ————————————————————
phi_grid = np.linspace(0, 0.5, 11)          # 0,0.05,…,0.50
best_val, best_phi = -1, None
records = []

for φ1, φ2, φ3 in it.product(phi_grid, repeat=3):
    if not (φ1 <= φ2 <= φ3):                # monotone
        continue
    V00 = value_iteration((φ1, φ2, φ3))[0, 0]
    records.append((φ1, φ2, φ3, V00))
    if V00 > best_val:
        best_val, best_phi = V00, (φ1, φ2, φ3)

print(f"Optimal φ* = {best_phi},   W(0,0) = {best_val:.2f}")

# ——————————————————— 3)  heat-map (fix φ3 = φ3*) —————————
φ3_star = best_phi[2]
heat = np.full((len(phi_grid), len(phi_grid)), np.nan)
for i, φ1 in enumerate(phi_grid):
    for j, φ2 in enumerate(phi_grid):
        if φ1 <= φ2 <= φ3_star:
            heat[i, j] = value_iteration((φ1, φ2, φ3_star))[0, 0]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(heat, origin="lower",
               extent=[phi_grid[0], phi_grid[-1], phi_grid[0], phi_grid[-1]],
               aspect="auto", cmap="viridis")
ax.scatter(best_phi[1], best_phi[0], s=120, c="r", marker="*",
           label="optimum (φ₁*,φ₂*)")
ax.set_xlabel(r"$\varphi_2$"); ax.set_ylabel(r"$\varphi_1$")
ax.set_title(f"3-tier value $W(0,0)$ with $\\varphi_3={φ3_star:.2f}$")
fig.colorbar(im, ax=ax, label="value"); ax.legend(); fig.tight_layout()
fig.savefig("heatmap_3tier.png", dpi=300)

