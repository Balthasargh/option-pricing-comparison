#!/usr/bin/env python3
"""
Comparaison des méthodes de pricing d'options européennes (Call)
- Formule fermée de Black-Scholes
- Simulation Monte Carlo
- Résolution de l'EDP de Black-Scholes par différences finies (Crank-Nicolson)

Mesure de la précision (erreur absolue vs formule exacte) et du temps de calcul.
"""

import time
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paramètres de l'option (Call européen)
# ---------------------------------------------------------------------------
S0 = 100.0      # Spot
K = 100.0       # Strike
T = 1.0         # Maturité (années)
r = 0.05        # Taux sans risque
sigma = 0.20    # Volatilité


# ---------------------------------------------------------------------------
# 1. Formule fermée de Black-Scholes
# ---------------------------------------------------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """Prix exact d'un call européen via la formule de Black-Scholes."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# 2. Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo_call(S0, K, T, r, sigma, n_paths=100_000, seed=42):
    """
    Prix d'un call européen par simulation Monte Carlo (schéma exact log-normal).
    Retourne (prix, écart-type de l'estimateur).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = np.mean(discounted)
    stderr = np.std(discounted, ddof=1) / np.sqrt(n_paths)
    return price, stderr


# ---------------------------------------------------------------------------
# 3. EDP – Différences finies (Crank-Nicolson)
# ---------------------------------------------------------------------------
def pde_crank_nicolson_call(S0, K, T, r, sigma, S_max=None, M=200, N=200):
    """
    Résolution de l'EDP de Black-Scholes par schéma de Crank-Nicolson
    sur une grille en S (prix de l'actif) et t (temps).
    
    M : nombre de pas en espace (S)
    N : nombre de pas en temps
    """
    if S_max is None:
        S_max = 4.0 * K   # domaine suffisamment large

    dS = S_max / M
    dt = T / N
    S = np.linspace(0.0, S_max, M + 1)

    # Conditions aux limites & condition terminale (payoff)
    V = np.maximum(S - K, 0.0)

    # Coefficients pour le schéma CN (forme standard sur grille uniforme en S)
    # On utilise la formulation en variables originales.
    alpha = 0.25 * dt * (sigma**2 * (S / dS)**2 - r * (S / dS))
    beta  = -0.5 * dt * (sigma**2 * (S / dS)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * (S / dS)**2 + r * (S / dS))

    # Matrices tridiagonales A (côté n+1) et B (côté n)
    # On stocke en format bandé pour solve_banded
    # A * V^{n+1} = B * V^n

    # Pour i = 1..M-1
    lower_A = -alpha[1:M]
    diag_A  = 1.0 - beta[1:M]
    upper_A = -gamma[1:M]

    lower_B = alpha[1:M]
    diag_B  = 1.0 + beta[1:M]
    upper_B = gamma[1:M]

    # Conditions aux bornes :
    # S=0  : V = 0
    # S=S_max : V = S - K e^{-r(T-t)}  (comportement asymptotique du call)

    for n in range(N):
        t = n * dt
        # Second membre
        rhs = np.zeros(M - 1)
        rhs[:] = (
            lower_B * V[0:M-1]
            + diag_B * V[1:M]
            + upper_B * V[2:M+1]
        )
        # Condition en S=0 (déjà V[0]=0)
        # Condition en S=S_max
        V_Smax = S_max - K * np.exp(-r * (T - (t + dt)))
        rhs[-1] += gamma[M-1] * V_Smax   # contribution de la borne supérieure côté A

        # Résolution du système tridiagonal
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = upper_A[:-1]   # super-diagonale
        ab[1, :]  = diag_A         # diagonale
        ab[2, :-1] = lower_A[1:]   # sous-diagonale

        V_interior = solve_banded((1, 1), ab, rhs)
        V[1:M] = V_interior
        V[0] = 0.0
        V[M] = V_Smax

    # Interpolation linéaire pour obtenir le prix en S0
    idx = np.searchsorted(S, S0)
    if idx == 0:
        price = V[0]
    elif idx >= len(S):
        price = V[-1]
    else:
        w = (S0 - S[idx-1]) / (S[idx] - S[idx-1])
        price = (1 - w) * V[idx-1] + w * V[idx]
    return price


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_comparison():
    print("=" * 70)
    print("COMPARAISON DES MÉTHODES DE PRICING D'OPTIONS")
    print("=" * 70)
    print(f"Paramètres : S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print()

    # 1. Formule exacte
    t0 = time.perf_counter()
    price_bs = black_scholes_call(S0, K, T, r, sigma)
    t_bs = time.perf_counter() - t0
    print(f"{'Méthode':<30} {'Prix':>12} {'Erreur abs.':>14} {'Temps (s)':>12}")
    print("-" * 70)
    print(f"{'Black-Scholes (formule)':<30} {price_bs:12.6f} {'—':>14} {t_bs:12.6f}")

    # 2. Monte Carlo – plusieurs tailles d'échantillon
    mc_results = []
    for n_paths in [10_000, 50_000, 100_000, 500_000]:
        t0 = time.perf_counter()
        price_mc, stderr = monte_carlo_call(S0, K, T, r, sigma, n_paths=n_paths)
        t_mc = time.perf_counter() - t0
        err = abs(price_mc - price_bs)
        mc_results.append((n_paths, price_mc, err, t_mc, stderr))
        print(f"{'Monte Carlo (' + str(n_paths) + ' chemins)':<30} {price_mc:12.6f} {err:14.6f} {t_mc:12.6f}")

    # 3. EDP Crank-Nicolson – plusieurs résolutions de grille
    pde_results = []
    for M, N in [(50, 50), (100, 100), (200, 200), (400, 400)]:
        t0 = time.perf_counter()
        price_pde = pde_crank_nicolson_call(S0, K, T, r, sigma, M=M, N=N)
        t_pde = time.perf_counter() - t0
        err = abs(price_pde - price_bs)
        pde_results.append((M, N, price_pde, err, t_pde))
        print(f"{'EDP CN (M=' + str(M) + ', N=' + str(N) + ')':<30} {price_pde:12.6f} {err:14.6f} {t_pde:12.6f}")

    print("-" * 70)
    print()
    print("Observations :")
    print("  • La formule de Black-Scholes est exacte (à la précision machine près) et quasi-instantanée.")
    print("  • Monte Carlo converge en O(1/√N) ; l'erreur est stochastique.")
    print("  • L'EDP (Crank-Nicolson) est déterministe et converge en O(ΔS² + Δt²).")
    print("  • Pour une précision donnée, la formule est imbattable ; l'EDP est plus rapide")
    print("    que Monte Carlo en dimension 1 ; Monte Carlo reste indispensable en haute dimension.")
    print()

    # Sauvegarde des résultats pour le README / graphiques
    return {
        "bs": (price_bs, t_bs),
        "mc": mc_results,
        "pde": pde_results,
    }


def plot_results(results):
    """Génère un graphique de comparaison erreur vs temps."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Erreur vs nombre de chemins / taille de grille ---
    ax = axes[0]
    n_paths = [r[0] for r in results["mc"]]
    err_mc = [r[2] for r in results["mc"]]
    ax.loglog(n_paths, err_mc, "o-", label="Monte Carlo", color="C0")
    # Référence 1/√N
    ax.loglog(n_paths, err_mc[0] * np.sqrt(n_paths[0] / np.array(n_paths)),
              "--", color="C0", alpha=0.5, label=r"$\propto 1/\sqrt{N}$")

    M_vals = [r[0] for r in results["pde"]]
    err_pde = [r[3] for r in results["pde"]]
    ax.loglog(M_vals, err_pde, "s-", label="EDP Crank-Nicolson", color="C1")
    ax.set_xlabel("N (chemins) ou M (pas spatiaux)")
    ax.set_ylabel("Erreur absolue")
    ax.set_title("Convergence de l'erreur")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    # --- Temps de calcul ---
    ax = axes[1]
    t_mc = [r[3] for r in results["mc"]]
    t_pde = [r[4] for r in results["pde"]]
    ax.loglog(n_paths, t_mc, "o-", label="Monte Carlo", color="C0")
    ax.loglog(M_vals, t_pde, "s-", label="EDP Crank-Nicolson", color="C1")
    ax.axhline(results["bs"][1], color="C2", ls="--", label="Black-Scholes (formule)")
    ax.set_xlabel("N (chemins) ou M (pas spatiaux)")
    ax.set_ylabel("Temps de calcul (s)")
    ax.set_title("Temps de calcul")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150)
    print("Graphique sauvegardé : comparison_results.png")
    plt.close()


if __name__ == "__main__":
    results = run_comparison()
    plot_results(results)
