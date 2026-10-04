#!/usr/bin/env python3
"""
Comparaison des méthodes de pricing d'options européennes (Call)
================================================================

Trois méthodes implémentées :
  1. Formule fermée de Black-Scholes          → exacte, O(1)
  2. Simulation Monte Carlo                   → erreur O(1/√N), flexible
  3. EDP Black-Scholes (Crank-Nicolson)       → erreur O(ΔS² + Δt²), déterministe

Mesure de la précision (erreur absolue vs formule exacte) et du temps de calcul.
Génère également un graphique de convergence (comparison_results.png).

Usage :
  python option_pricing.py
"""

import time
import numpy as np
from scipy.stats import norm
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paramètres de l'option (Call européen) — exemple pédagogique
# ---------------------------------------------------------------------------
S0 = 100.0      # Prix spot de l'actif sous-jacent
K = 100.0       # Prix d'exercice (strike)
T = 1.0         # Maturité en années
r = 0.05        # Taux d'intérêt sans risque (continu)
sigma = 0.20    # Volatilité annualisée


# ---------------------------------------------------------------------------
# 1. Formule fermée de Black-Scholes
# ---------------------------------------------------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """
    Prix exact d'un call européen via la formule de Black-Scholes.

    Formule :
        C = S · N(d1) − K · e^{−rT} · N(d2)
    avec
        d1 = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
        d2 = d1 − σ√T

    Hypothèses : taux et vol constants, pas de dividende, marché complet.

    Parameters
    ----------
    S : float
        Prix spot
    K : float
        Strike
    T : float
        Maturité (années)
    r : float
        Taux sans risque
    sigma : float
        Volatilité

    Returns
    -------
    float
        Prix du call
    """
    # À maturité, le call vaut le payoff
    if T <= 0:
        return max(S - K, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------------
# 2. Simulation Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo_call(S0, K, T, r, sigma, n_paths=100_000, seed=42):
    """
    Prix d'un call européen par simulation Monte Carlo.

    Schéma exact (pas d'erreur de discrétisation temporelle) :
        S_T = S_0 · exp( (r − σ²/2)·T + σ√T · Z )   avec Z ~ N(0,1)

    Le prix est la moyenne des payoffs actualisés :
        Ĉ = e^{−rT} · (1/N) · Σ max(S_T^{(i)} − K, 0)

    L'erreur standard de l'estimateur décroît en O(1/√N).

    Parameters
    ----------
    S0 : float
        Spot initial
    K : float
        Strike
    T : float
        Maturité
    r : float
        Taux sans risque
    sigma : float
        Volatilité
    n_paths : int
        Nombre de trajectoires simulées
    seed : int
        Graine du générateur (reproductibilité)

    Returns
    -------
    price : float
        Estimateur du prix
    stderr : float
        Écart-type de l'estimateur (erreur statistique)
    """
    rng = np.random.default_rng(seed)

    # Tirage de N variables gaussiennes indépendantes
    Z = rng.standard_normal(n_paths)

    # Prix terminaux (formule exacte de la solution de l'EDS de Black-Scholes)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Payoffs du call, actualisés
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs

    # Estimateurs
    price = np.mean(discounted)
    stderr = np.std(discounted, ddof=1) / np.sqrt(n_paths)
    return price, stderr


# ---------------------------------------------------------------------------
# 3. EDP – Différences finies (schéma de Crank-Nicolson)
# ---------------------------------------------------------------------------
def pde_crank_nicolson_call(S0, K, T, r, sigma, S_max=None, M=200, N=200):
    """
    Résolution de l'EDP de Black-Scholes par le schéma de Crank-Nicolson.

    EDP (variables S, t) :
        ∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0

    Schéma de Crank-Nicolson :
      - Moyenne des schémas explicite et implicite
      - Ordre 2 en temps et en espace
      - Inconditionnellement stable
      - À chaque pas de temps : résolution d'un système tridiagonal

    Conditions aux limites :
      V(0, t)    = 0                         (call vaut 0 si S = 0)
      V(Smax, t) = Smax − K·e^{−r(T−t)}      (comportement asymptotique)

    Condition terminale (t = T) :
      V(S, T) = max(S − K, 0)

    Parameters
    ----------
    S0 : float
        Spot auquel on évalue le prix (interpolation linéaire)
    K : float
        Strike
    T : float
        Maturité
    r : float
        Taux sans risque
    sigma : float
        Volatilité
    S_max : float or None
        Borne supérieure du domaine spatial (défaut : 4·K)
    M : int
        Nombre de pas en espace
    N : int
        Nombre de pas en temps

    Returns
    -------
    float
        Prix approximé du call en S0
    """
    if S_max is None:
        S_max = 4.0 * K   # domaine suffisamment large

    dS = S_max / M        # pas spatial
    dt = T / N            # pas temporel
    S = np.linspace(0.0, S_max, M + 1)

    # Condition terminale : payoff
    V = np.maximum(S - K, 0.0)

    # Coefficients du schéma CN (forme standard sur grille uniforme en S)
    # alpha, beta, gamma entrent dans les diagonales des matrices A et B
    alpha = 0.25 * dt * (sigma**2 * (S / dS)**2 - r * (S / dS))
    beta  = -0.5 * dt * (sigma**2 * (S / dS)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * (S / dS)**2 + r * (S / dS))

    # Diagonales de A (côté implicite, temps n+1) et B (côté explicite, temps n)
    # On ne traite que les nœuds intérieurs i = 1 … M−1
    lower_A = -alpha[1:M]
    diag_A  = 1.0 - beta[1:M]
    upper_A = -gamma[1:M]

    lower_B = alpha[1:M]
    diag_B  = 1.0 + beta[1:M]
    upper_B = gamma[1:M]

    # Marche arrière en temps : de t = T vers t = 0
    for n in range(N):
        t = n * dt

        # Second membre = B · V^n
        rhs = np.zeros(M - 1)
        rhs[:] = (
            lower_B * V[0:M-1]
            + diag_B * V[1:M]
            + upper_B * V[2:M+1]
        )

        # Contribution de la condition en S = S_max (côté A)
        V_Smax = S_max - K * np.exp(-r * (T - (t + dt)))
        rhs[-1] += gamma[M-1] * V_Smax

        # Assemblage de la matrice bande pour solve_banded
        # ab[0] = sur-diagonale, ab[1] = diagonale, ab[2] = sous-diagonale
        ab = np.zeros((3, M - 1))
        ab[0, 1:] = upper_A[:-1]
        ab[1, :]  = diag_A
        ab[2, :-1] = lower_A[1:]

        # Résolution du système tridiagonal A · V^{n+1} = rhs
        V_interior = solve_banded((1, 1), ab, rhs)
        V[1:M] = V_interior
        V[0] = 0.0          # CL en S = 0
        V[M] = V_Smax       # CL en S = Smax

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
# Benchmark : comparaison précision / temps
# ---------------------------------------------------------------------------
def run_comparison():
    """
    Exécute les trois méthodes pour plusieurs niveaux de précision,
    affiche un tableau récapitulatif et retourne les résultats structurés.
    """
    print("=" * 70)
    print("COMPARAISON DES MÉTHODES DE PRICING D'OPTIONS")
    print("=" * 70)
    print(f"Paramètres : S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print()

    # En-tête
    print(f"{'Méthode':<30} {'Prix':>12} {'Erreur abs.':>14} {'Temps (s)':>12}")
    print("-" * 70)

    # 1. Formule exacte (référence)
    t0 = time.perf_counter()
    price_bs = black_scholes_call(S0, K, T, r, sigma)
    t_bs = time.perf_counter() - t0
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

    return {
        "bs": (price_bs, t_bs),
        "mc": mc_results,
        "pde": pde_results,
    }


def plot_results(results):
    """
    Génère un graphique de comparaison erreur vs temps de calcul
    et le sauvegarde sous comparison_results.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Graphique 1 : convergence de l'erreur ---
    ax = axes[0]
    n_paths = [r[0] for r in results["mc"]]
    err_mc = [r[2] for r in results["mc"]]
    ax.loglog(n_paths, err_mc, "o-", label="Monte Carlo", color="C0")
    # Courbe de référence ∝ 1/√N
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

    # --- Graphique 2 : temps de calcul ---
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


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_comparison()
    plot_results(results)
