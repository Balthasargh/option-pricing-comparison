#!/usr/bin/env python3
"""
Interface Streamlit pour la comparaison des méthodes de pricing d'options.
Presets : Bitcoin et Soja (soybean futures).
"""

import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_banded
import pandas as pd

# ---------------------------------------------------------------------------
# Fonctions de pricing
# ---------------------------------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def monte_carlo_call(S0, K, T, r, sigma, n_paths=100_000, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0)
    discounted = np.exp(-r * T) * payoffs
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted, ddof=1) / np.sqrt(n_paths))
    return price, stderr


def pde_crank_nicolson_call(S0, K, T, r, sigma, S_max=None, M=200, N=200):
    if S_max is None:
        S_max = 4.0 * max(K, S0)
    dS = S_max / M
    dt = T / N
    S = np.linspace(0.0, S_max, M + 1)
    V = np.maximum(S - K, 0.0)

    alpha = 0.25 * dt * (sigma**2 * (S / dS)**2 - r * (S / dS))
    beta  = -0.5 * dt * (sigma**2 * (S / dS)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * (S / dS)**2 + r * (S / dS))

    lower_A = -alpha[1:M]
    diag_A  = 1.0 - beta[1:M]
    upper_A = -gamma[1:M]
    lower_B = alpha[1:M]
    diag_B  = 1.0 + beta[1:M]
    upper_B = gamma[1:M]

    for n in range(N):
        t = n * dt
        rhs = lower_B * V[0:M-1] + diag_B * V[1:M] + upper_B * V[2:M+1]
        V_Smax = S_max - K * np.exp(-r * (T - (t + dt)))
        rhs[-1] += gamma[M-1] * V_Smax

        ab = np.zeros((3, M - 1))
        ab[0, 1:] = upper_A[:-1]
        ab[1, :]  = diag_A
        ab[2, :-1] = lower_A[1:]

        V[1:M] = solve_banded((1, 1), ab, rhs)
        V[0] = 0.0
        V[M] = V_Smax

    idx = np.searchsorted(S, S0)
    if idx == 0:
        return float(V[0])
    if idx >= len(S):
        return float(V[-1])
    w = (S0 - S[idx-1]) / (S[idx] - S[idx-1])
    return float((1 - w) * V[idx-1] + w * V[idx])


# ---------------------------------------------------------------------------
# Presets marché (données approximatives début octobre 2026)
# ---------------------------------------------------------------------------
PRESETS = {
    "Personnalisé": {
        "S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20,
        "note": "Paramètres libres",
        "unit": "",
    },
    "Bitcoin (BTC)": {
        "S0": 85300.0,      # ~ prix spot USD début oct. 2026
        "K": 85000.0,       # strike ATM-ish
        "T": 30/365,        # 1 mois
        "r": 0.05,          # taux sans risque ≈ 5 %
        "sigma": 0.35,      # IV ATM ~ 30-38 % (Deribit / réalisées ~28-45 %)
        "note": "Call européen 1 mois ATM — vol ~35 % (marché options crypto)",
        "unit": "USD",
    },
    "Soja (Soybean ZS)": {
        "S0": 12.78,        # futures ~1278 ¢/bu = 12.78 USD/bu
        "K": 12.80,
        "T": 60/365,        # ~2 mois (vers expiration Nov)
        "r": 0.05,
        "sigma": 0.21,      # CVOL / réalisée ~20-22 %
        "note": "Call sur futures soja CBOT — vol ~21 % (CVOL / GARCH)",
        "unit": "USD/bu",
    },
}

# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Comparaison Pricing Options",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Comparaison des méthodes de pricing d'options")
st.markdown(
    "Comparez en temps réel la **formule de Black-Scholes**, "
    "la **simulation Monte Carlo** et la résolution de l'**EDP** (Crank-Nicolson). "
    "Presets disponibles pour **Bitcoin** et **Soja**."
)

# --- Sidebar ---
st.sidebar.header("Sous-jacent")
preset_name = st.sidebar.selectbox(
    "Preset marché",
    list(PRESETS.keys()),
    index=0,
    help="Charge des paramètres réalistes pour Bitcoin ou Soja",
)
preset = PRESETS[preset_name]
st.sidebar.caption(preset["note"])

st.sidebar.header("Paramètres de l'option")
S0 = st.sidebar.number_input(
    f"Spot / Futures \(S_0\) ({preset['unit']})",
    min_value=0.01,
    value=float(preset["S0"]),
    step=1.0 if preset["S0"] > 100 else 0.01,
    format="%.2f",
)
K = st.sidebar.number_input(
    f"Strike \(K\) ({preset['unit']})",
    min_value=0.01,
    value=float(preset["K"]),
    step=1.0 if preset["K"] > 100 else 0.01,
    format="%.2f",
)
T = st.sidebar.number_input(
    "Maturité \(T\) (années)",
    min_value=0.01,
    value=float(preset["T"]),
    step=0.01,
    format="%.4f",
)
r = st.sidebar.slider("Taux sans risque \(r\)", 0.0, 0.15, float(preset["r"]), 0.005)
sigma = st.sidebar.slider(
    "Volatilité \(\sigma\)",
    0.05, 1.50, float(preset["sigma"]), 0.01,
)

st.sidebar.header("Paramètres numériques")
n_paths = st.sidebar.select_slider(
    "Nombre de chemins Monte Carlo",
    options=[10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
    value=100_000,
)
grid_size = st.sidebar.select_slider(
    "Taille de grille EDP (M = N)",
    options=[50, 100, 200, 300, 400, 500],
    value=200,
)

run = st.sidebar.button("Lancer la comparaison", type="primary", use_container_width=True)

# --- Calcul ---
if run or "results" not in st.session_state:
    with st.spinner("Calcul en cours…"):
        t0 = time.perf_counter()
        price_bs = black_scholes_call(S0, K, T, r, sigma)
        t_bs = time.perf_counter() - t0

        t0 = time.perf_counter()
        price_mc, stderr_mc = monte_carlo_call(S0, K, T, r, sigma, n_paths=n_paths)
        t_mc = time.perf_counter() - t0
        err_mc = abs(price_mc - price_bs)

        t0 = time.perf_counter()
        price_pde = pde_crank_nicolson_call(S0, K, T, r, sigma, M=grid_size, N=grid_size)
        t_pde = time.perf_counter() - t0
        err_pde = abs(price_pde - price_bs)

        st.session_state.results = {
            "price_bs": price_bs, "t_bs": t_bs,
            "price_mc": price_mc, "err_mc": err_mc, "t_mc": t_mc, "stderr_mc": stderr_mc,
            "price_pde": price_pde, "err_pde": err_pde, "t_pde": t_pde,
            "n_paths": n_paths, "grid_size": grid_size,
            "params": (S0, K, T, r, sigma),
            "preset": preset_name,
            "unit": preset["unit"],
        }

res = st.session_state.results
unit = res.get("unit", "")

# --- Bannière preset ---
if res.get("preset") and res["preset"] != "Personnalisé":
    st.info(f"**Preset actif :** {res['preset']} — {PRESETS[res['preset']]['note']}")

# --- Métriques ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Black-Scholes (formule)",
        f"{res['price_bs']:.4f} {unit}",
        help="Prix exact (référence)",
    )
    st.caption(f"Temps : {res['t_bs']*1000:.2f} ms")

with col2:
    st.metric(
        "Monte Carlo",
        f"{res['price_mc']:.4f} {unit}",
        delta=f"{res['err_mc']:.4f}",
        delta_color="inverse",
        help=f"Erreur abs. vs BS | stderr ≈ {res['stderr_mc']:.5f}",
    )
    st.caption(f"Temps : {res['t_mc']*1000:.1f} ms | {res['n_paths']:,} chemins")

with col3:
    st.metric(
        "EDP Crank-Nicolson",
        f"{res['price_pde']:.4f} {unit}",
        delta=f"{res['err_pde']:.4f}",
        delta_color="inverse",
        help="Erreur abs. vs BS",
    )
    st.caption(f"Temps : {res['t_pde']*1000:.1f} ms | grille {res['grid_size']}×{res['grid_size']}")

st.divider()

# --- Tableau ---
st.subheader("Tableau comparatif")

df = pd.DataFrame([
    {
        "Méthode": "Black-Scholes (formule)",
        "Prix": res["price_bs"],
        "Erreur abs.": 0.0,
        "Temps (ms)": res["t_bs"] * 1000,
        "Type d'erreur": "—",
    },
    {
        "Méthode": f"Monte Carlo ({res['n_paths']:,} chemins)",
        "Prix": res["price_mc"],
        "Erreur abs.": res["err_mc"],
        "Temps (ms)": res["t_mc"] * 1000,
        "Type d'erreur": "Stochastique O(1/√N)",
    },
    {
        "Méthode": f"EDP CN ({res['grid_size']}×{res['grid_size']})",
        "Prix": res["price_pde"],
        "Erreur abs.": res["err_pde"],
        "Temps (ms)": res["t_pde"] * 1000,
        "Type d'erreur": "Déterministe O(ΔS²+Δt²)",
    },
])
st.dataframe(
    df.style.format({
        "Prix": "{:.6f}",
        "Erreur abs.": "{:.6f}",
        "Temps (ms)": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# --- Convergence ---
st.subheader("Courbes de convergence")
show_conv = st.checkbox("Afficher les courbes de convergence (peut prendre quelques secondes)", value=False)

if show_conv:
    with st.spinner("Calcul des courbes…"):
        mc_Ns = [10_000, 50_000, 100_000, 250_000, 500_000]
        mc_errs, mc_times = [], []
        for n in mc_Ns:
            t0 = time.perf_counter()
            p, _ = monte_carlo_call(S0, K, T, r, sigma, n_paths=n)
            mc_times.append(time.perf_counter() - t0)
            mc_errs.append(abs(p - res["price_bs"]))

        pde_Ms = [50, 100, 200, 300, 400]
        pde_errs, pde_times = [], []
        for m in pde_Ms:
            t0 = time.perf_counter()
            p = pde_crank_nicolson_call(S0, K, T, r, sigma, M=m, N=m)
            pde_times.append(time.perf_counter() - t0)
            pde_errs.append(abs(p - res["price_bs"]))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.loglog(mc_Ns, mc_errs, "o-", label="Monte Carlo", color="C0")
        ax.loglog(mc_Ns, mc_errs[0] * np.sqrt(mc_Ns[0] / np.array(mc_Ns)),
                  "--", color="C0", alpha=0.5, label=r"$\propto 1/\sqrt{N}$")
        ax.loglog(pde_Ms, pde_errs, "s-", label="EDP Crank-Nicolson", color="C1")
        ax.set_xlabel("N (chemins) ou M (pas spatiaux)")
        ax.set_ylabel("Erreur absolue")
        ax.set_title("Convergence de l'erreur")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)

        ax = axes[1]
        ax.loglog(mc_Ns, mc_times, "o-", label="Monte Carlo", color="C0")
        ax.loglog(pde_Ms, pde_times, "s-", label="EDP Crank-Nicolson", color="C1")
        ax.axhline(res["t_bs"], color="C2", ls="--", label="Formule BS")
        ax.set_xlabel("N (chemins) ou M (pas spatiaux)")
        ax.set_ylabel("Temps (s)")
        ax.set_title("Temps de calcul")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# --- Notes spécifiques ---
with st.expander("Notes sur Bitcoin et Soja"):
    st.markdown("""
**Bitcoin (BTC)**  
- Prix spot ~ 85 300 USD (début octobre 2026).  
- Volatilité implicite ATM options Deribit typiquement 30–40 % (1 mois) ; réalisée 30j souvent 28–45 %.  
- Taux sans risque ~ 5 %.  
- Pas de dividende → modèle Black-Scholes classique adapté.  
- Attention : jumps et vol stochastique importants → BS est une approximation.

**Soja (Soybean futures ZS – CBOT)**  
- Prix futures Nov ~ 1 278 ¢/bu ≈ **12,78 USD/bu**.  
- Volatilité (CVOL / GARCH) ~ 20–22 %.  
- Options sur futures → en pratique on utilise souvent Black-76 (équivalent à BS avec \( q = r \)).  
- Ici on applique BS « cash » pour la comparaison des méthodes numériques ; les conclusions sur précision/temps restent valides.

**Limites communes**  
Les trois méthodes comparent la **résolution numérique** du même modèle (BS).  
Elles ne capturent pas le smile de volatilité, les sauts ou le coût de stockage des commodities.
    """)

with st.expander("Quand utiliser quelle méthode ?"):
    st.markdown("""
| Situation | Méthode recommandée |
|-----------|---------------------|
| Option européenne vanilla, 1 actif | **Formule de Black-Scholes** |
| Besoin de Greeks / surface de prix | **EDP (différences finies)** |
| Options path-dependent ou multi-actifs | **Monte Carlo** |
| Options américaines | EDP (ou arbre binomial) |
| Haute dimension (> 3-4 actifs) | Monte Carlo |
    """)

st.caption(
    "Prix de référence = formule de Black-Scholes. "
    "Erreur = |prix méthode − prix BS|. "
    "Code : [option-pricing-comparison](https://github.com/Balthasargh/option-pricing-comparison)"
)
