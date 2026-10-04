/**
 * ============================================================================
 * Comparaison des méthodes de pricing d'options européennes (Call)
 * ============================================================================
 *
 * Trois méthodes implémentées :
 *   1. Formule fermée de Black-Scholes          → exacte, O(1)
 *   2. Simulation Monte Carlo                  → erreur O(1/√N), flexible
 *   3. EDP Black-Scholes (Crank-Nicolson)      → erreur O(ΔS² + Δt²), déterministe
 *
 * Compilation :
 *   g++ -O3 -std=c++17 -o option_pricing option_pricing.cpp -lm
 *   # Avec OpenMP (parallélisation MC) :
 *   g++ -O3 -std=c++17 -fopenmp -o option_pricing option_pricing.cpp -lm
 *
 * Exécution :
 *   ./option_pricing              # exemple vanilla S=K=100
 *   ./option_pricing --btc        # preset Bitcoin
 *   ./option_pricing --soja       # preset Soja (futures CBOT)
 *   ./option_pricing --all        # les trois presets
 *   ./option_pricing --help
 *
 * Auteur : projet option-pricing-comparison
 * Licence : MIT
 */

#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// Utilitaires mathématiques et mesure de temps
// ============================================================================

/**
 * Fonction de répartition de la loi normale centrée réduite N(0,1).
 *
 * Utilise l'approximation d'Abramowitz & Stegun (formule 26.2.17),
 * précise à ~1.5e-7 près. Suffisante pour le pricing d'options.
 *
 * @param x  Point d'évaluation
 * @return   P(Z ≤ x) où Z ~ N(0,1)
 */
inline double norm_cdf(double x) {
    // Coefficients de l'approximation rationnelle
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    // Gestion du signe (la formule est donnée pour x ≥ 0)
    int sign = (x < 0) ? -1 : 1;
    x = std::fabs(x) / std::sqrt(2.0);

    // Approximation de erf via fraction continue
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
                     * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

// Alias pour le chronomètre haute résolution
using Clock = std::chrono::high_resolution_clock;

/**
 * Calcule le temps écoulé depuis un instant de départ, en millisecondes.
 */
double elapsed_ms(const Clock::time_point& start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ============================================================================
// 1. Formule fermée de Black-Scholes
// ============================================================================

/**
 * Prix exact d'un call européen dans le modèle de Black-Scholes.
 *
 * Formule :
 *   C = S · N(d1) − K · e^{−rT} · N(d2)
 * avec
 *   d1 = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
 *   d2 = d1 − σ√T
 *
 * Hypothèses : taux et volatilité constants, pas de dividende, marché complet.
 *
 * @param S      Prix spot de l'actif sous-jacent
 * @param K      Prix d'exercice (strike)
 * @param T      Maturité en années (T > 0)
 * @param r      Taux d'intérêt sans risque (continu)
 * @param sigma  Volatilité annualisée
 * @return       Prix du call
 */
double black_scholes_call(double S, double K, double T, double r, double sigma) {
    // À maturité, le call vaut simplement le payoff max(S−K, 0)
    if (T <= 0.0) return std::max(S - K, 0.0);

    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;

    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// ============================================================================
// 2. Simulation Monte Carlo
// ============================================================================

/** Résultat d'une simulation Monte Carlo : prix estimé + erreur standard. */
struct MCResult {
    double price;   // Estimateur du prix (moyenne des payoffs actualisés)
    double stderr;  // Écart-type de l'estimateur (≈ erreur statistique)
};

/**
 * Pricing d'un call européen par simulation Monte Carlo.
 *
 * Schéma exact (pas d'erreur de discrétisation) :
 *   S_T = S_0 · exp( (r − σ²/2)·T + σ√T · Z )   avec Z ~ N(0,1)
 *
 * Le prix est la moyenne des payoffs actualisés :
 *   Ĉ = e^{−rT} · (1/N) · Σ max(S_T^{(i)} − K, 0)
 *
 * L'erreur standard décroît en O(1/√N).
 *
 * @param S0       Spot initial
 * @param K        Strike
 * @param T        Maturité
 * @param r        Taux sans risque
 * @param sigma    Volatilité
 * @param n_paths  Nombre de trajectoires simulées
 * @param seed     Graine du générateur aléatoire (reproductibilité)
 * @return         Structure {prix, écart-type de l'estimateur}
 */
MCResult monte_carlo_call(double S0, double K, double T, double r, double sigma,
                          int n_paths = 100000, unsigned seed = 42) {
    // Générateur Mersenne Twister 64 bits + distribution normale
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Facteurs constants (pré-calculés pour la performance)
    double drift = (r - 0.5 * sigma * sigma) * T;   // drift de la log-normale
    double vol   = sigma * std::sqrt(T);             // coefficient de diffusion
    double disc  = std::exp(-r * T);                 // facteur d'actualisation

    double sum    = 0.0;   // somme des payoffs actualisés
    double sum_sq = 0.0;   // somme des carrés (pour la variance)

    for (int i = 0; i < n_paths; ++i) {
        double Z = dist(rng);                              // bruit gaussien
        double ST = S0 * std::exp(drift + vol * Z);        // prix terminal
        double payoff = std::max(ST - K, 0.0);             // payoff du call
        double disc_payoff = disc * payoff;                // actualisation
        sum    += disc_payoff;
        sum_sq += disc_payoff * disc_payoff;
    }

    // Estimateur de la moyenne et de l'écart-type
    double mean = sum / n_paths;
    double var  = (sum_sq - sum * sum / n_paths) / (n_paths - 1);  // variance empirique
    double se   = std::sqrt(var / n_paths);                        // erreur standard

    return {mean, se};
}

// ============================================================================
// 3. EDP – Différences finies (schéma de Crank-Nicolson)
// ============================================================================

/**
 * Résolution d'un système linéaire tridiagonal A·x = d par l'algorithme de Thomas.
 *
 * Complexité O(n) — optimal pour une matrice bande.
 * Les vecteurs lower, diag, upper représentent les trois diagonales.
 * Le second membre d est modifié en place pour contenir la solution x.
 *
 * @param lower  Sous-diagonale  (taille n, lower[0] non utilisé)
 * @param diag   Diagonale principale (taille n)
 * @param upper  Sur-diagonale   (taille n, upper[n-1] non utilisé)
 * @param d      Second membre → devient la solution en sortie
 */
void solve_tridiagonal(const std::vector<double>& lower,
                       const std::vector<double>& diag,
                       const std::vector<double>& upper,
                       std::vector<double>& d) {
    int n = static_cast<int>(diag.size());
    std::vector<double> c_prime(n, 0.0);   // coefficients de la factorisation
    std::vector<double> d_prime(n, 0.0);

    // Phase avant (forward sweep)
    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = d[0] / diag[0];

    for (int i = 1; i < n; ++i) {
        double denom = diag[i] - lower[i] * c_prime[i - 1];
        c_prime[i] = (i < n - 1) ? upper[i] / denom : 0.0;
        d_prime[i] = (d[i] - lower[i] * d_prime[i - 1]) / denom;
    }

    // Phase arrière (back substitution)
    d[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        d[i] = d_prime[i] - c_prime[i] * d[i + 1];
    }
}

/**
 * Pricing d'un call européen par résolution de l'EDP de Black-Scholes
 * avec le schéma de Crank-Nicolson (différences finies).
 *
 * EDP de Black-Scholes (variables S, t) :
 *   ∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0
 *
 * Schéma de Crank-Nicolson :
 *   - Moyenne des schémas explicite et implicite → ordre 2 en temps et en espace
 *   - Inconditionnellement stable
 *   - À chaque pas de temps on résout un système tridiagonal
 *
 * Conditions aux limites :
 *   V(0, t)   = 0                          (call vaut 0 si S=0)
 *   V(Smax,t) = Smax − K·e^{−r(T−t)}       (comportement asymptotique)
 *
 * Condition terminale (t = T) :
 *   V(S, T) = max(S − K, 0)
 *
 * @param S0     Spot auquel on évalue le prix (interpolation linéaire)
 * @param K      Strike
 * @param T      Maturité
 * @param r      Taux sans risque
 * @param sigma  Volatilité
 * @param M      Nombre de pas en espace (grille S)
 * @param N      Nombre de pas en temps
 * @return       Prix approximé du call en S0
 */
double pde_crank_nicolson_call(double S0, double K, double T, double r, double sigma,
                               int M = 200, int N = 200) {
    // Domaine spatial suffisamment large pour que la CL en Smax soit précise
    double S_max = 4.0 * std::max(K, S0);
    double dS = S_max / M;   // pas spatial
    double dt = T / N;       // pas temporel

    // Grille en S : S_i = i · dS,  i = 0 … M
    std::vector<double> S(M + 1);
    for (int i = 0; i <= M; ++i) S[i] = i * dS;

    // Condition terminale : payoff du call
    std::vector<double> V(M + 1);
    for (int i = 0; i <= M; ++i) V[i] = std::max(S[i] - K, 0.0);

    // Coefficients du schéma CN pour chaque nœud spatial
    // alpha, beta, gamma interviennent dans les diagonales des matrices A et B
    std::vector<double> alpha(M + 1), beta(M + 1), gamma(M + 1);
    for (int i = 0; i <= M; ++i) {
        double Si_dS = S[i] / dS;
        alpha[i] = 0.25 * dt * (sigma * sigma * Si_dS * Si_dS - r * Si_dS);
        beta[i]  = -0.5 * dt * (sigma * sigma * Si_dS * Si_dS + r);
        gamma[i] = 0.25 * dt * (sigma * sigma * Si_dS * Si_dS + r * Si_dS);
    }

    // Matrices A (côté implicite, temps n+1) et B (côté explicite, temps n)
    // On ne stocke que les nœuds intérieurs i = 1 … M−1
    int n_int = M - 1;
    std::vector<double> lower_A(n_int), diag_A(n_int), upper_A(n_int);
    std::vector<double> lower_B(n_int), diag_B(n_int), upper_B(n_int);

    for (int j = 0; j < n_int; ++j) {
        int i = j + 1;  // index spatial correspondant
        lower_A[j] = -alpha[i];
        diag_A[j]  = 1.0 - beta[i];
        upper_A[j] = -gamma[i];
        lower_B[j] =  alpha[i];
        diag_B[j]  = 1.0 + beta[i];
        upper_B[j] =  gamma[i];
    }

    // Marche arrière en temps : de t=T vers t=0
    for (int n = 0; n < N; ++n) {
        double t = n * dt;

        // Construction du second membre B · V^n
        std::vector<double> rhs(n_int);
        for (int j = 0; j < n_int; ++j) {
            int i = j + 1;
            rhs[j] = lower_B[j] * V[i - 1]
                   + diag_B[j]  * V[i]
                   + upper_B[j] * V[i + 1];
        }

        // Contribution de la condition en S = S_max (côté A)
        double V_Smax = S_max - K * std::exp(-r * (T - (t + dt)));
        rhs[n_int - 1] += gamma[M - 1] * V_Smax;

        // Résolution du système tridiagonal A · V^{n+1} = rhs
        solve_tridiagonal(lower_A, diag_A, upper_A, rhs);

        // Mise à jour de la solution
        for (int j = 0; j < n_int; ++j) V[j + 1] = rhs[j];
        V[0] = 0.0;       // CL en S=0
        V[M] = V_Smax;    // CL en S=Smax
    }

    // Interpolation linéaire pour obtenir V(S0, 0)
    if (S0 <= 0.0)   return V[0];
    if (S0 >= S_max) return V[M];

    int idx = static_cast<int>(S0 / dS);
    if (idx >= M) idx = M - 1;
    double w = (S0 - S[idx]) / dS;          // poids d'interpolation
    return (1.0 - w) * V[idx] + w * V[idx + 1];
}

// ============================================================================
// Benchmark et affichage des résultats
// ============================================================================

/** Paramètres d'une option + métadonnées d'affichage. */
struct Params {
    double S0, K, T, r, sigma;
    std::string name;   // nom du sous-jacent (ex. "Bitcoin")
    std::string unit;   // unité monétaire (ex. "USD")
};

/**
 * Lance la comparaison des trois méthodes pour un jeu de paramètres donné
 * et affiche un tableau formaté (prix, erreur absolue, temps).
 */
void run_comparison(const Params& p) {
    std::cout << "======================================================================\n";
    std::cout << "COMPARAISON DES MÉTHODES DE PRICING D'OPTIONS (C++)\n";
    std::cout << "======================================================================\n";
    std::cout << "Sous-jacent : " << p.name << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Paramètres  : S0=" << p.S0 << " " << p.unit
              << ", K=" << p.K << ", T=" << p.T
              << ", r=" << p.r << ", σ=" << p.sigma << "\n\n";

    // En-tête du tableau
    std::cout << std::left << std::setw(32) << "Méthode"
              << std::right << std::setw(14) << "Prix"
              << std::setw(14) << "Erreur abs."
              << std::setw(12) << "Temps (ms)" << "\n";
    std::cout << std::string(72, '-') << "\n";

    // ----- 1. Formule de Black-Scholes (référence exacte) -----
    auto t0 = Clock::now();
    double price_bs = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma);
    double t_bs = elapsed_ms(t0);

    std::cout << std::left << std::setw(32) << "Black-Scholes (formule)"
              << std::right << std::setw(14) << std::setprecision(6) << price_bs
              << std::setw(14) << "—"
              << std::setw(12) << std::setprecision(3) << t_bs << "\n";

    // ----- 2. Monte Carlo pour plusieurs tailles d'échantillon -----
    std::vector<int> mc_paths = {10000, 50000, 100000, 500000};
    for (int n : mc_paths) {
        t0 = Clock::now();
        MCResult mc = monte_carlo_call(p.S0, p.K, p.T, p.r, p.sigma, n);
        double t_mc = elapsed_ms(t0);
        double err = std::fabs(mc.price - price_bs);

        std::string label = "Monte Carlo (" + std::to_string(n) + " chemins)";
        std::cout << std::left << std::setw(32) << label
                  << std::right << std::setw(14) << std::setprecision(6) << mc.price
                  << std::setw(14) << err
                  << std::setw(12) << std::setprecision(3) << t_mc << "\n";
    }

    // ----- 3. EDP Crank-Nicolson pour plusieurs résolutions de grille -----
    std::vector<int> grids = {50, 100, 200, 400};
    for (int m : grids) {
        t0 = Clock::now();
        double price_pde = pde_crank_nicolson_call(p.S0, p.K, p.T, p.r, p.sigma, m, m);
        double t_pde = elapsed_ms(t0);
        double err = std::fabs(price_pde - price_bs);

        std::string label = "EDP CN (M=N=" + std::to_string(m) + ")";
        std::cout << std::left << std::setw(32) << label
                  << std::right << std::setw(14) << std::setprecision(6) << price_pde
                  << std::setw(14) << err
                  << std::setw(12) << std::setprecision(3) << t_pde << "\n";
    }

    std::cout << std::string(72, '-') << "\n\n";
    std::cout << "Observations :\n";
    std::cout << "  • Formule BS : exacte, quasi-instantanée.\n";
    std::cout << "  • Monte Carlo : erreur O(1/√N), stochastique.\n";
    std::cout << "  • EDP Crank-Nicolson : déterministe, O(ΔS² + Δt²).\n";
    std::cout << "  • En C++ optimisé (-O3), les temps sont typiquement 5–20× plus\n";
    std::cout << "    bas qu'en Python pur pour Monte Carlo et EDP.\n\n";
}

// ============================================================================
// Point d'entrée
// ============================================================================

int main(int argc, char* argv[]) {
    // ----- Presets de marché (données approximatives début octobre 2026) -----

    // Exemple pédagogique classique
    Params vanilla{100.0, 100.0, 1.0, 0.05, 0.20, "Vanilla (exemple)", ""};

    // Bitcoin : spot ~85 300 USD, IV ATM ~35 %, maturité 1 mois
    Params btc{85300.0, 85000.0, 30.0 / 365.0, 0.05, 0.35, "Bitcoin (BTC)", "USD"};

    // Soja (futures CBOT) : ~12,78 USD/bu, vol ~21 %, maturité ~2 mois
    Params soja{12.78, 12.80, 60.0 / 365.0, 0.05, 0.21, "Soja (ZS futures)", "USD/bu"};

    // ----- Gestion des arguments en ligne de commande -----
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--btc" || arg == "-b") {
            run_comparison(btc);
        } else if (arg == "--soja" || arg == "-s") {
            run_comparison(soja);
        } else if (arg == "--all" || arg == "-a") {
            run_comparison(vanilla);
            run_comparison(btc);
            run_comparison(soja);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--btc|--soja|--all]\n"
                      << "  (sans argument : exemple vanilla S=K=100)\n";
            return 0;
        } else {
            std::cerr << "Argument inconnu. Utilisez --help.\n";
            return 1;
        }
    } else {
        // Comportement par défaut : exemple vanilla
        run_comparison(vanilla);
    }

    return 0;
}
