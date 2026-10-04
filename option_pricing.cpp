/**
 * Comparaison des méthodes de pricing d'options européennes (Call)
 * - Formule fermée de Black-Scholes
 * - Simulation Monte Carlo
 * - Résolution de l'EDP de Black-Scholes par différences finies (Crank-Nicolson)
 *
 * Compilation :
 *   g++ -O3 -std=c++17 -o option_pricing option_pricing.cpp -lm
 *   # ou avec OpenMP pour paralléliser le Monte Carlo :
 *   g++ -O3 -std=c++17 -fopenmp -o option_pricing option_pricing.cpp -lm
 *
 * Exécution :
 *   ./option_pricing
 *   ./option_pricing --btc
 *   ./option_pricing --soja
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

// ---------------------------------------------------------------------------
// Utilitaires
// ---------------------------------------------------------------------------

// CDF de la loi normale standard (approximation d'Abramowitz & Stegun)
inline double norm_cdf(double x) {
    // Constants for approximation
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = (x < 0) ? -1 : 1;
    x = std::fabs(x) / std::sqrt(2.0);

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

using Clock = std::chrono::high_resolution_clock;

double elapsed_ms(const Clock::time_point& start) {
    auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// ---------------------------------------------------------------------------
// 1. Formule fermée de Black-Scholes
// ---------------------------------------------------------------------------
double black_scholes_call(double S, double K, double T, double r, double sigma) {
    if (T <= 0.0) return std::max(S - K, 0.0);
    double sqrtT = std::sqrt(T);
    double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    double d2 = d1 - sigma * sqrtT;
    return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// ---------------------------------------------------------------------------
// 2. Monte Carlo
// ---------------------------------------------------------------------------
struct MCResult {
    double price;
    double stderr;
};

MCResult monte_carlo_call(double S0, double K, double T, double r, double sigma,
                          int n_paths = 100000, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol   = sigma * std::sqrt(T);
    double disc  = std::exp(-r * T);

    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = 0; i < n_paths; ++i) {
        double Z = dist(rng);
        double ST = S0 * std::exp(drift + vol * Z);
        double payoff = std::max(ST - K, 0.0);
        double disc_payoff = disc * payoff;
        sum += disc_payoff;
        sum_sq += disc_payoff * disc_payoff;
    }

    double mean = sum / n_paths;
    double var  = (sum_sq - sum * sum / n_paths) / (n_paths - 1);
    double se   = std::sqrt(var / n_paths);

    return {mean, se};
}

// ---------------------------------------------------------------------------
// 3. EDP – Crank-Nicolson (différences finies)
// ---------------------------------------------------------------------------
// Résolution d'un système tridiagonal Ax = d (algorithme de Thomas)
void solve_tridiagonal(const std::vector<double>& lower,
                       const std::vector<double>& diag,
                       const std::vector<double>& upper,
                       std::vector<double>& d) {
    int n = static_cast<int>(diag.size());
    std::vector<double> c_prime(n, 0.0);
    std::vector<double> d_prime(n, 0.0);

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = d[0] / diag[0];

    for (int i = 1; i < n; ++i) {
        double denom = diag[i] - lower[i] * c_prime[i - 1];
        c_prime[i] = (i < n - 1) ? upper[i] / denom : 0.0;
        d_prime[i] = (d[i] - lower[i] * d_prime[i - 1]) / denom;
    }

    d[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        d[i] = d_prime[i] - c_prime[i] * d[i + 1];
    }
}

double pde_crank_nicolson_call(double S0, double K, double T, double r, double sigma,
                               int M = 200, int N = 200) {
    double S_max = 4.0 * std::max(K, S0);
    double dS = S_max / M;
    double dt = T / N;

    std::vector<double> S(M + 1);
    for (int i = 0; i <= M; ++i) S[i] = i * dS;

    std::vector<double> V(M + 1);
    for (int i = 0; i <= M; ++i) V[i] = std::max(S[i] - K, 0.0);

    // Coefficients (taille M+1, on utilise indices 1..M-1)
    std::vector<double> alpha(M + 1), beta(M + 1), gamma(M + 1);
    for (int i = 0; i <= M; ++i) {
        double Si_dS = S[i] / dS;
        alpha[i] = 0.25 * dt * (sigma * sigma * Si_dS * Si_dS - r * Si_dS);
        beta[i]  = -0.5 * dt * (sigma * sigma * Si_dS * Si_dS + r);
        gamma[i] = 0.25 * dt * (sigma * sigma * Si_dS * Si_dS + r * Si_dS);
    }

    // Matrices A (côté n+1) et B (côté n) pour i = 1..M-1
    int n_int = M - 1;
    std::vector<double> lower_A(n_int), diag_A(n_int), upper_A(n_int);
    std::vector<double> lower_B(n_int), diag_B(n_int), upper_B(n_int);

    for (int j = 0; j < n_int; ++j) {
        int i = j + 1;  // index spatial
        lower_A[j] = -alpha[i];
        diag_A[j]  = 1.0 - beta[i];
        upper_A[j] = -gamma[i];
        lower_B[j] = alpha[i];
        diag_B[j]  = 1.0 + beta[i];
        upper_B[j] = gamma[i];
    }

    for (int n = 0; n < N; ++n) {
        double t = n * dt;
        std::vector<double> rhs(n_int);

        for (int j = 0; j < n_int; ++j) {
            int i = j + 1;
            rhs[j] = lower_B[j] * V[i - 1] + diag_B[j] * V[i] + upper_B[j] * V[i + 1];
        }

        // Condition en S = S_max
        double V_Smax = S_max - K * std::exp(-r * (T - (t + dt)));
        rhs[n_int - 1] += gamma[M - 1] * V_Smax;

        // Résolution A * V_new = rhs
        solve_tridiagonal(lower_A, diag_A, upper_A, rhs);

        for (int j = 0; j < n_int; ++j) V[j + 1] = rhs[j];
        V[0] = 0.0;
        V[M] = V_Smax;
    }

    // Interpolation linéaire en S0
    if (S0 <= 0.0) return V[0];
    if (S0 >= S_max) return V[M];

    int idx = static_cast<int>(S0 / dS);
    if (idx >= M) idx = M - 1;
    double w = (S0 - S[idx]) / dS;
    return (1.0 - w) * V[idx] + w * V[idx + 1];
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------
struct Params {
    double S0, K, T, r, sigma;
    std::string name;
    std::string unit;
};

void run_comparison(const Params& p) {
    std::cout << "======================================================================\n";
    std::cout << "COMPARAISON DES MÉTHODES DE PRICING D'OPTIONS (C++)\n";
    std::cout << "======================================================================\n";
    std::cout << "Sous-jacent : " << p.name << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Paramètres  : S0=" << p.S0 << " " << p.unit
              << ", K=" << p.K << ", T=" << p.T
              << ", r=" << p.r << ", σ=" << p.sigma << "\n\n";

    std::cout << std::left << std::setw(32) << "Méthode"
              << std::right << std::setw(14) << "Prix"
              << std::setw(14) << "Erreur abs."
              << std::setw(12) << "Temps (ms)" << "\n";
    std::cout << std::string(72, '-') << "\n";

    // 1. Formule
    auto t0 = Clock::now();
    double price_bs = black_scholes_call(p.S0, p.K, p.T, p.r, p.sigma);
    double t_bs = elapsed_ms(t0);

    std::cout << std::left << std::setw(32) << "Black-Scholes (formule)"
              << std::right << std::setw(14) << std::setprecision(6) << price_bs
              << std::setw(14) << "—"
              << std::setw(12) << std::setprecision(3) << t_bs << "\n";

    // 2. Monte Carlo
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

    // 3. EDP
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

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    Params vanilla{100.0, 100.0, 1.0, 0.05, 0.20, "Vanilla (exemple)", ""};
    Params btc{85300.0, 85000.0, 30.0 / 365.0, 0.05, 0.35, "Bitcoin (BTC)", "USD"};
    Params soja{12.78, 12.80, 60.0 / 365.0, 0.05, 0.21, "Soja (ZS futures)", "USD/bu"};

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
        run_comparison(vanilla);
    }

    return 0;
}
