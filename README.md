# Comparaison des méthodes de pricing d'options

Ce dépôt compare **trois méthodes classiques** pour le pricing d'un call européen dans le modèle de Black-Scholes :

| Méthode              | Type                    | Caractéristiques principales                          |
|----------------------|-------------------------|-------------------------------------------------------|
| **Formule fermée**   | Analytique              | Exacte, quasi-instantanée                             |
| **Monte Carlo**      | Simulation stochastique | Flexible (path-dependent, multi-actifs), erreur O(1/√N) |
| **EDP (Crank-Nicolson)** | Différences finies   | Déterministe, convergence O(ΔS² + Δt²), adapté à la dim. 1 |

## Paramètres de l'option

- Spot \( S_0 = 100 \)
- Strike \( K = 100 \)
- Maturité \( T = 1 \) an
- Taux sans risque \( r = 5\% \)
- Volatilité \( \sigma = 20\% \)

Prix de référence (formule de Black-Scholes) : **10.450584**

## Résultats numériques

```
Méthode                          Prix        Erreur abs.    Temps (s)
----------------------------------------------------------------------
Black-Scholes (formule)       10.450584              —     0.001168
Monte Carlo (10 000 chemins)  10.345182       0.105402     0.002108
Monte Carlo (50 000 chemins)  10.457692       0.007108     0.008895
Monte Carlo (100 000 chemins) 10.420541       0.030042     0.010073
Monte Carlo (500 000 chemins) 10.450050       0.000534     0.078750
EDP CN (M=50, N=50)           10.594607       0.144024     0.003412
EDP CN (M=100, N=100)         10.410792       0.039791     0.002857
EDP CN (M=200, N=200)         10.440692       0.009892     0.006071
EDP CN (M=400, N=400)         10.448114       0.002470     0.014342
```

Le graphique de convergence (erreur et temps) est généré automatiquement par le script (`comparison_results.png`).

## Analyse

### Précision
- **Formule de Black-Scholes** : exacte (erreur machine uniquement).
- **Monte Carlo** : erreur statistique qui diminue en \( O(1/\sqrt{N}) \). Avec 500 000 chemins on atteint déjà une précision de l'ordre de \( 5\cdot10^{-4} \).
- **EDP Crank-Nicolson** : erreur déterministe qui diminue en \( O(\Delta S^2 + \Delta t^2) \). Avec une grille 400×400 l'erreur est d'environ \( 2.5\cdot10^{-3} \).

### Temps de calcul
- La formule est **quasi-instantanée** (< 2 ms).
- L'EDP est très efficace en dimension 1 : une grille 400×400 prend ~14 ms.
- Monte Carlo devient plus coûteux dès que l'on veut une précision élevée (500 k chemins ≈ 80 ms).

### Quand utiliser quelle méthode ?

| Situation                              | Méthode recommandée      |
|----------------------------------------|--------------------------|
| Option européenne vanilla, 1 actif     | Formule de Black-Scholes |
| Besoin de Greeks / surface de prix     | EDP (différences finies) |
| Options path-dependent ou multi-actifs | Monte Carlo              |
| Options américaines                    | EDP (ou arbre binomial)  |
| Haute dimension (> 3-4 actifs)         | Monte Carlo              |

## Installation & exécution

```bash
# Dépendances
pip install numpy scipy matplotlib

# Lancer la comparaison
python option_pricing.py
```

Le script affiche le tableau de résultats et génère le graphique `comparison_results.png`.

## Structure du code

- `black_scholes_call` : formule analytique
- `monte_carlo_call` : simulation exacte (loi log-normale) avec estimateur d'écart-type
- `pde_crank_nicolson_call` : schéma de Crank-Nicolson sur grille uniforme en \( S \), résolution par `scipy.linalg.solve_banded`

## Licence

MIT
