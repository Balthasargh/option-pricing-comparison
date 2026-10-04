# Comparaison des méthodes de pricing d'options

Ce dépôt compare **trois méthodes classiques** pour le pricing d'un call européen dans le modèle de Black-Scholes :

| Méthode              | Type                    | Caractéristiques principales                          |
|----------------------|-------------------------|-------------------------------------------------------|
| **Formule fermée**   | Analytique              | Exacte, quasi-instantanée                             |
| **Monte Carlo**      | Simulation stochastique | Flexible (path-dependent, multi-actifs), erreur O(1/√N) |
| **EDP (Crank-Nicolson)** | Différences finies   | Déterministe, convergence O(ΔS² + Δt²), adapté à la dim. 1 |

## Interface utilisateur (Streamlit)

Une interface web interactive est disponible :

```bash
pip install -r requirements.txt
streamlit run app.py
```

Fonctionnalités de l'UI :
- Saisie des paramètres de l'option (spot, strike, maturité, taux, volatilité)
- Choix du nombre de chemins Monte Carlo et de la taille de grille EDP
- Affichage en temps réel des prix, erreurs et temps de calcul
- Tableau comparatif
- Courbes de convergence (optionnelles)

## Paramètres de l'option (exemple par défaut)

- Spot \( S_0 = 100 \)
- Strike \( K = 100 \)
- Maturité \( T = 1 \) an
- Taux sans risque \( r = 5\% \)
- Volatilité \( \sigma = 20\% \)

Prix de référence (formule de Black-Scholes) : **10.450584**

## Résultats numériques (script CLI)

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

Le graphique de convergence est généré par `python option_pricing.py`.

## Analyse

### Précision
- **Formule de Black-Scholes** : exacte (erreur machine uniquement).
- **Monte Carlo** : erreur statistique qui diminue en \( O(1/\sqrt{N}) \).
- **EDP Crank-Nicolson** : erreur déterministe qui diminue en \( O(\Delta S^2 + \Delta t^2) \).

### Temps de calcul
- La formule est **quasi-instantanée** (< 2 ms).
- L'EDP est très efficace en dimension 1.
- Monte Carlo devient plus coûteux pour une précision élevée.

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
pip install -r requirements.txt

# Interface web interactive
streamlit run app.py

# Ou script en ligne de commande
python option_pricing.py
```

## Structure du code

| Fichier | Description |
|---------|-------------|
| `app.py` | Interface Streamlit |
| `option_pricing.py` | Script CLI de benchmark |
| `requirements.txt` | Dépendances |

Fonctions principales :
- `black_scholes_call` : formule analytique
- `monte_carlo_call` : simulation exacte (loi log-normale)
- `pde_crank_nicolson_call` : schéma de Crank-Nicolson

## Licence

MIT
