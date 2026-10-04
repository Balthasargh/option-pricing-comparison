# Comparaison des méthodes de pricing d'options

Ce dépôt compare **trois méthodes classiques** pour le pricing d'un call européen dans le modèle de Black-Scholes :

| Méthode              | Type                    | Caractéristiques principales                          |
|----------------------|-------------------------|-------------------------------------------------------|
| **Formule fermée**   | Analytique              | Exacte, quasi-instantanée                             |
| **Monte Carlo**      | Simulation stochastique | Flexible (path-dependent, multi-actifs), erreur O(1/√N) |
| **EDP (Crank-Nicolson)** | Différences finies   | Déterministe, convergence O(ΔS² + Δt²), adapté à la dim. 1 |

## Interface utilisateur (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Presets marché inclus

| Preset | Spot / Futures | Strike | Maturité | Vol | Source (début oct. 2026) |
|--------|----------------|--------|----------|-----|--------------------------|
| **Bitcoin (BTC)** | ~85 300 USD | 85 000 | 1 mois | 35 % | Spot + IV Deribit / réalisée |
| **Soja (ZS)** | ~12,78 USD/bu | 12,80 | ~2 mois | 21 % | Futures CBOT + CVOL / GARCH |
| Personnalisé | libre | libre | libre | libre | — |

Fonctionnalités :
- Choix du preset + ajustement des paramètres
- Affichage prix / erreur / temps pour les 3 méthodes
- Tableau comparatif + courbes de convergence optionnelles
- Notes sur les spécificités Bitcoin (jumps) et soja (futures / Black-76)

## Paramètres de l'option (exemple par défaut CLI)

- Spot \( S_0 = 100 \), Strike \( K = 100 \), \( T = 1 \) an, \( r = 5\% \), \( \sigma = 20\% \)
- Prix de référence BS : **10.450584**

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

## Analyse

### Précision
- **Formule de Black-Scholes** : exacte (erreur machine uniquement).
- **Monte Carlo** : erreur statistique \( O(1/\sqrt{N}) \).
- **EDP Crank-Nicolson** : erreur déterministe \( O(\Delta S^2 + \Delta t^2) \).

### Temps de calcul
- Formule quasi-instantanée ; EDP très efficace en dim. 1 ; Monte Carlo plus coûteux pour haute précision.

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
pip install -r requirements.txt
streamlit run app.py          # interface web
python option_pricing.py      # benchmark CLI
```

## Structure

| Fichier | Description |
|---------|-------------|
| `app.py` | Interface Streamlit + presets BTC / Soja |
| `option_pricing.py` | Script CLI de benchmark |
| `requirements.txt` | Dépendances |

## Licence

MIT
