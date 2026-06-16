# EU Transport CO₂ Emissions – Empirical Analysis

**Bachelor's Thesis | Ca' Foscari University of Venice**  
**Author:** mhmd gholami  
**Dataset:** EU27 Member States, 2022

---

## Overview

This repository contains the data, analysis code and outputs for a Bachelor's thesis examining the **determinants of transport CO₂ emissions per capita** across all 27 EU Member States, using 2022 cross-sectional data from official Eurostat sources.

The thesis applies five quantitative methods:
1. Descriptive Statistics
2. Comparative Analysis
3. Pearson Correlation Analysis
4. Multiple OLS Regression
5. K-means Cluster Analysis

---

## Research Question

> Which economic, demographic and mobility-related factors are most closely associated with cross-country differences in transport CO₂ emissions per capita across EU Member States, and can distinct national mobility–emission profiles be identified empirically?

---

## Repository Structure

```
eu-transport-co2-thesis/
│
├── data/
│   └── DATASET-3.xlsx            # Final dataset (EU27, 2022, 13 variables)
│
├── analysis/
│   └── analysis.py               # Full Python analysis (all 5 methods + figures)
│
├── outputs/
│   ├── fig1_co2_per_capita.png   # Bar chart: CO₂/cap ranking
│   ├── fig2_gdp_vs_co2.png       # Scatter: GDP vs CO₂/cap
│   ├── fig3_correlation_heatmap.png
│   ├── fig4_regression_diagnostics.png
│   ├── fig5_elbow.png            # Elbow method for k selection
│   └── fig6_clusters.png         # Cluster scatter plot
│
├── docs/
│   └── Thesis_Chapters_2_3_4.docx  # Written chapters (Literature, Empirical, Conclusion)
│
├── requirements.txt
└── README.md
```

---

## Dataset Variables

| Variable | Description | Source (Eurostat) |
|---|---|---|
| `CO2_total` | Total transport CO₂ emissions (tonnes) | [env_ac_ainah_r2](https://ec.europa.eu/eurostat/databrowser/view/env_ac_ainah_r2/default/table) |
| `CO2_per_capita` | CO₂ per resident (derived: CO2_total ÷ Population) | Calculated by author |
| `Population` | National resident population | [demo_gind](https://ec.europa.eu/eurostat/databrowser/view/demo_gind/default/table) |
| `Population_density` | Inhabitants per km² | [demo_r_d3dens](https://ec.europa.eu/eurostat/databrowser/view/demo_r_d3dens/default/table) |
| `Passenger_km` | Total passenger-kilometres (rail) | [rail_pa_typepas](https://ec.europa.eu/eurostat/databrowser/view/rail_pa_typepas/default/table) |
| `Passenger_km_per_capita` | Passenger-km per resident (derived) | Calculated by author |
| `Freight_tkm` | Total freight tonne-kilometres (road) | [road_go_ta_tg](https://ec.europa.eu/eurostat/databrowser/view/road_go_ta_tg/default/table) |
| `Freight_tkm_per_capita` | Freight tkm per resident (derived) | Calculated by author |
| `GDP_per_capita` | GDP per capita in PPS index | [sdg_08_10](https://ec.europa.eu/eurostat/databrowser/view/sdg_08_10/default/table) |
| `EV_stock` | Total electric vehicle stock | [road_eqs_carpda](https://ec.europa.eu/eurostat/databrowser/view/road_eqs_carpda/default/table) |
| `EV_per_capita` | EV stock per resident (derived) | Calculated by author |

---

## Key Results

### Descriptive Statistics
| Variable | Mean | Median | Min | Max |
|---|---|---|---|---|
| CO₂ per capita (t/cap) | 1.640 | 0.863 | 0.317 (Romania) | 7.973 (Luxembourg) |
| Population density (inh/km²) | 187.0 | 105.3 | 18.3 (Finland) | 1696.8 (Malta) |
| GDP per capita (€ PPS) | 34,157 | 26,820 | 10,740 (Bulgaria) | 104,100 (Luxembourg) |
| EV per capita | 0.0064 | 0.0031 | 0.0006 (Greece) | 0.0213 (Luxembourg) |

### Correlation with CO₂ per capita
| Variable | r | p-value |
|---|---|---|
| GDP per capita | **0.68** | < 0.001 *** |
| EV per capita | 0.58 | 0.002 ** |
| Freight tkm per capita | 0.24 | 0.238 |
| Population density | 0.21 | 0.292 |
| Passenger-km per capita | 0.03 | 0.894 |

### OLS Regression (Reduced Model, n = 26)
- **R² = 0.589**, Adjusted R² = 0.511, F(4,21) = 7.52, p < 0.001
- Only **GDP per capita** is statistically significant (β = 5.04×10⁻⁵, p = 0.014)
- All VIF < 3 — no multicollinearity issue

### Cluster Analysis (k = 3)
| Cluster | Countries | Mean CO₂/cap |
|---|---|---|
| A – High-income, high-emission outliers | Denmark, Ireland, Luxembourg | 6.46 t/cap |
| B – Large mainstream group | 20 EU countries | 0.93 t/cap |
| C – Dense, wealthy, moderate-emission leaders | Belgium, Germany, Netherlands | 0.99 t/cap |

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/eu-transport-co2-thesis.git
cd eu-transport-co2-thesis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
# Copy DATASET-3.xlsx into the data/ folder

# 4. Run the full analysis
python analysis/analysis.py
```

Figures are saved automatically to the `outputs/` folder.

---

## Dependencies

See `requirements.txt`:
```
pandas
numpy
scipy
scikit-learn
statsmodels
matplotlib
openpyxl
```

---

## License

This project is for academic purposes. Data sourced from [Eurostat](https://ec.europa.eu/eurostat) — freely available under the Eurostat copyright notice.
