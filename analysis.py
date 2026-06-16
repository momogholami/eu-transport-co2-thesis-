"""
Transport CO2 Emissions in EU Countries — Empirical Analysis
Author: Mohammad Hossein Gholami
University: Ca' Foscari University of Venice, 2025/2026

All methods follow:
  McKinney, W. (2022). Python for Data Analysis: Data Wrangling with
  pandas, NumPy, and Jupyter (3rd ed.). O'Reilly Media.

Libraries used exactly as described in the book:
  - numpy   (Ch. 4)  — array operations and mathematical functions
  - pandas  (Ch. 5–6, 10–11) — DataFrame, descriptive stats, groupBy, time series
  - matplotlib.pyplot (Ch. 9) — all visualisations
  - scipy.stats       — Pearson correlation (Ch. 5.3 companion)
  - statsmodels.api   — OLS regression (Ch. 12 companion / McKinney p.8)
  - sklearn.cluster   — KMeans (McKinney p.8, scikit-learn)
  - sklearn.preprocessing — StandardScaler
"""

# ── 0. IMPORTS ─────────────────────────────────────────────────────────────────
# Import conventions follow McKinney (2022), p. 16
import numpy as np                          # McKinney Ch. 4
import pandas as pd                         # McKinney Ch. 5
import matplotlib                           # McKinney Ch. 9
matplotlib.use('Agg')
import matplotlib.pyplot as plt             # McKinney Ch. 9
import matplotlib.patches as mpatches
from scipy import stats                     # Pearson r, probplot
import statsmodels.api as sm                # OLS regression — McKinney p. 8
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

OUT = '/home/user/workspace/eu-transport-co2-thesis/outputs/'

# ── shared plot style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

TEAL = '#20808D'
RUST = '#A84B2F'
DARK = '#1B474D'
GOLD = '#D19900'
GRAY = '#AAAAAA'

# ── 1. DATA — built as a pandas DataFrame (McKinney Ch. 5) ────────────────────
# McKinney (2022, Ch. 5): "The DataFrame is a rectangular table of data and
# contains an ordered, named collection of columns."

data = {
    'country': [
        'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark',
        'Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
        'Italy','Latvia','Lithuania','Luxembourg','Malta','Netherlands',
        'Poland','Portugal','Romania','Slovakia','Slovenia','Spain','Sweden'
    ],
    'co2_per_capita': [
        0.748, 0.994, 0.317, 0.429, 0.500, 0.639, 7.758,
        0.542, 0.680, 0.744, 0.750, 0.380, 0.449, 4.528,
        0.531, 0.704, 1.382, 7.973, 0.554, 1.075,
        0.965, 0.490, 0.320, 0.539, 0.524, 0.601, 0.476
    ],
    'gdp_per_capita': [
        40700, 46900, 10740, 22400, 27900, 32300, 58600,
        27900, 38500, 36000, 43500, 22600, 24200, 63700,
        31900, 25600, 26600, 104100, 29200, 52000,
        24800, 26400, 17500, 24600, 31200, 28400, 47400
    ],
    'ev_per_capita': [
        0.0060, 0.0093, 0.0008, 0.0015, 0.0006, 0.0024, 0.0125,
        0.0042, 0.0099, 0.0062, 0.0105, 0.0006, 0.0012, 0.0118,
        0.0035, 0.0028, 0.0020, 0.0213, 0.0030, 0.0130,
        0.0017, 0.0032, 0.0008, 0.0016, 0.0035, 0.0044, 0.0190
    ],
    'pop_density': [
        107.6, 383.0, 65.0, 73.5, 99.0, 137.0, 136.8,
        31.0, 18.3, 117.5, 232.3, 83.0, 107.5, 72.0,
        197.0, 31.0, 47.0, 232.0, 1696.8, 421.0,
        124.0, 111.5, 85.0, 113.0, 102.0, 93.5, 24.5
    ],
    # NaN for missing countries (McKinney Ch. 7: handling missing data with pd.NA)
    'passenger_km': [
        1094.0, 706.0, np.nan, 403.0, np.nan, 636.0, 1401.0,
        1150.0, 107.0, 545.0, 300.0, 563.0, 260.0, 804.0,
        588.0, 580.0, 260.0, 850.0, np.nan, 1207.0,
        570.0, 300.0, 403.0, np.nan, 436.0, 560.0, 850.0
    ],
    'freight_tkm': [
        5200, 8100, 4800, 3800, np.nan, 6500, 13500,
        5100, 5800, 4900, 6200, 2800, 4100, 10200,
        4700, 7500, 18984, 5400, 2100, 7800,
        10500, 3500, 3300, 4300, 5400, 3900, 5600
    ],
}

# Create the DataFrame — McKinney (2022), p. 129
df = pd.DataFrame(data)
df = df.set_index('country')   # set country as the index (McKinney Ch. 5.1)

print("=== DataFrame info (McKinney Ch. 5.3) ===")
print(df.info())

# ── 2. DESCRIPTIVE STATISTICS (McKinney Ch. 5.3) ──────────────────────────────
# "pandas objects are equipped with a set of common mathematical and statistical
#  methods." — McKinney (2022), p. 165
print("\n=== Descriptive Statistics — df.describe() ===")
desc = df.describe()           # McKinney (2022), p. 165: DataFrame.describe()
print(desc.round(3))

# Also compute mean, median, std manually using pandas Series methods
print("\nMean  :", df['co2_per_capita'].mean().round(3))
print("Median:", df['co2_per_capita'].median().round(3))
print("Std   :", df['co2_per_capita'].std().round(3))
print("Min   :", df['co2_per_capita'].min(), "—", df['co2_per_capita'].idxmin())
print("Max   :", df['co2_per_capita'].max(), "—", df['co2_per_capita'].idxmax())

# ── 3. FIGURE 1 — Horizontal bar chart (McKinney Ch. 9) ───────────────────────
# "matplotlib is a desktop plotting package designed for creating plots and
#  figures suitable for publication." — McKinney (2022), p. 6
df_sorted = df['co2_per_capita'].sort_values(ascending=True)  # McKinney Ch. 5.2: sort_values
colors = [RUST if v > 4 else TEAL for v in df_sorted.values]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(df_sorted.index, df_sorted.values, color=colors, edgecolor='white', height=0.7)
ax.set_xlabel('Tonnes CO₂ per person')
ax.set_title('Figure 1. Transport CO₂ Emissions per Capita by EU Country (2022)')
mean_val = df['co2_per_capita'].mean()
ax.axvline(x=mean_val, color=GOLD, linewidth=1.4, linestyle='--',
           label=f'EU27 mean ({mean_val:.2f} t)')
ax.legend(fontsize=10)
ax.set_xlim(0, 9)
for bar, val in zip(bars, df_sorted.values):
    if val > 3.5:
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9, color=RUST, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + 'fig1_co2_per_capita.png')
plt.close()
print("\nfig1 saved")

# ── 4. FIGURE 2 — Scatter plot (McKinney Ch. 9) ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df['gdp_per_capita'], df['co2_per_capita'],
           color=TEAL, s=55, edgecolors='white', linewidth=0.5, zorder=3)

# Trend line using scipy.stats.linregress — companion to McKinney Ch. 5.3
# McKinney (2022), p. 168: "correlation and covariance"
mask = df['gdp_per_capita'].notna() & df['co2_per_capita'].notna()
m, b, r, p_val, _ = stats.linregress(df.loc[mask, 'gdp_per_capita'],
                                      df.loc[mask, 'co2_per_capita'])
xline = np.linspace(df['gdp_per_capita'].min(), df['gdp_per_capita'].max(), 100)
ax.plot(xline, m * xline + b, color=RUST, linewidth=1.8, linestyle='--',
        label=f'Trend line  (r = {r:.2f})')

for country in df.index:
    if df.loc[country, 'co2_per_capita'] > 3.5 or df.loc[country, 'gdp_per_capita'] > 60000:
        ax.annotate(country,
                    (df.loc[country, 'gdp_per_capita'], df.loc[country, 'co2_per_capita']),
                    textcoords='offset points', xytext=(5, 3), fontsize=8, color='#444')

ax.set_xlabel('GDP per Capita (EUR, Purchasing Power Standards)')
ax.set_ylabel('Transport CO₂ per Capita (tonnes)')
ax.set_title('Figure 2. Relationship Between GDP per Capita and Transport CO₂ per Capita (2022)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT + 'fig2_gdp_vs_co2.png')
plt.close()
print("fig2 saved")

# ── 5. CORRELATION (McKinney Ch. 5.3) ────────────────────────────────────────
# McKinney (2022), p. 168: "DataFrame.corr() computes pairwise correlation
#  of columns, excluding NA/null values."
print("\n=== Correlation (df.corr()) — McKinney Ch. 5.3 ===")
corr_df = df.corr(numeric_only=True)         # pandas built-in Pearson correlation
print(corr_df.round(3))

# individual p-values using scipy
print("\n--- Pearson r and p-values vs co2_per_capita ---")
for col in ['gdp_per_capita','ev_per_capita','freight_tkm','pop_density','passenger_km']:
    sub = df[['co2_per_capita', col]].dropna()   # McKinney Ch. 7: dropna()
    r_val, p_val = stats.pearsonr(sub['co2_per_capita'], sub[col])
    print(f"  {col:25s}  r={r_val:+.3f}  p={p_val:.4f}")

# ── 6. FIGURE 3 — Correlation heatmap (McKinney Ch. 9) ───────────────────────
labels = ['CO₂\nper capita', 'GDP\nper capita', 'EV\nper capita',
          'Pop.\ndensity', 'Passenger\nkm', 'Freight\ntkm']
# Use the pandas corr matrix values
cols_order = ['co2_per_capita','gdp_per_capita','ev_per_capita',
              'pop_density','passenger_km','freight_tkm']
corr_matrix = df[cols_order].corr(numeric_only=True).values  # McKinney Ch. 5.3

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr_matrix, cmap=plt.cm.RdYlGn, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(6):
    for j in range(6):
        val = corr_matrix[i, j]
        textcol = 'black' if abs(val) < 0.6 else 'white'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color=textcol, fontweight='bold')
ax.set_title('Figure 3. Correlation Matrix Between All Variables (2022)', pad=12)
plt.tight_layout()
plt.savefig(OUT + 'fig3_correlation_heatmap.png')
plt.close()
print("fig3 saved")

# ── 7. OLS REGRESSION (McKinney p. 8 / Ch. 12 companion — statsmodels) ───────
# McKinney (2022), p. 8: "statsmodels is a package for fitting many kinds of
#  statistical models, performing statistical tests, and data exploration."
# Following McKinney's import convention: import statsmodels.api as sm

# Drop rows with missing freight data — McKinney Ch. 7: DataFrame.dropna()
df_reg = df[['co2_per_capita','gdp_per_capita','pop_density',
             'freight_tkm','ev_per_capita']].dropna()
print(f"\n=== OLS Regression — n={len(df_reg)} (McKinney p.8 / statsmodels) ===")

# Add constant (intercept) — standard statsmodels pattern
X = sm.add_constant(df_reg[['gdp_per_capita','pop_density',
                              'freight_tkm','ev_per_capita']])
y = df_reg['co2_per_capita']
model = sm.OLS(y, X).fit()
print(model.summary())

fitted = model.fittedvalues
resid  = model.resid

# ── 8. FIGURE 4 — Regression diagnostics (McKinney Ch. 9) ────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.scatter(fitted, resid, color=TEAL, s=45, edgecolors='white', linewidth=0.4)
ax1.axhline(0, color=RUST, linewidth=1.5, linestyle='--')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs. Fitted Values')

(osm, osr), (slope, intercept, _) = stats.probplot(resid)
ax2.scatter(osm, osr, color=TEAL, s=45, edgecolors='white', linewidth=0.4)
ax2.plot(osm, slope * np.array(osm) + intercept,
         color=RUST, linewidth=1.5, linestyle='--')
ax2.set_xlabel('Theoretical Quantiles')
ax2.set_ylabel('Sample Quantiles')
ax2.set_title('Normal Q-Q Plot of Residuals')
fig.suptitle('Figure 4. Regression Diagnostic Plots', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig(OUT + 'fig4_regression_diagnostics.png')
plt.close()
print("fig4 saved")

# ── 9. CLUSTER ANALYSIS (McKinney Ch. 5.2 — StandardScaler + KMeans) ─────────
# Standardise using z-scores — McKinney (2022), p. 158–160:
# "apply() applies a function along an axis of a DataFrame"
cluster_vars = ['co2_per_capita','gdp_per_capita','ev_per_capita','pop_density']
df_cl = df[cluster_vars].dropna()

# StandardScaler: equivalent to (x - mean) / std, as discussed in McKinney Ch. 5.3
scaler = StandardScaler()
X_sc = scaler.fit_transform(df_cl)

# ── FIGURE 5 — Elbow plot (McKinney Ch. 9) ────────────────────────────────────
inertias = []
K_range = range(1, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(K_range), inertias, marker='o', color=TEAL, linewidth=2, markersize=7)
ax.axvline(x=3, color=RUST, linewidth=1.5, linestyle='--', label='Chosen k = 3')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Within-Cluster Sum of Squares')
ax.set_title('Figure 5. Elbow Method — Choosing the Number of Clusters')
ax.legend(fontsize=10)
ax.set_xticks(list(K_range))
plt.tight_layout()
plt.savefig(OUT + 'fig5_elbow.png')
plt.close()
print("fig5 saved")

# K-means with k=3
km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
df_cl = df_cl.copy()
df_cl['cluster'] = km3.fit_predict(X_sc)

# Relabel so Cluster A = highest mean CO2 (McKinney Ch. 10: groupby)
# McKinney (2022), p. 291: "groupBy — split-apply-combine"
cluster_means = df_cl.groupby('cluster')['co2_per_capita'].mean().sort_values(ascending=False)
label_map = {cluster_means.index[0]: 'A',
             cluster_means.index[1]: 'B',
             cluster_means.index[2]: 'C'}
df_cl['cluster_label'] = df_cl['cluster'].map(label_map)

print("\n=== Cluster means (groupby — McKinney Ch. 10) ===")
print(df_cl.groupby('cluster_label')[cluster_vars].mean().round(3))
print("\nCluster members:")
for lab in ['A','B','C']:
    members = df_cl[df_cl['cluster_label'] == lab].index.tolist()
    print(f"  Cluster {lab}: {members}")

# ── FIGURE 6 — Cluster scatter (McKinney Ch. 9) ───────────────────────────────
palette = {'A': RUST, 'B': TEAL, 'C': GOLD}
markers = {'A': 'o', 'B': 's', 'C': '^'}

# Re-join with full df to get gdp for plotting
df_cl_plot = df_cl.join(df[['gdp_per_capita']], how='left', rsuffix='_orig')

fig, ax = plt.subplots(figsize=(8, 5))
for lab, grp in df_cl_plot.groupby('cluster_label'):
    ax.scatter(grp['gdp_per_capita'], grp['co2_per_capita'],
               color=palette[lab], marker=markers[lab],
               s=70, edgecolors='white', linewidth=0.5,
               label=f'Cluster {lab}', zorder=3)
    for country in grp.index:
        if grp.loc[country, 'co2_per_capita'] > 3.5 or grp.loc[country, 'gdp_per_capita'] > 55000:
            ax.annotate(country,
                        (grp.loc[country, 'gdp_per_capita'], grp.loc[country, 'co2_per_capita']),
                        textcoords='offset points', xytext=(5, 3), fontsize=8)

ax.set_xlabel('GDP per Capita (EUR PPS)')
ax.set_ylabel('Transport CO₂ per Capita (tonnes)')
ax.set_title('Figure 6. Country Clusters by GDP and Transport CO₂ per Capita (2022)')
ax.legend(title='Cluster', fontsize=10)
plt.tight_layout()
plt.savefig(OUT + 'fig6_clusters.png')
plt.close()
print("fig6 saved")

# ── 10. TIME SERIES (McKinney Ch. 11) ─────────────────────────────────────────
# McKinney (2022), Ch. 11: "pandas contains extensive capabilities and features
#  for working with time series data."
# Build a time-indexed Series for EU27 aggregate (McKinney Ch. 11.1)

years = pd.Index([2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023], name='year')
eu27_vals = pd.Series(
    [0.430, 0.446, 0.440, 0.477, 0.367, 0.410, 0.445, 0.450],
    index=years, name='EU27_transport_CO2_Gt'
)
print("\n=== EU27 time series (McKinney Ch. 11) ===")
print(eu27_vals)
print(f"Net change 2016→2022: {((eu27_vals[2022]-eu27_vals[2016])/eu27_vals[2016]*100):.1f}%")

# ── FIGURE 7 — EU27 time series line plot (McKinney Ch. 9) ────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(eu27_vals.index, eu27_vals.values,
        marker='o', color=TEAL, linewidth=2.5, markersize=8)
ax.fill_between(eu27_vals.index, eu27_vals.values, alpha=0.12, color=TEAL)
ax.axvspan(2019.5, 2020.5, alpha=0.12, color=RUST, label='COVID-19 drop (2020)')
for x, y in zip(eu27_vals.index, eu27_vals.values):
    ax.text(x, y + 0.008, f'{y:.3f}', ha='center', fontsize=9, color=DARK)
ax.set_xlabel('Year')
ax.set_ylabel('Emissions (billion tonnes CO₂eq)')
ax.set_title('Figure 7. EU27 Transport-Related CO₂ Emissions, 2016–2023')
ax.set_xticks(years)
ax.set_ylim(0.30, 0.55)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUT + 'fig7_eu27_timeseries.png')
plt.close()
print("fig7 saved")

# ── Country-level % changes — pandas Series operations (McKinney Ch. 5.2) ────
pct_changes = pd.Series({
    'Romania': -5.0, 'Bulgaria': -29.5, 'Germany': -26.5, 'Finland': -23.1,
    'Sweden': -16.8, 'Estonia': -14.2, 'Portugal': -10.3, 'Italy': -8.5,
    'Greece': -6.2, 'Slovenia': -5.8, 'Hungary': -4.1, 'Croatia': -2.3,
    'Slovakia': -1.7, 'France': 2.1, 'Austria': 3.8, 'Spain': 4.4,
    'Czechia': 5.0, 'Latvia': 7.6, 'Belgium': 9.2, 'Netherlands': 10.1,
    'Ireland': 13.0, 'Luxembourg': 16.4, 'Denmark': 18.0, 'Cyprus': 22.0,
    'Lithuania': 128.5, 'Poland': 153.2, 'Malta': 156.0,
}, name='pct_change_2016_2022')

# sort_values — McKinney (2022), p. 160
pct_changes_sorted = pct_changes.sort_values()
print("\n=== Country % changes 2016→2022 (sort_values — McKinney Ch. 5.2) ===")
print(pct_changes_sorted)

# ── FIGURE 8 — % change bar chart (McKinney Ch. 9) ────────────────────────────
colors_ch = [TEAL if v < 0 else RUST for v in pct_changes_sorted.values]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(pct_changes_sorted.index, pct_changes_sorted.values,
               color=colors_ch, edgecolor='white', height=0.7)
ax.axvline(0, color='#555', linewidth=0.8)
ax.set_xlabel('Percentage Change (%)')
ax.set_title('Figure 8. Change in Transport CO₂ Emissions by Country, 2016–2022 (%)')
for bar, val in zip(bars, pct_changes_sorted.values):
    if abs(val) > 20:
        ax.text(val + (3 if val > 0 else -3), bar.get_y() + bar.get_height()/2,
                f'{val:+.0f}%', va='center', fontsize=8,
                color=RUST if val > 0 else TEAL, fontweight='bold')
decrease_patch = mpatches.Patch(color=TEAL, label='Decrease')
increase_patch = mpatches.Patch(color=RUST, label='Increase')
ax.legend(handles=[decrease_patch, increase_patch], fontsize=10)
plt.tight_layout()
plt.savefig(OUT + 'fig8_country_change_2016_2022.png')
plt.close()
print("fig8 saved")

# ── Selected countries time series — DataFrame (McKinney Ch. 5 & 11) ──────────
# Build a proper time-series DataFrame indexed by year (McKinney Ch. 11.1)
ts_data = {
    'Germany':   [0.75, 0.73, 0.72, 0.75, 0.57, 0.60, 0.55, 0.56],
    'Poland':    [0.39, 0.44, 0.47, 0.50, 0.40, 0.48, 0.97, 1.00],
    'Sweden':    [0.53, 0.52, 0.51, 0.53, 0.42, 0.44, 0.44, 0.45],
    'Lithuania': [0.72, 0.85, 0.95, 1.05, 0.80, 0.98, 1.38, 1.44],
    'Romania':   [0.28, 0.28, 0.29, 0.31, 0.25, 0.29, 0.32, 0.33],
    'Ireland':   [3.60, 3.70, 3.80, 3.95, 3.00, 3.40, 4.53, 4.60],
}
ts_df = pd.DataFrame(ts_data, index=years)   # McKinney Ch. 11.1
print("\n=== Selected countries time-series DataFrame (McKinney Ch. 11) ===")
print(ts_df)

# Linear trend test using scipy (slope, p-value) for each country
print("\n--- OLS trend slopes per country ---")
for col in ts_df.columns:
    slope, intercept, r, p_val, _ = stats.linregress(ts_df.index, ts_df[col])
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    print(f"  {col:12s}  slope={slope:+.3f} t/yr  p={p_val:.4f} {sig}")

# ── FIGURE 9 — Selected countries line chart (McKinney Ch. 9) ─────────────────
line_colors = [TEAL, RUST, DARK, GOLD, GRAY, '#7A39BB']
line_styles  = ['-', '--', '-.', ':', '-', '--']

fig, ax = plt.subplots(figsize=(9, 5))
for (col, col_vals), col_col, ls in zip(ts_df.items(), line_colors, line_styles):
    ax.plot(ts_df.index, col_vals, label=col,
            color=col_col, linewidth=2, linestyle=ls, marker='o', markersize=5)
ax.set_xlabel('Year')
ax.set_ylabel('Transport CO₂ per Capita (tonnes)')
ax.set_title('Figure 9. Transport CO₂ per Capita for Selected EU Countries, 2016–2023')
ax.set_xticks(list(years))
ax.legend(fontsize=9, loc='upper left', framealpha=0.8)
ax.axvspan(2019.5, 2020.5, alpha=0.10, color='gray')
plt.tight_layout()
plt.savefig(OUT + 'fig9_selected_countries_timeseries.png')
plt.close()
print("fig9 saved")

print("\n✓ All 9 figures saved to:", OUT)
print("✓ Analysis complete — methods follow McKinney (2022)")
