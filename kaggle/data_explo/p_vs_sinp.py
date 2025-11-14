
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------------------------
# Load data
# -------------------------
df = pd.read_csv("./df_imputed.csv")

# Quick cleaning: ensure Price numeric and drop NA prices
df = df.copy()
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price', 'Time_of_Day']).reset_index(drop=True)

# Normalize/clean Time_of_Day categories and set order if appropriate
# Adjust the list below to match your dataset's labels if needed.
order = ["morning", "afternoon", "evening"]
df['Time_of_Day'] = df['Time_of_Day'].astype(str).str.lower().str.strip()
df = df[df['Time_of_Day'].isin(order)].copy()
df['Time_of_Day'] = pd.Categorical(df['Time_of_Day'], categories=order, ordered=True)

# -------------------------
# Create log price to reduce skew and mitigate outliers
# -------------------------
df['Price_log'] = np.log1p(df['Price'])   # log(1 + price)

# -------------------------
# Basic group stats (raw and log)
# -------------------------
group_stats_raw = df.groupby('Time_of_Day')['Price'].agg(
    count='count', mean='mean', std='std', min='min', max='max', median='median'
)

group_stats_log = df.groupby('Time_of_Day')['Price_log'].agg(
    count='count', mean='mean', std='std', min='min', max='max', median='median'
)

# -------------------------
# Correlation with Price_Sine (if available)
# -------------------------
sine_corr = None
if 'Price_Sine' in df.columns:
    df['Price_Sine'] = pd.to_numeric(df['Price_Sine'], errors='coerce')
    sine_corr = df[['Price','Price_Sine']].dropna()['Price'].corr(df[['Price','Price_Sine']].dropna()['Price_Sine'])

# -------------------------
# Parametric ANOVA on raw and log price
# -------------------------
anova_p_raw = None
anova_p_log = None
anova_results_log = None

groups_raw = [g['Price'].values for n,g in df.groupby('Time_of_Day')]
groups_log = [g['Price_log'].values for n,g in df.groupby('Time_of_Day')]

if len(groups_raw) > 1:
    try:
        _, anova_p_raw = f_oneway(*groups_raw)
    except Exception as e:
        anova_p_raw = None

if len(groups_log) > 1:
    try:
        anova_stat_log, anova_p_log = f_oneway(*groups_log)
    except Exception as e:
        anova_p_log = None

# Also compute ANOVA table (sum of squares / eta-sq) with statsmodels on log-price
try:
    model = ols('Price_log ~ C(Time_of_Day)', data=df).fit()
    anova_results_log = sm.stats.anova_lm(model, typ=2)  # Type II
    ss_between = anova_results_log.loc['C(Time_of_Day)', 'sum_sq']
    ss_total = ss_between + anova_results_log.loc['Residual', 'sum_sq']
    eta_sq = ss_between / ss_total
except Exception:
    anova_results_log = None
    eta_sq = None

# -------------------------
# Non-parametric Kruskal-Wallis (robust to non-normality)
# -------------------------
kw_p = None
try:
    kw_stat, kw_p = kruskal(*groups_raw)
except Exception:
    kw_p = None

# -------------------------
# Tukey HSD post-hoc on log-price (if ANOVA suggests anything)
# -------------------------
tukey = None
try:
    tukey = pairwise_tukeyhsd(endog=df['Price_log'], groups=df['Time_of_Day'], alpha=0.05)
except Exception:
    tukey = None

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(9,5))
sns.boxplot(data=df, x='Time_of_Day', y='Price_log')
plt.title('Boxplot of log(1+Price) by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('log(1 + Price)')
plt.show()

plt.figure(figsize=(9,5))
sns.violinplot(data=df, x='Time_of_Day', y='Price_log', inner='quart')
plt.title('Violin plot of log(1+Price) by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('log(1 + Price)')
plt.show()

plt.figure(figsize=(9,5))
# mean and 95% CI suppressed by using errorbar=None and showing raw means
means = df.groupby('Time_of_Day')['Price_log'].mean().reindex(order)
counts = df.groupby('Time_of_Day')['Price_log'].count().reindex(order)
plt.errorbar(x=range(len(means)), y=means, yerr=None, fmt='o', capsize=0)
plt.xticks(range(len(means)), means.index)
plt.title('Mean log(1+Price) by Time of Day (no error bars shown)')
plt.ylabel('mean log(1 + Price)')
plt.show()

# -------------------------
# Print numerical summary
# -------------------------
print("\n" + "="*70)
print(" NUMERICAL SUMMARY FOR CYCLICALITY OF PRICE ")
print("="*70)
print("\nRaw Price group statistics:")
print(group_stats_raw.round(4))

print("\nLog(1+Price) group statistics:")
print(group_stats_log.round(4))

if sine_corr is not None:
    print(f"\nCorrelation between raw Price and Price_Sine: {sine_corr:.4f}")
else:
    print("\nPrice_Sine column not found or has NaNs; skipping correlation.")

print("\nANOVA on raw Price (one-way f-test):")
print(f"  p-value = {anova_p_raw if anova_p_raw is not None else 'n/a'}")

print("\nANOVA on log(1+Price) (one-way f-test):")
print(f"  p-value = {anova_p_log if anova_p_log is not None else 'n/a'}")

if anova_results_log is not None:
    print("\nANOVA table (log price) [Type II]:")
    print(anova_results_log.round(6))
    if eta_sq is not None:
        print(f"\nEstimated effect size (eta-squared) for Time_of_Day on log-price: {eta_sq:.6f}")
else:
    print("\nANOVA table not computed.")

print("\nKruskal-Wallis test (raw Price) — non-parametric:")
print(f"  p-value = {kw_p if kw_p is not None else 'n/a'}")

if tukey is not None:
    print("\nTukey HSD post-hoc (on log price):")
    print(tukey.summary())
else:
    print("\nTukey HSD not available or could not be computed.")

print("\nSUGGESTED INTERPRETATION / NEXT STEPS (automated text):")
# Automated interpretation hints
# Note: these are general — adapt to your actual p-values printed above.
if anova_p_log is not None:
    if anova_p_log < 0.05:
        print(" - ANOVA on log-price: p < 0.05 → there is a statistically detectable difference in average price across Time_of_Day (on the log scale).")
        if eta_sq is not None:
            print(f" - Effect size (eta^2) = {eta_sq:.4f}  (small ~0.01, medium ~0.06, large ~0.14)")
    else:
        print(" - ANOVA on log-price: p >= 0.05 → no statistically significant difference in mean log-price across Time_of_Day.")
else:
    print(" - ANOVA on log-price not computed.")

if kw_p is not None:
    if kw_p < 0.05:
        print(" - Kruskal-Wallis: p < 0.05 → distribution of raw prices differs between at least two Time_of_Day groups (robust test).")
    else:
        print(" - Kruskal-Wallis: p >= 0.05 → no evidence of distributional differences in raw Price between Time_of_Day groups.")

print("\nRecommended next steps:")
print("  * If you have extreme outliers / high max values (which you likely do), prefer analysis on log-price or after winsorizing.")
print("  * If no difference found, consider interactions (Price ~ Time_of_Day * Category) or per-category tests.")
print("  * If you expect a continuous cyclic variable (hour-of-day), create sin/cos encodings of hour and test regression with them.")
print("  * Consider removing or capping extreme outliers before reporting raw means.")
print("="*70)
