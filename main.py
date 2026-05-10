# ============================================================
# IMPORTS
# ============================================================
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson


# ============================================================
# CONFIG
# Sets up output directories and defines shared helper functions
# used throughout the script for saving text files and formatting
# model results.
# ============================================================
os.makedirs('figures', exist_ok=True)
os.makedirs('output', exist_ok=True)

def save_text(content, filename):
    # Writes content to output/<filename> and prints it to console.
    path = os.path.join('output', filename)
    with open(path, 'w') as f:
        f.write(content)
    print(content)

def significance_stars(p):
    # Returns significance stars based on p-value thresholds.
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''

def format_coef(coef, se, p):
    # Returns a two-line string: "coef*** \n  (se)" for the final results table.
    if pd.isna(coef): return ''
    return f"{coef:+.4f}{significance_stars(p)}\n  ({se:.4f})"

def compute_nw_lags(n):
    # Newey-West lag length for quarterly data: L = floor(4 * (n/100)^(2/9))
    return math.floor(4 * (n / 100) ** (2 / 9))

def save_model_summary(model, label, filename):
    # Writes a labeled model summary block to output/<filename>.
    content = "\n".join([
        "=" * 75,
        label,
        "=" * 75,
        str(model.summary()),
    ])
    save_text(content, filename)


# ============================================================
# DATA LOADING AND PREPARATION
# Loads the raw Excel data, computes the log-transformed dependent
# variable, and defines the variable lists used across all sections.
#
# Outputs:
#   output/summary_statistics.txt — descriptive stats for all analysis variables
# ============================================================
df = pd.read_excel('data/Econ318_RawData_Paper.xlsx')
df['log_Delinquency'] = np.log(df['Delinquency'])

Y_var = 'Delinquency'           # levels DV (Model 1)
Y_var_log = 'log_Delinquency'   # log DV (Models 2-4)
X_vars = ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'GDPGrowth', 'D_Crisis', 'D_Covid']
vars_no_dummy = [Y_var] + ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'GDPGrowth']

# Summary statistics for all analysis variables
summary_vars = ['Delinquency', 'log_Delinquency', 'UNRATE', 'MORTGAGE30US',
                'HPI_Growth', 'GDPGrowth', 'D_Crisis', 'D_Covid']
summary_stats = df[summary_vars].describe(percentiles=[0.25, 0.5, 0.75]).T
summary_stats = summary_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
summary_stats.columns = ['N', 'Mean', 'Std Dev', 'Min', 'P25', 'Median', 'P75', 'Max']
summary_stats = summary_stats.round(3)
summary_text = "\n".join([
    "=" * 75,
    "SUMMARY STATISTICS",
    "=" * 75,
    summary_stats.to_string(),
])
save_text(summary_text, 'summary_statistics.txt')


# ============================================================
# EXPLORATORY VISUALIZATIONS
# Produces three exploratory figures to inspect distributions,
# time-series patterns, and pairwise correlations among variables.
#
# Outputs:
#   figures/00_scatter_plots.png        — 2x3 grid of X vs Delinquency scatter plots
#   figures/01_dv_histogram_comparison.png — levels vs log DV histograms (paper figure)
#   figures/01_histograms.png           — 2x2 histograms of explanatory variables only
#   figures/02_time_series.png          — time series with crisis/covid shading
#   figures/03_correlation_heatmap.png  — heatmap of X-variable correlations
#   output/correlation_matrix.txt       — correlation matrix (numeric)
# ============================================================

# --- Scatter plots: all six X variables vs Delinquency in a 2x3 grid ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for i, var in enumerate(X_vars):
    axes[i].scatter(df[var], df[Y_var])
    axes[i].set_title(f'{var} vs {Y_var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel(Y_var)
plt.tight_layout()
plt.savefig('figures/00_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Dependent variable comparison: raw vs log transformation (paper figure) ---
# Justifies the log transformation by comparing the two DV distributions.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df['Delinquency'], bins=40)
axes[0].set_title('Delinquency (Raw)')
axes[0].set_xlabel('Delinquency Rate (%)')
axes[0].set_ylabel('Frequency')
axes[1].hist(df['log_Delinquency'], bins=40)
axes[1].set_title('log(Delinquency)')
axes[1].set_xlabel('log(Delinquency Rate)')
axes[1].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('figures/01_dv_histogram_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Histograms of explanatory variables (documentation, not in paper) ---
# Uses continuous_X_vars (excludes dummies and the DV) for a clean 2x2 grid.
continuous_X_vars = ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'GDPGrowth']
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()
for i, var in enumerate(continuous_X_vars):
    axes[i].hist(df[var], bins=40)
    axes[i].set_title(f'{var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('figures/01_histograms.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Time series ---
units = {
    'Delinquency':  'Percent',
    'UNRATE':       'Percent',
    'MORTGAGE30US': 'Percent',
    'HPI_Growth':   'YoY % change',
    'GDPGrowth':    'Annualized %',
}

crisis_start = pd.Timestamp('2007-07-01')
crisis_end   = pd.Timestamp('2010-12-31')
covid_start  = pd.Timestamp('2020-01-01')
covid_end    = pd.Timestamp('2021-09-30')

fig, axes = plt.subplots(3, 2, figsize=(13, 10))
axes = axes.flatten()
for i, var in enumerate(vars_no_dummy):
    axes[i].plot(df['observation_date'], df[var])
    axes[i].axvspan(crisis_start, crisis_end, alpha=0.6)   # 2007-2010 financial crisis
    axes[i].axvspan(covid_start, covid_end, alpha=0.6)     # 2020-2021 COVID period
axes[5].axis('off')
plt.suptitle('Time Series: 1991Q1 – 2025Q4')
plt.tight_layout()
plt.savefig('figures/02_time_series.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Correlation matrix ---
corr_matrix = df[X_vars].corr().round(3)
save_text(corr_matrix.to_string(), 'correlation_matrix.txt')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# DIAGNOSTICS — VIF
# Checks for multicollinearity among regressors by computing the
# Variance Inflation Factor for each X variable. VIF >= 5 is
# flagged as a concern (user-defined threshold).
#
# Outputs:
#   output/vif_table.txt — VIF values and severity flags per variable
# ============================================================
X_vif = sm.add_constant(df[X_vars])

vif_data = pd.DataFrame({
    'Variable': X_vif.columns[1:],
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
})
vif_data['VIF'] = vif_data['VIF'].round(3)

def flag(v):
    if v >= 10: return 'SEVERE (≥10)'
    elif v >= 5: return 'SEVERE (≥5, user threshold)'
    else: return 'OK (<5)'
vif_data['Flag'] = vif_data['VIF'].apply(flag)

vif_text = "\n".join([
    "=" * 55,
    "VARIANCE INFLATION FACTORS",
    "Threshold: VIF ≥ 5 flagged as concern",
    "=" * 55,
    vif_data.to_string(index=False),
])
save_text(vif_text, 'vif_table.txt')


# ============================================================
# MODEL FITTING
# Fits all six regression models used in the paper. Each model is
# stored in a named variable and reused throughout the remaining
# sections — no model is re-fit later.
#
# Models:
#   model_linear  — OLS, levels DV, full regressors (Model 1)
#   model_ols     — OLS, log DV, classical SEs (Model 2 baseline)
#   model_hc3     — OLS, log DV, HC3 heteroscedasticity-robust SEs
#   model_nw      — OLS, log DV, Newey-West HAC SEs, full regressors (Model 3)
#   model_co      — Cochrane-Orcutt GLS, log DV, corrects autocorrelation
#   model_nw_trim — OLS, log DV, Newey-West HAC SEs, trimmed regressors (Model 4)
# ============================================================
X = sm.add_constant(df[X_vars])
y = df[Y_var_log]
nw_lags = compute_nw_lags(len(df))   # lag length for all Newey-West models

regressors_trimmed = ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'D_Covid']
X_trim = sm.add_constant(df[regressors_trimmed])

model_linear = sm.OLS(df[Y_var], X).fit()
model_ols    = sm.OLS(y, X).fit()
model_hc3    = sm.OLS(y, X).fit(cov_type='HC3')
model_nw     = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})

# Cochrane-Orcutt: iteratively estimates rho, quasi-differences the data,
# and re-fits OLS until rho converges. Corrects for AR(1) autocorrelation.
rho_hat = model_ols.resid.autocorr(lag=1)
co_lines = [
    "=" * 60,
    "COCHRANE-ORCUTT ITERATIVE PROCEDURE",
    "=" * 60,
    f"\n  Step 1: Estimated ρ from residuals:  ρ̂ = {rho_hat:.4f}",
]
y_qd = (y - rho_hat * y.shift(1)).dropna()
X_qd = (X - rho_hat * X.shift(1)).dropna()
co_lines.append(f"\n  Step 2: Quasi-differenced data has {len(y_qd)} observations (one dropped)")
model_co = sm.OLS(y_qd, X_qd).fit()
co_lines.append(f"\n  Step 3: OLS on quasi-differenced data complete")

max_iter = 10
tol = 1e-6
for it in range(max_iter):
    rho_prev = rho_hat
    residuals_original = y - X @ model_co.params
    rho_hat = residuals_original.autocorr(lag=1)
    y_qd = (y - rho_hat * y.shift(1)).dropna()
    X_qd = (X - rho_hat * X.shift(1)).dropna()
    model_co = sm.OLS(y_qd, X_qd).fit()
    if abs(rho_hat - rho_prev) < tol:
        co_lines.append(f"\n  Converged after {it+1} iterations. Final ρ̂ = {rho_hat:.4f}")
        break
else:
    co_lines.append(f"\n  Reached max {max_iter} iterations. Final ρ̂ = {rho_hat:.4f}")

model_nw_trim = sm.OLS(y, X_trim).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})

model4_text = "\n".join([
    "=" * 75,
    f"MODEL 4 — LOG-LIN, NEWEY-WEST HAC, TRIMMED (maxlags={nw_lags})",
    "Excludes GDPGrowth and D_Crisis based on joint F-test (F = 0.434, p = 0.649)",
    "=" * 75,
    str(model_nw_trim.summary()),
])
save_text(model4_text, 'model4_nw_trim_summary.txt')


# ============================================================
# RESIDUAL DIAGNOSTICS
# Checks the residuals of the baseline OLS model (model_ols) for
# violations of classical assumptions: non-zero mean, non-normality,
# and autocorrelation patterns.
#
# Outputs:
#   figures/04_residual_comparison.png       — residuals vs fitted, linear vs log-lin
#   figures/05_qq_comparison.png             — QQ plots, linear vs log-lin
#   figures/06_autocorrelation_diagnostics.png — residuals over time +
#                                               residual vs lagged residual
# ============================================================

# Residuals vs fitted: side-by-side comparison of linear and log-linear specs.
# Checks for heteroscedasticity and non-linearity in each specification.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(model_linear.fittedvalues, model_linear.resid)
axes[0].axhline(0, linestyle='--')
axes[0].set_title('Linear Specification: Residuals vs Fitted')
axes[0].set_xlabel('Fitted Delinquency')
axes[0].set_ylabel('Residuals')
axes[1].scatter(model_ols.fittedvalues, model_ols.resid)
axes[1].axhline(0, linestyle='--')
axes[1].set_title('Log-Linear Specification: Residuals vs Fitted')
axes[1].set_xlabel('Fitted log(Delinquency)')
axes[1].set_ylabel('Residuals')
plt.tight_layout()
plt.savefig('figures/04_residual_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# QQ plots: checks whether residuals follow a normal distribution in each spec.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sm.qqplot(model_linear.resid, line='s', ax=axes[0])
axes[0].set_title('Linear Specification: QQ Plot')
sm.qqplot(model_ols.resid, line='s', ax=axes[1])
axes[1].set_title('Log-Linear Specification: QQ Plot')
plt.tight_layout()
plt.savefig('figures/05_qq_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Left panel: residuals over time — visual check for autocorrelation
# Right panel: residual vs lagged residual — slope ≈ AR(1) coefficient
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(df['observation_date'], model_ols.resid)
axes[0].axhline(0, linestyle='--')
axes[0].set_title('Residuals Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residual')
axes[0].grid(True)

resid_lag = model_ols.resid.shift(1)
axes[1].scatter(resid_lag, model_ols.resid, alpha=0.4)
axes[1].axhline(0, linestyle=':')
axes[1].axvline(0, linestyle=':')
rho = model_ols.resid.autocorr(lag=1)
axes[1].set_title(f'Residual vs. Lagged Residual (ρ̂ = {rho:.3f})')
axes[1].set_xlabel('Residual at t-1')
axes[1].set_ylabel('Residual at t')

plt.tight_layout()
plt.savefig('figures/06_autocorrelation_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# STATISTICAL TESTS
# Runs formal hypothesis tests for heteroscedasticity, autocorrelation,
# and joint significance of regressor groups. All tests use model_ols
# residuals; F-tests use model_nw (Newey-West) for robust inference.
#
# Outputs:
#   output/heteroscedasticity_tests.txt — Breusch-Pagan and White test results,
#                                         plus classical vs HC3 SE comparison
#   output/autocorrelation_tests.txt    — Durbin-Watson and Breusch-Godfrey
#                                         LM test results
#   output/stage9_ftests.txt            — three joint Wald F-tests under HAC SEs
# ============================================================

# --- Heteroscedasticity tests ---
# Breusch-Pagan and White tests both test H0: homoscedasticity.
# The SE comparison table shows how much SEs shift when switching to HC3.
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model_ols.resid, X)
w_stat, w_p, w_f, w_fp = het_white(model_ols.resid, X)

se_comparison = pd.DataFrame({
    'Coefficient': model_ols.params.round(4),
    'Classical SE': model_ols.bse.round(4),
    'Classical p': model_ols.pvalues.round(4),
    'HC3 Robust SE': model_hc3.bse.round(4),
    'HC3 Robust p': model_hc3.pvalues.round(4),
})
se_comparison['SE Change (%)'] = (100 * (model_hc3.bse - model_ols.bse) / model_ols.bse).round(1)

hetero_text = "\n".join([
    "=" * 60,
    "BREUSCH-PAGAN-GODFREY TEST",
    "H0: homoscedasticity",
    "=" * 60,
    f"  LM statistic:       {bp_stat:.4f}",
    f"  LM p-value:         {bp_p:.4f}",
    f"  F-statistic:        {bp_f:.4f}",
    f"  F p-value:          {bp_fp:.4f}",
    f"  Conclusion: {'REJECT H0 - heteroscedasticity present' if bp_p < 0.05 else 'DO NOT REJECT H0'}",
    "",
    "=" * 60,
    "WHITE TEST",
    "H0: homoscedasticity",
    "=" * 60,
    f"  LM statistic:       {w_stat:.4f}",
    f"  LM p-value:         {w_p:.4f}",
    f"  F-statistic:        {w_f:.4f}",
    f"  F p-value:          {w_fp:.4f}",
    f"  Conclusion: {'REJECT H0 - heteroscedasticity present' if w_p < 0.05 else 'DO NOT REJECT H0'}",
    "",
    "=" * 80,
    "COMPARISON: CLASSICAL vs. HC3-ROBUST STANDARD ERRORS",
    "=" * 80,
    se_comparison.to_string(),
])
save_text(hetero_text, 'heteroscedasticity_tests.txt')

# --- Autocorrelation tests ---
# Durbin-Watson detects first-order autocorrelation; Breusch-Godfrey
# extends the test up to 4 lags (one year of quarterly data).
dw = durbin_watson(model_ols.resid)
bg_stat, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(model_ols, nlags=4)

autocorr_text = "\n".join([
    "=" * 60,
    "DURBIN-WATSON TEST (H0: no 1st-order autocorrelation)",
    "=" * 60,
    f"  DW statistic: {dw:.4f}",
    f"  Interpretation:",
    f"    DW ≈ 2:   no autocorrelation",
    f"    DW < 1.5: positive autocorrelation likely",
    f"    DW > 2.5: negative autocorrelation likely",
    "",
    "=" * 60,
    "BREUSCH-GODFREY LM TEST (H0: no autocorrelation up to lag 4)",
    "=" * 60,
    f"  LM statistic:  {bg_stat:.4f}",
    f"  LM p-value:    {bg_p:.6f}",
    f"  F-statistic:   {bg_f:.4f}",
    f"  F p-value:     {bg_fp:.6f}",
])
save_text(autocorr_text, 'autocorrelation_tests.txt')

# --- Joint F-tests (Newey-West HAC) ---
# Three Wald tests on model_nw to assess whether regressor groups
# can be excluded. Results inform the trimmed specification (Model 4).
f1 = model_nw.f_test("D_Crisis = 0, D_Covid = 0")
f2 = model_nw.f_test("UNRATE = 0, MORTGAGE30US = 0, HPI_Growth = 0, GDPGrowth = 0")
f3 = model_nw.f_test("GDPGrowth = 0, D_Crisis = 0")

ftests_text = "\n".join([
    "=" * 70,
    "STAGE 9 — JOINT F-TESTS (Newey-West HAC, primary model)",
    "=" * 70,
    "",
    "F-TEST 1: Are the dummy variables jointly significant?",
    "H0: beta(D_Crisis) = beta(D_Covid) = 0",
    f"  F-statistic:  {float(f1.fvalue):.4f}",
    f"  p-value:      {float(f1.pvalue):.6f}",
    f"  df numerator: {int(f1.df_num)}",
    f"  Conclusion:   {'REJECT H0 — dummies jointly significant' if float(f1.pvalue) < 0.05 else 'FAIL TO REJECT H0'}",
    "",
    "F-TEST 2: Are the macro variables jointly significant?",
    "H0: beta(UNRATE) = beta(MORTGAGE30US) = beta(HPI_Growth) = beta(GDPGrowth) = 0",
    f"  F-statistic:  {float(f2.fvalue):.4f}",
    f"  p-value:      {float(f2.pvalue):.6f}",
    f"  df numerator: {int(f2.df_num)}",
    f"  Conclusion:   {'REJECT H0 — macro variables jointly significant' if float(f2.pvalue) < 0.05 else 'FAIL TO REJECT H0'}",
    "",
    "F-TEST 3: Can we drop GDPGrowth and D_Crisis?",
    "H0: beta(GDPGrowth) = beta(D_Crisis) = 0",
    f"  F-statistic:  {float(f3.fvalue):.4f}",
    f"  p-value:      {float(f3.pvalue):.6f}",
    f"  df numerator: {int(f3.df_num)}",
    f"  Conclusion:   {'REJECT H0 — cannot drop both' if float(f3.pvalue) < 0.05 else 'FAIL TO REJECT H0 — dropping GDPGrowth and D_Crisis is supported'}",
])
save_text(ftests_text, 'stage9_ftests.txt')


# ============================================================
# RESULTS REPORTING
# Writes all final output files: individual model summaries,
# side-by-side SE comparisons, Cochrane-Orcutt iteration log,
# and the consolidated four-model results table for the paper.
#
# Outputs:
#   output/model1_ols_summary.txt   — full statsmodels summary, classical OLS
#   output/model2_hc3_summary.txt   — full statsmodels summary, HC3 robust
#   output/model3_nw_summary.txt    — full statsmodels summary, Newey-West
#   output/model_se_comparison.txt  — M1/M2/M3 coefficients with all three SEs
#   output/cochrane_orcutt.txt      — CO iteration log and final DW statistic
#   output/model_comparison.txt     — HC3 vs Cochrane-Orcutt side-by-side
#   output/final_results_table.txt  — four-model paper table with stars and SEs
# ============================================================

# Individual model summaries
save_model_summary(model_ols, "MODEL 1 — LOG-LIN, CLASSICAL OLS", 'model1_ols_summary.txt')
save_model_summary(model_hc3, "MODEL 2 — LOG-LIN, HC3-ROBUST STANDARD ERRORS", 'model2_hc3_summary.txt')
save_model_summary(model_nw, f"MODEL 3 — LOG-LIN, NEWEY-WEST HAC STANDARD ERRORS (maxlags={nw_lags})", 'model3_nw_summary.txt')

# Side-by-side SE comparison: same coefficients, three different SE estimators
comparison_text = "\n".join([
    "=" * 110,
    "MODEL 1 (Classical) vs MODEL 2 (HC3) vs MODEL 3 (Newey-West HAC)",
    "=" * 110,
    f"{'Variable':<16}{'M1 coef':>10}{'M1 SE':>10}{'M1 p':>10}   "
    f"{'M2 coef':>10}{'M2 SE':>10}{'M2 p':>10}   "
    f"{'M3 coef':>10}{'M3 SE':>10}{'M3 p':>10}",
    "-" * 110,
])
for v in X.columns:
    c = model_ols.params[v]
    se1, p1 = model_ols.bse[v], model_ols.pvalues[v]
    se2, p2 = model_hc3.bse[v], model_hc3.pvalues[v]
    se3, p3 = model_nw.bse[v], model_nw.pvalues[v]
    comparison_text = comparison_text + "\n" + \
        f"{v:<16}{c:>+10.4f}{se1:>10.4f}{p1:>10.4f}   " \
        f"{c:>+10.4f}{se2:>10.4f}{p2:>10.4f}   " \
        f"{c:>+10.4f}{se3:>10.4f}{p3:>10.4f}"
comparison_text += "\n\n  Note: M1, M2, M3 have identical coefficients. Only SEs differ."
save_text(comparison_text, 'model_se_comparison.txt')

# Cochrane-Orcutt iteration log + post-correction DW check
dw_after = durbin_watson(model_co.resid)
co_lines.append(f"\n  Model 3 Durbin-Watson statistic: {dw_after:.4f}")
save_text("\n".join(co_lines), 'cochrane_orcutt.txt')

# HC3 vs Cochrane-Orcutt: compares coefficients and SEs across the two
# approaches to correcting for heteroscedasticity and autocorrelation
comp_lines = [
    "=" * 95,
    "MODEL 2 (HC3) vs MODEL 3 (Cochrane-Orcutt)",
    "=" * 95,
    f"{'Variable':<15}{'M2 Coef':>10}{'M2 SE':>10}{'M2 p':>10}    {'M3 Coef':>10}{'M3 SE':>10}{'M3 p':>10}",
    "-" * 95,
]
for v in X.columns:
    c2, se2, p2 = model_hc3.params[v], model_hc3.bse[v], model_hc3.pvalues[v]
    c3, se3, p3 = model_co.params[v], model_co.bse[v], model_co.pvalues[v]
    comp_lines.append(f"{v:<15}{c2:>+10.4f}{se2:>10.4f}{p2:>10.4f}    {c3:>+10.4f}{se3:>10.4f}{p3:>10.4f}")
comp_lines.extend([
    f"\n  DW Model 2: {durbin_watson(model_ols.resid):.4f}   →   DW Model 3: {dw_after:.4f}",
    f"  R² Model 2: {model_ols.rsquared:.4f}            →   R² Model 3: {model_co.rsquared:.4f}",
    f"  Note: Model 3 R² is on quasi-differenced data and not directly comparable to Model 2.",
])
save_text("\n".join(comp_lines), 'model_comparison.txt')

# Final four-model results table for the paper
# Rows: each regressor; columns: Model 1 (linear), 2 (classical), 3 (NW full), 4 (NW trimmed)
# Bottom rows: n, R², adjusted R², F-statistic, Durbin-Watson
all_vars = ['const'] + X_vars

def overall_f(model, regressors):
    # Computes a joint Wald F-test for all regressors in the list.
    joint = ", ".join([f"{v} = 0" for v in regressors])
    f = model.f_test(joint)
    return float(f.fvalue), float(f.pvalue)

f1_stat, f1_p = model_linear.fvalue, model_linear.f_pvalue
f2_stat, f2_p = model_ols.fvalue, model_ols.f_pvalue
f3_stat, f3_p = overall_f(model_nw, X_vars)
f4_stat, f4_p = overall_f(model_nw_trim, regressors_trimmed)

def fmt_f(f, p):
    return f"{f:.2f}{significance_stars(p)}"

table_lines = [
    "=" * 110,
    f"{'':<18}{'Model 1':>20}{'Model 2':>20}{'Model 3':>20}{'Model 4':>20}",
    f"{'':<18}{'(Linear)':>20}{'(Log-Lin Classical)':>20}{'(Log-Lin NW)':>20}{'(Log-Lin Trimmed)':>20}",
    "-" * 110,
]
for v in all_vars:
    c1 = format_coef(model_linear.params.get(v), model_linear.bse.get(v), model_linear.pvalues.get(v))
    c2 = format_coef(model_ols.params.get(v), model_ols.bse.get(v), model_ols.pvalues.get(v))
    c3 = format_coef(model_nw.params.get(v), model_nw.bse.get(v), model_nw.pvalues.get(v))
    if v in model_nw_trim.params.index:
        c4 = format_coef(model_nw_trim.params.get(v), model_nw_trim.bse.get(v), model_nw_trim.pvalues.get(v))
    else:
        c4 = "—"
    parts1 = c1.split('\n') if c1 else ['', '']
    parts2 = c2.split('\n') if c2 else ['', '']
    parts3 = c3.split('\n') if c3 else ['', '']
    parts4 = c4.split('\n') if c4 != "—" else ['—', '']
    while len(parts1) < 2: parts1.append('')
    while len(parts2) < 2: parts2.append('')
    while len(parts3) < 2: parts3.append('')
    while len(parts4) < 2: parts4.append('')
    table_lines.append(f"{v:<18}{parts1[0]:>20}{parts2[0]:>20}{parts3[0]:>20}{parts4[0]:>20}")
    table_lines.append(f"{'':<18}{parts1[1]:>20}{parts2[1]:>20}{parts3[1]:>20}{parts4[1]:>20}")

table_lines.extend([
    "-" * 110,
    f"{'n':<18}{model_linear.nobs:>20.0f}{model_ols.nobs:>20.0f}{model_nw.nobs:>20.0f}{model_nw_trim.nobs:>20.0f}",
    f"{'R-squared':<18}{model_linear.rsquared:>20.4f}{model_ols.rsquared:>20.4f}{model_nw.rsquared:>20.4f}{model_nw_trim.rsquared:>20.4f}",
    f"{'Adj. R-squared':<18}{model_linear.rsquared_adj:>20.4f}{model_ols.rsquared_adj:>20.4f}{model_nw.rsquared_adj:>20.4f}{model_nw_trim.rsquared_adj:>20.4f}",
    f"{'F-statistic':<18}"
    f"{fmt_f(f1_stat, f1_p):>20}"
    f"{fmt_f(f2_stat, f2_p):>20}"
    f"{fmt_f(f3_stat, f3_p):>20}"
    f"{fmt_f(f4_stat, f4_p):>20}",
    f"{'Durbin-Watson':<18}"
    f"{durbin_watson(model_linear.resid):>20.4f}"
    f"{durbin_watson(model_ols.resid):>20.4f}"
    f"{durbin_watson(model_nw.resid):>20.4f}"
    f"{durbin_watson(model_nw_trim.resid):>20.4f}",
    "-" * 110,
    "Notes:",
    "  Coefficients with standard errors in parentheses.",
    "  Significance: *** p<0.01, ** p<0.05, * p<0.10.",
    "  Model 1 dependent variable: Delinquency (level).",
    "  Models 2-4 dependent variable: log(Delinquency).",
    "  R-squared not directly comparable across linear (Model 1) and log-lin (Models 2-4).",
    f"  Model 3 and Model 4 use Newey-West HAC standard errors with {nw_lags} lags.",
    "  Model 4 excludes GDPGrowth and D_Crisis based on joint F-test (F = 0.43, p = 0.649).",
    "=" * 110,
])
save_text("\n".join(table_lines), 'final_results_table.txt')
