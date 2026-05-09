# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

# Create output directories
os.makedirs('figures', exist_ok=True)
os.makedirs('output', exist_ok=True)

def save_text(content, filename):
    path = os.path.join('output', filename)
    with open(path, 'w') as f:
        f.write(content)
    print(content)

# Load data
df = pd.read_excel('data/Econ318_RawData_Paper.xlsx')

df['log_Delinquency'] = np.log(df['Delinquency'])
df['log_UNRATE'] = np.log(df['UNRATE'])
df['log_MORTGAGE30US'] = np.log(df['MORTGAGE30US'])

# ----------------------------
# 1. SCATTER PLOT (X vs Y)
# ----------------------------
# Define variables
Y_var = 'Delinquency'
X_vars = ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'GDPGrowth','D_Crisis', 'D_Covid']

Y_var_log = 'log_Delinquency'


# for var in X_vars:
#     plt.figure()
#     plt.scatter(df[var], df[Y_var])
#     plt.xlabel(var)
#     plt.ylabel(Y_var)
#     plt.title(f'{Y_var} vs {var}')
#     plt.savefig(f'figures/scatter_{var}.png', dpi=150, bbox_inches='tight')
#     plt.close()
#
# ----------------------------
# 2. HISTOGRAM (lin-lin model)
# ----------------------------
# Combine Y and X variables
vars_no_dummy = [Y_var] + ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'GDPGrowth']

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()

for i, var in enumerate(vars_no_dummy):
    axes[i].hist(df[var], bins=40)
    axes[i].set_title(f'{var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequency')

# Remove extra empty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('figures/01_histograms.png', dpi=150, bbox_inches='tight')
plt.close()


# ----------------------------
# TIME SERIES: 5 non-dummy variables, separate subplots
# ----------------------------

units = {
    'Delinquency':  'Percent',
    'UNRATE':       'Percent',
    'MORTGAGE30US': 'Percent',
    'HPI_Growth':   'YoY % change',
    'GDPGrowth':    'Annualized %',
}

# Define the shaded windows
crisis_start = pd.Timestamp('2007-07-01')
crisis_end   = pd.Timestamp('2010-12-31')
covid_start  = pd.Timestamp('2020-01-01')
covid_end    = pd.Timestamp('2021-09-30')

fig, axes = plt.subplots(3, 2, figsize=(13, 10))
axes = axes.flatten()

for i, var in enumerate(vars_no_dummy):
    axes[i].plot(df['observation_date'], df[var])
    axes[i].axvspan(crisis_start, crisis_end, alpha=0.15, color='red', label='Financial Crisis')
    axes[i].axvspan(covid_start, covid_end, alpha=0.15, color='blue', label='COVID period')

# Hide the unused 6th subplot
axes[5].axis('off')

plt.suptitle('Time Series: 1991Q1 – 2025Q4')
plt.tight_layout()
plt.savefig('figures/02_time_series.png', dpi=150, bbox_inches='tight')
plt.close()


# ----------------------------
# CORRELATION MATRIX HEATMAP
# ----------------------------
corr_matrix = df[X_vars].corr().round(3)

save_text(corr_matrix.to_string(), 'correlation_matrix.txt')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('figures/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()


# ----------------------------
# VARIANCE INFLATION FACTORS (VIF)
# ----------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df[X_vars]
X_vif = sm.add_constant(X_vif)

# Compute VIF for each regressor (skip the constant at index 0)
vif_data = pd.DataFrame({
    'Variable': X_vif.columns[1:],
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
})
vif_data['VIF'] = vif_data['VIF'].round(3)

# Flag severity — user's threshold = 5
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


# ----------------------------
# 3. LINEAR REGRESSION
# ----------------------------
X = df[X_vars]
X = sm.add_constant(X)
y = df[Y_var_log]
# y = df[Y_var]

model = sm.OLS(y, X).fit()
save_text(str(model.summary()), 'model1_ols_summary.txt')

# ----------------------------
# 4. RESIDUAL PLOT
# ----------------------------
residuals = model.resid
fitted = model.fittedvalues

plt.figure()
plt.scatter(fitted, residuals)
plt.axhline(0, linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('figures/04_residual_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# ----------------------------
# 5. QQ PLOT
# ----------------------------
sm.qqplot(residuals, line='s')
plt.title('QQ Plot')
plt.savefig('figures/05_qq_plot.png', dpi=150, bbox_inches='tight')
plt.close()



# ----------------------------
# STAGE 7 — HETEROSCEDASTICITY TESTS AND CORRECTION
# ----------------------------
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Re-run Model Log to have it handy
model_log = sm.OLS(y, X).fit()

# --- Breusch-Pagan test ---
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model_log.resid, X)

# --- White test ---
w_stat, w_p, w_f, w_fp = het_white(model_log.resid, X)

# --- Corrected model with HC3 robust standard errors ---
model_log_robust = sm.OLS(y, X).fit(cov_type='HC3')

comparison = pd.DataFrame({
    'Coefficient': model_log.params.round(4),
    'Classical SE': model_log.bse.round(4),
    'Classical p': model_log.pvalues.round(4),
    'HC3 Robust SE': model_log_robust.bse.round(4),
    'HC3 Robust p': model_log_robust.pvalues.round(4),
})
comparison['SE Change (%)'] = (100 * (model_log_robust.bse - model_log.bse) / model_log.bse).round(1)

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
    comparison.to_string(),
])
save_text(hetero_text, 'heteroscedasticity_tests.txt')



# ----------------------------
# MODEL 2 — LOG-LIN WITH HC3 ROBUST STANDARD ERRORS
# ----------------------------
# Fit with HC3 robust standard errors
model_2 = sm.OLS(y, X).fit(cov_type='HC3')

model2_text = "\n".join([
    "=" * 75,
    "MODEL 2 — LOG-LIN, HC3-ROBUST STANDARD ERRORS",
    "=" * 75,
    str(model_2.summary()),
])
save_text(model2_text, 'model2_hc3_summary.txt')


# ----------------------------
# MODEL 3 — LOG-LIN WITH NEWEY-WEST HAC STANDARD ERRORS
# ----------------------------
# Lag length for quarterly data: L = floor(4 * (n/100)^(2/9))
import math
n_obs = len(df)
nw_lags = math.floor(4 * (n_obs / 100) ** (2 / 9))

model_3_nw = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})

model3_text = "\n".join([
    "=" * 75,
    f"MODEL 3 — LOG-LIN, NEWEY-WEST HAC STANDARD ERRORS (maxlags={nw_lags})",
    "=" * 75,
    str(model_3_nw.summary()),
])
save_text(model3_text, 'model3_nw_summary.txt')

# ----------------------------
# SIDE-BY-SIDE: MODEL 1 vs MODEL 2 vs MODEL 3
# ----------------------------
model_1 = sm.OLS(y, X).fit()
model_2 = sm.OLS(y, X).fit(cov_type='HC3')

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
    c = model_1.params[v]
    se1, p1 = model_1.bse[v], model_1.pvalues[v]
    se2, p2 = model_2.bse[v], model_2.pvalues[v]
    se3, p3 = model_3_nw.bse[v], model_3_nw.pvalues[v]
    comparison_text = comparison_text + "\n" + \
        f"{v:<16}{c:>+10.4f}{se1:>10.4f}{p1:>10.4f}   " \
        f"{c:>+10.4f}{se2:>10.4f}{p2:>10.4f}   " \
        f"{c:>+10.4f}{se3:>10.4f}{p3:>10.4f}"

comparison_text += "\n\n  Note: M1, M2, M3 have identical coefficients. Only SEs differ."
save_text(comparison_text, 'model_se_comparison.txt')



# ----------------------------
# STAGE 8 — AUTOCORRELATION TESTING AND CORRECTION
# ----------------------------
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# Use our baseline log-lin for diagnostics
model_log = sm.OLS(y, X).fit()  # classical fit for residual extraction

# --- Durbin-Watson ---
dw = durbin_watson(model_log.resid)

# --- Breusch-Godfrey LM test (up to 4 lags = 1 year of quarters) ---
bg_stat, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(model_log, nlags=4)

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

# --- Residual time series plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(df['observation_date'], model_log.resid)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Residuals Over Time')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Residual')
axes[0].grid(True, alpha=0.3)

# Residuals vs. lagged residuals (visualizes AR(1) structure)
resid_lag = model_log.resid.shift(1)
axes[1].scatter(resid_lag, model_log.resid, alpha=0.6, s=22, color='#8B0000',
                edgecolor='black', linewidth=0.3)
axes[1].axhline(0, color='black', linewidth=0.5, linestyle=':')
axes[1].axvline(0, color='black', linewidth=0.5, linestyle=':')
rho = model_log.resid.autocorr(lag=1)
axes[1].set_title(f'Residual vs. Lagged Residual (ρ̂ = {rho:.3f})')
axes[1].set_xlabel('Residual at t-1')
axes[1].set_ylabel('Residual at t')

plt.tight_layout()
plt.savefig('figures/06_autocorrelation_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Cochrane-Orcutt correction ---
rho_hat = model_log.resid.autocorr(lag=1)
co_lines = [
    "=" * 60,
    "COCHRANE-ORCUTT ITERATIVE PROCEDURE",
    "=" * 60,
    f"\n  Step 1: Estimated ρ from residuals:  ρ̂ = {rho_hat:.4f}",
]

# Step 2: Quasi-difference all variables
y_qd = y - rho_hat * y.shift(1)
X_qd = X - rho_hat * X.shift(1)
# Drop the first observation (which is NaN after shifting)
y_qd = y_qd.dropna()
X_qd = X_qd.dropna()
co_lines.append(f"\n  Step 2: Quasi-differenced data has {len(y_qd)} observations (one dropped)")

# Step 3: Re-run OLS on quasi-differenced data
model_3 = sm.OLS(y_qd, X_qd).fit()
co_lines.append(f"\n  Step 3: OLS on quasi-differenced data complete")

# Step 4: Iterate to convergence
max_iter = 10
tol = 1e-6
for it in range(max_iter):
    rho_prev = rho_hat
    residuals_original = y - X @ model_3.params
    rho_hat = residuals_original.autocorr(lag=1)

    y_qd = (y - rho_hat * y.shift(1)).dropna()
    X_qd = (X - rho_hat * X.shift(1)).dropna()
    model_3 = sm.OLS(y_qd, X_qd).fit()

    if abs(rho_hat - rho_prev) < tol:
        co_lines.append(f"\n  Converged after {it+1} iterations. Final ρ̂ = {rho_hat:.4f}")
        break
else:
    co_lines.append(f"\n  Reached max {max_iter} iterations. Final ρ̂ = {rho_hat:.4f}")

# --- Check whether autocorrelation is resolved in Model 3 residuals ---
dw_after = durbin_watson(model_3.resid)
co_lines.append(f"\n  Model 3 Durbin-Watson statistic: {dw_after:.4f}")

save_text("\n".join(co_lines), 'cochrane_orcutt.txt')

# --- Side-by-side comparison: Model 2 (HC3) vs Model 3 (Cochrane-Orcutt) ---
model_2 = sm.OLS(y, X).fit(cov_type='HC3')

comp_lines = [
    "=" * 95,
    "MODEL 2 (HC3) vs MODEL 3 (Cochrane-Orcutt)",
    "=" * 95,
    f"{'Variable':<15}{'M2 Coef':>10}{'M2 SE':>10}{'M2 p':>10}    {'M3 Coef':>10}{'M3 SE':>10}{'M3 p':>10}",
    "-" * 95,
]
for v in X.columns:
    c2, se2, p2 = model_2.params[v], model_2.bse[v], model_2.pvalues[v]
    c3, se3, p3 = model_3.params[v], model_3.bse[v], model_3.pvalues[v]
    comp_lines.append(f"{v:<15}{c2:>+10.4f}{se2:>10.4f}{p2:>10.4f}    {c3:>+10.4f}{se3:>10.4f}{p3:>10.4f}")

comp_lines.extend([
    f"\n  DW Model 2: {durbin_watson(model_log.resid):.4f}   →   DW Model 3: {dw_after:.4f}",
    f"  R² Model 2: {model_log.rsquared:.4f}            →   R² Model 3: {model_3.rsquared:.4f}",
    f"  Note: Model 3 R² is on quasi-differenced data and not directly comparable to Model 2.",
])
save_text("\n".join(comp_lines), 'model_comparison.txt')



# ----------------------------
# STAGE 9 — JOINT F-TESTS (Newey-West HAC)
# ----------------------------

# Primary model for all F-tests
import math
n_obs = len(df)
nw_lags = math.floor(4 * (n_obs / 100) ** (2 / 9))
model_primary = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})

# --- F-Test 1: Dummies jointly significant? ---
f1 = model_primary.f_test("D_Crisis = 0, D_Covid = 0")

# --- F-Test 2: Macro variables jointly significant? ---
f2 = model_primary.f_test("UNRATE = 0, MORTGAGE30US = 0, HPI_Growth = 0, GDPGrowth = 0")

# --- F-Test 3: Can we drop GDPGrowth and D_Crisis? ---
f3 = model_primary.f_test("GDPGrowth = 0, D_Crisis = 0")

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



# ----------------------------
# STAGE 10 — FINAL RESULTS TABLE
# ----------------------------
import math

# Define specifications
regressors_full = ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'GDPGrowth', 'D_Crisis', 'D_Covid']
regressors_trimmed = ['UNRATE', 'MORTGAGE30US', 'HPI_Growth', 'D_Covid']

# Newey-West lag length
n_obs = len(df)
nw_lags = math.floor(4 * (n_obs / 100) ** (2 / 9))

# Fit all four models
X_full = sm.add_constant(df[regressors_full])
X_trim = sm.add_constant(df[regressors_trimmed])
y_log = df['log_Delinquency']
y_lin = df['Delinquency']

m1_linear   = sm.OLS(y_lin, X_full).fit()                                                # Model 1: linear DV
m2_classic  = sm.OLS(y_log, X_full).fit()                                                # Model 2: log-lin classical
m3_nw       = sm.OLS(y_log, X_full).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})   # Model 3: log-lin NW
m4_trim     = sm.OLS(y_log, X_trim).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})   # Model 4: trimmed NW

# Helper: format coefficient with asterisks and SE
def fmt(coef, se, p):
    if pd.isna(coef): return ''
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    return f"{coef:+.4f}{stars}\n  ({se:.4f})"

# All variables that appear in any model (in order)
all_vars = ['const'] + regressors_full

# Build table rows
table_lines = [
    "=" * 110,
    f"{'':<18}{'Model 1':>20}{'Model 2':>20}{'Model 3':>20}{'Model 4':>20}",
    f"{'':<18}{'(Linear)':>20}{'(Log-Lin Classical)':>20}{'(Log-Lin NW)':>20}{'(Log-Lin Trimmed)':>20}",
    "-" * 110,
]

for v in all_vars:
    # M1 always has it, M2/M3 always have it (full), M4 only if in trimmed
    c1 = fmt(m1_linear.params.get(v), m1_linear.bse.get(v), m1_linear.pvalues.get(v))
    c2 = fmt(m2_classic.params.get(v), m2_classic.bse.get(v), m2_classic.pvalues.get(v))
    c3 = fmt(m3_nw.params.get(v), m3_nw.bse.get(v), m3_nw.pvalues.get(v))
    if v in m4_trim.params.index:
        c4 = fmt(m4_trim.params.get(v), m4_trim.bse.get(v), m4_trim.pvalues.get(v))
    else:
        c4 = "—"
    # Each variable takes 2 lines (coef + SE)
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

# Helper for robust F-statistic of overall model significance
def overall_f(model, regressors):
    joint = ", ".join([f"{v} = 0" for v in regressors])
    f = model.f_test(joint)
    return float(f.fvalue), float(f.pvalue)

# Compute overall F for each model
f1_stat, f1_p = m1_linear.fvalue, m1_linear.f_pvalue       # Classical OLS, classical F
f2_stat, f2_p = m2_classic.fvalue, m2_classic.f_pvalue     # Classical OLS, classical F
f3_stat, f3_p = overall_f(m3_nw, regressors_full)          # Newey-West Wald F
f4_stat, f4_p = overall_f(m4_trim, regressors_trimmed)     # Newey-West Wald F

# Format F-stat with stars based on its p-value
def fmt_f(f, p):
    stars = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    return f"{f:.2f}{stars}"

# Bottom rows: summary statistics
table_lines.extend([
    "-" * 110,
    f"{'n':<18}{m1_linear.nobs:>20.0f}{m2_classic.nobs:>20.0f}{m3_nw.nobs:>20.0f}{m4_trim.nobs:>20.0f}",
    f"{'R-squared':<18}{m1_linear.rsquared:>20.4f}{m2_classic.rsquared:>20.4f}{m3_nw.rsquared:>20.4f}{m4_trim.rsquared:>20.4f}",
    f"{'Adj. R-squared':<18}{m1_linear.rsquared_adj:>20.4f}{m2_classic.rsquared_adj:>20.4f}{m3_nw.rsquared_adj:>20.4f}{m4_trim.rsquared_adj:>20.4f}",
    f"{'F-statistic':<18}"
    f"{fmt_f(f1_stat, f1_p):>20}"
    f"{fmt_f(f2_stat, f2_p):>20}"
    f"{fmt_f(f3_stat, f3_p):>20}"
    f"{fmt_f(f4_stat, f4_p):>20}",
    f"{'Durbin-Watson':<18}"
    f"{durbin_watson(m1_linear.resid):>20.4f}"
    f"{durbin_watson(m2_classic.resid):>20.4f}"
    f"{durbin_watson(m3_nw.resid):>20.4f}"
    f"{durbin_watson(m4_trim.resid):>20.4f}",
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