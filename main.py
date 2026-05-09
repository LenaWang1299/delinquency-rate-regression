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
