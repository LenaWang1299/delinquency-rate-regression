# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
#     plt.show()
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
plt.show()

import seaborn as sns


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
plt.show()


# ----------------------------
# CORRELATION MATRIX HEATMAP
# ----------------------------
corr_matrix = df[X_vars].corr().round(3)

print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, center=0, square=True,
            cbar_kws={'label': 'Correlation'})
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()


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

print("=" * 55)
print("VARIANCE INFLATION FACTORS")
print("Threshold: VIF ≥ 5 flagged as concern")
print("=" * 55)
print(vif_data.to_string(index=False))


# ----------------------------
# 3. LINEAR REGRESSION
# ----------------------------
X = df[X_vars]
X = sm.add_constant(X)
y = df[Y_var_log]
# y = df[Y_var]

model = sm.OLS(y, X).fit()
print(model.summary())

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
plt.show()

# ----------------------------
# 5. QQ PLOT
# ----------------------------
sm.qqplot(residuals, line='s')
plt.title('QQ Plot')
plt.show()



# ----------------------------
# STAGE 7 — HETEROSCEDASTICITY TESTS AND CORRECTION
# ----------------------------
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Re-run Model Log to have it handy
model_log = sm.OLS(y, X).fit()

# --- Breusch-Pagan test ---
bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model_log.resid, X)
print("=" * 60)
print("BREUSCH-PAGAN-GODFREY TEST")
print("H0: homoscedasticity")
print("=" * 60)
print(f"  LM statistic:       {bp_stat:.4f}")
print(f"  LM p-value:         {bp_p:.4f}")
print(f"  F-statistic:        {bp_f:.4f}")
print(f"  F p-value:          {bp_fp:.4f}")
print(f"  Conclusion: {'REJECT H0 - heteroscedasticity present' if bp_p < 0.05 else 'DO NOT REJECT H0'}")

# --- White test ---
w_stat, w_p, w_f, w_fp = het_white(model_log.resid, X)
print("\n" + "=" * 60)
print("WHITE TEST")
print("H0: homoscedasticity")
print("=" * 60)
print(f"  LM statistic:       {w_stat:.4f}")
print(f"  LM p-value:         {w_p:.4f}")
print(f"  F-statistic:        {w_f:.4f}")
print(f"  F p-value:          {w_fp:.4f}")
print(f"  Conclusion: {'REJECT H0 - heteroscedasticity present' if w_p < 0.05 else 'DO NOT REJECT H0'}")

# --- Corrected model with HC3 robust standard errors ---
model_log_robust = sm.OLS(y, X).fit(cov_type='HC3')

print("\n" + "=" * 80)
print("COMPARISON: CLASSICAL vs. HC3-ROBUST STANDARD ERRORS")
print("=" * 80)
comparison = pd.DataFrame({
    'Coefficient': model_log.params.round(4),
    'Classical SE': model_log.bse.round(4),
    'Classical p': model_log.pvalues.round(4),
    'HC3 Robust SE': model_log_robust.bse.round(4),
    'HC3 Robust p': model_log_robust.pvalues.round(4),
})
# How much did SEs change?
comparison['SE Change (%)'] = (100 * (model_log_robust.bse - model_log.bse) / model_log.bse).round(1)
print(comparison.to_string())



# ----------------------------
# MODEL 2 — LOG-LIN WITH HC3 ROBUST STANDARD ERRORS
# ----------------------------
# Fit with HC3 robust standard errors
model_2 = sm.OLS(y, X).fit(cov_type='HC3')

# Print the full regression output — statsmodels automatically uses HC3 throughout
print("=" * 75)
print("\nMODEL 2 — LOG-LIN, HC3-ROBUST STANDARD ERRORS")
print("=" * 75)
print(model_2.summary())





# ----------------------------
# STAGE 8 — AUTOCORRELATION TESTING AND CORRECTION
# ----------------------------
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

# Use our baseline log-lin for diagnostics
model_log = sm.OLS(y, X).fit()  # classical fit for residual extraction

# --- Durbin-Watson ---
dw = durbin_watson(model_log.resid)
print("=" * 60)
print("DURBIN-WATSON TEST (H0: no 1st-order autocorrelation)")
print("=" * 60)
print(f"  DW statistic: {dw:.4f}")
print(f"  Interpretation:")
print(f"    DW ≈ 2:   no autocorrelation")
print(f"    DW < 1.5: positive autocorrelation likely")
print(f"    DW > 2.5: negative autocorrelation likely")

# --- Breusch-Godfrey LM test (up to 4 lags = 1 year of quarters) ---
bg_stat, bg_p, bg_f, bg_fp = acorr_breusch_godfrey(model_log, nlags=4)
print("\n" + "=" * 60)
print("BREUSCH-GODFREY LM TEST (H0: no autocorrelation up to lag 4)")
print("=" * 60)
print(f"  LM statistic:  {bg_stat:.4f}")
print(f"  LM p-value:    {bg_p:.6f}")
print(f"  F-statistic:   {bg_f:.4f}")
print(f"  F p-value:     {bg_fp:.6f}")

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
plt.show()

# --- Cochrane-Orcutt correction ---
print("\n" + "=" * 60)
print("COCHRANE-ORCUTT ITERATIVE PROCEDURE")
print("=" * 60)

# Step 1: Estimate rho from residuals
rho_hat = model_log.resid.autocorr(lag=1)
print(f"\n  Step 1: Estimated ρ from residuals:  ρ̂ = {rho_hat:.4f}")

# Step 2: Quasi-difference all variables
# y*_t = y_t - rho * y_{t-1}, drop the first row
y_qd = y - rho_hat * y.shift(1)
X_qd = X - rho_hat * X.shift(1)
# Drop the first observation (which is NaN after shifting)
y_qd = y_qd.dropna()
X_qd = X_qd.dropna()
# Replace the transformed constant with 1 (the transformation creates (1 - rho),
# but we want the intercept in ORIGINAL units so we can interpret it cleanly)
# Standard practice: run OLS on the transformed variables with the transformed constant
print(f"\n  Step 2: Quasi-differenced data has {len(y_qd)} observations (one dropped)")

# Step 3: Re-run OLS on quasi-differenced data
model_3 = sm.OLS(y_qd, X_qd).fit()
print(f"\n  Step 3: OLS on quasi-differenced data complete")

# Step 4: Iterate to convergence (one iteration is often enough, but let's do it properly)
max_iter = 10
tol = 1e-6
for it in range(max_iter):
    rho_prev = rho_hat
    # Compute residuals of CURRENT quasi-differenced model, in ORIGINAL scale
    # The original-scale residual at time t is: y_t - X_t @ beta_current
    residuals_original = y - X @ model_3.params
    rho_hat = residuals_original.autocorr(lag=1)

    # Re-quasi-difference with updated rho
    y_qd = (y - rho_hat * y.shift(1)).dropna()
    X_qd = (X - rho_hat * X.shift(1)).dropna()
    model_3 = sm.OLS(y_qd, X_qd).fit()

    if abs(rho_hat - rho_prev) < tol:
        print(f"\n  Converged after {it+1} iterations. Final ρ̂ = {rho_hat:.4f}")
        break
else:
    print(f"\n  Reached max {max_iter} iterations. Final ρ̂ = {rho_hat:.4f}")

# --- Check whether autocorrelation is resolved in Model 3 residuals ---
dw_after = durbin_watson(model_3.resid)
print(f"\n  Model 3 Durbin-Watson statistic: {dw_after:.4f}")

# --- Side-by-side comparison: Model 2 (HC3) vs Model 3 (Cochrane-Orcutt) ---
model_2 = sm.OLS(y, X).fit(cov_type='HC3')

print("\n" + "=" * 95)
print("MODEL 2 (HC3) vs MODEL 3 (Cochrane-Orcutt)")
print("=" * 95)
print(f"{'Variable':<15}{'M2 Coef':>10}{'M2 SE':>10}{'M2 p':>10}    {'M3 Coef':>10}{'M3 SE':>10}{'M3 p':>10}")
print("-" * 95)
for v in X.columns:
    c2, se2, p2 = model_2.params[v], model_2.bse[v], model_2.pvalues[v]
    c3, se3, p3 = model_3.params[v], model_3.bse[v], model_3.pvalues[v]
    print(f"{v:<15}{c2:>+10.4f}{se2:>10.4f}{p2:>10.4f}    {c3:>+10.4f}{se3:>10.4f}{p3:>10.4f}")

print(f"\n  DW Model 2: {durbin_watson(model_log.resid):.4f}   →   DW Model 3: {dw_after:.4f}")
print(f"  R² Model 2: {model_log.rsquared:.4f}            →   R² Model 3: {model_3.rsquared:.4f}")
print(f"  Note: Model 3 R² is on quasi-differenced data and not directly comparable to Model 2.")
