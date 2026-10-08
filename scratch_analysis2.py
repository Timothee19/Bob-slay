import pickle, numpy as np, pandas as pd
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features import build_features

X_train, y_train = load_dataset()
X_test = load_test_dataset()

with open('scratch_residuals.pkl','rb') as f:
    d = pickle.load(f)
X_val = d['X_val']
residuals = d['residuals']
y_true = d['y_true']
y_pred = d['y_pred']
df = X_val.copy()
df['resid'] = residuals
df['y_true'] = y_true
df['y_pred'] = y_pred

print('=== ANALYSE: OFF EST-IL VRAIMENT UN PROXY LINEAIRE DU TARGET? ===')
# On regarde off vs target sur le train
mask_off = X_train['off'].notna() & (y_train > 0)
off_vals = X_train.loc[mask_off, 'off'].values
tgt_vals = y_train.loc[mask_off].values
ratio = off_vals / tgt_vals
print(f'Ratio off/target: mean={ratio.mean():.4f}, std={ratio.std():.4f}, median={np.median(ratio):.4f}')
print(f'Corr(off, target): {np.corrcoef(off_vals, tgt_vals)[0,1]:.4f}')
# Non-linearity: does a polynomial fit of off->target improve?
from numpy.polynomial import polynomial as P
coeffs1 = np.polyfit(off_vals, tgt_vals, 1)
coeffs2 = np.polyfit(off_vals, tgt_vals, 2)
res1 = tgt_vals - np.polyval(coeffs1, off_vals)
res2 = tgt_vals - np.polyval(coeffs2, off_vals)
print(f'Linear fit RMSE: {np.sqrt((res1**2).mean()):.4f}')
print(f'Quadratic fit RMSE: {np.sqrt((res2**2).mean()):.4f}')

print()
print('=== BIAIS RATER: OFF vs ON sur memes visites ===')
mask_both = X_train['off'].notna() & X_train['on'].notna() & (y_train > 0)
df_both = X_train[mask_both].copy()
df_both['target'] = y_train[mask_both].values
df_both['ratio_off'] = df_both['off'] / df_both['target']
df_both['ratio_on'] = df_both['on'] / df_both['target']
df_both['diff_ratios'] = df_both['ratio_off'] - df_both['ratio_on']
# Per visite: est-ce que le biais off est correlé au biais on?
print(f'Corr(ratio_off, ratio_on) intra-visite: {df_both["ratio_off"].corr(df_both["ratio_on"]):.4f}')
print(f'diff = ratio_off - ratio_on: mean={df_both["diff_ratios"].mean():.4f}, std={df_both["diff_ratios"].std():.4f}')
# Partial correlation: controlling for time_since_intake_on
mask_t = mask_both & X_train['time_since_intake_on'].notna()
df_t = X_train[mask_t].copy()
df_t['target'] = y_train[mask_t].values
df_t['ratio_off'] = df_t['off'] / df_t['target']
df_t['ratio_on'] = df_t['on'] / df_t['target']
# residualize ratio_on from time_since_intake_on
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(df_t[['time_since_intake_on']], df_t['ratio_on'])
ratio_on_resid = df_t['ratio_on'] - lr.predict(df_t[['time_since_intake_on']])
print(f'Partial corr(ratio_off, ratio_on | time): {df_t["ratio_off"].corr(ratio_on_resid):.4f}')
print(f'=> Meme en controlant le timing, les biais sont corréles à {df_t["ratio_off"].corr(ratio_on_resid):.3f}')

print()
print('=== ANALYSE PATIENT INTERCEPT OFFSET ===')
# Par patient, quel est l'écart moyen entre off_corrected et target?
# C'est le "niveau absolu" non capté
from parkinsons.evaluate import get_cv_folds
from parkinsons.features import build_features

folds = get_cv_folds(X_train, y_train)
tr_idx, va_idx = folds[0]
X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
X_va = X_train.iloc[va_idx].reset_index(drop=True)
y_va = y_train.iloc[va_idx].reset_index(drop=True)

X_tr_fe, X_va_fe = build_features(X_tr, y_tr, X_va)
# patient_off_mean vs mean target per patient (in val fold)
val_pat_stats = pd.DataFrame({
    'off_mean': X_va_fe['patient_off_mean'],
    'target_mean': y_va.values,
    'trend': X_va_fe['patient_trend_at_age'],
    'pid': X_va['patient_id'].values
}).groupby('pid').mean()

print(f'Corr(patient_off_mean, target_mean): {val_pat_stats["off_mean"].corr(val_pat_stats["target_mean"]):.4f}')
print(f'Corr(patient_trend_at_age, target_mean): {val_pat_stats["trend"].corr(val_pat_stats["target_mean"]):.4f}')
# Residual from patient trend
val_pat_stats['bias'] = val_pat_stats['target_mean'] - val_pat_stats['trend']
print(f'Patient trend bias: mean={val_pat_stats["bias"].mean():.3f}, std={val_pat_stats["bias"].std():.3f}')
print(f'=> Ce biais de {val_pat_stats["bias"].std():.2f} est du au level shift systematique par patient non capte par la trend')
