import pickle, numpy as np, pandas as pd
from sklearn.metrics import root_mean_squared_error

with open('scratch_residuals.pkl','rb') as f:
    d = pickle.load(f)

residuals = d['residuals']
y_true = d['y_true']
y_pred = d['y_pred']
X_val = d['X_val']

df = X_val.copy()
df['resid'] = residuals
df['y_true'] = y_true
df['y_pred'] = y_pred

print('=== DISTRIBUTION DES RESIDUS ===')
print(f'mean={residuals.mean():.3f}, std={residuals.std():.3f}')
for thr in [5, 7, 10]:
    pct = 100*(np.abs(residuals) > thr).mean()
    n = (np.abs(residuals) > thr).sum()
    print(f'  |resid|>{thr}: {n} rows ({pct:.1f}%)')

print()
print('=== RMSE PAR SOUS-GROUPE ===')
# RMSE quand off est manquant vs présent
for label, mask in [
    ('off present', df['off'].notna()),
    ('off missing', df['off'].isna()),
    ('on present', df['on'].notna()),
    ('on missing', df['on'].isna()),
    ('ledd present', df['ledd'].notna()),
    ('ledd missing', df['ledd'].isna()),
]:
    if mask.sum() > 0:
        r = root_mean_squared_error(df.loc[mask,'y_true'], df.loc[mask,'y_pred'])
        print(f'  {label:20s}: RMSE={r:.4f}  n={mask.sum()}')

print()
print('=== RMSE PAR QUINTILE DE TARGET ===')
quintiles = pd.qcut(df['y_true'], 5)
for q, grp in df.groupby(quintiles):
    r = root_mean_squared_error(grp['y_true'], grp['y_pred'])
    print(f'  target {q}: RMSE={r:.4f}  n={len(grp)}')

print()
print('=== BIAIS PAR RATER (top/bottom) ===')
rater_stats = df.groupby('rater_id').apply(
    lambda g: pd.Series({
        'rmse': root_mean_squared_error(g['y_true'], g['y_pred']),
        'bias': g['resid'].mean(),
        'n': len(g)
    })
).sort_values('rmse', ascending=False)
print('Top 10 worst raters:')
print(rater_stats.head(10).to_string())
print('Rater RMSE std:', rater_stats['rmse'].std())

print()
print('=== BIAIS SYSTEMATIQUE PAR PATIENT ===')
pat_stats = df.groupby('patient_id').apply(
    lambda g: pd.Series({
        'bias': g['resid'].mean(),
        'rmse': root_mean_squared_error(g['y_true'], g['y_pred']),
        'n': len(g)
    })
)
print('Distribution des biais patient (mean residu par patient):')
print(pat_stats['bias'].describe())
print(f'Patients biais>3: {(pat_stats["bias"]>3).sum()}, biais<-3: {(pat_stats["bias"]<-3).sum()}')
# Variance du biais patient = variance "fixe" non captee
var_resid = residuals.var()
var_pat_bias = pat_stats['bias'].var()
print(f'Variance residus totale: {var_resid:.3f}')
print(f'Variance biais patient:  {var_pat_bias:.3f}  ({100*var_pat_bias/var_resid:.1f}% des residus)')
print(f'RMSE patient moyen: {pat_stats["rmse"].mean():.4f}')
print(f'RMSE si on corrigeait le biais patient: {np.sqrt(var_resid - var_pat_bias):.4f}  (borne basse theorique)')

print()
print('=== AUTOCORRELATION DES RESIDUS INTRA-PATIENT ===')
autocorrs = []
for pid, grp in df.groupby('patient_id'):
    grp_sorted = grp.sort_values('age')
    if len(grp_sorted) >= 4:
        r = grp_sorted['resid'].values
        if np.std(r) > 0:
            ac1 = np.corrcoef(r[:-1], r[1:])[0,1]
            autocorrs.append(ac1)
print(f'Autocorr lag-1 residus (n={len(autocorrs)} patients): mean={np.mean(autocorrs):.3f}, std={np.std(autocorrs):.3f}')
print(f'Patients avec autocorr>0.3: {sum(a>0.3 for a in autocorrs)} ({100*sum(a>0.3 for a in autocorrs)/len(autocorrs):.0f}%)')
