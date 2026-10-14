# %% [markdown]
# # Exp 21 — exp 19 + estimateur bayesien par patient (features bayes_*)
#
# Pipeline independant du package (features fold-aware calculees uniquement sur le fold train) :
# - biais medecin = q75 de log(off/target) par rater_id (normalise sur la mediane globale)
# - courbe PK f(t) de ON, reponse au medicament k par patient (onk)
# - z = mesure log (off en priorite, sinon on corrige), courbe MLE par patient avec densite KDE des residus
# - enveloppes / quantiles / fenetres locales / cummax de z detrende (zq, loc, loc2)
# - estimateur bayesien empirique par patient (eb), fiabilite medecin (rel), features de mode (mode)
# - LightGBM + XGBoost moyennes, lissage poly2 par patient, puis ratchet x1.02
#   (dans y_train, par patient trie par age, le target ne redescend jamais : quand la courbe
#    stagne il monte de x1.02 par visite)
# Toute statistique qui utilise target est calculee sur le fold train uniquement.

# %%
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import skore
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import get_cv_folds
from parkinsons.hub import load_skore_credentials

VARIANT = "onk_norater_mle_zq_loc_loc2_eb_rel_bq75_offz_mode_ensx"
SEED = 0
PRIOR = 1.0
RATCHET = 1.02

N_CPU = os.cpu_count() or 4
CV_JOBS = min(5, N_CPU)
INNER_JOBS = max(1, N_CPU // CV_JOBS)
print(f"CPU: {N_CPU}  -> CV_JOBS={CV_JOBS}  INNER_JOBS={INNER_JOBS}")

DATA_DIR = PROJECT_ROOT / "data"
CATS = ["cohort", "gene", "rater_id"]

# %%
T_BINS = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0])


# ---------------------------------------------------------------- fold-aware stats (use target)
def rater_affine(o):
    """per-rater log off = a + b log T, fitted on medians of 6 T-quantile bins (robust)."""
    out = {}
    for r, g in o.groupby(o.rater_id.astype(str)):
        lt = np.log(g.target.values); lo = np.log(g.off.values)
        qs = np.quantile(lt, np.linspace(0, 1, 7))
        xs, ys = [], []
        for a_, b_ in zip(qs[:-1], qs[1:]):
            k = (lt >= a_) & (lt <= b_)
            if k.sum() > 10:
                xs.append(np.median(lt[k])); ys.append(np.median(lo[k]))
        if len(xs) >= 3:
            c = np.polyfit(xs, ys, 1); out[r] = (c[1], c[0])
    return out


def fit_calib(tr):
    t = tr[tr.target > 5]
    o = t[t.off.notna()]
    b_r = (o.off / o.target).groupby(o.rater_id, observed=True).median()
    b_glob = float((o.off / o.target).median())
    if "bq" in VARIANT:
        q_ = 0.75 if "bq75" in VARIANT else 0.85
        o2 = o[o.off > 0]
        lr_ = np.log(o2.off / o2.target)
        g_ = np.exp(lr_.quantile(q_))
        b_r = np.exp(lr_.groupby(o2.rater_id, observed=True).quantile(q_)) * (b_glob / g_)   # keep global level comparable to median-based

    aff = None
    if "rcal" in VARIANT:
        aff = rater_affine(o[o.off > 0])
    n = t[t.on.notna()].copy()
    br_n = n.rater_id.map(b_r).astype(float).fillna(b_glob)
    if aff is not None:
        n = n[n.on > 0]
        ab = n.rater_id.astype(str).map(aff)
        a_ = np.array([v[0] if isinstance(v, tuple) else np.nan for v in ab]); b_ = np.array([v[1] if isinstance(v, tuple) else np.nan for v in ab])
        n["r"] = np.exp((np.log(n.on.values) - a_) / b_) / n.target
    else:
        n["r"] = n.on / (n.target * br_n)
    nt = n[n.time_since_intake_on.notna()]
    mids, meds = [], []
    for lo, hi in zip(T_BINS[:-1], T_BINS[1:]):
        s = nt[(nt.time_since_intake_on >= lo) & (nt.time_since_intake_on < hi)]
        if len(s) > 30:
            mids.append(s.time_since_intake_on.median()); meds.append(s.r.median())
    f_missing = float(n[n.time_since_intake_on.isna()].r.median())
    # LEDD/target ratio as function of disease duration (population)
    l = t[t.ledd.notna()]
    foff = None
    if "toffc" in VARIANT:
        o3 = o[o.off > 0].copy()
        o3["lu"] = np.log(o3.off / (o3.target * o3.rater_id.map(b_r).astype(float)))
        TB = [0, 10, 11, 12, 13, 14, 15, 16, 18, 40]
        o3["tb"] = pd.cut(o3.time_since_intake_off, TB)
        ref = o3[o3.time_since_intake_off.isna()].lu.quantile(0.75)
        qb = o3.groupby("tb", observed=True).lu.quantile(0.75) - ref
        foff = (TB, qb)
    return dict(b_r=b_r, b_glob=b_glob, t_mid=np.array(mids), f_med=np.array(meds),
                f_missing=f_missing, aff=aff, foff=foff)


def apply_calib(d, cal):
    d = d.copy()
    b = d.rater_id.map(cal["b_r"]).astype(float).fillna(cal["b_glob"])
    d["b_r"] = b
    d["off_c"] = d.off / b
    f = np.interp(d.time_since_intake_on.fillna(-1), cal["t_mid"], cal["f_med"])
    f = np.where(d.time_since_intake_on.isna(), cal["f_missing"], f)
    d["f_t"] = f
    d["on_c"] = d.on / (b * f)
    if cal.get("foff") is not None:
        TB, qb = cal["foff"]
        tb = pd.cut(d.time_since_intake_off, TB)
        adj = tb.map(qb).astype(float).fillna(0.0).values
        d["off_c"] = d.off_c / np.exp(adj)
        d["off_tadj"] = adj
    if cal.get("aff") is not None:
        ab = d.rater_id.astype(str).map(cal["aff"])
        a_ = np.array([v[0] if isinstance(v, tuple) else np.nan for v in ab]); b_ = np.array([v[1] if isinstance(v, tuple) else np.nan for v in ab])
        ok_ = ~np.isnan(a_)
        offc = np.where(ok_ & (d.off.values > 0), np.exp((np.log(np.where(d.off.values > 0, d.off.values, 1.0)) - a_) / b_), d.off_c.values)
        offc = np.where(d.off.values == 0, 0.0, offc)
        onr = np.where(ok_ & (d.on.values > 0), np.exp((np.log(np.where(d.on.values > 0, d.on.values, 1.0)) - a_) / b_), d.on.values / b)
        d["off_c_lin"] = d.off_c; d["on_c_lin"] = d.on_c
        d["off_c"] = np.where(d.off.isna(), np.nan, offc)
        d["on_c"] = np.where(d.on.isna(), np.nan, onr / f)
        d["r_slope"] = b_; d["r_icpt"] = a_
    return d


# ---------------------------------------------------------------- patient features (no target)
def robust_poly(age, val, deg, w=None, iters=3):
    """Weighted poly fit with Huber-like reweighting. returns coeffs or None."""
    if len(age) < deg + 1:
        return None
    w0 = np.ones(len(age)) if w is None else w.copy()
    ww = w0.copy()
    for _ in range(iters):
        try:
            c = np.polyfit(age, val, deg, w=np.sqrt(ww))
        except Exception:
            return None
        r = val - np.polyval(c, age)
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        k = 1.5 * s
        ww = w0 * np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return c


def on_correction(d):
    """Patient-specific drug response, estimated without target."""
    d = d.sort_values(["patient_id", "age"]).copy()
    prior = PRIOR
    if VARIANT.startswith("onk2") or VARIANT.startswith("onk3"):
        # off reference = robust linear fit of off_c vs age, per patient (uses all visits)
        off_ref = np.full(len(d), np.nan)
        pos = {idx: i for i, idx in enumerate(d.index)}
        for pid, gg in d.groupby("patient_id", observed=True):
            o = gg[gg.off_c.notna() & (gg.off_c > 1)]
            if len(o) == 0:
                continue
            deg = 1 if len(o) >= 3 else 0
            c = robust_poly(o.age.values, o.off_c.values, deg)
            if c is None:
                continue
            off_ref[[pos[i] for i in gg.index]] = np.polyval(c, gg.age.values)
        d["off_ref"] = off_ref
        ref = d.off_ref.where(d.off_ref > 1)
        # same-visit off is more precise when available: blend
        ref = np.where(d.off_c.notna() & (d.off_c > 1), 0.5 * d.off_c + 0.5 * ref.fillna(d.off_c), ref)
        ref = pd.Series(ref, index=d.index)
    else:
        ref = d.off_c.where(d.off_c > 1)
    ok = d.on_c.notna() & ref.notna()
    if VARIANT.startswith("onk2"):
        # effect-scaling model: on_b/target = 1 - s_i * E(t),  E = 1 - f(t)
        E = (1 - d.f_t).clip(0.05, None)
        on_b = d.on / d.b_r
        sv = ((1 - on_b / ref) / E).where(ok & (E > 0.2)).clip(-1, 3)
        n = sv.notna().groupby(d.patient_id, observed=True).transform("sum")
        sm = sv.groupby(d.patient_id, observed=True).transform("sum").fillna(0)
        s_i = (sm + prior * 1.0) / (n + prior)
        d["on_s"] = s_i; d["on_k_n"] = n
        d["on_c"] = on_b / (1 - s_i * E).clip(0.15, None)
    else:
        lr = np.log((d.on_c / ref).where(ok).clip(0.05, 20))
        sm = lr.groupby(d.patient_id, observed=True).transform("sum").fillna(0)
        n = ok.groupby(d.patient_id, observed=True).transform("sum")
        if "kp" in VARIANT:
            P = pd.DataFrame({"lk": lr, "pid": d.patient_id.astype(str)}).groupby("pid").lk.agg(["mean", "count"])
            g_ = d.groupby(d.patient_id.astype(str))
            F = pd.DataFrame({"aad": g_.age_at_diagnosis.first(), "sex": g_.sexM.first(), "am": g_.age.mean(),
                              "ledd": g_.ledd.mean(), "ledd_n": g_.ledd.count(), "n_on": g_.on.count(), "n_off": g_.off.count(),
                              "t_on": g_.time_since_intake_on.mean(), "onmean": g_.on_c.mean(), "offmean": g_.off_c.mean(),
                              "nv": g_.age.size()})
            F["dd"] = F.am - F.aad
            F = F.join(P)
            trn = F[F["count"] >= 3]
            fc = [c for c in F.columns if c not in ("mean", "count")]
            km = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1, n_jobs=INNER_JOBS).fit(trn[fc], trn["mean"])
            lkp = pd.Series(km.predict(F[fc]), index=F.index)
            pk = d.patient_id.astype(str).map(lkp).values
            d["on_k_prior"] = np.exp(pk)
            k = np.exp((sm + prior * pk) / (n + prior))
        else:
            k = np.exp(sm / (n + prior))
        d["on_k"] = k; d["on_k_n"] = n
        d["on_c"] = d.on_c / k
    return d.sort_index()


def patient_features(d):
    if "onk" in VARIANT:
        d = on_correction(d)
    d = d.sort_values(["patient_id", "age"]).copy()
    g = d.groupby("patient_id", observed=True)
    d["visit_idx"] = g.cumcount()
    d["n_vis"] = g.age.transform("size")
    d["age_c"] = d.age - g.age.transform("mean")
    d["age_span"] = g.age.transform("max") - g.age.transform("min")
    for c in ["off_c", "on_c"]:
        d[f"p_{c}_mean"] = g[c].transform("mean")
        d[f"p_{c}_med"] = g[c].transform("median")
        d[f"p_{c}_max"] = g[c].transform("max")
        d[f"p_{c}_std"] = g[c].transform("std")
        d[f"p_{c}_n"] = g[c].transform("count")
        d[f"{c}_prev"] = g[c].shift(1)
        d[f"{c}_next"] = g[c].shift(-1)
        d[f"{c}_ffill"] = g[c].ffill()
        d[f"{c}_bfill"] = g[c].bfill()
    d["p_ledd_mean"] = g.ledd.transform("mean")
    d["p_ledd_max"] = g.ledd.transform("max")
    d["p_ledd_n"] = g.ledd.transform("count")
    d["ledd_prev"] = g.ledd.shift(1)
    d["ledd_next"] = g.ledd.shift(-1)
    d["off_cummax"] = g.off_c.cummax()

    # combined robust curve of corrected measurements vs age
    curve_vals = np.full(len(d), np.nan)
    curve_lin = np.full(len(d), np.nan)
    curve_slope = np.full(len(d), np.nan)
    ledd_scaled = np.full(len(d), np.nan)
    ledd_k = np.full(len(d), np.nan)
    n_pts = np.zeros(len(d))
    pos = {idx: i for i, idx in enumerate(d.index)}
    wt_on = 1.0 if VARIANT == "base" else 1.0
    for pid, gg in d.groupby("patient_id", observed=True):
        ages, vals, ws = [], [], []
        o = gg[gg.off_c.notna()]
        ages += list(o.age); vals += list(o.off_c); ws += [1.0] * len(o)
        n = gg[gg.on_c.notna()]
        ages += list(n.age); vals += list(n.on_c); ws += [wt_on] * len(n)
        ages = np.array(ages); vals = np.array(vals); ws = np.array(ws)
        rows = [pos[i] for i in gg.index]
        n_pts[rows] = len(ages)
        if len(ages) == 0:
            continue
        deg = 2 if len(ages) >= 6 else (1 if len(ages) >= 3 else 0)
        c = robust_poly(ages, vals, deg, ws)
        if c is not None:
            curve_vals[rows] = np.polyval(c, gg.age.values)
        c1 = robust_poly(ages, vals, 1 if len(ages) >= 2 else 0, ws)
        if c1 is not None:
            curve_lin[rows] = np.polyval(c1, gg.age.values)
            curve_slope[rows] = c1[0] if len(c1) == 2 else 0.0
        # LEDD-shape estimate: scale ledd by patient's robust ratio est/ledd
        if gg.ledd.notna().sum() >= 1:
            lmap = dict(zip(gg.age, gg.ledd))
            ratios = [v / lmap[a] for a, v in zip(ages, vals) if a in lmap and lmap[a] > 0 and not np.isnan(lmap[a])]
            if len(ratios) >= 1:
                k = np.median(ratios)
                ledd_k[rows] = k
                ledd_scaled[rows] = gg.ledd.values * k
    d["curve"] = curve_vals
    d["curve_lin"] = curve_lin
    d["curve_slope"] = curve_slope
    d["ledd_scaled"] = ledd_scaled
    d["ledd_k"] = ledd_k
    d["curve_npts"] = n_pts
    d["off_resid"] = d.off_c - d.curve
    d["on_resid"] = d.on_c - d.curve
    return d.sort_index()


def fit_density(tr_feat):
    t=tr_feat[tr_feat.target>3]
    u=(t.z-np.log(t.target)).dropna().values
    grid=np.linspace(-4,1.5,551); bw=0.04
    dens=np.zeros_like(grid)
    hist,edges=np.histogram(u,bins=np.linspace(-4,1.5,1101),density=True)
    mids=(edges[:-1]+edges[1:])/2
    for m,h in zip(mids,hist):
        if h>0: dens+=h*np.exp(-0.5*((grid-m)/bw)**2)
    dens/=np.trapezoid(dens,grid); dens=np.maximum(dens,1e-4)
    return grid, np.log(dens)

def add_z(d):
    d=d.copy()
    zo=np.log(d.off_c.where(d.off_c>0)); zn=np.log(d.on_c.where(d.on_c>0))
    d['z']=zo.fillna(zn) if "offz" in VARIANT else pd.concat([zo,zn],axis=1).mean(axis=1)
    if "zboth" in VARIANT:
        d['z_mean2']=pd.concat([zo,zn],axis=1).mean(axis=1)
    return d

def mle_curves(d, grid, logp, slope_mu, slope_sd):
    out=np.full(len(d),np.nan); outs=np.full(len(d),np.nan)
    d=d.reset_index(drop=True)
    for pid,g in d.groupby('patient_id',observed=True):
        m=g.z.notna().values
        if m.sum()==0: continue
        a=g.age.values; am=a[m].mean(); x=a[m]-am; z=g.z.values[m]
        deg2 = m.sum()>=6
        def nll(p):
            lt=p[0]+p[1]*x+(p[2]*x*x if deg2 else 0)
            r=z-lt
            lp=np.interp(r,grid,logp,left=logp[0],right=logp[-1])
            pen=0.5*((p[1]-slope_mu)/slope_sd)**2 + (0.5*(p[2]/0.01)**2 if deg2 else 0)
            return -lp.sum()+pen
        p0=np.array([np.median(z)+0.1, slope_mu, 0.0])
        best=None
        for da in (-0.2,0.0,0.2):
            r=minimize(nll,p0+np.array([da,0,0]),method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-4,maxiter=600))
            if best is None or r.fun<best.fun: best=r
        p=best.x; xa=a-am
        out[g.index.values]=np.exp(p[0]+p[1]*xa+(p[2]*xa*xa if deg2 else 0))
        outs[g.index.values]=p[1]
    return out, outs



PIDS=[]
EDG=np.log(np.array([0.5,8,15,25,35,45,55,70,200]))
GR=np.linspace(-4,1.5,551)
def kde(u,bw=0.05):
    h,e=np.histogram(u,bins=np.linspace(-4,1.5,1101),density=True); m=(e[:-1]+e[1:])/2
    dens=(h[None,:]*np.exp(-0.5*((GR[:,None]-m[None,:])/bw)**2)).sum(1)
    dens/=np.trapezoid(dens,GR); return np.log(np.maximum(dens,1e-5))
def fit_noise(t):
    t=t[(t.target>0.5)&t.z.notna()]; lt=np.log(t.target.values); u=t.z.values-lt
    tabs=[];cent=[]
    for lo,hi in zip(EDG[:-1],EDG[1:]):
        k=(lt>=lo)&(lt<hi); tabs.append(kde(u[k]) if k.sum()>200 else kde(u)); cent.append(np.median(lt[k]) if k.sum()>0 else (lo+hi)/2)
    return np.array(tabs),np.array(cent)
def logp(u,lt,tabs,cent):
    # u, lt arrays same shape; bilinear lookup
    w=np.interp(lt,cent,np.arange(len(cent))); i0=np.clip(np.floor(w).astype(int),0,len(cent)-2); a=np.clip(w-i0,0,1)
    gi=np.clip((u-GR[0])/(GR[1]-GR[0]),0,len(GR)-1.001); j0=gi.astype(int); b=gi-j0
    T=tabs
    l00=T[i0,j0]; l01=T[i0,j0+1]; l10=T[i0+1,j0]; l11=T[i0+1,j0+1]
    return (1-a)*((1-b)*l00+b*l01)+a*((1-b)*l10+b*l11)
def curves(t):
    rows=[]; PIDS.clear()
    for pid,g in t.groupby('patient_id',observed=True):
        if len(g)<3 or (g.target<=0.3).any(): continue
        am=g.age.mean(); x=g.age.values-am; deg=2 if len(g)>=4 else 1
        c=np.polyfit(x,np.log(g.target.values),deg); 
        if deg==1: c=np.r_[0,c]
        rows.append((c[2],c[1],c[0],am-g.age_at_diagnosis.iloc[0],am,(g.age.max()-g.age.min())/2+0.5)); PIDS.append(pid)
    return np.array(rows)

def vtype(t):
    return np.where(t.off_c.notna().values & t.on_c.notna().values, 2, np.where(t.off_c.notna().values, 0, 1)).astype(int)


def fit_noise_typed(t):
    t = t[(t.target > 0.5) & t.z.notna()]; ty = vtype(t)
    return {v: fit_noise(t[ty == v]) for v in np.unique(ty)}


def eb_feature(f, tabs, cent, C, pids_C, exclude_self, TY=None):
    A,B,Cc,DD,AM,HS=C.T
    out=np.full(len(f),np.nan); ent=np.full(len(f),np.nan)
    fr=f.reset_index(drop=True)
    for pid,g in fr.groupby('patient_id',observed=True):
        m=g.z.notna().values; ages=g.age.values; amp=ages[m].mean() if m.any() else ages.mean(); x=ages-amp
        ddp=amp-g.age_at_diagnosis.iloc[0]
        prior=-0.5*((AM-amp)/8)**2 + (0 if np.isnan(ddp) else np.where(np.isnan(DD),-0.5,-0.5*((DD-ddp)/1.2)**2))
        if exclude_self: prior=np.where(pids_C==pid,-np.inf,prior)
        if m.any():
            xm=x[m]; z=g.z.values[m]
            xq=np.clip(xm[None,:],-HS[:,None],HS[:,None]); LT=A[:,None]+B[:,None]*xm[None,:]+Cc[:,None]*(xq**2+2*xq*(xm[None,:]-xq))
            if TY is None: ll=logp(z[None,:]-LT,LT,tabs,cent).sum(1)
            else:
                tv=vtype(g)[m]; ll=0
                for v in np.unique(tv):
                    mm=tv==v; tb,ce=TY.get(v,(tabs,cent)); ll=ll+logp(z[None,mm]-LT[:,mm],LT[:,mm],tb,ce).sum(1)
        else: ll=0
        lw=prior+ll; w=np.exp(lw-lw.max()); w/=w.sum()
        xq=np.clip(x[None,:],-HS[:,None],HS[:,None]); LTa=A[:,None]+B[:,None]*x[None,:]+Cc[:,None]*(xq**2+2*xq*(x[None,:]-xq))
        out[g.index.values]=(w[:,None]*np.exp(LTa)).sum(0)
        ent[g.index.values]=1.0/np.sum(w**2)
    return out, ent


DROP = {"Index", "patient_id", "target", "on", "off", "z"}


def feat_cols(d):
    drop = set(DROP)
    if "norater" in VARIANT:
        drop.add("rater_id")
    return [c for c in d.columns if c not in drop]


def smooth_poly2(d, pred):
    out = pred.copy()
    tmp = pd.DataFrame({"pid": d.patient_id.values, "age": d.age.values, "p": pred})
    for pid, gg in tmp.groupby("pid", observed=True):
        n = len(gg)
        deg = 2 if n >= 4 else 1
        if n < 2:
            continue
        c = np.polyfit(gg.age, gg.p, deg)
        out[gg.index.values] = np.polyval(c, gg.age)
    return out


PARAMS = dict(n_estimators=1500, learning_rate=0.02, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              n_jobs=INNER_JOBS, verbose=-1, random_state=SEED)

if "hpA" in VARIANT:
    PARAMS.update(num_leaves=47, min_child_samples=80, colsample_bytree=0.4, objective="huber", alpha=6.0)
if "hpB" in VARIANT:
    PARAMS.update(num_leaves=31, min_child_samples=100, colsample_bytree=0.5)
if "reg" in VARIANT:
    PARAMS.update(num_leaves=31, min_child_samples=150, colsample_bytree=0.5, n_estimators=2500, reg_lambda=5.0)
RESID = "resid" in VARIANT
# ---------------------------------------------------------------- estimateur bayesien par patient
# Constat sur y_train : par patient (trie par age) target = cubique de l'age centre, et quand la cubique
# stagne ou baisse le target monte de x1.02 par visite. Chaque visite donne une mesure bruitee z (log).
# On calcule la moyenne a posteriori de T a chaque visite (vraisemblance = densite empirique du bruit).
B_S = 3000
B_EDGES = np.linspace(-4, 0.6, 231)
B_CENT = (B_EDGES[:-1] + B_EDGES[1:]) / 2


def _bayes_logdens(r):
    h, _ = np.histogram(r, bins=B_EDGES)
    k = np.exp(-0.5 * (np.arange(-8, 9) / 2.0) ** 2); k /= k.sum()
    h = np.convolve(h.astype(float), k, mode="same") + 1e-5
    h = h / (h.sum() * (B_EDGES[1] - B_EDGES[0]))
    return np.log(h)


def _bayes_curves(A, u):
    C = A[:, [0]] + A[:, [1]] * u + A[:, [2]] * u ** 2 + A[:, [3]] * u ** 3
    T = np.empty_like(C)
    T[:, 0] = np.maximum(C[:, 0], 0.0)
    for i in range(1, C.shape[1]):
        T[:, i] = np.where(C[:, i] > T[:, i - 1], C[:, i], RATCHET * T[:, i - 1])
    return T


def add_bayes(ff, of):
    t0_ = time.time()
    # 1. densite du bruit (ff seulement)
    t = ff[(ff.target > 3) & ff.z.notna()]
    r = (t.z - np.log(t.target)).values
    is_off = (t.off_c > 0).values
    ld_off = _bayes_logdens(r[is_off]); ld_on = _bayes_logdens(r[~is_off])
    # 2. prior des courbes (ff seulement)
    rows = []
    for _, g in ff.groupby("patient_id", observed=True):
        if len(g) < 4 or not (g.target > 0).all():
            continue
        u = g.age.values - g.age.values.mean()
        c = np.polyfit(u, g.target.values, 3)[::-1]
        rows.append((c[0], c[1], c[2], c[3], g.dd.mean(), g.ledd.notna().mean()))
    P = np.array(rows, float)
    dd_med = float(np.nanmedian(P[:, 4]))
    P[np.isnan(P[:, 4]), 4] = dd_med
    Xr = np.c_[np.ones(len(P)), P[:, 4], P[:, 5]]
    beta = np.linalg.lstsq(Xr, P[:, 0], rcond=None)[0]
    V = np.c_[P[:, 0] - Xr @ beta, P[:, 1:4]]
    mu = V.mean(0); sig = np.cov(V.T) * 1.5
    prior = multivariate_normal(mu, sig, allow_singular=True)
    prop13 = multivariate_normal(mu[1:], 2 * sig[1:, 1:], allow_singular=True)
    # 3. posterieur par patient (pour ff et of, prior et densite appris sur ff)
    for f in (ff, of):
        bm = np.full(len(f), np.nan); bs = np.full(len(f), np.nan); be = np.full(len(f), np.nan)
        for _, g in f.groupby("patient_id", observed=True):
            g = g.iloc[np.argsort(g.age.values, kind="stable")]
            pos = f.index.get_indexer(g.index)
            rng = np.random.default_rng(0)
            u = g.age.values - g.age.values.mean()
            z = g.z.values.astype(float); has = ~np.isnan(z)
            typ_off = (g.off_c.values > 0)
            ddm = g.dd.mean(); ddm = dd_med if np.isnan(ddm) else ddm
            a0_hat = beta[0] + beta[1] * ddm + beta[2] * g.ledd.notna().mean()
            off_ = np.array([a0_hat, 0.0, 0.0, 0.0])
            if not has.any():
                A = prior.rvs(B_S, random_state=rng) + off_
                w = np.full(B_S, 1.0 / B_S)
                T = _bayes_curves(A, u)
            else:
                zi = z[has]; oi = typ_off[has]

                def loglik(T):
                    rr = zi[None, :] - np.log(np.maximum(T[:, has], 0.5))
                    l = np.where(oi[None, :], np.interp(rr, B_CENT, ld_off), np.interp(rr, B_CENT, ld_on))
                    return l.sum(1)

                L = max(float(np.exp(np.quantile(zi, 0.75))), 1.0)
                a0 = rng.normal(L, 0.3 * L, B_S)
                a13 = prop13.rvs(B_S, random_state=rng)
                A = np.c_[a0, a13]
                lq = norm.logpdf(a0, L, 0.3 * L) + prop13.logpdf(a13)
                T = _bayes_curves(A, u)
                lw = loglik(T) + prior.logpdf(A - off_) - lq
                w = np.exp(lw - lw.max()); w /= w.sum()
                m2 = w @ A
                C2 = ((A - m2) * w[:, None]).T @ (A - m2) * 1.5 + 1e-6 * np.eye(4)
                q2 = multivariate_normal(m2, C2, allow_singular=True)
                A = q2.rvs(B_S, random_state=rng)
                T = _bayes_curves(A, u)
                lw = loglik(T) + prior.logpdf(A - off_) - q2.logpdf(A)
                w = np.exp(lw - lw.max()); w /= w.sum()
            mT = w @ T
            bm[pos] = mT
            bs[pos] = np.sqrt(np.maximum(w @ (T - mT) ** 2, 0))
            be[pos] = 1.0 / np.sum(w ** 2) / B_S
        f["bayes_mean"] = bm; f["bayes_sd"] = bs
        f["bayes_logmean"] = np.log(np.clip(bm, 0.5, None)); f["bayes_ess"] = be
    print(f"   add_bayes {time.time() - t0_:.0f}s", flush=True)
    return ff, of


def make_feats(fit_df, other_df):
    cal = fit_calib(fit_df)
    ff = patient_features(apply_calib(fit_df, cal))
    of = patient_features(apply_calib(other_df, cal))
    if "mle" in VARIANT:
        ff = add_z(ff); of = add_z(of)
        grid, logp = fit_density(ff)
        sl = [np.polyfit(g.age, np.log(g.target), 1)[0] for _, g in ff.groupby("patient_id", observed=True) if len(g) >= 4 and (g.target > 3).all()]
        smu, ssd = np.median(sl), np.std(sl)
        for f in (ff, of):
            p_, s_ = mle_curves(f, grid, logp, smu, ssd)
            f["mle"] = p_; f["mle_slope"] = s_
            f["z_resid"] = f.z - np.log(f.mle)
            if "zq" in VARIANT:
                g = f.groupby("patient_id", observed=True)
                am = g.age.transform("mean")
                zd = f.z - f.mle_slope * (f.age - am)          # detrended log measurement
                f["_zd"] = zd
                g = f.groupby("patient_id", observed=True)
                for q in (0.5, 0.75, 0.9):
                    f[f"zd_q{int(q*100)}"] = np.exp(g._zd.transform(lambda s: s.quantile(q)) + f.mle_slope * (f.age - am))
                f["zd_max"] = np.exp(g._zd.transform("max") + f.mle_slope * (f.age - am))
                f["zd_top2"] = np.exp(g._zd.transform(lambda s: s.nlargest(2).mean()) + f.mle_slope * (f.age - am))
                f["zd_std"] = g._zd.transform("std")
                f["zd_n"] = g._zd.transform("count")
                f.drop(columns="_zd", inplace=True)
            if "loc" in VARIANT:
                f.sort_values(["patient_id", "age"], inplace=True)
                am = f.groupby("patient_id", observed=True).age.transform("mean")
                tr_ = f.mle_slope * (f.age - am)
                f["_zd"] = f.z - tr_
                g = f.groupby("patient_id", observed=True)
                for w in (1, 2):
                    r = g._zd.rolling(2 * w + 1, center=True, min_periods=1)
                    f[f"loc{w}_max"] = np.exp(r.max().reset_index(level=0, drop=True) + tr_)
                    f[f"loc{w}_mean"] = np.exp(r.mean().reset_index(level=0, drop=True) + tr_)
                    f[f"loc{w}_n"] = r.count().reset_index(level=0, drop=True)
                f["z_cummax"] = np.exp(g._zd.cummax() + tr_)
                f["z_revcummax"] = np.exp(f.iloc[::-1].groupby("patient_id", observed=True)._zd.cummax().iloc[::-1] + tr_)
                f["z_cummax_raw"] = np.exp(g.z.cummax())
                f["z_revcummin_raw"] = np.exp(f.iloc[::-1].groupby("patient_id", observed=True).z.cummin().iloc[::-1])
                f.drop(columns="_zd", inplace=True)
                f.sort_index(inplace=True)
            if "loc2" in VARIANT:
                f.sort_values(["patient_id", "age"], inplace=True)
                am = f.groupby("patient_id", observed=True).age.transform("mean")
                tr_ = f.mle_slope * (f.age - am)
                f["_zd"] = f.z - tr_
                f["_ro"] = np.log(f.off_c.where(f.off_c > 0)) - tr_
                f["_rn"] = np.log(f.on_c.where(f.on_c > 0)) - tr_
                g = f.groupby("patient_id", observed=True)
                for w in (3,):
                    r = g._zd.rolling(2 * w + 1, center=True, min_periods=1)
                    f[f"loc{w}_max"] = np.exp(r.max().reset_index(level=0, drop=True) + tr_)
                    f[f"loc{w}_mean"] = np.exp(r.mean().reset_index(level=0, drop=True) + tr_)
                for w in (1, 2):
                    r = g._zd.rolling(2 * w + 1, center=True, min_periods=2)
                    f[f"loc{w}_q75"] = np.exp(r.quantile(0.75).reset_index(level=0, drop=True) + tr_)
                for nm in ("_ro", "_rn"):
                    r = g[nm].rolling(5, center=True, min_periods=1)
                    f[f"loc2{nm}_max"] = np.exp(r.max().reset_index(level=0, drop=True) + tr_)
                    f[f"loc2{nm}_mean"] = np.exp(r.mean().reset_index(level=0, drop=True) + tr_)
                for sh in (1, 2, -1, -2):
                    f[f"zd_sh{sh}"] = np.exp(g._zd.shift(sh) + tr_)
                f.drop(columns=["_zd", "_ro", "_rn"], inplace=True)
                f.sort_index(inplace=True)
            if "zq2" in VARIANT:
                lm = np.log(f.mle)
                f["_r"] = f.z - lm
                f["_ro"] = np.log(f.off_c.where(f.off_c > 0)) - lm
                f["_rn"] = np.log(f.on_c.where(f.on_c > 0)) - lm
                g = f.groupby("patient_id", observed=True)
                for nm in ("_r", "_ro", "_rn"):
                    for q in (0.6, 0.8, 0.95):
                        f[f"e{nm}_q{int(q*100)}"] = np.exp(lm + g[nm].transform(lambda s: s.quantile(q)))
                    f[f"e{nm}_max"] = np.exp(lm + g[nm].transform("max"))
                    f[f"e{nm}_top3"] = np.exp(lm + g[nm].transform(lambda s: s.nlargest(3).mean()))
                    f[f"e{nm}_n"] = g[nm].transform("count")
                f.drop(columns=["_r", "_ro", "_rn"], inplace=True)
        if "eb" in VARIANT:
            tabs, cent = fit_noise(ff); C = curves(ff); pc = np.array(PIDS)
            for f, ex in ((ff, True), (of, False)):
                e_, n_ = eb_feature(f, tabs, cent, C, pc, ex, fit_noise_typed(ff))
                f["eb"] = e_; f["eb_neff"] = n_
        if "loc3" in VARIANT:
            for f in (ff, of):
                f.sort_values(["patient_id", "age"], inplace=True)
                le = np.log(f.eb.clip(0.5))
                f["_re"] = f.z - le
                g = f.groupby("patient_id", observed=True)
                for q in (0.75, 0.9):
                    f[f"eb_q{int(q*100)}"] = np.exp(le + g._re.transform(lambda s: s.quantile(q)))
                f["eb_max"] = np.exp(le + g._re.transform("max"))
                f["eb_top2"] = np.exp(le + g._re.transform(lambda s: s.nlargest(2).mean()))
                for w in (1, 2):
                    r = g._re.rolling(2 * w + 1, center=True, min_periods=1)
                    f[f"ebloc{w}_max"] = np.exp(le + r.max().reset_index(level=0, drop=True))
                    f[f"ebloc{w}_mean"] = np.exp(le + r.mean().reset_index(level=0, drop=True))
                f["eb_cummax"] = np.exp(le + g._re.cummax())
                f["eb_revcummax"] = np.exp(le + f.iloc[::-1].groupby("patient_id", observed=True)._re.cummax().iloc[::-1])
                f.drop(columns="_re", inplace=True)
                f.sort_index(inplace=True)
        if "locmle" in VARIANT or "iso" in VARIANT:
            gridd, logpd = fit_density(ff)
            AG = np.linspace(-1.5, 1.5, 301)
            for f in (ff, of):
                f.sort_values(["patient_id", "age"], inplace=True)
                am = f.groupby("patient_id", observed=True).age.transform("mean")
                tr_ = (f.mle_slope * (f.age - am)).values
                zd = (f.z.values - tr_)
                lm = np.log(f.mle.values)
                base_ = lm - tr_          # patient level (log) at mean age from mle
                pidv = f.patient_id.astype(str).values
                outs = {k: np.full(len(f), np.nan) for k in ("lm3", "lm5", "lmall", "iso", "isoup")}
                starts = np.r_[0, np.where(pidv[1:] != pidv[:-1])[0] + 1, len(f)]
                for s0, s1 in zip(starts[:-1], starts[1:]):
                    zz = zd[s0:s1]; b0 = base_[s0]; ok = ~np.isnan(zz)
                    if "locmle" in VARIANT and ok.any():
                        # log-lik table: L[i, a] = logp(zz_i - (b0 + AG_a))
                        L = np.where(ok[:, None], np.interp(zz[:, None] - (b0 + AG[None, :]), gridd, logpd), 0.0)
                        cs = np.vstack([np.zeros(len(AG)), np.cumsum(L, 0)])
                        n = len(zz)
                        for w, key in ((1, "lm3"), (2, "lm5")):
                            for i in range(n):
                                lo, hi = max(0, i - w), min(n, i + w + 1)
                                if ok[lo:hi].any():
                                    outs[key][s0 + i] = b0 + AG[np.argmax(cs[hi] - cs[lo])]
                        outs["lmall"][s0:s1] = b0 + AG[np.argmax(cs[-1])]
                    if "iso" in VARIANT and ok.sum() >= 2:
                        from sklearn.isotonic import IsotonicRegression
                        x_ = np.where(ok)[0]; zr = f.z.values[s0:s1][ok]
                        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
                        fit1 = ir.fit_transform(x_, zr)
                        zc = np.maximum(zr, fit1 - 0.15)          # pull low outliers up, refit -> upper-ish envelope
                        fit2 = ir.fit_transform(x_, zc)
                        outs["iso"][s0 + x_] = fit1; outs["isoup"][s0 + x_] = fit2
                for key in ("lm3", "lm5", "lmall"):
                    if "locmle" in VARIANT:
                        f[f"locmle_{key}"] = np.exp(outs[key] + tr_)
                if "iso" in VARIANT:
                    f["iso_z"] = pd.Series(np.exp(outs["iso"]), index=f.index).groupby(f.patient_id, observed=True).transform(lambda s: s.interpolate(limit_direction="both"))
                    f["iso_up"] = pd.Series(np.exp(outs["isoup"]), index=f.index).groupby(f.patient_id, observed=True).transform(lambda s: s.interpolate(limit_direction="both"))
                f.sort_index(inplace=True)
        if "rel" in VARIANT:
            t_ = ff[(ff.target > 3) & ff.z.notna()]
            u_ = t_.z - np.log(t_.target)
            rs = pd.DataFrame({"u": u_.values, "r": t_.rater_id.astype(str).values}).groupby("r").u
            st = pd.DataFrame({"r_med": rs.median(), "r_sd": rs.std(), "r_q10": rs.quantile(0.1), "r_q90": rs.quantile(0.9)})
            if "rel2" in VARIANT:
                st["r_q75"] = rs.quantile(0.75); st["r_q95"] = rs.quantile(0.95); st["r_q25"] = rs.quantile(0.25)
                st["r_ptail"] = rs.apply(lambda x: (x < -0.5).mean())
                # upper-half spread (sharpness of the upper edge)
                st["r_upsd"] = rs.apply(lambda x: x[x > x.median()].std())
            for f in (ff, of):
                rr = f.rater_id.astype(str)
                for c in st.columns:
                    f[c] = rr.map(st[c]).astype(float).values
                f.sort_values(["patient_id", "age"], inplace=True)
                am = f.groupby("patient_id", observed=True).age.transform("mean")
                tr_ = f.mle_slope * (f.age - am)
                f["_zr"] = f.z - f.r_q90 - tr_          # rater-specific upper-edge alignment
                g = f.groupby("patient_id", observed=True)
                f["zr_max"] = np.exp(g._zr.transform("max") + tr_)
                f["zr_top2"] = np.exp(g._zr.transform(lambda s: s.nlargest(2).mean()) + tr_)
                f["zr_mean"] = np.exp(g._zr.transform("mean") + tr_)
                r = g._zr.rolling(3, center=True, min_periods=1)
                f["zr_loc1max"] = np.exp(r.max().reset_index(level=0, drop=True) + tr_)
                f["zr_cummax"] = np.exp(g._zr.cummax() + tr_)
                f["zr_revcummax"] = np.exp(f.iloc[::-1].groupby("patient_id", observed=True)._zr.cummax().iloc[::-1] + tr_)
                if "rel3" in VARIANT:
                    ty_ = vtype(f)
                    qt = {}
                    tt_ = ff[(ff.target > 3) & ff.z.notna()]; ut_ = (tt_.z - np.log(tt_.target)).values; tyt_ = vtype(tt_)
                    rrt = tt_.rater_id.astype(str).values
                    for v_ in (0, 1, 2):
                        qt[v_] = np.quantile(ut_[tyt_ == v_], 0.9) if (tyt_ == v_).sum() > 50 else np.quantile(ut_, 0.9)
                    # rater x type q90 (shrunk to type q90)
                    dfq = pd.DataFrame({"u": ut_, "r": rrt, "t": tyt_})
                    rq = dfq.groupby(["r", "t"]).u.agg(["count", lambda x: x.quantile(0.9)]); rq.columns = ["n", "q"]
                    keyv = list(zip(f.rater_id.astype(str).values, ty_))
                    q_rt = np.array([ (rq.loc[k_, "q"] * rq.loc[k_, "n"] + qt[k_[1]] * 50) / (rq.loc[k_, "n"] + 50) if k_ in rq.index else qt[k_[1]] for k_ in keyv])
                    f["q_rt"] = q_rt
                    f["_zt"] = f.z - q_rt - tr_
                    g3 = f.groupby("patient_id", observed=True)
                    f["zt_max"] = np.exp(g3._zt.transform("max") + tr_)
                    f["zt_top2"] = np.exp(g3._zt.transform(lambda s: s.nlargest(2).mean()) + tr_)
                    f["zt_loc1max"] = np.exp(g3._zt.rolling(3, center=True, min_periods=1).max().reset_index(level=0, drop=True) + tr_)
                    f["zt_loc2max"] = np.exp(g3._zt.rolling(5, center=True, min_periods=1).max().reset_index(level=0, drop=True) + tr_)
                    f["zt_cummax"] = np.exp(g3._zt.cummax() + tr_)
                    f["zt_revcummax"] = np.exp(f.iloc[::-1].groupby("patient_id", observed=True)._zt.cummax().iloc[::-1] + tr_)
                    f.drop(columns="_zt", inplace=True)
                if "rel2" in VARIANT:
                    for w in (2, 3):
                        r = g._zr.rolling(2 * w + 1, center=True, min_periods=1)
                        f[f"zr_loc{w}max"] = np.exp(r.max().reset_index(level=0, drop=True) + tr_)
                        f[f"zr_loc{w}mean"] = np.exp(r.mean().reset_index(level=0, drop=True) + tr_)
                    f["_zr2"] = f.z - f.r_q75 - tr_
                    g2 = f.groupby("patient_id", observed=True)
                    f["zr2_top2"] = np.exp(g2._zr2.transform(lambda s: s.nlargest(2).mean()) + tr_)
                    f["zr2_q75"] = np.exp(g2._zr2.transform(lambda s: s.quantile(0.75)) + tr_)
                    f["zr2_loc1max"] = np.exp(g2._zr2.rolling(3, center=True, min_periods=1).max().reset_index(level=0, drop=True) + tr_)
                    f["p_r_sd_mean"] = g.r_sd.transform("mean")
                    f.drop(columns="_zr2", inplace=True)
                f.drop(columns="_zr", inplace=True)
                f.sort_index(inplace=True)
        if "vrel" in VARIANT:
            NF_ = ["r_med", "r_sd", "r_q10", "r_q90", "_vt", "_htoff", "_hledd", "time_since_intake_on", "f_t", "on_k_n", "_off0",
                   "n_vis", "visit_idx", "_lmle", "dd", "_hon", "_hoff"]
            for f in (ff, of):
                f["_vt"] = vtype(f); f["_htoff"] = f.time_since_intake_off.notna().astype(int); f["_hledd"] = f.ledd.notna().astype(int)
                f["_off0"] = (f.off == 0).astype(int); f["_lmle"] = np.log(f.mle.clip(0.5)); f["_hon"] = f.on.notna().astype(int); f["_hoff"] = f.off.notna().astype(int)
            msk = (ff.target > 3) & ff.z.notna()
            u_all = (ff.z - np.log(ff.target.clip(0.5))).values
            PQ = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200, verbose=-1, n_jobs=INNER_JOBS)
            def fit3(tr_df, u_tr):
                return (lgb.LGBMRegressor(**PQ).fit(tr_df[NF_], u_tr),
                        lgb.LGBMRegressor(**dict(PQ, objective="quantile", alpha=0.9)).fit(tr_df[NF_], u_tr),
                        lgb.LGBMRegressor(**dict(PQ, objective="quantile", alpha=0.5)).fit(tr_df[NF_], u_tr))
            # OOF for train rows
            oofp = np.full((len(ff), 3), np.nan)
            pid_s = ff.patient_id.astype(str).values
            for itr, iva in GroupKFold(n_splits=4).split(ff, groups=pid_s):
                itr_m = itr[msk.values[itr]]
                ms = fit3(ff.iloc[itr_m], u_all[itr_m])
                for j, m_ in enumerate(ms): oofp[iva, j] = m_.predict(ff.iloc[iva][NF_])
            msf = fit3(ff[msk], u_all[msk.values])
            ofp = np.column_stack([m_.predict(of[NF_]) for m_ in msf])
            for f, P_ in ((ff, oofp), (of, ofp)):
                f["vu_mean"] = P_[:, 0]; f["vu_q90"] = P_[:, 1]; f["vu_q50"] = P_[:, 2]
                f.sort_values(["patient_id", "age"], inplace=True)
                am = f.groupby("patient_id", observed=True).age.transform("mean")
                tr_ = f.mle_slope * (f.age - am)
                f["_zv"] = f.z - f.vu_q90 - tr_
                f["_zm"] = f.z - f.vu_q50 - tr_
                g = f.groupby("patient_id", observed=True)
                f["zv_max"] = np.exp(g._zv.transform("max") + tr_)
                f["zv_top2"] = np.exp(g._zv.transform(lambda s: s.nlargest(2).mean()) + tr_)
                f["zv_loc1max"] = np.exp(g._zv.rolling(3, center=True, min_periods=1).max().reset_index(level=0, drop=True) + tr_)
                f["zv_loc2max"] = np.exp(g._zv.rolling(5, center=True, min_periods=1).max().reset_index(level=0, drop=True) + tr_)
                f["zv_cummax"] = np.exp(g._zv.cummax() + tr_)
                f["zv_revcummax"] = np.exp(f.iloc[::-1].groupby("patient_id", observed=True)._zv.cummax().iloc[::-1] + tr_)
                f["zm_med"] = np.exp(g._zm.transform("median") + tr_)
                f["zm_loc1"] = np.exp(g._zm.rolling(3, center=True, min_periods=1).median().reset_index(level=0, drop=True) + tr_)
                f.drop(columns=["_zv", "_zm", "_vt", "_htoff", "_hledd", "_off0", "_lmle", "_hon", "_hoff"], inplace=True)
                f.sort_index(inplace=True)
        if "mode" in VARIANT:
            for f in (ff, of):
                f.sort_values(["patient_id", "age"], inplace=True)
                am = f.groupby("patient_id", observed=True).age.transform("mean")
                tr_ = (f.mle_slope * (f.age - am)).values
                zo = np.log(f.off_c.where(f.off_c > 0)).values - tr_
                za = f.z.values - tr_
                pidv = f.patient_id.astype(str).values
                starts = np.r_[0, np.where(pidv[1:] != pidv[:-1])[0] + 1, len(f)]
                out = {k: np.full(len(f), np.nan) for k in ("mo_off", "mo_off_sup", "mo_off_h2", "mo_off_sup2", "mo_all", "mo_all_sup")}
                for s0, s1 in zip(starts[:-1], starts[1:]):
                    for arr, key, h in ((zo, "mo_off", 0.03), (zo, "mo_off_h2", 0.06), (za, "mo_all", 0.04)):
                        v = arr[s0:s1]; v = v[~np.isnan(v)]
                        if len(v) == 0: continue
                        K = np.exp(-0.5 * ((v[:, None] - v[None, :]) / h) ** 2).sum(1)
                        best = np.max(K); c = v[K >= best - 1e-9].max()
                        out[key][s0:s1] = c
                        if key == "mo_off": out["mo_off_sup"][s0:s1] = best
                        if key == "mo_off_h2": out["mo_off_sup2"][s0:s1] = best
                        if key == "mo_all": out["mo_all_sup"][s0:s1] = best
                for key in ("mo_off", "mo_off_h2", "mo_all"):
                    f[key] = np.exp(out[key] + tr_)
                for key in ("mo_off_sup", "mo_off_sup2", "mo_all_sup"):
                    f[key] = out[key]
                f.sort_index(inplace=True)
        if "ransac" in VARIANT:
            for f in (ff, of):
                f.sort_values(["patient_id", "age"], inplace=True)
                pidv = f.patient_id.astype(str).values; ages = f.age.values; oc = f.off_c.values
                starts = np.r_[0, np.where(pidv[1:] != pidv[:-1])[0] + 1, len(f)]
                rv = np.full(len(f), np.nan); ri = np.full(len(f), np.nan); rs = np.full(len(f), np.nan); rr = np.full(len(f), np.nan)
                for s0, s1 in zip(starts[:-1], starts[1:]):
                    a = ages[s0:s1]; y = oc[s0:s1]; ok = ~np.isnan(y) & (y > 0)
                    ia = np.where(ok)[0]
                    if len(ia) < 2: continue
                    best = (-1, 0, None)
                    for i in range(len(ia)):
                        for j in range(i + 1, len(ia)):
                            p, q = ia[i], ia[j]
                            if a[q] - a[p] < 0.3: continue
                            sl = (y[q] - y[p]) / (a[q] - a[p])
                            if sl < -0.5: continue
                            line = y[p] + sl * (a[ia] - a[p])
                            tol = 1.0 + 0.015 * line
                            inl = np.abs(y[ia] - line) <= tol
                            sc = inl.sum() + 1e-3 * line.mean()
                            if sc > best[0] + best[1] * 0:
                                best = (sc, 0, inl)
                    if best[2] is None: continue
                    inl = best[2]; xi = a[ia][inl]; yi = y[ia][inl]
                    c = np.polyfit(xi, yi, 1) if len(np.unique(xi)) >= 2 else np.array([0.0, yi.mean()])
                    rv[s0:s1] = np.polyval(c, a); ri[s0:s1] = inl.sum(); rs[s0:s1] = c[0]; rr[s0:s1] = inl.sum() / len(ia)
                f["rs_val"] = rv; f["rs_inl"] = ri; f["rs_slope"] = rs; f["rs_frac"] = rr
                f.sort_index(inplace=True)
    ff, of = add_bayes(ff, of)
    return ff, of




# %% [markdown]
# ## Post-traitement : poly2 par patient puis ratchet

# %%
def ratchet(d, pred, r=RATCHET):
    """par patient trie par age : si la prediction stagne ou baisse, elle monte de x r par visite."""
    out = np.asarray(pred, float).copy()
    for _, g in d.groupby("patient_id", observed=True):
        idx = g.index.values[np.argsort(g["age"].values, kind="stable")]
        cv = out[idx].copy(); t = cv.copy()
        for i in range(1, len(t)):
            if cv[i] <= t[i - 1]:
                t[i] = r * t[i - 1]
        out[idx] = t
    return out


def _to_xgb(A, ref_cats):
    A = A.copy()
    for c, cats in ref_cats.items():
        A[c] = pd.Categorical(A[c], categories=cats).codes.astype(float)
    return A


# %% [markdown]
# ## Estimateur sklearn (compatible skore CrossValidationReport)

# %%
class BenchRatchet(BaseEstimator, RegressorMixin):
    def __init__(self, ratchet_r=RATCHET):
        self.ratchet_r = ratchet_r

    def _prep(self, X, y=None):
        d = X.copy().reset_index(drop=True)
        d["dd"] = d["age"] - d["age_at_diagnosis"]
        d["target"] = np.asarray(y, float) if y is not None else np.nan
        return d

    def fit(self, X, y):
        d = self._prep(X, y)
        self.cats_ = {c: d[c].astype("category").cat.categories for c in CATS}
        for c in CATS:
            d[c] = pd.Categorical(d[c], categories=self.cats_[c])
        self.fit_df_ = d
        dummy = d.iloc[:200].copy(); dummy["target"] = np.nan
        trf, _ = make_feats(d, dummy)
        self.cols_ = feat_cols(trf)
        self.xcats_ = {c: trf[c].cat.categories for c in self.cols_ if str(trf[c].dtype) == "category"}
        self.lgb_ = lgb.LGBMRegressor(**PARAMS).fit(trf[self.cols_], trf.target)
        self.xgb_ = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=7, min_child_weight=20,
                                     subsample=0.8, colsample_bytree=0.6, tree_method="hist",
                                     n_jobs=INNER_JOBS).fit(_to_xgb(trf[self.cols_], self.xcats_), trf.target)
        return self

    def predict(self, X):
        d = self._prep(X)
        for c in CATS:
            d[c] = pd.Categorical(d[c], categories=self.cats_[c])
        _, of = make_feats(self.fit_df_, d)
        of = of.reset_index(drop=True)
        p = (self.lgb_.predict(of[self.cols_]) + self.xgb_.predict(_to_xgb(of[self.cols_], self.xcats_))) / 2
        p = smooth_poly2(of, p)
        p = ratchet(of, p, self.ratchet_r)
        # make_feats peut reordonner : on remet dans l'ordre de X via Index
        s = pd.Series(np.clip(p, 0, None), index=of["Index"].values)
        return s.loc[d["Index"].values].values


# %% [markdown]
# ## Donnees + skore

# %%
if __name__ == "__main__":
    cfg = load_skore_credentials()
    skore.login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

    X_train, y_train = load_dataset(DATA_DIR)
    print(f"Train: {X_train.shape}")
    folds = get_cv_folds(X_train, y_train)

    # test rapide 1 fold (sans skore) pour voir que tout tourne
    QUICK_FOLD = True
    if QUICK_FOLD:
        t0 = time.time()
        tr_idx, va_idx = folds[0]
        X_val = X_train.iloc[va_idx]
        m = BenchRatchet().fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pv = m.predict(X_val)
        yv = np.asarray(y_train.iloc[va_idx], float)
        print(f"Fold 1 (test rapide) modele complet RMSE = {np.sqrt(np.mean((pv - yv) ** 2)):.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        d = m._prep(X_val)
        for c in CATS:
            d[c] = pd.Categorical(d[c], categories=m.cats_[c])
        _, of = make_feats(m.fit_df_, d)
        of = of.reset_index(drop=True)
        pb = ratchet(of, smooth_poly2(of, of["bayes_mean"].values.copy()))
        pb = pd.Series(np.clip(pb, 0, None), index=of["Index"].values).loc[d["Index"].values].values
        print(f"Fold 1 (test rapide) bayes_mean seul (poly2 + ratchet) RMSE = {np.sqrt(np.mean((pb - yv) ** 2)):.4f}", flush=True)
        print(f"   ess median {of.groupby('patient_id', observed=True).bayes_ess.first().median():.3f}", flush=True)

    print(f"\nRunning CrossValidationReport exp 21 (n_jobs={CV_JOBS})… (barre a 0% tant qu'aucun fold n'est fini, normal)")
    t_cv = time.time()
    report = skore.CrossValidationReport(BenchRatchet(), X=X_train, y=y_train, splitter=folds, n_jobs=CV_JOBS)
    print(f"CV terminee en {(time.time() - t_cv) / 60:.1f} min")
    project.put("21_bayes", report)
    print("Report saved: 21_bayes")
    print(report.metrics.rmse())
    try:
        per_fold = report.metrics.rmse(aggregate=None)
        print(per_fold)
    except Exception as e:
        print("(rmse par fold indisponible :", e, ")")

    # soumission : refit sur tout le train, tous les coeurs
    INNER_JOBS = N_CPU
    PARAMS["n_jobs"] = N_CPU
    X_test = load_test_dataset(DATA_DIR)
    print(f"Test: {X_test.shape}")
    t_fit = time.time()
    final_model = BenchRatchet().fit(X_train, y_train)
    test_preds = final_model.predict(X_test)
    print(f"Refit + predict en {(time.time() - t_fit) / 60:.1f} min")
    sub_dir = PROJECT_ROOT / "submissions"
    sub_dir.mkdir(exist_ok=True)
    pd.DataFrame({"Index": X_test["Index"].values, "target": np.clip(test_preds, 0, None)}).to_csv(
        sub_dir / "21_bayes.csv", index=False)
    print("Submission saved: submissions/21_bayes.csv")