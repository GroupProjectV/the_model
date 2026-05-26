"""Approach 1 (swing) + Approach 2 (reweighted/audit-mitigated) — does Twitter actually predict 2024?"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
RANDOM_STATE = 42

df_sample = pd.read_parquet('app_artifacts/df_sample.parquet')
df_model  = pd.read_parquet('app_artifacts/df_model.parquet')
with open('app_artifacts/political_topic_ids.json') as f: political_topic_ids = json.load(f)

# Two samples: full, and audit-mitigated (power users excluded)
samples = {
    'full':      df_sample.dropna(subset=['state_code']).copy(),
    'reweighted (no power users)': df_sample[~df_sample['is_power_user']].dropna(subset=['state_code']).copy(),
}

# Petrocik ownership
ownership = {'Healthcare':'D','Climate / Environment':'D','Education':'D','Abortion / Reproductive Rights':'D',
             'Immigration / Border':'R','Gun Policy':'R','Economic Policy':'R','Criminal Justice':'R','Foreign Policy':'R'}

margin_2020 = {  # R-D pp
    'AL':25.5,'AK':10.1,'AZ':-0.3,'AR':27.6,'CA':-29.2,'CO':-13.5,'CT':-20.0,'DE':-19.0,'DC':-86.8,'FL':3.4,
    'GA':-0.2,'HI':-29.5,'ID':30.8,'IL':-17.0,'IN':16.1,'IA':8.2,'KS':14.6,'KY':25.9,'LA':18.6,'ME':-9.1,
    'MD':-33.2,'MA':-33.5,'MI':-2.8,'MN':-7.1,'MS':16.5,'MO':15.4,'MT':16.4,'NE':19.1,'NV':-2.4,'NH':-7.3,
    'NJ':-15.9,'NM':-10.8,'NY':-23.1,'NC':1.3,'ND':33.4,'OH':8.0,'OK':33.1,'OR':-16.1,'PA':-1.2,'RI':-20.7,
    'SC':11.7,'SD':26.2,'TN':23.2,'TX':5.6,'UT':20.5,'VT':-35.1,'VA':-10.1,'WA':-19.2,'WV':38.9,'WI':-0.6,'WY':43.3}
demo = {  # %bach, %white-nh
    'AL':(26.2,64.6),'AK':(30.7,59.4),'AZ':(30.7,53.4),'AR':(24.2,70.8),'CA':(35.3,35.2),'CO':(43.2,66.4),
    'CT':(40.6,63.2),'DE':(33.0,60.8),'DC':(60.6,36.8),'FL':(31.1,51.6),'GA':(32.0,50.0),'HI':(33.6,21.4),
    'ID':(28.7,80.4),'IL':(36.0,60.2),'IN':(27.8,78.0),'IA':(29.7,84.5),'KS':(34.5,75.4),'KY':(25.4,82.6),
    'LA':(25.2,57.8),'ME':(33.2,90.6),'MD':(41.6,47.8),'MA':(45.3,67.6),'MI':(30.5,73.0),'MN':(37.5,76.6),
    'MS':(23.2,55.4),'MO':(30.0,77.6),'MT':(33.2,85.4),'NE':(32.4,76.4),'NV':(26.4,45.6),'NH':(38.4,87.4),
    'NJ':(41.2,53.2),'NM':(29.2,35.4),'NY':(38.2,53.8),'NC':(33.0,60.6),'ND':(32.0,82.0),'OH':(29.8,76.0),
    'OK':(26.7,62.0),'OR':(35.2,73.6),'PA':(33.8,73.6),'RI':(35.7,70.4),'SC':(30.0,62.8),'SD':(30.6,80.4),
    'TN':(29.8,71.4),'TX':(32.3,39.0),'UT':(37.0,75.2),'VT':(41.0,89.0),'VA':(40.6,59.0),'WA':(38.3,65.8),
    'WV':(23.4,91.0),'WI':(32.0,79.6),'WY':(28.6,82.4)}

state_partisan_lean = df_model.set_index('state_code')['partisan_lean']

def build_features(sample_df):
    pol = sample_df[sample_df['topic'].isin(political_topic_ids)].copy()
    ts = pol.groupby(['state_code','topic_label']).size().unstack(fill_value=0)
    ts = ts.div(ts.sum(axis=1).replace(0,1), axis=0)
    d = [c for c in ts.columns if ownership.get(c)=='D']
    r = [c for c in ts.columns if ownership.get(c)=='R']
    pisi = ts[d].sum(axis=1) - ts[r].sum(axis=1)
    sst = pol.groupby(['state_code','topic_label'])['vader_compound'].mean().unstack(fill_value=0)
    sd = sst.reindex(columns=d, fill_value=0).reindex(index=ts.index, fill_value=0)
    sr = sst.reindex(columns=r, fill_value=0).reindex(index=ts.index, fill_value=0)
    pisi_w = (ts[d]*sd).sum(axis=1) - (ts[r]*sr).sum(axis=1)
    mean_sent = sample_df.groupby('state_code')['vader_compound'].mean()
    rows=[]
    for s in margin_2020:
        if s not in demo: continue
        b,w = demo[s]
        m24 = df_model.loc[df_model['state_code']==s, 'margin_r']
        if len(m24)==0: continue
        rec = {'state':s, 'margin_2024':m24.iloc[0], 'margin_2020':margin_2020[s], 'swing':m24.iloc[0]-margin_2020[s],
               'pct_bachelor':b, 'pct_white':w,
               'pisi':pisi.get(s,0), 'pisi_weighted':pisi_w.get(s,0),
               'stance_lean':state_partisan_lean.get(s,0) if pd.notna(state_partisan_lean.get(s,np.nan)) else 0,
               'mean_sentiment':mean_sent.get(s,0)}
        for c in d+r: rec[f'sal_{c}'] = ts.loc[s,c] if s in ts.index else 0
        rows.append(rec)
    return pd.DataFrame(rows).set_index('state').replace([np.inf,-np.inf],0).fillna(0)

def loocv(X,y,a):
    if X.shape[1]==0: return np.full_like(y,y.mean(),dtype=float)
    p=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(X):
        pipe=Pipeline([('sc',StandardScaler()),('rg',Ridge(alpha=a,random_state=RANDOM_STATE))])
        pipe.fit(X[tr],y[tr]); p[te]=pipe.predict(X[te])
    return p

def best_a(X,y,grid=(0.05,0.1,0.5,1,2,5,10,25,50,100)):
    if X.shape[1]==0: return 0
    return max(grid, key=lambda a: r2_score(y, loocv(X,y,a)))

def metrics(y,p):
    return dict(r2=r2_score(y,p), rmse=float(np.sqrt(mean_squared_error(y,p))),
                mae=mean_absolute_error(y,p), r=float(np.corrcoef(y,p)[0,1]) if p.std()>0 else 0,
                bin=float(np.mean(np.sign(p)==np.sign(y))))

def boot_ci(feats_a, feats_b, df_ext, y, n=300, seed=42):
    rng = np.random.RandomState(seed)
    Xa = df_ext[feats_a].values if feats_a else np.zeros((len(y),0))
    Xb = df_ext[feats_b].values if feats_b else np.zeros((len(y),0))
    aa = best_a(Xa,y) if feats_a else 0; ab = best_a(Xb,y) if feats_b else 0
    diffs=[]
    for _ in range(n):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(idx))<10: continue
        try:
            r2a = r2_score(y[idx], loocv(Xa[idx] if Xa.size else np.zeros((len(idx),0)), y[idx], aa))
            r2b = r2_score(y[idx], loocv(Xb[idx] if Xb.size else np.zeros((len(idx),0)), y[idx], ab))
            diffs.append(r2b-r2a)
        except: pass
    return float(np.percentile(diffs,2.5)), float(np.percentile(diffs,97.5)), float(np.mean(diffs))

for sample_name, sample_df in samples.items():
    print('\n' + '='*100)
    print(f'SAMPLE: {sample_name}    (n={len(sample_df)} tweets, {sample_df["state_code"].nunique()} states)')
    print('='*100)
    df_ext = build_features(sample_df)
    twitter = ['pisi','pisi_weighted','stance_lean','mean_sentiment'] + [c for c in df_ext.columns if c.startswith('sal_')]

    # raw correlations
    print('Raw correlations with outcomes:')
    for tgt in ['margin_2024','swing']:
        for f in ['pisi','pisi_weighted','stance_lean','mean_sentiment']:
            r,p = pearsonr(df_ext[f], df_ext[tgt])
            sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else ''))
            print(f'  {tgt:<12} <-> {f:<18} r={r:+.3f} p={p:.3f} {sig}')

    for target_name in ['margin_2024 (LEVEL)', 'swing (2024 minus 2020)']:
        target_col = 'margin_2024' if 'LEVEL' in target_name else 'swing'
        y = df_ext[target_col].values
        print(f'\n  --- Target: {target_name}   (mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.1f}, max={y.max():.1f}) ---')
        sets = {
            'M0 intercept':                  [],
            'M1 2020 margin':                ['margin_2020'],
            'M2 2020 + demographics':        ['margin_2020','pct_bachelor','pct_white'],
            'M3 M2 + Twitter':               ['margin_2020','pct_bachelor','pct_white']+twitter,
            'D demographics only':           ['pct_bachelor','pct_white'],
            'T Twitter only':                twitter,
            'D+T demographics + Twitter':    ['pct_bachelor','pct_white']+twitter,
        }
        print(f'  {"Model":<32} {"R²":>7} {"RMSE":>7} {"MAE":>7} {"r":>6} {"BinAcc":>8} {"alpha":>7}')
        print('  ' + '-'*78)
        res={}
        for name,feats in sets.items():
            X = df_ext[feats].values if feats else np.zeros((len(y),0))
            a = best_a(X,y) if feats else 0
            p = loocv(X,y,a)
            m = metrics(y,p); m['alpha']=a
            res[name]=m
            print(f'  {name:<32} {m["r2"]:>7.3f} {m["rmse"]:>7.2f} {m["mae"]:>7.2f} {m["r"]:>6.3f} {m["bin"]:>8.3f} {m["alpha"]:>7.2f}')
        # incremental R² and CI: T vs D (no 2020) and M3 vs M2
        lo,hi,mn = boot_ci(['pct_bachelor','pct_white'], ['pct_bachelor','pct_white']+twitter, df_ext, y)
        lo2,hi2,mn2 = boot_ci(['margin_2020','pct_bachelor','pct_white'], ['margin_2020','pct_bachelor','pct_white']+twitter, df_ext, y)
        print(f'  Incremental R² (Twitter beyond demographics):       mean={mn:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]')
        print(f'  Incremental R² (Twitter beyond 2020+demographics):  mean={mn2:+.3f}, 95% CI [{lo2:+.3f}, {hi2:+.3f}]')
