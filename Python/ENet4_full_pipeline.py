#!/usr/bin/env python3
# Full ENet4 pipeline script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2, norm
from math import sqrt
import argparse, os

def delong_roc_variance(ground_truth, predictions):
    order = np.argsort(-predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]
    distinct_value_indices = np.where(np.diff(predictions))[0]
    threshold_idxs = np.r_[distinct_value_indices, ground_truth.size - 1]
    tps = np.cumsum(ground_truth)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]; fps = np.r_[0, fps]
    P = np.sum(ground_truth); N = ground_truth.size - P
    if P == 0 or N == 0: return np.nan, np.nan
    fpr = fps / N; tpr = tps / P
    auc_val = np.trapz(tpr, fpr)
    pos_preds = predictions[ground_truth == 1]
    neg_preds = predictions[ground_truth == 0]
    m, n = len(pos_preds), len(neg_preds)
    def v(u, v_): return 1.0 if u>v_ else (0.5 if u==v_ else 0.0)
    v_vec = np.array([np.mean([v(up, vn) for vn in neg_preds]) for up in pos_preds])
    u_vec = np.array([np.mean([v(up, vn) for up in pos_preds]) for vn in neg_preds])
    auc_cov = (np.var(v_vec, ddof=1)/m) + (np.var(u_vec, ddof=1)/n)
    return auc_val, auc_cov

def delong_ci(y_true, y_score):
    auc_val, auc_var = delong_roc_variance(np.asarray(y_true), np.asarray(y_score))
    if np.isnan(auc_val) or np.isnan(auc_var): return np.nan, (np.nan, np.nan)
    se = sqrt(auc_var); z = norm.ppf(1 - 0.05/2)
    return auc_val, (max(0, auc_val - z*se), min(1, auc_val + z*se))

def brier_score(y, p): return float(np.mean((p - y)**2))

def spiegelhalter_test(y, p):
    y = np.asarray(y).astype(float); p = np.asarray(p).astype(float)
    Z = np.sum(y - p) / np.sqrt(np.sum(p*(1-p)) + 1e-12)
    return float(Z), float(2*(1 - norm.cdf(abs(Z))))

def hosmer_lemeshow(y, p, g=10):
    df = pd.DataFrame({'y':y,'p':p})
    df['bin'] = pd.qcut(df['p'], q=g, duplicates='drop')
    grp = df.groupby('bin'); obs = grp['y'].sum().values; n = grp.size().values
    exp = grp['p'].sum().values; chi2_stat = np.sum((obs-exp)**2/(exp+1e-12) + ((n-obs)-(n-exp))**2/((n-exp)+1e-12))
    dfree = len(n) - 2; pval = 1 - chi2.cdf(chi2_stat, dfree)
    cal_table = pd.DataFrame({'mean_pred': grp['p'].mean().values, 'obs_rate': obs/n})
    return float(chi2_stat), int(dfree), float(pval), cal_table

def nested_enet_oof(X, y, outer_splits, inner_splits, seed):
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed+123)
    Cs = np.logspace(-3,3,20); l1_grid=[0.1,0.3,0.5,0.7,0.9]
    oof = np.zeros(len(y))
    for tr, te in outer.split(X,y):
        X_tr, X_te = X[tr], X[te]; y_tr = y[tr]
        best_loss=np.inf; best_pipe=None
        for l1 in l1_grid:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegressionCV(Cs=Cs, cv=inner, penalty='elasticnet', solver='saga',
                                             l1_ratios=[l1], scoring='neg_log_loss', max_iter=5000, n_jobs=-1))])
            pipe.fit(X_tr, y_tr)
            scores = pipe.named_steps['clf'].scores_[1].mean(axis=0)
            cur_loss = -scores.max()
            if cur_loss < best_loss: best_loss = cur_loss; best_pipe = pipe
        oof[te] = best_pipe.predict_proba(X_te)[:,1]
    return oof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', default='Combined demo and imaging data.xlsx')
    ap.add_argument('--output_dir', default='enet4_outputs')
    ap.add_argument('--outer_splits', type=int, default=5)
    ap.add_argument('--inner_splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=321)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.data_path, engine='openpyxl')
    rename_map = {
        'Lung cancer (0=no, 1=yes)': 'lung_cancer',
        'AIRC nodule malignancy risk (0=low, 1=high)': 'airc_high',
        'Suspicious nodule morphology (0=no, 1=yes)': 'suspicious',
        'Nodule invades adjacent structures on CT (0=no, 1=yes)': 'inv_adj',
        'COPD': 'copd'
    }
    for col in df.columns:
        if 'emphy' in col.lower() and 'ct' in col.lower(): rename_map[col]='emphysema_ct'
    df = df.rename(columns=rename_map)

    features4=['airc_high','suspicious','inv_adj']; emphy='emphysema_ct' if 'emphysema_ct' in df.columns else 'copd'
    features4.append(emphy)

    data=df[['lung_cancer']+features4].copy().apply(pd.to_numeric, errors='coerce')
    for c in features4: data[c]=(data[c]>0).astype(int)
    data=data.dropna().copy(); X=data[features4].values; y=data['lung_cancer'].astype(int).values

    oof=nested_enet_oof(X,y,args.outer_splits,args.inner_splits,args.seed)

    auc,(auc_lo,auc_hi)=delong_ci(y,oof); bs=brier_score(y,oof)
    Z,p_sp=spiegelhalter_test(y,oof)
    HL_chi2,HL_df,HL_p,cal_table=hosmer_lemeshow(y,oof,g=10)

    sm_df=data.copy(); sm_df['intercept']=1.0
    res=sm.Logit(sm_df['lung_cancer'], sm_df[['intercept']+features4]).fit(disp=False)
    params=res.params; conf=res.conf_int(); pvals=res.pvalues
    OR=np.exp(params); CI_low=np.exp(conf[0]); CI_high=np.exp(conf[1])
    or_table=pd.DataFrame({'term':OR.index,'OR':OR.values,'CI_low':CI_low.values,'CI_high':CI_high.values,'p_value':pvals.values})
    or_table=or_table[or_table['term']!='intercept']
    or_table.to_excel(os.path.join(args.output_dir,'ENet4_ORs.xlsx'),index=False)

    X_vif=sm_df[features4].astype(float)
    vif_table=pd.DataFrame({'variable':features4,'VIF':[variance_inflation_factor(X_vif.values,i) for i in range(len(features4))]})
    vif_table.to_excel(os.path.join(args.output_dir,'ENet4_VIF.xlsx'),index=False)

    plt.figure(figsize=(13,4))
    fpr,tpr,_=roc_curve(y,oof); plt.subplot(1,3,1); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.title('ROC')
    plt.subplot(1,3,2); plt.scatter(cal_table['mean_pred'],cal_table['obs_rate']); plt.plot([0,1],[0,1],'--'); plt.title('Calibration')
    plt.subplot(1,3,3); plt.errorbar(or_table['OR'], range(len(or_table)), xerr=[or_table['OR']-or_table['CI_low'], or_table['CI_high']-or_table['OR']], fmt='o'); plt.xscale('log'); plt.title('OR Forest')
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir,'ENet4_combined_panel.png'),dpi=600)

    print('AUC:',auc,'CI:',(auc_lo,auc_hi)); print('Brier:',bs)

if __name__=='__main__': main()