"""Spectrum-level confidence vs anomaly score for in-taxonomy errors on the test split; 10% abstention."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[0])); from _studypaths import REPO, STUDY, VARIANTS_ROOT  # noqa: E402
import sys, numpy as np; from pathlib import Path; from scipy.special import softmax; from sklearn.metrics import roc_auc_score
S=Path(sys.argv[1]).resolve(); spec=sys.argv[2]; sys.path.insert(0,str(S)); from evaluate import ecdf
F=dict(np.load(S/'features.npz')); t=np.load(REPO/f'cache/{spec}/test.npz')
p=softmax(t['fine_logits'].astype(float),1); err=p.argmax(1)!=t['labels']; conf=p.max(1)
R=lambda X:np.column_stack([ecdf(F['val_emb_scores'][:,j],X[:,j]) for j in range(16)]).mean(1)
a_test=R(F['test_emb_scores']); a_val=R(F['val_emb_scores'])
keep=conf>=np.quantile(conf,.10); ab=~keep
print(S.name,'acc',round(100*(~err).mean(),1),'AUROC conf',round(roc_auc_score(err,-conf),3),'AUROC anomaly',round(roc_auc_score(err,a_test),3),
      'retained acc',round(100*(~err[keep]).mean(),1),'abstained below val95 anomaly',round(100*(a_test[ab]<np.quantile(a_val,.95)).mean(),1))
