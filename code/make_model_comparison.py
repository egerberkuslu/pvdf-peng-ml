"""Regenerate fig04_model_comparison.pdf: LOO-CV R2 of the 14 canonical
regressors on the four targets (SVR-ayarli excluded; it is not part of the
fixed-hyperparameter benchmark described in the paper)."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
_p = "reg_models.json" if os.path.exists("reg_models.json") else os.path.join(os.path.dirname(__file__), "..", "results", "reg_models.json")
d = json.load(open(_p))
targets = ["rms", "vpp", "peak_abs", "energy"]
ttitle = {"rms": "RMS", "vpp": r"V$_{pp}$", "peak_abs": "Peak |V|", "energy": "Energy"}
label = {"Lineer": "Linear", "Ridge": "Ridge", "Lasso": "Lasso",
         "Polinom2-OLS": "Poly-2 (OLS)", "Polinom2-Ridge": "Poly-2 (ridge)",
         "SVR-RBF": "SVR (RBF)", "KernelRidge": "Kernel ridge", "KNN": "kNN",
         "RandomForest": "Random forest", "ExtraTrees": "Extra-trees",
         "GradientBoosting": "Gradient boosting", "XGBoost": "XGBoost",
         "ANN-MLP": "MLP", "ARD-GP": "ARD-GP"}
order = list(label)  # JSON insertion order, bottom-to-top like the original

BLUE, RED = "#4C72B0", "#C44E52"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5})
fig, axes = plt.subplots(1, 4, figsize=(11, 4.0), sharey=True)
for ax, tg in zip(axes, targets):
    md = d["targets"][tg]["models"]
    vals = [max(md[m]["cv_r2"], 0.0) for m in order]
    best = max(order, key=lambda m: md[m]["cv_r2"])
    colors = [RED if m == best else BLUE for m in order]
    ax.barh(range(len(order)), vals, color=colors, height=0.72)
    ax.set_title(f"{ttitle[tg]} (best: {label[best]} {md[best]['cv_r2']:.2f})")
    ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel(r"LOO-CV $R^2$")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([label[m] for m in order])
fig.tight_layout()
for out in ["../manuscript-elsevier/figures/fig04_model_comparison.pdf"]:
    fig.savefig(out, bbox_inches="tight")
print("written:", len(order), "models, best per target:",
      {t: max(order, key=lambda m: d['targets'][t]['models'][m]['cv_r2']) for t in targets})
