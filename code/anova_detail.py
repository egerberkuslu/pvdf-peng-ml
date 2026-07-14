#!/usr/bin/env python3
"""E8: full ANOVA specification — main effects + two-way interactions
(three-way pooled into residual, df=32), Type II & III SS, partial eta^2
with residual-bootstrap CIs, diagnostics. Writes anova_results.json."""
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings("ignore")

df = pd.read_parquet("targets_design.parquet")
df = df.copy()
df["log_rms"] = np.log10(df["rms_Voc"])

FORMULA_T = (
    "{y} ~ C(composition) + C(force_N) + C(freq_Hz)"
    " + C(composition):C(force_N) + C(composition):C(freq_Hz)"
    " + C(force_N):C(freq_Hz)"
)
FORMULA_S = (
    "{y} ~ C(composition, Sum) + C(force_N, Sum) + C(freq_Hz, Sum)"
    " + C(composition, Sum):C(force_N, Sum)"
    " + C(composition, Sum):C(freq_Hz, Sum)"
    " + C(force_N, Sum):C(freq_Hz, Sum)"
)


def clean_key(k):
    return (
        k.replace("C(composition, Sum)", "composition")
        .replace("C(force_N, Sum)", "force")
        .replace("C(freq_Hz, Sum)", "frequency")
        .replace("C(composition)", "composition")
        .replace("C(force_N)", "force")
        .replace("C(freq_Hz)", "frequency")
    )


def partial_eta2_from_anova(tab):
    ss_res = float(tab.loc["Residual", "sum_sq"])
    out = {}
    for idx in tab.index:
        if idx == "Residual":
            continue
        ss = float(tab.loc[idx, "sum_sq"])
        out[clean_key(idx)] = ss / (ss + ss_res)
    return out


def analyze(ycol, seed=0, n_boot=2000):
    fit_t = smf.ols(FORMULA_T.format(y=ycol), data=df).fit()
    fit_s = smf.ols(FORMULA_S.format(y=ycol), data=df).fit()
    a2 = sm.stats.anova_lm(fit_t, typ=2)
    a3 = sm.stats.anova_lm(fit_s, typ=3).drop(index="Intercept", errors="ignore")
    eta2_t2 = partial_eta2_from_anova(a2)
    eta2_t3 = partial_eta2_from_anova(a3)

    # fixed-X residual bootstrap for partial eta^2 CIs (Type II)
    rng = np.random.default_rng(seed)
    fitted = fit_t.fittedvalues.values
    resid = fit_t.resid.values
    boots = {k: [] for k in eta2_t2}
    dboot = df.copy()
    for _ in range(n_boot):
        dboot["_yb"] = fitted + rng.choice(resid, size=len(resid), replace=True)
        fb = smf.ols(FORMULA_T.format(y="_yb"), data=dboot).fit()
        tb = sm.stats.anova_lm(fb, typ=2)
        eb = partial_eta2_from_anova(tb)
        for k in boots:
            boots[k].append(eb[k])
    ci = {
        k: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
        for k, v in boots.items()
    }

    sw_stat, sw_p = stats.shapiro(resid)
    bp = het_breuschpagan(resid, fit_t.model.exog)
    lev = stats.levene(
        *[df.loc[df.composition == c, ycol].values for c in df.composition.unique()]
    )
    return {
        "residual_df": int(fit_t.df_resid),
        "model_R2": float(fit_t.rsquared),
        "typ2_partial_eta2": eta2_t2,
        "typ3_partial_eta2": eta2_t3,
        "typ2_p_values": {
            clean_key(i): float(a2.loc[i, "PR(>F)"])
            for i in a2.index
            if i != "Residual"
        },
        "partial_eta2_90CI_typ2_residboot": ci,
        "diagnostics": {
            "shapiro_p": float(sw_p),
            "breusch_pagan_p": float(bp[1]),
            "levene_composition_p": float(lev.pvalue),
        },
    }


results = {}
for ycol in ["rms_Voc", "log_rms", "Vpp"]:
    results[ycol] = analyze(ycol)

raw_ok = results["rms_Voc"]["diagnostics"]["shapiro_p"] > 0.05
log_ok = results["log_rms"]["diagnostics"]["shapiro_p"] > 0.05
results["recommended_scale_for_rms"] = (
    "raw" if raw_ok else ("log10" if log_ok else "log10 (closer to normal)")
)

with open("anova_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
print("DONE")
