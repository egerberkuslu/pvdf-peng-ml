"""LCR measurements of the five PENG specimens: summaries, RC chain, macros.

Reads data/lcr/lcr_tidy.csv (five compositions x 201 frequencies, 1 kHz to 1 MHz) and
regenerates results/lcr/{lcr_summary,lcr_rc,fig_cp_vs_f,fig_D_vs_f}.csv, results/lcr/tables/
and results/lcr/numbers_lcr.tex. R_in = 144 kOhm and the tap range 5-25 Hz are given constants.
Run:  python3 code/lcr_analysis.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from peng_paths import ROOT  # repository root (env PENG_ROOT overrides)
TIDY = ROOT / "data" / "lcr" / "lcr_tidy.csv"
OUT = ROOT / "results" / "lcr"
(OUT / "tables").mkdir(parents=True, exist_ok=True)

EPS0 = 8.854e-12  # value typed in the workbook (F/m)
R_IN = 144e3  # ohm, NI USB-6009 input (given)
F_TAP = (5.0, 25.0)  # Hz (given)
OLD_CP_NF = (0.19, 7.7)  # previously assumed range of the coupling capacitance
OLD_THICK_UM = (140.0, 170.0)  # earlier assumed thickness range (sensitivity only)

# composition key, label, macro suffix, freq col, Cs, Cp, D, cached eps col, geometry cols (area cell, d cell)
BLOCKS = [
    ("PVDF", "PVDF", "Pvdf", "B", "C", "D", "E", "F", "A3", "A7"),
    ("PVDF/BaTiO3", "PVDF/BaTiO3", "Bto", "K", "L", "M", "N", "O", "J3", "J7"),
    ("PVDF/BaTiO3/1wt%MWCNT", "+1 wt% MWCNT", "CntOne", "T", "U", "V", "W", "X", "S3", "S7"),
    ("PVDF/BaTiO3/2wt%MWCNT", "+2 wt% MWCNT", "CntTwo", "AB", "AC", "AD", "AE", "AF", "AA3", "AA7"),
    ("PVDF/BaTiO3/3wt%MWCNT", "+3 wt% MWCNT", "CntThree", "AJ", "AK", "AL", "AM", "AN", "AI3", "AI7"),
]
R0, R1 = 25, 225


def eps_r(C, d_um, a_cm2):
    return C * (d_um * 1e-6) / (EPS0 * a_cm2 * 1e-4)


def load():
    df = pd.read_csv(TIDY)
    lab = {b[0]: (b[1], b[2]) for b in BLOCKS}
    df["label"] = df["composition"].map(lambda k: lab[k][0])
    df["suffix"] = df["composition"].map(lambda k: lab[k][1])
    df["eps_r_recomputed_stated_geometry"] = eps_r(df["Cs_F"], df["d_um_stated"], df["area_cm2_stated"])
    return df


def checks(df):
    """Sanity checks that are printed and stored in the README."""
    msgs = []
    s2 = df
    # recomputed eps vs the cached value in the sheet (column F, O, X, AF, AN)
    dev = (s2["eps_r_recomputed_stated_geometry"] - s2["eps_r_sheet_cached"]).abs().max()
    msgs.append(f"max |eps_r recomputed - sheet cached| over all 5 blocks = {dev:.3g}")
    # Cs vs Cp consistency Cp = Cs/(1+D^2) (series/parallel equivalent circuit)
    pred = s2["Cs_F"] / (1 + s2["D"] ** 2)
    msgs.append(f"max relative |Cp - Cs/(1+D^2)| = {((s2['Cp_F']-pred).abs()/s2['Cp_F']).max():.3g}")
    # sheet extra columns G,P,... are eps_r+0.01 and tan+0.0015: manual offsets
    msgs.append("Sheet columns G (eps_r+0.01), I (D+0.0015), P (+0.01), R (+0.00015) are hand offsets: NOT used")
    return msgs


def summarize(df):
    out = []
    s2 = df
    groups = [(k, g.sort_values("frequency_Hz")) for k, g in s2.groupby("composition", sort=False)]
    for k, g in groups:
        f = g["frequency_Hz"].to_numpy()
        C = g["Cp_F"].to_numpy() if g["Cp_F"].notna().all() else g["Cs_F"].to_numpy()
        cname = "Cp" if g["Cp_F"].notna().all() else "Cs (no Cp in block)"
        D = g["D"].to_numpy()
        i1k = int(np.where(f == 1000)[0][0])
        dC = np.diff(C)
        rho = pd.Series(f).rank().corr(pd.Series(D).rank())
        rhoC = pd.Series(f).rank().corr(pd.Series(C).rank())
        out.append(dict(
            composition=k, capacitance_quantity=cname, n_points=len(f), f_min_Hz=f.min(), f_max_Hz=f.max(),
            C_1kHz_F=C[i1k], C_fmin_F=C[0], C_fmax_F=C[-1],
            C_dispersion_pct=100 * (C[-1] - C[0]) / C[0],
            C_range_over_C1k_pct=100 * (C.max() - C.min()) / C[i1k],
            C_n_increases=int((dC > 0).sum()), C_spearman_vs_f=rhoC,
            D_1kHz=D[i1k], D_fmin=D[0], D_fmax=D[-1], D_min=D.min(), f_at_D_min_Hz=f[D.argmin()],
            D_max=D.max(), f_at_D_max_Hz=f[D.argmax()], D_ratio_fmax_over_fmin=D[-1] / D[0], D_spearman_vs_f=rho,
            area_cm2_stated=g["area_cm2_stated"].iloc[0], d_um_stated=g["d_um_stated"].iloc[0],
            eps_r_1kHz_stated_geometry=eps_r(C[i1k], g["d_um_stated"].iloc[0], g["area_cm2_stated"].iloc[0]),
            eps_r_1kHz_if_d_140um_A_12p25=eps_r(C[i1k], OLD_THICK_UM[0], 12.25) if k in [b[0] for b in BLOCKS] else np.nan,
            eps_r_1kHz_if_d_170um_A_12p25=eps_r(C[i1k], OLD_THICK_UM[1], 12.25) if k in [b[0] for b in BLOCKS] else np.nan,
            A_max_cm2_for_eps_r_1_at_stated_d=1e4 * C[i1k] * g["d_um_stated"].iloc[0] * 1e-6 / EPS0,
            note=("TO CONFIRM: eps_r<1 at stated geometry; do not present as a material property" if eps_r(C[i1k], g["d_um_stated"].iloc[0], g["area_cm2_stated"].iloc[0]) < 1
                  else "TO CONFIRM: geometry and poling state")))
    return pd.DataFrame(out)


def rc_table(summ):
    rows = []
    s = summ
    for _, r in s.iterrows():
        Cs_ = {"1kHz": r["C_1kHz_F"], "fmin": r["C_fmin_F"], "fmax": r["C_fmax_F"]}
        fc = 1 / (2 * math.pi * R_IN * r["C_1kHz_F"])
        Cmax_over = max(r["C_1kHz_F"], r["C_fmin_F"], r["C_fmax_F"])
        Cmin_over = min(r["C_1kHz_F"], r["C_fmin_F"], r["C_fmax_F"])
        row = dict(composition=r["composition"], Cp_1kHz_F=r["C_1kHz_F"], R_in_ohm=R_IN,
                   fc_Hz=fc, fc_Hz_at_Cp_max_over_range=1 / (2 * math.pi * R_IN * r["C_range_max_F"]) if "C_range_max_F" in r else np.nan,
                   tau_s=R_IN * r["C_1kHz_F"])
        row["ratio_5Hz_over_fc"] = F_TAP[0] / fc
        row["ratio_25Hz_over_fc"] = F_TAP[1] / fc
        # first-order high-pass magnitude |H| = x/sqrt(1+x^2), x = f/fc (R_p >> R_in assumed)
        for ft in F_TAP:
            x = ft / fc
            row[f"H_mag_{int(ft)}Hz_pct"] = 100 * x / math.sqrt(1 + x * x)
        row["R_p_eq_1kHz_ohm_loss_only"] = 1 / (2 * math.pi * 1000 * r["C_1kHz_F"] * r["D_1kHz"])
        row["Cp_needed_for_fc_eq_5Hz_F"] = 1 / (2 * math.pi * R_IN * F_TAP[0])
        row["Cp_needed_for_fc_eq_25Hz_F"] = 1 / (2 * math.pi * R_IN * F_TAP[1])
        row["Cp_factor_to_reach_fc_eq_25Hz"] = row["Cp_needed_for_fc_eq_25Hz_F"] / r["C_1kHz_F"]
        row["old_assumed_Cp_nF_low"], row["old_assumed_Cp_nF_high"] = OLD_CP_NF
        row["Cp_1kHz_vs_old_range"] = ("below old range" if r["C_1kHz_F"] * 1e9 < OLD_CP_NF[0] else
                                       "inside old range" if r["C_1kHz_F"] * 1e9 <= OLD_CP_NF[1] else "above old range")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = load()
    msgs = checks(df)
    summ = summarize(df)
    # add Cp extremes over the whole range for RC bound
    s2 = df
    mx = s2.groupby("composition", sort=False)["Cp_F"].max()
    summ["C_range_max_F"] = summ["composition"].map(mx)
    mn = s2.groupby("composition", sort=False)["Cp_F"].min()
    summ["C_range_min_F"] = summ["composition"].map(mn)
    summ.to_csv(OUT / "lcr_summary.csv", index=False)
    rc = rc_table(summ)
    rc["fc_Hz_at_Cp_max_over_range"] = [1 / (2 * math.pi * R_IN * mx[c]) for c in rc["composition"]]
    rc["fc_Hz_at_Cp_min_over_range"] = [1 / (2 * math.pi * R_IN * mn[c]) for c in rc["composition"]]
    rc.to_csv(OUT / "lcr_rc.csv", index=False)

    # figure data
    s2f = s2[["composition", "frequency_Hz", "Cs_F", "Cp_F", "D"]].copy()
    c1k = s2f[s2f["frequency_Hz"] == 1000].set_index("composition")["Cp_F"]
    s2f["Cp_pF"] = s2f["Cp_F"] * 1e12
    s2f["Cp_over_Cp_1kHz"] = s2f["Cp_F"] / s2f["composition"].map(c1k)
    s2f[["composition", "frequency_Hz", "Cp_F", "Cp_pF", "Cp_over_Cp_1kHz"]].to_csv(OUT / "fig_cp_vs_f.csv", index=False)
    s2f[["composition", "frequency_Hz", "D"]].to_csv(OUT / "fig_D_vs_f.csv", index=False)

    # table body
    lab = {b[0]: b[1] for b in BLOCKS}
    lines = ["% body of booktabs tabular; column order: composition & Cp(1 kHz) pF & Cp(1 MHz) pF & change % & D(1 kHz) & D(1 MHz) & f_c kHz",
             "% generated by code/lcr_analysis.py; eps_r deliberately not tabulated (geometry TO CONFIRM)"]
    s = summ.reset_index(drop=True)
    for i, r in s.iterrows():
        fc = rc.loc[i, "fc_Hz"] / 1e3
        lines.append(f"{lab[r['composition']]} & {r['C_1kHz_F']*1e12:.1f} & {r['C_fmax_F']*1e12:.1f} & {r['C_dispersion_pct']:.1f} & "
                     f"{r['D_1kHz']:.4f} & {r['D_fmax']:.4f} & {fc:.2f} \\\\")
    (OUT / "tables" / "tab_lcr_summary.tex").write_text("\n".join(lines) + "\n")

    # macros
    M = ["% numbers_lcr.tex: generated by code/lcr_analysis.py from data/lcr/lcr_tidy.csv",
         "% No digits in macro names. Units are in the name suffix (Pf, Nf, Khz, Hz, Pct)."]
    def add(name, val, fmt, note=""):
        M.append(f"\\newcommand{{\\lcr{name}}}{{{format(val, fmt)}}}% {note}")
    sufm = {b[0]: b[2] for b in BLOCKS}
    for i, r in s.iterrows():
        sf = sufm[r["composition"]]
        add("CpAtOneKhzPf" + sf, r["C_1kHz_F"] * 1e12, ".1f", "Cp at 1 kHz = lowest measured frequency")
        add("CpAtOneMhzPf" + sf, r["C_fmax_F"] * 1e12, ".1f", "Cp at 1 MHz = highest measured frequency")
        add("CpDispersionPct" + sf, r["C_dispersion_pct"], ".1f", "(Cp(1MHz)-Cp(1kHz))/Cp(1kHz), signed")
        add("DAtOneKhz" + sf, r["D_1kHz"], ".4f", "loss tangent D at 1 kHz")
        add("DAtOneMhz" + sf, r["D_fmax"], ".4f", "loss tangent D at 1 MHz")
        add("DMax" + sf, r["D_max"], ".4f", f"max D over range")
        add("FcKhz" + sf, rc.loc[i, "fc_Hz"] / 1e3, ".2f", "1/(2 pi R_in Cp(1kHz))")
        add("RatioFiveHzOverFc" + sf, rc.loc[i, "ratio_5Hz_over_fc"], ".2e", "5 Hz / f_c")
        add("RatioTwentyFiveHzOverFc" + sf, rc.loc[i, "ratio_25Hz_over_fc"], ".2e", "25 Hz / f_c")
        add("HighPassTwentyFiveHzPct" + sf, rc.loc[i, "H_mag_25Hz_pct"], ".2f", "|V_rec/V_oc| at 25 Hz if R_p >> R_in")
        add("EpsStatedGeom" + sf, r["eps_r_1kHz_stated_geometry"], ".2f", "DO NOT CITE as material property: 12.25 cm2 / 50 um TO CONFIRM")
        add("EpsSensLowThick" + sf, r["eps_r_1kHz_if_d_140um_A_12p25"], ".2f", "sensitivity only: d=140 um, A=12.25 cm2")
        add("EpsSensHighThick" + sf, r["eps_r_1kHz_if_d_170um_A_12p25"], ".2f", "sensitivity only: d=170 um, A=12.25 cm2")
        add("RpEqOneKhzMohm" + sf, rc.loc[i, "R_p_eq_1kHz_ohm_loss_only"] / 1e6, ".0f", "1/(2 pi 1kHz Cp D): dielectric-loss resistance at 1 kHz, NOT leakage")
    add("RinKohm", R_IN / 1e3, ".0f", "given")
    add("AreaStatedCmSq", 12.25, ".2f", "stated in workbook, TO CONFIRM")
    add("ThickStatedUm", 50, ".0f", "stated in workbook, TO CONFIRM")
    add("FreqLowKhz", s["f_min_Hz"].min() / 1e3, ".0f", "LCR range low")
    add("FreqHighMhz", s["f_max_Hz"].max() / 1e6, ".0f", "LCR range high")
    add("NFreqPoints", int(s["n_points"].iloc[0]), "d", "points per specimen")
    add("CpMinPf", s["C_1kHz_F"].min() * 1e12, ".1f", "min over specimens, Cp(1 kHz)")
    add("CpMaxPf", s["C_1kHz_F"].max() * 1e12, ".1f", "max over specimens, Cp(1 kHz)")
    add("FcMinKhz", rc["fc_Hz"].min() / 1e3, ".2f", "lowest f_c (largest Cp)")
    add("FcMaxKhz", rc["fc_Hz"].max() / 1e3, ".2f", "highest f_c (smallest Cp)")
    add("FcMinOverRangeKhz", rc["fc_Hz_at_Cp_max_over_range"].min() / 1e3, ".2f", "f_c at the largest Cp over 1 kHz to 1 MHz of any specimen")
    add("RatioMinFiveHz", rc["ratio_5Hz_over_fc"].min(), ".2e", "smallest f_tap/f_c")
    add("RatioMaxTwentyFiveHz", rc["ratio_25Hz_over_fc"].max(), ".2e", "largest f_tap/f_c")
    add("HighPassMinFiveHzPct", rc["H_mag_5Hz_pct"].min(), ".3f", "smallest |V_rec/V_oc| across specimens at 5 Hz if R_p >> R_in")
    add("HighPassMaxTwentyFiveHzPct", rc["H_mag_25Hz_pct"].max(), ".3f", "largest |V_rec/V_oc| at 25 Hz if R_p >> R_in")
    add("CpNeededFiveHzNf", rc["Cp_needed_for_fc_eq_5Hz_F"].iloc[0] * 1e9, ".0f", "Cp that would put f_c at 5 Hz")
    add("CpNeededTwentyFiveHzNf", rc["Cp_needed_for_fc_eq_25Hz_F"].iloc[0] * 1e9, ".1f", "Cp that would put f_c at 25 Hz")
    add("CpFactorMin", rc["Cp_factor_to_reach_fc_eq_25Hz"].min(), ".0f", "smallest factor by which Cp(1 kHz) must grow for f_c = 25 Hz")
    add("OldCpLowNf", OLD_CP_NF[0], ".2f", "previously assumed range")
    add("OldCpHighNf", OLD_CP_NF[1], ".1f", "previously assumed range")
    add("NSpecBelowOldRange", int((rc["Cp_1kHz_vs_old_range"] == "below old range").sum()), "d", "specimens with Cp(1 kHz) below the old lower bound")
    (OUT / "numbers_lcr.tex").write_text("\n".join(M) + "\n")

    print("\n".join(msgs))
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(summ.drop(columns=["note"]).T)
    print(rc.T)


if __name__ == "__main__":
    main()
