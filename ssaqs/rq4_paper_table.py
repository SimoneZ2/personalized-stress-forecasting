#!/usr/bin/env python3
"""RQ4: the SSAQS table of the paper (seed 42), from the metrics written by evaluate.py.

rq4_table.py summarizes the three seeds as mean ± sd; the paper reports seed 42.

Usage: python ssaqs/rq4_paper_table.py --dir rq4_eval [--seed 42]
"""

import argparse

import pandas as pd

ROWS = [("Fitbit Stress Score (raw)", "Fitbit Stress Score (raw)"),
        ("Fitbit Stress Score (calibrated)", "Fitbit Stress Score (calibrated)"),
        ("Personalized LSTM", "Personalized LSTM"),
        ("Full system", "LSTM+GBM (full system)")]
COLUMNS = [("MAE", "MAE", 2), ("MAE_peak", "MAE_peak", 2), ("rho_pooled", "ρ pooled", 3),
           ("CCC", "CCC", 3), ("rho_user", "ρ per-user", 3)]

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

md = [f"# RQ4 - SSAQS, next-day forecast (seed {a.seed}, table of the paper)", "",
      "| Scheme | Predictor | " + " | ".join(c[1] for c in COLUMNS) + " |",
      "|---|---|" + "---|" * len(COLUMNS)]
for schema in ["WF", "LUO"]:
    d = pd.read_csv(f"{a.dir}/rq4_metrics_{schema}.csv")
    d = d[d["file"].astype(str).str.startswith(f"seed{a.seed}__")].set_index("predictor")
    for name, label in ROWS:
        r = d.loc[name]
        md.append(f"| {schema} | {label} | " + " | ".join(f"{r[c]:.{n}f}" for c, _, n in COLUMNS) + " |")

open(f"{a.dir}/rq4_paper_table.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
