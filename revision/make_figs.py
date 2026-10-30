"""Figures + tables for the corrected results (results/revision/)."""
from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parent.parent / "results" / "revision"; F = R / "figures"; F.mkdir(exist_ok=True)
d = pd.read_csv(R / "transfer_matrix_fixed.csv")
p = d.groupby(["src", "tgt", "regime", "method"])[["mcc", "auc"]].mean().reset_index()
single = p[p.regime != "multi-source"]
order = ["generator", "language", "modality", "generator+language", "generator+modality",
         "language+modality", "generator+language+modality"]
lab = ["G", "L", "M", "G+L", "G+M", "L+M", "G+L+M"]

# Fig 1: regime x method (MCC)
g = single.pivot_table(index="regime", columns="method", values="mcc").reindex(order)
fig, ax = plt.subplots(figsize=(7, 3.2)); x = np.arange(len(order)); w = 0.2
for i, (m, n) in enumerate([("naive", "Naive"), ("coral", "CORAL"), ("selftrain", "Self-training"), ("upper", "Supervised (upper)")]):
    ax.bar(x + (i - 1.5) * w, g[m], w, label=n)
ax.set_xticks(x, lab); ax.set_ylabel("Mean MCC"); ax.set_xlabel("Shifted axes (G=generator, L=language, M=modality)")
ax.set_ylim(0, 1.05); ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)
fig.tight_layout(); fig.savefig(F / "fig_regime.png", dpi=300)

# Fig 2: few-shot label budget
ks = [0, 10, 50, 100, 500]; ms = ["naive", "fewshot_10", "fewshot_50", "fewshot_100", "fewshot_500"]
sub = single[single.method.isin(ms)]
both = sub.pivot_table(index=["src", "tgt"], columns="method", values="mcc").dropna().index  # pairs with all budgets
fig, ax = plt.subplots(figsize=(4.2, 3))
for met, mk in (("mcc", "o-"), ("auc", "s--")):
    t = sub.set_index(["src", "tgt"]).loc[both].reset_index().groupby("method")[met].mean()[ms]
    ax.plot(range(len(ks)), t.values, mk, label=met.upper())
    up = single[single.method == "upper"].set_index(["src", "tgt"]).loc[both][met].mean()
    ax.axhline(up, ls=":", c="grey"); ax.text(0, up + 0.01, f"full-target upper bound ({met.upper()})", fontsize=7)
ax.set_xticks(range(len(ks)), [str(k) for k in ks])
ax.set_xlabel("Labelled target messages k"); ax.set_ylabel("Score"); ax.set_ylim(0.2, 1.05); ax.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(F / "fig_fewshot.png", dpi=300)

# Fig 3: language heatmap (naive MCC) with pair counts
n = single[single.method == "naive"].copy()
n["sl"] = n.src.str.split("|").str[1]; n["tl"] = n.tgt.str.split("|").str[1]
L = ["en", "fr", "ar"]
h = n.pivot_table(index="sl", columns="tl", values="mcc").reindex(index=L, columns=L)
c = n.pivot_table(index="sl", columns="tl", values="mcc", aggfunc="count").reindex(index=L, columns=L)
fig, ax = plt.subplots(figsize=(3.4, 2.9)); im = ax.imshow(h, cmap="viridis", vmin=0.1, vmax=0.8)
ax.set_xticks(range(3), [s.upper() for s in L]); ax.set_yticks(range(3), [s.upper() for s in L])
ax.set_xlabel("Target"); ax.set_ylabel("Source")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{h.iloc[i, j]:.2f}\n(n={c.iloc[i, j]})", ha="center", va="center", fontsize=7,
                color="w" if h.iloc[i, j] < 0.5 else "k")
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(F / "fig_lang.png", dpi=300)
print("figures ->", F)
