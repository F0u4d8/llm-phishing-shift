"""Corrected transfer-matrix rerun (CSNT 2026 camera-ready / JISA revision).

Fixes found while answering the CSNT reviews:
  1. LEAK: generator domains reused the same human legit messages (same sample seed),
     so source and target shared legit texts. Now every legit message belongs to exactly
     one domain (disjoint partition per language x modality).
  2. Scenario overlap: all generators used the same 18 pretexts. Now source-side phishing
     uses scenarios s01-s09 and target-side phishing uses s10-s18.
  3. FR-SMS cell dropped: its only legit file had 51 unique texts, half labelled spam.
  4. Human domains capped at CAP per class so all domains are of comparable size.
Adds: ROC-AUC, few-shot target labels (k), multi-source baseline, per-method wall time.
Reuses the notebook's own features/CORAL/self-training code (revision/nb.py).

Run:  python3 revision/rerun_fixed.py        -> results/revision/*.csv
"""
import sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import numpy as np, pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "revision"; OUT.mkdir(parents=True, exist_ok=True)
CAP = 1500                     # max messages per class per domain
SRC_SCEN = {f"s{i:02d}" for i in range(1, 10)}   # s01-s09 -> source side; s10-s18 -> target side
KS = (10, 50, 100, 500)
SEEDS = range(3)
G = None                       # notebook library, loaded per process


def _lib():
    global G
    if G is None:
        G = nb.load(("import re, json", "def numeric_features", "def load_all_real"))
    return G


def load_generated():
    rows = []
    for f in sorted((ROOT / "data" / "generated").glob("*.csv")):
        if f.stem.startswith("legit_"):
            continue                                             # FR-SMS legit file: dropped (see 3.)
        gen, lang, mod = f.stem.rsplit("_", 2)
        if (lang, mod) == ("fr", "sms"):
            continue
        df = pd.read_csv(f, engine="python", on_bad_lines="skip")
        df.columns = [c.strip().lower() for c in df.columns]
        if "body" in df.columns:
            text = df.get("subject", "").fillna("").astype(str) + "  " + df["body"].fillna("").astype(str)
        else:
            text = df[next(c for c in ("message", "text", "full_text") if c in df.columns)].astype(str)
        rows.append(pd.DataFrame(dict(text=text.str.strip(), label=1, scen=df["scenario_id"].astype(str),
                                      generator=gen, lang=lang, modality=mod)))
    g = pd.concat(rows, ignore_index=True)
    return g[g.text.str.len() >= 15]


def build_domains():
    real = _lib()["load_all_real"]().drop_duplicates("text")
    gen = load_generated().drop_duplicates("text")
    gen = gen[~gen.text.isin(real.text)]
    rng = np.random.RandomState(0)
    doms = {}
    for (lang, mod), r in real.groupby(["lang", "modality"]):
        gens = sorted(gen[(gen.lang == lang) & (gen.modality == mod)].generator.unique())
        legit = r[r.label == 0].sample(frac=1, random_state=0)
        parts = np.array_split(legit, 1 + len(gens))             # disjoint legit shares
        hum_ph = r[r.label == 1]
        hum_ph = hum_ph.sample(min(len(hum_ph), CAP), random_state=0)
        doms[f"human|{lang}|{mod}"] = pd.concat([hum_ph.assign(scen="na"), parts[0].head(CAP).assign(scen="na")])
        for g_, part in zip(gens, parts[1:]):
            ph = gen[(gen.generator == g_) & (gen.lang == lang) & (gen.modality == mod)]
            doms[f"{g_}|{lang}|{mod}"] = pd.concat([ph.head(CAP), part.head(CAP).assign(scen="na")])
    for k, d in doms.items():
        doms[k] = d[["text", "label", "scen"]].reset_index(drop=True)
    # sanity: no text in two domains
    seen = {}
    for k, d in doms.items():
        for t in d.text:
            assert seen.setdefault(t, k) == k, f"text shared by {seen[t]} and {k}"
    return doms


def side(d, role):
    """Source side keeps scenarios s01-s09; target side s10-s18. Human/legit rows (scen=na) kept."""
    keep = d.scen.eq("na") | (d.scen.isin(SRC_SCEN) if role == "src" else ~d.scen.isin(SRC_SCEN))
    return d[keep]


def _scores(m, X):
    return m.predict_proba(X)[:, 1]


def _metrics(y, s, tau=0.5):
    yp = (s >= tau).astype(int)
    auc = roc_auc_score(y, s) if len(set(y)) == 2 else np.nan
    return matthews_corrcoef(y, yp), auc


def eval_pair(args):
    a, b, src, tgt = args
    L = _lib()
    rows = []
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(tgt)); cut = len(tgt) // 2
        test, pool = tgt.iloc[idx[:cut]], tgt.iloc[idx[cut:]]
        if test.label.nunique() < 2 or pool.label.nunique() < 2:
            continue
        tf = L["make_transform"](list(src.text))
        Xs, ys = tf(list(src.text)), src.label.values
        Xp, Xt, yt = tf(list(pool.text)), tf(list(test.text)), test.label.values

        def rec(method, s, t0, tau=0.5):
            mcc, auc = _metrics(yt, s, tau)
            rows.append(dict(src=a, tgt=b, seed=seed, method=method, mcc=mcc, auc=auc, secs=time.time() - t0))

        t0 = time.time(); m = L["make_model"]("logreg").fit(Xs, ys); rec("naive", _scores(m, Xt), t0)
        t0 = time.time(); Xa = L["coral"](Xs, Xp); m = L["make_model"]("logreg").fit(Xa, ys)
        rec("coral", _scores(m, Xt), t0, L["recalibrate_threshold"](_scores(m, Xp)))
        t0 = time.time(); m = L["self_train"]("logreg", Xs, ys, Xp); rec("selftrain", _scores(m, Xt), t0)
        t0 = time.time(); m = L["make_model"]("logreg").fit(Xp, pool.label.values); rec("upper", _scores(m, Xt), t0)
        for k in KS:                                             # source + k labelled target messages
            if k >= len(pool):
                continue
            t0 = time.time()
            sel = pool.groupby("label", group_keys=False).apply(
                lambda g: g.sample(max(1, round(k * len(g) / len(pool))), random_state=seed))
            m = L["make_model"]("logreg").fit(np.vstack([Xs, tf(list(sel.text))]),
                                              np.concatenate([ys, sel.label.values]))
            rec(f"fewshot_{k}", _scores(m, Xt), t0)
    return rows


def eval_multi(args):
    b, src, tgt = args
    return [dict(r, src="multi") for r in eval_pair(("multi", b, src, tgt))
            if r["method"] in ("naive", "upper")]


def regime(a, b):
    if a == "multi":
        return "multi-source"
    (ga, la, ma), (gb, lb, mb) = a.split("|"), b.split("|")
    d = [n for n, x, y in (("generator", ga, gb), ("language", la, lb), ("modality", ma, mb)) if x != y]
    return "+".join(d)


if __name__ == "__main__":
    doms = build_domains()
    pd.DataFrame([dict(domain=k, n=len(d), phish=int(d.label.sum()), legit=int((d.label == 0).sum()),
                       median_chars=int(d.text.str.len().median())) for k, d in doms.items()]
                 ).to_csv(OUT / "domains.csv", index=False)
    print(f"{len(doms)} domains, all disjoint")
    names = sorted(doms)
    jobs = [(a, b, side(doms[a], "src"), side(doms[b], "tgt")) for a in names for b in names if a != b]
    multi = [(b, pd.concat([side(doms[a], "src").groupby("label", group_keys=False)
                            .apply(lambda g: g.sample(min(len(g), 300), random_state=0))
                            for a in names if a != b]), side(doms[b], "tgt")) for b in names]
    t = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for i, r in enumerate(ex.map(eval_pair, jobs, chunksize=2)):
            rows += r
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} pairs  {time.time() - t:.0f}s", flush=True)
        for r in ex.map(eval_multi, multi):
            rows += r
    df = pd.DataFrame(rows)
    df["regime"] = [regime(a, b) for a, b in zip(df.src, df.tgt)]
    df.to_csv(OUT / "transfer_matrix_fixed.csv", index=False)
    print("done", len(df), "rows", f"{time.time() - t:.0f}s")
