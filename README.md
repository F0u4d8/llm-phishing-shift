# Lightweight LLM-Phishing Detection under Generator, Language and Channel Shift

Code, generated data, and results for:

> M. F. Mahkouka, S. Benkrama, M. A. Kenanda. *When Lightweight Phishing Detectors Meet Unseen LLMs and Languages: A Label-Free Domain Adaptation Study.* CSNT 2026, Mostaganem, Algeria.

A lightweight CPU detector (10 surface features + character 3–5-gram TF-IDF, logistic regression) is trained on one domain and tested on another. A domain is a (author, language, channel) triple: authors = human or one of three LLMs, languages = English/French/Arabic, channels = email/SMS. We compare naive transfer, CORAL, self-training, few-shot target labels (k = 10/50/100/500), a multi-source baseline, and a full-supervision upper bound, reporting MCC and ROC-AUC over 20 disjoint domains and 380 transfers.

## Repository layout

| Path | Content |
|---|---|
| `phishing_uda.ipynb` | Original pipeline: data normalisation, features, CORAL, self-training, generation prompts |
| `revision/rerun_fixed.py` | **Corrected, leakage-free experiment** that produces every number in the paper |
| `revision/nb.py` | Loads the notebook's functions so the rerun uses identical code |
| `revision/make_figs.py` | Figures from the results |
| `data/generated/` | LLM-generated phishing (DeepSeek-V3, GPT-5.5, Claude Opus 4.8; EN/FR/AR; email/SMS), with scenario ids |
| `results/revision/` | `transfer_matrix_fixed.csv` (every transfer, seed and method), `domains.csv`, figures |

## Generated data

Each file `<generator>_<lang>_<channel>.csv` holds synthetic phishing generated through the public chat interface of the named assistant from 18 pretext scenarios (about 40 messages per scenario). The messages are for **defensive research only**: links are de-fanged placeholders (`hxxp://verify-example[.]tld`), and the messages contain no real URLs or personal data.

## Human corpora (not redistributed)

Download these from their sources, then run the notebook's data-preparation cell (Part 1b). It writes `data/real/{lang}_{channel}.csv` with columns `text,label,lang,modality`.

| Domain | Source |
|---|---|
| EN email | TREC 2005–2007 emails in the MeAJOR release — https://zenodo.org/records/18471483 |
| EN SMS | UCI SMS Spam Collection — https://archive.ics.uci.edu/dataset/228 |
| FR email | https://www.kaggle.com/datasets/kinoux/french-spamham-detection-free |
| AR email | https://www.kaggle.com/datasets/zeyadkarim/arabic-phishing-dataset-10k-csv (280 unique messages after deduplication) |
| AR SMS | https://www.kaggle.com/datasets/sultannizar/arabic-phishing-dataset |

In these corpora the positive class is spam, including phishing.

## Reproduce

```bash
pip install -r requirements.txt
python3 revision/rerun_fixed.py   # about 80 min on an 8-core Apple M3
python3 revision/make_figs.py
```

## Protocol (leakage controls)
- Exact duplicates are removed within and across all corpora.
- The legitimate messages of each language and channel are partitioned into disjoint shares, one per domain, so no message appears in two domains.
- Source-side phishing uses scenarios s01–s09; target-side phishing uses s10–s18.
- The target is split 50/50 into an unlabelled pool (used for adaptation) and a held-out test half (used for every reported score). The three seeds control this split.
- French SMS is excluded: no native corpus exists, and its generated legitimate file is unusable.

## Licence
Code: MIT. Generated data: CC BY 4.0.
