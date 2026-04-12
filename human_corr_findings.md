# Human Agreement Findings

Date: 2026-04-12

This note records the investigation into unexpectedly low human reviewer agreement on the ICLR 2026 unbalanced dataset.

## What was checked

- Re-fetched the full ICLR 2025 venue from OpenReview into `iclr2025_data/all_notes.json` using the `neg` conda environment.
- Re-fetched the full ICLR 2026 venue into `tmp/iclr2026_refetch_all_notes.json`.
- Tightened OpenReview parsing in `fetch_iclr.py` to:
  - target `ICLR.cc/2026/Conference` explicitly
  - count only exact `.../-/Official_Review` invitations
  - count only exact `.../-/Decision` invitations
- Replaced random split-half estimation with exact half-split enumeration in `metric.py`.
- Added `plot_human_corr_compare.py` and generated `human_corr_2025_vs_2026.png`.

## Main result

Using exact split-half enumeration on the full-note dumps:

- ICLR 2025:
  - papers: 11,512
  - split-half Pearson: 0.5509
  - split-half Spearman: 0.5638
  - split-half MAE: 1.0081
  - Spearman-Brown corrected split-half Pearson: 0.7104
  - one-vs-rest Pearson: 0.4937

- ICLR 2026:
  - papers: 18,790
  - split-half Pearson: 0.2940
  - split-half Spearman: 0.2805
  - split-half MAE: 1.3305
  - Spearman-Brown corrected split-half Pearson: 0.4544
  - one-vs-rest Pearson: 0.2612

## Interpretation

- The low 2026 agreement is real and is not explained by the cached fetch being malformed.
- Fresh 2026 and cached 2026 had identical score lists and identical `gt_binary` labels.
- The only cached-vs-fresh 2026 differences were metadata drift in venue / decision strings.
- 2025 shows substantially stronger reviewer agreement than 2026 under the same parsing and metric code.

## Output artifact

- Comparison figure: `human_corr_2025_vs_2026.png`
