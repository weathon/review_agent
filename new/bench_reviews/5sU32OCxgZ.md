Now let me run calibration searches to anchor my scoring.Now I have enough context to write the final review.Now I have sufficient calibration. Let me write the final consolidated review.

---

## Summary

TTVD proposes a geometric framework for test-time adaptation (TTA) by interpreting neighbor-based TTA methods through the lens of Voronoi Diagrams. Starting from a basic VD formulation (entropy minimization over distance-softmax logits), the paper progressively extends to a Cluster-induced Voronoi Diagram (CIVD) that uses rotation-augmented class cluster sites, and a Power Diagram (PD) for flexible boundary-based noisy sample filtering. Experiments on CIFAR-10-C, CIFAR-100-C, ImageNet-C, and ImageNet-R using the standardized TTAB toolkit show improvements over prior methods in both error rate and ECE calibration.

---

## Strengths

- **Geometric reframing with interpretable visualizations (Figure 1, Definition 3.1–3.4):** Connecting neighbor-based TTA to Voronoi Diagrams provides a unified vocabulary and the 2D visualization of partition boundaries (MNIST-C) is genuinely illuminating. The progression VD → CIVD → CIPD in Figure 3 gives a clean theoretical narrative.
- **Calibration metric included (Table 1):** Reporting ECE alongside error is a meaningful addition largely absent in TTA papers. The ECE improvements are substantial (3.4–4.3%), well exceeding the error improvements, and suggest TTVD produces more trustworthy confidence estimates.
- **Clean progressive ablation (Table 2):** The within-framework ablation clearly shows VD (28.4%) → CIVD (22.7%) → CIPD (20.5%) on CIFAR-10-C, with each component providing measurable gains.
- **Rigorous standardized evaluation:** Using TTAB as a peer-reviewed, open-source codebase with grid-searched hyperparameters is a good-practice choice, lending credibility to the experimental setup.
- **Analysis of batch size and label shift effects (Appendix B):** Investigating small-batch and non-i.i.d. stream behavior directly addresses deployment realism and differentiates this work from papers that only evaluate on the canonical setting.

---

## Weaknesses

### Fatal
None that fully invalidate the paper.

### Major

- **Backbone trained with label augmentation, but no baseline is (Section 4.1 — Key confound):** TTVD uses a backbone trained with "self-supervised label augmentation" (Lee et al., 2020), which extends the training objective to jointly predict original class labels *and* rotation-augmented pseudo-labels for four rotation angles. The CIVD mechanism then constructs Voronoi sites $\mu_k^{(\alpha)}$ precisely from these rotation-augmented class means (Section 3.2). Every comparison method in Table 1 (TENT, SAR, T3A, TAST, SHOT, etc.) uses a standard backbone without this augmentation. Since the 5.7% VD→CIVD gain (Table 2) directly leverages the rotation structure embedded into the backbone, it is impossible to determine whether this gain comes from CIVD's multi-site geometric mechanism or from the richer feature space induced by label-augmented pre-training. The critical ablation — CIVD applied on a standard backbone, or all baselines re-run on the label-augmented backbone — is entirely absent. As stated in Section 4.1: *"For TTVD, we trained ResNet-26…using label augmentation (Lee et al., 2020)"*. This asymmetry undermines the central empirical claim.

- **Most relevant baseline (AdaNPC) excluded from main comparison (Section 4.2):** AdaNPC (Zhang et al., 2023) is described in the related work as the closest mechanistic neighbor to TTVD, yet it appears only in a restricted Table 3 covering four blur corruption types on ImageNet-C. No rationale is given for why AdaNPC is absent from Table 1's cross-dataset comparison. Given that the backbone confound (above) makes the full-table comparison already uncertain, restricting the nearest-neighbor baseline to a partial comparison compounds the evidential gap.

### Minor

- **Small absolute margins without variance (Table 1):** The headline improvements over prior work are 0.8%, 0.7%, 1.6%, and 0.7% in error. These are single-run, point estimates with no standard deviations, confidence intervals, or multi-seed results. While single-run evaluation is common in TTA (following TTAB), the margins are close to run-to-run noise, and the paper claims "remarkable improvements" in the abstract — a characterization that is not warranted by 0.7% gaps. At minimum, the language should be moderated.

- **Mechanistic claim for CIVD avoiding negative transfer is unsubstantiated (Section 3.2):** The paper states that CIVD "avoids negative transfer since the objective is now unified" (Section 3.2, final paragraph). However, there is no gradient conflict analysis, no ablation comparing CIVD's unified loss against a naïve sum of self-supervision + entropy minimization on the same backbone, and no theoretical derivation supporting this. The claim is asserted without evidence.

- **Non-monotonic result in Table 4 unexplained:** Table 4 shows error of 59.8% at 10% data, 59.8% at 5%, and 59.9% at 1%. The 1% result is *slightly worse* than both others, but the paper frames this as "high robustness." The non-monotonicity is not discussed and the differences are within noise, making this table inconclusive as evidence of robustness to class-mean precision.

- **Negative power in Eq. 6 for non-integer γ (Section 3.3):** CIPD's influence function (Equation 6) contains the term $\{d(\mu_k^{(\alpha)}, z)^2 - v_k^2\}^\gamma$. The quantity $d^2 - v_k^2$ can be negative for points inside the power cell, making $(\cdot)^\gamma$ undefined for non-integer $\gamma$. The paper does not acknowledge or constrain this case.

### Trivial
- Figure 4's adaptation curves start from different initial error rates for TTVD vs. TENT/SAR (because of the different backbone), making it unclear whether TTVD adapts faster or simply starts from a better-calibrated initial state.

---

## Nice-to-Haves

- Ablation of CIVD with a standard (non-label-augmented) backbone, which would unambiguously demonstrate the contribution of the multi-site geometric mechanism.
- Evaluation on a continual/non-i.i.d. shifting corruption benchmark (e.g., CoTTA's setting), which would validate the paper's claim that CIVD is particularly suited for "small batches and non-i.i.d. streams."
- Separate ablation of the PD's two roles — (a) weight-adjusted cell boundaries vs. (b) PD–VD filtering — to identify which drives the CIVD→CIPD gain.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

1. **Harsh Critic — "Geometric framework is just rebranding":** Partially valid (VD = nearest-prototype classification by definition, CIVD = multi-augmentation TTA), but too sweeping. The geometric vocabulary yields interpretability advantages (e.g., the entropy landscape analysis in Figure 2a) and connects to an established mathematical literature. While the practical novelty of the mechanisms is modest, calling it purely descriptive ignores the unification of the loss and the geometric justification for filtering. Removed as a separate fatal weakness; merged into the minor concern about unsubstantiated claims.

2. **Harsh Critic — Equation 4 exponent shows "7" instead of γ:** This is a PDF parser artifact (the γ symbol rendered as "7" in the extracted text). The hard rule on formatting artifacts applies.

3. **Harsh Critic — "Temperature τ not specified":** The paper states hyperparameters are grid-searched following TTAB guidelines. Requesting explicit disclosure of τ is a reproducibility nitpick of a hyperparameter covered by the TTAB grid-search protocol. Removed per nitpick rule.

4. **Harsh Critic — Gridsearching more hyperparameters gives more overfitting freedom:** This is generic criticism applicable to any TTA paper that uses grid search. Not specific enough to be a substantive weakness.

5. **Strength Finder — "Sustained adaptation over time (Figure 4)":** While Figure 4 does show TTVD decreasing monotonically over 750 batches, the comparison is confounded by the different starting point from the label-augmented backbone. Removed as a standalone strength per the rule that weaknesses trump conflicting strengths.

6. **Strength Finder — "Robustness to class mean precision (Table 4)":** Results are within noise (59.8%, 59.8%, 59.9%) and the 1% result is slightly *worse*. The table does not compellingly demonstrate robustness. Removed as a genuine strength; re-classified as a minor concern above.

---

## Novel Insights

The paper's most genuinely novel observation is the geometric reinterpretation of why neighbor-based TTA methods fail: not because of prototype drift per se, but because nearest-prototype classification creates fixed Voronoi cells that cannot respond to multi-source influences or flex their boundaries for noisy sample rejection. The CIVD framing makes explicit that the 4×-augmented rotation sites impose a richer partition structure, and the PD subtraction trick provides a geometrically motivated alternative to entropy-threshold filtering. These ideas could guide future work on adaptive partition geometry for online adaptation. However, whether the empirical gains are attributable to this geometric mechanism vs. the backbone pre-training advantage remains unresolved.

---

## Suggestions

1. **Re-run a critical ablation on a standard backbone**: Train TTVD (especially CIVD) on a backbone without label augmentation, and compare VD vs. CIVD on equal footing with the baselines. This single experiment would either confirm or challenge the paper's central claim.
2. **Either add AdaNPC to Table 1 or explicitly justify its exclusion** in the caption (e.g., if AdaNPC's published paper only reports on blur corruptions, say so explicitly).
3. **Moderate the abstract's claim**: Replace "remarkable improvements" with accurate quantitative characterization (e.g., 0.7–1.6% error and 1.8–4.3% ECE reduction).
4. **Disentangle the PD contribution**: Add a two-cell ablation within CIPD separating the boundary-shifting effect from the PD–VD filtering effect.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Relation to paper under review |
|---|---|---|
| `3Z2flzXzBY.md` (PASLE TTA) | **6.4**, Accept | Similar TTA paper on CIFAR/ImageNet-C; no backbone confound; accepted despite novelty questions |
| `eXrUdcxfCw.md` (CTA EMA prototypes) | **4.8**, Reject | TTA prototype method rejected for tiny margins (<0.5%), missing baselines, and low technical novelty; this paper's issue severity is comparable but TTVD has a more principled framework |
| `sEMJ1PLSZR.md` (AEA for TTA) | **6.25**, Accept | TTA paper with energy-based reinterpretation; one reviewer flagged backbone mismatch with a baseline but it didn't kill the paper |
| `rW3NVhKtQ2.md` (TT-GREB GNN) | **4.5**, Reject | Test-time adaptation under distribution shifts on graphs; rejected for weak technical contribution |
| `BmG88rONaU.md` (TCR cross-modal) | **7.5**, Accept (Spotlight) | Strong TTA paper with clearly novel design and large improvements; clearly above TTVD |

**Reasoning:** TTVD sits between the rejected eXrUdcxfCw (4.8) and the accepted PASLE (6.4). The geometric framework is more principled than EMA prototypes and the improvement margins are larger; however, the label-augmentation backbone confound — absent in PASLE and most accepted TTA papers — is a genuine major issue that leaves the central algorithmic claim unsupported by the existing experiments. The AEA paper (6.25) was accepted despite a reviewer flagging a backbone architecture difference, but in TTVD the confound is more targeted: the backbone was explicitly trained to embed rotation structure that CIVD directly exploits. This places TTVD just below borderline acceptance. Score: **5.0**.

**Originality:** Moderate — connects existing techniques to a well-studied geometric structure; the CIVD and PD applications are genuinely new to TTA but mechanistically equivalent to multi-augmentation prototype TTA plus boundary-based filtering.  
**Importance of research question:** High — TTA is a practically critical problem.  
**Claims supported by evidence:** Partially — the ablation within TTVD (Table 2) is self-consistent, but the cross-method comparisons in Table 1 are confounded by backbone choice.  
**Soundness of experiments:** Moderate — TTAB is rigorous, but the backbone asymmetry and missing AdaNPC in Table 1 weaken evidential force.  
**Clarity of writing:** Good — the progressive construction and geometric definitions are clearly presented.  
**Value to research community:** Moderate — the geometric perspective is transferable to future work, but the core claim needs the missing ablation to be convincing.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>