Now let me check a few calibration papers to anchor my score.Now I have enough context to write the consolidated review.

---

## Summary

TTVD proposes a geometric framework for test-time adaptation (TTA) by formalizing neighbor-based TTA methods through Voronoi Diagrams (VD) and extending them to two advanced structures: the Cluster-induced Voronoi Diagram (CIVD), which integrates self-supervision and entropy minimization via multi-source influence, and the Power Diagram (PD), which identifies noisy boundary samples through flexible cell boundaries. Experiments on CIFAR-10/100-C, ImageNet-C, and ImageNet-R using the standardized TTAB toolkit show TTVD achieves competitive accuracy and notably strong Expected Calibration Error (ECE) improvements over prior methods.

---

## Strengths

- **Novel geometric perspective.** Formalizing TTA through Voronoi/Power Diagrams is a genuinely novel conceptual lens. The progression VD → CIVD → CIPD provides a principled and readable geometric narrative, and connections to prior geometric deep learning work (DeepVoro, iVoro) are well-situated.
- **Strong ECE improvements.** The calibration gains are substantial and consistent: 3.4% on CIFAR-10-C, 1.8% on CIFAR-100-C, 4.1% on ImageNet-C, and 4.3% on ImageNet-R. These are practically significant for deployment scenarios where trustworthy confidence estimates matter.
- **Standardized evaluation.** Use of the peer-reviewed TTAB toolkit with grid-searched hyperparameters increases reproducibility credibility and reduces cherry-picking risk.
- **Robustness to class mean precision (Table 4).** The finding that TTVD maintains stable ImageNet-C performance with only 1% of training data for class means is practically relevant and honestly reported.
- **Ablation with VD → CIVD → CIPD (Table 2).** The incremental decomposition is clear and the CIVD improvement (5.7%) and CIPD improvement (2.2%) over CIFAR-10-C provide structured evidence for the component design.

---

## Weaknesses

### Fatal
*None that individually undermine the entire contribution.*

### Major

- **Information asymmetry: source training data used for Voronoi sites, but not disclosed as a setting assumption.** Section 4.1 states "We use the full training set of CIFAR-10, CIFAR-100 to compute the class means for Voronoi sites and 10% of ImageNet for similar calculation." This is transparently reported, but the paper's Problem Setup (Section 3) defines TTA as "without accessing the original training data or labels during test time," then quietly uses source data for class means as an "offline" precomputation. The core baselines (Tent, SAR, NOTE, etc.) use *only* the pretrained model checkpoint — no source data at all. TTVD has a privileged source-statistic advantage that is not acknowledged as a design assumption and is not used to level-set comparisons. The paper should explicitly declare that TTVD assumes access to source training data (or model outputs over it) for prototype initialization, compare against baselines in the same setting, and discuss this limitation rather than burying it in implementation details.

- **Thin mechanistic justification for CIVD and PD contributions.** The paper claims (Section 3.2) that CIVD "avoids the negative transfer since the objective is now unified" and (Section 3.3) that PD-based diagram subtraction identifies noisy boundary samples more effectively than entropy-based filtering. These are causal claims, but the evidence consists almost entirely of performance deltas in Table 2 and qualitative 2D MNIST visualizations. There is no gradient conflict analysis comparing objectives with/without CIVD, no quantitative evaluation of filtering quality (precision/recall), and no ablation isolating PD filtering from the influence of CIVD expansion. The claims may be true, but the paper does not establish *why* these structures help.

- **Small accuracy margins without uncertainty quantification.** On CIFAR-10-C and CIFAR-100-C, gains over the best prior method are 0.8% and 0.7% respectively, and on ImageNet-R, 0.7%. These are plausible but fragile without variance across runs or error bars. The paper emphasizes TTAB's "rigid" settings, but with single-run results at these margins, the headline claim of state-of-the-art accuracy is not fully established. (ECE improvements are substantially larger and more convincing.)

### Minor

- **Ablation confined to CIFAR-10-C.** Table 2 ablates VD → CIVD → CIPD on CIFAR-10-C only. Whether the ~5.7% (CIVD) and ~2.2% (CIPD) gains replicate on harder, higher-dimensional datasets (CIFAR-100-C, ImageNet-C) is unknown. The geometric structures likely behave differently in higher-dimensional feature spaces.

- **Exponent 7 in CIVD influence function (Eq. 4) unjustified.** No theoretical motivation or empirical sensitivity analysis is provided for this highly specific value. Given that γ controls the influence scale and the exponent shapes the geometry, this is a non-trivial choice.

- **Core CIPD algorithm (Algorithm 3) relegated to Appendix H.** The method used in all experiments is CIPD, but its full algorithm appears only in the appendix, with the main text deferring to "Algorithm 3 in Appendix H." This makes the paper difficult to follow and evaluate as a standalone submission.

- **Table 4 "identical" results are suspicious.** ImageNet-C error is 59.8 → 59.8 → 59.9 for 10%, 5%, 1% of training data used for class means. While the paper frames this as demonstrating robustness, the near-identical values raise questions about whether the geometric structure meaningfully depends on the class mean precision, or whether the evaluation is not sensitive enough to discriminate. A brief analysis of what changes (if anything) in the actual Voronoi sites would be informative.

- **Adaptation curve narrative is internally contradictory.** Section 4.2 first states "Tent and SAR do not show signs of overfitting," then later says their "early stagnation may indicate potential overfitting." These are different claims and neither is carefully substantiated.

### Trivial
- Section 3.1 uses "$\mathcal{D}_{test}$" where "$\mathcal{D}_{train}$" is meant (noted as likely typo by Harsh Critic; confirmed in the text: "since the training distribution $\mathcal{D}_{test}$ deviates from $\mathcal{D}_{train}$").

---

## Nice-to-Haves

- **Ablation on harder datasets (CIFAR-100-C, ImageNet-C):** Reproducing Table 2's VD/CIVD/CIPD breakdown on harder benchmarks would significantly strengthen the component validation.
- **Gradient conflict analysis:** A cosine similarity analysis of self-supervision vs. entropy gradients with and without CIVD would directly test the "negative transfer avoidance" claim.
- **Computational overhead comparison:** TTVD adds geometric computations (multi-site influences, PD weight computation, diagram subtraction) per batch. A wall-clock comparison against Tent and SAR would help practitioners evaluate the method.
- **Sensitivity analysis for γ and τ:** Brief ablations on the influence scale and temperature would validate that the method is robust to hyperparameter choices.
- **Feature-space visualization (t-SNE/UMAP):** Showing how VD/CIVD/PD partitions shift during adaptation in realistic high-dimensional settings (not just 2D MNIST-C) would validate the geometric intuition.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic] Headline improvement claim overstated in the abstract:** The abstract says "remarkable improvements." This is a mild overstatement for accuracy but is reasonably accurate for ECE. While imprecise, this is a style/framing nitpick rather than a factual error, and the paper does not misreport numbers. REMOVED as pure framing critique.

- **[Harsh Critic] Evaluation protocol not clearly matched to online TTA setting:** The paper describes single-sample online adaptation in Eq. 1 but implements mini-batch-based adaptation, and Figure 4 shows "retrospective" error curves. This is inconsistent notation but is standard practice in the TTA literature (Tent, SAR also use mini-batches). The TTAB framework standardizes this across all methods. REMOVED as a standard-practice concern.

- **[Spark / Human Finder] Missing recent TTA baselines (CoTTA, EcoTTA, ROTI, ViDA, SENTRY, MEMO):** Per the hard rule against mentioning missing related works that cannot be independently verified, these are excluded.

- **[Human Finder] Reproducibility concerns about hyperparameter disclosure:** The paper notes grid search follows TTAB guidelines and method-specific hyperparameters follow original setups. Criticizing the search space disclosure is a nitpick about trivial implementation detail. REMOVED per hard rules.

- **[Neutral Reviewer] General complexity concern:** The "many interacting components make it hard to reproduce" critique is a generic one-size-fits-all concern not specific to this paper's failure mode. REMOVED per soft rules (generic).

---

## Novel Insights

The most genuinely novel observation across reviewers is the connection between Voronoi Diagrams and neighbor-based TTA: formalizing T3A-style prototype assignment as a VD cell assignment is non-trivial and opens a principled path to importing computational geometry's richer structures (CIVD, PD) into TTA. The ECE improvements are notably larger and more consistent than accuracy improvements, suggesting that the geometric alignment enforced by distance-to-prototype objectives may primarily improve calibration rather than raw accuracy — a distinction worth exploring explicitly, as it reframes the paper's core contribution from accuracy-focused to calibration-focused adaptation.

---

## Suggestions

1. **Reframe the problem setup to honestly include source-data access.** Declare upfront that TTVD requires precomputed source class means, add at least one experiment with class means derived from test data (e.g., online-updated prototypes) or a SFDA baseline that also uses source statistics, and discuss when source access is realistic.
2. **Extend the ablation to CIFAR-100-C and ImageNet-C.** At minimum, show VD vs. CIPD (the two extremes) on all four datasets.
3. **Move Algorithm 3 (CIPD) to the main paper.** It is the core method.
4. **Add a sensitivity plot for the exponent in Eq. 4 and the temperature τ.**
5. **Explicitly discuss the ECE vs. accuracy trade-off**: the paper's strongest result is calibration, and framing TTVD as a calibration-centric TTA method may be more honest and impactful than emphasizing small accuracy margins.

---

## Score and Decision

**Calibration against retrieved papers:**

| Paper | Decision | Scores | Comparable features |
|---|---|---|---|
| PROGRAM | Accept (Poster) | 5,6,6,6,8 (avg ~6.2) | Prototype-based TTA, similar benchmarks, clear novelty, decent margins |
| NGTTA | Reject | 6,5,6,5 (avg ~5.5) | Geometry-driven TTA, thin mechanistic analysis, single domain |
| Continual TTA | Reject | 3,5,5,6,5 (avg ~4.8) | Prototype-based TTA, marginal gains (+0.5%), limited novelty |

TTVD sits between PROGRAM and NGTTA. It has more novelty than the rejected Continual TTA paper and the VD geometric framing is richer than NGTTA's heuristic use of geometry. However, the source-data information asymmetry is a real fairness concern not present in PROGRAM, and the mechanistic analysis is thinner. ECE gains are genuine and notable; accuracy gains on CIFAR benchmarks are marginal without variance reporting.

**Assessment:**
- **Originality:** Good — the VD/CIVD/PD framing for TTA is novel and principled.
- **Importance of research question:** High — TTA is a practically important problem with active community interest.
- **Claims vs. support:** Moderate — ECE claims are well-supported; accuracy claims are marginal; mechanistic claims are poorly supported.
- **Soundness of experiments:** Moderate — TTAB toolkit is a strength, but source-data asymmetry and single-run results without error bars are weaknesses.
- **Clarity:** Fair — progression of structures is clear, but core algorithm is buried in the appendix and source-data assumption is obscured.
- **Value to community:** Moderate — the geometric interpretation is intellectually interesting and calibration improvements are practically useful.

The paper has real contributions (novel geometric framing, strong ECE improvements, standardized evaluation) but is undercut by a significant information asymmetry in comparison (source training data for class means vs. purely source-free baselines), weak mechanistic evidence, and marginal accuracy gains without uncertainty quantification. It is a borderline paper that would benefit substantially from revisions.

**Final Score: 5.0** — Marginally below acceptance threshold. The geometric novelty and calibration contributions are genuine, but the fairness of comparison requires clarification and the mechanistic claims need stronger empirical support.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>