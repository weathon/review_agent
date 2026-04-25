Now let me search for calibration anchors.Now I have sufficient information to write the final review. Let me synthesize everything.

---

## Summary

SuMi proposes a multimodal wild Test-Time Adaptation (TTA) framework that addresses the practically important setting where the test stream contains a mixture of weak OOD samples (single modality corrupted) and strong OOD samples (multiple modalities corrupted or missing). The method combines three components: (1) IQR-based smoothing that gradually expands the sample pool over iterations, (2) Unimodal Assistance (UA) that filters for low multimodal entropy but high unimodal entropy, and (3) a Mutual Information Sharing (MIS) loss aligning unimodal and multimodal predictions via KL divergence. Experiments on Kinetics50-C and VGGSound-C show clear gains over prior TTA baselines on strong OOD scenarios.

---

## Strengths

- **Novel, well-motivated problem formulation (§1, Figure 1(b,c,d))**: The multimodal wild TTA setting is genuinely underexplored. Figure 1 clearly demonstrates that all existing TTA baselines—including the multimodal-specific READ—fail catastrophically on strong OOD (missing or multi-corrupted modalities), collapsing to ~10–16% accuracy, which is a compelling motivation for the new challenge.

- **Strong empirical results on the hardest scenarios (Table 2, Table 4, Figure 5)**: SuMi achieves 33.4% average strong-OOD accuracy on Kinetics50-C vs. next-best READ at 29.1%, and most other baselines at 11–16%. The robustness under increasing strong-OOD ratio (Figure 5), where baselines degrade rapidly while SuMi holds its performance, is the paper's most convincing evidence.

- **Unimodal assistance insight is empirically grounded (Figure 3(c), Table 6)**: The observation that the [20,40] unimodal entropy quantile outperforms the [0,20] quantile (Figure 3(c)) is supported by Table 6, where Area 1 samples (low multimodal entropy, rich multimodal information) achieve 39.4% vs. Area 3 (low multimodal entropy, little multimodal info) at 32.1%. This is a concrete and novel insight.

- **Comprehensive benchmarks**: The construction of Kinetics50-C and VGGSound-C with 15 video and 6 audio corruption types at five severity levels, plus four strong OOD scenarios, provides a thorough and reusable evaluation framework.

---

## Weaknesses

### Fatal

None.

### Major

- **Table 5 contains two indistinguishable full-model rows that give substantially different results, and the paper's textual interpretation of the ablation is factually wrong.** Table 5 has two rows both labelled "IQR ✓, UA ✓, MIS ✓" that report (54.3, 44.6, 51.3) and (59.3, 52.0, 59.1) on Kinetics50-C—a difference of ~7.4 pp at severity 5—with no explanation of what differs between them. More critically, the lower full-model row (54.3 at severity 3) is even *worse* than UA alone (52.1) or IQR+MIS (58.0), which is inexplicable if all three components are genuinely complementary. The paper offers no clarification about what these two rows represent (different MIS stopping time? different β?). This makes the ablation uninterpretable as the primary evidence for each component's contribution.

  Beyond the duplication, §4.3 claims: *"IQR smoothing brings the most improvements to the model."* This is directly contradicted by Table 5: UA alone achieves 45.1% vs. IQR alone at 31.7% and MIS alone at 39.4% at severity 5. By any reading—whether comparing single components in isolation or computing marginal gains—UA is the strongest single component and IQR is the weakest. The text's central interpretive claim about the ablation is factually wrong.

- **IQR+UA combination actively hurts relative to UA alone (38.1% < 45.1% at severity 5), and this negative interaction is never acknowledged or explained.** The two mechanisms are presented as complementary (IQR for smoothing, UA for quality filtering), but their combination degrades performance by ~7 pp on Kinetics50-C severity 5. Understanding why these components conflict would be essential to validating the architecture—instead, the paper ignores this.

### Minor

- **IQR applied to high-dimensional feature vectors lacks statistical justification.** Algorithm 1 (lines 5–6) computes Q1 = quantile(**h**, 0.25) and Q3 = quantile(**h**, 0.75) where **h** ∈ ℝ^d is a concatenated representation vector. Tukey's fence is a univariate rule for detecting outliers *across a scalar sample*; applying it across feature dimensions of a single sample representation is an unconventional extension. The paper's claim that this naturally selects "weak OOD" samples early is supported only by a qualitative t-SNE visualization (Figure 3(b)), not by direct measurement of which samples (weak vs. strong OOD) are retained at each iteration. Whether the apparent curriculum effect is caused by IQR or by the entropy threshold in Equation 4 is not disentangled.

- **β hyperparameter differs substantially across datasets (0.6 vs. 0.9) with no ablation.** The smoothing coefficient governing sample admission is set to 0.6 for Kinetics50-C and 0.9 for VGGSound-C (§4.1), a large gap suggesting dataset-specific tuning. No sensitivity analysis over β is presented, making it unclear how robust the method is to this choice.

- **μ hyperparameter shows divergent behavior across datasets.** Figure 7(a) shows performance increasing with μ on Kinetics50-C but decreasing on VGGSound-C. The authors attribute this to modality dominance, but selecting the optimal μ requires prior knowledge of which modality dominates—this should be discussed more honestly as a limitation of the method's claimed generality.

- **Unexplained catastrophic failure of several baselines on VGGSound-C audio corruptions (Table 4).** SAR achieves 3.6%, SoTTA 7.7%, DeYO 4.2% on audio corruption (well below the source model), while EATA achieves 32.5%. These failures warrant explanation—whether this is a hyperparameter issue or a known failure mode of these methods, the paper should address it rather than simply reporting the numbers.

### Trivial

- The "mutual information sharing" name is misleading—the loss actually minimizes KL divergence between unimodal and multimodal predictions, which is cross-modal prediction alignment, not mutual information optimization in the information-theoretic sense. This creates a false impression of stronger theoretical grounding.

---

## Nice-to-Haves

- A "multimodal-aware Tent or EATA" baseline (e.g., filtering by unimodal entropy without IQR or MIS) would clarify whether the gains come from SuMi's specific design choices or simply from incorporating any multimodal signal.
- A direct per-iteration measurement of what fraction of admitted samples are weak vs. strong OOD would validate or refute the smoothing narrative (rather than relying on t-SNE visualization).
- Ablation over β (0.6–0.9) and t₀ (MIS stopping iteration) to demonstrate stability.
- Evaluation on a three-modality dataset to validate the generalization claim for M > 2 modalities (the paper presents M-modality formulation in Equation 5 but only tests M=2).

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Comparison against unimodal TTA baselines is structurally unfair (Harsh Critic)**: Removed. The baselines are indeed designed for unimodal models but are legitimately applied here to assess whether they generalize. The comparison structure (all methods on the same multimodal model) is standard. The one multimodal-specific baseline (READ) is the most relevant, and the paper properly highlights gains over it.

- **Figure 1(b-d) uses approximate values and collapses baselines**: Removed as a pure presentation nitpick. The approximate values are illustrative, and exact numbers are given in Tables 1–4.

- **Learning rate difference (1e-4 vs. 1e-5) suggests unfair comparison**: Removed. The paper states different learning rates for different datasets, which is standard practice. There is no evidence baselines were not also tuned per dataset.

- **Termination of MIS loss (t₀ = iter/2) gets no ablation**: Partially valid concern, but moved to Nice-to-Haves as it's not central enough to constitute a Major weakness on its own.

- **Strength Finder claim that IQR smoothing is validated by t-SNE (Figure 3b)**: Downgraded. The t-SNE is only qualitative and does not directly measure which OOD type is selected at each iteration. Listed as a concern in Minor weaknesses instead.

---

## Novel Insights

The paper's most genuinely novel observation is that unimodal entropy has a *non-monotone* relationship with multimodal adaptation quality: very low unimodal entropy is actually *harmful* because it signals the sample doesn't require cross-modal integration, removing informative multimodal signal. Samples in the moderate [20,40] unimodal entropy quantile outperform those in the [0,20] range (Figure 3(c), Table 6). This insight—that the most "confident" unimodal predictions are the *least useful* for multimodal adaptation—could be broadly applicable to multimodal learning research beyond TTA.

---

## Suggestions

1. Clearly distinguish and label the two full-model rows in Table 5, explaining what differs between them (e.g., whether MIS is applied for all iterations vs. only iter/2). If one row represents the default SuMi and the other a variant, say so explicitly.
2. Correct §4.3: the claim that "IQR smoothing brings the most improvements" is inconsistent with Table 5 and should accurately reflect that UA or MIS contributes more.
3. Investigate and explain the IQR+UA negative interaction—this is the most important missing piece of analysis. 
4. Provide a sensitivity analysis for β, since it varies substantially between datasets.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison |
|------|----------------|------------|
| TPZRq4FALB (READ) | 8.0 | Direct precursor to SuMi; cleaner methodology, consistent ablation, excellent presentation. SuMi clearly below this bar. |
| 9w3iw8wDuE (DeYO) | 7.0 | TTA with novel confidence metric, strong theory+experiments. SuMi's contributions are more incremental. |
| sEMJ1PLSZR (AEA) | 6.25 | TTA paper accepted as poster; solid but mixed scores. Comparable scope to SuMi but cleaner analysis. |
| UhKkWHkvfg (MDAA) | 5.0 | Multimodal Continual TTA, rejected. Similar setting, similar motivation-as-combination-of-ideas concern. SuMi has stronger empirical gains but worse ablation issues. |
| nc0XGK40dn (IDKR) | 4.67 | Continual TTA, rejected for overlapping motivation with existing methods. Comparable quality tier. |
| ws0F5NTzGw (AdapTable) | 4.5 | Tabular TTA, rejected. Low technical depth. SuMi is clearly stronger. |
| Wure6HljpJ (CoSDA) | 3.67 | Source-free domain adaptation, rejected. Lower technical quality. SuMi is clearly above this. |

SuMi's empirical results on the strong OOD problem are genuinely useful, and the problem formulation is a meaningful extension of READ's work. However, the ablation study has a fundamental presentation failure (two unexplained identical rows with divergent results), the primary textual claim about component contributions is factually wrong per the authors' own data, and a key claimed synergy (IQR+UA) actually degrades performance relative to UA alone with no explanation. These are substantive analytical failures, not mere presentation issues.

Positioning relative to anchors: READ (8.0) is the gold standard for multimodal TTA papers at this venue—clean methodology, clear ablations, strong contributions. SuMi falls well short of READ on analytical rigor. MDAA (5.0, rejected) is the closest comparator: both are multimodal TTA extensions with real empirical gains but methodological concerns. SuMi has larger empirical gains but more serious ablation problems. I place SuMi slightly below MDAA, at 4.5.

**Originality**: Moderate — problem is a meaningful extension of READ, but IQR mechanism is heuristic without clear justification.  
**Importance**: Moderate-high — multimodal wild TTA is a real problem.  
**Claim support**: Weak — main textual claim about ablation is factually wrong per Table 5.  
**Experimental soundness**: Mixed — results are broad but ablation is unreliable.  
**Writing clarity**: Adequate, with the notable exception of the ablation discussion.  
**Community value**: Moderate — benchmarks are useful; MIS loss concept is clean.

**Decision: Reject.** The paper addresses a genuine and practically important problem, but the ablation study is both internally inconsistent (two indistinguishable full-model rows with very different results) and textually misinterpreted. The central claimed advantage of IQR smoothing is contradicted by the paper's own data. These issues would need to be resolved before acceptance.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>