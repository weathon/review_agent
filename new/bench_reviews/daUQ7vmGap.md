Now I have a thorough understanding of the paper and calibration anchors. Let me write the final review.

## Summary

This paper proposes the Dynamic Sparsity Corruption Robustness (DSCR) Hypothesis: that Dynamic Sparse Training (DST) at low sparsity levels helps improve model robustness against image corruption compared to Dense Training. The authors validate this claim across 9 scenarios spanning multiple DST algorithms (SET, RigL, MEST, GraNet), architectures (VGG, ResNet, EfficientNet, DeiT, 3D ConvNets), datasets (CIFAR10/100-C, TinyImageNet-C, ImageNet-C/C̄, ImageNet-3DCC, UCF101), and provide spatial and spectral domain analyses attributing the robustness gain to an implicit regularization that reduces attention to high-frequency information.

## Strengths

- **Genuinely surprising and important empirical finding**: The observation that DST — typically used for efficiency — can also improve corruption robustness over dense training is counterintuitive and valuable. This opens a new research direction connecting sparse training and robustness.

- **Extensive experimental evaluation across 9 diverse scenarios**: The paper tests 4+ DST algorithms, 5+ architectures, 7 corruption benchmarks, video data, and transformer-based models (Table 1, Figures 2–4). This breadth provides strong evidence that the phenomenon is real and not an artifact of a specific setting.

- **Insightful corruption-type-specific analysis (Section 4.2, Figure 3)**: The finding that DST's advantage is largest for high-frequency corruptions (noise types) and at higher severity levels (e.g., ~25% relative improvement for MEST_g at severity 5 for impulse/Gaussian noise on ImageNet-C) is specific, informative, and directly connects to the proposed spectral explanation.

- **Clean spectral analysis framework (Section 5.2, Equations 1–2, Figure 7)**: The Radius-Accuracy curves for high-frequency and low-frequency attenuation provide a well-designed experimental probe. The two observations — equal sensitivity to low-frequency removal but reduced sensitivity to high-frequency removal for DST models — are clearly articulated and supported by the data.

- **Unified experimental framework (Section 3.2)**: Testing both random regrow and gradient-based regrow across all DST methods (SET, MEST_r, GraNet_r, RigL, MEST_g, GraNet_g) enables controlled comparisons and ensures findings are not artifacts of a particular regrowth strategy.

## Weaknesses

### Fatal
None.

### Major

- **Missing critical baselines that undermine attribution to DST specifically**: The paper compares DST at sparsity *s* against the *same architecture trained densely*, but never tests (a) a **smaller dense model** with equivalent parameter count/FLOPs (e.g., a ResNet34 with half the channel width at sparsity 0.5), (b) **static sparse training** at the same sparsity (no topology updates), or (c) **dense training with comparable regularization** (dropout, stronger weight decay). If a smaller dense model or a static sparse model achieves similar robustness, then the robustness gain is attributable to reduced capacity or simple regularization — not to *dynamic* sparsity per se. The DSCR Hypothesis specifically names "Dynamic" and "Sparse" as the operative factors, but the experiments cannot disentangle these from the confound of parameter count. The paper itself acknowledges (line 36) that DST "utilizes hard sparsity—on top of the soft sparsity introduced by weight decay or other regularization methods—which provides additional regularization," yet never tests whether simpler regularization achieves comparable gains. This gap directly weakens the headline claim.

- **No variance reporting or statistical significance testing**: The paper reports no error bars, confidence intervals, or results from multiple random seeds. Many of the reported improvements are small: 0.32% on ImageNet-C/ResNet50, 0.46% on ImageNet-3DCC, 0.4% on CIFAR10-C/VGG16 (Table 2). A 0.3% difference without variance information is indistinguishable from noise. Even the larger improvements (4% on CIFAR100-C) lack any reproducibility measure. For a paper whose core claim is "consistent outperformance," this is a fundamental evidential gap.

### Minor

- **"Consistently outperform" claim is overstated relative to the evidence**: The abstract states DST "can consistently outperform Dense Training," and Table 2 presents a clean 9/0 sweep. However, Figure 2 shows this is conditional on sparsity ratio — for CIFAR10-C/VGG16 (Figure 2, top-left), many DST methods at many sparsity ratios roughly match or only marginally exceed dense performance. Table 2 selects the single best sparsity ratio per scenario, which is a form of selection. The claim should acknowledge that outperformance is conditional on sparsity ratio and dataset.

- **Spectral analysis establishes correlation, not causation**: Section 5.2 shows DST models are less degraded by high-frequency attenuation, and Section 5.1 shows structured sparsity patterns. The paper concludes DST "introduces a form of implicit regularization, reducing the focus on high-frequency information." However, no causal experiment is conducted — e.g., artificially reducing high-frequency sensitivity in a dense model to test whether it reproduces the robustness gain, or forcing a DST model to retain high-frequency sensitivity to test whether it eliminates the gain. The alternative explanation — that lower-capacity models naturally cannot exploit high-frequency information as effectively — is not ruled out. The "implicit regularization" explanation remains plausible but unverified.

- **Abstract claim about resource costs is misleading for gradient-based methods**: The abstract states DST achieves robustness gains "without adding (or even reducing) resource cost." However, gradient-based methods like RigL require full-gradient computation for regrowth steps (acknowledged in the Figure 1 caption but contradicted by the abstract). The actual FLOP savings depend heavily on implementation and hardware, which the paper acknowledges in footnote 4 but does not quantify in the abstract.

### Trivial
None.

## Nice-to-Haves

- Training smaller dense models at equivalent parameter budgets to isolate whether reduced capacity alone explains the robustness gain.
- Comparing against static sparse training at the same sparsity to isolate whether the *dynamic* aspect matters.
- Testing dense training with dropout or stronger weight decay to assess whether DST provides benefits beyond standard regularization.
- Causal experiments: e.g., applying low-pass filtering during dense training to test whether it reproduces DST's robustness advantage.
- Reporting mean ± std over multiple seeds, especially for the ImageNet-C/3DCC scenarios where improvements are < 1%.
- Systematic sparsity ratio sweep with finer granularity to define the boundary where the DSCR benefit disappears.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Unified framework means tested versions are not identical to original algorithms"**: The paper acknowledges this explicitly in Section 3.2 and argues it enables controlled comparison. This is a deliberate methodological choice, not a flaw.

- **"40% FLOP savings claim is inaccurate for gradient-based methods"**: The paper does acknowledge this in footnote 4 and the Figure 1 caption. The claim about computational savings is qualified — this is not a hidden flaw but an acknowledged nuance.

- **"Qualitative visualization in Figure 5 does not establish mechanism"**: This is already a minor point acknowledged by the paper's own hedged language ("we suggest that dynamic sparsity introduces a form of implicit regularization"). The spectral analysis (Section 5.2) provides the quantitative support.

- **Strength removed: "Robustness gains achieved with computational savings"** — This strength conflicts with the verified weakness that the resource cost claim is misleading for gradient-based methods. The savings are conditional and implementation-dependent, so presenting this as a clean benefit overstates the case.

## Novel Insights

The spectral analysis reveals an asymmetric frequency sensitivity profile: DST models maintain equal sensitivity to low-frequency information but are less sensitive to high-frequency information compared to dense models. This asymmetric profile directly explains why DST excels specifically against noise-type corruptions (high-frequency) while showing smaller advantages for blur/weather-type corruptions (low-frequency). This frequency-based explanation is more specific and testable than a generic "regularization" account, and it predicts that any method that selectively reduces high-frequency sensitivity should show similar corruption-type-specific robustness patterns — a prediction that could be tested in future work.

## Suggestions

- Add at minimum a static sparse baseline (same architecture, same sparsity, fixed topology) and a smaller dense model baseline (e.g., reduced-width ResNet) to isolate the contribution of dynamic sparsity from reduced capacity. Even one scenario with these baselines would substantially strengthen the claims.
- Report results over 3+ random seeds with standard deviations for all scenarios in Table 2, or at minimum for the ImageNet-C and ImageNet-3DCC scenarios where improvements are < 1%.
- Refine the abstract claim from "consistently outperform" to "can outperform at specific sparsity ratios" to match the actual evidence in Figure 2.

## Evaluation

**Originality**: The paper identifies a genuinely novel and counterintuitive phenomenon — DST improving robustness — that had not been systematically studied. The spectral analysis framework is a useful methodological contribution. However, the attribution to "dynamic sparsity" specifically is not well-isolated from confounds.

**Importance of research question**: The question of whether to train densely or sparsely for robustness is practically important and timely, given the growing interest in both efficient training and robust ML.

**Claims support**: The core empirical observation (DST often improves robustness over dense training of the same architecture) is well-supported by the breadth of experiments. However, the stronger claims — that this is due to *dynamic* sparsity specifically, that it constitutes *implicit regularization* beyond simple capacity reduction, and that outperformance is *consistent* — are not well-supported by the evidence presented.

**Experimental soundness**: Extensive in breadth but lacking in critical controls (equal-resource baselines, variance reporting, static sparse comparison).

**Clarity**: The paper is generally well-written and clearly structured. The spectral analysis framework is cleanly presented.

**Community value**: The finding opens a new direction connecting sparse training and robustness, and the spectral analysis provides concrete hypotheses for future work.

## Calibration

Compared against the following anchors:

- **Low band (<3)**: 64vO8qoJfb (empirical robustness study, narrow scope, overclaimed, avg 3) — Our paper is significantly stronger with much broader experimental evaluation and deeper analysis.
- **Medium band (4–6)**: 0ydseYDKRi (overclaimed "SOTA," no error bars, missing baselines, avg 5.5, Reject) — Similar pattern of overclaiming and missing baselines, but our paper has more comprehensive experiments and a more surprising finding. qbw861vueP (DST bi-level optimization, avg 4.33, Reject) — Topically related but weaker contribution.
- **High band (>7)**: wJv4AIt4sK (sparsity-quantization interplay with theoretical proofs, avg 7.5, Spotlight) — Our paper is weaker due to lack of theoretical backing and methodological gaps. hJ1BaJ5ELp (probabilistic pruning with theoretical guarantees, avg 7.5, Spotlight) — Again, much stronger theoretical contribution.

Our paper sits in the medium band. It is stronger than the typical reject (interesting finding, extensive evaluation) but has significant methodological weaknesses that prevent a higher score. The missing baselines are the most damaging: without them, the paper cannot support its specific claims about dynamic sparsity being the operative factor. At the same time, the empirical phenomenon is real and worth reporting.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>