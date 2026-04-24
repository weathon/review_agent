## Summary

This paper proposes CAPGen, a framework for generating adversarial patches using a constrained color palette derived from the environment. Its central contribution is twofold: (i) a differentiable color-probability matrix that maps each patch pixel to one of a few base colors, and (ii) an empirical claim that adversarial effectiveness is driven primarily by patch patterns rather than colors, enabling rapid recoloring for new environments. While the technical mechanism is clean, the evaluation suffers from severe disconnects between the proposed method and the experiments actually conducted.

## Strengths

- **Clean differentiable mechanism for palette-constrained patches.** The color probability matrix with low-temperature softmax (Eq. 3, Sec 3.2) provides a concrete, gradient-friendly way to restrict patch pixels to a small set of base colors. This is a useful technical building block.
- **Cross-architecture evaluation breadth.** The white-box and black-box experiments span six detectors (YOLOv2/3/4/5s/5m and Faster R-CNN, Tables 1–2), which is more extensive than many patch-attack papers.
- **Practical motivation.** The direction of environment-adaptive, visually constrained adversarial patches is well motivated for physical-world attack scenarios.

## Weaknesses

### Fatal
None.

### Major

- **The core pattern-versus-color analysis is methodologically confounded and cannot support the central claim.** The paper asserts that “patterns significantly influence attack performance than colors” (Sec 3.3) based on comparing CAPGen-P1 (recoloring AdvPatch with base colors Bc1/Bc2) against CAPGen-T1/T2 (training a color probability matrix from scratch with fixed colors). This comparison is confounded: AdvPatch was optimized with full RGB freedom and no color-cardinality constraint, whereas CAPGen-T1 must learn a pattern from scratch under the severe constraint of only three base colors. The large performance gap (Table 1: 22.92 vs. 48.04 avg. mAP$_{50}$) may reflect optimization difficulty under constraints rather than an intrinsic dominance of pattern over color. Moreover, AdvPatch does not use the paper’s color-probability-matrix formulation, so how its “pattern” is extracted and preserved during recoloring is entirely unspecified (Sec 3.3 says “we only replace the base colors without altering the color probability matrix,” yet AdvPatch has no such matrix). Without a controlled ablation that trains both variants under identical constraints, the evidence does not validly support the claim.

- **The proposed environment-adaptive mechanism is described but abandoned in the main quantitative experiments.** Section 3.2 motivates K-means clustering to extract base colors “from the surrounding environment,” and Figure 3 shows color extraction from training images. However, Section 4.2 explicitly uses two arbitrarily chosen random color sets (Bc1 and Bc2) that are **not extracted from any environment** for the quantitative results in Tables 1–2. Because the reported numbers rely on random palettes rather than environmentally extracted colors, the main experiments do not test the actual adaptive pipeline proposed in the method. The quantitative claim that CAPGen “adapts to new environments” or achieves “environmental harmony” is therefore unsupported.

- **Visual stealthiness, the primary motivation, is never quantitatively measured.** The abstract promises patches that “seamlessly blend with their background for superior visual stealthiness,” yet no human perceptual study, detection-rate metric, or perceptual similarity score (e.g., LPIPS, SSIM, saliency) is reported. The only evidence is a single qualitative physical-world image (Figure 1) and unsupported assertions. Without measurement of the property the method is designed to optimize, the core claim remains unsubstantiated.

- **Physical-world applicability is claimed but lacks quantitative validation.** The introduction and abstract frame CAPGen as a physical attack method, and Figure 1 shows a physical coat deployment. However, every quantitative result is digital (INRIA dataset). No physical-world detection rates, attack success rates under varying distances/angles/lighting, or robustness metrics are reported. The practical applicability claims are therefore speculative.

### Minor

- **Results reporting is inconsistent and occasionally misleading.** In Section 4.3, the authors state: “Even AdvPatch is only 3.34 points lower than CAPGen-P1, further illustrating our approach’s advantage.” Since lower mAP$_{50}$ indicates better attack performance, AdvPatch (19.58) actually outperforms CAPGen-P1 (22.92). Framing the baseline’s superior performance as an advantage for the proposed method is a misinterpretation. Additionally, the introduction claims that with YOLOv4 as substitute, CAPGen-P1 “surpasses the mainstream algorithm by about 1.7%” in black-box transfer; the actual margin in Table 2 is 0.65 mAP$_{50}$ points (37.99 vs. 38.64), and the origin of the 1.7% figure is unexplained.

- **Selective emphasis in black-box narrative.** The paper highlights the YOLOv4 substitute-model result (CAPGen-P1: 37.99 vs. AdvPatch: 38.64) to claim preserved transferability. However, across the six substitute models in Table 2, AdvPatch achieves lower (better) average mAP$_{50}$ than CAPGen-P1 in four out of six settings. The narrative focuses on the single setting where CAPGen-P1 is marginally better, without acknowledging the mixed overall picture.

- **Figure 5 (right) makes an unsupported trend claim.** The ablation on the number of base colors uses only three data points (7, 8, 9 colors) showing a non-monotonic relationship (~52 → ~25 → ~35). The caption nevertheless claims “the overall trend … tends to improve as the number of colors increases,” which the plotted data do not clearly support.

### Trivial

- **Eq. (2) introduces a stealth term $S(P; \epsilon)$ and stealth parameters $\epsilon$ that are never instantiated or optimized in the actual method.** The real stealth mechanism is simply fixing colors to base colors, which is not equivalent to the formulated $S(P; \epsilon)$.

- **Section 3.2 describes “regularizing the color probability matrix” to enforce one-hot pixel assignments, but no explicit regularization term appears in the loss; only a low-temperature softmax ($\tau = 0.1$) is used.** The description is slightly imprecise but the operational effect is clear.

## Nice-to-Have

- A controlled ablation in which both pattern and color variants are trained *from scratch* under the same color-cardinality constraint (e.g., train an unconstrained 3-color patch and compare it against a recolored version) would cleanly isolate the effect of pattern versus color.
- Quantitative perceptual metrics or a small human study to validate that K-means-extracted colors actually improve concealment relative to random or AdvPatch colors.
- Small-scale physical-world robustness evaluation (varying distance, angle, lighting) to support the physical attack framing.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **“AdvPatch outperforms CAPGen-P1 in 5 out of 6 black-box transfer settings.”** This is factually incorrect. Table 2 shows AdvPatch achieves better (lower) average mAP$_{50}$ than CAPGen-P1 in 4 out of 6 substitute-model settings, not 5. The broader concern about selective reporting is valid, but the specific count is wrong.
- **Missing appendix, proofs, or references.** These sections were stripped by the parser and may exist in the original submission.
- **Formatting/style nitpicks and typos.** These are parser artifacts, not author errors.

## Novel Insights

None beyond the paper's own contributions. The color probability matrix is a sensible mechanism, and the ambition to disentangle pattern from color in physical adversarial patches is a worthwhile direction that has been understudied.

## Suggestions

1. Redesign the pattern-vs-color experiment so that both variants start from the same optimization conditions (e.g., train a CAPGen patch with 3 colors from scratch, then recolor it while freezing the probability matrix, and compare against re-optimizing the matrix with the same fixed colors).
2. Replace the arbitrary Bc1/Bc2 colors in the main tables with K-means-extracted palettes from distinct environment image sets, and report both attack performance and concealment metrics.
3. Correct the misstatements in Sections 4.3 and 4.4 regarding relative performance and percentage margins.

## Score and Decision

**Calibration comparison:**
- **High anchor (6.75):** *Embodied Active Defense* — strong theoretical grounding, extensive 3D real-world evaluation, and attack-agnostic experiments. CAPGen is well below this: it lacks real-world quantitative validation and its core experiment is confounded.
- **Medium anchor (5.25):** *Efficient Diversified Attack* — plentiful experiments with marginal improvements and some missing evaluations. CAPGen is below this because its central experimental design does not validly test its main claim, and its proposed adaptive mechanism is untested in the quantitative results.
- **Low anchor (3.50):** *Human-Producible Adversarial Examples* — had actual physical user studies but was rejected for insufficient baselines and motivation. CAPGen has a more focused technical idea but similarly fails to quantitatively validate its core claims (stealth, environment adaptation, physical robustness). Its mechanism is cleaner, but the evaluation gaps are comparably severe.

This paper identifies a sensible problem and proposes a technically concrete mechanism, but the disconnect between its claims and evidence is too large. The confounded pattern-vs-color comparison, the abandonment of the K-means adaptive mechanism in the main experiments, the complete absence of stealth quantification, and the lack of physical-world numbers collectively mean the central contributions are not empirically supported. The paper needs substantial revision and re-evaluation before it can be considered for acceptance.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>