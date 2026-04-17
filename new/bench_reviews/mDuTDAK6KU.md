Now I have sufficient calibration data. Let me write the final review.

## Summary

The paper proposes KOALA, an adversarial detection method that flags inputs as attacked when predictions from two complementary similarity metrics—KL divergence and an L0-based distance—disagree on the class label of the input embedding relative to class prototypes. The motivation is that energy-bounded adversarial perturbations must manifest as either dense (detected by KL) or sparse (detected by L0) shifts, making it difficult for a single perturbation to fool both metrics simultaneously. The authors provide a conditional theorem (Theorem 1) asserting guaranteed detection under assumptions about normalized embeddings, bounded perturbations in feature space, coordinate-wise bounds, and clean alignment, and experiment on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet.

## Strengths

- **Novel and well-motivated detection principle.** The geometric intuition that energy-bounded perturbations must manifest as either dense or sparse shifts, and that KL and L0 are naturally complementary in capturing these respectively, is elegant and clearly communicated (Figure 1 provides a helpful illustration). This is a genuine conceptual contribution to the adversarial detection literature.

- **Theoretical grounding, even if conditional.** Unlike purely empirical detectors, KOALA provides a formal theorem (Theorem 1) establishing conditions under which detection is guaranteed. The proof sketch provides meaningful intuition about the incompatibility between the conditions needed to fool KL vs. L0 metrics. This is a step above purely heuristic detection approaches.

- **Lightweight, semantics-free design.** The method requires only clean-image fine-tuning with a composite loss and no adversarial training or architectural changes. It operates purely on representation geometry, making it architecture-agnostic and potentially applicable across modalities.

- **Honest analysis of CLIP results.** The paper openly acknowledges that the L0+KL+Cosine combination achieves high detection on CLIP by "breaking the underlying classification" rather than genuine robustness (Section 4.3). This level of self-critical analysis is welcome.

- **ResNet/CIFAR-10 improvements are non-trivial.** Table 3 shows that KL+L0 fine-tuning achieves adversarial accuracy of 54-57% under PGD/CW/AutoAttack at ε=4/255, compared to ~31-36% for the baseline, using only clean data. This is a substantive improvement.

## Weaknesses

### Major:

- **No evaluation against adaptive attacks that target the detector.** This is the most significant empirical gap. All attacks (PGD, CW, AutoAttack) are optimized against the underlying classifier, not against KOALA's disagreement mechanism. An adaptive attacker with white-box knowledge of both the KL and L0 heads could incorporate the disagreement criterion into their optimization objective, seeking perturbations where both metrics agree on the same wrong class. The detection rule (ŷ_KL ≠ ŷ_L0) is differentiable almost everywhere via the smoothed L0 surrogate, making gradient-based adaptive attacks feasible. The adversarial detection literature has repeatedly shown that detectors appearing robust against non-adaptive attacks can collapse under adaptive evaluation (Carlini & Wagner 2017; Athalye et al. 2018; Tramer et al. 2020). The paper repeatedly frames KOALA as a "robust, theoretically grounded defense for safety-critical applications," but this framing is unsupported without adaptive evaluation.

- **The "formal proof of correctness" claim is overstated relative to what is established.** The theorem is conditional on four assumptions that are either unverified or difficult to verify in practice: (a) A3 imposes a coordinate-wise bound |δ_i| ≤ 3/2|p*_i| on feature-space perturbations, which is strong and not standard; no empirical evidence confirms that real adversarial perturbations in feature space satisfy this. (b) A4 (clean alignment) is precisely what the fine-tuning is supposed to achieve, but no quantitative measurement of how well it holds is provided; the theorem's guarantee is circular if A4 is only approximately satisfied. (c) The threshold function Γ_i(ε) in Theorem 1 is not defined in the main paper, making it impossible for a practitioner to verify whether the conditions hold for a given model. (d) There is no connection established between the feature-space threat model (norm-bounded perturbations δ with ‖δ‖ ≤ ε) and the pixel-space ℓ∞ attacks used in experiments; the paper cites Lipschitz continuity of the backbone (A2) but provides no Lipschitz constant or argument. The abstract's claim that "detection is not a probabilistic outcome but a mathematical certainty" is misleading given these gaps.

- **Low theorem-compliance coverage undermines practical relevance of the guarantee.** On CLIP/Tiny-ImageNet, only ~10% of test samples (510-556 out of ~5000) satisfy Theorem 1's conditions. For the remaining ~90% of samples, the method has no provable guarantee, and precision drops to 0.62-0.66. Even on ResNet/CIFAR-10, compliance ranges from ~59% to ~67%. The paper does not analyze what makes samples non-compliant or whether compliance can be improved through better fine-tuning or alternative architectures.

- **Non-standard and conflated evaluation metrics.** The definitions of TP and FP conflate detection with classification: TP includes adversarial examples that the detector does not flag but the classifier correctly classifies anyway (â=0, ŷ=y*), and FP includes clean inputs that are simply misclassified without the detector triggering. This means "precision" and "recall" mix classification accuracy with detection accuracy, making it unclear how well KOALA performs as a *detector* specifically. Under these definitions, a perfect classifier with no detector at all would score 1.0 on all metrics, which reveals that these are not pure detection metrics.

- **No comparison with existing adversarial detection methods.** Despite citing feature squeezing, LID, Mahalanobis, MagNet, and other detection methods in related work, no empirical comparison is provided. It is impossible to assess whether KOALA offers a genuine improvement over prior detection approaches, which is essential for a paper positioning itself as an effective adversarial detector.

### Minor:

- **Limited attack diversity.** Only ℓ∞-bounded attacks at ε ∈ {2/255, 4/255} are tested. No ℓ2 or ℓ1 attacks, spatial transformations, or transfer-based black-box attacks are evaluated. Given that the L0 metric is designed to capture sparse perturbations, testing against sparse/ℓ0-bounded attacks would have been particularly informative.

- **No hyperparameter sensitivity analysis.** The L0 threshold τ=0.75, smoothness parameter φ=0.5, and loss weights ω_L0=0.9, ω_KL=0.1 are set without systematic study. Since the detection rule is driven entirely by disagreement between the two metrics, mis-tuning could easily cause systemic false alarms or collapse to trivial agreement.

- **The "plug-and-play" claim is slightly misleading.** The method requires fine-tuning the backbone encoder with a novel composite loss, which modifies the model's weights. This is not a zero-change addition to existing models—it requires retraining.

### Trivial:

- **Minor notation issue:** Paper uses card(·) for cardinality, which while clear, could be made more standard.

## Nice-to-Haves

- Compare against at least 2-3 established detection baselines (Mahalanobis, feature squeezing, LID) on the same benchmarks with standard detection metrics (AUROC, TPR at fixed FPR).
- Run adaptive attacks that jointly target both the KL and L0 heads, incorporating the disagreement criterion into the attack objective.
- Report standard detection metrics (not conflation metrics) and explicitly analyze false positive rates on clean data as a standalone number.
- Test on ℓ2 and ℓ0 perturbation norms to validate the claimed generality of the guarantee.
- Visualize the embedding space (e.g., t-SNE/UMAP) before and after fine-tuning to verify whether the composite loss actually creates the inter-class separation the theorem requires.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Plug-and-play claim is misleading" as raised by the neutral reviewer.** While technically true that "plug-and-play" is imprecise (the method requires backbone fine-tuning), the paper is transparent about this: "The only training required is a simple fine-tuning step on a pre-trained image encoder using clean images." The method does not require adversarial training, architectural changes, or knowledge of specific attacks. The claim is partially misleading but not severely so; I've kept a minor version of this point.

- **"Assumption A1 is restrictive"** — The paper explicitly implements A1 via softmax normalization on the embeddings, and it appears to work in practice. This is a design choice, not an unverified assumption. Kept a softened mention in the theorem discussion but not as a standalone weakness.

- **"CLIP results are underwhelming"** — The CLIP results show 0.66 precision and 0.85 recall. These are not stellar but are informative. The paper already discusses the CLIP limitations honestly. Not a separate weakness.

- **"No investigation of non-compliant majority"** — This is already covered by the "low theorem-compliance coverage" weakness; the non-compliant samples are precisely the ones the theorem doesn't cover.

- **"Softmax normalization changes the geometry"** — While true, the paper shows empirically that it works. This is a design choice with tradeoffs but not a fundamental flaw.

- **"False positive rate on clean non-compliant samples is unreported"** — This is partially captured by the non-standard metrics weakness. The paper does report overall accuracy on the full test set which implicitly includes false positives.

## Novel Insights

The core insight—that dense and sparse perturbations are fundamentally incompatible under an energy budget, and can be detected via disagreement between metrics sensitive to each type—is genuinely useful and provides a principled geometric motivation for multi-metric detection. However, the gap between the conditional theoretical guarantee and the empirical reality (especially on CLIP where only ~10% of samples are theorem-compliant) substantially limits the practical impact of the theoretical contribution. The observation that KL+L0+Cosine achieves high detection on CLIP by degrading classification rather than genuinely detecting attacks is a refreshingly honest finding that should serve as a cautionary note for the community about conflating detection rates with robust classification.

## Suggestions

1. **Run adaptive attacks immediately.** Craft attacks that minimize a combined loss: misclassify under both KL and L0 metrics while keeping them in agreement. This is the single most important empirical gap.

2. **Report standard detection metrics** (AUROC, TPR@FPR=1% or 5%) alongside the current conflated metrics, and report false positive rates on clean data separately. This enables comparison with prior detection work.

3. **Make Theorem 1 operational.** Define Γ_i(ε) in the main paper, measure prototype separation margins, and report what fraction of samples meet the theorem's exact conditions with the measured margins.

4. **Soften the "formal proof of correctness" and "mathematical certainty" language.** The theorem provides a conditional guarantee under assumptions that are not verified for the experimental models. More accurate phrasing would be "guaranteed detection under verifiable conditions on prototype separation and perturbation structure."

5. **Test against at least one established detection baseline** (e.g., Mahalanobis distance detection) to establish relative standing.

## Score and Decision

**Calibration:**
- Papers scoring 3-4 in adversarial defense/detection at similar venues: "Detecting Adversarial Examples" (KAWlH5pfQu, scores 1-5, rejected), "Neural Fingerprints" (eG56H9teXv, scores 3,3,3, rejected), "Multi-Task Consistency Detection" (adhxppqQAn, scores 3-6, rejected), "Statistical Attack-Agnostic Detection" (kz78RIVL7G, scores 1-3, rejected). These papers share the absence of adaptive evaluation and overclaimed theoretical guarantees.
- Papers scoring 5-6: "Adversarial Training Can Provably Improve Robustness" (inLUnCpDIB, scores 6,6,6,6,6, accepted poster) has genuine theoretical contributions with clear assumptions and empirical validation of the theory's predictions.
- KOALA has a genuine conceptual contribution (the KL/L0 disagreement principle) and shows empirical improvements on ResNet/CIFAR-10, but the theoretical claims overreach given the conditional and unverified assumptions, there is no adaptive evaluation, the evaluation metrics are non-standard, and there are no detection baselines for comparison. This places it below papers with verified theoretical guarantees and above papers with completely wrong proofs, but the gap between the "formal proof of correctness" claim and what is actually established is substantial.

The paper sits below papers with verified theoretical contributions (score ~6) and above truly broken papers (score ~1-2). The absence of adaptive evaluation and detection baselines, combined with the overclaimed theoretical guarantees and non-standard metrics, places it in the 3.5-4 range.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>