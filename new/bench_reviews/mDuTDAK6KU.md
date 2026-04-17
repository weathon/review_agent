Now I have sufficient context from calibration papers. Let me synthesize the final review.

## Summary

KOALA proposes an adversarial detector based on disagreement between two complementary metrics—KL divergence and an L0-based distance—in a nearest-prototype classification framework. When the two metrics predict different classes, the input is flagged as adversarial. The paper provides a formal theorem (Theorem 1) claiming guaranteed detection under specified conditions, and demonstrates the method on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet using only clean-image fine-tuning.

## Strengths

- **Novel and well-motivated detection principle.** The idea of exploiting disagreement between KL divergence (sensitive to dense, low-amplitude shifts) and L0-based distance (sensitive to sparse, high-impact changes) is creative and geometrically intuitive. Figure 1 effectively communicates this motivation.

- **Lightweight deployment.** The method requires only clean-image fine-tuning with a composite loss—no adversarial examples, no architectural changes, no domain-specific semantic priors. This makes it a genuine plug-and-play solution as claimed.

- **Empirical robustness gains on ResNet/CIFAR-10.** Table 3 shows that the KL+L0 fine-tuning objective achieves 57.32% adversarial accuracy under PGD ε=2/255, compared to 45.5% for the baseline—an 11.8 percentage point improvement from clean-only fine-tuning, which is nontrivial.

- **Honest analysis of failure modes.** The paper candidly discusses why the KL+L0+Cosine combination achieves high detection on CLIP for the wrong reasons (Section 4.3: "a high detection rate does not always equate to a truly robust model"), and acknowledges the low compliance rate for CLIP models.

- **Thorough ablation over metric combinations.** Tables 2-4 provide meaningful insight into which metric pairings are complementary and why adding Cosine similarity can hurt adversarial robustness due to optimization conflicts.

## Weaknesses

### Major:

- **The "formal proof of correctness" is advertised as the central contribution but cannot be operationalized as a practical guarantee.** The theorem (Theorem 1) states detection is guaranteed when |c*_i − ĉ_i| > Γ_i(ε) "for some threshold Γ_i(ε)," but Γ_i(ε) is never explicitly defined or computable in the main text. The proof is relegated to the appendix, leaving the main-text reader unable to verify or apply the condition. Crucially, the "theorem-compliant" vs "non-compliant" partition in Experiment 1 (Table 1) is determined post hoc using the same detector geometry, making the "recall = 1.0 on compliant subset" near-tautological: samples where margins are large under KOALA's own metrics are exactly where KOALA works well. The paper never provides a certified guarantee of the form "for X% of inputs, all ℓ∞-ε attacks are detected," which is what the abstract and introduction promise. This disconnect between the marketing of provable detection and the actual deliverable undermines the paper's core claim.

- **No comparison to any existing adversarial detection method.** The paper discusses prior detectors (feature squeezing, LID, Mahalanobis, MagNet, CADet) in Section 2 but never evaluates any of them as baselines. Even a simple comparison against one or two established detectors on the same benchmark would clarify KOALA's relative standing. Without this, it is impossible to assess whether KOALA represents an advance over the detection landscape or merely a different heuristic with comparable performance.

- **No evaluation against adaptive attacks targeting the detector.** All attacks (PGD, CW, AutoAttack) target the backbone classifier, not the composed system (backbone + KOALA head + disagreement rule). An adaptive attacker with full knowledge of KOALA could potentially optimize perturbations that simultaneously cause agreement between the KL and L0 predictions on an incorrect class, bypassing the disagreement-based detection entirely. This is a standard and critical gap for any defense/detection paper—it was flagged as essential in reviews of multiple similar papers (e.g., KAWlH5pfQu, eG56H9teXv, 4HL2aiDV97).

- **Non-standard and potentially misleading confusion matrix definitions conflate robustness and detection.** The TP definition includes cases where an attacked input is correctly classified but the detector does not fire (â=0, ŷ=y*). In standard adversarial detection, if the detector does not flag an attacked input, that should be a false negative regardless of whether the underlying classifier happens to be robust. This inflates precision/recall numbers and obscures the distinction between attacks blocked by disagreement detection, attacks that fail to change the label, and attacks that succeed without detection.

### Minor:

- **Feature-space theorem assumptions are not verified against input-space experiments.** Theorem 1 operates in feature space (Assumption A2: ∥δ∥ ≤ ε in embedding space) but attacks use ℓ∞ bounds in input space. The paper mentions Lipschitz continuity as justification but provides no quantitative bound linking input ε to feature-space ε, and never checks whether Assumption A3 (|δ_i| ≤ 3/2|p*_i|) holds in practice.

- **Low practical coverage of the formal guarantee on CLIP.** Only ~10% (510/5000) of CLIP/Tiny-ImageNet samples are theorem-compliant, with non-compliant recall of only 0.80–0.84 and precision 0.62–0.63 (Table 1). The paper acknowledges this but does not propose mechanisms to increase coverage.

- **The proof claims existence of per-input τ but a fixed τ=0.75 is used in practice.** Proposition 4 in the proof sketch states "we can always find a threshold τ for the L0 metric that forces a trade-off," suggesting τ may depend on the input. The gap between this existential claim and the fixed operational choice is not addressed.

- **Results on CLIP/Tiny-ImageNet contradict the paper's core narrative.** The KL+L0 combination is not the strongest for adversarial accuracy on CLIP (Table 4); L0 alone is better for PGD and AutoAttack. This undercuts the claim of universal complementarity between KL and L0.

### Trivial:

- Detection performance (precision/recall) is only reported under PGD, while classification accuracy appears under CW and AutoAttack (Tables 3-4). Extending detection metrics to stronger attacks would be more informative.

## Nice-to-Haves

- Sensitivity analysis for hyperparameters (τ, φ, ω_L0, ω_KL), especially since τ is central to both the theoretical guarantee and practical performance.
- Visualizations of feature-space "stability bands" (e.g., t-SNE projections) to empirically verify the dense/sparse geometric motivation.
- Compliance rate as a function of perturbation budget ε to clarify the regime where the guarantee applies.
- Computational overhead analysis at inference time.

## Removed Points

- **"The proof is wrong" or "Theorem 1 is invalid":** The theorem is mathematically stated and a proof is provided in the appendix. The concern is about its *applicability* and *operationalization*, not its formal validity. Removed as mischaracterization.

- **"Softmax normalization destroys embedding structure" (implied by Human Finder point 7 about replacing the classifier head):** The paper reports clean accuracy of 94.78% for KL+L0 on ResNet vs. 95.16% baseline (Table 3), showing only a minor degradation. This suggests softmax normalization does not catastrophically harm the embeddings. The concern is reasonable but not substantiated as fatal.

- **"Only two model/dataset combinations" as a standalone weakness without context:** While more diverse evaluation would be beneficial, ResNet/CIFAR-10 and CLIP/Tiny-ImageNet represent meaningfully different architectures and scales. Two combinations is standard for an initial study but not sufficient for claims of generality. Downgraded to trivial.

- **"Claims of being modality-agnostic are speculative":** The paper claims to be "semantics-free" (which is supported: no semantic priors are used), but uses "various data modalities" language in the abstract without testing beyond images. The "semantics-free" claim is fair; "modality-agnostic" is mildly overstated. Weakened.

## Novel Insights

The most revealing observation from the experiments—one the paper partially acknowledges but does not fully grapple with—is that KOALA's detection mechanism can function via two fundamentally different pathways: (1) genuine disagreement detection where the attack pushes the embedding past one metric's stability boundary but not the other's, and (2) "broken classification" where the underlying classifier is destabilized so that all metrics randomly guess, yielding high disagreement rates for trivial reasons. The paper identifies pathway (2) for the KL+L0+Cosine combination on CLIP but does not systematically disentangle these two pathways across all configurations. This distinction is essential for any detector that replaces the classifier head: if the detector mostly works by making the classifier fragile rather than by detecting adversarial structure, its value as a security tool is questionable.

## Suggestions

1. **Add comparisons to at least 2-3 existing detectors** (e.g., feature squeezing, Mahalanobis distance) on the same benchmarks to establish KOALA's relative position.

2. **Design and evaluate adaptive attacks** that jointly optimize over both metric heads to maintain agreement while causing misclassification—this is essential for any detection paper claiming security guarantees.

3. **Separate detection metrics from classification metrics.** Redefine TP/TN/FP/FN so that the detector's performance is evaluated independently of the underlying classifier's robustness, providing a cleaner picture of what the detector itself contributes.

4. **Make the theorem's conditions operational.** Provide explicit computable Γ_i(ε) or at minimum an algorithmic check for compliance at test time (not just post-hoc partitioning), and report the fraction of inputs that are certified.

## Score and Decision

**Calibration anchors:**

- **Low end:** "Detecting Adversarial Examples" (KAWlH5pfQu) — scores 1,3,5,3,3 (withdrawn): wrong proof, non-adaptive evaluation, no baselines. "Neural Fingerprints" (eG56H9teXv) — scores 3,3,3 (withdrawn): no baselines, no adaptive attacks, single dataset. "Statistical Method" (kz78RIVL7G) — scores 3,3,1,3,3 (withdrawn): no novelty, poor baselines.

- **Mid range:** "Provably Safeguarding" (kwCHcaeHrf) — scores 6,5,6,5 (accept poster): has a formal guarantee with similar theory-practice gap issues, but includes baseline comparisons and tests across 3 datasets/architectures.

- **High end:** "Illusory Attacks" (F5dhGCdyYh) — scores 8,6,8 (accept spotlight): novel framing, thorough evaluation against adaptive attacks. "Expressive Losses for Verified Robustness" (mzyZ4wzKlM) — scores 8,8,6,5 (accept poster): provable guarantees with strong empirical results.

KOALA sits below "Provably Safeguarding" (which at least had baselines and multiple datasets) and shares many weaknesses with the low-scoring detection papers (no baselines, no adaptive attacks, theory-practice gap). However, it offers a genuinely novel detection principle with some solid empirical gains on ResNet/CIFAR-10. The overselling of the "formal proof of correctness" combined with the non-standard evaluation metrics, missing baselines, and no adaptive attack evaluation places this firmly below the acceptance threshold.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>