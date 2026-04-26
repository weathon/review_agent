Now I have thoroughly reviewed the paper and verified the critical claims from the harsh critic. Let me synthesize the final review.

## Summary

KOALA proposes an adversarial detection method that flags inputs as attacked when predictions from two complementary similarity metrics (KL divergence and L0-based) disagree. The method requires only clean-image fine-tuning of a pre-trained encoder to align embeddings with both metrics. A formal theorem guarantees detection when inter-class prototype separation is sufficiently large.

## Strengths

- **Novel and well-motivated detection principle**: The core idea—that dense and sparse perturbations are captured by different metrics, and their disagreement signals an attack—is intuitive, geometrically motivated (Figure 1), and genuinely novel in the adversarial detection landscape.
- **Formal guarantee structure**: Theorem 1 establishes that, under Assumptions A1–A4, no single perturbation can simultaneously fool both the KL and L0 classifiers. The proof sketch (mutual exclusivity of stability bands via Props. 2–4) is a reasonable theoretical direction for providing provable detection conditions, differentiating KOALA from purely empirical detectors.
- **Honest failure analysis**: The paper explicitly acknowledges (§4.3) that the best detection metric on CLIP (KL+L0+Cosine) achieves high detection rates partly because it degrades classification accuracy, making it "not always equate to a truly robust model." This kind of self-critical analysis is unusual and welcome.
- **Clean training procedure**: Fine-tuning only on clean images without adversarial examples is practically appealing compared to adversarial training.

## Weaknesses

### Fatal

None.

### Major

- **Inflated detection metrics due to conflation of detection and classification robustness**: The TP definition (Eq. in §4.2) counts an adversarial example as a "true positive" for detection even when the detector outputs (â=0, ŷ=y*) — meaning the detector failed to flag the attack but the underlying classifier happened to classify correctly. From Table 3, the KL+L0 ResNet model correctly classifies 57.32% of PGD adversarial examples; every such example counts as a TP regardless of whether the detection mechanism triggered. This makes the reported precision of 0.94 and recall of 0.81 (Abstract, Table 2) impossible to interpret as measuring detection capability alone — they conflate detector performance with the model's inherent classification robustness. The "perfect" P=1.0/R=1.0 on theorem-compliant samples is partly an artifact of this definition: if compliant samples are those where the model classifies correctly, they are automatically TPs even if the detector does nothing. This significantly undermines the empirical contribution.

- **No comparison with any established adversarial detection baseline**: The paper's Related Work section discusses Mahalanobis detection (Lee et al., 2018), LID (Ma et al., 2018), feature squeezing (Xu et al., 2018), and MagNet (Meng & Chen, 2017), yet none appear in the experiments. All comparisons are between metric combinations within the KOALA framework. Without external baselines, there is no way to assess whether KOALA advances the state of the art, or whether a simple baseline detector would outperform it.

- **Formal guarantee applies to a small minority of inputs on the more realistic benchmark**: Table 1 shows that only 510–556 out of ~5000 samples (~10%) on CLIP/Tiny-ImageNet satisfy Theorem 1's conditions. On ResNet/CIFAR-10, ~60–67% are compliant. On the harder, more realistic benchmark, the guarantee is effectively vacuous for the vast majority of inputs. The paper repeatedly emphasizes a "formal proof of correctness" (Abstract, §1, §3.2) without adequately foregrounding this limited practical scope.

### Minor

- **Misleading "no architectural changes" claim**: The abstract and introduction state the method "requires no architectural changes," yet §3.1 explicitly describes replacing the conventional classifier head with the KOALA Detector. This is an architectural change. The claim should be revised or qualified.

- **Gap between existential threshold guarantee and fixed τ = 0.75**: The proof argues τ can always be found, but in practice τ = 0.75 is a fixed hyperparameter. There is no analysis showing this specific τ satisfies Theorem 1's conditions for any particular input set. This is a standard theory-practice gap but should be acknowledged.

- **No adaptive attack evaluation**: The detector's rule (metric disagreement) is simple and published. Standard practice for adversarial defense papers includes evaluation against attacks that target the defense mechanism directly. The paper evaluates against PGD, CW, and AutoAttack but not against attacks designed to make KL and L0 predictions agree.

- **Unverified assumptions**: Assumption A2 (bounded feature-space perturbation from Lipschitz continuity) and A3 (coordinate-wise bound) are stated but not verified for the specific models used. No Lipschitz bounds are computed for ResNet-18 or CLIP ViT-B/32.

### Trivial

- The "semantics-free" claim, while reasonable in contrast to methods requiring domain-specific priors, does require labeled data for fine-tuning with class prototypes, which is more restrictive than truly unsupervised detection methods.

## Nice-to-Haves

- Re-evaluate using pure detection metrics (e.g., AUROC of the detector's disagreement signal) on adversarial examples that successfully fool the base classifier, separating detection from classification robustness.
- Include at least 2–3 established detection baselines (Mahalanobis, LID, feature squeezing) in the comparison table.
- Evaluate against adaptive attacks that explicitly optimize for agreement between KL and L0 predictions.
- Characterize what fraction of input-class pairs are theorem-compliant as a function of model architecture, dataset, and fine-tuning strategy, providing guidance on when the guarantee is practically useful.

## Removed Points

These points were flagged for removal and should be treated with caution:

- *Harsh critic: "perfect 1.0 scores on compliant samples are artifacts of the flawed TP definition"* — This is a real concern (see Major weakness above), but the claim that ALL perfect scores are merely artifacts is too strong. The theorem does guarantee detection on compliant samples, so perfect scores there are theoretically predicted. The concern is specifically that the TP definition also counts correct classifications as detection successes, inflating the apparent magnitude, but not that the entire result is artifactual.
- *Harsh critic: "CLIP/Tiny-ImageNet detection works because model is effectively broken"* — The paper itself acknowledges this point explicitly in §4.3, so this is not an overlooked weakness but a known limitation already discussed.
- *Harsh critic: "Adversary could craft perturbations lowering μ(c,p) to make L0 less sensitive"* — This is an adaptive attack concern, already captured in the "no adaptive attack evaluation" minor weakness. It's speculative without experimental evidence.
- *Harsh critic: "Standard deviations/confidence intervals not reported"* — This is a minor reproducibility concern; single-run evaluation is standard practice in this field for large-scale adversarial evaluations.
- *Harsh critic: "L0 threshold depends on adversarial embedding p"* — This is a correct observation but conflated with the adaptive attack concern. The paper's L0 metric uses μ(c,p) which depends on the perturbed embedding; however, the dependence is via the average distance, making it a relatively minor vulnerability rather than a fundamental flaw.
- *Harsh critic: "Replacing classifier head contradicts 'no architectural changes'"* — Kept as a minor weakness (see above) since it is a valid criticism, though the method is still "plug-in" in spirit as it only replaces the final layer.
- *Strength finder: "Improved adversarial classification accuracy alongside detection"* — This is correct but somewhat undermined by the detection metric inflation concern; however, the classification accuracy tables (Table 3, 4) are separately reported and valid, so this strength is partially retained.

## Novel Insights

The most interesting observation that emerges from the interaction between the paper's claims and the evaluation concerns is the fundamental tension between classification robustness and detection. On CLIP/Tiny-ImageNet, the metric combination with the highest detection rate (KL+L0+Cosine) simultaneously achieves the worst classification robustness, suggesting that making the two metrics disagree more often (and thus detecting more attacks) and maintaining correct classification under attack may be partially contradictory objectives. This tension — that a better "detector" might be one that ruins the classifier — is a genuine structural challenge for disagree-based detection methods that the community should grapple with more explicitly.

## Suggestions

- Report a pure detection metric (e.g., AUROC of the binary disagreement signal â) alongside the current system-level metrics, evaluated only on adversarial examples that successfully change the classifier's prediction. This would isolate detection capability from model robustness.
- Include at least Mahalanobis-based detection (Lee et al., 2018) and feature squeezing (Xu et al., 2018) as baselines, as they are the most directly comparable methods.
- Add a short analysis of what fraction of samples satisfy Theorem 1's conditions on additional model architectures to help readers assess the practical scope of the guarantee.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| KAWlH5pfQu (Adversarial detection, overclaimed AUC > 0.99, no adaptive eval, weak baselines) | /human_reviews/KAWlH5pfQu.md | 3.0 | Lower — KOALA has a real formal theorem and more honest analysis, but similar evaluation methodology issues |
| AHqXvTK4KG (Adversarial detection, no baselines, metric issues) | /human_reviews/AHqXvTK4KG.md | 3.5 | Comparable — similar lack of baselines and metric concerns, but KOALA has stronger theoretical contribution |
| kwCHcaeHrf (Provably safeguarding classifier, formal guarantees but weak experiments) | /human_reviews/kwCHcaeHrf.md | 5.5 | Higher — KOALA has more serious evaluation methodology issues (TP definition, no baselines) |
| F5dhGCdyYh (Adversarial detection with formal detectability constraints) | /human_reviews/F5dhGCdyYh.md | 7.33 | Significantly higher — that paper had rigorous formal framework and well-designed experiments |

KOALA falls between the 3.0–3.5 weak detection papers and the 5.5–6.0 provable defense papers. Its core idea is genuinely novel and the formal guarantee direction is sound, but the evaluation has serious issues: the TP metric conflation, lack of baseline comparisons, and the guarantee's limited practical scope (10% compliance on CLIP) collectively undermine the paper's claims. The paper is more substantial than the truly weak detection papers (avg ~3) that had fabricated metrics and wrong proofs, but falls well short of the stronger provable defense papers (avg ~5.5–6) that had cleaner experimental methodology. The most comparable anchor is AHqXvTK4KG at 3.5, but KOALA's theoretical contribution merits a small premium.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>