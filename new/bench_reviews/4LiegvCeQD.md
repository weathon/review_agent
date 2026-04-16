## Summary

This paper proposes Intra-model Ensemble Learning (IEL), a test-time adaptation method that simultaneously adapts multiple independently pre-trained classifiers by selecting, for each test sample, the softmax output with the highest probability for the majority-voted class as a soft pseudo-label, then minimizing the cross-entropy between this target and all member models. IEL operates on single-sample batches and updates all trainable parameters. Experiments on CIFAR-10-C, CIFAR-100-C, and ImageNet-C show improvements over static ensemble baselines on most corruption types.

## Strengths

1. **Practically relevant single-sample TTA setting**: The paper addresses a genuinely restrictive scenario where only one sample is available per adaptation step, which is important in domains where batch statistics cannot be acquired (e.g., medical imaging, time-sensitive applications). This fills a gap left by batch-dependent methods like TENT. The paper correctly identifies that batch-dependent methods degrade at batch size 1, providing clear motivation (Section 1, Section 2.2).

2. **Simple, transparent mechanism**: The core update rule—pick majority-vote, highest-confidence softmax as teacher and distill into all models via cross-entropy—is straightforward and easy to reproduce. Algorithm 1 is clear enough that reimplementation would be trivial.

3. **Individual models improve, not just the ensemble**: Unlike standard ensemble methods (bagging, stacking) that leave members unchanged, IEL's backpropagation through all members means each model genuinely adapts. Figure 2 clearly shows ResNet50 accuracy rising from ~41% to ~47% across epochs on Glass Blur, demonstrating this property.

4. **Extensive corruption coverage**: Evaluation across all 15 corruption types on CIFAR-10-C, CIFAR-100-C, and ImageNet-C with multiple architectures per dataset provides a reasonably broad empirical picture. Positive results are shown on a majority of corruption types, particularly strong on blur-type corruptions (e.g., +20% on Zoom Blur for some models).

5. **Negative results openly reported**: Tables 1–2 show substantial degradation on noise-type corruptions rather than hiding them, which allows readers to assess the method's limitations.

## Weaknesses

### Major:

1. **No comparison to any existing TTA methods**: This is the most significant empirical gap. The paper positions itself as a TTA method and discusses TENT, EATA, CoTTA, and ROID in related work, but compares only against static (unadapted) ensembles. Without single-model TTA baselines, it is impossible to determine whether IEL's gains come from the adaptation mechanism or simply from having multiple models. Even a naive single-model entropy minimization applied independently to each ensemble member would serve as an instructive baseline. The paper does not demonstrate that IEL outperforms any adaptive baseline whatsoever—only that it beats doing nothing.

2. **"Diversity-based" framing is contradictory to the actual mechanism**: The paper's central novelty claim is that it "proposes diversity as a new optimization signal" (Contributions). In reality, the method explicitly minimizes diversity by forcing all models to converge toward the majority-voted softmax via cross-entropy. The paper itself states "we minimize the diversity of the ensemble" (Section 1). The actual mechanism is majority-vote pseudo-label self-distillation, which has clear precedents in mutual learning and co-training. While the authors acknowledge this tension, the conceptual framing overclaims novelty—the "diversity signal" is actually a consensus/agreement signal. This mischaracterization affects the claimed conceptual contribution.

3. **Cherry-picking best-epoch results without a fixed stopping criterion**: Tables 1–3 report "highest accuracy improvements over all epochs," which represents an oracle selection. Figure 3 shows that accuracy peaks around epoch 4-5 and then declines or fluctuates. In practice, without access to ground-truth labels, a practitioner has no principled way to determine when to stop. The method can produce results *worse* than the static baseline if run too long, and there is no mechanism proposed for detecting this. This inflates the reported performance relative to any practical deployment scenario.

4. **Evaluation limited to stationary, per-corruption adaptation with weight resets**: Between each corruption type, all model weights are reset. This avoids the exact problem (catastrophic forgetting across domains) that is central to TTA research. In realistic TTA deployment, one does not know when the distribution shifts or have the ability to revert weights. The broader claims about TTA and continual learning (Abstract, Contributions, Conclusion) are not supported by the experimental protocol, which is closer to unsupervised per-corruption domain adaptation than to online TTA.

### Minor:

5. **Severe failure on noise-type corruptions is insufficiently analyzed**: On CIFAR-10-C, majority vote accuracy drops by 20.56%/17.70% on Gaussian Noise (Table 1). On CIFAR-100-C, four corruption types show degradation (Table 2). The paper states this without analysis of why noise corruptions are particularly problematic or any suggestion for mitigation. This is a significant practical limitation that is acknowledged but not investigated.

6. **No ablation studies**: Key design choices are unexplored: (a) effect of ensemble size, (b) architectural diversity vs. same-architecture ensembles, (c) cross-entropy vs. KL divergence as the distance function (the paper mentions KL as future work), (d) temperature scaling of the soft target, (e) learning rate sensitivity. The effective learning rate (0.001 × 10e-11 regularization) is essentially zero, suggesting the actual step size may be very small—this deserves clarification or analysis.

7. **Computational cost unquantified**: IEL requires backpropagation through M full models per test sample. The paper acknowledges this is "more computationally heavy" but provides no latency, FLOP, or memory comparison. For a method explicitly targeting real-time inference, this is a notable gap.

### Trivial:

8. **The human collaboration analogy (Section 1, "two heads are better than one") is overextended**: The analogy doesn't translate into concrete design principles beyond "models learn from each other." The actual mechanism (cross-entropy minimization toward a pseudo-label) is standard distillation with a particular teacher-selection rule.

## Nice-to-Haves

- Add comparisons to at least TENT, EATA, and CoTTA under single-sample settings to position IEL in the TTA landscape. Even if these methods degrade at batch size 1, demonstrating this empirically would be informative.
- Report results at a fixed epoch (e.g., epoch 1 or 5) alongside best-epoch results, or propose a practical stopping criterion (e.g., based on model agreement or loss stabilization).
- Evaluate under non-stationary shifts where corruption types change during the test stream without weight resets, which is the more realistic TTA scenario.
- Analyze how ensemble diversity evolves over IEL epochs and correlates with performance, which would substantiate the paper's engagement with diversity theory.

## Removed Points

- *Reviewers questioned the existence/release status of pre-trained models and datasets.* The paper cites publicly available sources (PyTorch Vision models, GitHub repos for CIFAR models, and standard corruption benchmarks). These are standard resources that exist and are widely used; this is not a valid concern.

- *Harsh critic claimed Section 2.1 on ensemble diversity is "conceptually inconsistent" because the paper argues against diversity via a counterexample.* The paper presents a nuanced position: diversity is generally beneficial for ensembles, but it is *possible* to reduce diversity while also reducing generalization error (e.g., identical perfect models). This is a reasonable observation, not an inconsistency. The issue is with the "diversity as optimization signal" framing in the contributions, not with Section 2.1's discussion.

- *Harsh critic claimed the $H(x)$ definition is "unclear" and hurts reproducibility.* The paper states $H(x) := \arg\max_{\{h_{\theta_i}(x)|i=1,...,M\}} h_{\theta_i}(x)^{(c)}$ where $c$ is the majority-voted class. While slightly informal, the meaning (select the model with highest probability for the majority class, use its full softmax) is recoverable from context and Algorithm 1. This is a minor clarity issue, not a reproducibility barrier.

- *Harsh critic claimed the ImageNet ensemble uses "only ResNet variants" undermining the "distinct architectures" claim.* The CIFAR experiments use genuinely diverse architectures (ResNet, VGG, MobileNet, ShuffleNet, RepVG). The ImageNet experiments use ResNet-50, ResNet-101, ResNet-101_64x4d, and ResNet-152. While these are from the same family, they are distinct architectures with different depths and widths. The paper says "unless otherwise specified we will assume that any two models with distinct weights also have distinct architectures" (Section 3), which is satisfied. The concern is valid for generalization claims but does not invalidate the experimental setup.

- *Spark suggested checking whether the regularization constant α = 10e-11 is "intentional or a bug."* The paper states "a regularization constant α = 10e−11 which effectively makes our learning rate even smaller." This appears intentional—it is a weight decay term. While its near-zero effect is curious, this is not a bug claim the reviewers have grounds to make.

## Novel Insights

The observation that IEL's losses naturally correlate with Shannon Entropy (Figure 1) is interesting but expected given the mathematical relationship between cross-entropy toward a peaked distribution and entropy. A more insightful observation from the results is that IEL fails dramatically on noise-type corruptions but succeeds on blur-type corruptions. This likely reflects a fundamental property of the method: when all models are severely corrupted (noise), their majority vote provides a poor pseudo-label, and the distillation process amplifies errors. On blur corruptions, models retain partial discriminability, making the majority vote more reliable. This asymmetry reveals that the method's usefulness depends on a minimum level of residual model capability across the ensemble—a condition the paper does not formalize or test.

## Suggestions

1. **Add comparisons to TENT, EATA, and CoTTA** even under their suboptimal batch-size-1 settings, and to ROID which directly builds ensembles at test time. This is the single most important improvement for positioning the contribution.

2. **Reframe the contribution** around "consensus-based ensemble self-distillation for test-time adaptation" rather than "diversity-based optimization." The actual novelty is in the specific teacher-selection rule (majority-vote, highest-confidence) and the application to the single-sample TTA setting, not in a diversity signal.

3. **Report fixed-epoch results** alongside peak results, or provide a stopping criterion that does not require oracle access.

4. **Investigate the noise-corruption failure mode**: Quantifying the accuracy of the majority-vote pseudo-labels per corruption type would reveal when consensus fails and when it succeeds, providing practical guidance for deployment.

## Score and Decision

**Calibration**: I compared against several TTA papers with similar evaluation issues. The TTA-with-Auxiliary-Tasks paper (PxL35zAxvT, scores 5/6/3, Reject) was rejected partly for lacking proper baselines and having evaluation concerns. The "Controlling Forgetting" paper (fRNDDFkPiv, scores 5/6/8/8, Reject) was rejected for unrealistic evaluation settings and missing comparisons. The TTE paper (4wk2eOKGvh, scores 6/8/6/6, Accept Poster) was accepted partly because it integrated with existing TTA baselines and evaluated on continual TTA. DeYO (9w3iw8wDuE, scores 8/6/6/8, Accept Spotlight) had a novel insight with strong evaluation against baselines. IEL has an interesting idea but falls between the rejected papers (missing baselines, limited evaluation protocol) and the accepted papers (no baseline comparison, stationary-only protocol).

The core idea—mutual adaptation of an ensemble at test time using majority-vote pseudo-labels—is reasonable and the single-sample setting is valuable. However, the paper has three significant issues that collectively undermine its claims: (1) no comparison to any adaptive baselines, making it impossible to assess the method's TTA contribution; (2) best-epoch cherry-picking without a practical stopping criterion; and (3) stationary-only evaluation with weight resets that avoids the hardest TTA challenges. Additionally, the conceptual framing around "diversity" mischaracterizes what is actually a consensus mechanism. These are not minor gaps—they affect whether the paper's core claims are supported.

MY FINAL SCORE: <pineapple>4</pineapple>
MY FINAL DECISION: <orange>Reject</orange>