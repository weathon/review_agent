Now I have enough context. Let me verify a few key claims in the paper text before writing the final review.

The paper text is included in the user message. Let me verify key claims:

1. The paper says "complete unlearning" means 0% forget accuracy (Section 5.1: "we aim to achieve complete unlearning - achieving minimal accuracy on the forget set"). This is operationalized narrowly.

2. The paper acknowledges "weak unlearning" in Section 2: "Although one could argue that our methods lean towards a weak unlearning strategy—given the pre-trained backbone might retain some information about the forget set—our approach deviate from the strict definition of weak unlearning."

3. The DKVB retrain baseline is indeed absent from the paper.

4. MIA is deferred to Appendix A.7.

5. The paper does add ImageNet-1k and ResNet-50 results in this version, which addresses reviewer concerns from the prior submission.

Let me also check calibration scores:
- Prior submission (TLBPjECC5D): scores 5,5,6,5 (avg 5.25, rejected)
- Deep Unlearning (pUOesbrlw4): scores 5,5,3,8 (rejected)
- Simulated Oracle Matching (3vXpZpOn29): scores 6,6,8 (accepted poster)
- Information Theoretic Evaluation (NGF1wDDBMm): scores 5,6,6,6 (rejected)
- NegMerge (bKQJzuBSRJ): scores 6,6,6,6 (rejected)

Now let me write the final review.

## Summary

This paper proposes a low-compute machine unlearning method based on the Discrete Key-Value Bottleneck (DKVB), where unlearning is achieved by masking key-value pairs activated by the forget set—requiring only inference, no gradient updates. Two variants are introduced: "Unlearning via Activations" (masking the top-$N_a$ most frequently activated pairs) and "Unlearning via Examples" (masking all pairs selected by $N_e$ forget-set examples). Experiments on CIFAR-10/100, LACUNA-100, and ImageNet-1k with ViT-B/32 and ResNet-50 backbones show forget-class accuracy dropping to 0% with negligible retain-set damage, reportedly at 20× lower compute than SCRUB.

## Strengths

- **Elegant and simple core mechanism:** The insight that architectural sparsity via DKVB enables unlearning through simple key-value masking is conceptually clean, easy to implement, and requires no gradient-based retraining. This is a genuine practical advantage.

- **Strong empirical class suppression:** Across 4 datasets and 2 backbones, forget-class accuracy reliably drops to 0% while retain-set accuracy changes by ≤1–2% in most cases (Table 1). Figures 2–3 provide thorough hyperparameter sweeps of $N_a$ and $N_e$, giving practitioners clear guidance.

- **Clear computational advantage:** Table 2 documents that the proposed methods require zero backward FLOPs and orders of magnitude fewer forward FLOPs than all baselines, which is a meaningful practical result for deployment scenarios requiring rapid unlearning.

- **Improved evaluation over prior version:** Adding ImageNet-1k, ResNet-50 backbone, and explicit FLOPs comparison addresses concerns from the prior submission's reviewers.

## Weaknesses

### Fatal

None.

### Major

- **The "unlearning" claim substantially overreaches the evidence.** The paper motivates the work with privacy and "right to be forgotten" (§1), yet all evaluation measures only top-1 classification accuracy. A model can achieve 0% forget-class accuracy while still encoding forget-class information in its frozen backbone—an adversary with access to activations or gradients could recover class membership. The paper itself acknowledges (§2) that its method "leans towards a weak unlearning strategy—given the pre-trained backbone might retain some information about the forget set," yet elsewhere uses language like "complete unlearning," "removing the influence of the forget class," and "making the model behave as if it had not been trained on certain data" (§1). This gap between strong claims and narrow evaluation understates the method's limitations. What is demonstrated is **class-specific output suppression with minimal collateral damage**, not erasure of training influence. Membership inference attack evaluation is deferred to Appendix A.7 without quantitative results in the main paper.

- **Architecturally unbalanced comparison confounds architecture with unlearning method.** The proposed method uses a DKVB head with ~1M codebook entries and a non-parametric decoder, while all baselines use a standard linear head on the same backbone. The paper then claims its approach is "competitive with or better than SCRUB" (Abstract, Conclusion), but the comparison conflates (a) the effect of DKVB's localized representations—which inherently separate class information—with (b) the effect of the masking intervention. Without applying SCRUB (or other baselines) to the DKVB architecture, it is unclear whether the DKVB head itself provides most of the "unlearning" benefit, and the masking procedure is simply the natural intervention given the architecture. The "20× compute efficient" claim also mixes architectural decisions into the comparison, since forward FLOPs differ between DKVB and linear heads.

- **No retrain-from-scratch baseline within the same architecture.** The paper includes "Linear Layer + Retraining" (trained from scratch on the retain set) for the linear-head model, but does not train a DKVB model from scratch on the retain set only. This is the natural "gold standard" for unlearning: how close is the masked model to the model that never saw the forget data? Without this comparison, we cannot assess whether key masking approximates retraining within the DKVB framework—only that it zeros out forget-class accuracy. The linear-head retraining baseline is not a fair surrogate because the architectures differ.

- **Incremental novelty.** DKVB (Träuble et al., 2023) was designed for class-incremental learning, where its sparse localized representations allow new classes to be learned with minimal interference. Class unlearning is essentially the inverse operation, and the core insight—that one can delete class-specific key-value pairs—follows directly from DKVB's design. The paper's contribution is primarily empirical validation that this direct application works well, rather than a conceptual or methodological advance.

### Minor

- **Zero-shot unlearning claim is speculative.** Section 4 states the methods "can also be circumvented under appropriate assumptions, making the proposed approaches zero-shot unlearning methods." However, all experiments use the actual forget-class training data (via examples) or its activations. No experiment tests proxy data or cached activations from prior epochs. This framing should be qualified as future capability rather than a demonstrated property.

- **Class unlearning only; no instance-level demonstration.** The entire evaluation removes whole classes. Privacy regulations like GDPR typically require removing specific individuals' data, not entire categories. The paper does not discuss how the method would handle removing a subset of images from a class while keeping the rest. While not a fatal flaw for a class-unlearning paper, the strong privacy motivation in §1 makes this gap notable.

- **No analysis of key-value overlap across classes.** The method's success depends on class-specific key-value pairs being well-separated. The paper cites Träuble et al. (2023) for this property but provides no direct empirical verification (e.g., overlap statistics, heatmaps). If overlap is high, the approach may not scale to fine-grained or many-class settings; if overlap is near-zero, the result is nearly trivial.

- **Relative-accuracy metric in Table 1 obscures absolute performance.** The table reports relative change in accuracy, making it hard to assess whether DKVB's base accuracy is comparable to the linear baseline. If DKVB starts at lower accuracy, maintaining it is a weaker result than it appears.

### Trivial

- The $N_a$ counts in Table 2 (shown as e.g., "2193" TFLOPs for CIFAR-10 via Activations with ViT) lack decimal formatting, making some values ambiguous (e.g., "075" → 75 or 0.75?).

## Nice-to-Haves

- Evaluate membership inference attack resistance in the main paper, not just in the appendix.
- Apply SCRUB or NegGrad+ to the DKVB architecture to isolate the contribution of the masking method from the architectural bias.
- Include a retrain-from-scratch DKVB baseline as the true unlearning gold standard.
- Analyze key-value overlap across classes empirically.
- Demonstrate or discuss instance-level unlearning.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"FLOP accounting is imprecise / no wall-clock time"**: The harsh critic questioned backward FLOP approximations and absence of wall-clock time. The paper clearly describes its FLOP methodology (§5.1), approximating backward passes using the standard 2× parameter count convention. Wall-clock time is less relevant than FLOPs for a compute-efficiency claim since it varies with hardware. This is a nice-to-have, not a weakness.

- **"Require pre-trained encoders" as a fundamental limitation**: The harsh critic cited this as structural. The paper explicitly acknowledges this limitation (§6). It is inherent to the DKVB framework and does not undermine the method's contribution within its stated scope.

- **"Choosing the best-learned class as forget class makes the task not maximally challenging for entanglement"**: The critic suggested the selected classes may not be the most entangled. However, the paper chose the best-learned class precisely to make unlearning hardest in terms of initial confidence—a reasonable and standard choice. Entanglement analysis would be a nice-to-have additional experiment, not a weakness.

- **"Retain accuracy sometimes improves after unlearning"**: The critic suggested this warrants investigation. This is actually expected behavior with sparse representations—masking forget-class keys can act as a regularizer. It does not indicate a problem.

- **"Missing related works"**: Per the rules, I should not flag missing citations.

- **"Reproducibility concerns about hyperparameters or implementation details"**: Per the rules, I remove nitpicks about undisclosed hyperparameters. The paper provides sufficient implementation details for the core method, and additional details are in the appendix.

- **"Strong unmatching between motivation (privacy/LLMs) and experiments (image classifiers)"**: While the gap exists, the paper explicitly scopes itself to class unlearning in supervised classification. The LLM motivation is aspirational framing. This is a valid scope limitation but not a fatal flaw; it should be noted as a limitation rather than treated as disqualifying.

## Novel Insights

The most interesting observation across the reviews is that the method's success likely *derives from* DKVB's architectural inductive bias rather than from the masking intervention alone. Because DKVB enforces sparse, localized representations during training, the mapping between classes and key-value pairs is already nearly disjoint before any unlearning intervention. This means masking is essentially "deleting the right indices" rather than a sophisticated learning procedure—a trivially effective operation made possible entirely by the architecture. This is simultaneously the paper's core strength (the intervention is simple and compute-free) and its core limitation (without the specific architecture, there is no contribution). The paper could be significantly strengthened by decoupling these factors: showing what happens when baselines are applied to the same DKVB architecture.

## Suggestions

1. **Run SCRUB or NegGrad+ on the DKVB architecture** to disentangle architectural bias from unlearning method effectiveness. This is the single most impactful experiment the authors could add.

2. **Add a DKVB retrain-from-scratch baseline** (train DKVB on retain set only) to establish the true gold standard within the same architecture.

3. **Tone down "unlearning" language** throughout the paper—replace "complete unlearning" with "complete class suppression" or "zero forget-class accuracy," and qualify privacy claims rather than implying data erasure.

4. **Include MIA evaluation in the main paper** with quantitative results, not just a brief appendix mention.

5. **Report absolute base accuracies** for both DKVB and linear models before unlearning, to contextualize relative-change metrics.

## Score and Decision

**Calibration:**
- Prior submission of this paper (TLBPjECC5D, 3 datasets, ViT only): scores 5,5,6,5 = avg 5.25 (rejected)
- Deep Unlearning (pUOesbrlw4, same domain, training-free class unlearning): scores 5,5,3,8 = avg 5.25 (rejected)
- Simulated Oracle Matching (3vXpZpOn29, stronger unlearning method with principled evaluation): scores 6,6,8 = avg 6.7 (accepted poster)
- Information Theoretic Evaluation (NGF1wDDBMm, unlearning metric paper): scores 5,6,6,6 = avg 5.75 (rejected)
- NegMerge (bKQJzuBSRJ, unlearning method): scores 6,6,6,6 = avg 6.0 (rejected)

This paper has improved over its prior version (added ImageNet-1k, ResNet-50, FLOPs comparison), but the core concerns persist: (1) the overclaim on "unlearning" when what is shown is output suppression, (2) the confounded comparison across architectures, and (3) incremental novelty as a straightforward application of DKVB properties. These are substantial issues that limit the contribution. The paper is comparable in quality to the prior submission (5.25 avg) with modest improvement from expanded experiments, but the fundamental weaknesses are not addressed. I place it slightly above the prior version due to the expanded evaluation but below NegMerge (which had a more novel methodology but still was rejected at 6.0).

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>