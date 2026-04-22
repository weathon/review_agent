# How much correction is adequate? A Unified Bias-Aware Loss for Long-Tailed Semi-Supervised Learning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Long-tailed semi-supervised learning (LTSSL) suffers from class imbalance-induced biases in both training and inference. Existing debiasing methods typically rely on distribution priors, which fail to capture two critical dynamic factors: the pseudo-labeling-induced shifts in effective priors and the model’s intrinsic evolving bias. To address this limitation, we propose Bias-Aware Loss (BiAL), a unified objective that replaces static distribution priors with the model’s current bias. This straightforward substitution enables BiAL to generate plug-and-play bias-aware variants of cross-entropy/logit adjustment and contrastive heads, thereby unifying prior correction across diverse network architectures and training paradigms. Through theoretical analysis and empirical validation, we prove that our BiAL provides a singular, unified mechanism to align training with model’s evolving state and achieves state-of-the-art performance on multiple datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper attempts to address the issue of dynamic bias in long-tailed semi-supervised learning (LTSSL). The authors propose BiAL, which replaces the static class prior with an online estimate of the model's own bias, measured from its output on no-information inputs. The authors subtract this estimated bias from the model's logits to form debiased energy, and uniformly applying this energy across all loss functions provides a more adaptive and effective correction mechanism. The experiments reportedly achieve state-of-the-art performance on multiple datasets.

### Strengths
1. The idea of probing model bias with no-information inputs seem to be simple and effective. And the experiments verified that the method achieve sota performance.
2. The paper writing is clear and easy to understand.

### Weaknesses
1. The core idea of this paper that using the model's response to no-information inputs to estimate and correct for its bias has been proposed in CDMAD [1]. The use of bias just dynamic variant of the classic Logit Adjustment (LA). [2]
2. Lack theoretical foundation for why $b_{\theta}$ is a good estimator for the "effective training prior. The paper provides no rigorous proof or analysis whatsoever.
3. The paper is replete with grandiose terms like unified, universal, and "fundamental," which do not align with the actual substance of the contribution. The so-called "unification" is just applying a simple logit subtraction to different components.
[1] Cdmad: class-distribution-mismatch-aware debiasing for classimbalanced semi-supervised learning.
[2] Long-tail learning via logit adjustment.

### Questions
See in Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper makes the following key contributions for long-tailed semi-supervised learning (LTSSL), suitable for international machine learning conference review:  

1. **Unified Bias-Aware Objective**: Proposes Bias-Aware Loss (BiAL), which replaces static distribution priors (limiting existing methods) with the model’s current class bias (estimated from no-information inputs). BiAL unifies bias correction across cross-entropy/logit adjustment and contrastive heads, and extends to supervised learning, enabling consistent debiasing in diverse architectures (e.g., FixMatch, CCL).  

2. **Theoretical Guarantees**: Establishes Fisher consistency for balanced error rate (BER) with BiAL’s debiased energy, derives dynamic-regret advantages under prior drift (induced by pseudo-labeling), and proves that static-prior methods suffer from unavoidable mismatch—quantifying their excess BER to justify BiAL’s adaptivity.  

3. **Plug-and-Play Implementation**: Adds negligible computational overhead (only lightweight bias probing and logit adjustment) without extra components. It integrates seamlessly into existing SSL pipelines via warm-up/ramp scheduling for stability.  

4. **Empirical Validation**: Achieves state-of-the-art (SOTA) performance on CIFAR10/100-LT, STL10-LT, and ImageNet-127 across consistent/uniform/reverse unlabeled distributions. It concurrently improves pseudo-label quality and test accuracy, outperforming strong baselines (e.g., CPE, Meta-Experts, CCL).

### Strengths
# 1. **Well-motivated and unified formulation**
The paper proposes a unified Bias-Aware Loss (BiAL) that replaces static class priors with model-induced bias estimated from no-information inputs. This principled abstraction enables a plug-and-play correction mechanism compatible with multiple paradigms such as CE, LA, and contrastive heads . The idea is conceptually clean and addresses a central limitation of prior long-tailed SSL approaches.

# 2. **Solid theoretical justification**
The authors provide clear theoretical insights, showing that pseudo-labeling induces dynamic class-prior drift and that static prior correction becomes misspecified. The analysis within the balanced-error framework demonstrates that BiAL can reduce dynamic regret and align more closely with the evolving effective prior This theoretical grounding strongly supports the method.

### Weaknesses
# 1. **Bias estimation stability remains unclear**
The bias is estimated using model predictions on no-information images (e.g., black images) and stabilized with EMA and warm-up strategies . However, the accuracy and robustness of such estimation—particularly during early training—may be questionable. More analysis on sensitivity to batch size, input type, and noise would be helpful.
# 2. **Performance improvements can be marginal in some setups**
Although the method achieves state-of-the-art results on several benchmarks, improvements over strong baselines (e.g., FixMatch+ACR/CPE) are sometimes relatively small and appear within statistical variance. More significance analysis or discussion would help contextualize these gains.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Bias-Aware Loss (BiAL), a unified framework for long-tailed semi-supervised learning (LTSSL) that replaces static distribution priors with the dynamically estimated bias of the model. The core idea is to measure this bias by probing the model on no-information inputs (e.g., solid black images) and then use it to correct logits throughout training and inference. The method is simple, theoretically grounded and empirically strong. It achieves highly competitive performance across multiple datasets (CIFAR-10/100-LT, STL-10-LT, ImageNet-127). The paper provides good theoretical guarantees (Fisher consistency, dynamic regret bounds) and several experiments that validate the method's robustness.

### Strengths
1. Novel conceptual framework
    - Introduces a dynamic bias estimation mechanism that generalizes prior static approaches.
    - Unifies existing bias-corrective losses under a single principle.
2. Strong theoretical foundation
    - Fisher consistency and dynamic regret proofs provide mathematical backing.
    - Gradient-level analysis clarifies improvements in minority class margins.
3. Comprehensive empirical validation
    - Benchmarks across 4 datasets and multiple distribution regimes.
    - Thorough ablations and sensitivity checks.
4. Practical utility
    - Plug-and-play integration with minimal overhead.
    - Includes practical engineering details (warm-up, EMA smoothing, ramp-up).
    - Clear implementation and reproducibility potential.
5. The paper is well written, with clear motivation, good organization, and visual presentation.

### Weaknesses
**Limited bias source analysis:** The method captures aggregate bias, but the paper does not separate its components (e.g. class imbalance vs. architectural difficulty).

**Hyperparameter sensitivity:** The introduction of new parameters $(\beta, E_\mathrm{est}, E_\mathrm{warm}, E_\mathrm{ramp}, \alpha)$ adds to the tuning cost. While a sensitivity analysis is provided, clear heuristics for setting these on new datasets are limited.

**Minor writing issues:** Includes missing references to figures and a minor equation labeling error, which slightly impact the reading flow.

**Limited exploration of "No-Information" inputs:** The paper uses all-black images but does not ablate this choice. Exploring other types of non-informative inputs (e.g., noise patterns) could have strengthened the methodological foundation.

**Domain-specific semantic meaning of "No-Information" inputs:** The method's core assumption is that a solid black image serves as a neutral, non-informative baseline. However, in specialized domains like medical imaging, the color black can carry significant clinical meaning (e.g. specific tissue types, or the absence of a finding). In such cases, using a black image would not probe the model's intrinsic class bias but would instead measure its response to a semantically charged input, leading to a corrupted and misleading bias estimate.

**Contextual bias from training data:** The bias estimation relies on the model's output for a constant-colour input. However, if the original training dataset contains correlations between plain backgrounds and specific classes, the model may learn these spurious associations. Consequently, the estimated bias vector $b_\theta$ would capture this dataset-specific contextual bias (e.g., a bias towards classes frequently appearing with blank slides) alongside the intended class-frequency bias.

### Questions
1. Bias composition: The measured bias $b_\theta$ is treated as a unified vector. Can you disentangle how much of this bias originates from class imbalance versus other factors, such as the model's architectural prior or dataset-specific visual biases (e.g., background correlations)? Is the bias on no-information inputs a pure reflection of the label distribution?
2. Hyperparameter tuning: What heuristics or adaptive schemes could help set $\beta, E_\mathrm{warm}$, and $E_\mathrm{ramp}$ for new datasets?
3. Scalability: How does computational cost scale with model size and class count?
4. Failure cases: When might BiAL underperform relative to static priors?
5. Bias estimation robustness: What happens when bias estimation is noisy or unstable?
6. Visual similarity: Could you analyse how BiAL affects discrimination among visually similar head vs. tail classes?
7. Experimental fairness: Were the same data samples used consistently across all method comparisons?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of long-tailed semi-supervised learning (LTSSL), where both label imbalance and pseudo-label noise cause strong class bias during training. The authors observe that most existing debiasing methods use static distribution priors (e.g., class frequencies), which become inaccurate as the model evolves and pseudo-labels change the effective class distribution.

To address this, the paper proposes Bias-Aware Loss (BiAL), a unified bias-aware objective that replaces static priors with the model’s current bias, estimated directly from its responses to no-information inputs (e.g., blank images). This approach allows consistent correction during both training and inference, and can be easily plugged into different SSL frameworks such as FixMatch and CCL.

### Strengths
1. Paper is well-written and overall well organized.

### Weaknesses
The main idea of using model bias estimated from no-information inputs is conceptually close to DebiasPL (“Debiased Learning from Naturally Imbalanced Pseudo-Labels,” CVPR 2022). Both approaches rely on the model’s self-bias for correction without external priors. DebiasPL's causal inference pipeline The new method mainly wraps this idea into a unified loss formulation (BiAL) but does not introduce a very different underlying mechanism. The contribution seems more incremental than fundamentally new. 

Most experiments are conducted on small benchmarks such as CIFAR10/100-LT and STL10-LT, with limited data diversity and visual complexity. While the method shows nice improvements there, it is unclear whether the gains can generalize to large-scale or real-world long-tailed semi-supervised scenarios (e.g., ImageNet-LT, WebVision, or domain-shifted data).

The paper only discusses classification tasks. It is not clear whether the proposed bias-aware correction can be generalized to other settings like detection, segmentation, or multimodal learning. Since those tasks often involve structured outputs and continuous predictions, the practical applicability of BiAL outside classification remains uncertain.

Marginal performance gains. As shown in table 1 and table 2, the performance gains is often within 0.5%.

### Questions
Please check the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2
