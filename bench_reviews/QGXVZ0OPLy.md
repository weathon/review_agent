## Summary

The paper proposes DualPrompt, a training-free method to improve CLIP's zero-shot multi-label classification by incorporating label co-occurrence information through dual prompts: a discriminative prompt (DiP) containing only the target label, and a correlative prompt (CoP) that includes co-occurring labels. The authors identify that while CoP helps recognize multiple objects, it causes object hallucination; they propose combining both prompts via a causal inference framework to retain benefits while mitigating hallucination. Experiments on MS-COCO, VG-256, and Objects365 show improvements over baselines.

## Strengths

- **Interesting empirical finding about co-occurrence:** The paper convincingly demonstrates that correlative prompts have a dual nature—Figure 2 and Appendix B show they improve ~50% of classes while degrading the other half, which is a valuable insight for the community. The visualization of co-occurrence probability gaps in Figure 1 provides concrete motivation for the work.

- **Practical and simple implementation:** The method requires only dual prompt inference and probability addition (Eq. 2), making it easy to adopt and computationally efficient (no model fine-tuning required). The approach works with multiple backbones (ResNet-101, ViT-B/16) and can combine with TagCLIP for further gains.

- **Empirical effectiveness:** DualPrompt achieves meaningful improvements over vanilla CLIP (+4.6 mAP on MS-COCO with ResNet-101, +2.8 mAP with ViT-B/16). When combined with TagCLIP and using 1% data for co-occurrence estimation, it reaches 70.0 mAP on MS-COCO, outperforming prior methods.

- **Analysis of co-occurrence sources:** The paper compares ChatGPT-generated vs. data-estimated co-occurrences (Section 6.5, Figure 7), showing that even small amounts of labeled data (1%) provide better co-occurrence statistics than generic LLM knowledge.

## Weaknesses

- **Theoretical derivation lacks rigor:** The transformation from Eq. 1 (subtraction form) to Eq. 2 (addition form) relies on unstated conditional independence assumptions and a proportionality constant λ that is set to 1 empirically without justification. The paper states Eq. 1 "hardly works" due to CLIP overestimating p(y=1|x, L^c_k), but if Eq. 2 is mathematically equivalent as claimed, this should not matter. This inconsistency suggests Eq. 2 is an empirical heuristic rather than a principled causal intervention, which weakens the causal framing of the paper.

- **Missing critical ablations:** The number of co-occurring labels (l=2 for ChatGPT, top co-occurring for data estimation) and the λ=1 setting are chosen without sensitivity analysis. The paper does not test whether simpler alternatives (e.g., averaging DiP and CoP scores, or using learned weights) would perform similarly, which would help validate whether the causal framework is necessary or merely post-hoc justification.

- **Modest gains over the strongest baseline:** DualPrompt alone (ViT-B/16, no data) achieves 67.7 mAP, which is actually lower than TagCLIP's 68.7 mAP. The best results require combining DualPrompt with TagCLIP (70.0 mAP), raising questions about the method's standalone value. The improvements over TagCLIP alone are small (+1.3 mAP with 1% data for co-occurrence).

- **"Training-free" claim is misleading:** The title and abstract emphasize "training-free," but the best results require 1% labeled training data for co-occurrence estimation. While the model weights are not updated, this is not truly zero-shot—downstream labeled data is used to compute statistics. The paper should explicitly reframe this as "minimal-data" or clarify the distinction.

- **Causal graph is conceptual rather than formal:** Figure 3 presents variables F^d and F^c ("discriminative" and "correlative" features) without formally defining them as random variables in a structural causal model. The narrative explanation of causal paths is intuitive but the formal causal inference claims (Total Direct Effect) lack proper grounding.

- **No failure mode analysis:** The paper claims DualPrompt "removes" the negative effects of co-occurrence, but provides no systematic analysis of remaining hallucination cases. Figure 5 shows per-class AP changes but does not quantify residual false positives or analyze which image types still suffer from hallucination.

## Nice-to-Haves

- **Cross-dataset transfer of co-occurrence:** Testing whether co-occurrence statistics from one dataset (e.g., COCO) transfer to another (e.g., VG-256) would validate whether the method captures universal patterns or dataset-specific biases.

- **Computational overhead quantification:** Dual prompts require encoding twice as many text inputs. While relatively minor, reporting inference time comparisons would be helpful for practical adoption.

- **Threshold sensitivity analysis:** Multi-label classification requires decision thresholds. Analyzing how DualPrompt affects optimal threshold selection and whether results are robust to threshold choices would strengthen the empirical evaluation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Softmax formulation concern (Harsh Critic):** The review criticizes using softmax for multi-label classification. However, this is how standard CLIP works—the paper inherits this formulation from CLIP itself and is not proposing to change it. This is not a novel critique of the method.

- **Title being "misleading" (Harsh Critic):** Calling "Unlocking the Power" misleading is stylistic nitpicking. The paper does unlock co-occurrence as a useful signal via prompt engineering.

- **Subset co-occurrence matrices shown (Harsh Critic):** The paper acknowledges Figure 1 shows subsets and provides complete results in Appendix B. This is addressed in the paper.

- **Missing comparison with other training-free methods (Spark Finder):** The paper compares with CLIP (vanilla), TagCLIP (training-free), DualCoOp (training-based), and TaICLIP. TagCLIP is the relevant SOTA training-free baseline, which is included.

- **Statistical significance testing (Spark Finder):** While valuable, single-run evaluation is standard in this research area. The paper follows common practice.

- **Prompt overhead as a weakness (Neutral Review):** The computational cost of dual prompts is mentioned but is minor (2x text encoding) and doesn't represent a fundamental limitation of the method.

## Novel Insights

The paper's most interesting finding is the double-edged nature of co-occurrence information: correlative prompts help recognize multiple objects by activating "correlative features," but simultaneously cause object hallucination when co-occurring objects exist without the target object. This trade-off is empirically demonstrated in Figure 2(a), which shows nearly half of classes improve with CoP while the other half degrade. The insight that simply adding discriminative prompt scores back mitigates hallucination—while intuitive in retrospect—is a practical contribution. However, the causal framework intended to explain *why* this works remains underdeveloped, leaving the core mechanism partially unexplained.

## Suggestions

1. **Provide ablations for hyperparameters:** Test multiple values of l (number of co-occurring labels: 1, 2, 3, 5) and λ (combination weight: 0.5, 1.0, 1.5, 2.0) to justify the current settings.

2. **Clarify the "training-free" scope:** Either reframe the contribution as "minimal-data adaptation" or clearly separate results into "zero external data" (ChatGPT co-occurrence) vs. "1% data" settings throughout the paper.

3. **Add a failure case analysis:** Report the remaining false positive rate after DualPrompt calibration and show example images where hallucination persists. This would honestly characterize the method's limitations.

4. **Simplify or formalize the causal derivation:** Either provide a rigorous derivation of Eq. 2 from Eq. 1 with explicit assumptions, or acknowledge that Eq. 2 is an empirical heuristic inspired by causal reasoning. The current claim of mathematical equivalence is not substantiated.

5. **Test a simpler baseline:** Compare DualPrompt against naive averaging of DiP and CoP probabilities (without the causal framing). If this performs similarly, it would suggest the causal framework is unnecessary; if DualPrompt is substantially better, it would validate the approach.