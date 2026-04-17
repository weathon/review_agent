# Functional-level Uncertainty Quantification for Calibrated Fine-tuning on LLMs

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Accurate uncertainty quantification in large language models (LLMs) is essential for providing credible confidence estimates over their outputs. However, fine-tuned LLMs often exhibit overconfidence in uncertain predictions, which stems from their limited ability to generalize with sparse data. Existing parameter efficient fine-tuning (PEFT) uncertainty quantification methods for LLMs focus on post fine-tuning stage, and thus fail to address the core issue: limited specialization of PEFT adapters to accurately capture task-specific input-output relationships. To address these limitations, we propose Functional-Level Uncertainty Quantification for Calibrated Fine-Tuning (UQ4CT), which captures and calibrates uncertainty over the space of functions that map input prompts to outputs. We implement UQ4CT during the fine-tuning stage via a mixture-of-experts framework that hierarchically decomposes the functional space. Empirically, UQ4CT achieves over 25% reduction in Expected Calibration Error (ECE) while preserving high accuracy across five benchmarks. Even under distribution shift, UQ4CT maintains superior ECE performance with high accuracy, showcasing improved generalizability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a new method for quantifying uncertainty in large language models. The proposed method focuses on quantifying functional uncertainty using a mixture of LoRA experts to all parameter matrices during finetuning. The model is deemed to be more confident for a particular generation given an input if it assigns high probability to a single expert. The overall functional uncertainty score computed across all model trainable components is incorporated as an auxiliary loss function during fine-tuning. Overall experimental results show that the proposed approach offers better uncertainty estimation and high performance retention using a Llama-3.1 8B model across several standard datasets.

### Strengths
- Really nice and creative approach for efficiently calibrating LLMs during fine-tuning.

### Weaknesses
- While I really liked the method, the experiments are limited. Only a single model size from a single family is tested. Furthermore, fine-tuning on the datasets you used in Table 2 or OBQA is not realistic (see my suggestions below). The current experimental setting does not convincingly show if the proposed method generalizes.

### Questions
1. You should report results using at least one more model family (e.g., Qwen3, OLMO 2 or similar) and model size (e.g. 3B or larger than 8B if you have enough compute).

2. You should use standard instruction tuning data (Alpaca or similar) to SFT your models instead of your current setting.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a fine-tuning approach for calibrating MixLoRA-adapted LLMs. The authors define functional uncertainty (FLU) as the predictive difference between the adapted model and its vanilla LLM counterpart. The paper then shows that this FLU can be linearly expressed by the MoE weights in the MixLoRA architecture. To calibrate this FLU score, the paper introduces a Brier-score-like calibration loss as the auxiliary loss.

### Strengths
1. The analysis presented in Fact 3.1 is insightful. The resulting equation, which linearly approximates the perturbation using $\Delta \alpha(h)$ and $g(h)$ (omitting super- and subscripts for simplicity), is surprising.
2. The experiments are basically comprehensive, including both in-distribution and out-of-distribution evaluations, and the reported calibration performance appears strong.
3. The methodology does not require modifications to the model's architecture, making it agnostic to the underlying vanilla LLM.

### Weaknesses
1. The derivation of uncertainty in Fact 3.1, defined as the difference between the fine-tuned (MixLoRA-adapted) LLM and the vanilla LLM, is counter-intuitive. The authors should clarify which model is the subject of the uncertainty analysis. If, as understood, it is the fine-tuned LLM, a more logical definition of uncertainty would seem to involve a comparison between the MixLoRA LLM and a perturbed version of itself, rather than a comparison against the vanilla LLM.

2. In Fact 3.1, $\Delta f(x)$ is expressed using both $\Delta \alpha(h)$ and $g(h)$. However, $g(h)$ is omitted from the subsequent definition of FLU. The authors should justify this omission.

3. The definition of FLU in Eq. (12), the calibration loss in Eq. (13), and Proposition (3.2) are highly counter-intuitive. Eq. (12) appears to model the confidence that "Expert $i$ should be involved in the computation," and the calibration loss (Eq. 13) reinforces this interpretation. This, however, represents confidence, not uncertainty. This contradiction is evident from Proposition (3.2): a perfect (100% accurate) fine-tuned LLM would have optimal MoE selection, implying zero uncertainty. In contrast, the proposed formulation would result in an FLU of 1.

4. The computational budget comparison with BLOB (Lines 399-402) is not straightforward. BLOB requires multiple forward passes, whereas the proposed method utilizes an MoE structure.
Given these fundamental architectural differences, a more rigorous and detailed complexity analysis is needed to make the comparison solid.

5. Figure 2 requires polishing, and its caption is misleading. Furthermore, when referenced in the text (e.g., Lines 273-274), there is a lack of proper guidance for the reader to interpret it.

6. The Introduction and Related Work sections could be strengthened by discussing other lines of LLM calibration, such as evidential methods [1], to provide a more complete background.

[1] Li, Yawei, et al. "Calibrating LLMs with Information-Theoretic Evidential Deep Learning." ICLR 2025.

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new approach to uncertainty estimation and calibration in large language models during fine-tuning. Instead of modeling uncertainty at the parameter level or requiring multiple stochastic forward passes, the authors introduce functional-level uncertainty , which quantifies uncertainty directly from the Mixture-of-Experts routing probabilities in the model.

### Strengths
1. The paper tackles a timely and interesting topic.
2. The proposed functional-level uncertainty formulation, leveraging MoE routing probabilities, is conceptually novel and computationally efficient.

### Weaknesses
1. Limited application scope. The method only applies to Mixture-of-Experts (MoE) architectures and to discriminative tasks with ground-truth labels. It is not directly applicable to generative or open-ended tasks, which limits its broader relevance in LLM research.
2. Theoretical justification is oversimplified. A more rigorous or relaxed analysis would strengthen the paper.
3. Comparable performance to BLoB (N=10). Although the proposed method is more efficient (single-pass), its calibration and accuracy improvements over BLoB(N=10) are modest, suggesting a trade-off between novelty and practical gains.

### Questions
1. Unclear token-level aggregation. In Eq. (12), the aggregation of routing probabilities across tokens is not well defined — it should specify whether it averages over all tokens, the [CLS] token, or the last token.
2. Missing ablation results in main text. The ablation study on the calibration loss weight (Lines 454–458) should be moved from the appendix to the main body for better visibility, since it demonstrates the effectiveness of the calibration component.

### Soundness
2

### Presentation
2

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
This paper proposes UQ4CT, which quantifies uncertainty at the functional level during fine-tuning using a Mixture-of-Experts (MoE) architecture with LoRA adapters. The method computes functional-level uncertainty (FLU) from router weights and uses a calibration loss to align FLU with predictive correctness. Experiments on five multiple-choice QA benchmarks show >25% ECE reduction while maintaining accuracy.

### Strengths
1. The paper presents a novel calibration mechanism where router weights naturally encode functional-level uncertainty.
2. Achieves calibration during training with only one forward pass at inference, unlike ensemble methods (8× slower) or BLoB(N=10) (10× slower).

### Weaknesses
1. All evaluated benchmarks (OBQA, ARC, BOOLQ, ClimateQA, MMLU) are classification tasks. This may limit the method's applicability, as multiple-choice QA is inherently more suitable for calibration compared to open-ended short-form or long-form QA. Can this approach be extended to open-ended QA datasets such as TriviaQA[1]?
2. There is a lack of comparison with alternative uncertainty estimation methods, including self-consistency[2] and training-based approaches[3].
3. Proposition 3.2 only holds at global optimum over true distribution. No analysis of: (a) convergence with non-convex MoE optimization, (b) finite sample effects, (c) conflicts between CE, Lb, and Lcal in Eq. 14. Need empirical verification that learned FLU ≈ P(correct).

[1]TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
[2]Self-Consistency Improves Chain of Thought Reasoning in Language Models
[3]Can AI Assistants Know What They Don't Know?

### Questions
1. Line 108: The model name appears inconsistently as "Llama3.1-8B" and "LLaMA3.1-8B." Please standardize the naming convention.
2. What is the performance of the untuned LLaMA3.1-8B on these tasks? This information is necessary to distinguish whether improvements are due to better task learning or improved calibration.
3. In Table~1, it would be clearer to present ECE and ACC on the same row under each dataset for better visual alignment and comparison.

### Soundness
2

### Presentation
2

### Contribution
2
