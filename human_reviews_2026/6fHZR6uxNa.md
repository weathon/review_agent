# Evaluating and Explaining Prompt Sensitivity of LLMs Using Interactions

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
The remarkable capabilities of large language models (LLMs) are often undermined by their instability. That is, LLMs are sensitive to prompts, as even subtle and semantically irrelevant changes in prompts can cause dramatic fluctuations in performance, a phenomenon known as prompt sensitivity. Previous studies typically evaluate prompt sensitivity by comparing the LLM's final outputs when prompts change. However, such coarse-grained metrics fail to explain the internal reasons for prompt sensitivity. 
In this paper, we introduce the game-theoretic interaction framework as a fine-grained tool to analyze prompt sensitivity of LLMs. Specifically, we disentangle the output score of the LLM into a set of interactions. Each interaction represents a nonlinear relationship associated with a combination of input variables. We discover that subtle changes to prompts can trigger significant instability in interactions, even when the final outputs of the LLM remain the same. To this end, we propose an Interaction-based Prompt Sensitivity (IPS) metric by quantifying changes in interactions when we introduce subtle changes to prompts. We apply the proposed IPS metric to 50 open-source LLMs and uncover four factors that reduce the prompt sensitivity of LLMs, including supervised fine-tuning, increased model scales, dense architectures, and few-shot learning. More crucially, we discover a common mechanism by which these four factors reduce prompt sensitivity: all these four factors tend to reduce the prompt sensitivity of low-order interactions (*i.e.*, interactions involving few input variables).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper investigates prompt sensitivity in large language models using a novel fine-grained analytical framework that combines prompt masking with Harsanyi interaction decomposition. Leveraging this framework, the authors introduce an Interaction‑based Prompt Sensitivity (IPS) metric and conduct a comprehensive benchmark across 50 open‑source LLMs. Their analysis uncovers four effective strategies for mitigating prompt sensitivity; all of these approaches consistently reduce low‑order interactions, thereby improving model robustness.

### Strengths
1. The proposed IPS metric as an interpretability tool provides a fine-grained measurement of prompt sensitivity for LLM. This newly proposed framework helps in further research on LLM's stability.
2. Admitted that the four discovered methods are already common sense, the observation that they jointly reduce the low-order interactions further enhanced the credibility of the proposed framework. 
3. The experiment section, together with the results presented in Section 3.3, is well-written and well-structured.

### Weaknesses
1. The sensitivity results discussed in the paper can be categorized into capitalization and symbol addition/modification. It would be interesting to see a sensitivity comparison between altering tokens (e.g., from answers to ANSWERS)  vs adding tokens (e.g., using two colons instead of one)
2. The interaction-based decomposition proposed in this paper requires mutual exclusiveness, i.e., we assume the model output is the perplexity of predicting one specific word. A discussion on how to control the prediction mass to meet this assumption, or how to generalize the model into non-exclusive answers, will greatly improve the practical usefulness of this work

**Non-desicive Suggestions**:
1. Line 239, please double-check if the description matches with Fig. 3. `The top row shows examples where the LLM's output remains the same` 
2. I suggest introducing a bit more on the Harsanyi model implemented in this work. I am a bit confused if we need to train the surrogate model $\phi(x)$ or not, and need to look into the supplementary material together with Li & Zhang (2023)'s GitHub repo

### Questions
1. As the work is masked-based, what will happen if a token also contains information that should not be masked? e.g. the edging token contains neighboring words.

### Soundness
3

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
5

### Summary
This paper introduces a novel, interaction-based framework for fine-grained analysis of prompt sensitivity in LLMs. The authors propose the IPS metric, which quantifies sensitivity by measuring the instability of internal interaction patterns under semantically-irrelevant prompt changes. Through extensive experiments on 50 models, they identify four factors that reduce sensitivity and uncover the counter-intuitive mechanism that these factors primarily stabilize already-robust low-order interactions.

### Strengths
- The application of the game-theoretic interaction framework to dissect prompt sensitivity offers a principled look into the model's internal reasoning shifts.

- The experiments across 50 LLMs provide generalizable findings on the factors influencing prompt stability.

### Weaknesses
- The computational cost of the interaction analysis, which appears to require evaluating $2^n$ masked inputs, is a major concern. Clarification on how salient interactions can be efficiently identified and how the method scales to longer contexts is critically needed.

- The study only tests template modifications. Its significance is limited without testing paraphrases, a more realistic threat. Relevant work like [1] should be discussed.

- The proposed framework is interesting but feels somewhat preliminary. It successfully quantifies prompt sensitivity, but does not further investigate the underlying mechanisms or explore how to mitigate it. It comes across as using one problem to describe another.

[1] On the Worst Prompt Performance of Large Language Models

### Questions
Please refer to Weaknesses.

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
2

### Summary
This paper addresses the well-known problem of prompt sensitivity in LLMs. The authors propose a new fine-grained metric, Interaction-based Prompt Sensitivity , which is based on a game-theoretic decomposition of the model's output score into a set of input variable interactions.This metric is used to conduct a large-scale study on 50 open-source LLMs. The study finds that supervised fine-tuning, larger model scale, dense architectures, and few-shot learning consistently reduce sensitivity. Crucially, the authors identify a common underlying mechanism: these factors primarily stabilize low-order interactions, while high-order interactions remain highly unstable.

### Strengths
1.  A key strength of the paper is its large-scale empirical study, evaluating 50 open-source LLMs from six different model families. This provides a robust and broad empirical foundation for the paper's claims, which is a significant undertaking.
2.  The identification of four concrete factors (supervised fine-tuning, model scale, dense-vs-MoE architecture, and few-shot learning) that consistently reduce prompt sensitivity is a valuable and actionable contribution for the community .
3.  The paper's main conceptual contribution is shifting the analysis of sensitivity from coarse-grained, output-based metrics to the model's fine-grained internal logic. The discovery of "latent instability"—where interaction patterns are highly unstable even when the final output remains the same —is an important finding that highlights the limitations of traditional evaluation.
4.  The discovery of a common underlying mechanism for these stability improvements is a novel insight. The finding that these diverse factors all reduce sensitivity primarily by stabilizing low-order interactions, rather than the more volatile high-order ones , is counter-intuitive and significant.

### Weaknesses
1.  The conceptual novelty of the proposed *method* is limited. The game-theoretic interaction framework is not new but is imported from prior work in explain ability . The paper's contribution is therefore a large-scale *application* and *analysis*, not the development of a new fundamental method.
2.  The paper's primary weakness is the severe scalability limitation of the chosen methodology. The $O(2^n)$ computational cost is prohibitive and restricts the entire analysis to short-input MCQ datasets. This makes it difficult to assess whether the findings generalize to more common and practical long-context NLP tasks. The proposed solutions in Appendix G are speculative and not empirically validated .
3.  The paper fails to sufficiently justify the necessity of this specific, and computationally expensive, interaction framework over simpler alternatives. It is unclear why this method is superior to scalable proxy metrics for internal stability (e.g., measuring the L2 distance or cosine similarity of hidden state activations), which are not explored or compared against.
4.  The proposed IPS metric relies on a threshold $\tau$ (set to 0.1) to distinguish "salient" interactions from "noise patterns". While a sensitivity analysis is provided in the appendix, the selection and impact of this critical hyperparameter warrant a more central discussion, as it directly influences the final quantitative results.

### Questions
1.  Regarding Weakness #2 & #3: Given the severe $O(2^n)$ computational cost, have the authors considered comparing their IPS metric against simpler, scalable proxy metrics? For instance, would measuring the cosine similarity of final-layer hidden states under prompt perturbations (T vs. $\hat{T}$) yield similar conclusions regarding the four factors? Such a baseline would be computationally trivial and would help justify the necessity of the complex interaction framework.
2.  Regarding Generalizability: The findings are derived exclusively from MCQ tasks . Do the authors have any evidence (preliminary or theoretical) to suggest that the core finding—that stability is governed by low-order interactions—would generalize to open-ended generation tasks, which arguably rely on more complex, higher-order reasoning?
3.  Regarding MoE Models: The paper concludes that MoE models are generally more sensitive. However, it also notes the Mistral family as an exception, attributing this to a higher *activated* parameter count . This seems to conflate two of the paper's factors. Is the instability caused by the MoE architecture itself, or is this finding simply a proxy for Factor 2 (model scale), where MoE models are less stable because they typically activate fewer parameters? Could the authors please clarify this?

### Soundness
3

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
3

### Summary
This paper introduces a novel fine-grained metric, Interaction-based Prompt Sensitivity (IPS), to measure the robustness of Large Language Models (LLMs) through the lens of interaction patterns between input variables. The authors decompose the scalar output score for the ground-truth answer into a sparse set of AND/OR interactions among input words, grounded in the "universal matching" property and sparsity guarantees. IPS quantifies changes in salient interaction effects when comparing pairs of semantically equivalent prompt templates (T and T̂) that differ only in superficial formatting. Evaluating 50 open-source LLMs across two multiple-choice question (MCQ) benchmarks (ARC and MMLU), the authors identify four factors that reduce prompt sensitivity: (1) supervised fine-tuning, (2) increased model scale, (3) dense architectures relative to Mixture-of-Experts (MoE), and (4) few-shot learning. Notably, they demonstrate that these factors primarily stabilize low-order interactions while leaving high-order interactions vulnerable to prompt perturbations.

### Strengths
1.	Solid theoretical framing: Theorem 1 establishes the universal matching property with comprehensive proof in Appendix B, grounding the framework in established prior work (Ren et al., 2023; Li & Zhang, 2023). The sparsity property provides additional mathematical guarantees for the faithfulness of interaction-based explanations.
2.	Insightful findings: The transition from output-level to interaction-level sensitivity analysis represents a meaningful contribution. The empirical demonstration that interaction patterns exhibit substantial instability (Figure 3) even when final predictions remain unchanged offers insights that coarse-grained metrics inherently fail to capture.
3.	Comprehensive evaluations: The study encompasses 50 LLMs across six model families evaluated on two datasets with consistent methodological controls. The factorial analysis of influences (fine-tuning, scale, architecture, and prompting strategy) demonstrates careful experimental design.

### Weaknesses
1,	Task scope is restricted: The evaluation is exclusively restricted to MCQ datasets (ARC and MMLU) with highly structured template variations. Although justified in Appendix E.1, this limitation fundamentally undermines claims of broad applicability. The authors acknowledge applicability to open-ended generation tasks but provide no empirical validation on such domains. Real-world prompt sensitivity often manifests through semantic reformulations (e.g., paraphrases, instruction reordering) rather than superficial formatting changes, yet these variations remain untested. 
2,	Computational feasibility and selection bias: The theoretical framework refers to 2^n masked variants, but in practical approximations were used in the actual experiment. proposed solutions (selecting informative words, phrase-level aggregation, approximation) are not empirically validated, raising concerns about selection bias and reproducibility of IPS.
3,	Causal interpretation and confounds: Dense vs. MoE and base vs. instruct comparisons may be confounded by differences in training data, RLHF/DPO, activated parameter counts, quantization precision, etc. The largest models are quantized (Int4) while most others are FP16—this can affect internal representations and sensitivity, potentially biasing the “larger is more stable” conclusion.

### Questions
1.	Can you provide any empirical evidence (e.g. preliminary results on a small pilot) that the low-order stabilization finding holds on non-MCQ tasks?
2.	How does the method scale to practical long-context scenarios (document understanding, long-form summarization, multi-turn dialogue) where the number of input variables n substantially exceeds 50? Have computational approximations been empirically validated on such tasks?
3.	The evaluation exclusively tests superficial formatting variations (letter case, punctuation). Have you evaluated robustness to semantic paraphrases (e.g., "Which animal does NOT lay eggs?" vs. "Which creature is not an egg-layer?") or instruction reordering? Would the four identified stabilizing factors exhibit consistent effects under such variations?

### Soundness
2

### Presentation
2

### Contribution
2
