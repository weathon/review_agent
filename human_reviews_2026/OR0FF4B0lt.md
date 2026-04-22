# Harnessing Model Uncertainty for Adaptive Causal Debiasing in Visual Question Answering

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Visual Question Answering (VQA) models often exploit spurious correlations, hindering true multimodal reasoning. While causal inference offers principled debiasing methods, current approaches pair complex causal graphs with overly simplistic, static counterfactual interventions (e.g., feature subtraction). This limits effectiveness. We challenge this by proposing a novel framework synergistically integrating uncertainty estimation with causal counterfactual reasoning for robust VQA debiasing. This is the first work, to our knowledge, to leverage uncertainty within a causal VQA framework. We systematically explore uncertainty quantification techniques (entropy, prediction margin) to assess model confidence. This estimated uncertainty dynamically modulates the counterfactual intervention, allowing adaptive adjustment of biased information sources based on real-time confidence. This moves beyond rigid interventions. Furthermore, we introduce a tailored Curriculum Learning strategy that dynamically assesses sample difficulty using uncertainty-aware metrics, enhancing the adaptive mechanism. Our uncertainty-guided intervention module is architecture-agnostic, enabling integration into diverse VQA networks. This adaptive, uncertainty-aware approach offers a more flexible, robust, and theoretically grounded pathway towards mitigating VQA biases.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes an extension to the existing method for debiasing the VAQ models. VQA models have priors that are caused by the statistical answer distribution in training data. Many existing methods subtract a question-only baseline to correct the output logits. Currently, the question-only baseline is often a constant vector. This paper extends the constant to an adaptive estimate. The adaptation is done through a gating mechanism driven by a collection of uncertainty metrics. This enables a more flexible debiasing.

### Strengths
- Using per-example uncertainty to adapt prior static subtraction methods is a meaningful exploration in VQA debiasing. Introducing uncertainty as a control signal for adaptive debiasing is reasonable.
- The adaptive intervention is simple.

### Weaknesses
- Method is a feature ablation, not counterfactuals. Counterfactuals involve abduction, action, prediction (do calculus). The method uses only contrastive residual. So causal semantics are overstated.
- Various figures contradict the math. Figure 1 suggests cutting the question path, but the math replaces the vision and vision-question paths with a constant.
- Sloppy/incorrect math at times. The gating mechanism involves a learnable sigmoid function; I believe that the author incorrectly forces the hyperparameter $\gamma$ to be positive. Please check.
- Novelty is modest compared with the claim. The core contribution is a soft, uncertainty-gated generalisation of an existing subtractive debiasing scheme (CF-VQA). It is overclaimed as "the first to dynamically guide a causal intervention".

### Questions
- What is C? How come the same C was used for both Z_{V} and Z_{VQ}?

### Soundness
2

### Presentation
1

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
This paper introduces a novel framework for debiasing Visual Question Answering (VQA) models by integrating uncertainty estimation with causal counterfactual reasoning, addressing the limitations of static interventions in existing methods. The proposed Adaptive Uncertainty-Guided Intervention (AUGI) module dynamically modulates counterfactual interventions based on model uncertainty metrics such as predictive entropy, prediction margin, or ensemble disagreement, allowing for instance-specific adjustments to mitigate spurious correlations. Complementing this, an Uncertainty-Aware Curriculum Learning (UACL) strategy sequences training samples from low to high uncertainty, enhancing the model's robustness, while the architecture-agnostic design ensures broad applicability across VQA models.

### Strengths
Pros:
1. The paper introduces uncertainty estimation into causal VQA debiasing, enabling adaptive intervention that dynamically adjusts debiasing strength per instance, overcoming the rigidity of static methods, e.g., over-correcting in some cases and under-correcting in others.
2. The proposed UACL strategy sequences training from simple to hard samples, aligning learning progression with the adaptive mechanism and enhancing model robustness on challenging samples.
3. Extensive experiments demonstrate improvements in debiasing performance over baselines, validating the effectiveness of the uncertainty-guided approach.

### Weaknesses
Cons:
1. The UACL strategy depends on initial uncertainty from a warm-up model and uses static scheduling post-warm-up, which may introduce instability if uncertainty evolves significantly during training.
2. The bottom example in Figure 1 is misleading and does not reflect how Static Causal Intervention works. If "eating" is more common in the training set, Static Causal Intervention should reduce its probability on test cases, yet your figure shows "cooking" decreasing instead, which won't happen if you truly understood how Static Causal Intervention methods like CF-VQA works.  A correct illustration would show Static Causal Intervention over-correcting a true "eating" test case to "cooking", while your method correctly predicts "eating".

### Questions
Please address the above two questions I raised in the weaknesses. Your response will directly influence whether I raise or lower my final score.

### Soundness
2

### Presentation
2

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
This paper proposes a VQA debiasing framework that couples uncertainty estimation with counterfactual intervention. Uncertainty drives instance-level intervention strength, and an uncertainty-aware curriculum schedules training. The design is backbone-agnostic and targets robust multimodal reasoning. The claims include introducing uncertainty into causal VQA for adaptive intervention (AUGI), an uncertainty-aware curriculum (UACL), and broad empirical gains.

### Strengths
1. Clear motivation that fixed, static interventions fail to match per-example bias intensity.
2. Using uncertainty as the control signal improves adaptivity and aligns with robust learning principles.
3. Learnable gating with α, consistency regularization, and training objectives form a coherent end-to-end procedure.

### Weaknesses
1. Evaluation leans on older VQA backbones and lacks systematic tests on stronger Transformer-based or recent VLM architectures.
2. Computational overhead remains unclear, especially with ensemble-style uncertainty and multi-head designs. No FLOPs, memory, or latency analysis.
3. UACL orders samples by early uncertainty, which may interact with dataset bias. Sensitivity and robustness analyses are limited.
4. Edge behavior when α approaches its limits could suppress the debiasing signal; theoretical and empirical analysis is thin.

### Questions
1.How much training and inference overhead vs. CF-VQA in wall-clock time, memory, and latency, and what is the incremental cost attributable to ensembles or extra heads?
2. Does the plug-in work with modern VLMs such as BLIP/BLIP-2 or LXMERT, and do trends hold on those backbones?

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
1. The paper proposes a novel uncertainty-driven causal debiasing framework for Visual Question Answering (VQA), integrating uncertainty estimation with counterfactual reasoning to better assess model confidence and adapt intervention strength.

2. The approach dynamically evaluates sample difficulty using uncertainty-aware metrics, forming an adaptive curriculum learning strategy that enhances training progression from easy to hard samples.

3. Compared with baseline methods (e.g., CF-VQA), the proposed uncertainty-guided framework achieves superior performance on both VQA-CP v2 and VQA v2 datasets, demonstrating improved robustness and generalization.

### Strengths
1. The paper proposes a novel uncertainty-driven causal debiasing framework for Visual Question Answering (VQA), integrating uncertainty estimation with counterfactual reasoning to better assess model confidence and adapt intervention strength.

2. The approach dynamically evaluates sample difficulty using uncertainty-aware metrics, forming an adaptive curriculum learning strategy that enhances training progression from easy to hard samples.

3. Compared with baseline methods (e.g., CF-VQA), the proposed uncertainty-guided framework achieves superior performance on both VQA-CP v2 and VQA v2 datasets, demonstrating improved robustness and generalization.

### Weaknesses
1. The paper does not clearly describe the underlying model architecture. A brief background subsection should be added to explain the baseline model pipeline to help readers understand how the model is trained. In addition, at line 211,the authors should explicitly cite CF-VQA (Niu et al., 2021) when it is mentioned as the “baseline our work builds upon.” In addition, the term should be consistently written as CF-VQA, not CFVQA, to maintain clarity and consistency throughout the paper.

2. During writing, ensure references to Table 4 and Table 6 are explicitly mentioned where relevant to the discussion.

3. In Figure 2, the purple arrow line representing UACL is visually unclear and should be enhanced for better readability.

4. In LaTeX, the left double quotation mark should be written as two backticks (``) rather than a single quotation mark ("). Otherwise, the rendered text will display incorrect quotation marks (e.g., ”...”).

### Questions
1. For current VQA tasks, large vision-language models (e.g., LLaVA, Qwen-VL) have become dominant. Could the proposed uncertainty-driven framework also be extended or fine-tuned within such large-scale VLM architectures?

### Soundness
3

### Presentation
1

### Contribution
3
