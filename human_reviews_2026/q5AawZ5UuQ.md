# Emergent Misalignment is Easy, Narrow Misalignment is Hard

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Finetuning large language models on narrowly harmful datasets can cause them to become emergently misaligned, giving stereotypically `evil' responses across diverse unrelated settings. Concerningly, a pre-registered survey of experts failed to predict this result, highlighting our poor understanding of the inductive biases governing learning and generalisation in LLMs. We use emergent misalignment (EM) as a case study to investigate these inductive biases, and find that although models can learn the narrow dataset task, the general solution is measurably more stable and more efficient. To establish this, we first demonstrate that EM is a robust phenomena by introducing new datasets which induce misalignment more consistently and coherently than prior work. We show that different EM finetunes converge to the same linear representation of general misalignment, which can be used to mediate misaligned behaviour. However, a linear representation of the narrow solution also exists, and can be learned by introducing a KL divergence loss. Comparing these representations reveals that general misalignment achieves lower loss, is more robust to perturbations, and is more influential in the pre-training distribution. This work isolates a concrete representation of general misalignment for monitoring and mitigation. More broadly, it offers a detailed case study and metrics for understanding how inductive biases shape generalisation in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper investigates the phenomenon of emergent misalignment (EM) in LLMs, where finetuning on a narrow set of harmful examples leads to broad, stereotypical misaligned behavior across diverse, unrelated settings. The central contribution is an investigation into the inductive biases that favor this general misaligned solution over a narrow, task-specific one. The authors find that the general solution is demonstrably more efficient, stable, and influential in the pretraining distribution compared to the narrow solution, which they successfully enforce using a KL divergence loss. Overall, the paper is highly interesting, and its findings provide significant insights into the generalization and alignment properties of LLMs.

### Strengths
The originality of this work stems from its novel framing of emergent misalignment as a problem of competing solutions—general versus narrow—for a localized finetuning task. The work isolates and compares concrete linear representations for both the general and narrow solutions within the model's activation space, which is a powerful and highly original methodological approach to studying inductive biases. The quality of the empirical evidence is high, demonstrating the robustness of the EM phenomenon across different datasets, model scales, and architectures. The analysis comparing the efficiency, stability, and influence of the two solution types provides a clear, quantitative explanation for the observed inductive bias, which significantly advances our understanding of generalization in LLMs. The clarity of the core findings is strong. The significance is substantial, as it offers a foundational case study for monitoring and potentially mitigating future, more complex forms of emergent misalignment.

### Weaknesses
The analysis of the narrow misalignment solution, which is achieved via the introduction of a KL divergence constraint, introduces both practical and theoretical limitations. The success of this method critically relies on the assumption that the base pre-trained model is perfectly aligned. If the base model contains subtle, unknown, or slight misalignments, the KL penalty could inadvertently lock in these undesirable biases, thereby preventing the realization of a truly safe and narrowly-focused behavior. Furthermore, the practical deployment of this technique is hampered by the significant hyperparameter tuning challenge of balancing the KL divergence weight $\beta$ against the task-specific loss. The paper successfully demonstrates the existence of the narrow solution but does not offer a theoretically robust framework or an automatically adaptable mechanism for setting the $\beta$ value across different tasks or models.

A second major weakness concerns the explanatory power of the linear representation analysis. While the finding that both general and narrow misalignments can be captured by linear directions is extremely useful, the core arguments about the inductive bias being driven by the linear solution’s efficiency and stability are based on an approximation of a fundamentally non-linear process. The comparison of linear projections, while insightful, ultimately explains what happened in the model's feature space, but it does not fully uncover the deeper non-linear mechanisms responsible for organizing the pretraining feature space in a way that makes the linear general solution globally optimal or locally most efficient for the fine-tuning loss.

Finally, the overall clarity and presentation of the paper could be significantly improved. The prose is occasionally non-fluent, requiring careful editing for an ICLR submission. Specifically, Figure 3, which currently appears to be a screenshot or a less polished visualization, requires professional reformatting to better present the complex comparison data it contains, which would enhance the overall readability of the results.

### Questions
Could the authors elaborate on the stability of the narrow misalignment solution (NM) with respect to different values of the KL divergence weight $\beta$? Specifically, is there a phase transition where a small change in $\beta$ causes the model to jump from an NM solution to an EM solution, and what does this imply for the practical robustness of the NM solution?

The paper argues the general solution is more influential in the pre-training distribution. Can the authors provide a more detailed information-theoretic or geometric explanation as to why the pre-training objective would favor the encoding of a general concept that is so easily activated by a narrow, harmful dataset? What specific properties of the pre-training data or objective create this specific inductive bias?

Please ensure that all figures, particularly Figure 3, are redrawn and polished to meet the professional standards of a conference submission, ensuring a more fluid reading experience. The overall prose requires a pass for improved fluency and clarity.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper systematically investigates the phenomenon of EM that arises in large language models after fine-tuning. By constructing new datasets—including cases such as erroneous medical advice and risky financial recommendations—the authors validate the robustness of this phenomenon and reproduce the effect across models of different scales and families. Furthermore, the paper reveals that EM can be represented by a linear direction in the model’s activation space, which remains highly consistent across different fine-tuning experiments. This direction can be used both to induce misaligned behaviors and to eliminate them through ablation. In addition, the authors introduce efficiency and stability metrics to explain why models tend to favor general solutions, showing that such solutions achieve lower training loss under the same parameter norm and exhibit greater robustness to perturbations.

### Strengths
1. The paper systematically and comprehensively reproduces the EM phenomenon, confirming its generality and revealing that generic misalignment exhibits a linear representation.

2. It proposes innovative evaluation metrics for the EM phenomenon and provides a concrete explanation for the model’s preference for general solutions.

### Weaknesses
1. The experimental analysis on LoRA and SFT is thorough and insightful; however, the presence of the EM phenomenon in broader alignment algorithms still requires further empirical validation. I am particularly curious whether similar EM behaviors might also emerge in algorithms such as RLHF or DPO.

2. The distinction between narrow misalignment and general misalignment needs to be clarified. The definition of the narrow domain should be more explicit, and the core differences between the narrow and broader domains are not clearly articulated. Moreover, the paper lacks more quantifiable methods to characterize or delineate the distributional or data features that distinguish “general” from “narrow” misalignment.

### Questions
Please refer to the relevant points in the Weaknesses section. If the authors can provide clarification and improvements, I would be very happy to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies why narrow, harmful fine-tuning tasks can induce broad, “stereotypically evil” behaviour in LLMs (“emergent misalignment,” EM). It (i) introduces synthetic medical/finance/sports datasets and shows EM is robust/coherent; (ii) identifies a linear “misalignment direction” in mid layers that transfers across finetunes and can both elicit and ablate EM; (iii) trains narrow misalignment by adding a KL penalty outside the dataset’s domain; and (iv) argues the general (EM) solution is preferred because it’s more efficient (lower loss at smaller parameter norms) and stable (more robust to perturbations), and is more influential on pre-training data.

### Strengths
1. Clear empirical contributions & model organisms. New datasets produce higher-coherency EM than prior “insecure code,” with a clean evaluation protocol and judge prompts described in appendices. 

2. Mechanism that transfers. A mid-layer mean-diff “misalignment” vector reliably induces EM when added and significantly reduces EM when ablated, including across different finetunes, useful for monitoring/mitigation. 

3. Narrow vs. general comparison is well-posed. KL-regularised SFT learns narrow misalignment at similar in-domain rates while avoiding OOD misalignment—an informative counterfactual baseline. 

4. Well-designed metrics. Efficiency (loss/‖θ‖²) and stability (loss under orthogonal noise) concretise why general solutions are favored during optimisation; the paper ties these to pre-training significance via KL on web data. 

13475_Emergent_Misalignment_is

### Weaknesses
1. Heavy reliance on LLM judges. Safety/coherence and domain-harm labels depend on GPT-4o; while prompts are provided, this leaves open judge drift and bias. A subset of human-rated validation or cross-judge checks (e.g., different families) would strengthen claims.
2. “Why” remains partly speculative. The link from efficiency/stability to pre-training influence is suggestive but not causal; further ablations (e.g., controlling corpus slices) would bolster the pre-training hypothesis.
3. Scope of models. Results span multiple families/sizes, but many key demonstrations use specific Qwen layers (e.g., ~layer 24). Clarify layer selection, variability across depths, and families with different depth/architecture.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3
