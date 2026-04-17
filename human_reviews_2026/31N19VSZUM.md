# Mitigating Over-Refusal in Adversarial Tuning via Subspace-guided Sample Selection

- Decision: Reject
- Scores: 8, 4, 4

## Abstract
As the adoption of large language models (LLMs) increases, their vulnerability to jailbreaks poses a significant concern. Adversarial tuning offers an effective means of enabling LLMs to resist jailbreak prompts, but it inevitably introduces the problem of over-refusal, where benign queries are mistakenly rejected, thereby comprising the model utility. To address the limitation, we propose the Soft Adversarial Tuning (SAT) framework, which selects “soft samples” that balance robustness and over-refusal for adversarial tuning. Specifically, SAT decomposes the model’s hidden states into two behavioral subspaces via representation engineering: one for producing robust responses to malicious queries and another for avoiding over-refusal on benign queries. By projecting the gradients of candidate adversarial-tuning samples onto these subspaces, we quantify each sample’s influence on jailbreak defense and over-refusal. We then select ”soft samples” that exert strong influence in the robustness subspace while having minimal effect in the over-refusal subspace for soft adversarial tuning. We evaluate SAT with six existing defense methods across different settings. Experimental results show that SAT consistently outperforms these methods, reducing the over-refusal rate by more than 22%, while maintaining an attack success rate below 2.8% against five representative jailbreak attacks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses over-refusal in LLM adversarial tuning, which degrades utility. It proposes SAT, a subspace-guided soft sample selection framework. Key results: It reduces over-refusal by over 22%, maintains ASR below 2.8% across 5 attacks, and preserves utility (e.g., AlpacaEval winrate 54.61% vs SafeLoRA’s 47.37%).

### Strengths
1. Precise subspace decomposition: It splits model activation space into robustness and over-refusal subspaces, using gradient projection to quantify sample impact (e.g., projecting candidate gradients to select samples with strong robustness and weak over-refusal influence), enabling targeted tuning. 
2. Extensive baseline comparisons: It compares 6 SOTA defenses (PPL, Self-Reminder, etc.) across 3 models (Vicuna, Llama2, Dolphin), showing SAT’s superiority (e.g., Vicuna’s SAT reduces ASR to 2.8% vs Self-Examination’s 10%).
 3. Utility preservation: Unlike Goal Priority (which harms GSM8K accuracy), SAT maintains or improves utility (e.g., Llama2’s GSM8K accuracy 53.20% vs SafeLoRA’s 52.20%).

### Weaknesses
1. Limited sample generation sources: It relies on GCG to generate candidate adversarial samples; other attack methods (e.g., AutoDAN, TAP) are untested—using diverse attack-generated samples could enhance sample selection generality. 
2. Lack of long-term evaluation: It does not test over-refusal drift after extended fine-tuning (e.g., 10+ epochs); evaluating durability would confirm long-term effectiveness. 
3. Narrow domain coverage: It only tests semantic QA and math reasoning; high-stakes domains (e.g., healthcare, finance) are unexamined—adding domain-specific tests would improve real-world relevance.

### Questions
Please refer to the weaknesses above.

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
4

### Summary
This paper tackles a common side-effect of adversarial fine-tuning for LLM safety: over-refusal of benign queries. The authors propose Soft Adversarial Tuning (SAT), a data-selection framework that: (i) constructs two behavior subspaces from hidden activations, one for jailbreak robustness and one for over-refusal avoidance, using difference-in-means “steering” vectors; (ii) projects per-sample gradients onto these subspaces to estimate how each candidate training example would shift behavior; and (iii) scores and selects “soft” samples that push strongly along the robustness direction while minimally affecting the over-refusal direction. The selected set is then used for LoRA fine-tuning. Evaluated on Vicuna-7B, Llama-2-7B-Chat, and Dolphin-7B against five jailbreak attacks, SAT is reported to lower attack success rates and reduce refusal on safe inputs relative to several baselines, with an ablation indicating the explicit over-refusal penalty in the scoring term is important.

### Strengths
1.	The paper centers on over-refusal on LLM, then proposes a specific mechanism (dual subspaces, gradient projections, sample scoring) rather than an open-ended recipe. The subspace construction via contrastive positive/negative pairs and difference-in-means vectors is well-defined and easy to implement. 
2.	SAT is a pre-fine-tuning data curation step that could be bolted onto diverse PEFT schemes; the actual model update uses standard LoRA, keeping the method pragmatic for practitioners. 
3.	On three 7B chat models and five representative jailbreaks, SAT often shows lower ASR and lower harmful scores than baselines such as Self-Examination, ICD, Retokenization, and Paraphrase; safe-set refusal is also reduced on XSTest. The tables make these cross-method comparisons explicit.

### Weaknesses
1.	The approach assumes a linear correspondence between activation-space projections at a single layer and subsequent parameter-space updates, then uses one-dimensional directions per behavior to rank samples. There is no sensitivity analysis across layers, pooling choices, or multi-dimensional subspaces, and no theoretical or empirical justification that a single direction captures over-refusal vs robustness without leakage between them. At minimum, a study varying layer index, using multi-basis PCA/LDA subspaces, and checking orthogonality would be needed. 
2.	Since SAT is a sample selection method, it should be compared against strong selection baselines: e.g., gradient-norm filtering, loss-based filtering, influence-function/DShapley-style data valuation, or even simple intermediate-iteration only heuristics. Current baselines are mostly defense mechanisms at inference or full training, not data curation approaches, so the incremental value of subspace-guided selection is unclear. 
3.	The pipeline generates 500 optimized candidates per seed over 400 seeds, then projects gradients for scoring yet finally keeps only 500 samples. There is no accounting of attack-generation time, gradient-projection overhead, or overall wall-clock vs baselines. For a method pitched as “efficient pre-selection,” this omission is important.

### Questions
1.	Table 1 mixes ASR and harmful scores per cell but has a few oddities/typos (e.g., ICA, duplicated averages formatting) and the narrative highlights dramatic gains without confidence intervals, seed variability, or per-category breakdowns. The claim reduce over-refusal by more than 22% while keeping ASR below 2.8% would benefit from statistical tests across runs and categories. 
2.	One figure panel includes an error message (index out of bounds / lda (Failed)), which undermines polish and raises questions about the robustness of the dimensionality-reduction diagnostics. Implementation details crucial to reproducing subspaces are not fully specified.

### Soundness
2

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
This paper addresses the "over-refusal" problem in LLMs. This is when models reject benign queries after adversarial tuning. The authors propose the Soft Adversarial Tuning (SAT) framework. SAT uses representation engineering to define two subspaces: a "robustness subspace" and an "over-refusal subspace". It then uses gradient projections to select "soft samples". These samples have a strong influence on robustness but a minimal effect on over-refusal. The main contribution is this automatic sample selection mechanism.

### Strengths
1. The paper is well-written and easy to understand. The figures are also clear and well-done.
2. The paper's method is highly novel. The experimental results also show that the proposed method is effective.

### Weaknesses
1. The quotation marks for "soft samples" in the abstract are not correct.

2. The paper uses mean-pooled hidden states from layer $l$. Many methods use the hidden state of the last token. The authors should explain why they chose mean-pooling and discuss how this choice impacts the results.

3. Please provide a detailed explanation for the rationale behind Equation (8). Specifically, why is the absolute value of $p_2$ used? The paper describes $v_2$ as the direction vector pointing from the over-refusal space to the normal response space. Does this imply that more robust samples also tend to cause over-refusal, thereby resulting in a negative $p_2$ value?

4. Equation (5) is unclear. Please specify what the gradient is taken with respect to.

5. The paper has small formatting problems. For example, lines 316 and 297 are missing spaces after the colons. Many citation formats are also incorrect.

6. A formatting error: The caption for a table should be placed above it.

7. It would be beneficial to add results on the latest Qwen models.

### Questions
These are all in the 'Weaknesses' section above.

### Soundness
3

### Presentation
2

### Contribution
3
