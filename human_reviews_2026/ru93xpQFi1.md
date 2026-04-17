# SFT Doesn’t Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper's premise is the observation that SFT does not always significantly harm a model's general capabilities ; this issue can, in fact, be largely mitigated by using a smaller learning rate. A theoretical analysis further suggests this performance degradation is linked to the adverse impact of low-probability tokens. Based on this, the authors propose TALR, a method that suppresses the decline in general capabilities by adaptively reducing the loss weights of these "hard" tokens.

Experiments show that TALR outperforms existing methods in balancing domain-specific and general performance , though it cannot fully resolve the trade-off. The paper's final practical guideline for SFT is to prioritize using a small learning rate and to supplement with TALR when a better balance is desired.

### Strengths
1. Experiments across multiple models (Qwen and Gemma series) and diverse datasets (MedCalc, ESCI, MetaMathQA) consistently demonstrate the effectiveness of using a small learning rate.
2. The paper moves beyond mere empirical observation to provide a theoretical foundation. The perspective linking SFT dynamics to information compression and code-length change is novel. This theory successfully explains why small learning rates are beneficial and why label-only SFT behaves differently.
3. The proposed TALR method is not ad-hoc; it is derived from the theoretical analysis, giving it a strong theoretical grounding.

### Weaknesses
1. The primary recommendation—using a small learning rate—is a standard and well-known heuristic in continual learning to mitigate catastrophic forgetting. As such, the method itself lacks significant innovation.
2. As the authors acknowledge, TALR offers negligible benefits over standard SFT when the recommended small learning rate is used. Its utility is confined to a niche scenario where one *must* use a large learning rate (to pursue peak in-domain performance) and needs to balance the resulting degradation in general capabilities.
3. The dynamic heuristic for setting the $\tau$ hyperparameter (using the batch median loss) is not sufficiently validated. The ablation study only compares it to a single fixed value ($\tau=1$) rather than a proper hyperparameter sweep or comparison with other dynamic strategies.
4. The experiment in Figure 3, which is used to motivate the "curriculum-like" behavior, does not match the actual TALR algorithm. The experiment uses gradient clipping (truncation to zero) for hard tokens, whereas TALR only uses down-weighting. Therefore, this analysis does not represent the true dynamics of TALR. Furthermore, the aggregate statistic in Figure 3c (the rising fraction of p>0.2 tokens) fails to prove that the hardest tokens (e.g., p=0.05) are eventually learned; it could simply reflect that medium-difficulty tokens have become easier.
5. All experiments are conducted on relatively small-scale models (8B or less). The paper lacks validation on whether its core conclusions generalize to current state-of-the-art large models (e.g., 70B+) or MoE architectures, whose fine-tuning dynamics may differ significantly.

### Questions
1. TALR balances performance by down-weighting hard tokens. As noted in the weaknesses, it is not well-argued whether these hard tokens are eventually learned. Could the authors provide an analysis that tracks the probability of a few specific, hard tokens (like the example in Fig 3a) throughout the training process to demonstrate their learning dynamics?
2. The paper lacks an analysis of the actual training dynamics of TALR, instead showing a related "extreme experiment" (Fig 3c). Could the authors provide an analysis (e.g., of token probabilities or loss weights) based on the true TALR algorithm as presented in Algorithm 1?
3. The paper's primary recommendation is to use a small learning rate, a setting where TALR offers minimal advantages. The utility of TALR hinges on a scenario where one must use a large LR to achieve peak domain performance. Could the authors provide a clearer experimental case where the domain performance of the small-LR SFT significantly lags behind that of the large-LR SFT, thereby strengthening the practical case for TALR?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Based on experiments showing that a smaller learning rate can partially mitigate catastrophic forgetting, this paper validates the conclusion through theoretical analysis. Furthermore, building on both theoretical and experimental findings, the authors propose the TALR method. By limiting the magnitude of updates for low-probability tokens, the approach alleviates catastrophic forgetting to some extent.

### Strengths
1. This paper is well-written and really easy to read and follow.

2. The method proposed in this paper is derived from an experimental finding and theoretical analysis, with a well-justified motivation.

### Weaknesses
1. I think the paper's core motivation, using a smaller learning rate to mitigate catastrophic forgetting, is clear. However, a significant issue is that this finding is not novel, as it is already a widely recognized common-sense approach. The paper excessively emphasizes this observation, dedicating substantial space to elaborating the advantages of a low learning rate, while its actual contribution, the proposed TALR method, does not receive sufficient attention or elaboration. This may lead readers to perceive the paper's main contribution as the experimental and theoretical analysis of the low learning rate's advantages, thereby overlooking the more important innovation.

2. Building on the previous point, if the main contribution of this paper is merely a more thorough experimental and theoretical analysis of the low learning rate, I do not believe the contribution is sufficient, as the experimental results do not present any particularly striking or novel conclusions.

3. Regarding Figure 2, could you provide the results in a table format? From the figure, it appears that TALR does not show a clear advantage under many settings, which suggests that the TALR method may not have a significant effect.

4. Regarding the extensive discussion on low learning rates in the paper, I believe the experimental analysis is not sufficiently comprehensive. In both the preliminary analysis and the main experiments, the authors only used three different learning rate settings (with even just two shown in Figure 2). Since TALR is derived from this experimental finding, I think the experiments should be more thorough and robust—relying on only three learning rate points is inadequate.

### Questions
What is the main contribution of this paper? Is it the experimental and theoretical analysis of the low learning rate, or is it the TALR method?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the common belief that domain-specific supervised fine-tuning (SFT) significantly harms the general capabilities of large language models (LLMs). Through empirical and theoretical analysis, the authors demonstrate that using a smaller learning rate can mitigate this performance degradation while maintaining strong domain-specific performance. They further propose a novel method, Token-Adaptive Loss Reweighting (TALR), which dynamically down-weights the loss of "hard" tokens during training to better preserve general capabilities.

### Strengths
1. The writing is experienced and clear, effectively motivating the problem and delivering its overall message.

2. The work revisits a critical topic—the trade-off between in-distribution and out-of-distribution performance during SFT. Its core finding, that smaller learning rates are a simple yet effective strategy, is a valuable and practical insight for practitioners.

3. The token-level analysis is a particular strength, yielding interesting findings. For instance, it shows that most tokens in SFT data are easy for the model, and performance bottlenecks are likely due to a sparse set of hard tokens.

4. The theoretical analysis, framed through an information-theoretic and compression-based lens, is interesting and provides a valid foundation to explain the empirical phenomena observed, such as why smaller learning rates help.

### Weaknesses
1. Figure 2 can be difficult to interpret in its current form. Splitting it into more straightforward visualizations, such as bar charts, could make the conclusions more visually clear and accessible.

2. The advantage of the proposed TALR method over baselines is not entirely consistent. At a learning rate of 1e-6, methods like L2 regularization perform on par with TALR. TALR shows a clearer advantage at 5e-6 on Qwen models, but this benefit does not consistently extend to the Gemma models. This suggests that the effectiveness of mitigation strategies may be dependent on the nature or distribution of the base model. Authors could provide more explanations/explorations on this angle

### Questions
1. In Figure 1, for some models (e.g., Qwen2.5-7B on ESCI w/ CoT), domain-specific SFT appears to slightly improve general performance compared to the initialization. Could the authors provide hypotheses or explanations for this unexpected improvement?

2. The appendix states that 20 training epochs were set for MedCalc and ESCI, which seems high for domain-specific SFT. What was the rationale for this choice? Providing a plot of performance versus epochs for key models and learning rates would be helpful. This would clarify whether models fine-tuned with large learning rates might have been overtrained, potentially confounding the results.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper shows that domain-specific SFT does not necessarily hurt LLM general abilities: small learning rates suffice to retain them while keeping domain gains. It gives a compression-theoretic explanation and introduces TALR, an adaptive token re-weighting method that further curbs forgetting.

### Strengths
- Revisits and refutes a prevailing assumption—namely, that domain-specific SFT necessarily degrades general capabilities—showing instead that optimization choices, particularly learning-rate magnitude, are the primary determinant of forgetting.

- Employ learning rate as a single, easily tunable knob that largely eliminates catastrophic forgetting without extra parameters or data.

- TALR is lightweight (closed-form, no extra forward passes), yet consistently outperforms recent regularizers and MoE-style baselines.

### Weaknesses
- Scope limited to decoder-only LLMs ≤8 B; no evidence that conclusions hold for larger or encoder-decoder models.

- No probing of emergent skills (e.g., long-context) or fairness/toxicity side-effects after SFT.

### Questions
- Did you experiment with top-k or sparse routing variants of TALR, and how do they affect convergence speed and final trade-offs?

- How does TALR behave when the domain corpus is large-scale (>>1 M examples) or when continual domain increments arrive sequentially—does τ need scheduling?

### Soundness
3

### Presentation
3

### Contribution
3
