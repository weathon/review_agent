# MixReasoning: Switching Modes to Think

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4

## Abstract
Reasoning models enhance performance by tackling problems in a step-by-step manner, decomposing them into sub-problems and exploring long chains of thought before producing an answer. However, applying extended reasoning to every step introduces substantial redundancy, as sub-problems vary widely in difficulty and complexity: a small number of pivotal steps are genuinely challenging and decisive for the final answer, while many others only involve straightforward revisions or simple computations. Therefore, a natural idea is to endow reasoning models with the ability to adaptively respond to this variation, rather than treating all steps with the same level of elaboration. To this end, we propose MixReasoning, a framework that dynamically adjusts the depth of reasoning within a single response. The resulting chain of thought then becomes a mixture of detailed reasoning on difficult steps and concise inference on simpler ones. Experiments on GSM8K, MATH-500, and AIME show that MixReasoning shortens reasoning length and substantially improves efficiency without compromising accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper tackles redundancy in chain-of-thought (CoT) reasoning, where all sub-steps are treated with equal detail despite varying difficulty. It introduces MixReasoning, a framework that dynamically adjusts reasoning depth within a single CoT—using concise reasoning for trivial steps and detailed reasoning for critical ones. The approach employs a LoRA adapter enabling a “concise mode,” and monitors token-level uncertainty (via next-token entropy) to switch between modes during inference, with minimal overhead through KV-cache reuse. Experiments on reasoning benchmarks (e.g., GSM8K, MATH, AIME) show significant token reduction (≈47%) while maintaining or improving accuracy. Overall, the paper offers a novel, practical method for adaptive reasoning that better balances efficiency and accuracy by mirroring human-like focus on challenging reasoning steps.

### Strengths
The paper is an original and well-executed contribution addressing redundancy in chain-of-thought reasoning. It identifies the overlooked issue of intra-chain heterogeneity—that not all reasoning steps require equal depth—and introduces MixReasoning, which dynamically switches between concise and detailed reasoning modes based on token-level uncertainty. The approach is novel in both concept and implementation, using a LoRA adapter for lightweight mode switching and entropy-based triggers for adaptive control.

The work is technically solid and clearly presented: experiments on standard benchmarks show that it reduces reasoning length while maintaining or improving accuracy, and the method’s efficiency (via KV-cache reuse) makes it practical. The exposition is clear and logically structured, supported by quantitative and qualitative analysis.

Overall, the paper’s strengths lie in its originality, methodological soundness, and practical significance. It offers a compelling step toward adaptive, efficient reasoning in LLMs—an idea likely to influence future research on dynamic CoT and budgeted inference.

### Weaknesses
1. Limited Baseline Context & Novelty Clarification: The concept of adaptive reasoning depth within a chain is innovative, but the paper does not adequately compare it with existing global mode switching methods (e.g., AdaptThink, Emergent Mind). Without clearer positioning, the claim of originality appears overstated. The paper would benefit from a more detailed comparison to highlight the unique contributions of this approach.

2. Insufficient Heuristic Sensitivity Analysis: The entropy-based mode-switching heuristic lacks a thorough sensitivity analysis regarding key hyperparameters (e.g., τ↑/τ↓, rollback window B/F). These parameters likely have a significant impact on both performance and computational cost. Ablation studies should be conducted to analyze the effects of these hyperparameters and their influence on false positive/negative triggers in reasoning.

3. Narrow Evaluation Scope: Experiments are limited to mathematical reasoning tasks (GSM8K, MATH, AIME), and the broader claim of "reasoning efficiency" is not supported across different task domains. To validate the generality of the approach, experiments on non-mathematical reasoning tasks should be added, or the paper should acknowledge the method's domain limitations.

4. Reproducibility & Deployment Concerns: The paper lacks detailed implementation information and underreports practical performance, especially latency due to rollbacks. While token savings are mentioned, real-world efficiency may still degrade because of rollback delays. The authors should provide more implementation details, pseudocode, and wall-clock latency results to evaluate the feasibility of real-world deployment.

5. Interpretability & Readability of Reasoning Chains: The paper claims improved readability of reasoning chains, but evidence for this is minimal. The concise mode, while reducing reasoning length, may compromise the transparency or completeness of reasoning. Human evaluation of reasoning quality and explanation completeness across different modes is needed to validate these claims.

### Questions
1. Could you clarify how your method differs in terms of granularity (token/step vs. question-level) and switching criteria (entropy vs. RL)? A direct comparison or ablation against a question-level switching baseline would be helpful.

2. How sensitive are the results to changes in the hyperparameters (τ↑, τ↓, B, F, α_low/α_high)? Did you perform grid or annealing experiments on hold-out sets to select them?

3. What is the end-to-end inference latency (including rollback overhead) compared to a baseline without switching? How often did rollbacks occur in your experiments?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper addresses the computational inefficiency inherent in Large Language Models that apply a uniform, step-by-step reasoning process to all problems, regardless of their intrinsic difficulty. The authors posit that not all steps in a reasoning chain are equally complex; a few pivotal steps are decisive, while many others are routine. To address this, they introduce **MixReasoning**, a framework that dynamically switches between a detailed "thinking" mode and a concise "non-thinking" mode within a single response generation. This mode-switching is triggered by monitoring token-level uncertainty during inference. The framework is evaluated on the GSM8K, MATH-500, and AIME benchmarks, where it is shown to shorten reasoning lengths and improve efficiency, often without a loss in accuracy.

### Strengths
**Targets a Critical Problem:** The research focuses on the important and practical challenge of adaptive reasoning. Reducing the substantial inference cost and latency of long chain-of-thought (CoT) processes is crucial for making Large Language Models viable in interactive applications. The paper correctly identifies that a finer-grained, intra-response approach is a logical next step beyond problem-level routing, which treats entire problems as either simple or complex.

### Weaknesses
* **Evaluation on Saturated Benchmarks:** The primary benchmark used, GSM8K, is largely considered saturated, with top models achieving over 95% accuracy. On such benchmarks, it is difficult to demonstrate meaningful accuracy improvements or robustly differentiate the performance benefits of a new method from random variance.

* **Limited Model Diversity:** The experiments rely heavily on the Qwen model family (Qwen3-8B and Qwen3-14B), which constitutes two of the three models tested. This lack of diversity raises questions about the generalizability of the findings and the "model-agnostic" claim.

* **Insufficient Ablation Studies:** The paper lacks critical ablation studies to validate its core claims. While it includes an analysis of LoRA targets (MLP vs. attention layers), it fails to investigate:
    * **The efficacy of the uncertainty trigger:** There is no comparison against simpler or random switching heuristics to prove that the uncertainty metric is the key factor driving performance.
    * **The standalone performance of the concise mode and verbose mode:** The paper does not report the baseline performance of the model when forced to use only the concise (LoRA-adapted) mode. This makes it hard to discern what generates performance in MixReasoning .

* **Marginal Performance Improvements:** The reported results do not demonstrate strong performance gains across the board in terms of performance compared to number of tokens.

* **No comparison to Continuous Chain of Thoughts method [1]:** Continuous Chain of Thought has showed the ability to reason with less tokens which would be an important baseline to compare to 

[1] Hao, S. (2024). Training large language models to reason in a continuous latent space

### Questions
1.  The uncertainty-based trigger is the core of MixReasoning's dynamic behavior. How does this trigger compare to simpler switching heuristics, such as switching at fixed intervals, randomly, or based on the presence of certain keywords (e.g., "Therefore," "Let's calculate")?

2.  What is the standalone accuracy of the concise model and non-concise model (i.e., the base model with the LoRA adapter always active with strength α_low) on the evaluation benchmarks? This baseline is essential for understanding the trade-offs and determining whether the concise mode is a capable-but-brief reasoner or a degraded one that requires the "thinking" mode to salvage accuracy.

3.  The framework is tested on two Qwen models and one other model. To better substantiate the "model-agnostic" claim, have you considered applying MixReasoning to models with different architectures and training schemes, such as models from the Llama or Mistral families, which are widely-used open-source baselines?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
MixReasoning is an inference-time framework that dynamically adjusts reasoning depth within a single CoT response. Rather than applying uniform elaboration across all steps or making binary problem-level decisions, the method adaptively switches between detailed "thinking" and concise "non-thinking" modes based on local token-level uncertainty. This achieves substantial token compression (25-47% reduction on benchmarks) while maintaining or improving accuracy, offering practitioners explicit controllability over the accuracy-efficiency trade-off.

### Strengths
1. Different from previous problem-wise reasoning mode selection, the proposed method adjusts the reasoning in a more flexible way: it determines whether a detailed explanation is needed during the generation of the single response. 
2. The plug-in LoRA-based adaptor for uncertainty estimation without sacrificing the capability ofthe  base model. 
3. MixReasoning is runtime-efficient by allowing reuse of KV cache: the KV cache tokens generated in the previous thinking mode can be prefilled and reused in the new thinking mode.
4. Figure 3 shows that with the same generation tokens, the accuracy of MixReasoning is higher than baseline models, demonstrating the better accuracy–efficiency Pareto frontier the proposed method achieves.
5. They also investigate the control of reasoning length and find that MLP layers contribute more than the attention. Besides, without modifying the attention layers, the KV cache can also be reused to speed up inference.

### Weaknesses
1. Experiment results are not strong enough to show the effectiveness of the proposed method. In Table 1, the improvement of pass@1 is not significant. For example, on GSM8K, the average improvement (compared with the best baseline models) is around 0.2%, indicating that only 2 more problems can be solved. It is unclear whether this improvement is caused by the randomness in sampling. Similarly, the improvement on Math-500 and AIME 2024 is also marginal. The significance test and standard deviation should be added to verify the effectiveness.
2. Some evaluation details are missing. For example, what is the temperature used during evaluation? Besides, it is unclear to me how the evaluation results are calculated. In the appendix, it says that all results are averaged over 5 runs. Considering the 500 cases in Math-500, the pass@1's granularity should be 1/500 = 0.002. After averaging on 5 runs, the last digit should still be an even number. The pass@1, such as 0.8937, looks weird to me because it indicates that 2,234.25 problems are correct in 500 * 5 instances. 
3. Figure and Table typos
	1. In Table 1, under the setting of Qwen3-8B on Math-500, the best performance that should be bolded is 0.9320 instead of 0.9313. 
	2. Figure 5: The meaning of the y-axis is unclear. Is a higher y value better, or reverse? According to Lines 459-464, it seems that Figure 5(a) aims to compare the reduced token count. If that is the case, there should be one baseline without finetuning to show the initial token count.

### Questions
1. Can the LoRA for uncertainty estimation trained on one dataset be transferred to another dataset or domain? 
2. While the window size and uncertainty threshold can control the mix of modes, is there an empirical way to select suitable hyperparameters?

### Soundness
2

### Presentation
3

### Contribution
3
