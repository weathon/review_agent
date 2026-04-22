# Scalable LLM Math Reasoning Acceleration with Low-rank Distillation

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2

## Abstract
Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a resource-efficient distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1\% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters ($\sim$2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>16\% time-to-next-token reduction) while encouraging response brevity (up to 8.5\% fewer tokens).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Caprese, a method to recover model performance after pruning / efficiency adaptions, by training a LoRA-like adapter to minimize the residual error. Specifically the authors use CATS and GRIFFIN as FF sparsification methods, and show that Caprese can help to regain some of the lost Math performance with minimal computational overhead as the additional linear layers can be fused with existing ones that run in parallel. The distillation is data and memory efficient, just learning small adapters. The authors propose a reselection procedure on top that can improve the performance of Caprese in combination with GRIFFIN.

### Strengths
- Recovering strong math performance
- Minimal inference overhead
- Fast training
- Reselection procedure for improved performance similar to speculative decoding
- Analysis of the response length of efficiency-optimized models vs. original

### Weaknesses
- Similarity to LoRA, LESS just applied to FF layers resulting in a lack of novelty
- Not reproducible given private training set

### Questions
- What is "a common synthetic math training set"?
- How do you expect this to work around attention layers, e.g. when using sparsified / pruned attention, in the reasoning case?
- Your analysis on average response length is quite important. In the reasoning setting with a late final answer, wouldn't it make sense to compare overall runtime as efficiency metric - this way GRIFFIN / Caprese-GRIFFIN would underperform, despite per-token speedup, right? Could you add an analysis on this overall cost / runtime?

### Soundness
4

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
This paper introduces an efficient method of distilling models, with a primary focus on adapting the feed-forward blocks. Despite its relatively low budget, the method achieves promising results in almost fully recovering the mathematical capabilities of the distilled models compared to the teacher model. More exactly, Caprese recovers substantial performance with a small training budget (only 20K synthetic samples) and minimal overhead (roughly 0.8% additional parameters).

### Strengths
- One strength for the Caprese method is the efficiency compared to baselines, achieving near teacher-level performance on challenging benchmarks like MATH and GSM8K with a minimal training budget. This trade-off between computational cost and capability is a significant practical contribution, making high-performing, specialized models more accessible for real-world deployment.
- The strategy of performing low-rank distillation specifically on the FFN blocks to learn the residual necessary for complex tasks is an original and effective architectural choice.
- Accordingly, the paper clearly explains the proposed mechanism and its implementation. The methodology is straightforward and the experiments generally are well-structured.

### Weaknesses
- The primary focus and strongest empirical evidence are on mathematical reasoning tasks. It remains unclear how the method would transfer effectively to a diverse set of other domains, such as code generation or more complex general language understanding tasks.
- While the authors claim "strong" language capabilities, the empirical evidence is restricted to only a few benchmarks (CoQA, QASPER). On benchmarks like QASPER, the Caprese method does not consistently outperform the established GRIFFIN baseline (e.g., Table 1). The evaluation of language capabilities should be extended to support the strong claim.
- The initial performance of the original teacher models (e.g., Llama 3.1 8B Instruct GSM8K: 27.90, MATH: 13.94) appears unusually low for a model of that scale. This raises a question about the specific prompting strategy or evaluation setup used for the teacher model versus the student model during inference. Clarifying this discrepancy is essential for fully evaluating the true performance "recovery."
- The low-rank size of 256 is considered as being low-cost, but a small ablation study showing the sensitivity of the performance recovery to the rank dimension would significantly strengthen the analysis of the method's trade-off between parameter overhead and performance gains.

### Questions
1. Could you please elaborate on the surprisingly low performance of the Llama 3.1 8B Instruct teacher model on the GSM8K and MATH benchmarks as shown in Table 6? Specifically, is the observed performance gain of Caprese potentially related to the specific prompts or few-shot settings used during the benchmark evaluation for the student vs. teacher models?

2. Please provide a small analysis on the effect of the batch size for the distillation setting. How does the efficiency of your method scale, and how effective is it to use Caprese for a larger batch size during training compared to conventional methods?

3. The claim of "strong [...] language capabilities" requires more evidence. Could you please justify the focus on only CoQA and QASPER for language tasks? I strongly suggest extending the evaluation to a more diverse set of standard LLM benchmarks, such as MMLU, DROP, or a selection from the HELM suite, to substantiate this claim.

4. Would it be possible to include a comparison for an additional domain? Even a small experiment, perhaps on a code generation task (e.g., HumanEval), would be beneficial. This would demonstrate that, even if the primary focus is math, the method does not perform drastically worse than the teacher on a different, yet still STEM-related, task.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Caprese recovers math reasoning performance lost from sparse FF methods (GRIFFIN, CATS) by distilling residuals into rank-256 low-rank layers (~1% parameters) trained on 20K synthetic samples. Results show full recovery (DS-Qwen 7B: 62.5%→78.1% AMC) without harming language tasks, preserving efficiency (~2B parameter cut, >16% latency reduction). Works across instruct/thinking models, scales with Best-of-N, and unexpectedly reduces output length by 5-8.5%.

### Strengths
**Strong Empirical Evidence**: Sparse FF methods devastate math (Gemma 2 2B: 51%→34% GSM8K) despite working well on language. Figure 2's oracle analysis shows rank-256 approximation reduces residual error more than doubling sparsity density, motivating low-rank design.

**Comprehensive Experiments**: Robust across 1B-32B models (Llama, Gemma, Qwen, DeepSeek), math/language tasks, and test-time scaling. Figure 3 shows +7% Pass@100 vs. full model with 15.8% less compute. Table 2 demonstrates recovery on 32K-token thinking models.

Only 20K samples, 23 epochs, frozen weights. Figure 6 shows ~1% parameter overhead. Reselection (Table 3) improves DS-Qwen 1.5B from 34%→59% AMC by updating GRIFFIN metrics. Unexplained brevity (Figure 4): 5-8.5% fewer tokens than full models despite no length penalties.

### Weaknesses
**No Mechanistic Understanding**: What do L,R matrices capture? No SVD analysis, probing, or visualization. Why does layer-wise work better for Gemma but E2E for Llama (Table 6)? No ablation isolating layer-wise vs. E2E contributions.

**Narrow Scope**: Claims "any task" but only math + 2 language QA tasks. Missing: code generation, creative writing, dialogue, multi-modal. No baselines: full→sparse KD, error-focused training, adaptive rank

### Questions
1. Does Caprese work with MoEs, pruning, quantization? Or only activation-based sparsity?
2. Provide SVD analysis of L,R, visualization of captured components, and ablation isolating layer-wise vs. E2E.
3. Why shorter responses? Is it filtering redundancy, implicit regularization, or correlation with error reduction?
4. Compare to full→sparse KD, LoRA (main text), and more training data for sparse methods.

### Soundness
2

### Presentation
2

### Contribution
2
