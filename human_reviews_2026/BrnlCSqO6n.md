# SpecBranch: Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Speculative decoding (SD) has emerged as a promising technique to accelerate LLM inference by employing a small, efficient draft model to propose draft tokens in advance, and subsequently validating them in parallel with the large target model. However, the existing SD methods still remain fundamentally constrained by their serialized execution, which inevitably causes mutual waiting bubbles between the draft and target models. To address this critical challenge, we draw inspiration from sophisticated branch prediction mechanisms in modern processors and propose a novel framework, \textbf{SpecBranch}, to fully unlock branch parallelism in SD. Specifically, we first conduct an in-depth analysis of the potential of branch parallelism in SD, and recognize that the key challenge lies in the intricate trade-offs between parallelization and token rollback. Based on this analysis, we introduce parallel speculative branches to preemptively hedge against likely rejections. Meanwhile, to significantly enhance parallelism, we jointly orchestrate adaptive draft lengths with a hybrid combination of the implicit draft model confidence and explicit reusing of target model features. Extensive experiments conducted across various models and benchmarks show that \textbf{SpecBranch} achieves impressive speedups of over \textbf{1.8}$\times \sim$ \textbf{4.5}$\times$ against the standard auto-regressive decoding and reduces rollback tokens by \textbf{50}\% for poorly aligned models, while maintaining an identical sampling distribution. Our code is available at \url{https://github.com/Sylvan820/Specbranch}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The method introduces parallel speculative branches, adaptively chooses draft lengths, and reuses features of the target model to hedge against incorrect drafts and reduce rollback tokens.

### Strengths
The idea of modelling speculative decoding as analogous to branch prediction in CPUs is clever, and the introduction of parallel speculative branches is conceptually interesting.

### Weaknesses
It is not fully clear how generally applicable the method is across diverse architectures, tasks, or deployment settings.

### Questions
I suggest adding ablation experiments exploring branch length, number of speculative branches, draft vs. target model similarity, and rollback frequency under different task/benchmark settings.

### Soundness
3

### Presentation
4

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
Based on PEARL's parallel framework, SpecBranch introduces the Hybrid Rollback-Aware Draft Structure (H-RAD) and ingeniously integrates the branch resampling method, enabling branch-parallel execution of speculative decoding (SD). This design effectively addresses the serialized waiting bottleneck and high rollback issue inherent in traditional SD methods. The authors rigorously validate and elaborate on these two core innovations through comprehensive theoretical analyses—including Theorem 1 on latency under rollback and the truncated geometric distribution of accepted tokens—and extensive experimental evaluations.

### Strengths
1.	Comprehensive figures and textual descriptions, clear structure, and detailed experiments.
2.	Starting from the weaknesses of PEARL—including its rollback-oblivious nature, static draft length, and inadequate handling of pre-verify/post-verify rollbacks—the authors explain the rationale behind proposing the Hybrid Rollback-Aware Draft Structure (H-RAD). They further validate the design of H-RAD as a three-class classification framework through experiments, and combine it with in-depth theoretical analyses (e.g., Theorem 1 on latency under rollback, truncated geometric distribution of accepted tokens) to develop the SpecBranch method. This approach achieves excellent speedup ratios among training-free speculative sampling methods.
3.	Detailed comparative analyses are conducted on hyperparameters (e.g., maximum branch number kmax, confidence threshold ϵ, feature layer count K) and core components (H-RAD, branch resampling) that influence the speedup ratio.
4.	The paper’s analysis of the Rollback Rate (RB) clearly identifies the time-consuming bottlenecks of speculative decoding (SD) methods, and effectively demonstrates the superiority of SpecBranch in reducing rollback tokens (up to 50% reduction for poorly aligned model pairs).

### Weaknesses
1. There are several errors in the paper, such as the incorrect labeling of "Lookahead" in Figure 1(c) and the syntax issue with "foreach" at Line 668.
2. The superiority of the proposed method lies in its training-free characteristic, yet its acceleration performance is not as excellent as that of training-required methods like EAGLE-3. Furthermore, the so-called "training-free" essentially relies on pre-existing draft-target model pairs. Additionally, during inference, this parallel training-free method consumes more GPU resources than EAGLE and poses greater challenges in deployment. Therefore, it is anticipated that the authors will provide deployment implementations on frameworks such as vLLM or integrate SpecBranch into EAGLE, delivering more practical solutions to the industry.

### Questions
The authors have conducted numerous analytical experiments, and the paper is quite complexly written. However, I believe the method in the paper is actually very simple—it merely adds an MLP on top of PEARL for judgment. I need the authors to explain the timing sequence of the entire process, which should be described most clearly in the main text, especially how the subsequent tokens of the branches are generated and verified.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper seeks to address the pipeline bubble that occurs during the verification step in speculative decoding. While previous research has looked at speculating in parallel to performing verification, they have done so by making simplistic assumptions about draft acceptance length. SpecBranch on the other hand uses a hybrid predictor to intelligently decide how to spend parallel speculation resources. The SpecBranch method leads to substantial speed ups on multiple benchmarks.

### Strengths
* The writing and figures used throughout the paper are very clean.
* The theoretical model of parallel speculative decoding is well motivated and provides a good model for understanding tradeoffs.
* The idea of rollback aware branched parallelism is novel and addresses a large gap with current speculative decoding methods.
* The empirical analysis presented is robust as the method is tested across a large range of tasks and across multiple model sizes and model families. Furthermore, the speedup that was achieved is substantial compared to most improvements to speculative decoding.

### Weaknesses
* The largest weakness is that the analysis is only presented for the batch size = 1 setting. While this makes sense as the low batch setting is most relevant for speculative decoding methods, it would be helpful to profile the scaling behavior as the batch size is increased.

### Questions
None right now.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel and well-executed study on improving draft model training for speculative decoding using structural re-parameterization. The method is innovative, empirically thorough, and practically significant.

### Strengths
1 The paper introduces structural re-parameterization—commonly used in CNNs—into speculative decoding for the first time. The idea is clever and directly addresses the core limitation of draft model capacity. The dual design of Pure Linear (no inference cost) and Hybrid (better performance with minimal cost) offers flexible solutions for different deployment needs.
2 The authors evaluate RepSpec across multiple target models (LLaMA-3.1-8B, LLaMA-2-13B, Vicuna-7B), tasks (dialogue, code, math, etc.), and draft frameworks (EAGLE, Medusa, Hydra). The consistent improvements in acceptance length and speed-up demonstrate the generality and effectiveness of the method.
3 The ablation studies are systematic and insightful.

### Weaknesses
1 While Appendix E attempts to explain why re-parameterization helps (gradient variance reduction, implicit spectral regularization), the arguments are intuitive rather than rigorous. A more formal analysis—e.g., connecting the training dynamics to a well-defined optimization objective—would strengthen the foundation.

2 The paper compares RepSpec against standard training. However, it would be more compelling to compare against a baseline that directly increases the model size(e.g., adding more layers or hidden units) to match the parameter count of RepSpec during training. This would better isolate the benefit of the re-parameterization structure itself.

3 The inference overhead of the Hybrid method is well documented, but the additional training cost(time, GPU hours) is only briefly mentioned. A quantitative comparison (e.g., how much longer RepSpec takes to train than the baseline) would help users assess the overall cost-benefit trade-off.

### Questions
The results on LLaMA-2-13B suggest that Hybrid RepSpec performs better relative to Pure Linear as the target model grows. Do the authors have any results or intuition about how RepSpec would behave with very large target models(e.g., 70B+)? Would the Hybrid method become even more favorable?

### Soundness
3

### Presentation
3

### Contribution
3
