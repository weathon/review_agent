# GRPO-MA: Multi-Answer Generation in GRPO for Stable and Efficient Chain-of-Thought Training

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Recent progress, such as DeepSeek-R1, has shown that the GRPO algorithm, a Reinforcement Learning (RL) approach, can effectively train Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) and Vision-Language Models (VLMs).
In this paper, we analyze three challenges of GRPO: gradient coupling between thoughts and answers, sparse reward signals caused by limited parallel sampling, and unstable advantage estimation. To mitigate these challenges, we propose GRPO-MA, a simple yet theoretically grounded method that leverages multi-answer generation from each thought process, enabling more robust and efficient optimization. Theoretically, we show that the variance of thought advantage decreases as the number of answers per thought increases.
Empirically, our gradient analysis confirms this effect, showing that GRPO-MA reduces gradient spikes compared to GRPO.
Experiments on math, code, and diverse multimodal tasks demonstrate that GRPO-MA substantially improves performance and training efficiency. Our ablation studies further reveal that increasing the number of answers per thought consistently enhances model performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GRPO-MA, an extension of the GRPO designed for training CoT reasoning in LLMs and VMs. The method addresses three specific problems in GRPO: gradient coupling between thoughts and answers, sparse reward signals, and unstable advantage estimation. GRPO-MA tackles these by sampling multiple answers per thought, decoupling the learning signals, and aggregating rewards to stabilize training. The paper provides a theoretical analysis of variance reduction with multi-answer sampling ande demonstrates through experiments and ablation studies across mathematical, coding and diverse multimodal tasks that GRPO-MA both improves model performance and enhances training stability and efficiency.

### Strengths
1. Clear identification of GRPO's limitations. The paper articulates three concrete challenges with existing GRPO-based RL for CoT—namely, gradient coupling, sparse rewards, and variance in advantage estimation—which are well-motivated in both text and multimodal tasks.
2. Theoretical Contribution. The variance analysis using the delta method is both non-trivial and practically relevant. Equations and accompanying text (see Section 4.2.2, Page 5) give explicit form for the variance of standardized thought advantages, clarifying the distinct influence of increasing the number of thoughts K and answers per thought M.
3. Gradient Stability. The inclusion of the Gradient Spike Score (GSS) is insightful; GRPO-MA consistently achieves lower GSS@10, lending quantitative evidence to theoretical claims of improved training stability (see Table 1 and 2).

### Weaknesses
1. Limited Scope of Baseline Comparisons. Although GRPO and SFT are extensively compared, the absense of GRPO-CARE limits the comprehensiveness of the evaluation. Since both GRPO-CARE and the proposed GRPO-MA aim to reduce the inconsistency between reasoning processes (thoughts) and final answers, including GRPO-CARE in the comparison would offer a more direct and compelling demonstration of GRPO-MA’s advantage in mitigating such inconsistencies.
2. Lack of Empirical Analysis on Variance. The paper assumes that the same thought may produce different answers, leading to increased reward variance, which serves as the main motivation for introducing multi-answer sampling. However, the authors do not empirically verify whether a single thought can produce multiple distinct answers to a significant extent in practice, especially under different sampling temperatures. Moreover, although the paper presents a detailed theoretical derivation of variance reduction, the experiments focus primarily on the Gradient Spike Score (GSS@10) as a proxy for training stability. But there is no explanation provided for why 10 is a meaningful cutoff A more direct analysis, such as reporting the step-wise dynamics of gradient norms, would substantially strengthen the empirical support for the claimed stability improvements.
3. Possible Overstatement of Computation Gains. While the paper claims that, for example, T4A4 achieves “comparable or even slightly better performance … with a 40% reduction in training time” vs T16A1 (Page 6), the per-training-step time metric may not reflect overall wall-clock or sample efficiency gains, as actual convergence rates and QA throughput are not shown. It is unclear if there are task or scale-dependent tradeoffs in the number of answers M versus thoughts K, especially for larger models or data regimes. 
4. Poor presentation. The paper’s presentation can be improved. The formatting of equations (2) and (3), as well as the description in Section 5.1.1, should be refined. Moreover, a substantial portion of the paper is devoted to explaining GRPO, while the authors should allocate more space to clearly describe their own method and highlight its novel contributions.
5. Lack of Anonymous Repository for Reproducibility. While the paper presents detailed experimental results and comparisons, it does not provide an anonymized code or data repository to support independent replication.

### Questions
1. The paper’s approach of drawing multiple samples from the same state and using Monte Carlo estimation to reduce the variance of advantage estimation reminds me of [1]. Could this method be considered a special case of VinePPO, where repeated sampling is applied only at the answer stage? If the authors could clearly explain the similarities and differences between the two methods, it would help make the contribution and novelty of this work clearer.
2. What is the sensitivity of the approach to the independence assumption between thought-value estimates? Could the authors experimentally ablate the effect of correlated thoughts on variance/stability?
3. How does GRPO-MA perform on other challenging mathematical and coding test sets?

[1] VinePPO: Refining Credit Assignment in RL Training of LLMs

### Soundness
2

### Presentation
2

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
This paper discusses three key challenges in Generalized Reinforcement Policy Optimization (GRPO): gradient coupling between thoughts and answers, sparse reward signals due to limited parallel sampling, and unstable advantage estimation. To address these challenges, the authors introduce GRPO-MA, a straightforward yet theoretically sound method. GRPO-MA improves optimization efficiency and enhances the overall performance by generating multiple answers from each thought process.

### Strengths
1. This paper proposes the GRPO-MA algorithm, a simple but effective and general improvement
strategy for GRPO.
2. Multiple task domains are discussed, such as mathematics and code, with a focus on visual reasoning tasks.

### Weaknesses
1. Normalizing rewards can lead to zero advantages when rewards are all-zero, stalling optimization. The proposed method reduces multiple samplings of thoughts and increases the number of answers per thought. However, this could exacerbate the issue, as fewer thoughts with limited sampling may lead to invalid reasoning, especially when correct thoughts appear less frequently. The paper offers incremental innovation, limiting its overall contribution.

2. Contribution 1 mentions compatibility with other RL techniques such as DAPO, but the experiments do not validate its effectiveness in other methods, so this cannot be considered a contribution of the paper.
3. Similarly, the paper does not compare its results with DAPO, GRPO-CARE, or Dr.GRPO/GPG. Therefore, some of the issues mentioned in the challenges raise doubts about whether the proposed method truly addresses these challenges compared with other methods.
4. The experiments in this paper are conducted only on Qwen2.5-VL-3B-Ins, which has a small model size and lacks evidence for broader applicability. The improvements in results are minimal, with outcomes comparable to or even lower than those with TN=16. Additionally, the performance on code-related tasks is inferior to GRPO, which undermines the validity of Contribution 3.

### Questions
1. How does the reduction in multiple thought samplings and the increase in answer samples per thought impact optimization?
2. Is the proposed method effective when applied to other RL techniques like DAPO?
3. How does the proposed method compare to other approaches like DAPO, GRPO-CARE, or Dr.GRPO/GPG in addressing the identified challenges?
4. Is the performance of the proposed method consistent across other common model sizes  (such as 7B size)?

### Soundness
1

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
4

### Summary
This paper proposes GRPO-MA, an extension of Group Relative Policy Optimization for Chain-of-Thought training. The method generates multiple answers per thought to reduce gradient variance, decouple thought–answer updates, and stabilize training. Theoretically, it shows that more answers yield lower variance; empirically, GRPO-MA improves stability and performance across math, code, and multimodal tasks.

### Strengths
1. **Extensive experiments**: The paper evaluates GRPO-MA on a wide range of reasoning and multimodal tasks (math, code, vision, and manipulation), demonstrating consistent gains in both performance and training stability.
2. **Strong theoretical grounding**: The derivation using the delta method clearly connects multi-answer sampling with variance reduction in advantage estimation, providing a solid mathematical basis for the approach.
3. **Cross-modal applicability**: The method generalizes well across text and vision tasks, highlighting its robustness and potential for broad adoption in reinforcement learning for reasoning.

### Weaknesses
1. **Overly long introduction**: The introduction is too long and repeats background information that could be moved to the related work section for better focus and readability.
2. **Limited novelty**: The proposed multi-answer strategy is intuitive and builds naturally on GRPO. While it’s a solid and useful extension, it feels more like a practical enhancement than a major conceptual change.
3. **Limited model diversity**: The experiments focus mainly on the Qwen2.5-VL family, leaving it unclear whether the proposed GRPO-MA method generalizes equally well to other model architectures or sizes.
4. **Lack of scaling analysis**: The paper would benefit from an experiment exploring how GRPO-MA scales with model size and training time compared to the original GRPO. Such a scaling law study could clarify the method’s efficiency and practicality for larger models.
5. **Missing comparisons with other RL baselines**: While the paper mentions several GRPO variants (e.g., DAPO, Dr.GRPO, GPG), it only compares GRPO-MA against the original GRPO. Including results or discussion on how GRPO-MA performs relative to these methods would provide a clearer picture of its advantages and limitations.

### Questions
Please refer to weaknesses

### Soundness
2

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
4

### Summary
The paper proposes GRPO-MA, a variant of GRPO that samples M answers per thought, uses the average reward per thought to compute a thought-level advantage, and applies token-level PPO-style updates to both thoughts and answers. Theoretically, the authors argue (via the delta method) that increasing M monotonically reduces the variance of the thought advantage; empirically, they show higher accuracy, fewer gradient spikes, and better sample-efficiency across math, code, vision, and an embodied manipulation task.

### Strengths
1. The paper is generally well-written and structured, followed by the motivation and explanation.
2. Multi-answer sampling decouples noisy answer outcomes from thought quality and directly generates reward signals.
3. Experiments span math/code and diverse multimodal tasks with consistent gains.

### Weaknesses
1. The only difference I can think of between multi-answer with one rollout and single-answer generation is that, in the former, at least one answer among the multiple generations is correct. However, how can the authors guarantee that, given the same instruction and thinking trajectory, the model can generate the final answer directly without rethinking? It would be helpful to include an example illustrating this process.
2. This method can apply to both LLMs and VLMs. Why did the authors not evaluate it on text-only LLMs? One concern is that multimodal models produce shorter CoTs compared to LLMs, which makes the reasoning process simpler. It would be better to test the approach on longer reasoning tasks using LLMs.
3. It's better to add stronger baselines, such as DAPO, for a more comprehensive comparison.
4. Although the authors evaluate different values of $K$ and $M$, since these are the two most critical hyperparameters, I suggest conducting a more extensive exploration to determine the optimal combination of $K$ and $M$.

### Questions
1. The authors train a 3B model using LoRA. I suggest testing with a larger base model with LoRA.
2. In Figure 3, the performance appears to decrease slightly when $M = 6$ and then increase afterward. Could the authors explain the reason behind this?

### Soundness
3

### Presentation
3

### Contribution
2
