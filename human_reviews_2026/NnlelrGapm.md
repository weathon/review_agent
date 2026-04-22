# UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Decision-Making

- Avg Score: 5.50
- Decision: Reject
- Scores: 6, 6, 6, 4

## Abstract
As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios being underexplored. In this paper, we introduce a principled, information‑theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (MPI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of uncertainty quantification in large language models within multi-step decision-making scenarios. It proposes an information-theoretic framework that decomposes uncertainty into ​intrinsic uncertainty​ and ​extrinsic uncertainty. The authors introduce the ​UProp method, which efficiently estimates extrinsic uncertainty by sampling from a ​trajectory-dependent decision process (TDP)​​ and approximating ​pointwise mutual information (PMI)​, thereby tackling the challenge of the inherent non-computability of extrinsic uncertainty. Experiments conducted on benchmarks including ​AgentBench-OS, HotpotQA, and StrategyQA​ validate the superiority of UProp, demonstrating AUROC improvements of 2.3% to 11%. Further analyses on sampling efficiency and applications such as selective prediction are also provided.

### Strengths
1. Framework Novelty. It is the first to decompose multi-step decision-making uncertainty into Intrinsic Uncertainty (IU) and Extrinsic Uncertainty (EU), providing a principled framework.
2. Theoretical Soundness and Methodological Innovation. The convergence proofs and the local smoothness assumption enhance the method's credibility. UProp addresses the high computational complexity of Mutual Information (MI) estimation through Trajectory-Dependent Decision Process (TDP) sampling and Pointwise Mutual Information (PMI) approximation, effectively balancing efficiency with accuracy.
3. Comprehensive experiments. The evaluation covers realistic scenarios such as operating system interactions and multi-hop question answering, and includes comparisons with multiple baselines, making the results highly convincing.

### Weaknesses
1. High Sampling Dependency.​​ The TDP sampling process requires multiple LLM invocations, which may introduce inference latency issues.
2. Dependence on Theoretical Assumptions.​​ The local smoothness assumption may not always hold in highly uncertain decision chains.
3. Lack of Theoretical Boundary Analysis.​​ The study fails to explicitly specify the upper bound of the impact of PMI estimation errors on the overall uncertainty quantification (UQ) results.

### Questions
1. How does the estimation error of Extrinsic Uncertainty (EU) propagate as the number of steps increases?​ Is there an analysis of an upper bound for this error?​
2. In the context of extremely long decision sequences, does extrinsic uncertainty exhibit cumulative drift?​
3. In large-scale tasks, particularly when the decision space expands rapidly, the computational burden of Monte Carlo sampling may become non-negligible. How can the algorithm be optimized to maintain efficiency when handling larger-scale decision-making tasks, especially for application scenarios with high real-time requirements?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes UProp, a method to quantify uncertainty in Large Language Models for multi-turn question-answering and sequential decision-making—critical for judging when to trust LLM outputs. UProp decomposes uncertainty into two parts: Intrinsic Uncertainty(IU) and Extrinsic Uncertainty(EU). And Uprop calculates the EU through Trajectory-Dependent Pointwise MI since directly calculating EU is intractable. Experiments across multi-turn benchmarks (e.g., AgentBench-OS, HotpotQA) and state-of-the-art LLMs show UProp achieves more accurate uncertainty quantification than single-turn UQ baselines.

### Strengths
1. The paper is clearly written and aims to tackle the more realistic and challenging setting of multi-turn QA, providing a rigorous, information-theoretic foundation for it.
2. UProp proposes a novel perspective for quantifying uncertainty in multi-turn question-answering. Its "PMI in TDP" method offers an efficient solution by converting the computationally expensive direct estimation of Mutual Information (MI) into a tractable process.

### Weaknesses
1. Although UProp employs sampling to approximate distributions, this strategy introduces non-trivial computational costs, resulting in slower inference speeds and increased resource consumption. These factors may limit the method's applicability in real-time or large-scale deployment scenarios.
2. The method heavily depends on a smoothness assumption. While intuitive, this assumption lacks theoretical grounding and may not hold in all practical conditions, potentially compromising the framework's generalizability.
3. UProp is primarily evaluated on commonsense reasoning tasks with definitive answers. Consequently, its effectiveness and contributions remain unclear for other problem categories, such as those involving open-ended generation or creative tasks.
4. There are some typos in the paper, e.g, Figure 1, Section 5.

### Questions
1. The results on StrategyQA (Table 1) show a higher success rate for Qwen2.5 but a much lower AUROC compared to Gemma-2. Please discuss this result. Does it indicate that the estimated uncertainty does not always directly correlate with the final task success rate?
2. Could the authors elaborate on the potential of Uprop in more exploratory environments (e.g., creative writing, simulations), where leveraging uncertainty is crucial for guiding the agent's actions to achieve higher performance?
3. Is it possible to integrate the EU component with other single-turn UQ methods, and if so, what would be the effect?
4. UProp employs ReAct as the reasoning method. How well would it perform with other reasoning methods?
5. Quantifying output uncertainty in LLMs is a critical consideration, and recent work, PlanU [1], explores a similar direction. PlanU extends uncertainty quantification to the context of LLMs in reasoning tasks, aligning with the objectives of UProp. However, PlanU also introduces the additional aspect of considering environmental uncertainty. It would be valuable to further explore the distinctions between UProp and PlanU, especially in terms of how each approach handles their implications for uncertainty quantification.

[1] PlanU: Large Language Model Reasoning through Planning under Uncertainty. NeurIPS 2025

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
2

### Summary
This paper addresses the challenge of Uncertainty Quantification (UQ) for LLMs in multi-step decision-making. The authors propose UProp, a method that decomposes the total uncertainty into Intrinsic Uncertainty (IU) from the current decision and Extrinsic Uncertainty (EU) inherited from previous steps, and efficiently estimates the computationally challenging EU.

### Strengths
1. The research topic is valuable, targeting the under-explored uncertainty propagation in LLM multi-step decision-making (a critical issue for reliable agentic LLMs), the work aligns with current academic and industrial needs.
2. UProp avoids complex auxiliary models or exponential space exploration, relying on lightweight sampling and approximation—striking a good balance between computational efficiency and estimation accuracy.

### Weaknesses
- The reliance on a simple random sampling strategy for trajectory candidates lacks justification. The lack of discussion or ablation studies on sampling strategies leaves open the question of how sensitive UProp's performance is to the quality and diversity of the sampled trajectories.
- The paper quantifies extrinsic uncertainty but does not verify its correlation with final decision error rates (e.g., whether higher extrinsic uncertainty correlates with more frequent multi-step reasoning errors)
- Unaddressed performance in long-sequence scenarios, no experiments or discussions are provided on UProp’s behavior in multi-step decisions with over 10 steps (e.g., whether step-length normalization fully mitigates length bias, or if PdMI approximation error accumulates), limiting the method’s applicability to complex long-horizon tasks.

### Questions
- How was the sampling strategy designed? Could the estimator's accuracy be significantly improved by employing a more intelligent sampling technique that prioritizes informative or diverse decision paths?
- Is there a measurable correlation between the estimated extrinsic uncertainty and the actual error rate in multi-step decisions? Could the authors provide experimental evidence (e.g., error rate vs. EU scatter plots) to strengthen the practical utility of UProp?
- How does UProp perform in long-sequence decision tasks? If estimation accuracy degrades with more steps, what strategies could be applied to maintain stability and mitigate bias?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces an estimate method to estimate the LLM sequential decision uncertainty. First, the paper decomposes LLM sequential decision uncertainty into two parts, including internal uncertainty intrinsic to the current decision and extrinsic uncertainty as a Mutual-Information (MI) quantity of how much uncertainty is inherited from preceding decisions. Second, the proposed method as UProp achieves effective extrinsic uncertainty estimation, which converts the direct estimation of MI to the estimation of Pointwise Mutual Information over multiple Trajectory-Dependent Decision Processes. The results show that UProp significantly outperforms existing single-turn Uncertainty Quantification baselines, including sampling efficiency, potential applications, and intermediate uncertainty propagation.

### Strengths
1. The theoretical analysis is relatively rigorous.
2. The experimental results show that the algorithm is effective.
3. The method simply estimates the proposed Extrinsic uncertainty in the multi-step decision-making of LLM with the pointwise mutual information by the Monte Carlo sampling.

### Weaknesses
1. The innovation of the paper needs further clarification.
2. There is a lack of computational cost analysis of the method.
3. The method may be hard to generalize due to the large output space and hallucination of LLM.

### Questions
1. Did you first propose to decompose the uncertainty of decision-making into intrinsic and extrinsic uncertainty with the information theory? If not, please introduce some previous related work. If yes, please provide more evidence.
2. The backbone of LLM model may be additionally selected, including Claude 3 or Gemini 2.5 Pro, which has a powerful ability of decision-making or reasoning.
3. Please add some benchmarks such as complex mathematical reasoning tasks or long-chain knowledge reasoning, etc.
4. After quantifying intrinsic and Extrinsic uncertainties, how can you optimize LLM decisions through ReAct-style prompts?
5. There are some problems with the manuscript. Why are some words underlined? And the symbols such as “➋” are uncommon in the academic papers. The researchers usually use “2)” or “b.” instead.

### Soundness
3

### Presentation
2

### Contribution
2
