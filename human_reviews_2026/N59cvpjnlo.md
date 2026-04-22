# POLLINATOR: OPTIMAL MATCHMAKING IN AN INTELLIGENCE MARKETPLACE

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
The rapid growth of the intelligence marketplace has created an abundance of Large Language Model (LLM) producers, each with different cost–performance tradeoffs, making optimal selection challenging and resource-intensive. We present POLLINATOR, a novel router that integrates a frugal, data-efficient predictor with an online dual-based optimizer. The predictor combines graph-based semi-supervised learning with an Item Response Theory (IRT) head, reducing training cost by up to 49% while improving predictive accuracy over prior state-of-the-art. The optimizer formulates matchmaking as a strongly convex problem, which allows efficient dual-to-primal conversion for real-time serving. Extensive experiments demonstrate that POLLINATOR delivers superior cost–performance tradeoffs: achieving 0.43%-1.5% gains at 71%-93% of the cost of state-of-the-art router, 3-5% gains at only 1.9-3% of the cost of the best individual producer,
and up to 10.6% higher accuracy at just 0.3-35.7% of the cost on challenging real-world benchmarks such as BFCL-V3 and MMLU-Pro. Finally, the interpretability of learned query difficulties and model abilities demonstrates POLLINATOR’s effectiveness for dynamic and cost-efficient intelligence matchmaking.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes an elastic model selection framework that dynamically allocates large models to optimize performance and cost. The idea is interesting and practically valuable for large-scale deployment. However, the paper lacks clarity on how dynamic model switching is feasible in real settings, given that invoking multiple cloud or local services incurs significant latency and token overhead. Moreover, the method’s handling of prediction–optimization mismatch is insufficiently analyzed, and scalability issues related to operational and inference costs are not well quantified. Overall, the paper presents a promising direction but requires deeper technical explanation and broader empirical validation to be fully convincing.

### Strengths
The paper addresses an important and timely problem in efficient large-model deployment by proposing an elastic selection mechanism to balance accuracy and cost. Its idea of dynamically routing tasks to different models based on predicted utility is conceptually appealing and reflects practical system needs. The formulation is well-motivated, and the integration of optimization principles into model selection demonstrates good technical insight.

### Weaknesses
The main limitation of the paper lies in the practicality and scalability of its proposed elastic model routing mechanism. Although the idea of adaptively selecting large models is appealing, the paper lacks a concrete analysis of how such routing can be implemented efficiently in real deployment scenarios. Frequent switching between models or services would inevitably introduce latency, overhead, and token costs, yet these factors are not sufficiently quantified or discussed.
﻿
Moreover, the framework’s predictive component relies heavily on estimated performance and cost metrics without a clear strategy for handling uncertainty or prediction errors. The absence of robustness tests, sensitivity analysis, and ablation studies weakens the empirical evidence. Additionally, operational constraints such as service rate limits, model drift, or dynamic pricing are not modeled or evaluated, raising concerns about the system’s stability and reliability under real-world conditions.

### Questions
1. How does the proposed elastic model selection framework manage the latency and token overhead introduced by dynamically invoking multiple large models across cloud or local services?
﻿
2. What mechanisms are used to mitigate prediction–optimization mismatch in the estimated performance and cost metrics ? Are calibration or uncertainty-aware strategies incorporated during training or inference?
﻿
3. Can the authors provide additional experiments or analysis to evaluate scalability, particularly in terms of end-to-end latency, system throughput, and robustness under dynamic model pricing or service availability changes?

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
This paper addresses the challenge of optimal LLM selection in the intelligence marketplace in which numerous LLMs with varying cost-performance tradeoffs make it difficult for users to  select the right model according to the request.  The proposed method is a novel router integrating a predictor and an online optimizer, aiming to balance inference performance, cost savings, and real-time serving capability for different scenarios.

### Strengths
The paper focuses a critical challenge in the intelligence marketplace. This is a timely problem of practical importance. 

It combines GCN) with IRT for prediction, as well as convex optimization for online serving. This integration results into significant performance improvement over existing methods.

Authors have conducted comprehensive experiments that cover a total of 14 datasets. Results demonstrate clear advantages.

### Weaknesses
Somehow the modeling comes with oversimplified cost and constraints. The paper only models cost based on token counts, did not take into account other practical factors such as hardware deployment overhead or regional pricing differences, which plays an important role in real-world applications. Other common industrial constraints are not considered in the proposed approach e.g., minimum LLM usage volume commitments. 

The paper emphasizes on engineering implementation, it lacks in-depth theoretical analysis and comparisons with RL-based dynamic routing methods.  Also, it assumes static LLM abilities, with no mechanism to adapt to LLM updates e.g., iterative model fine-tuning or sudden shifts in query types.

### Questions
Is the proposed method sensitive to the hyper parameter selection ?  
What are some alternative methods for cost optimization in LLM deployment?

### Soundness
2

### Presentation
2

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
This paper presents POLLINATOR, a system for optimal LLM matchmaking in the Intelligence Marketplace. The framework includes two key components: a) Predictor, a data-efficient prediction module that integrates a GCN with an IRT head to improve accuracy while reducing training cost; b) Optimizer, a strongly convex, dual-based optimization module that supports efficient online routing and enforces safety constraints during model selection. Experiments across multiple benchmarks show consistent improvements in cost–performance trade-offs over existing routing methods.

### Strengths
- The integration of GCN and IRT for performance prediction is innovative, effectively capturing task–model relationships while offering interpretability via parameters representing model ability and query difficulty.
- The optimization is elegantly formulated as a strongly convex program, enabling closed-form solutions for utility allocation. This avoids costly iterative optimization and supports real-time decision-making with solid theoretical grounding.
- Extensive experiments on 14 diverse benchmarks demonstrate consistent gains in both accuracy and cost efficiency compared with strong baselines.

### Weaknesses
- The paper mentions potential mismatch between predicted and actual values but does not quantify its impact on constraint satisfaction or budget adherence. A sensitivity analysis would strengthen the evaluation.
- Although Algorithm 1 is well-designed, the paper lacks runtime and complexity analysis. The scalability of the approach for large model pools (tens or hundreds of models) remains unclear.

### Questions
see above

### Soundness
3

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
This paper presents a novel router designed to select the optimal LLM producer for a request, balancing cost and performance in real-time. The system integrates a frugal predictor that combines graph-based semi-supervised learning with an Item Response Theory head. This is coupled with an online dual-based optimizer that formulates the matchmaking as a strongly convex problem for efficient, real-time serving. Experiments demonstrate superior cost-performance tradeoffs, achieving performance gains at a fraction of the cost  across various in-domain and out-of-domain benchmarks.

### Strengths
- The paper proposes a novel approach that leverages GCN with an IRT head, reducing the training cost significantly. 
- The dual serving scheme allows online optimization and offers better cost-performance tradeoff.
- Empirical results show that the method achieves superior performance gain at a lower cost.

### Weaknesses
- It's unclear how the GCN helps improve the router. The ablation study does not show any particular trend on the graph neighbour size (tab. 4), questioning its role. What if we remove the GCN part (similar to size 1), or simply apply k NN averaging? 
- The experimental setting is not presented well. How do you create the Performance-First/Cost-First/Balance setting? The paper should include a thorough comparison across the whole spectrum of performance and cost (see Figure 4 in RouterBench for example). Also, the oracle result where we choose the best and cheapest LLM for each sample should be reported for reference.
- The results look strange. How can we achieve lower costs than the cheapest model (Tab. 1 Cost-First, Tab. 2 Balanced/Cost-First, and Tab. 3)?

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
