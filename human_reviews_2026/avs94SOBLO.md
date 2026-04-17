# Explaining the Reasoning of Large Language Models Using Attribution Graphs

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large language models (LLMs) exhibit remarkable capabilities, yet their reasoning remains opaque, raising safety and trust concerns. Attribution methods, which assign credit to input features, have proven effective for explaining the decision making of computer vision models. From these, context attributions have emerged as a promising approach for explaining the behavior of autoregressive LLMs.
However, current context attributions produce incomplete explanations by directly relating generated tokens to the prompt, discarding inter-generational influence in the process. To overcome these shortcomings, we introduce the Context Attribution via Graph Explanations (CAGE) framework. CAGE introduces an *attribution graph*: a directed graph that quantifies how each generation is influenced by both the prompt and all prior generations. The graph is constructed to preserve two properties---causality and row stochasticity. The attribution graph allows context attributions to be computed by marginalizing intermediate contributions along paths in the graph. Across multiple models, datasets, metrics, and methods, CAGE improves context attribution faithfulness, achieving average gains of up to 40%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces CAGE, a context attribution method that builds a causal row stochastic graph over prompt and generated tokens and propagates influence to capture prompt to output and intergenerational effects in autoregressive language models. Experiments on facts, math, and multihop question answering with llama and qwen show consistent gains, with large faithfulness improvements and perfect wins in some settings.

### Strengths
* **Clear problem and fix:** Prior methods miss how earlier outputs affect later ones. CAGE builds a simple cause-and-effect graph that captures this and solves the gap directly.
* **General and clean:** It works on top of any attribution method, with straightforward normalization and a closed-form formula to spread influence, making it easy to reuse.
* **Strong results:** Consistent gains across models and datasets—up to 40% better on faithfulness, 17/20 wins on attribution coverage, and 40/40 wins in some tests.

### Weaknesses
* The paper enforces a row stochastic constraint (incoming edge weights sum to 1) without proper mathematical justification. This creates a critical logical flaw. The constraint assumes that 100% of a token's generation can be attributed to previous tokens, which contradicts how LLMs actually work. LLMs have inherent randomness, learned biases, and model parameters that contribute to generation beyond just the input context. By forcing weights to sum to 1, the method artificially inflates attribution scores when there are few contributing tokens, potentially creating misleading explanations

* The normalization in Equation (1) using Φ(x) = max(x, 0) can cause division by zero issues if all attribution scores for a token are negative. This is a serious mathematical flaw in the method.

* The mathematical formulation for computing context attribution has a fundamental error:
a^xτ = Aτ,: + Σ(i=1 to τ-1) Aτ,i · Ai, This assumes attribution influences combine linearly and additively, which is problematic because It treats intermediate tokens as simple linear transmitters of influence, ignoring the non-linear transformations in transformer architectures

* Evaluation is underpowered: no ablations relaxing causality or row-stochasticity to test necessity; reported gains lack confidence intervals or significance tests on the 500/250-example sets; key recent baselines (e.g., Barkan et al., EMNLP 2024) are omitted; and results exclude larger 14B–80B models—leaving generality and robustness unproven.

[1] Oren Barkan, Yonatan Toib, Yehonatan Elisha, Jonathan Weill, and Noam Koenigstein. Llm explainability via attributive masking learning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 9522–9537, 2024.

### Questions
Same as weakness

### Soundness
2

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
CAGE builds an attribution graph over prompt and generated tokens (causal, row-stochastic), then marginalizes along paths to explain outputs; this improves attribution coverage and perturbation-based faithfulness versus row-summation baselines across Llama/Qwen on Facts, Math, and MorehopQA.

### Strengths
* Captures inter-generational influence (not just prompt→token), aligning with CoT behavior. 

* Principled construction (nonnegative, row-stochastic adjacency; DAG; closed-form total influence)

* Consistent empirical gains (AC ↑ max/avg 134%/40%; faithfulness wins 40/40; up to 30%/11% improvement). 

* Method-agnostic: wraps perturbation, CLP, IG, Attn×IG, ReAGent.

### Weaknesses
* Because CAGE applies Φ(x)=max(x,0) and then row-normalizes the attribution table into a stochastic adjacency, it discards inhibitory (negative) effects and collapses absolute magnitudes into relative shares that must sum to one, so negative influence and true effect size cannot be represented.

* The faithfulness tests remove entire prompt sentences and replace them with EOS tokens, which can introduce distribution shift and confound measured effects with artifacts of degraded input rather than true causal importance

* The method assumes that “influence” from earlier words simply adds up and passes straight through the sequence. Transformers don’t work that way, they apply non-linear, state-dependent updates where effects can interact or cancel.

### Questions
* Please add ablation studies to increase the faithfulness of the method.

* Also please refer to weakness as questions.

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
This paper introduces Context Attribution via Graph Explanations (CAGE), a novel framework for explaining the reasoning of autoregressive Large Language Models (LLMs). The authors argue that existing "row attribution" methods provide incomplete explanations because they only measure the direct influence of prompt tokens on each generated token, ignoring the crucial influence of intermediate generations. The CAGE framework addresses this limitation by constructing an attribution graph, a directed acyclic graph that models the causal flow of influence throughout the entire generation process. To generate a final context attribution for a specific output, CAGE marginalizes the influence of intermediate tokens by tracing all causal paths from the output back to the prompt tokens.

### Strengths
- Current "row attribution" methods treat each generated token's relationship to the prompt in isolation, which is fundamentally incompatible with the sequential, stateful nature of autoregressive generation, especially in chain-of-thought processes. The proposed CAGE framework is a novel and principled solution to this problem, utilizing a causal perspective on the generation process.
- The paper is well-written and easy to follow. The motivation is clearly laid out in the introduction, and the distinction between CAGE and prior work is made both in the text and through effective visualizations. Especially, Figure 3 greatly aids in understanding the influence propagation process.
- By providing a tool to trace the flow of information through intermediate generation steps, CAGE claims to offer a more faithful and complete picture of a model's reasoning process than was previously possible with attribution methods. The framework is general and can be applied on top of any existing or future base attribution method.

### Weaknesses
- The model of influence propagation, calculated via matrix multiplication ($A_{\tau,:}^{\prime}=A_{\tau,:}+A_{\tau,\tau-1}\cdot A_{\tau-1,:}$), implicitly assumes that influence propagates through the network in a way that can be modeled by a linear combination of path weights. The paper could be strengthened by acknowledging this as a simplifying assumption and briefly discussing why it is a reasonable one in this context, or contemplating what might be lost by this linearization of the influence flow.

- The construction of the adjacency matrix $A$ involves clipping all attribution scores to be non-negative ($\Phi(x) = \max(x, 0)$) before row-normalization. Many attribution methods can produce meaningful negative scores, indicating that a feature actively suppresses a certain output. By discarding this information, the model may lose a crucial part of the explanation. For example, in the "Facts" dataset, a model might rely on already-generated facts to down-weight the probability of generating them again. This "negative" influence is an important part of the reasoning process that is currently ignored. The paper would benefit from a discussion of this design choice and its potential limitations.

- The context attribution evaluation relies heavily on the Attribution Coverage (AC) metric, which is newly introduced in this paper. While novel metrics can be valuable, their introduction requires strong justification. The core assumption of the AC metric—that attribution should be uniformly distributed across all ground-truth sentences—is not well-defended and may not be appropriate for complex reasoning tasks. It is highly plausible that certain pieces of information are more pivotal than others. By penalizing non-uniform distributions, the AC metric may unfairly disadvantage methods that correctly identify and focus on the most crucial context. The paper would be more convincing if it either provided a stronger theoretical justification for the uniformity assumption or supplemented the AC results with a more standard metric.

### Questions
- The AC metric, as defined in the paper, rewards explanations where attribution is spread uniformly across all ground-truth sentences. However, in complex reasoning tasks, it's plausible that some pieces of evidence are more critical than others. Why is uniform attribution considered an ideal property, rather than allowing for a non-uniform distribution that might more accurately reflect the model's focus? Furthermore, could the authors provide the rationale for selecting the specific range $[\frac{1}{2}\mathbb{E}(a_{GT}), \frac{3}{2}\mathbb{E}(a_{GT})]$ for this metric?
- To isolate the goal of identifying all relevant prompt sentences, have the authors considered an alternative metric that measures coverage without enforcing a uniformity constraint? For example, a metric based on the average rank of ground-truth sentences (when all prompt sentences are sorted by their attribution scores) could provide a more direct assessment of whether an explanation method successfully highlights the most important information.
- The faithfulness evaluation in line 463 mentions calculating the "area under this perturbation curve". Could the authors explicitly define the axes of this curve? Based on the procedure described, is it correct to assume the x-axis represents the fraction of prompt sentences removed and the y-axis represents the probability of generating the original output?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes CAGE (Context Attribution via Graph Explanations), a framework for explaining the reasoning process of autoregressive large language models (LLMs) through attribution graphs. The authors argue that existing row-wise attribution methods fail to effectively capture inter-generational dependencies during token generation, resulting in incomplete and inaccurate explanations. CAGE addresses this issue by constructing a directed acyclic graph (DAG) that preserves both causality and row stochasticity, and by marginalizing intermediate contributions along graph paths to compute contextual attributions. Experiments across multiple models and datasets demonstrate that CAGE achieves significant improvements in attribution faithfulness and coverage.

### Strengths
1. Clear motivation and presentation. The paper articulates its motivation clearly and presents the research problem and solution in a well-structured and intuitive manner.

2. The empirical evaluations cover four models, three different task datasets, five base attribution methods, and multiple evaluation metrics. CAGE achieves the best performance in 85% of comparisons (17/20 on the AC metric, 40/40 on faithfulness), lending strong empirical support to its claims.

### Weaknesses
1. Limited technical novelty and theoretical depth. While the paper identifies clear shortcomings in row-wise attribution methods, the proposed CAGE framework primarily relies on standard graph-based computations (e.g., path accumulation operations). The approach appears straightforward. The authors could further clarify the technical or theoretical challenges involved in the research, and articulate any new conceptual insights it provides regarding the semantics of contextual attribution.

2. Potential issues with the AC metric design. The AC metric assumes that ground-truth sentence attributions follow an approximately uniform distribution. However, in reality, different sentences naturally contribute unequally to overall generation. This assumption may unintentionally penalize explanations that accurately reflect such differences in importance.

### Questions
1. What is the main challenge and novelty of the CAGE framework?

### Soundness
3

### Presentation
3

### Contribution
2
