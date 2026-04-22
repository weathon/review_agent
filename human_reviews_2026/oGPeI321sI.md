# Max-Speedup Speculative Sampling: A Generic Tree Construction Principle

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 6

## Abstract
Speculative sampling has emerged as a promising approach for accelerating large language models (LLMs) inference by leveraging a lightweight draft model to propose multiple candidate tokens, which are then verified in parallel by a target model. 
Recent methods enhance this process by structuring candidate sequences into a token tree for more efficient verification. 
However, existing tree construction methods overly rely on acceptance length as a proxy for speedup.
Such an indirect pursuit renders it challenging to achieve the optimal tree structure for maximum speedup.
In this paper, we first revisit prior approaches and find they suffer from two key limitations: analytical intractability and the assumption of node independence.
We then redefine the costs and benefits of each tree node, derive a function that characterizes the relationship between time reduction and draft length, and prove its convexity. 
Finally, we extend this analytical framework to tree structures and propose a general principle for tree construction aimed at maximizing speedup.
Applying this principle to state-of-the-art tree-based speculative sampling methods consistently delivers significant gains, improving overall performance by 4% to 14% and achieving end-to-end speedup of 1.97× to 2.68×.
The implementation is publicly available at: 
https://anonymous.4open.science/r/GTCP-CC76/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates speculative sampling for accelerating large language model inference.
The authors propose a Generic Tree Construction Principle to decide which candidate nodes should be added during tree-based speculative decoding. They provide a convex-optimization-style analysis of expected time vs. acceptance rate and claim that the derived principle can yield maximum speedups under a given cost budget.

### Strengths
The attempt to formalize benefit-cost trade-offs in tree-based speculative decoding is interesting.

### Weaknesses
1 The proposed “generic principle” essentially reformulates existing tree-based speculative sampling as an optimization problem with pruning rules. It does not introduce a fundamentally new mechanism or algorithmic insight beyond prior work (e.g., Sequoia, EAGLE-2, SpecInfer). The contribution is more of an incremental improvement rather than a conceptual breakthrough.
2 The analysis relies on assuming that each node’s acceptance probability can be estimated accurately and independently. In real LLM decoding, acceptance rates depend on context, sampling temperature, and model randomness—these assumptions rarely hold.
The convexity proof therefore has limited practical significance.
3 The paper’s tone is conversational (“we first revisit…”, “we then extend…”) and the exposition lacks precision.
Some equations and definitions are introduced loosely, and related work is not carefully positioned against the most recent 2024–2025 literature on LLM inference acceleration.

### Questions
The paper does not address when the principle might fail (e.g., under high temperature, small batch size, or hardware constraints).
Practical deployment concerns (GPU memory, caching, latency) are absent.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a generic tree construction principle (GT principle) for speculative sampling that directly maximizes inference speedup—rather than optimizing the proxy metric of average acceptance length. By redefining per-node cost and benefit and proving the convexity of time reduction with respect to tree size, the method selects only nodes with positive net benefit. Applied to both static (Sequoia) and dynamic (EAGLE-2) tree methods, it achieves consistent end-to-end speedups of 1.97×–2.68× across model sizes.

### Strengths
1. Principled formulation: The paper provides a clear, theoretically motivated criterion for tree construction, moving beyond heuristic design. It correctly identifies the core limitation of prior work—optimizing for average acceptance length as a proxy—which can be misaligned with the true objective of minimizing total inference time.
2. Strong empirical validation: Results across model sizes (7B–33B), tasks, temperatures, and batch sizes demonstrate robustness. The paper goes beyond simple benchmarking by including ablation studies on memory efficiency, sensitivity to hyperparameters (like the cost threshold `c`), and generalization across different LLM architectures (Llama-2/3, Qwen2, Vicuna).
3. Plug-and-play applicability: GT integrates seamlessly with both static (Sequoia) and dynamic (EAGLE-2) methods, showing broad utility. The simplicity of the principle—select nodes where estimated benefit outweighs cost—makes it highly practical and easy to adopt by the community.

### Weaknesses
1. The GT principle evaluates nodes based on individual net benefit, assuming independence. However, in speculative decoding trees, the utility of a node may stem from its role in enabling high-value subtrees—such as acting as a necessary prefix for multiple high-probability continuations—making isolated evaluation potentially suboptimal.
1. In the integration with EAGLE-2, the adaptive tree depth is governed solely by the acceptance probability along the leftmost path. This choice lacks theoretical justification and may overlook more informative aggregation strategies that better reflect overall tree quality at each depth.
1. The GT principle relies on accurate estimates of token acceptance probabilities to compute node benefit. Since the true acceptance rate ε is unobservable during inference, any systematic bias or variance in its proxy could degrade the effectiveness of node selection.

### Questions
1. Could the GT principle miss structurally important “bridge” nodes—those with low immediate benefit but essential for reaching high-benefit descendants—due to its greedy, node-wise selection criterion? If so, how might one extend the framework to account for subtree-level value?
2. Why is the leftmost path used as the sole indicator for adaptive depth in the EAGLE-2 integration? Have alternatives—such as using the maximum, average, or entropy-weighted acceptance probability across all nodes at a given depth—been considered, and how do they affect speedup and stability?
3. In real-world deployment, the acceptance probability must be approximated. How robust is the GT principle to inaccuracies in this estimation? Are there practical strategies (e.g., calibration) to mitigate performance degradation under estimation noise?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- The paper proposes an early stopping algorithm for tree-based speculative decoding to reduce extra draft runs, aiming to improve speed-up with only a slight reduction in acceptance.
- Claims that expected speed-up is a convex function over draft length—but this seems like a known property already.

### Strengths
- Introduces a practical idea to reduce unnecessary draft generation, which can save compute during inference.
- Focuses on speed-up optimization, which is critical for real-world deployment of speculative decoding.

### Weaknesses
- No comparison with adaptive draft length methods, despite the proposed approach being conceptually similar (stopping draft generation dynamically).
- Eq. (4) is missing $M_T$​ on $\epsilon$; text below Eq. (4) should include $M_T \epsilon$ since they come together.
- Full form of GT is unclear—should clarify if it means General Tree.
- Criticism of Sequoia in the static tree section seems invalid:
  - Sequoia is offline, so its cost is a one-time calibration, which is standard and not a runtime overhead unless extremely large.
  - The claim that Sequoia fails to find an optimal speed-up tree needs evidence.
- Eq. (10) and Eq. (11) lack intuition—currently appear arbitrary.
- Improvement in speed for static tree in Table 1 is marginal (second decimal point).
- Fig. 3 legend says “GE,” likely should be “GT.”
- Table 4 does not specify the evaluation task.
- Overall, the paper does not convincingly show significant gains compared to existing methods.

### Questions
- Please check weaknesses
- Can the authors include SpecBench comparisons using EAGLE2/3 with LLaMA3 family, I'm curious to see performance on models with  large vocabulary models?

### Soundness
2

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
2

### Summary
This paper proposes a "Generic Tree Construction Principle" (GT principle) for tree-based speculative sampling in large language models (LLMs), aiming to maximize inference speedup by selecting nodes with positive net benefits based on redefined costs and benefits. The authors derive a convex function for time reduction, extend it to tree structures, and integrate it into existing methods like Sequoia and EAGLE-2, claiming 4-14% performance gains and speedups up to 2.68×. Experiments are conducted on benchmarks like Spec-Bench with models from 7B to 33B parameters.

### Strengths
1. This paper has a reasonable structure and clear writing. The authors elaborate on the research methods in detail, making it highly readable.
2. The proposed method can be easily integrated with existing speculative decoding algorithms to achieve performance improvements.
3. The experiments are comprehensive, covering multiple model scales (Vicuna with 7B–33B parameters), tasks (machine translation, transformation, summarization, question answering, multi-turn dialogue, and retrieval-augmented generation from Spec-Bench), and baseline methods (Sequoia, EAGLE-2, with additional methods including EAGLE, EAGLE-3, Medusa, and Hydra in the appendix).

### Weaknesses
1. Although the method proposed by the authors achieves a 4%-14% performance improvement compared to the baselines, the improvement margin is relatively small. The end-to-end speedup is largely contributed by the baseline methods, and the enhancement brought by the authors' method to the baselines is quite limited. Additionally, the improvement in the R value (memory-to-speedup ratio) is negligible.
2. Code generation tasks (such as the HumanEval benchmark) are not included, and there are few experimental results on larger models (e.g., 70B models).

### Questions
1. Could you provide the quality metrics for each benchmark in Spec-Bench (such as BLEU for the WMT task, accuracy for the GSM8K task, etc.) to demonstrate that the GT principle does not reduce the utility of the model?
2. The authors claim that "all experiments were repeated 5 times under the same hyperparameters, and the final results are presented as averages". Could you please provide the variance of the 5 experiments?

### Soundness
3

### Presentation
4

### Contribution
3
