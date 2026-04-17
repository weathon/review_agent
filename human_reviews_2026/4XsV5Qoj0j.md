# Textual Equilibrium Propagation for Deep Compound AI Systems

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Large language models (LLMs) are increasingly deployed as part of compound AI systems which coordinate multiple modules (e.g., retrievers, tools, verifiers) over long-horizon workflows. Although recent frameworks that propagate textual feedback globally (e.g., TextGrad make it feasible to optimize such pipelines, we identify two depth-scaling failure modes in long-horizon agentic workflows: 1) exploding textual gradient, where textual feedback grows exponentially with depth, leading to prohibitively long message and amplifies evaluation biases; and 2) vanishing textual gradient, where limited long-context ability causes models overemphasize recent or early feedback, while compression of lengthy feedback causes downstream messages to lose specificity gradually as they propagate many hops upstream. To mitigate these issues, we introduce Textual Equilibrium Propagation (TEP), a local learning principle inspired by Equilibrium Propagation in energy-based models. TEP includes two phases: 1) a free phase where a local LLM critics iteratively refine prompts until reaching equilibrium (no further improvements are suggested); and 2) a nudged phase which applies proximal prompt edits with bounded modification intensity, using task-level objectives that propagate via forward signaling rather than backward feedback chains. This design supports local prompt optimization followed by controlled adaptation toward global goals without the computational burden and signal degradation of global textual backpropagation. Across long-horizon QA benchmarks and multi-agent tool-use dataset, TEP consistently improves accuracy and efficiency over global propagation methods such as TextGrad, with gains that increase at greater depths, while preserving the practicality of black-box LLM components in deep compound AI system.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The work (i) formalizes exploding and vanishing textual gradients in global textual backprop frameworks like TextGrad, (ii) introduces a depth-robust, node-local alternative inspired by Equilibrium Propagation, and (iii) reports consistent empirical gains on multi-step QA, retrieval, and code-generation suites, alongside ablations and depth-scaling analyses.

### Strengths
1. This paper clearly identifies and formalizes depth-dependent pathologies (exploding/vanishing textual gradients) with simple but compelling measures for message length and specificity.

2. This paper presents a principled, modular algorithm (free-phase local refinement; nudged-phase bounded edits) with parallelizable node-local procedures and clean pseudocode.

3. This paper provides breadth across tasks (PubMedQA, STARK-PRIME, HotpotQA, BigCodeBench) and shows consistent improvements, with larger gains at greater depth.

### Weaknesses
1. The novelty is unclear because the “free phase + nudged edits” design is very close to existing self-refinement and actor–critic prompt editing methods, and the paper does not include matched-budget comparisons against those strongest local baselines.

2. The definition of “equilibrium” relies on a heuristic stopping rule (score stabilization over three evaluations or a cap of 20 iterations), and the paper does not provide a sensitivity analysis to show that the results are not artifacts of this choice.

3. The computational cost of TEP is underreported; running 20 free iterations plus 40 nudged iterations per node appears expensive, and the paper does not normalize token, latency, or dollar costs against the baselines.

### Questions
1. Can you precisely distinguish TEP from prior local methods such as Self-Refine, PACE (actor–critic editing), and verifier-guided prompt tuning, beyond the free/nudged phrasing?

2. Which concrete properties of TEP (and not general self-refinement) are necessary for the reported gains?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the issue of vanishing and exploding textual gradients in compound AI systems by introducing Textual Equilibrium Propagation (TEP). This framework, inspired by equilibrium propagation in energy-based models employs two phases: iterative refinement and nudging. Experimental results indicate that the proposed framework achieves superior performance over TextGrad and offers a robust approach for black-box optimization.

### Strengths
- The paper is clearly written, and the inclusion of illustrative examples enhances clarity and comprehension.
- Extensive experiments are conducted, demonstrating consistent empirical gains.
- Evaluation includes both prompt optimization and solution optimization tasks, even though the latter is limited to two datasets.

### Weaknesses
- The rationale behind the choice of large language models (LLMs) is insufficiently explained.
- There is limited discussion of potential LLM biases and how they may influence results.
- Details regarding hyperparameter selection and tuning are lacking.
- The paper does not analyze the trade-off between performance gains and the associated increase in computational cost.

### Questions
- What is the rationale for selecting the specific large language models (LLMs) used for different tasks?
- How does the size of the LLM influence the performance of various components within the proposed framework?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles a timely and relevant problem: the optimization of deep compound AI systems that perform multi-step reasoning and multi-tool coordination at realistic scales. The authors identify that current textual optimization methods—particularly TextGrad—encounter depth-scaling failures analogous to the vanishing and exploding gradient problems in neural networks. To address this, they propose **Textual Equilibrium Propagation (TEP)**, a two-phase optimization method inspired by equilibrium propagation in energy-based models.
In TEP, global feedback propagation is replaced by local refinement in a “free phase”, where each node independently updates its prompt using a local LLM critic until equilibrium is reached, and a “nudged phase,” where small task-specific perturbations are introduced to align local equilibria with the global objective. While the authors cite conceptual inspiration from equilibrium propagation, the mathematical connection is loose—TEP does not define or optimize an explicit energy function ($E(x, \theta)$).

### Strengths
1. The paper clearly articulates the depth-scaling issues of existing methods and communicates them effectively through illustrative figures.
2. The proposed method has well-defined computational properties: local optimization has bounded token complexity ($O$(1)) per node versus exponential (*O*($2^{depth}$)) complexity in multi-hop textual backpropagation, allowing scalability to deeper systems (10–20+ nodes).
3. Empirically, TEP demonstrates strong performance improvements over TextGrad on multiple benchmarks (PubMedQA, STaRK-PRIME, HotpotQA, BigCodeBench).
4. The depth-scaling experiments (Fig. 3) are particularly compelling: TextGrad’s feedback length grows from 2K to 32K tokens with decreasing update effectiveness (36%→5%), while TEP maintains near-constant token complexity and only modest performance degradation (37%→33%).
5. The ablation studies on TEP’s free and nudged phases effectively show that both components are necessary for the observed gains.

### Weaknesses
1. The connection to classical Equilibrium Propagation (Scellier & Bengio, 2017) remains metaphorical rather than formal. TEP does not define a differentiable energy function, nor does it derive a learning rule corresponding to a negative gradient step. As a result, claims of “consistent descent direction” and “convergence under standard conditions” are not theoretically substantiated in the paper.
2. The “convergence analysis” in Appendix A is largely qualitative. It assumes information contraction and bounded channel capacity but does not provide a formal proof of convergence or conditions guaranteeing equilibrium stability.
3. Computational cost analysis is incomplete. While token complexity is analyzed, the paper omits wall-clock time or total iteration comparisons against baseline methods. The trade-off between token efficiency and the number of required refinement steps remains underexplored.
4. The experimental section would benefit from deeper analysis of model generality. Although the authors report results using different base LLMs (GPT-4o, Claude 3 Haiku, Llama 3.2), there are no controlled ablations where the same task is optimized using multiple base models. This weakens the generalization claim.
5. It is unclear whether the same model or a separate model serves as the critic LLM for local refinement. Clarifying this choice (and its impact on stability or bias) would help contextualize results.

### Questions
1. Could the authors provide more rigorous mathematical support for the claimed “local optimization until equilibrium” behavior? In particular, can they formalize the contraction assumptions or provide sufficient conditions under which the refinement operator is guaranteed to converge?
2. Clarify whether “consistent descent direction” refers to an empirical observation or a theoretical guarantee. If the latter, please include a proof or clearly stated theorem.
3. Include wall-clock comparisons and iteration counts to quantify the trade-off between per-node efficiency and convergence speed.
4. Strengthen generalization claims by running at least one benchmark task with multiple base models under the same conditions.
5. Specify whether the critic LLM is identical to the base model or distinct, and discuss how this choice affects equilibrium stability and sample efficiency.

### Soundness
2

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
4

### Summary
This paper investigates textural propagation in large language models (LLMs) to enhance compound AI systems. The authors point out that existing textural propagation approaches, such as textgrad, suffer from two key problems: exploding textural gradients and vanishing textural gradients in the middle context. They propose Text Equilibrium Propagation (TEP), a local learning principle in which a local LLM critic iteratively refines prompts until no further performance gain is observed. The method also introduces forward propagation of prompt edits. Experimental results show that TEP achieves larger improvements than previous methods, especially as task depth increases.

### Strengths
1.The problem is well motivated.
2.The proposed method is conceptually sound.

### Weaknesses
1. It is not entirely clear how the local critic model operates. Does it require ground truth labels for intermediate outputs?
2. The baseline method Revolve[1] is mentioned. 
3. While performance gains are significant, the paper does not analyze the tradeoff between accuracy and computational cost,e.g. how does the computational cost look like for each of the baseline and TEP?

[1]Revolve: Optimizing AI Systems by Tracking Response Evolution in Textual Optimization

### Questions
1.Can you explain what is the main difference between the forward signaling and backward feedback chains?
2.In Table 1, the improvement of TextGrad over Chain of Thought (CoT) is relatively small. Do you have any insights into why this happens?

### Soundness
3

### Presentation
3

### Contribution
3
