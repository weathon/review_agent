# MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The generation quality of large language models (LLMs) is often improved by utilizing inference-time sequence-level scaling methods (e.g., Chain-of-Thought). We introduce hyper-parallel scaling, a complementary framework that improves prediction quality at the token level. Hyper-parallel scaling computes and aggregates multiple output proposals for a single token from the model. We implement this concept in Mixture-of-Experts (MoE) models, which we refer to as Roster of Experts (RoE). RoE is a training-free inference algorithm that turns a single MoE into a dynamic ensemble of MoEs. RoE injects controlled stochasticity into the expert routing mechanism, enabling it to sample multiple diverse experts for each token and aggregate their outputs for a more accurate final prediction.To overcome the computational cost, we introduce an efficient batching strategy and a specialized KV-caching mechanism that minimizes compute and memory overhead. For example, RoE enables a 7B MoE model to match the performance of a 10.5B MoE model while using 30% less compute for inference. These gains are achieved without any fine-tuning of model parameters.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce a training-free framework complementary to chain-of-thought scaling by enabling parallel inference in MoE via repeatedly sampling activated experts, before aggregating the outputs from multiple computational paths to produce a more accurate next-token prediction. Experiments on three different MoE models across mathematical and commonsense reasoning and code generation benchmarks demonstrate that the proposed RoE method consistently improves performance.

### Strengths
1. The paper is well written and well motivated. It complements existing inference scaling paradigms.

2. The empirical results are comprehensive and demonstrate a new axis for scaling.

### Weaknesses
1. The "equivalent model size" is a weak argument, as the authors point out the disconnect between perplexity and downstream task performance. While this remains a challenge in scaling law research, which is out of scope of this investigation, this nevertheless renders the argument less convincing. In addition, Figure 5 seems to suggest a diminishing return as the number of samples increases. What is the performance ceiling of RoE, and how does it compare to a vanilla best-of-N scaling?

2. The author presents clean cache as a major contribution, but only evaluated it on one of the three models tested without offering a clear explanation. This seems to greatly undermine the case the authors attempt to build.

3. The authors did not mention the cost associated with the Bayesian Optimization-based search. The authors present the layer-wise temperature as a parameter that requires careful tuning, but did not further discuss how difficult it is to obtain a good configuration nor how much time it takes.

4. MoE as a class of models also varies significantly in how sparse they are. More discussions on how sparsity affects performance improvements could be added to understand the tradeoff.

5. While the method is orthogonal to sequential scaling paradigms (such as CoT), iso-compute ablation studies are notably missing from the experiment sections. For instance, a simple baseline to compare against could be self-consistency.

### Questions
Please see my points above.

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
3

### Summary
This paper proposes hyper-parallel scaling, a new inference-time compute scaling paradigm that aims to improve the prediction quality at the token level, as opposed to existing test-time scaling methods that mainly work at the sequence level (chain of thought, self-consistency). The authors develop this idea to Mixture-of-Experts language models through the Roster of Experts (RoE), a training-free inference time scaling algorithm. RoE introduces randomness to the expert router using Gumbel-Top-k sampling with temperature. This creates multiple diverse expert choices for each token, then their outputs are aggregated to form a single next-token prediction. To make this practical, the authors propose batched inference so that multiple routed paths for the same token are evaluated in parallel and a "Clean Cache" KV-cache sharing method that avoids the memory blowup. Empirically, RoE is evaluated on three instruction-tuned MoE models and across math reasoning, common sense, and code-generation benchmarks. The proposed method increases the accuracy of MoE models with manageable memory and latency overheads.

### Strengths
- To my knowledge, the paper introduces a novel test-time scaling axis, per-token compute diversification compared to sequential and parallel scaling. I think that this contribution is valuable.
- RoE is training-free and it can be applied post hoc to existing MoE LLMs without any finetuning. It just modifies routing at inference, samples multiple expert pathways.

### Weaknesses
- The paper provides only one baseline, the greedy decoding. I think that it could be interesting to see how the proposed method compares with other inference-time scaling tricks.
- While the paper claims hyper-parallel scaling is domain-agnostic and could apply to other inference methods, the paper focuses only on RoE, scaling for MoE. The presentation could be improved by clarifying this scope or providing a broader set of applications of hyper-parallel scaling.

### Questions
- The Clean Cache trick assumes that only the current token needs diverse expert routing, and all candidates can share a single deterministic KV-cache history. Intuitively, that means RoE only widens computation locally at the next step, not globally across the whole generated prefix. Do you have evidence that applying stochastic routing at every step with separate KV histories would produce larger gains?

### Soundness
3

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
This paper proposes a hyper-parallel scaling method for test-time scaling. Based on RoE, it aggregates several logits for more robust predictions. The highlight is ...?
To overcome the computational cost, this paper introduces a batching strategy and a KV-caching mechanism. From experiments, the method works best on small and weak models, e.g. OLMoE-7B.

### Strengths
1) The paper is well-written, with clearly designed experiments on three LLMs.
2) Empirically, the method achieves obvious improvements on all benchmarks of OLMoE-1b-7b.

### Weaknesses
1) There is a formatting issue on both the main paper and the appendix: the header should be "ICLR 2026", instead of "ICLR 2025".
2) Lack of novelty. Adding controlled variation and caching mechanisms are common remedies, as mentioned in the introduction. 
3) In Fig. 3, the improvements are not consistent. However, the quality and efficiency breakthrough of SOTA models are desired. 
4) From the hyperparameter tuning and manually designed steps, it is unclear whether the authors have truly addressed the issue of LLM generation quality.

### Questions
1) Please analyze the inconsistent performance observed across the three backbones, considering that weaker models tend to benefit more from inference-time methods than SOTA models.
2) The sensitivity towards temperature should be an ablation study in the main paper, instead of in the appendix. Since unreasonable selection could harm the baseline, will this "task-specific temperature" undermine the robustness of your method?
3) What‘s the main design contributing to the performance gains and its rationale in your framework?
4) Is the noise scheme essential to the overall performance, or just providing limited diversity?

### Soundness
3

### Presentation
3

### Contribution
3
