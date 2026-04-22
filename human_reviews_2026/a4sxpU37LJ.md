# BUDDY: BUdget-Driven DYnamic Depth Routing for Adaptive Large Language Models Inference

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Large language models require substantial computational resources for inference due to their massive number of parameters. Model layer pruning accelerates inference by eliminating redundant layers. However, existing layer pruning methods fail to meet users' flexible budget constraints and lack the ability to adaptively adjust the inference path. To address these issues, we propose Buddy, a budget-driven and adaptive inference framework. Specifically, we design a Decision Module that adaptively selects important layers to execute based on user input while satisfying a given budget constraint. Additionally, Buddy reuses the KV cache from the first layer and dynamically updates the context during inference, enabling adaptive adjustments to the inference path based on evolving contextual information. Furthermore, when no explicit budget is provided, a Budget Predictor automatically determines an appropriate inference cost to achieve an optimal trade-off between performance and computational efficiency. Extensive experiments on the Llama model demonstrate that Buddy consistently outperforms baseline methods under various pruning configurations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Buddy, a budget-driven adaptive inference framework with: (1) a Decision Module for input-dependent dynamic layer selection, (2) a Dynamic KV cache reuse and context updating mechanism, (3) a Budget Predictor for automatic cost determination.

### Strengths
1. Novel decode-adaptive mechanism: The framework dynamically adjusts layer selection during the decode phase (not just prefill), which represents an interesting and innovative approach to inference optimization.
2. Automatic Budget control: Automatic control over executed layers provides benefits for real-world deployment needs.
3. KV cache reuse: Leveraging the first-layer KV cache for global context is a low-overhead solution.

### Weaknesses
1. The experiments conducted on Llama1/Llama2, which are a bit old， Newer models (e.g., Llama3, Qwen), or thinking models (e.g., o1, DeepSeek-R1) present more challenging scenarios and would provide more relevant validation for current deployment needs.
2. The gain is a bit marginal. ShortGPT actually outperforms Buddy at 12.5% sparsity (63.16 vs 62.63), and Buddy shows only marginal improvements at higher sparsity levels.
3. Cost analysis: What is the memory overhead of maintaining the KV cache for routing decisions? Detailed latency comparisons between static and this dynamic approach are missing.
4. Only common sense reasoning benchmarks are reported. These relatively simple tasks may not effectively demonstrate the benefits of dynamic routing. (like in Figure 4, it seems all the tasks tend to prune the later layer) I would wonder if this approach would benefit more on tasks requiring multi-step reasoning or long-context understanding from decode-adaptive mechanisms.

### Questions
1. The Decision Module training process lacks details: Which dataset was used for training? How many samples were required? How was the ground truth routing path obtained?
2. Ablations comparing different KV cache selection strategies: Why specifically use the first layer's KV cache? What about other layers?
3. In real deployment, contexts can shift completely (e.g., switching from technical discussion to casual chat, or processing unrelated prompts in the same session).  Would KV cache from irrelevant prior contexts influence layer selection?

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
This paper introduces a budget-driven and adaptive inference framework for large language models. It dynamically selects important layers during inference through a Decision Module, reuses and updates KV caches for contextual adaptation, and predicts budgets when unspecified.

### Strengths
Quality

1. The paper is well written and easy to follow.

Significance

2. The analysis showing that the optimal routing path can change during decoding is interesting and provides new insight into adaptive inference.

### Weaknesses
Originality

1. The gating-based layer skipping mechanism is not entirely novel. Similar techniques have been explored in prior works [1-4].

Quality

2. The model used in experiments is somewhat outdated, which limits the relevance of the reported results.

Clarity

3. The paper does not clearly explain how the Decision Module is trained, leaving ambiguity around the optimization and supervision of the gating mechanism.
4. The calibration (validation) set needs to be described in more detail for better clarity.

Significance 

5. The experimental results are not strong enough to convincingly highlight the benefits of the proposed framework. Evaluations on more challenging or reasoning-intensive tasks could better showcase its advantages.
6. The Budget Predictor is presented as part of the framework but appears tangential to the main contribution, with minimal results shown in the main paper.

[1] DASH: Input-Aware Dynamic Layer Skipping for Efficient LLM Inferencewith Markov Decision Policies  
[2] DiffSkip: Differential Layer Skipping in Large Language Models  
[3] What Layers When: Learning to Skip Compute in LLMs with Residual Gates  
[4] Skip Transformers: Efficient Inference throughSkip-Routing

### Questions
1. How is the Decision Module trained? Please clarify the optimization process for the gating mechanism.
2. The gating mechanism for layer skipping has been explored in prior works. The authors are encouraged to discuss these related works [1–4] in the related work section and include some of them as baselines in the experiments.
3. Minor typo: remove the duplicate period in “a single deployed LLM to serve heterogeneous budgets..”.

[1] DASH: Input-Aware Dynamic Layer Skipping for Efficient LLM Inferencewith Markov Decision Policies  
[2] DiffSkip: Differential Layer Skipping in Large Language Models  
[3] What Layers When: Learning to Skip Compute in LLMs with Residual Gates  
[4] Skip Transformers: Efficient Inference throughSkip-Routing

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
4

### Summary
This work introduces a dynamic layer-skipping framework that enables each input to traverse different Transformer blocks while satisfying a given budget constraint. Given an input sequence, the KV vectors from the first layer are extracted to compute a global context, which is then fed into a decision module to determine the most important layers to execute. During the decoding phase, the global context is updated using the latest token’s KV cache, allowing the decision module to refresh the depth path dynamically. When no user-specified budget is provided, a budget predictor is trained to estimate an appropriate number of layers based on the input’s KV cache. Experiments are conducted on the LLaMA family.

### Strengths
This work effectively analyzes the varying importance of Transformer blocks across different inputs and throughout the decoding process. It presents a principled and lightweight approach to dynamic layer selection and budget-aware inference.

### Weaknesses
- While I find the motivation of this work compelling, I am concerned about whether batch processing can be efficiently realized for higher throughput, particularly when different samples (e.g., sample 1 and sample 2) have distinct depth paths. This may also lead to inefficient parallelism and increased complexity in cache management.
- The main results in Table 2 do not appear remarkably superior. The performance gap between ShortGPT (static pruning) and the proposed method seems rather marginal.
- It appears that in the Decision Module, the model-predicted scores from the lightweight MLP are combined with the prior scores. However, it was difficult to understand whether the MLP in the Decision Module itself is trained through any learning process. In contrast, the Budget Predictor is clearly described as being trained using GRPO.
- I believe Table 2 reports results under specified budgets, and I could not find corresponding results or analysis for the Budget Predictor.
- The experimental validation seems relatively weak. It would strengthen the work to demonstrate the applicability of the proposed method on other popular LLMs such as Qwen or Phi, which would further support the generality and effectiveness of the approach.
- Minor
  * Softmsax in Eqn (5), 5p -> Softmax
  * I was not able to find which specific LLaMA model was used for the analysis in Figures 1 and 2.

### Questions
Please refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Buddy, a budget-driven dynamic depth routing framework that enables adaptive inference for large language models under explicit compute constraints. Unlike prior static or dynamic pruning methods that either fix the sparsity pattern or lack control over computation, Buddy introduces three coordinated modules: (1) a Decision Module that scores and selects Transformer layers based on the current context while strictly satisfying user-defined budgets, (2) a KV-aware Planner that reuses the first-layer kv cache to provide global context during decoding, and (3) a Budget Predictor trained with Group-Relative Policy Optimization (GRPO) to automatically choose an appropriate inference cost when no budget is provided.

### Strengths
1. The paper convincingly identifies two under-addressed challenges in dynamic depth pruning: input adaptivity and decode adaptivity, and motivates them with quantitative observations on layer importance variation (Figures 1–2). 

2. Buddy provides deterministic compute control. Its Top-k selection directly enforces user-specified budgets via a single model, while the Budget Predictor offers flexibility when explicit constraints are absent, enhancing the practical usability.

3. The paper presents comprehensive analyses and ablation studies that systematically align with the challenges and motivations outlined earlier, empirically verifying how each module of Buddy contributes to input adaptivity, decode adaptivity, and budget-controlled inference.

### Weaknesses
1. The paper didn’t specify how the Decision Module’s MLP scorer. Even though equations (4–5) describe inference-time scoring and Top-k selection, no loss, gradient flow, or differentiable approximation is presented in the paper. (I checked the code in model/buddy_model.py in the supplementary material; the authors used STE). Furthermore, no cost analysis of training the Decision Module was provided.

2. The evaluation is confined to the LLaMA-2 family, which is relatively outdated. Assessing Buddy on more recent architectures (e.g., LLaMA-3 or Qwen-series models) would provide stronger evidence of its practical effectiveness and generalizability.

3. The performance improvement over competing methods is modest at certain sparsity levels. For instance, at 12.5 % sparsity, Buddy achieves an average score of 62.63 compared to 63.16 for ShortGPT, suggesting that the claimed superiority is not consistent across all regimes.

4. While the paper emphasizes decode-adaptive routing, quantitative generation results are limited to SamSum ROUGE scores; broader decoding-style evaluations (e.g., long-context, reasoning, or open-ended tasks) are absent, leaving unclear how much decoding adaptivity benefits real-world generation workloads.

5. The reported speedups (Table 3, Table 8) show modest decode-phase gains at low sparsity (×1.01–×1.19), but omit breakdowns of routing overheads (Decision Module forward, Top-k selection, synchronization, and so on). Without detailed latency and memory profiling, it is hard to assess actual system-level efficiency.

### Questions
1. In Figure 1, the per-layer removal order distribution on WikiText-2 indicates that layer 1 often ranks among the least important layers. However, the authors later state that both the first and the last blocks are excluded from pruning due to their critical impact on overall model quality. Does the author have any comments on that?

2. How does the Decision Module get trained? Is the Decision Module optimized jointly with the LLM? What will be the dataset and loss then?

3. What is the deployment cost of Buddy? For example, what is the training cost of the Decision Module? Does it only fit one model, thus individual training is required?

4. Has Buddy been tested beyond the LLaMA-2/1 family to evaluate its generality? For instance, how would it perform on newer architectures like LLaMA-3 or Qwen-3, and on reasoning-intensive generation tasks such as GSM8K or MATH?

Typos: “Softmsax” (line 244) in equation 5 should be Softmax.

### Soundness
2

### Presentation
3

### Contribution
3
