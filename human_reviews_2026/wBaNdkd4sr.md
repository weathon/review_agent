# Runtime Adaptive Pruning for LLM Inference

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. 
Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests.
To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. 
Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV‑cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter‑light attention layers dominate KV‑cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state.
Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes RAP (Runtime-Adaptive Pruning), a reinforcement learning-based framework for dynamically pruning Large Language Models (LLMs) during inference. Unlike existing static pruning methods, RAP adapts pruning decisions based on runtime conditions including memory constraints, batch sizes, and sequence lengths. The framework uses Greedy Sequential Importance (GSI) analysis to iteratively evaluate block importance and an RL agent to select which transformer blocks to prune. While the paper shows improvements over baselines on Llama and Qwen models, there are significant concerns about the evaluation methodology and practical applicability.

### Strengths
1. The paper effectively demonstrates that memory bottlenecks shift dynamically between parameters and KV cache depending on workload characteristics.This is a valuable insight for the community.
2. The evaluation covers multiple model families (Llama2, Llama3, Qwen) and includes both generation quality metrics and downstream task performance across 7 benchmarks.
3. The three key observations in Section3 provide compelling evidence for adaptive approaches, particularly the insight that KV cache becomes the dominant memory bottleneck at larger batch sizes and sequence lengths.

### Weaknesses
1. The paper fails to properly isolate the RL agent's contribution. While it compares against one-shot GSI and random selection, it critically lacks comparison with iterative GSI without RL (i.e., greedily removing blocks with highest GSI scores and re-evaluating). The comparison to random selection is hardly competitive and doesn't demonstrate the value of the learned policy. Without this ablation, it's unclear whether the RL agent adds any value over simply following GSI scores greedily.

2. Computing GSI requires running inference through the full model initially, meaning the method can only be deployed on machines with enough memory for the unpruned model. This defeats the primary purpose - you cannot use RAP on memory-constrained devices where the full model doesn't fit, which is exactly where such methods are most needed. This is a critical flaw that severely limits practical deployment.

3. The paper claims minimal overhead by focusing on the RL agent's 18K parameters, but completely ignores that GSI requires multiple forward passes through the model for importance evaluation. Each GSI iteration requires a full forward pass, making the actual latency overhead potentially orders of magnitude higher than reported. This is a serious misrepresentation of the computational cost.

4. The paper doesn't clarify which baseline methods were designed to handle parameters + KV cache pruning vs just parameter pruning. If baselines only prune parameters while RAP prunes both, this creates an unfair advantage. Even if all methods prune parameters + KV cache but baselines were not designed to work for KV cache, it is still an unfair comparison. The evaluation protocol makes KV cache pruning the most important component, potentially biasing results if baselines aren't designed for this

5. The paper's valuable contribution - demonstrating that "KV cache pruning is an important part of model pruning for memory optimization" - is buried in the experimental analysis rather than being a central claim. This insight about the dominance of KV cache in memory bottlenecks could be the paper's great contribution if properly emphasized.

6. GSI is essentially iterative pruning with re-evaluation (a known technique), and the RL formulation uses standard DQN without innovations. The combination doesn't justify the complexity when simpler approaches might work equally well.

7. No evaluation on truly dynamic workloads despite "runtime-adaptive" claims

### Questions
1. Can you provide results for iterative GSI without RL (greedily selecting highest GSI scores with re-evaluation)? This is essential to understand if the RL agent adds value beyond following GSI scores.

2. How do you address the fundamental issue that GSI requires full model memory initially? Can the method work on devices where the full model doesn't fit? If the target deployment device cannot run the full model, how can RAP be used at all given GSI's requirements?

3. What is the actual wall-clock latency overhead including all GSI computations? How many forward passes are required in practice?

4. Please provide a clear table showing: 1/ which baseline methods prune parameters only vs parameters + KV cache, 2/ which methods were originally designed for KV cache pruning, 3/ whether you modified any baselines to handle KV cache

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
3

### Summary
The paper proposes RAP, a runtime-adaptive pruning framework that (i) computes Greedy Sequential Importance scores for FFN/MHA blocks; and (ii) uses a lightweight RL controller to pick a pruning policy per request under a memory budget that includes both parameters and KV-cache.

### Strengths
1. Considering parameters and KV cache as the target is novel as most pruning work optimizes only weights.

2.  Design with MLP with small overhead makes it easy and efficient to deploy.

3. The ablation study shows the effectiveness of this method.

### Weaknesses
1. Only zero-shot short-answer benchmarks. But long-context tasks (where KV matters) or real generation quality would better show the purported advantage.

2. Need end-to-end latency and throughput comparison.  Real-world servers aslo care about tokens/sec and tail latency besides memory savings.

3. If heads or layers are dropped at runtime, how are pre-existing KV tensors handled across decoding steps?

4. If GSI already orders blocks and the agent “iteratively removes the least important,” where does RL refine this order?

### Questions
See Weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes RAP,  framework for the runtime-adaptive pruning of Large Language Models (LLMs) based on RL. The core motivation is that real-world inference workloads and system memory availability are highly dynamic, which static pruning strategies cannot accommodate. RAP introduces a Greedy Sequential Importance (GSI) algorithm to better assess block importance and an RL agent that observes real-time system state and request characteristics to dynamically select which MHA or FFN blocks to prune, covering input-driven and system-level variance. Experiments show that RAP outperforms static pruning baselines under this budget-aware evaluation.

### Strengths
- The shift from evaluating at a "fixed sparsity ratio" to a "fixed memory budget" is interesting. It more accurately reflects the deployment constraints on resource-limited devices.
- The results clearly show that RAP makes more intelligent pruning decisions than static baselines, especially under aggressive memory budget.
- The paper includes thorough ablation studies that demonstrate the necessity of both the GSI component and the RL agent.

### Weaknesses
-  The entire motivation is built on optimizing for a Memory Budget. However, it fails to compare against the most effective and widely-adopted technique post-training quantization (e.g., INT4). A simple INT4 quantized model would occupy a smaller memory footprint than RAP's pruned FP16 model under the same budget.

- In real-world applications, FP16 would not be deployed.  A convincing demonstration of RAP's value would be to show that it can further reduce memory on top of a quantized model.

- The core idea is training a single and adaptive policy to replace a collection of static configurations. This concept was explored by the Once-for-All. The authors should discuss the connection to Once-for-All.

- While the memory budget focus is practical, the complete absence of a standard fixed-sparsity comparison makes it difficult to isolate and appreciate the core algorithmic improvement.

Once-for-All: Train One Network and Specialize it for Efficient Deployment. ICLR 2020

### Questions
- How does RAP compare to a strong INT4 quantization baseline?  Can you show the performance of "INT4 + RAP" to demonstrate complementary benefits?

- What is the computational cost of GSI?

- I am curious whether, at the same parameter sparsity level (e.g., 30%), does RAP's GSI-based policy yield higher accuracy than a one-shot importance scoring method?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a key limitation of existing LLM pruning approaches, namely their reliance on static importance criteria that fail to reflect the dynamic workload variations encountered in real-world deployment. To address this issue, the authors propose RAP (Runtime-Adaptive Pruning), a reinforcement learning-based framework that adjusts the degree of structured pruning according to runtime conditions.

The authors argue that conventional one-shot pruning evaluates block importance independently under a fixed model structure, thereby neglecting strong inter-layer dependencies within Transformers and leading to cumulative performance degradation. To mitigate this, they introduce Greedy Sequential Importance (GSI), which sequentially removes blocks while re-evaluating perplexity at each step.

RAP further incorporates runtime signals such as batch size, sequence length, system memory, and KV-cache ratio as state inputs to an RL policy that determines block removal decisions. As a result, pruning intensity varies across requests, allowing the model to adapt its structure to meet memory constraints while minimizing degradation in perplexity and downstream task performance. Through block-level structured pruning, the framework removes entire parameter and attention components, indirectly reducing KV cache usage.

Experimental results show that RAP achieves significantly lower perplexity and better commonsense reasoning performance compared to random drop and static pruning baselines under identical memory budgets, with the combination of GSI and RL yielding the most stable performance–compression trade-off.

### Strengths
* Clearly formulates the problem of runtime-aware pruning in realistic LLM inference settings (varying input length, batch size, and memory constraints) and convincingly highlights the limitations of static pruning approaches.
* Introduces Greedy Sequential Importance (GSI) as a principled mechanism that accounts for inter-layer dependencies through sequential importance re-evaluation, effectively mitigating performance degradation associated with one-shot pruning.
* Employs an RL-based controller to dynamically adjust pruning strength on a per-request basis, enabling adaptive responses to heterogeneous workload conditions.
* Adopts structured block-level pruning, enhancing hardware efficiency and practical deployability.
* Demonstrates empirical performance across multiple models and memory budgets, showing superior perplexity retention compared to heuristic baselines.

### Weaknesses
* The construction of the calibration set used for GSI computation in Table 1 is insufficiently specified, making it difficult to rule out the possibility that benchmark test data distributions were indirectly utilized during pruning, raising concerns regarding the fairness of the evaluation protocol.
* Since GSI is repeatedly recomputed based on a proxy corpus and sampled request distributions, the stability and reproducibility of the resulting importance scores remain unclear, potentially introducing variability in pruning outcomes.
* Although GSI is framed as an offline calibration step using a proxy corpus, generating an optimal pruning model for specific user scenarios would in practice require full-model access and repeated GSI execution, for which the associated computational overhead is not sufficiently quantified or analyzed.
* Despite claims regarding edge deployment feasibility, the necessity of full-model access and GSI computation during the pruning phase suggests a non-trivial operational gap between the proposed framework and practical deployment constraints.
* The pruning mechanism is inherently irreversible within an inference session, preventing mid-generation structural readjustments, which limits adaptiveness in multi-turn or reasoning-heavy scenarios where block importance may shift over time.
* The framework does not introduce an explicit KV-cache-specific pruning strategy, instead relying on indirect reduction through block removal, thereby weakening the claim of joint optimization between model parameters and KV cache memory.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
