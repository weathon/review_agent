# Bring Future Vision: Dynamic Computation Allocation Guided by Lightweight Feature Forecaster

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
The deployment of large language models (LLMs) in practical scenarios is hindered by their massive computational overhead. While token-wise computation allocation emerges as a promising solution, existing methods suffer from irreversible information loss and suboptimal token selection due to the $\textit{greedy routing}$ paradigm. This paper introduces a novel paradigm, $\textit{informed routing}$, which proactively addresses these limitations. Our key insight is to employ Lightweight Feature Forecasters (LFF) — simple, low-cost networks that learn to approximate the transformations of individual model components — before making any routing decisions. This allows the router to assess a token's recoverability (ease of approximation) rather than just its immediate importance. Extensive experiments demonstrate that our approach achieves state-of-the-art performance across various sparsity levels on language modeling and reasoning tasks. Notably, even without final LoRA fine-tuning, our method matches or surpasses strong baselines that require full fine-tuning, all while reducing training time by over 50\%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Informed Routing, a new dynamic computation allocation framework for large language models that aims to improve both efficiency and performance in token-wise pruning. Existing DCA approaches rely on greedy routing, where routers make binary “execute-or-skip” decisions that can cause irreversible information loss.

The proposed method introduces a Lightweight Feature Forecaster, a small, efficient module that approximates the output of transformer submodules. Instead of skipping tokens outright, the router decides whether to process them normally or approximate them via LFF, based on their recoverability.

This shift from “skip” to “approximate” allows smoother performance degradation under sparsity. Experiments on LLaMA-3B and 8B models show that Informed Routing achieves better accuracy–perplexity trade-offs than baselines like SkipGPT and static pruning methods, especially at 25–40% sparsity.

### Strengths
1. The proposed Informed Routing paradigm and LFF module offer a new way to dynamically adjust computation without severe performance loss. Results on reasoning and language modeling tasks show clear gains over both static pruning and greedy dynamic baselines, proving its effectiveness.
2. The experiments are comprehensive and systematic, covering both static and dynamic pruning comparisons across sparsity levels, model sizes (3B and 8B), and diverse benchmarks. The ablation studies further clarify how different LFF sizes, sparsity ratios, and module balances influence performance.

### Weaknesses
1. Although the framework improves theoretical performance under sparsity, its real-world efficiency gain is questionable. In large-batch inference, skipped or approximated tokens must wait for dense batches, making parallel execution inefficient. Thus, the approach may not outperform dense models in actual deployment scenarios.
2. Pruning inevitably causes non-negligible accuracy loss. The authors’ own results (e.g., Table 4 and 5) show that while moderate sparsity (25%) retains quality, higher sparsity ratios rapidly degrade performance. This limits the usefulness of the framework in real-world LLM compression, where the efficiency–performance trade-off remains hard to beat compared to dense baselines or specialized architectures.

### Questions
1. Since pruning inevitably leads to information loss, how can this dynamic computation model meaningfully outperform dense models in overall efficiency once practical acceleration and batch synchronization are considered?
2. In parallel inference scenarios, the dynamic skipping introduces heterogeneous batch latencies. How is this addressed to ensure consistent throughput? Are there strategies to mitigate the “waiting problem” of sparse batches?

### Soundness
3

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
2

### Summary
This paper proposes the Informed Routing paradigm, where a lightweight network called LFF (Lightweight Feature Forecaster) predicts the output of a computation block before the router decides whether to execute or approximate it. This replaces “skipping” with “approximating”,  mitigating information loss and improving routing decisions. Experiments on LLaMA-family models  for language modeling  and reasoning tasks show that the proposed method achieves better performance and training efficiency (reporting over 50% training time savings) than greedy routing baselines.

### Strengths
1. Novel and intuitive motivation: The idea of using predictability as a routing criterion directly addresses the weaknesses of greedy routing and is well motivated with quantitative metrics (cosine similarity and L1 error).


2. Lightweight design: The LFF module is compact, adding only about 0.02% extra parameters (≈0.82M), achieving a good trade-off between model capacity and efficiency.

3. Comprehensive experiments on small and mid-sized models: The method is evaluated on 3B and 8B models, with multiple sparsity ratios (25/40/70%) and ablations (balanced/unbalanced routing). Results demonstrate consistent improvements across settings.

### Weaknesses
1. Code release and reproducibility: The paper states “we will open-source our code upon receipt,” but no repository or scripts are currently provided. The authors should release code and scripts for LFF initialization, router training, LoRA fine-tuning, and inference benchmarks (including environment setup and seeds).

2. Lack of evaluation on larger models: Experiments are limited to 3B and 8B models. It is unclear whether LFF remains effective for larger models such as 13B, 30B, or 70B, where attention/FFN balance and KV-cache behavior differ.

3. Missing wall-clock speed and latency benchmarks: The paper reports training time savings (“>50% faster”) but does not provide inference-time throughput or latency results, which are critical to validate claims of deployment efficiency.

### Questions
See weaknesses.

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
This paper identifies and addresses key limitations in existing dynamic computation allocation (DCA) methods for large language models (LLMs). The authors argue that the prevalent "greedy routing" paradigm, which forces a binary "execute-or-skip" decision for each token, leads to irreversible information loss and suboptimal token selection based on short-sighted criteria. To overcome this, they propose a new paradigm called "informed routing," which replaces the skip option with an efficient approximation. The core of this approach is the Lightweight Feature Forecaster (LFF), a small network trained to mimic the transformation of its corresponding model component. This allows the router to base its decision on a token's "recoverability" (i.e., how well its transformation can be approximated) rather than its immediate importance. The proposed method involves a three-stage training pipeline: LFF initialization, router training, and optional LoRA fine-tuning. Extensive experiments demonstrate that this approach achieves state-of-the-art performance, significantly improves training stability, and can match or exceed fully fine-tuned baselines even without the final fine-tuning step, reducing training time by over 50%.

### Strengths
- The paper introduces a novel and significant paradigm shift from "greedy routing" to "informed routing". The concept of replacing an information-destroying "skip" action with an information-preserving "approximate" action is a creative and powerful idea. Using the LFF to assess a token's recoverability is an original criterion for routing decisions that directly addresses the core weaknesses of prior work.
- The empirical evaluation is comprehensive and of high quality. The authors validate their method on multiple model scales (3B and 8B Llama models) and across a diverse set of reasoning and language modeling benchmarks. The inclusion of strong static and dynamic baselines, along with insightful ablation studies like the balanced computation experiment, provides robust evidence for the method's effectiveness.
- The paper is exceptionally clear and well-written. The motivation is well-established by clearly diagnosing the "All-or-Nothing Dilemma" and "Short-Sighted Token Selection" problems of existing methods. The proposed architecture and training pipeline are explained logically and are well-supported by clear diagrams (e.g., Figure 2), making the work easy to understand and reproduce.
- The work carries significant practical implications. The finding that the method, after only router training, can outperform a fully fine-tuned baseline (SkipGPT-LORA) while saving over 50% of the training time is a major advantage. This highlights the efficiency and stability gains from the LFF pre-fitting stage, making advanced model compression more accessible.

### Weaknesses
- The exploration of the Lightweight Feature Forecaster (LFF) architecture is somewhat limited. The paper defaults to a simple two-layer linear network, which, while efficient, may not capture the full potential of the informed routing paradigm. The ablation study shows performance saturating with a small intermediate dimension, suggesting that this simple linear model hits a performance ceiling quickly. An investigation into slightly more complex, non-linear LFFs could provide valuable insights into the trade-off between forecaster capacity and overall model performance, especially at higher sparsity levels where the current LFF struggles.
- The method's contribution to Key-Value (KV) cache reduction feels underdeveloped. The authors adopt a straightforward masking strategy that predictably degrades performance and is not deeply integrated with the core LFF mechanism. Since the LFF approximates a token's output rather than eliminating it, the token's key and value are still required for subsequent layers, meaning the approach does not inherently reduce the KV cache. A more novel approach that leverages the LFF's predictive capabilities to perhaps generate approximate KV pairs could have been a more impactful contribution.
- While the paper demonstrates strong performance at moderate sparsity (25-40%), it also concedes that the approach does not extend the ultimate upper bound of computation reduction, with performance dropping off significantly at 70% sparsity. This positions the method as an improvement upon existing techniques rather than a complete breakthrough. A discussion on potential hybrid strategies or adaptive LFFs that become more powerful as sparsity increases could have strengthened the paper's forward-looking perspective.

### Questions
- The LFF initialization uses a fixed, relatively small dataset of 2,000 samples. How sensitive is the LFF's approximation quality, and consequently the final model's performance, to the size and composition of this initialization dataset? Could a larger or more strategically sampled dataset allow the LFF to learn more robust transformations?
- The paper provides an interesting analysis showing that your method prunes attention modules more heavily, attributing this to their "linear simplicity". This is a fascinating claim. Could you provide more qualitative or quantitative analysis to support this? For instance, what types of tokens or linguistic phenomena are most frequently routed through the LFF in attention layers versus FFN layers?
- The use of LFFs introduces approximation errors for a subset of tokens at each layer. How do these small errors propagate and accumulate through the network? Is there a risk that the accumulated noise from multiple LFFs could negatively impact the representations of tokens that are consistently processed by the main computational units in later layers?
- The paper's evaluation focuses primarily on model quality metrics (accuracy and perplexity) rather than practical acceleration. Could the authors provide a quantitative analysis of the wall-clock inference speedup (e.g., in tokens/second or latency) for both the LFF method and the SkipGPT baseline at various sparsity levels? It would be particularly insightful to see how these realized speedups compare to the theoretical speedup suggested by the sparsity ratio, and what overheads (e.g., router logic, non-contiguous memory access) limit performance. Similarly, for the training phase, what is the acceleration per training step compared to a dense model, to isolate the computational savings from the faster convergence you already reported?

### Soundness
3

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
2

### Summary
This paper proposes Informed Routing for dynamic computation allocation in LLMs. Instead of the traditional greedy execute-or-skip decision, the authors introduce a new option: execute-or-approximate via Lightweight Feature Forecasters (LFFs), which are small bottleneck networks trained to mimic the output of each compute unit (self-attention or FFN) before routing decisions are made. The method is trained in three stages: (1) prefit LFFs to each unit, (2) train routers (with Gumbel-Softmax) to choose between the original unit and its LFF under a global sparsity target, and (3) optional LoRA fine-tuning. Experiments on LLaMA-3B/8B report lower perplexity and higher reasoning accuracy than strong static and dynamic baselines (e.g., SkipGPT), especially at 25% sparsity, with >50% training-time savings for the router+LoRA pipeline. Analyses indicate attention transformations are often linearly simple and thus more predictable by LFFs, while very high sparsity (e.g., 70%) exposes forecasting limits.

### Strengths
1. Clear motivation: The authors motivates the invention of LFFs well by pointing out that current DCA method adopt a all-or-nothing approach which has room for improvements. This gives the router a recoverability signal in terms of how well a token's transformation can be forecast.

2. Clean methodology: The three stage pipeline is well explained with clear equations as well as training specifics (timings, optimizers etc.)

### Weaknesses
1. Limited forecasting capacity at high sparsity: The experiment results shows that at 70% sparsity the quality of LFF performance degrades notably. The author already acknowledges this but I wonder what are some potential resolution to this that you have in mind?

2. Evaluation breadth: It seems that this approach would be particularly promising for long context tasks, but the authors didn't directly benchmarked on this part. I am wondering if there is any related result on long context evaluation?

### Questions
1. The author mentioned that KV-cache reduction would hurt quality, I wonder if you could provide an understanding of KV memory saving vs. improvement/loss in PPL/benchmark accuracies for me to better understand this trade-off?

(See weaknesses for the rest)

### Soundness
3

### Presentation
3

### Contribution
3
