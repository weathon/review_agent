# ResLR: Residual-Low-Rank Surrogates for Stable and Fast Context Adaptive Computing in Large Language Models

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Large Language Models (LLMs) achieve state-of-the-art results on diverse tasks, yet inference remains expensive because every token traverses the full Transformer stack. Recent context adaptive computing methods mitigate this cost by token-wise layer skipping, but their per-layer routing is volatile, leading to accuracy oscillations and an extended fine-tuning process. We trace this instability to two issues: (i) direct skips violate the model’s functional hierarchy, and (ii) per-layer routing fails to exploit the similarity of activations between neighboring layers. We therefore propose a unified acceleration framework addressing both problems. First, we introduce the Residual-Low-Rank (ResLR) surrogate, a lightweight bypass that distills the residual transformation between consecutive layers into a low-rank operator within a compact subspace, thus synthesizing the effect of the skipped layers and preserving hierarchy. Second, we devise Block-Wise Multi-Path Routing, which clusters neighboring layers into blocks and issues a single routing decision per block, explicitly leveraging activation similarity to stabilize computation and reduce gating overhead. The method integrates into standard LoRA fine-tuning without extra stages. Across question answering, mathematical reasoning, and commonsense inference benchmarks, it reduces FLOPs by 48%–52\% and yields $\sim$1.9$\times$ wall-time speed-ups while outperforming static and dynamic baselines. With feature probing suggests a $\sim$90% functional preservation, variance analysis shows 42.3% lower score standard deviation and 53.7\% more stable routing than layer-skipping approaches, establishing ResLR and block-wise routing as a robust approach for practical, low-cost LLM inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses routing instability in token-wise layer-skipping methods for efficient LLM inference. The authors propose ResLR surrogates, which is a low-rank approximation of multi-layer residuals trained via self-distillation—combined with block-wise routing that makes unified decisions for groups of layers. Experiments on LLaMA2-7B/13B show 48-52% FLOPs reduction with improved stability (53.7% lower routing variance) and performance gains over baselines.

### Strengths
1. The problem is well motivated. The paper identifies the routing instability issue with existing layer-skipping methods caused by violating functional hierarchy and per-layer routing volatility. The empirical evidence (Figure 2) effectively demonstrates this problem.

2. The experimental results are sound, as table one clearly shows empirical gains.

### Weaknesses
1. The self-distillation formulation Is Circular. In particular, equation defines the main loss term, where E_{i+1} is the "teacher" output. But the paper never clarifies if E_{i+1} is computed with or without the surrogate f participating in the forward pass. If the surrogate participates: The target E_{i+1} depends on f, creating a moving target problem. This invalidates the entire theoretical analysis, as lemma 1 assumes approximating a fixed residual \delta E, but here \delta E depends on the thing being trained. The bias-variance decomposition (Eq. 9) breaks down because \phi* is no longer well-defined. If the surrogate is frozen: This should be stated explicitly, and you need ablations showing training stability with/without this constraint.The self-distillation literature addresses these circularity issues with EMA teachers or momentum updates. Your formulation ignores this entirely, making the theoretical contribution questionable.

2. "Preserving Functional Hierarchy" Is not validated. The core claim is that ResLR preserves the model's functional hierarchy better than direct skipping (lines 79-91), but this is never directly tested. The evidence provided is indirect: higher mutual information just shows correlated decisions, not functional correctness; lower variance shows stability, not hierarchy preservation; better task performance has many possible explanations.

### Questions
The paper claims all baselines use "self-distillation joint training" but D-LLM's original paper doesn't. Did you retrain everything under your protocol? If so, are the comparisons fair to methods designed for different training regimes?

Also: the paper only compares layer-skipping methods. Speculative decoding achieves 2-3× speedups with zero accuracy loss. How does ResLR compare? Claiming "state-of-the-art" without comparing these is misleading.

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
2

### Summary
This paper addresses LLM inference costs through dynamic layer skipping. The authors identify routing instability in existing methods and propose ResLR with two components: (1) low-rank surrogates to approximate skipped layers, motivated by SVD analysis showing 90% information retention in 50 components, and (2) block-wise routing that groups adjacent layers for unified decisions. Results show 48-52% FLOPs reduction with 1.9× speedup and improved stability vs D-LLM.

### Strengths
1. Well-motivated problem. Figure 2 quantifies routing instability with mutual information and variance metrics, providing clear motivation.

2. Reasonable design choices. SVD analysis supports the low-rank assumption. Block-wise routing addresses both efficiency and stability in a straightforward way.

3. Consistent improvements across 9 benchmarks with both FLOPs reduction and wall-time speedup.

### Weaknesses
1. There are some reproducibility concerns, as several essential implementation details are not fully specified:
- Specific rank r values and their selection criteria
- Gating network architecture and training procedure
- Loss balancing between distillation and task objectives
- ResLR insertion strategy (which layers, all or selective?)

With these details clarified, it would be easier to reproduce the method.

2. The paper lacks explanation for why inter-layer residuals have low-rank structure. More critically, the claim that ResLR "preserves functional hierarchy" is not rigorously justified. How does low-rank projection preserve hierarchy differently than direct skipping? Without probing experiments or formal analysis, this core claim remains unsubstantiated.

3. Testing only 7B and 13B is insufficient for "excellent scalability" claims. No evaluation on:
- Models larger than 13B (30B, 70B)
- Long-context scenarios (8k+ tokens)
- Different architectures beyond LLaMA

The generalization remains questionable.

### Questions
1. What specific rank r do you use? How is it selected?The paper mentions r << d but never specifies actual values. Is it the same across all layers? This is fundamental for reproducibility.
2. How exactly does ResLR preserve functional hierarchy? This is your core claim, but the mechanism is unclear. Can you provide probing experiments or formal analysis showing that low-rank projection preserves hierarchy better than direct skipping?
3. Does the low-rank structure hold across different models and tasks?
4. Why is block-wise routing necessary if ResLR already addresses the hierarchy problem?

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
This paper addresses a critical bottleneck in deploying Large Language Models (LLMs): high inference cost due to every token passing through all layers of the Transformer stack. While recent work has explored layer skipping to reduce this cost, such methods often suffer from unstable routing decisions and degrade the model’s internal functional structure, leading to volatile accuracy and inefficient fine-tuning.

To solve this, the authors propose ResLR, a unified acceleration framework that improves both efficiency and stability in dynamic inference. It combines two key innovations:
1。 ResLR surrogates that replaces skipped Transformer layers with a learned low-rank operator, preserving the model’s functional hierarchy.
2. Block-Wise Multi-Path Routing: Instead of deciding skip decisions per layer, the model groups layers into blocks and makes a single routing decision per block, reducing overhead and stabilizing the computation.

This framework is plug-and-play with standard LoRA fine-tuning and demonstrates state-of-the-art trade-offs on multiple reasoning and language benchmarks, reducing FLOPs by over 50%, increasing inference speed by 1.9×, and improving output stability over previous dynamic approaches.

### Strengths
1. The paper identifies two causes of instability in prior layer-skipping methods—(1) disrupting the model’s learned depth hierarchy, and (2) noisy, per-layer routing without inter-layer coordination.
2. The proposed ResLR surrogate is proposed to learn to approximate the combined residual transformation of skipped layers using a low-rank structure, trained via self-distillation, and achieve good empirical results.
3. The paper provides a clear bias-variance decomposition for the surrogate’s approximation error and justifies its rank selection.
4. The method can be integrated seamlessly into existing LoRA fine-tuning pipelines.

### Weaknesses
1. Unlike standard LoRA or other low-rank fine-tuning methods, the proposed ResLR surrogate operates as an external bypass module rather than a low-rank correction to existing Transformer weights. As a result, the trained ResLR components cannot be merged back into the original model weights, making deployment more complex and limiting compatibility with existing model-serving optimizations that rely on weight merging (e.g., static inference graph export).
2. Because the dynamic router determines which layers to execute per token, the method requires custom inference logic and is not directly compatible with existing static inference engines, possibly complicating real-world deployment.
3. While tested on LLaMA2, the method’s performance and compatibility with other LLM families (e.g., Qwen3, Llama3.x) is not assessed.

### Questions
While block-wise routing amortizes decision cost, inference still requires executing token-level gating logic. In high-throughput or multi-token batch settings, how significant is this routing overhead, especially as block size or model size increases?

### Soundness
3

### Presentation
3

### Contribution
3
