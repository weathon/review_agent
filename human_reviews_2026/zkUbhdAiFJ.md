# Attention Is All You Need for KV Cache in Diffusion LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
This work studies how to adaptively recompute key–value (KV) caches for diffusion large language models (DLMs) to maximize prediction accuracy while minimizing decoding latency. Prior methods' decoders recompute QKV for all tokens at every denoising step and layer, despite KV states changing little across most steps, especially in shallow layers, leading to substantial redundancy. We make three observations: (1) distant MASK tokens primarily act as a length-bias and can be cached block-wise beyond the active prediction window; (2) KV dynamics increase with depth, suggesting that selective refresh starting from deeper layers is sufficient; and (3) the most-attended token exhibits the smallest KV drift, providing a conservative lower bound on cache change for other tokens. Building on these, we propose Elastic-Cache, a training-free, architecture-agnostic strategy that jointly decides ${when}$ to refresh (via an attention-aware drift test on the most-attended token) and ${where}$ to refresh (via a depth-aware schedule that recomputes from a chosen layer onward while reusing shallow-layer caches and off-window MASK caches). Unlike fixed-period schemes, Elastic-Cache performs adaptive, layer-aware cache updates for diffusion LLMs, reducing redundant computation and accelerating decoding with negligible loss in generation quality. Experiments on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V across mathematical reasoning and code generation tasks demonstrate consistent speedups: $8.7\times$ on GSM8K (256 tokens), $45.1\times$ on longer sequences, and $4.8\times$ on HumanEval, while consistently maintaining higher accuracy than the baseline. Our method achieves significantly higher throughput ($6.8\times$ on GSM8K) than existing confidence-based approaches while preserving generation quality, enabling practical deployment of diffusion LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the computational overhead in Diffusion Large Language Model inference, which stems from recomputing the full QKV cache for all tokens at all layers during each denoising step. The authors propose Elastic-Cache that adaptively decides when and where to recompute the KV cache.

### Strengths
1. The method is training-free and architecture-agnostic, making it a drop-in component that can be easily adopted.

2. Comprehensive experiments and ablations showing the effectness of Elastic-Cache

### Weaknesses
1. It seems that the authors used the wrong template.

2. The primary assumption is that the most-attended token "typically exhibits the smallest drift" and thus serves as a "conservative lower bound" . This is a strong claim that is central to the method's success, but it is presented as an observation without comprehensive supporting data.

3. Would the authors clarify the exact mechanics of this window? Does it advance one token at a time, or does it process a full block of $\beta$ tokens and then move to the next block (which would make it very similar to Fast-dLLM, just with a different caching rule)?

4. At each step and each layer, the algorithm must find the most-attended token. What is the computational overhead of this process? a brief analysis of the trigger's cost relative to the cost of a single QKV projection would better for analysing.

### Questions
see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Elastic-Cache, a training-free KV cache refresh strategy for diffusion LLMs. Based on KV cache drift statistic, Elastic-Cache adaptively controls when to recompute the KV cache and selectively updates only deeper layers. By removing redundant cache computation, it achieves significant speedups (up to 45×) with minimal accuracy loss across diverse tasks.

### Strengths
- The idea is novel and well-motivated. The core idea is built upon a key observation: significant redundancy exists in DLLM's KV cache computation across denoising steps and layers, a majority of the KV cache can be reused. Similar phenomena have been discussed in KV cache acceleration for LLMs (Duoattention, Razorattention, CateKV, etc.)
- Strong performance across different models: Elastic-Cache consistently outperforms baselines in performance and speedup, with up to 45.1× speedup over LLaDA-1.5. It also maintains similar accuracy, demonstrating efficient computation reduction without sacrificing quality.

Effective Adaptive Mechanism: The attention-aware drift test (using most-attended tokens) and layer-aware updates reduce cache refresh frequency to as low as 20% of baseline (Fig. 3b), validating its lightweight and adaptive design.

### Weaknesses
- The author claims that Elastic-Cache architecture-agnostic. I suggest adding experiments on other DLLMs like Dream to prove it.

- Limited Multimodal Speedup: On LLaDA-V, speedup is modest (29.7 t/s), suggesting less cache redundancy in multimodal tasks or architecture-specific limitation of the method. And why not report the performance and tps of original LLaDA-V in Table 3?

### Questions
Please refer to the weaknesses

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
3

### Summary
The paper diagnoses redundancy in diffusion‑style LLM decoding: most methods recompute QKV for all tokens and all layers at every denoising step, even though KV states often change little—especially in shallow layers and for faraway MASK tokens. It proposes Elastic‑Cache, a training‑free, architecture‑agnostic policy that (i) block‑caches distant MASK tokens, (ii) triggers cache refresh when needed via an attention‑aware drift test on the most‑attended token, and (iii) refreshes where needed by recomputing only from a learned boundary layer ℓ^⋆ upward (layer‑aware schedule). Figure 2 contrasts the pipeline against blockwise caching; Algorithm 1 gives the full procedure. Reported results show up to 45.1× speedups on long sequences and 6.8× higher throughput on GSM8K while maintaining or improving accuracy versus baselines (e.g., Fast‑dLLM).

### Strengths
++ The paper clearly explains why AR KV caching fails in diffusion LLMs (bidirectional attention ⇒ evolving KVs) and motivates KV drift as the right signal. Figure 1 visualizes depth‑dependent drift and shows the most‑attended token’s attention changes track KV change, justifying the drift test.

++ Three components—block‑caching far MASK tokens, sliding‑window decoding for active MASKs, and a layer‑aware refresh boundary—are well matched to diffusion masking dynamics and are easy to integrate.

++ The method does not alter training or model internals; it only changes cache policy, which supports deployment claims and broad applicability across diffusion LLM variants. The motivation and contributions list emphasize this.

### Weaknesses
-- The most‑attended‑token drift is a clever heuristic, but the paper lacks sensitivity/theory: when does it fail (e.g., diffuse attention or multi‑focus regimes)? A risk analysis or bounds linking attention drift to KV error would strengthen the claim. Algorithm 1 hints at cosine‑similarity thresholds but ablations are not detailed.

-- While Fast‑dLLM and related works are discussed, it’s unclear if comparisons include recent adaptive policies (e.g., learned thresholds) under identical hardware/token budgets and for very long masks; more head‑to‑head with dKV‑Cache/DeepCache over a shared protocol would improve fairness.

-- The method introduces β (window size), γ (drift threshold), and ℓ^⋆ (boundary). The paper should show cross‑task robustness and guidance for setting them vs. accuracy/latency trade‑offs. (§3.2–3.3 describe β/γ, but systematic sweeps are not reported.)

-- The narrative focuses on compute and latency; a detailed profile of memory bandwidth, cache footprint (esp. with sliding windows and partial refresh), and impact on KV tensor residency would help practitioners plan deployments.

### Questions
1. How sensitive are results to the cosine‑similarity threshold γ and to choosing the single most‑attended token vs. a small set? Can you report AUROC of the drift test to predict accuracy deltas on held‑out runs?

2. How is ℓ^⋆ determined at runtime? Static per‑model, or adaptive per input? Please include ablations showing latency/quality as ℓ^⋆ varies.

3. In sequences where distant MASKs become influential (e.g., closing brackets, long‑range constraints), how often does block‑caching degrade predictions? Any fallback to expand the window β on‑the‑fly?

4. Please provide per‑sequence memory footprints and DRAM traffic for Elastic‑Cache vs. baselines at 256/512/1024 tokens so practitioners can size deployments.

5. Have you tested on masked‑diffusion LLMs with different denoising schedules or bidirectional attention variants? Any caveats for multimodal diffusion decoders?

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
4

### Summary
This paper introduces Elastic-Cache, a training-free, architecture-agnostic strategy to accelerate decoding in diffusion large language models (DLMs) by adaptively managing key-value (KV) cache recomputation. The method leverages three key observations: (1) distant MASK tokens act primarily as length priors and can be block-cached; (2) KV drift increases with layer depth, justifying depth-selective updates; and (3) the most-attended token exhibits minimal KV drift, serving as a reliable proxy for global cache stability. Elastic-Cache combines an attention-aware drift test (triggering refresh when cosine similarity of attention weights drops below threshold γ) with a layer-aware update policy (refreshing only from a boundary layer ℓ* onward). Experiments on LLaDA variants show substantial speedups—up to 45.1× on GSM8K-512, 8.7× on GSM8K-256, and 4.8× on HumanEval—while maintaining or even improving accuracy over baselines like Fast-dLLM.

### Strengths
* Strong empirical results: Consistent throughput gains across models (LLaDA-Instruct, LLaDA-1.5, LLaDA-V) and tasks (math, code, multimodal), with negligible accuracy loss.
* Principled design: The use of attention-weight drift on the most-attended token as a low-overhead trigger is well-motivated and novel for DLMs.
* Practical impact: Training-free, compatible with existing confidence-based decoding, and enables real-world deployment of DLMs.
* Comprehensive ablation: Thorough analysis of γ, sliding window size, and generation length validates design choices.

### Weaknesses
* Limited theoretical grounding: While empirically sound, the link between attention drift and KV state change lacks formal justification.
* Hyperparameter sensitivity: Performance depends on γ and β; optimal values vary across tasks (e.g., γ=0.9 for text, γ=0.7 for multimodal).
* Baseline scope: Comparisons focus on Fast-dLLM and dKV-Cache; broader context with other diffusion acceleration methods (e.g., consistency models) is sparse.
* Hardware assumption: All experiments on single A100; scalability to multi-GPU or devices is unverified.

### Questions
* How does Elastic-Cache perform under extreme KV drift (e.g., highly ambiguous prompts or adversarial inputs)?
* Could the attention-drift threshold γ be learned or adapted per token/layer instead of being global?
* What is the overhead of computing attention drift? Does it scale linearly with context length?
* Is the method applicable to block diffusion?

### Soundness
3

### Presentation
3

### Contribution
3
