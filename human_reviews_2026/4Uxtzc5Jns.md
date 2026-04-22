# G-KV:  Decoding-Time  KV Cache Eviction  with Global Attention

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Recent reasoning large language models (LLMs) excel in complex tasks but encounter significant computational and memory challenges due to long sequence lengths. KV cache compression has emerged as an effective approach to greatly enhance the efficiency of reasoning. However, existing methods often focus on prompt compression or token eviction with local attention score, overlooking the long-term importance of tokens. We propose G-KV, a KV cache eviction method that employs a global scoring mechanism, combining local and historical attention scores to more accurately assess token importance. Additionally, we introduce post-training techniques, including reinforcement learning and distillation, to optimize models for compressed KV cache settings.  The code of this paper is available on:https://anonymous.4open.science/r/G-KV-B3C0 .

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces G-KV, a decoding-time KV Cache eviction method designed to address the computational and memory bottlenecks that large language models face during long-sequence reasoning tasks. The authors argue that existing methods, which often rely on local attention scores to determine which tokens to evict, fail to capture the long-term importance of tokens. To solve this, G-KV introduces a "global score" that combines the current local attention score with historical scores, allowing for a more accurate assessment of a token's long-term value. Furthermore, the paper explores post-training techniques, including reinforcement learning and distillation, to better adapt the model to a compressed KV Cache environment. Experiments on the AMC and AIME mathematical reasoning benchmarks demonstrate that G-KV achieves significant performance improvements over existing methods.

### Strengths
1. The core motivation—that local attention is insufficient for capturing long-term token importance—is both intuitive and critical. The experiment in Section 4 (Figure 1), which convincingly demonstrates that the set of attended tokens shifts across different time windows, provides strong empirical support for the proposed “global score.”

2. I appreciate the paper’s writing style, especially the clarity of its figures and the *Observations* presented throughout, which make the motivation and overall argument easy to follow.

### Weaknesses
1. Limited Benchmark Coverage: The evaluation is primarily focused on mathematical reasoning tasks. While this effectively demonstrates the model's capabilities in long chain-of-thought scenarios, it may not fully prove the generalizability of the G-KV method to other types of long-text tasks. For instance, many state-of-the-art methods like SnapKV are also tested on benchmarks such as **LongBench** (for comprehensive long-text understanding) and **Needle-in-a-Haystack** (for long-range information retrieval). Including experiments on these more general long-context benchmarks would strengthen the paper's conclusions.

2. Insufficient Analysis of Method Overhead: The G-KV method introduces a "global score," which requires storing historical scores. Although the paper mentions in Appendix G that this overhead is negligible, it does not provide detailed empirical data to support this claim. Does calculating the global score introduce additional computational latency at each compression step? It would be beneficial for the authors to provide more specific experimental data, such as: **(1) What is the exact increase in memory (VRAM) usage for G-KV compared to methods that only use a local score? (2) On the same hardware and with the same batch size, what is the time cost of the G-KV algorithm itself (i.e., the scoring and sorting process)?** This would help readers more fully assess the method's efficiency.

### Questions
1. Regarding the benchmarks: Have you considered evaluating G-KV on more general long-context benchmarks like LongBench or Needle-in-a-Haystack? This would help validate your method's effectiveness on tasks beyond mathematical reasoning.

2. Regarding efficiency overhead: Could you provide more detailed data on the computational and memory overhead introduced by the global score calculation? Specifically, what is the added latency and extra memory consumption per compression step compared to a local-score-only method?

3. Regarding the hyperparameter $\alpha$: Figure 4 shows that the method's performance is quite sensitive to the decay factor $\alpha$. For new models or tasks, do you have any recommendations or heuristics for setting this hyperparameter, other than extensive experimental search?

4. Regarding token distribution: The finding in Figure 5 is very interesting—it shows G-KV retains tokens more uniformly across the entire sequence, whereas other methods are biased toward the end. Could you elaborate on why this uniform distribution is beneficial? Does it suggest that G-KV is better at preserving key information from the initial prompt, thus preventing context loss during long-range generation?

### Soundness
2

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
5

### Summary
The authors propose a KV cache selection strategy that comprehensively considering both local and historical attention contexts to enhances accuracy after cache compression. Furthermore, they introduce an additional post-training optimization step designed to adapt the model to the compressed KV Cache, which subsequently yields further performance improvements.

### Strengths
- The paper is well-structured and easy to follow.
- The improvement achieved through the post-training optimization is noteworthy.

### Weaknesses
- The experimental results are unstable and require validation on a broader range of models, particularly larger-scale models (e.g., 32B and 70B parameters).
    * Specifically, while the performance gain for the DeepSeek-R1-Distill-Qwen-7B model in Table 1 appears acceptable, the improvements for the DeepSeek-R1-Distill-Llama-8B model in Table 2 are very inconsistent.
- The experimental results indicate that significant performance improvements are only achieved under very low cache budgets(like 256), which are likely impractical or unusable in real-world scenarios. The improvement is marginal in large budget scenarios.
- Lack of Novelty.
- The post-training optimization appears largely disconnected from the proposed global score mechanism, suggesting an ad hoc addition. 
    - Fine-tuning a model to adapt to the sparsity inherent in cache compression is a general optimization technique and is not specifically tied to the authors' KV cache compression algorithm.
- The proposed method will likely impact the Time-to-First-Token(TTFT) performance. The authors need to include an experimental analysis of the TTFT overhead.

### Questions
- Performance and concern: In Table 4, the results show that G-KV (R-KV w/ global score ) achieves some performance gain over plain R-KV. Since the consideration of a "global score" is expected to introduce additional computational overhead, could the authors explain why G-KV still maintains an advantage?
- Data Inconsistency between Tables: The data presented in Table 1 and Table 3 does not align. Specifically, the "Untrained" baseline data in Table 3 does not match the "G-KV" data in Table 1. An explanation for this discrepancy is required.

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
G-KV proposes a decoding-time KV-eviction rule that forms a per-token global score by combining a normalized local window score with an attenuated historical score (via a hard max), plus an optional RL fine-tuning objective; evaluation is largely math-reasoning with 7B/8B models.

### Strengths
The proposed global scoring mechanism is intuitive, which improves over R-KV at 256 tokens.

### Weaknesses
**Q1.**  The related-work section names several 2025 SOTA methods (R-KV, CAKE, KVzip) but they are not used as baselines in the main tables; RocketKV and ShadowKV from ICML’25, Raas ACL’25 are ignored for baseline comparison.   

**Q2.** Novelty is incremental relative to temporal/union-aware and redundancy-aware methods.
 The global score (Eq. 3) is max(α·F_{t−1}, normalized S_t). a form of temporal accumulation via attenuation and a hard max. CAKE explicitly models temporal shifts and layer preferences; R-KV adds redundancy scores to avoid keeping near-duplicates; ShadowKV/KVzip reconstruct/repurpose KV context. G-KV’s “historical-local max” feels like a special case of these broader families and needs a stronger differentiation (e.g., theory or empirical wins against them).

**Q3.**  Authors state the RL objective “simplifies GRPO,” but the description aligns with REINFORCE w/ baseline; also, arguing the online setup “eliminates the need for clipping” is not supported—PPO/GRPO-style stability typically relies on clipping/trust-region constraints. Please justify with analysis or ablations. In general, I am not sure if, in practice, these RL work and deliver promising results.


**Q4.** Heuristic design issues:

The hard max in Eq. (3) may overreact to outliers; compare to (i) EMA / weighted average, (ii) max-pool w/ temperature, (iii) top-k smoothing. No ablation is shown.


Sensitivity to α, w (window), and s (interval) is missing; these hyper-params directly govern stability and recall. (Authors define them but don’t study sensitivity.)


**Q5.** Benchmarks are math-centric (AMC-23, AIME-24) with distilled 7B/8B models; The following ones are ignored GSM8K, MATH-500, CSQA, LiveCodeBench, and a long-context retrieval suite (LongBench/BabiLong).


**Q6.** Authors state decoding time is ~40% of Full-KV and ~90% memory reduction at 16k (appendices), but no system-level comparisons vs optimized kernels (e.g., FlashAttention-3 baselines in SeerAttention-R, or ShadowKV’s throughput).

**Q7.** Missing ablations/analyses 

A) Hard-max design must be validated against smoother alternatives.
 Add an ablation that replaces the hard max with: (a) exponential moving average (EMA) of local and historical scores; (b) a convex combination with a learned or tuned mixing weight; and (c) a temperatured max/LogSumExp. For each, report pass@1, average KV-retention, and variance of retention across steps to assess robustness to attention spikes.

B) Aslo, the author should run structured sweeps: window size (w from {64, 128, 256}), compression interval (s from {8, 16, 32}), attenuation (α from {0.8, 0.9, 0.95, 0.99}), and budgets (b from {128, 256, 512, 1024, 2048}). Plot accuracy vs. retention and accuracy vs. decode-time to identify stable operating regions.

C) Head-wise analysis (are some heads persistently favored/evicted?) to contrast with HeadKV


**Typos:**

“PREILIMINARY” to  “PRELIMINARY”.

 (Eq. 2): The summation bounds (k=0 to w) imply w+1 elements, but the division is by w. This should be clarified (e.g., k=0 to w-1).

### Questions
I already indicated them in the weakness section.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes G-KV, a decoding-time KV cache eviction method that introduces a global scoring mechanism combining historical and local attention scores to better preserve long-term token importance. It further enhances performance via post-training techniques—reinforcement learning with sparse attention masks and knowledge distillation. The method is evaluated on challenging mathematical reasoning benchmarks (AMC-23, AIME-24) using strong reasoning LLMs, showing significant gains over prior SOTA, especially under tight KV cache budgets (e.g., +19.15% pass@1 at 256 tokens).

The work is well-motivated, empirically solid, and addresses a critical bottleneck in long-context reasoning LLMs. The global scoring idea is simple yet effective, and the integration with existing methods (e.g., R-KV) demonstrates strong generalizability. The training-aware extensions (RL-Sparse, Distill) are thoughtfully designed to close the train-inference gap.
However, while the empirical results are compelling, the technical novelty is incremental, and several methodological and evaluation concerns limit the strength of the contribution for a top-tier conferences.

### Strengths
**Strong Empirical Results:**
  - Clear and consistent improvements over strong baselines across multiple models (Qwen-7B, Llama-8B), datasets (AMC, AIME), and KV budgets.
  - Gains are especially pronounced under low budgets (256–512 tokens), which are practically relevant for deployment.

**Insightful Motivation via Empirical Observation:**
  - Figure 1 provides compelling evidence that token importance is non-stationary across decoding windows—justifying the need for global scoring.
  - The analysis of token retention distribution (Figure 5) convincingly shows that G-KV preserves more diverse context than local methods.

**Practical and Modular Design:**
  - The global score is a drop-in replacement for local scores in existing eviction frameworks (H2O, SnapKV, R-KV).
  - Minimal computational overhead (confirmed by throughput results in Table 4).

**Comprehensive Evaluation:**
  - Includes efficiency metrics (throughput, memory, decoding time), ablation on α and λ, cross-model validation, and qualitative case studies (Appendix J).
  - Post-training methods (RL-Sparse, Distill) are well-motivated and show meaningful gains.

**Reproducibility:**
  - Code, distilled data, and environment configs are provided.

### Weaknesses
**Limited Technical Novelty:**
   - The global score (Eq. 3) is essentially an exponentially weighted moving average (EWMA) of normalized attention scores—a well-known technique in online learning and signal processing. While effective, it lacks deep algorithmic innovation.
  - The core idea resembles “heavy-hitter” tracking with decay, similar in spirit to H2O but with historical memory.

**Evaluation Scope is Narrow:**
  - Experiments are restricted to mathematical reasoning on two datasets. No evaluation on general QA, coding, or open-ended generation.
  - All models are distilled reasoning models from DeepSeek-R1. Performance on standard LLMs (e.g., Llama-3, Qwen2) or non-reasoning tasks is unverified.
  - No comparison to non-eviction compression methods (e.g., quantization, low-rank) that may offer better trade-offs.

**Hyperparameter Sensitivity:**
  - Performance heavily depends on α (decay) and λ (redundancy weight). While tuned, the paper doesn’t provide robustness analysis or automatic tuning strategies.
 - The optimal α ≈ 0.8–1.0 suggests historical scores dominate—raising questions about the marginal utility of local scores.

**Ambiguity in Training Protocol:**
  - RL-Sparse uses a sparse attention mask during training, but it’s unclear how this interacts with RoPE (rotary embeddings), which assumes full positional context. This could introduce positional bias.
  - Distillation uses teacher outputs from full KV, but student trains with sparse attention—a distributional mismatch not fully addressed.

**Claimed SOTA May Be Overstated:**
  - The 19.15% improvement is vs. R-KV under 256 tokens—but R-KV itself may not be the strongest baseline (e.g., CAKE, LightThinker are mentioned but not compared in main results).
  - No comparison to recent methods like StreamingLLM or Infini-Attention that handle long contexts differently.

### Questions
- How does G-KV perform on non-reasoning tasks (e.g., narrative continuation, summarization) or with standard LLMs (e.g., Llama-3-8B) without reasoning distillation?

- Besides math related task, what about those extremely long context reasoning task in coding domain like SWE bench? Besides, could you provide a computing and memory cost for different context length such as 4K, 8K, 16K, 32K, 64K, etc?

- Since G-KV evicts early tokens, how does the model handle positional information for retained early tokens under RoPE?

### Soundness
3

### Presentation
3

### Contribution
2
