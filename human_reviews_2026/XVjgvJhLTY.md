# OjaKV: Context-Aware Online Low-Rank KV Cache Compression with Oja’s Rule

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
The expanding long-context capabilities of large language models are constrained by a significant memory bottleneck: the key-value (KV) cache required for autoregressive generation. This bottleneck is substantial; for instance, a Llama-3.1-8B model processing a 32K-token prompt at a batch size of 4 requires approximately 16GB for its KV cache, a size exceeding the model's weights. While KV-cache compression via low-rank projection is a promising direction, existing methods' rely on a static, offline-learned subspace that performs poorly under data distribution shifts. To overcome these limitations, we introduce $\textbf{OjaKV}$, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation. First, OjaKV recognizes that not all tokens are equally important for compression; it preserves the crucial first and most recent tokens in full-rank, maintaining high-fidelity anchors for attention. Second, for the vast majority of intermediate tokens, it applies low-rank compression by incrementally adapting the projection basis using Oja’s algorithm for online principal component analysis. This adaptation involves a comprehensive update during prompt prefilling and lightweight periodic updates during decoding, ensuring the subspace remains aligned with the evolving context. Crucially, our framework is fully compatible with modern attention modules like $\textit{FlashAttention}$. Experiments demonstrate that OjaKV maintains or even improves zero-shot accuracy at high compression ratios. In particular, OjaKV achieves its strongest gains on very long-context benchmarks that require complex reasoning, highlighting the importance of online subspace adaptation in dynamically tracking context shifts. Furthermore, our approach is compatible with token-selection methods, enabling compounded memory savings. These results establish our hybrid framework as a practical, plug-and-play solution for memory-efficient long-context inference without requiring model fine-tuning. Code at https://anonymous.4open.science/r/OjaKV-9D76.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
For long-context inference, static low-rank projection causes a distribution shift, which harms accuracy. The authors propose online adaptation and keep the first (32) and most recent (32) tokens full rank, and compress the rest, adapting them with Ojass  rule. It is compatible with flashattn as they reconstruct KV on the fly, overall, this avoids context drift while delivering memory savings (60%), at the expense of TTFT (30%).

### Strengths
- Plug and play, works with flash-attention
- online adaptation is indeed quite important for reliable long decodes.

### Weaknesses
- baselines are a bit limited, focusing on static low-rank methods. Comparisons with minicache and even potentially MLA would be helpful. For e.g., can this work on MLA for further gains?
- why only TTFT? can we show throughput, scaling of online reconstruction with batching, etc?
- 0.6-0.8x compression with very notable dips in accuracy are a bit concerning (Table 2,3) in long-context settings, where this work is most applicable.

### Questions
- TTFT is great, but if that set-up exists, it should be possible to provide more latency / throughput numbers to strengthen the paper.
- Comments on baselines (from weaknesses) would be appreciated
- The average decrease in Table 2 for all low-rank options are very aggressive, (38→21; 74 → 51) feels like a bit of a difficult position to be in, and opens up questions wrt MLA or even pre-training a model with smaller hidden dimensions (the latter is definitely out of scope, but would appreciate comments on this!)

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
4

### Summary
This paper proposes OjaKV, a framework to reduce the KV cache memory bottleneck in long-context LLMs. It combines a hybrid storage policy (preserving the first and last tokens in full-rank) with an online adaptation mechanism using Oja's algorithm to update the low-rank projection basis for all intermediate tokens.

### Strengths
- The problem of KV cache memory consumption is important.

- The core idea of applying online subspace adaptation (via Oja's rule) to the KV cache to handle context shifts is novel. The author demonstrates the compatibility with FlashAttention and orthogonality to token selection methods.

### Weaknesses
- This Oja update adds significant latency (TTFT increases from 2102ms to 2801ms at 32K tokens)  for a negligible accuracy benefit, making its practical value questionable.

- Evaluated models & benchmarks are relatively old & small. 

- It seems most performance gain comes from the hybrid attention strategy.

### Questions
What is the TTFT of the StaticPCA-H baseline? This is essential to isolate the true latency cost of only the Oja update mechanism.

The paper's motivation is to handle complex, dynamic contexts. Why not evaluate OjaKV on newer models with dedicated reasoning capabilities (e.g., Qwen3-Thinking) or on more recent, challenging reasoning benchmarks like AIME, which would truly test the method's ability to adapt?

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
3

### Summary
The paper proposes OjaKV, a plug-and-play KV-cache compression method for long-context LLMs that combines a hybrid storage policy, keeping the first and most recent tokens full-rank as anchors, with online, context-aware low-rank projection updated via Oja’s rule. It selects salient tokens to adapt the subspace during prefill and performs lightweight periodic updates during decoding, while remaining fully compatible with FlashAttention by reconstructing full-rank tensors on the fly. The authors prove equivalence to reduced-space computation, analyze memory/latency trade-offs, and show OjaKV composes multiplicatively with token-selection methods.

### Strengths
- Introduces online, context-adaptive low-rank KV-cache compression using Oja’s rule, a novel application of online PCA to LLM inference.
- Removes a major limitation of prior static low-rank methods by enabling real-time subspace adaptation to evolving prompts.
- Proposes a hybrid storage policy (full-rank anchors + adaptive low-rank middle) that creatively combines insights from token selection and subspace learning.

### Weaknesses
- Insufficient exploration of trade-offs between adaptation frequency, learning rates, and performance
- Lack of end-to-end latency and throughput benchmarks under real inference workloads
- While the paper highlights OjaKV’s memory savings and Time-To-First-Token (TTFT) latency, it does not report actual decoding throughput (e.g., tokens per second) or total end-to-end inference time. Since OjaKV reconstructs full-rank tensors on the fly and performs periodic Oja updates, these operations could introduce nontrivial overhead during decoding. Without decoding speed or wall-clock runtime comparisons against baselines (Full KV, StaticPCA, FlashAttention-only), it is unclear whether OjaKV provides net speedups or even maintains comparable throughput under long-context workloads. This omission weakens the practical evaluation, especially because the method’s main motivation is efficient inference.
- Although OjaKV consistently outperforms static low-rank baselines, the absolute gains reported in most benchmarks are relatively modest—often within 1–2% accuracy or a few points on average. At the same time, these improvements come with increased algorithmic complexity (Oja updates, token selection, reconstruction). For instance, on LongBench and lm-eval-harness, OjaKV’s advantage over StaticPCA-H is minimal, suggesting that the benefit of online adaptation may not always justify the added overhead.

### Questions
- How stable are the online Oja updates across long decoding sequences (e.g., 64K–128K tokens)?

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
The paper focuses on the problem of the KV cache occupying a large amount of memory during long-context LLM reasoning. To address the issue, it proposes the OjaKV framework, which has two main points:

Keep full-rank KV for the first n_start tokens and the last n_recent tokens, while applying low-rank projection to the remaining tokens, thereby providing high-fidelity "anchors" for attention.  

For the K/V of intermediate tokens, perform online PCA using Oja’s rule. During the prefill phase, select salient tokens based on attention and update them in batches. During the decode phase, perform lightweight updates periodically using a buffer of recently generated tokens.

In experiments, the authors evaluated on: RULER， LongBench, and lm-eval-harness using models such as Llama-2-7B, Llama-3.1-8B, and LongChat-7B, comparing FullKV, Eigen-N, StaticPCA, and StaticPCA-H. The results show that under 0.8× and 0.6× KV memory budgets, OjaKV consistently outperforms static low-rank baselines on long context tasks and barely loses performance on short context tasks.

### Strengths
The problem is highly targeted with sufficient motivation. 

The method design is clean and feasible. 

The experimental design is relatively comprehensive. 

Writing and structural logic are smooth.

### Weaknesses
1 The compression rate range is relatively conservative, and there is a significant gap compared with the 'Extreme compression' approach.  
The core experiments focus on 0.8× and 0.6× (20–40% savings), while recent works like KVQuant, SVDq, CSR, and dictionary learning have already explored the range of 4×, 8×, and even 10×+.  
The current results are more like 'online subspace can help static PCA be more stable under moderate compression,' but it is not enough to demonstrate a dominant advantage in extreme compression scenarios.  

2 Currently, the main comparison is among Eigen-N / StaticPCA / StaticPCA-H, focusing on the 'low-rank projection system'; However, related work has already mentioned mature solutions such as KVQuant, Palu, MatryoshkaKV, and SVDq. The main experiments do not compare with these stronger baselines, which may weaken the persuasiveness; Even if OjaKV is orthogonal to quantization/eviction, it is recommended to provide: At least one combined baseline with a representative quantization/sequence compression method; Or a more systematic explanation: under equivalent compression ratios, a comparison of accuracy/latency/implementation complexity between pure quantization/eviction and OjaKV.

3 Insufficient analysis of the overhead and stability of online Oja updates. The paper reports an overhead of TTFT from 2102ms → 2801ms, indicating that it is not "zero cost"; however, it lacks a formula-level estimation of computational complexity and a sensitivity analysis of latency and performance with respect to different hyperparameters such as rank, update frequency T, and the number of salient tokens.

4 The design of the hybrid policy/token selection strategy is somewhat empirical.  
It only provides the configuration “keep first n_start + last n_recent full-rank + attention-based salient sampling” and a small amount of results, but there is no systematic ablation on different window sizes, whether to keep only the beginning/only the end, ways of calculating importance, etc.  
This makes it hard to judge how much of OjaKV's gains come from the useful hybrid heuristic rather than the online subspace update itself.

5 The theoretical part is rather light, leaning more towards a rational explanation rather than new theory. Regarding the orthogonality + bounded error argument with token eviction, it essentially reflects the contraction property of a linear operator and is at the level of a sanity check. There is no deeper analysis of online PCA regarding convergence, forgetting, and noise robustness under distribution shift scenarios, nor is there a comparison with other online PCA methods.

6 The application perspective is somewhat abstract. Although there are long-context benchmarks, evaluations of end-to-end latency, GPU memory budget, and throughput in real system scenarios such as multi-turn chat, code editing, or RAG agents are still quite limited. In particular, the behavior of online updates under multiple requests or mixed distributed traffic is not clearly explained.

### Questions
1 Is the online update of OjaKV independent per sequence, or does it share subspaces across multiple requests? If shared, how can we prevent cross-task contamination?

2 Have you tried comparing StaticPCA at more extreme compression rates? Are the benefits of online updates more apparent in high compression regimes?

3 Is there an automated approach for selecting the hybrid full-rank interval (n_start, n_recent), the update interval T, and the learning rates η_pre/η_dec? Are the current empirical values robust across models/tasks?

4 Has there been any preliminary testing combining with representative quantization methods or SVDq? At the same compression ratio, how does OjaKV plus quantization compare to pure quantization?

5 Do multi-head, multi-layer updates share subspaces or update independently per head/layer? In the current implementation, is it per-head/per-layer or a shared subspace? How does this affect performance and computational cost?

### Soundness
2

### Presentation
2

### Contribution
2
