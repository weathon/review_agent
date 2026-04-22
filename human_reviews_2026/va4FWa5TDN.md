# Cylon: Asynchronous Linear Attention

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Sequence models face stark tradeoffs between recall quality and memory efficiency. Recall -- the ability to use information over long sequences -- is critical for sequence modeling tasks ranging from information extraction to reasoning. 
Prior work has shown that in theory, \textit{linear} attention models with sufficient recurrent state sizes can expand the Pareto frontier of the recall-memory tradeoff space beyond alternative architectures such as softmax attention and state space models.
However, it is difficult to scale the linear attention state size due to hardware bottlenecks. I/O aware algorithms store the linear attention states in thread registers, however state sizes beyond $\approx 3$ megabytes exhaust register memory and trigger expensive register spills. 
In this work, we introduce CYLON, a hardware-aware strategy for partitioning linear attention's recurrent state across the registers of multiple GPU processors and asynchronously combining the partitions. When applying CYLON to popular architectures, such as Hedgehog and Mamba-2, we unlock 3$\times$ higher throughput compared to prior linear attention algorithms for these architectures on both Hopper and Blackwell GPUs. Finally, CYLON makes large states available to model designers by unlocking sizes (e.g., $\geq$ 131 MB) that are not achievable by the existing linear attention kernels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the fundamental tradeoff between recall quality and memory efficiency in sequence models. The authors identify that while linear attention models can theoretically outperform alternatives by using large recurrent states, they are practically limited by a hardware bottleneck: recurrent states larger than a few megabytes ($\approx3$ MB) exhaust fast GPU register memory, causing expensive "register spills" to the L1 cache.

To solve this, the authors propose CYLON, a hardware-aware linear attention architecture44. CYLON's core mechanism is to partition a large recurrent state across the registers of multiple streaming multiprocessors (SMs). Each SM processes its partition independently, and the partial results are then combined efficiently using modern asynchronous GPU operations, particularly the Tensor Memory Accelerator (TMA) for I/O and asynchronous global reductions.

### Strengths
1. Clear Problem Identification and Analysis: The paper does an excellent job of identifying a specific, critical, and practical bottleneck in modern sequence models. The analysis in Section 2.3, supported by Table 1, clearly profiles existing linear attention kernels and pinpoints register spills as the key limiting factor for scaling recurrent state size in efficient chunkwise algorithms.

2. Strong Empirical Performance: The performance improvements are a major strength. The paper demonstrates up to 3x higher throughput (iterations per second) on both H100 and Blackwell GPUs (Figure 3). This is a significant speedup that makes these models more practical.

3. Expanding the Pareto Frontier: Perhaps the most important contribution is that CYLON doesn't just speed up existing models; it enables new models. By unlocking state sizes ($\ge 131$ MB) that prior kernels could not support, the paper delivers on its promise of expanding the recall-efficiency trade-off space. This allows model designers to flexibly scale state size for better recall, which was previously not feasible

### Weaknesses
1. Normalization Relaxation (Eq. 10): A critical step for efficiency is the "normalization relaxation" proposed in Equation 1025. This avoids a global memory roundtrip 26by introducing a single "learned scalar $e$" 27instead of performing the full, correct normalization shown in Equation 728. While the results in Table 3 suggest quality is preserved29, this is a significant approximation. The paper lacks an ablation study to quantify the quality (and performance) impact of this specific relaxation. It is unclear if the quality parity is due to the partitioning being truly equivalent (as Theorems 1-2 suggest) or if this approximation happens to work well for the tested tasks.

2. Complexity of Partitioned Feature Maps: The CYLON architecture requires $m$ distinct feature maps ($\phi_k$) 30303030, which are implemented as learned MLPs 3131, to ensure diversity32. This seems to add implementation and potential training complexity compared to a standard linear attention model. The paper does not fully explore the parameter overhead or the impact on training dynamics (e.g., convergence, stability) resulting from this design choice.

3. Unsubstantiated "Path to Softmax": Section 3.2 also proposes an "alternate path to softmax attention" by approximating the $exp(x)$ function with polynomial expansions distributed across SMs 33. This is an interesting concept but is presented without any empirical validation. It feels disconnected from the paper's main contribution (linear attention) and leaves the reader wondering if this is a practical proposal or a purely theoretical aside.

### Questions
1. Could the authors provide an ablation study on the normalization relaxation in Equation 10? What is the quality and throughput impact of implementing the "correct" normalization (Equation 7) —which would require a global synchronization—compared to the proposed relaxation? This would help isolate the gains from partitioning from those from this approximation.

2. Regarding the proposed "alternate path to softmax attention": Have the authors benchmarked this approach? How does its practical throughput and, more importantly, its numerical stability (especially with low-precision formats like FP8) compare to standard, fused-kernel implementations of softmax attention like FlashAttention?

3. How sensitive is the model's quality and performance to the number of partitions, $m$? Table 3 shows results for $m \in \{1, 2, 4\}$, but the throughput benchmarks in Figure 3 are not broken down by $m$. How does throughput scale as $m$ increases, and is there a point where the overhead of the asynchronous reduction outweighs the benefit of smaller, register-resident partitions?

4. The paper focuses on inference throughput, specifically prefill. How does CYLON affect training performance? Does the partitioned architecture and the addition of $m$ distinct feature maps impact the training-time compute graph, memory footprint, or end-to-end training speed?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CYLON,  a hardware-aware linear attention architecture, which partitions the recurrent state across the registers of multiple GPU processors and asynchronously combines the partitions. When applying CYLON linear attention to popular architectures, such as Hedgehog and Mamba-2,  it unlocks 3× higher throughput compared to prior linear attention algorithms for these architectures on both Hopper and Blackwell GPUs. Additionally, the proposed  CYLON makes large states available to model designers by unlocking
sizes (e.g., ≥ 131 MB) that are not by the existing linear attention kernels.

### Strengths
1. The paper introduces an asynchronous linear attention mechanism that allows decoupling of query and key-value updates. This improves efficiency and makes the model more flexible in streaming or autoregressive scenarios—an interesting departure from traditional synchronous designs.

2. The formulation appears mathematically grounded, with clear derivations showing how the asynchronous update maintains stability and approximate equivalence to standard attention under certain conditions.

3. The paper includes solid experimental validation on multiple benchmarks, showing that CYLON achieves better or comparable performance with less computation and memory, demonstrating both effectiveness and scalability.

### Weaknesses
1. Most experiments are on standard benchmarks. The benefits of asynchrony (e.g., in real-time or streaming settings) are not thoroughly demonstrated in practical large-scale or latency-critical applications
'
2. The intuition behind why asynchronous updates yield better representations is somewhat under-explained. The paper could clarify whether the asynchronous mechanism introduces a temporal inductive bias, improves stability, or reduces redundancy — and under what conditions this matters. It would be better to include visualizations (e.g., attention maps over time, update trajectories) or toy examples could help convey the underlying intuition.

3. While comparisons with standard linear attention methods are included, it may lack evaluation against the latest state-space models (like Mamba or Hyena), which could provide a stronger empirical contrast to justify the novelty and advantage.

### Questions
1. The paper claims that asynchronous updates between queries and key-values improve efficiency and representation quality. Could the authors elaborate on the mechanistic reason behind this improvement? For instance, does asynchrony introduce beneficial temporal diversity or implicit regularization, or does it merely approximate a form of delayed gradient update? Some empirical or theoretical clarification would help justify why asynchrony leads to better results beyond computational scheduling advantages.

2. Since state-space models (e.g., Mamba, Hyena) also enable efficient long-sequence modeling with sub-quadratic complexity, how does CYLON compare conceptually and practically to these approaches?

### Soundness
3

### Presentation
4

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
Sequence models, crucial for tasks like information extraction and reasoning, face a trade-off between recall quality (ability to use long-sequence information) and memory efficiency. Linear attention models, in theory, can expand this trade-off frontier beyond other architectures like softmax attention and state-space models, but are limited by hardware bottlenecks, particularly regarding the size of their recurrent states. Current linear attention implementations struggle to scale beyond a few megabytes due to register memory exhaustion and expensive register spills, leading to significant slowdowns.

CYLON addresses this by partitioning the recurrent state across multiple GPU streaming multiprocessors (SMs) and asynchronously combining these partitions. This design leverages modern GPU features such as Tensor Memory Accelerators (TMA) for asynchronous I/O and reductions, and Tensor Cores for asynchronous multiplication. By ensuring each partition remains register-resident within an SM, CYLON maximizes locality and enables significantly larger effective state sizes. The paper demonstrates that this approach achieves up to 3x higher throughput compared to prior linear attention algorithms on both Hopper and Blackwell GPUs, and unlocks state sizes (e.g., ≥ 131 MB) that were previously inaccessible. CYLON-partitioned models maintain or exceed the quality of default linear attention models on language modeling and NLU tasks, effectively expanding the Pareto frontier for recall-efficiency.

### Strengths
1. The paper tackles a well-identified and significant challenge in linear attention: the inability to scale recurrent state sizes due to hardware memory limitations. This is a crucial problem for improving the recall quality of sequence models.
2. CYLON is meticulously designed with a deep understanding of modern GPU architectures, leveraging specific features like TMAs and Tensor Cores for asynchronous operations. This hardware-software co-design is key to its performance gains.
3. The reported 3x higher throughput and the ability to unlock much larger state sizes (≥ 131 MB) represent a substantial leap forward in the practical application of linear attention, particularly on modern GPUs.
4. Despite the architectural changes for efficiency, CYLON-partitioned models match or exceed the quality of non-partitioned linear attention models on various language modeling and NLU tasks, indicating that efficiency gains do not come at the cost of performance.
5. The paper includes theoretical proofs (Theorems 1, 2, 3) demonstrating that partitioning the state does not reduce the expressivity of linear attention and that the decomposition can be computationally efficient.

### Weaknesses
1. The asynchronous nature and distributed state management across SMs, while efficient, likely introduce significant complexity in implementation and debugging compared to traditional linear attention kernels.
2. While the paper discusses using learned MLPs for feature maps to ensure diversity, the specifics of how these distinct feature maps are learned and their impact on model training dynamics could be explored further.
3. The reliance on features like TMA and Warp-group matrix multiply means that CYLON's direct applicability might be limited to NVIDIA's Hopper and Blackwell architectures. While these are prevalent, performance on other hardware (e.g., AMD, custom AI accelerators) is not discussed.
4. The argument for "increasingly asynchronous machines" is based on current high-end NVIDIA GPUs. While this trend is likely to continue, future architectural shifts could potentially alter the optimal strategies for handling recurrent states.
5. While the paper evaluates language modeling and NLU tasks, a broader set of applications where long-context recall is paramount could further solidify CYLON's benefits.
6. While the paper mentions "long-latency roundtrips and inter-SM synchronization overhead," a more detailed breakdown of the specific overheads that CYLON successfully mitigates, perhaps with microbenchmark results for each component, could further strengthen the argument.

### Questions
see Weaknesses

### Soundness
3

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
4

### Summary
The paper tackles the practical limit that prevents linear-attention models from scaling recurrent state size due to GPU register pressure and spill-induced stalls. The proposed CYLON architecture partitions the linear-attention recurrent state across multiple streaming multiprocessors (SMs), keeps each partition register-resident, and aggregates partial results via asynchronous global reductions (TMA) while overlapping I/O and compute with wave specialization and async MMAs. Algorithmically, CYLON replaces a single feature map with per-partition maps and introduces a lightweight normalization relaxation with a learnable scalar. On Hopper and Blackwell GPUs, CYLON reports up to 3 times throughput improvements over widely used linear-attention kernels and enables large effective state sizes that prior kernels could not support. Language-model pretraining at 1.3B scale on 10B tokens shows that CYLON preserves perplexity and recall-oriented downstream performance relative to non-partitioned linear attention at matched state sizes. Theoretical results show that partitioning the feature map preserves expressivity and that simple polynomial expansions can be distributed across SMs.

### Strengths
- **(S1) Clear motivation grounded in profiling evidence.** The introduction and Section 2.3 convincingly identify register spills and long scoreboard stalls as the limiting factor for large linear‑attention states, supported by Nsight Compute profiles in Table 1. This isolates the true bottleneck and motivates the partition‑plus‑async design.

- **(S2) Hardware‑aware design with principled asynchrony.** CYLON combines async MMA, async I/O, and async global reductions to keep state in registers and avoid expensive HBM round trips. For example, Table 2 shows TMA reductions are 8–11% faster throughput and 12–15% lower latency than atomics in isolation, matching the end‑to‑end speedups. Table 3 reports comparable perplexity to non‑partitioned baselines at the same state size for both Hedgehog and Mamba‑2; recall‑heavy tasks improve as state size increases, with CYLON keeping pace or winning.

- **(S3) Reasonable reproducibility and clarity for a systems paper.** The manuscript is well organized and written in clear and professional English with comprehensive figures and tables. Meanwhile, the methodology is well structured and described with enough specificity (kernel pseudocode at the right abstraction, state‑size accounting, training stacks, hyper-parameters), and Appendices A–C document settings, extra sweeps, and proofs. Figures and writing are generally clear.

### Weaknesses
- **(W1) Limited theoretical depth relative to systems novelty.** The “equivalence” results largely formalize block decomposition of a unified feature map (Theorems 1–2) and a standard feature‑lifting observation for polynomials (Theorem 3). They do not analyze finite‑precision effects, async reduction order, or the learned scalar in Eq. (10) for stability/bias—precisely where implementation deviates from the idealized algebra. Strengthening this link would improve the scientific tightness.

- **(W2) Evaluation scope for model quality is narrow.** The main pre-training is 1.3B parameters on 10B tokens, with quality measured on standard NLU and three recall tasks. Long‑context tasks with explicit reasoning chains or multi‑hop retrieval are absent; comparisons to other efficient non‑LA long‑context mixers (e.g., SSM variants, RetNet) at comparable compute are not included. Hence, the claim that CYLON “expands the Pareto frontier” is well‑supported on throughput/state, but only suggestive on end‑task quality and reasoning.

- **(W3) Missing end‑to‑end inference metrics and profiling breakdowns.** Results emphasize iterations/sec for prefill at fixed (N, batch, d). However, decode‑token latency, HBM GB/s, occupancy, tensor‑core utilization, and power/energy are not reported, despite the contribution centering on memory hierarchy and asynchrony. Practitioners need these to make deployment decisions.

- **(W4) Minor issues of writing and limitations.** Firstly, the authors should double-check the typos in the main text.  Meanwhile, there are several limitations. (i) The effective partition count mm is bounded by available SMs and on‑chip memory, so speedups may vary substantially across SKUs (SXM vs PCIe) and future architectures with different async primitives. (ii) Overlapping TMA, async MMAs, and reductions increases kernel complexity and sensitivity to ISA/driver changes; without upstreaming, sustaining performance across generations may be non‑trivial.

### Questions
- **(Q1)** How are gradients and optimizer states synchronized with async reductions? Any measurable **gradient variance** or **load imbalance** across $\{\phi_k\}$ affecting convergence speed? Please report training stability signals (e.g., grad norms, step‑to‑step variance).

- **(Q2)** What are the end‑to‑end results (quality + stability) in FP8 for CYLON, given Figure 2’s approximation argument? What accumulation precisions are used along the reduction path, and are there guardrails (loss scaling, Kahan‑style compensation)?

- **(Q3)** Minor concerns about the fair experimental setup for decode latency. Were baseline kernels re‑tuned for large states (e.g., launch parameters, chunk shapes) before concluding the 3× speedup? Can you add decode latency (ms/token) at several state sizes to clarify real‑time implications?

### Soundness
3

### Presentation
3

### Contribution
3
