# Trion: FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of LLMs

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6, 4

## Abstract
Low-rank optimization has emerged as a promising direction in training large language models (LLMs) to improve running time and reduce the memory usage of adaptive optimizers by constraining learning to a lower-dimensional space. Prior work typically projects gradients of linear layers using approaches based on Singular Value Decomposition (SVD) or QR-decomposition. Applying these techniques individually to each layer in large models is computationally expensive and incurs additional memory costs due to storing the projection matrices. In this work, we propose a computationally efficient and conceptually simple, two-step procedure to approximate SVD/QR-based gradient projections into lower-dimensional spaces by using a predefined orthogonal matrix of the Discrete Cosine Transform (DCT). We dynamically select columns from the DCT matrix based on their alignment with the gradient of each layer. The effective projection matrices are obtained via a simple \texttt{matmul} with the DCT matrix in $O(n^3)$ time, followed by a lightweight sorting step to identify the most relevant basis vectors. For large layers, DCT can be computed via \texttt{Makhoul}'s $N$-point algorithm based on Fast Fourier Transform (FFT) in $O(n^2 \log(n))$ time, yielding speed-ups for low-end GPUs. Due to the predefined nature of the orthogonal bases, they are computed once at the start of training. Our numerical experiments on both pre-training and fine-tuning tasks demonstrate the effectiveness of our dual strategy in approximating optimal low-rank projections, obtaining an approach with rank-independent running time that matches the performance of costly SVD/QR-based methods while achieving faster runtime and reduced memory usage by up to $25\%$ across different model sizes. Our code is available at \href{https://github.com/IST-DASLab/Trion}{\texttt{https://github.com/IST-DASLab/Trion}}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
In this paper, the authors have attempted to address the computational and memory overhead in optimizers such as AdamW, Muon, Dion, and GaLore in LLM training. Prior low rank method relies on using SVD or QR decomposition to project gradients into a lower dimensional subspace. However, they are computationally expensive. Therefore, in this paper, the authors attempt to replace these with an alternative low-rank projection approach, which is cheaper to compute.

### Strengths
The strengths of this paper are summarized as follows:

1. SVD is time-consuming in LLM training, and it is widely used in various methods, like GaLore, FIRA, and FRUGAL. Replacing SVD with the Fast Fourier Transform based algorithm can reduce the time complexity.

2. Empirically, it has shown improvements on multiple optimizers and model sizes, like LLaMA 350M, 800M, and 1.3B. Also, on all of these model sizes, Trion has shown better performance and better running time compared with Dion. 

3. It has a strong theoretical guarantee for justifying the column selection approach may give the most significant column.

### Weaknesses
The weaknesses of this paper are summarized as follows:

1. The largest model size in the experiment is 1.3B. It would be better if the authors may consider running an experiment on larger models since modern transformer architectures are getting much larger than this size. Also, only the C4 dataset and LLaMA are tested and there is no fine-tuning or downstream benchmarks.

2. Galore has numerous follow-up works, such as Galore 2 [1], Golore [2], and Sara [3]. Could the authors give a comparison with these works?

[1] DiJia Su, Andrew Gu, Jane Xu, Yuandong Tian, and Jiawei Zhao. "Galore 2: Large-scale llm pre-training by gradient low-rank projection." arXiv preprint arXiv:2504.20437 (2025).

[2] Yutong He, Pengrui Li, Yipeng Hu, Chuyan Chen, and Kun Yuan. "Subspace optimization for large language models with convergence guarantees." ICML'25.

[3] Haochen Zhang, Junze Yin, Guanchu Wang, Zirui Liu, Tianyi Zhang, Anshumali Shrivastava, Lin Yang, and Vladimir Braverman. "Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining". NeurIPS'25.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the high compute and memory cost of SVD/QR decompositions in low-rank optimizers  The authors propose replacing these expensive, per layer projections with a "dynamic column selection" from a single, predefined orthogonal basis (the Discrete Cosine Transform, or DCT, matrix) computed once at the start . The method efficiently selects the top $r$ most aligned DCT basis vectors for each layer's gradient, replacing SVD/QR with a fast matrix multiplication (or FFT) and a sorting step. This saves memory, as each layer only stores $r$ indices instead of a full projection matrix. The authors integrate this technique into two new optimizers: Trion (improving Dion) and DCT AdamW (improving AdamW variants).

### Strengths
1. The method's runtime is rank independent, whereas SVD/QR based methods like Dion get slower as the rank $r$ increases. This is a significant practical advantage for using larger and more expressive ranks.

2. The core idea is simple yet elegant. The "dynamic column selection" is not just heuristic; the authors prove it is the optimal strategy for minimizing reconstruction error, given a fixed orthogonal basis $Q$. The motivation for using the DCT as that basis is also well justified.

3. Strong Empirical Results: 

i. Trion vs. Dion (Table 1): Trion consistently achieves better validation perplexity, lower memory usage (~7-10%), and faster runtimes (up to 18% faster) than its direct baseline, Dion, across all tested model sizes and ranks.

ii. DCT-AdamW vs. LDAdamW: DCT-AdamW achieves better validation perplexity, drastically lower memory, and is significantly faster (~25%).

### Weaknesses
1. Theoretical speedup not realized. The paper heavily motivates using DCT by citing the fast $O(n^2 \log n)$ FFT-based algorithm. However, the authors admit that for the model sizes tested (up to $d=2048$), this speedup was not significant, and a standard $O(n^3)$ matmul was used15. The primary speedup comes from replacing SVD/QR, not from the FFT.

2. While DCT-AdamW beats its low-rank competitor LDAdamW, it is still significantly outperformed by full rank AdamW (Val. PPL 13.69 vs. 11.73). This shows the method is a better low rank compromise, but still a compromise.

3. As the authors note, experiments are limited to 1.3B models. The method's true scalability and the benefit of the FFT based algorithm would only be evident on much larger models.

### Questions
Please see Weaknesses section

### Soundness
3

### Presentation
3

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
The authors introduce a Discrete Cosine Transform (DCT)-based dynamic column selection technique that approximates optimal low-rank projections by selecting columns of a fixed orthogonal DCT matrix aligned with each layer’s gradients to replace expensive SVD/QR-based low-rank projections used in adaptive optimizers for large language models. This method reduces computation and memory overhead by avoiding per-layer decompositions and storing only column indices. They apply this idea in two new optimizers: Trion, which improves Dion, and DCT-AdamW, a low-rank variant of AdamW.

### Strengths
1. The work targets a bottleneck in large-model training: the cost of SVD/QR-based low-rank projections used in adaptive optimizers.
2. The proposed DCT-based projection method is straightforward and easily integrable into existing optimizers.
3. The approach achieves good efficiency gains.
4. The paper provides mathematical rationale for why DCT approximates the gradient eigenbasis and the effectiveness of norm-based selection. 
5. The presentation is easy to follow and the paper is well-written.

### Weaknesses
1. All experiments stop at 1.3B-parameter models; there is no validation on pretraining larger LLMs (7B+), which undermines claims of scalability.
2. The paper lacks certain ablations of critical design choices (e.g., norm type, rank sensitivity, DCT variant).
3. Distributed and FSDP discussions are not empirically backed with wall-clock or communication cost benchmarks.
4. While the paper targets efficiency, comparison with fast efficient baselines that are not using SVD/QR decomposition like APOLLO [1] and SubTrack++ [2] would help to strengthen the paper. 
---
[1] Zhu et al., 2025. APOLLO: SGD-like Memory, AdamW-level Performance.
[2] Rajabi et al., 2025. SubTrack++: Gradient Subspace Tracking for Scalable LLM Training

### Questions
Please refer to the weaknesses.

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
3

### Summary
The paper proposes replacing per step SVD or QR low rank projections in adaptive optimizers with a fixed orthogonal basis using DCT and dynamic column selection. At each step the method scores basis columns by gradient to basis correlation and selects the top r to form projections, which avoids repeated factorizations and heavy state storage. Two instances, Trion and DCT AdamW, show lower memory use, faster training, and comparable or better perplexity on mid size LLM pretraining. The theory motivates norm based selection and provides simple error bounds. The implementation also explains how to integrate with distributed training to reduce communication.

### Strengths
The paper uses a precomputed DCT with on the fly selection to replace repeated SVD or QR, and only column indices are stored. Column norm ranking aligns with minimizing reconstruction error, and there is clear intuition for why DCT approximates dominant directions.

The method has consistent memory reduction and wall clock speedups while matching or improving perplexity across several model sizes.

The notes on DDP or FSDP usage and local reconstruction make adoption straightforward.

### Weaknesses
Evidence for scaling to very large hidden sizes remains limited. The paper primarily reports results on mid size models where the benefit of fast transforms over plain matrix multiplication is muted. Without end to end wall clock measurements at dimensions around 8k to 16k, and without a breakdown of time in the similarity computation, basis selection, and reconstruction kernels, it is hard to assess whether the claimed speedups persist when layers become wide and deep. A thorough profiling study across model width, batch size, and rank would make the efficiency claims more convincing.

It is unclear whether all systems level optimizations such as ZeRO style redundancy removal, identical communication and precision settings, and error feedback or quantization choices are enabled symmetrically for both the proposed method and the baselines. Differences in these controls can easily dominate the observed speed or memory gains. The paper should report results under strictly matched configurations and, if desired, separately include best tuned variants for each method.

The paper mixes square and rectangular gradient matrices without clearly specifying when to apply left versus right projections, and how this choice is made across different layer types such as attention projections and output layers. The exact dimensional assumptions of the DCT matrices and the consistency of symbols drift across sections, which complicates reproduction and theoretical interpretation. A precise, layer wise rule set and a short ablation on these choices would improve clarity.

The set of comparative baselines is too narrow to establish superiority. Strong structured projections such as Hadamard or FWHT, CountSketch style mappings, and recent online or streaming low rank methods are not evaluated side by side under matched rank and refresh frequency. Because many of these alternatives also offer O(n log n) or near linear time with tiny memory footprints, omitting them leaves open whether DCT based selection is uniquely effective. Head to head comparisons on identical tasks and budgets are needed to justify the design choice.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Low-rank optimization can speed LLM training and cut optimizer memory, but per-layer SVD/QR gradient projections are costly and require storing projection matrices. We propose a simple DCT-based alternative: multiply each layer’s gradient by a fixed orthogonal DCT basis, then rank-select the most aligned columns to form the projection. The DCT is efficiently computed (via FFT-based routines), precomputed once, and reused—so projections need only a matmul plus lightweight sorting. Across pre-training and fine-tuning, this yields rank-independent runtime, matches SVD/QR accuracy, and achieves faster training with lower memory use.

### Strengths
- Provided a two-stage DCT-based projection method to avoide the computational cost of SVD
- Developed the DCT variant Trion and DCT-AdamW
- Demenstrated the contractivenss of the proposed compressor
- Conduct extensive experiments

### Weaknesses
1. **Effectiveness in tracking gradients.** Constructing SVD-free, contractive projection matrices is straightforward (e.g., random projections). The real challenge is to remain SVD-free and contractive *while* faithfully capturing the gradient’s low-rank structure. The proposed two-stage method fixes a DCT basis $D_C$ for the entire training and, at each iteration, selects only a few of its columns. In effect, it tracks gradients using subsets of a preset basis—an approach not obviously aligned with evolving, layer-specific low-rank subspaces. The paper does not explain why this selection should match the true gradient subspaces or under what conditions it would; clearer insight or evidence (e.g., principal-angle analyses or SVD/QR approximation errors over training) is needed. 

2. **Necessity of compressing optimizer states.** ZeRO-style sharding already reduces optimizer-state memory by ~1/N per data-parallel replica without degrading quality. In Figure 2, DCTAdamW underperforms AdamW, suggesting that extra low-rank compression of states may be unnecessary—or even harmful—unless it delivers clear end-to-end gains. I personally think compressing optimizer states is unnecessary due to ZeRO optimizer. 

3.  **Necessity of saving computations in Newton–Schulz.** In Trion/Muon-style preconditioning, forward–backward passes dominate step time; Newton–Schulz iterations are typically a small fraction—especially with long context windows. The paper should provide profiler traces showing that low-rank $b_t$ meaningfully reduces *end-to-end* step time or time-to-target. Please also compare with vanilla Muon (no low-rank) to assess any convergence slowdown from low-rank preconditioning, and include both complexity estimates (per-token FLOPs) and wall-clock measurements to substantiate the claimed benefit.

### Questions
1. **Error-feedback memory overhead**. Trion’s error-feedback buffer appears to store a full-size residual per parameter. Does this negate the memory savings from low-rank gradient projection?

2. **Low-rank during forward–backward**. Beyond optimizer/state compression, can the low-rank structure be exploited to reduce the dominant forward–backward costs (FLOPs and memory)—e.g., via factored weight updates or structured bases that lower gradient-computation cost?

3. I noticed that the work [R1] uses a similar idea to save SVD computations. Could the authors highlight the difference from [R1]

[R1] Wavelet Meets Adam: Compressing Gradients for Memory-Efficient Training

### Soundness
3

### Presentation
3

### Contribution
2
