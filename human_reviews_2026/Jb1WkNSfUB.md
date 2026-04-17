# TileLang: Bridge Programmability and Performance in Modern Neural Kernels

- Decision: Accept (Oral)
- Scores: 8, 8, 4, 8

## Abstract
Modern AI algorithms increasingly adopt fused kernels for performance, but implementing them remains complex due to the lack of fine-grained control in existing compilers like Triton. We introduce TileLang, a controllable programming system for fused neural kernels. TileLang provides explicit tile-level primitives for memory placement, data movement, and parallel scheduling. To guide developers in hardware-aware programming, the TileLang introduces two key techniques: tile inference which models tile programs as fused graphs and automatically deduces tile configuration from partial annotations; and tile recommendation that suggests efficient tile configurations based on hardware profiles and heuristics. TileLang makes it easy to express a wide range of fused attention kernels in under 80 lines of Python code, reducing code size by up to 90% compared to manual implementations. Evaluations show that TileLang achieves up to 5x speedup over Triton on NVIDIA H100 and up to 6 on AMD GPUs, demonstrating its ability to bridge programmability and performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
TileLang is a tile-level domain-specific language (DSL) for developing high-performance AI kernels. It is built on top of TVM, and provides {unified fused tile-level dataflow graph (FTG), tile recommendation, tile inference}, and explicit control over {memory placement, data movement, parallel scheduling} to bridge the gap between programmability and performance.

Overall, I like the paper.

### Strengths
**S1.** Excellent contribution from both research and engineering perspectives. TileLang addresses a real pain point in ML kernel development by bridging the programmability-performance gap, which will greatly help in migrating ML kernels to new hardware platforms, as well as developing new ML kernels on existing hardware.

**S2.** Strong empirical results across diverse kernels and hardware platforms. The benchmarks demonstrate substantial speedups (over Triton, PyTorch) and sufficiently close to the hand-crafted kernels (e.g. FlashMLA).

**S3.** The unified FTG abstraction combined with tile recommendation and tile inference provides a principled approach to kernel optimization. The two-stage workflow effectively balances automation with fine-grained developer control.

### Weaknesses
(Here, the weaknesses are more about how the paper writing can be improved to provide more informative and intuitive insights to the readers, but less about the technical contributions.)

**W1.** Missing comparison with TVM. Since TileLang is built directly on top of TVM, the paper lacks benchmarks comparing against TVM itself. While comparisons with Triton/Torch/hand-crafted kernels are valuable, without TVM we're missing a key component to understand TileLang's contribution over its foundation.

**W2.** Lack of software architecture overview. The relationship between TileLang and TVM is unclear. I would like to see an illustration showing the compilation pipeline: Python TileLang DSL → FTG → TVM IR (TensorIR/Relay IR?) → CUDA. What is the relationship between FTG and TVM's IRs? Is FTG a standalone IR or must it combine with other IRs to fully represent the computation? This is not stated clearly in the paper and would help distinguish TileLang's real contribution from TVM's existing capabilities.

**W3.** Missing direct code comparisons with TVM/Triton/etc. Instead of just describing syntactic differences and abstraction levels, showing actual side-by-side Python code comparing TileLang vs TVM vs Triton would be much more intuitive. ("Shut up and show me the code").

**W4.** No comparison with TaichiLang, which sits at a higher abstraction level and supports autodiff. A performance/LoC/syntax comparison would help position TileLang in the landscape of kernel programming frameworks. Or, at least, a mention of the differences.

### Questions
**Q1.** I was wondering about the benchmarking details - do your performance numbers include the Python wrapper overhead, or are they just measuring the kernel execution time? Also, what warmup strategy did you use and how many repetitions for the statistics?

**Q2.** It would be helpful to see some measurements of compilation and code generation time in your benchmarks. This would help understand the overhead for first-time graph execution, especially for JIT compilation scenarios.

**Q3.** Similar to W3 asking for direct code comparisons, could you provide a direct IR-level comparison between TileLang and TVM for a simple matmul example? It would be very helpful to see the actual IR representations side-by-side to understand how different TileLang's tile-level abstractions are from TVM's TensorIR at the IR level.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose a new domain-specific language (DSL) for developing efficient kernels on modern GPUs, called TileLang. Similar to the widely used tile-level kernel programming language Triton, TileLang also treats tiles as first-class citizens. However, unlike Triton, TileLang exposes more control over underlying hardware components—such as memory scope, tile layout, and thread mapping—to the programmer. This additional control allows developers to implement more efficient kernels and achieve better performance in cases where the Triton compiler cannot automatically apply these optimizations. Extensive experiments demonstrate the advantages of TileLang over existing languages.

### Strengths
1. The problem addressed by this work is highly important. We need effective DSLs to simplify writing peak-performance kernels for modern GPUs, which are increasingly complex to program.
2. Both the implementation and experimental results are solid.
3. The paper is well-written and easy to follow.

### Weaknesses
1. Some claims could be more conservative.
2. Experiments are limited to a few hardware platforms.
3. It would strengthen the paper to compare with more recent related works.

### Questions
Thank you to the authors for submitting to ICLR 2026!

I have been following TileLang for quite some time. It is a solid piece of work that makes meaningful academic contributions and provides a robust implementation suitable for real-world model deployment. In my view, TileLang may be an even better fit for top systems conferences like ASPLOS or OSDI, where the strong implementation and in-depth design details could be more fully appreciated.

I have the following suggestions for improving the paper:

**1. Be more conservative with claims**

In the introduction, the authors use the MLA workload on H100 to highlight TileLang’s advantages over Triton, showing TileLang as over 5x faster. Other workloads reportedly achieve up to 10x speedups. While Triton can indeed be suboptimal in many cases, performance differences of several times are uncommon. The current phrasing may give readers the impression that TileLang consistently outperforms Triton by a large margin. I suggest reporting average speedups and explaining the reasons behind the 5x improvement on Hopper’s MLA workload—such as potential issues in the Triton program implementation or current compiler limitations.

**2. Add experiments on more hardware**

The evaluation primarily focuses on H100 and MI300. To demonstrate the generality of TileLang, it would be valuable to include benchmarks on other widely used architectures like Ampere and Blackwell.
For example, I previously - two or three months ago - benchmarked MHA implementations of TileLang and Tilus [1] on an RTX 4090 (q=k=v=4096, qkv heads=32, dim=128, using the example code from both GitHub projects). TileLang achieved 128 TFLOPs, while Tilus reached 148 TFLOPs. My motivation was to test whether TileLang’s pipelined loop effectively handles the prefetching optimization used in FlashAttention, which differs from that in matrix multiplication.

**3. Compare with more recent related works**

Triton’s programming model is elegant and easy to use, but its abstraction level is higher than that of the underlying hardware. As hardware evolves, achieving peak performance increasingly requires complex optimizations (e.g., software pipelining, warp specialization), which are difficult for the Triton compiler to apply automatically. Consequently, several tile-based DSLs have been proposed that sit between CUDA C and Triton in abstraction level.
To my knowledge, these include Triton Gluon [2], Tilus [1], and TileLang (the subject of this paper). A qualitative comparison among these DSLs would strengthen the paper. If possible, adding quantitative results -- even preliminary ones -- would make the comparison more compelling.

**Minor Suggestion**

Consider adding a short section acknowledging TileLang’s implementation foundation in TVM, to give appropriate credit.

**Summary**

Overall, TileLang is a novel and well-executed work. I strongly support acceptance and would champion this paper. Happy to increase the score if above concerns are addressed.

**References**

- [1] Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation (https://arxiv.org/abs/2504.12984, https://github.com/NVIDIA/tilus)
- [2] Triton Gluon: https://github.com/triton-lang/triton/tree/main/python/tutorials/gluon

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes TileLang, a framework to help simplify GPU kernel programming. The framework automates decisions around memory movement and scheduling, while exposing tile primitives and Pythonic operators. The work evaluates on NVIDIA and AMD GPUs and across a breadth of workloads.

### Strengths
The results are strong and the kernel examples / level of simplicity is compelling. This seems like a useful system for the AI community. The baselines for evaluation are well-chosen and the chosen level of abstraction to expose to the developer (between programming primitives, compiler automations) seems effective.

### Weaknesses
It is not very precise how the automation works within TileLang / what the underlying algorithms are. The automation is very nice, but it's not clear if it breaks down in any settings since that's not fully analyzed/explained to the reader. The paper makes clear *what* is done, but not *how* it's accomplished. 
- It is not clear how TileLang handles the memory access patterns for NVIDIA vs. AMD.
- It’s is not clear where the line is that divides the developer’s tasks from what TileLang automates. 
- It’s clear that the code is compact from the plots, but it’s not explained what deltas from existing frameworks contribute to the code size delta. 
- It is not clear what contributes to the speed ups over the baselines

The paper does not clearly compare to prior work, which makes it difficult to assess novelty. For instance, ThunderKittens (over a year old) is not mentioned in the related works or intro despite also using automatically optimized tile primitives and Pythonic functions. Mojo is another compiled language that works on both AMD and NVIDIA with tiles. In the intro Contribution #1 is tile abstractions, which has been shown in extensive prior work and it’s not clear how TileLang differs. Contribution #2 is tile configuration. Again prior frameworks provide default swizzle patterns and register layouts for matrix instruction alignment; how does the proposed method differ? 

The writing could be clearer in a few places: 
- What is the backend of TileLang? The intro doesn’t explain what language / frameworks it’s built on. For instance, the first time it’s mentioned that TileLang is a compiler/uses compiler passes is on L122. 
- L122: Why are compiler analyses needed versus making the load/store functions use coalesced load patterns and loop tiling to begin with? What additional value is the compiler providing? 

Results:
- Does the MI300X kernel match the performance of FlashAttention-3 on the H100? It would be useful to know the absolute TFLOP numbers everywhere, since PyTorch is a constantly changing baseline. For instance, some analysis has shown that PyTorch SDPA achieves 20-100 TFLOPs in some AMD Docker containers, so it is difficult to understand the results.
- Is there an analysis that shows that the “Pipelined” abstraction is sufficient for all warp specialization a user might want to do? I notice that synchronization is abstracted away from the user. Is there ever a case where the user would want to manage synchronization points more carefully and how does TileLang support this if so? 
- Are the appendix kernels the same on NVIDIA and AMD? 

Overall this is a very solid and useful system, but the research aspects -- how the philosophy and contributions are novel, and how the algorithms work -- have not been sufficiently explained.

### Questions
See questions above.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces TILELANG, a novel programming system that bridges the programmability-performance gap in AI accelerator programming. It treats "tiles" (hyper-rectangular tensor slices) as first-class entities in a Pythonic DSL, enabling explicit control over memory, data movement, and scheduling. Core innovations include:

Tile Recommendation: Uses a roofline model to suggest hardware-aware defaults (tile sizes, memory placement, warp partitioning).
Tile Inference: Autocompletes low-level details (memory layouts, schedules) via constraint propagation from partial specifications or recommendations.
On NVIDIA H100 and AMD MI300X GPUs, TILELANG matches or exceeds hand-tuned libraries (FlashAttention, CUTLASS) and outperforms Triton, with code size closer to manual implementations but simpler.

### Strengths
Solves a Critical Problem: Precisely targets the core trade-off between programmability and peak performance in AI kernel development, a highly relevant issue.

Innovative Human-in-the-Loop Framework: The novel "Tile Recommendation + Tile Inference" model is highly effective, lowering the barrier for entry while retaining expert control and drastically reducing manual tuning effort.

Elegant Programming Model: The tile-centric DSL is clean and expressive, allowing developers to control low-level hardware behavior (memory, parallelism) in a high-level language with concise code.

Highly Convincing Evaluation: Achieves state-of-the-art (SOTA) or near-SOTA performance on both NVIDIA and AMD GPUs with significantly less code than hand-tuned libraries, demonstrating its effectiveness and cross-platform capability.

### Weaknesses
1. The roofline-based analytical model ignores complex microarchitectural details, which could lead to suboptimal suggestions in certain scenarios.

2. There are several works  about kernel generators that are not cited properly.

### Questions
1. What performance trade-offs were made to create a unified API for both NVIDIA and AMD? Does this abstraction hide hardware-specific features that might be key for hitting peak performance on a single platform?

2. How would TILELANG's tile-based abstractions extend to highly irregular computations, such as sparse operations or Graph Neural Networks, where data access patterns are not uniform?

3. The appendix reports impressive tuning times of around 10-15 seconds. However, this is not directly compared to the tuning time of other auto-tuning frameworks (e.g., TVM's Ansor/AutoTVM, Triton's autotuner) for the same tasks. A direct comparison would better highlight the efficiency advantage of the recommendation/inference approach over empirical search.

### Soundness
4

### Presentation
3

### Contribution
4
