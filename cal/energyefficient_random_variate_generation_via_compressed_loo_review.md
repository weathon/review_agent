=== CALIBRATION EXAMPLE 70 ===

# Harsh Critic Review
Now I have enough information to write a thorough review. Let me produce it.

---

## Section-by-Section Critical Review

### Title & Abstract

The title accurately reflects the contribution. However, the abstract contains numerical inconsistencies with the rest of the paper. The abstract states "up to 40% savings in time and 50% in energy compared to state-of-the-art approaches" and "100× time improvement" versus Python, but the contributions section (§1) lists "30-40% faster and saves 25-50% energy," and the conclusion says "up to 50% in energy." The shifting figures (25–50%, "up to 50%", "up to 40%") are not wrong individually but signal imprecise summary language that leaves the reader uncertain about what the method actually achieves. A single, coherent quantitative headline would be stronger.

The abstract also claims cLUT achieves "near-optimal entropy cost" — this is a meaningful theoretical property, but the precise sense in which it is near-optimal (relative to the Knuth-Yao bound of H(**p**)+2 bits) is not stated, leaving readers to work it out from §4.

---

### Introduction & Problem Formulation

The energy motivation is timely and well-placed. The problem formulation is crisp: generate samples from a finite discrete distribution with n outcomes, probability vector **p**, efficiently and at controllable precision.

The framing of competing methods as "approximate" while cLUT is "exact" needs more careful qualification. The paper correctly observes that floating-point arithmetic in NumPy/PyTorch/JAX introduces uncontrolled rounding errors. However, cLUT is also approximate when the input probabilities p_i are irrational — the method quantizes them to fi/2^b. The paper's claim to "exactness" means "exact relative to the discretized distribution," which is a meaningful but narrower guarantee than the word "exact" implies. Theorem 1 (Appendix C) correctly bounds the KL divergence as a function of the minimum frequency κ, but this applies to cLUT as well. The distinction from Alias is that cLUT's rounding errors are *controlled and quantified* rather than uncontrolled — this is the genuine advantage and should be stated more precisely in §1.

---

### Related Work

Coverage is solid: Knuth-Yao DDGs, Lumbroso, the interval algorithm, FLDR, Alias, and Index methods are all discussed. The prior work by Marsaglia et al. (2004) on compressed lookup tables is acknowledged, and the paper clearly distinguishes its single-table direct indexing from Marsaglia's multi-table approach. This is an important and correctly made distinction.

There is a broken citation at line 116: "**?** improved this to H(**p**)+2 bits with faster sampling speed for the ALDR algorithm." This is obviously Draper & Saad (2025), the same paper cited elsewhere. This is a substantial editorial error that should be corrected regardless of whether it is a parser artifact or a genuine submission error.

The related work omits any comparison to the Index method (Chen & Asau, 1974) in the experiments despite mentioning it here. This is a gap.

---

### Method / Approach

**Compression scheme and sampling.** The core idea — represent the lookup table as a 2D array of r+1 rows and 2^c columns, fill via binary expansion of frequencies, sample the row from a truncated geometric distribution and the column uniformly — is elegant, well-motivated, and clearly described. The compression ratio ρ = 2^r/(r+1) growing exponentially in r is a genuine advantage.

**Choice of r.** Algorithm 2, line 2 defines r as:

> max{v ∈ [0,b] : Σ_j^w Σ_i^n f_i^(j) · 2^{v-b-1} ≤ 1 ∀w ∈ {0,...,b}}

This is the most technically critical line in the paper and is presented without prose explanation. A reader must reverse-engineer why this particular constraint on the cumulative bit-level counts determines the maximum legal number of rows. The paper states that the compressed and naive table coincide for r=0, c=b, and that r=0 is always valid, but the intuition for maximizing r subject to this constraint is unclear and deserves at least a sentence of explanation in the main text.

**Rectification procedure (Algorithm 3).** The distribute() function redistributes bit-level entries from higher rows to lower rows to equalize row lengths. The correctness relies on the property that moving a count from level k to level k-1 doubles it (since each row represents half the frequency of the row above). This is correct but the termination guarantee and the correctness invariant (that total frequency is preserved) are not proven in the text. For a paper at ICLR making theoretical claims, a brief proof sketch here would strengthen the work.

**Bit efficiency.** The expected number of random bits per sample is stated as b-r+2-2^{-(r-1)}. For large r this approaches b-r+2 = c+2. This matches the Knuth-Yao lower bound H(**p**)+2 asymptotically only for near-uniform distributions (where H(**p**) ≈ b ≈ c+r). For low-entropy distributions the entropy H(**p**) is small while b can still be 16-23 bits. The paper correctly shows empirically (Figure 4, bottom subfigure) that consumed bits approach H(**p**) for high-entropy distributions, but does not show clearly whether near-optimality holds for low-entropy distributions. This is not a fatal gap, but it weakens the "near-optimal entropy" claim in the abstract for distributions that are the common case in ML (e.g., peaked class distributions, rare-event heavy tails).

---

### Experiments & Results

**Evaluation hardware.** All experiments are conducted on a single laptop (Intel i7-1255U, 16 GiB DDR4). This is a legitimate and reproducible setup for a microbenchmark paper, but all quantitative claims are tied to this one microarchitecture. Intel's i7-1255U is a hybrid P+E core design; the paper pins experiments to P-cores. Results on AMD, ARM/Apple Silicon, or RISC-V could differ substantially, especially for energy efficiency due to different cache hierarchies and instruction-level parallelism. The paper should include a caveat that results are architecture-specific.

**Python vs. C comparison.** The 10–100× speedup over NumPy/PyTorch (Figure 3, Table 1) is heavily confounded: cLUT is implemented in C while the competitors are Python library calls. The comparison is legitimate only if the paper's intended use case is "replace NumPy discrete sampling with a C extension," which is indeed a reasonable use case. But the framing ("we achieve a 100× time improvement") in the abstract overstates this as a property of the *algorithm* rather than the *implementation stack*. JAX with cLUT vs. JAX default (Appendix F) is a fairer comparison: the speedup is substantial but more modest.

**C-vs-C comparison (Figure 4, Table 2).** This is the core algorithmic comparison and is much more meaningful. cLUT is 30-40% faster and 25-50% more energy-efficient than FLDR and Alias on the test distributions. These are solid, consistent wins. The standard deviations in Table 2 are large (e.g., ALDR energy: 263±50 nJ for n∈[10^3,10^4)), which is understandable given the diverse distribution set but makes individual pairwise comparisons noisy.

**Distribution diversity.** The main evaluation uses discretized Exponential distributions with varying parameters to span "a broad range of entropy values." While Appendix G adds sparse Dirichlet distributions, the set of test distributions is still limited. In ML, the most common discrete sampling scenarios are: softmax outputs from neural networks (peaked, near-low-entropy), categorical embeddings (variable), and importance-sampling weights (often very heavy-tailed). It is not obvious that Exponential and Dirichlet families cover the distributions where cLUT is actually used in practice.

**Variable precision across distribution sizes.** The bit precision b is set to 16 for n<10^4, 20 for n<10^6, and 23 for n<10^7. This is necessary (because b ≥ log2(n) roughly) but means that comparisons across distribution sizes conflate two variables (n and b) simultaneously. The paper does not isolate the effect of b vs. n in the main text.

**Memory.** Figure 5 shows cLUT "consumes slightly more memory at peak times." The "slight" qualifier is questionable: during preprocessing, cLUT must construct intermediary structures. The paper says peak memory during preprocessing is highest for cLUT but the final compressed table is small. The exact peak memory overhead should be quantified more precisely (e.g., a factor of X over Alias).

**Break-even analysis.** Figure 5 (right panel) shows the break-even point n* for sampling time vs. Alias is "approximately linear in distribution size." This implies that for n=10^8, you need ~10^8 samples to break even — which is a significant amortization requirement. This is buried in the discussion and should be emphasized as a genuine limitation.

**TrueSkill application (§4.2).** The application is deliberately chosen because "sampling accounts for a significant share of total energy consumption." This is selection bias. TrueSkill is not a modern deep learning workload; it is a factor graph-based inference system from 2006. The choice of a non-deep-learning application for an ICLR submission is puzzling. The paper reports 72% reduction in execution time and 34% in energy vs. NumPy's discrete sampler — but since sampling is the dominant cost, these numbers primarily reflect cLUT's per-sample speedup applied to a sampling-heavy task, not a general insight about how cLUT helps ML pipelines.

**Diffusion model (Appendix I).** The model is a two-layer MLP trained on synthetic bimodal data with batch size 8 for 3×10^5 steps. The enormous energy savings (37–65%) are impressive but almost certainly driven by the fact that, at batch size 8, sampling constitutes an unusually large fraction of total training time. Real diffusion model training (e.g., DDPM on CIFAR-10 or ImageNet) uses GPU-batched operations where sampling overhead is amortized differently. The experiment is presented in Appendix I and the paper wisely does not make it a main result, but the use of the word "core ML problem" in the appendix is an overstatement for a toy experiment.

**Statistical validity.** The 10-repetition regime for preprocessing and 10^7-repetition regime for sampling appears reasonable. The TrueSkill p-values (>0.2 for t-tests on moments) support distributional equivalence. No multiple-comparison correction is applied, but the number of tests is small.

**Security configuration.** The paper disables CPU security features (mitigations against PLATYPUS and related side-channel attacks, which the paper itself cites) to reduce energy measurement noise. This is a reasonable scientific choice for reproducibility of microbenchmarks, but the paper should more explicitly note that production deployments would not run with these settings and that energy measurements could differ.

---

### Writing & Clarity

The description of the compression scheme in §3 is clear and the running example (Figures 1b and 2) is effective. Section 3's "Preprocessing step" paragraph is dense and would benefit from a forward reference to Algorithm 2 earlier in the discussion. The choice of notation (r+1 rows but I ∈ {1,...,r+1}) creates minor confusion because the "(r+1)-th row" behaves identically to the r-th row in the truncated geometric sampler — this should be explained explicitly.

---

### Limitations & Broader Impact

The paper's limitations section is essentially absent — there is no explicit limitations paragraph in the main text. Issues that deserve acknowledgment include:

1. **Single-architecture evaluation.** Results are from one Intel laptop. Generalizability is asserted but not demonstrated.
2. **High preprocessing cost.** The break-even point scales linearly with n, making cLUT unsuitable for one-shot or low-sample-count scenarios with large distributions.
3. **The "exactness" claim.** Quantized distributions are used, so the claim to exactness requires the reader to understand the discretization step. The KL divergence from the true continuous target distribution is not bounded end-to-end in the main text.
4. **Peak memory during preprocessing** is higher than competing methods, which matters in memory-constrained settings (e.g., mobile/edge devices — paradoxically, the very settings where energy efficiency matters most).
5. **No multithreaded or vectorized comparison.** The paper explicitly avoids vectorized implementations "to improve understandability." This is a legitimate choice for a reference implementation, but it means competing methods may be more favorable in comparison than a fair engineering comparison would show.

---

### Overall Assessment

The paper presents a technically sound and practically motivated contribution: a compressed lookup table scheme for discrete sampling that is faster and more energy-efficient than state-of-the-art methods in the C-vs-C comparison, and dramatically faster than Python library calls due to the implementation stack advantage. The compression idea — mapping a naïve 1D lookup table to a 2D structure indexed via a truncated geometric row sampler — is elegant, and the claim that this achieves near-optimal entropy efficiency is credibly supported. The energy measurement methodology is more rigorous than typical ML systems papers.

However, several issues must be addressed before acceptance at ICLR. The "exactness" claim in the introduction conflates the method's controlled-quantization guarantee with true exactness, and needs to be more precisely scoped. The broken citation ("**?**") for the ALDR algorithm is a clear error that undermines confidence in the manuscript's readiness. The ML application examples (TrueSkill, toy diffusion model) are too narrow and artificially sampling-heavy to demonstrate the method's general impact on modern ML workflows; stronger motivation for ICLR specifically would require showing cLUT's benefit in a realistic training or inference pipeline where sampling is not the sole bottleneck. The single-hardware evaluation and the missing break-even discussion as an explicit limitation are additional gaps. Finally, the absence of any comparison with the Index method (cited in related work but absent from experiments) is unexplained. The core algorithmic contribution does stand, but the paper reads more naturally as a systems/algorithms venue submission (e.g., ALENEX, SC, or ICML as an efficiency paper) than as an ICLR contribution, and the authors should substantially strengthen the ML application evidence and address the theoretical precision issues to be competitive at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Compressed Lookup Tables (cLUT), an exact random variate generation method designed to reduce computational time and energy consumption in machine learning applications. The approach utilizes a novel lossless compression strategy for lookup tables based on the binary expansion of outcome frequencies, enabling sampling via fast geometric row and uniform column index generation. The method is evaluated against Python libraries and C-based state-of-the-art sampling algorithms, demonstrating significant speedups (up to 100x against Python) and energy savings (up to 50% against SOTA C implementations) in static distribution scenarios like TrueSkill and diffusion models.

### Strengths
1.  **Focus on Energy Efficiency:** Unlike most ML sampling papers that focus solely on latency, this work provides rigorous energy measurements using RAPL counters on x86 CPUs. This aligns well with current concerns regarding the carbon footprint of AI deployment (Section 1, Section 4).
2.  **Exact Sampling Guarantees:** The method guarantees exact sampling (subject to finite precision discretization) compared to common approximations like the Alias method (Section 2). It includes theoretical bounds on KL divergence related to discretization precision (Appendix C, Theorem 1).
3.  **Strong Empirical Validation on ML Tasks:** Beyond microbenchmarks, the authors evaluate the method in end-to-end pipelines such as TrueSkill (importance sampling) and diffusion models, showing concrete reductions in execution time and energy (Section 4.2, Tables 3, 7).
4.  **Entropy Efficiency:** The sampling scheme approaches the information-theoretic lower bound (Knuth-Yao) for bit consumption, which is theoretically significant and practically relevant for entropy-constrained environments (Section 4, Figure 4).

### Weaknesses
1.  **Preprocessing Latency and Static Constraint:** The preprocessing step (building the compressed table) is significantly slower than state-of-the-art alternatives like the Alias method (Table 1, Figure 5). The method assumes static distributions; dynamic updates would require costly reconstruction, limiting applicability to online learning scenarios without modification.
2.  **Memory Overhead:** While the compressed runtime table is smaller, the peak memory usage during preprocessing is higher than competitors (Figure 5, Subfigure 1). For extremely large distributions, the initial memory spike could exceed cache limits or available RAM on edge devices, counteracting the benefits of the compression.
3.  **Novelty vs. Prior Work:** The approach builds on lookup tables, specifically referencing Marsaglia et al. (2004). The distinction between the proposed compression scheme and existing compressed lookup table techniques needs a clearer, more detailed comparison in the Related Work (Section 2) to establish unique contribution against Marsaglia’s method.
4.  **Python Comparison Context:** The 100x speedup over Python libraries (Table 1) is partly due to the C implementation vs. Python overhead rather than purely algorithmic superiority. While illustrative, this specific comparison might be seen as comparing "apples to oranges" to a degree, though the authors mitigate this by also benchmarking against C-based SOTA methods.
5.  **Discretization Impact on Convergence:** While Theorem 1 bounds KL divergence, the paper does not fully quantify how this discretization error propagates through iterative optimization (e.g., in VAEs or Diffusion) beyond the specific TrueSkill/Diffusion examples (Appendix I, Section B).

### Novelty & Significance
The paper demonstrates significant novelty by combining entropy-optimal sampling with a specific geometric compression scheme optimized for modern memory hierarchies. The significance lies in addressing the energy bottleneck in probabilistic inference and generative modeling, a topic gaining traction in the AI community ("Green AI"). For ICLR, this fits well within the scope of efficient learning representations and inference techniques, especially given the integration with JAX and GPU evaluation (Appendix F). However, the theoretical novelty requires careful framing against established randomized algorithm literature to avoid being seen as merely an engineering optimization of existing methods.

### Suggestions for Improvement
1.  **Clarify Dynamism:** Provide a discussion or small study on how the method handles updates to the distribution. If dynamic, discuss the amortized cost or alternative strategies for frequent updates to address the preprocessing bottleneck.
2.  **Deepen Marsaglia Comparison:** In Section 2, explicitly differentiate the compression logic proposed here from Marsaglia et al. (2004). A specific analysis of why this compression yields better cache locality or instruction throughput would strengthen the claim of superiority.
3.  **Quantify Error Propagation:** Expand the analysis in Appendix C regarding the impact of the KL divergence bound on the convergence rate of gradient descent in variational inference, rather than just static posterior accuracy in TrueSkill.
4.  **CPU vs. GPU Nuance:** While GPU integration is shown (Appendix F), clarify if the speedup persists when accounting for the memory bandwidth on the GPU, as the method relies heavily on table lookups which may be memory-bound rather than compute-bound.
5.  **Benchmarking Rigor:** Standardize the "SOTA" comparisons by ensuring compiler optimizations (e.g., vectorization, unrolling) are explicitly compared or neutralized across all C implementations to ensure the cLUT efficiency is not due to under-optimized baselines.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **GPU Energy Measurements:** The core energy claims rely on CPU RAPL, but ML workloads run on accelerators; measure GPU power draw to verify efficiency gains transfer to actual deployment hardware.
2. **Dynamic Distribution Overhead:** Benchmark preprocessing time when distributions update frequently (e.g., per batch), as static distribution assumptions rarely hold in attention mechanisms or RL.
3. **FP32/FP64 Memory Footprint:** Report exact memory consumption for standard 32-bit and 64-bit precision to validate the "arbitrary precision" claim against realistic hardware limits.
4. **Standard Model Training Loop:** Evaluate speed/energy within a full-scale training job (e.g., ResNet or Transformer) rather than isolated microbenchmarks or niche applications like TrueSkill.

### Deeper Analysis Needed (top 3-5 only)
1. **Memory Bandwidth Saturation:** Analyze cache miss rates and memory bandwidth usage, as lookup-heavy methods often become memory-bound rather than compute-bound at scale.
2. **Entropy Source Cost Isolation:** Quantify the energy cost of generating the required random bits; cLUT's efficiency gains may disappear if the underlying RNG is the dominant cost.
3. **Quantization Error Impact on ML:** Empirically demonstrate whether the $b$-bit discretization error degrades model convergence or final accuracy in standard neural network tasks.

### Visualizations & Case Studies
1. **Precision vs. Memory Trade-off Curve:** Plot table size against precision $b$ to visually identify the threshold where memory requirements become prohibitive for common ML precisions.
2. **System-Level Energy Breakdown:** Visualize the split between preprocessing, sampling logic, and entropy generation energy to expose where actual savings occur.
3. **Cache Locality Heatmap:** Compare memory access patterns of cLUT vs. Alias method to reveal potential cache thrashing that undermines speed claims.

### Obvious Next Steps
1. **SIMD/Vectorized Implementation:** The paper explicitly excludes vectorization; adding this is mandatory to claim state-of-the-art throughput for ML workloads.
2. **Integrated PRNG Benchmarking:** Measure total system energy including the random bit generator, not just the sampling algorithm in isolation.
3. **GPU Kernel Comparison:** Compare against native GPU sampling kernels (e.g., cuRAND) rather than CPU-based Python wrappers ported to GPU.

# Final Consolidated Review
## Summary

This paper introduces compressed lookup tables (cLUT), a method for fast and energy-efficient exact sampling from arbitrary discrete distributions. The approach compresses a naive lookup table into a 2D structure indexed by geometric (row) and uniform (column) random variates, achieving near-optimal entropy consumption while enabling single-lookup sampling with controlled precision guarantees.

## Strengths

- **Energy measurement rigor:** Unlike most ML efficiency papers, this work provides detailed energy measurements using RAPL counters, isolating CPU core and package domains, and validates with wall-socket power measurement (MCP device) for the TrueSkill application. The methodology accounts for noise sources (security mitigations, CPU frequency scaling) and reports standard deviations.

- **Strong algorithmic comparison:** The C-vs-C comparison against ALDR, FLDR, and Alias (Table 2, Figure 4) shows consistent 30-40% speed improvements and 25-50% energy savings across distribution sizes. This is the fair comparison since implementation language is controlled.

- **Near-optimal entropy efficiency:** The sampling scheme provably uses b-r+2-2^(-(r-1)) random bits per sample, approaching the Knuth-Yao lower bound H(p)+2. Figure 4 (bottom) empirically confirms that consumed bits approach the entropy H(p) for high-entropy distributions.

- **Controlled discretization error:** Unlike standard library samplers that produce uncontrolled floating-point approximations, cLUT quantizes probabilities to precision b with bounded KL divergence (Theorem 1, Appendix C). This is a genuine advantage for applications requiring reproducibility or theoretical guarantees.

## Weaknesses

- **Single-architecture evaluation:** All experiments run on one Intel i7-1255U laptop. Energy efficiency depends on memory hierarchy and instruction-level parallelism, which differ substantially across AMD, ARM/Apple Silicon, and RISC-V. The paper should explicitly caveat that results may not generalize.

- **Peak memory overhead during preprocessing:** Figure 5 shows cLUT has the highest peak memory usage among compared methods during table construction. This matters for memory-constrained deployments (edge devices)—paradoxically the settings where energy efficiency matters most. The paper acknowledges this but does not quantify the overhead factor.

- **Preprocessing amortization requirement:** The break-even analysis (Figure 5, right) shows the sampling break-even point scales roughly linearly with distribution size. For n=10⁸ outcomes, approximately 10⁸ samples are needed before cLUT's sampling speed advantage overcomes its higher preprocessing cost. This makes cLUT unsuitable for one-shot or low-sample-count scenarios with large distributions—a limitation not prominently discussed.

- **Application scope limited:** TrueSkill (2006 factor graph inference) and a toy diffusion model (Appendix I, two-layer MLP, batch size 8) do not demonstrate impact on modern deep learning workloads where sampling is typically GPU-batched. The 37-65% energy savings in the diffusion experiment reflect an artificially sampling-heavy configuration; real training loops would show smaller relative gains.

- **Broken citation:** Line 116 contains "**?** improved this to H(p)+2 bits"—a missing reference to the ALDR algorithm that undermines manuscript readiness.

- **"Exactness" framing needs precision:** The paper claims the method is "exact" (§2), but it quantizes probabilities to b-bit precision, introducing controlled KL divergence from the target. The genuine advantage is *controlled and quantified* discretization error, not true exactness. This should be stated precisely.

## Nice-to-Haves

- **GPU energy measurements:** ML workloads typically run on accelerators; validating energy savings on GPU architectures would strengthen deployment relevance.

- **Dynamic distribution handling:** Real ML systems often update distributions per batch (e.g., attention mechanisms, RL policy distributions). Discussing amortized update costs or incremental reconstruction would broaden applicability.

- **Vectorized implementation:** The paper deliberately excludes SIMD/vectorized code for clarity, but real deployment would require this for fair comparison against optimized library implementations.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Numerical inconsistency claim:** The reviewer claimed inconsistencies between abstract (40% time, 50% energy) and contributions (30-40% time, 25-50% energy). The ranges are appropriate given that actual savings vary by distribution size (Table 2 shows 24% to 63% energy savings depending on n), and the abstract correctly uses "up to" language.

- **Index method comparison demand:** The Index method (Chen & Asau, 1974) is correctly described as requiring "expensive search operations" and is excluded because the comparison is against faster exact/approximate methods (Alias, FLDR, ALDR). Demanding inclusion of an inferior method is scope creep.

- **Marsaglia comparison depth:** The paper states the distinction clearly: "In contrast, our approach uses a single compressed table with direct indexing, eliminating conditional overhead" versus Marsaglia's multi-table approach requiring conditional branching. This is a substantive technical difference, not merely "engineering optimization."

## Novel Insights

The geometric decomposition of the lookup table—mapping frequencies to binary expansions and sampling rows from a truncated geometric distribution—is an elegant insight that bridges information theory (entropy-optimality) with systems efficiency (cache locality, single memory access). The observation that memory subsystems dominate energy consumption (rather than pure compute) motivates the compression scheme: fewer active transistors during lookup directly reduces energy even when wall-clock gains are modest. The application to uniform floating-point generation (Appendix K) by interpreting row/column indices as exponent/mantissa is a clever connection that yields better statistical uniformity than standard approaches.

## Suggestions

- Fix the broken ALDR citation before final submission.

- Add a "Limitations" paragraph explicitly discussing: (1) single-architecture results, (2) preprocessing amortization requirements, (3) memory overhead during construction, (4) static distribution assumption.

- Clarify the "exact" claim as "exact relative to the b-bit discretized distribution with bounded KL divergence from the target."

- Report the peak memory overhead factor during preprocessing (e.g., "cLUT requires X× peak memory during construction compared to Alias").

- Consider adding one modern ML experiment where sampling is embedded in a standard training loop (e.g., dropout sampling in a small network) to demonstrate practical relevance beyond microbenchmarks.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 6.0, 8.0]
Average score: 7.0
Binary outcome: Accept
