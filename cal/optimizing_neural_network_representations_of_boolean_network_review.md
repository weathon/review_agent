=== CALIBRATION EXAMPLE 29 ===

# Final Consolidated Review
## Summary

This paper addresses the problem of losslessly optimizing neural network (NN) representations of Boolean networks (BNs) — a specific pipeline in which Boolean logic circuits are mapped to Heaviside-threshold NNs to exploit hardware matrix-multiplication acceleration. The authors establish that optimizing the NN size is equivalent to minimizing the multilinear polynomial (MP) representation of each Boolean function sub-problem, formulate this as an integer linear program (ILP) with linear σ-equivalence constraints (Theorem 3.1, Corollary 3.1), and accelerate the optimization by exploiting NPN equivalence classes shared among sub-problems. Experimentally, they demonstrate up to 70%/60% reduction in connections/neurons relative to the state-of-the-art (SOTA) of Gavier et al. (2023), and up to 34.3×/5.9× speedup in optimization time over naive and caching baselines.

---

## Strengths

- **Elegant theoretical characterization of σ-equivalence.** Theorem 3.1 — that the σ-equivalence class of an MP is exactly the feasible set of a linear inequality system — is non-trivial and directly enables a tractable ILP formulation. This is the key insight of the paper and is not found in prior NN compression or logic synthesis literature.

- **Principled lossless guarantee.** Unlike standard pruning/quantization, the proposed method provably preserves functional equivalence (∀x ∈ {0,1}ⁿ, NN(x) = BN(x)) for the specific Heaviside-threshold architecture resulting from technology mapping. This is non-trivial because it relies on the σ-equivalence class structure rather than heuristic removal.

- **PN-invariance enables non-trivial NPN class reuse.** Lemma 3.2 (both the uniform and degree criteria are PN-invariant) is the key enabler for the NPN acceleration, and the authors carefully handle the phase-assignment subtlety via the permuted phase assignment τ_{π(φ)} in Theorems 3.3–3.4. This is technically careful and closes a gap that a naive NPN reuse scheme would miss.

- **Architecture-aware depth-constrained optimization.** The `optMaintainDepth` algorithm, based on the leeway concept (Definition 3.8), provides a clean way to preserve latency (depth) constraints while still compressing the layer-merged NN. The O(V(V+E)) algorithm is practical and the trade-off between `optAll` and `optMaintainDepth` is well-illustrated.

- **Consistent experimental validation of speedup.** Figure 6 shows that the NPN-class algorithm consistently outperforms both naive and caching baselines across all four circuits and all tested K, with the speedup persisting under both MMP criteria. This is credible evidence that NPN classification provides genuine structural reuse.

---

## Weaknesses

### Fatal
None identified.

### Major

- **No inference latency measurement — the core performance claim is unsubstantiated.** The abstract states "faster simulation performance" and the introduction motivates the work by "exploit[ing] hardware matrix-multiplication acceleration." However, no experiment measures the actual wall-clock inference time of the optimized NN on any hardware (GPU, neuromorphic chip, or FPGA). Reducing structural size does not trivially translate to lower latency on modern accelerators (memory bandwidth, batching, sparsity support all intervene). The paper reports optimization time speedups (how fast the compilation runs), but this is entirely different from simulation/inference speedups. The headline claims in the abstract about "faster simulation performance" and "high-throughput circuit simulation" are currently unsupported. This is the most significant empirical gap.

- **No comparison with logic synthesis baselines.** The paper motivates NN representations as superior to direct circuit simulation for hardware acceleration, yet provides no comparison against running the original BN (after ABC/standard optimization) directly, or against the SOTA NN representation *before* optimization on inference tasks. Without this, the reader cannot assess whether the proposed NN pipeline with MMP optimization outperforms a well-optimized circuit mapped to conventional hardware. This is not about ABC being a competitor — it is about establishing the practical benefit of the NN representation that motivates the entire paper.

- **Optimality gap between ℓ₁ objective and true ℓ₀ minimum is uncharacterized.** The paper correctly acknowledges (below Lemma 3.1) that ℓ₁ minimization is a relaxation of the ℓ₀ objective and that "the solution to the relaxed problem may differ from the actual minimum." However, it nowhere quantifies or bounds this gap empirically. The reported compression figures (70%/60%) reflect ℓ₁-ILP quality, not provably optimal lossless compression. Even a small case study comparing the ℓ₁-ILP solution against exhaustive search for small K (e.g., K ≤ 4) would establish how much is being left on the table. As it stands, calling the result "minimal" is technically accurate per Definition 3.3 (which is stated w.r.t. ℓ₁), but the framing throughout the paper implies near-optimal compression without evidence.

- **Correctness of layer merging with MMP intermediate vertices lacks formal proof.** The inter-depth composition (Equations 3–5) crucially relies on W₂^(d) being linear (no threshold). When a non-PO vertex at depth d is in 𝓜 (MMP-represented), its output is σ(W₂σ(W₁x + b₁)), i.e., W₂ has a threshold applied. The `optMaintainDepth` algorithm handles this by augmenting the graph with a predecessor hidden vertex and restricting 𝓜 to vertices with positive leeway, but the paper does not provide a formal proof that the composed network remains functionally equivalent after merging under these conditions. This is especially non-trivial when multiple MMP vertices appear at the same depth. Section 3.3 gives an informal description but stops short of a theorem.

### Minor

- **ILP solver unspecified.** The experimental results depend entirely on an ILP solver (for K up to 11, the variable space is 2^11 = 2048 integers). The main text does not identify the solver used (Gurobi, CBC, SCIP, etc.), solver version, or configuration. Performance of ILP solvers varies by an order of magnitude across implementations. This is a reproducibility issue that should be addressed in the main text, not only the appendix.

- **Functional equivalence verification protocol absent.** The paper claims "lossless" optimization but does not describe how functional equivalence of the full composed network was verified after optimization. For safety/security applications cited in the introduction, a formal verification step (e.g., SAT-based equivalence checking) would be needed. At minimum, the paper should describe how the correctness of the experiments was validated.

- **Figure 5(b) counter-intuitive trend unexplained.** For `optMaintainDepth` (layer-merged NN), Figure 5(b) shows compression increasing with K for some circuits and decreasing for others, with a less clear trend than Figure 5(a). The paper states "the degree criterion leads to larger reduction" but gives no mechanistic explanation for why compression changes with K in the depth-constrained setting, nor why the trend differs from Figure 5(a). This merits at least a brief discussion.

- **Benchmark scope in main paper.** The main paper reports results on only four circuits. While the appendix includes additional circuits (acknowledged in the paper), a summary table of all circuits with sizes and compression ratios in the main paper would significantly strengthen the empirical case.

### Tiny

- **Three trials for timing.** For optimization runs that span 10⁰ to 10⁴ seconds, three trials is marginal for statistical robustness. The error bars shown in Figure 6 appear small in most cases, which is reassuring, but this should be noted.

- **Missing K=11 naive baselines.** The paper notes that Naive (uart, K=11) and (aes, K=11) are unreported "due to compute constraints." The speedup comparison is thus incomplete at the largest K for the largest circuit.

---

## Nice-to-Haves

- **Runtime breakdown by component.** A stacked chart separating NPN canonicalization time from ILP solving time, for varying K, would clarify at what K the NPN overhead itself becomes a bottleneck and inform the practical operating range of the method.

- **Comparison of ℓ₁-ILP against exhaustive ℓ₀ search for small K.** Even for K ≤ 4 or K ≤ 5, an empirical characterization of the gap between the ℓ₁-ILP solution and the true ℓ₀ minimum would be informative and easy to compute.

- **Scalability analysis for K > 11.** A brief discussion or plot of ILP solve time as a function of K extrapolated beyond K=11 would help practitioners understand the method's upper limits.

- **Hardware inference latency experiment.** Even a single proof-of-concept measurement on a target GPU or neuromorphic accelerator comparing optimized vs. unoptimized NN inference time would substantiate the "high-throughput simulation" motivation.

- **Walkthrough example for NPN optimization.** Section 3.2 is mathematically dense. A step-by-step numerical example tracing a specific BF through NPN canonicalization, MMP computation, and back-transformation would substantially aid readability for readers unfamiliar with NPN classification.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Deterministic" is misleading (Harsh Reviewer Concern 1).** The abstract uses "deterministic algorithm" correctly — it means the procedure is not stochastic (produces the same output on each run), which is accurate. The abstract explicitly acknowledges NP-hardness in context. This is not a framing error.

- **Heaviside convention concern (Harsh Reviewer Concern 3).** The paper defines σ explicitly in footnote 1 (σ(x) = 1 if x > 0) and the σ-equivalence class structure in Theorem 3.1 is derived from this specific convention. The OR example (p' = a+b as MMP of p = a+b−ab) is verifiably correct under this convention. Not a real concern.

- **Remark 2.1 depth count issue.** The paper explicitly notes that the "no skip connections" assumption is achieved by adding identity vertices, and depth claims are conditioned on this. Not a gap.

- **Output polarity / output threshold (Concern 6 framing).** The paper explicitly addresses this in §3.3: MMP-represented vertices require σ on their output, and the algorithm explicitly accounts for this added layer in the architecture-aware optimization. The concern is partially valid (the formal proof gap is captured under Major Weaknesses), but the framing that "this introduces a non-linearity in what was a linear layer" is already handled by the paper's own design.

- **Speedup baseline asymmetry as an inflated claim.** The 34.3× speedup vs. Naive is legitimate to report — Naive is a proper lower bound on algorithmic baselines (it is the baseline without any structure exploitation). The 5.9× vs. Cached is reported alongside it and is clearly labeled. Reporting both is standard practice; this is not misleading.

- **EDA-specific venue fit concern.** The paper addresses neural network representations for hardware simulation — a topic of direct relevance to the neurosymbolic AI and ML systems communities. The venue fit concern reflects a community preference judgment that is out of scope for this review.

- **Cyclic Boolean networks.** Definition 2.4 restricts to DAGs, which is the standard setting for combinational logic and for the prior work (Gavier et al., 2023) this paper extends. Criticizing the absence of sequential/cyclic network support is scope creep.

- **Broader impact in appendix.** Not a substantive weakness; appendix placement of broader impact is common practice.

---

## Novel Insights

The most underappreciated insight in this paper is the connection between *σ-equivalence under threshold* and *linear feasibility*. The fact that the entire equivalence class [p]_σ — which is a priori an infinite set of integer vectors — is exactly characterized by a finite system of linear inequalities (Theorem 3.1) is non-obvious and makes the otherwise NP-hard ℓ₀ problem tractable (as an ℓ₁-ILP). The secondary insight — that PN-invariance of the MMP criterion is sufficient (but phase-invariance is not, requiring the permuted phase assignment trick) — is a clean structural observation that rescues the NPN acceleration from what would otherwise be a subtle correctness bug. These two insights together constitute a genuinely novel interface between linear programming, Fourier analysis of Boolean functions, and NN synthesis.

---

## Suggestions

1. **Add an inference latency experiment.** Run the optimized and unoptimized NNs on a GPU or available hardware, reporting wall-clock throughput for batch Boolean network simulation. Even one circuit at one K would validate the "faster simulation performance" claim.

2. **Provide a formal theorem + proof (or proof sketch in the appendix) for the correctness of layer merging when intermediate vertices are MMP-represented.** The current informal discussion in §3.3 is insufficient given that functional equivalence is the central claim of the paper.

3. **State the ILP solver and version in the main text.** Add a sentence to §4 or the experimental setup.

4. **Add a brief empirical optimality-gap study.** For K ≤ 4, compare the ℓ₁-ILP solution against the true ℓ₀ minimum (computable by exhaustive search). Report the fraction of cases where they agree.

5. **Explain Figure 5(b) trend mechanistically.** Provide a one-paragraph explanation of why compression under `optMaintainDepth` behaves differently from `optAll` as K grows, and why the degree criterion consistently dominates.

6. **Add functional equivalence verification detail.** Describe how the lossless claim was validated for the full composed networks in the experiments (e.g., exhaustive truth-table comparison for smaller circuits, or SAT-based checking).

---

**Axis Assessment:**
- *Novelty*: Meaningful — the σ-equivalence linear characterization and PN-invariance-based NPN acceleration are genuinely new contributions not present in prior NN compression or logic synthesis work.
- *Technical soundness*: Mostly sound, with one notable gap in the formal proof of layer merging correctness with MMP intermediate vertices.
- *Empirical support*: Incomplete — compression results are well-supported, but the central motivation (hardware simulation speedup) is not empirically validated; EDA comparison baselines are absent.
- *Significance*: Moderate — highly significant for the NN-based technology mapping niche; broader ICLR relevance depends on whether the hardware simulation claims can be substantiated.
- *Clarity*: Good overall; §3.2 is dense and would benefit from a worked example.

# Actual Human Scores
Individual reviewer scores: [8.0, 8.0, 6.0, 5.0]
Average score: 6.8
Binary outcome: Accept
