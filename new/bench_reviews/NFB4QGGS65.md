## Summary

This paper establishes that GPTQ, when executed in back-to-front dimension order, is mathematically identical to Babai's nearest plane algorithm for the closest vector problem (CVP) on a lattice defined by the Hessian matrix of a layer's inputs. This equivalence provides a geometric interpretation of GPTQ's error propagation step and imports Babai's approximation guarantee to derive a tight, layer-wise error bound for no-clipping GPTQ. Leveraging these insights, the authors propose two no-clipping quantization methods (SSQR and HPTQ) with efficient CUDA kernels, demonstrating improvements over vanilla GPTQ.

## Strengths

- **Genuine and illuminating theoretical insight**: The equivalence between GPTQ and Babai's nearest plane algorithm (Theorem 4) is non-trivial and transforms GPTQ from a sequence of seemingly ad hoc algebraic updates into a well-understood lattice algorithm. This provides a principled answer to why a local greedy procedure works globally—the algorithm is performing orthogonal projections through nested affine subspaces in a CVP-lattice geometry. The geometric visualization (Figures 1–3) and the concept dictionary (Table 1) are effective in communicating this insight.

- **Tight error bound with structural implications**: Theorem 5 provides a proven-tight absolute error bound (1/4 · s_i^T D s_i, where D is the LDL diagonal of the permuted Hessian) and a relative bound for the no-clipping regime. The dependence on pivot order directly motivates the min-pivot heuristic (Algorithm 3) and explains the empirical effectiveness of act-order, which is a valuable byproduct of the theory.

- **Theory-to-practice pipeline**: The paper goes beyond pure theory, translating the no-clipping constraint into two practical methods (SSQR with scale-adjusted outliers, HPTQ with Huffman-coded integers) that outperform original GPTQ on perplexity-vs-bitwidth curves (Figure 4a–b), plus efficient CUDA kernels achieving ~2× speedup (Figure 4c).

- **Rigorous proof that composition is redundant**: Section C.4's proof that adding an extra GPTQ-style correction after Babai is algebraically redundant confirms the equivalence is tight and not an artifact of truncation—a useful negative result.

## Weaknesses

### Major:

- **Novelty relative to QuIP/LDLQ is incompletely delineated**: QuIP (Chee et al., 2023) already provided an LDL-based reformulation of GPTQ (called LDLQ) with an error guarantee. The paper mentions this in Related Work but does not clearly state how its CVP/Babai-derived bound differs from QuIP's existing guarantee—whether it is strictly stronger, weaker, or essentially equivalent in practical regimes. Without this comparison, the incremental novelty of the theoretical contribution beyond QuIP's algebraic framework is unclear. The concurrent Birnick (2025) result provides "a short equivalence proof" of the same GPTQ-Babai connection; the paper should specify what additional depth this work provides beyond that proof (e.g., the OBQ geometric interpretation, the error propagation visualization, the composition-redundancy result). This is a structural concern because the central claim of being "first" to provide a geometric interpretation is contested.

- **The error bound applies only to the no-clipping regime**: The tight bound in Theorem 5 requires Z† = ℤ, which excludes the most common practical setting (INT4/INT8 with finite grids). The paper acknowledges that standard GPTQ "violates the bound" via clipping but does not quantify how often clipping occurs in typical layers, how large the gap between the bound and actual errors is when clipping is present, or whether any approximate bound extends to the clipped case. While the authors propose no-clipping methods (SSQR, HPTQ) motivated by this limitation, the proposed methods do not use the bound itself in their design (HPTQ's scale is chosen via entropy-based binary search, not by minimizing s_i^T D s_i). This weakens the claim that the theory provides "firm footing" for practical methods.

- **Narrow empirical evaluation**: The main results (Figure 4) are on Qwen3-8B WikiText-2 perplexity. Zero-shot benchmarks and LLaMA results are relegated to the appendix. The baselines are RTN, GPTQ, HRTN, and SSQR variants—not including modern vector quantization methods (QuIP#, AQLM, VPTQ) that are highly relevant at the bitwidths considered. For a paper claiming its methods "outperform the original GPTQ," the evaluation is insufficient to establish competitiveness with the state of the art.

- **Min-pivot ordering not used in experiments**: The theoretically motivated min-pivot order (Algorithm 3) is the most direct algorithmic consequence of the Babai-based bound, yet all experiments use act-order. The paper concedes "downstream accuracy gains are modest" (Section 4.5), which weakens the claim that the lattice viewpoint leads to practically effective ordering heuristics.

### Minor:

- **Back-to-front execution order**: The equivalence (Theorem 4) holds when GPTQ is run back-to-front, while standard implementations run front-to-back. The paper does not empirically compare these orders, leaving unclear whether the bound is descriptive of the widely-deployed variant.

- **LLL basis reduction unexplored**: The paper's closing vision is that "decades of CVP heuristics can refine practical quantizers," with LLL/BKZ reduction being the most natural next step. While reasonable to leave as future work, no preliminary experiment (even on small layers) validates this promise.

- **HPTQ's Huffman decoding overhead**: Huffman coding introduces variable-length decoding that is known to be GPU-unfriendly. The paper does not discuss or measure this overhead during inference.

### Trivial:

- None beyond the above.

## Nice-to-Haves

- Empirical comparison of back-to-front vs. front-to-front GPTQ to demonstrate whether the Babai-derived bound is descriptive of the standard implementation.
- Analysis of clipping frequency per layer in standard INT4 GPTQ, and the actual vs. bound error ratio, to quantify the theory-practice gap.
- Preliminary LLL-reduced GPTQ experiments, even on small layers, to validate the lattice perspective as generative (not merely retrospective).
- Direct comparison of the Theorem 5 bound with QuIP/LDLQ's guarantee to clarify the incremental contribution.
- Expand baselines to include QuIP#, AQLM, or VPTQ at comparable bitwidths.

## Removed Points

- **Reproducibility concerns about concurrent Birnick work**: The paper cites Birnick (2025) as a concurrent arXiv preprint; per the rules, we assume this exists and is available. Removed any questioning of its existence or availability.

- **Doubts about SSQR/HPTQ code or model availability**: All cited models and code are assumed available. Removed.

- **Formatting nitpicks**: Removed.

- **Demand for wall-clock quantization time**: Removed as a reproducibility nitpick; iterative scale search overhead is a valid practical concern but belongs in nice-to-haves rather than a core weakness.

- **Rank-deficient Hessian assumption**: The paper uses damping (λ) to ensure full rank, which is a standard practice. The concern about the geometric implications of a perturbed lattice is minor and not a flaw.

- **Claims that the paper "overclaims" the phrase "first"**: While the concurrent Birnick work means the claim needs qualification, the paper already footnotes this. The novelty question is substantive and covered in Major weakness #1.

## Novel Insights

The paper's most genuinely novel insight is that the *reason* GPTQ's greedy local procedure works globally is not because of some lucky local approximation, but because it is performing a well-studied lattice algorithm (Babai's nearest plane) that comes with provable approximation guarantees. The inverse basis interpretation of the error propagation step (N = B^{−⊤} connecting to OBQ's Hessian inverse coefficients) and the geometric realization that quantization order corresponds to the order of Gram-Schmidt orthogonalization in CVP are contributions that transcend the algebraic formulation and open a two-way channel between lattice algorithm design and LLM quantization.

## Suggestions

- Explicitly compare and contrast Theorem 5's bound with QuIP/LDLQ's guarantee—show whether the Babai-derived bound is tighter, looser, or equivalent, and under what conditions.
- Add a table showing the actual-vs-bound error ratio across layers of a real LLM (even just one model) in the no-clipping regime, to demonstrate that the bound is informative in practice.
- Run GPTQ back-to-front and front-to-back on at least one model and report perplexity and layer-wise errors, to clarify whether the order matters empirically.
- Replace or amend the phrase "first" in the introduction and abstract to clarify the relationship with Birnick's concurrent proof.
- Tone down "firm theoretical footing" in the abstract and conclusions to explicitly delimit the no-clipping scope.

## Score and Decision

**Calibration**: I compared against:
- SpQR (accepted poster, scores 6/8/6/6): Similar profile—novel representation + kernels for LLM quantization, limited benchmark breadth. SpQR had stronger practical results but no theoretical depth.
- PVQ for LLMs (rejected, scores 5/1/6/8): Applied lattice ideas to LLM quantization with narrow baselines. Weaker than this paper theoretically.
- DiscQuant (rejected, scores 6/3/6/3): Theoretical quantization paper with limited practical impact. Weaker theory (heavy assumptions) and narrower impact.
- CDQuant (rejected, scores 5/5/6): Incremental improvement over GPTQ with weak experiments.
- Lattice Transform Coding (accepted spotlight, scores 6/8/6/8/8): Connected lattice theory to practical compression. Stronger paper than this one—deeper theory-practice integration and broader evaluation.

This paper sits between the rejected quantization papers (CDQuant, DiscQuant, PVQ) and the accepted ones (SpQR, LTC). The theoretical insight (GPTQ ≡ Babai) is genuine and non-trivial, but its novelty is partially preempted by QuIP's LDL analysis and concurrently by Birnick. The practical gap (no-clipping only, limited experiments, min-pivot unused) is significant. The paper is above the rejected quantization papers due to its theoretical depth, but below the accepted ones because the theory does not yet translate into clearly superior practical outcomes.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>