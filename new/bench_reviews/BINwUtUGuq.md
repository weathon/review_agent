## Summary

This paper introduces FISTAPruner, a layer-wise post-training pruning method for LLMs based on a convex LASSO-like optimization model solved via FISTA, augmented with an intra-layer sequential reconstruction mechanism and adaptive λ tuning. The method is evaluated extensively across OPT, LLaMA, LLaMA-2, and LLaMA-3 model families under both unstructured and 2:4 semi-structured sparsity, consistently outperforming SparseGPT, Wanda, DSnoT, and PERP on perplexity and zero-shot benchmarks. The empirical results are thorough and convincing, but the paper's framing of its contributions significantly overstates the novelty of its components.

## Strengths

- **Comprehensive and consistent empirical results**: Tables 1–5 demonstrate that FISTAPruner consistently improves perplexity over SparseGPT, Wanda, DSnoT, and even the retraining-based PERP across model sizes from 125M to 70B, both sparsity patterns, and multiple architectures. The LLaMA-3-70B results (retaining 98.6% and 95.6% of zero-shot performance at 50% sparsity) are particularly notable.
- **Well-specified optimization pipeline**: Algorithm 1 and Sections 3.1–3.5 provide a clear, reproducible description of the FISTA iterations, hard thresholding, adaptive λ bisection, and stopping criteria. The proximal gradient derivations (Eqs. 4–5) are standard but correctly applied.
- **Robustness to initialization**: Table 6 shows comparable perplexity whether initialized from dense weights or magnitude pruning, demonstrating that the optimization procedure itself—not initialization quality—drives the gains.

## Weaknesses

### Fatal
None.

### Major

- **The convex optimization does not determine the pruning mask; mask selection is post-hoc magnitude thresholding** — The paper's abstract claims "a LASSO-like convex optimization model crafted to induce sparsity." However, Section 3.3 (Eq. 7) and Section 3.4 both explicitly acknowledge that a hard thresholding step $\mathcal{H}$ is applied after FISTA convergence to enforce the exact target sparsity: *"the smallest-magnitude weights [are set] to zero until the exact sparsity level is achieved."* This means the convex FISTA solver is performing weight rescaling/adjustment while the actual mask is selected via magnitude comparison—identical in principle to Wanda or magnitude pruning. The central methodological claim that convex optimization "induces" the sparsity pattern is significantly overstated.
  
  Why it matters: If the mask is ultimately determined by magnitude thresholding rather than the LASSO solution, the claimed novelty of "convex optimization for mask determination" is misleading. The true contribution is better weight adjustment given a mask—a meaningful but narrower claim than what the paper makes.

- **"Intra-layer error correction" is standard sequential within-layer reconstruction, reframed as a novel mechanism** — Section 3.1 describes $X^* = Z_{\text{prev}}^*$ for subsequent operators, which is the standard sequential processing mode already used in SparseGPT, GPTQ, and other layer-wise reconstruction methods. The ablation in Figure 4(a) compares sequential vs. parallel (independent per operator) reconstruction, not "error correction vs. none." The paper's Related Work acknowledges SparseGPT as the prominent method but does not cite that it already uses sequential intra-layer processing.
  
  Why it matters: This overclaiming inflates the perceived novelty of the method. While the sequential approach does empirically help (as shown in the ablation), it is an established technique, not a new invention.

- **Performance gains cannot be disentangled from unequal compute/calibration budgets** — FISTAPruner runs FISTA for $K=20$ iterations up to $T=3$ times per layer with bisection tuning of λ, while SparseGPT and Wanda are single-pass analytical methods. Figure 4(b) shows FISTAPruner's perplexity drops sharply with more calibration data while baselines plateau immediately. Without equalizing optimizer steps or total compute budget, the experiments cannot determine whether gains come from the convex formulation or simply from more computational effort and repeated data exposure.
  
  Why it matters: This confounds the primary comparison that motivates the paper. The method may simply be doing more work rather than being inherently superior per unit of computation.

### Minor

- **2:4 semi-structured extension abandons the optimization framework entirely** — Section 3.3 explicitly states that the n:m sparsity constraint renders the problem non-convex, and the paper resorts to a deterministic "set the two smallest-magnitude elements to zero in each group of four" after FISTA convergence. The convex optimization plays no role in selecting the 2:4 pattern. While the authors acknowledge this limitation ("the non-convex nature of this extension introduces complexities"), the paper still presents 2:4 as a natural extension of the same framework, which it is not.

- **Theorem 1's convergence guarantee assumes smoothness violated by the actual pipeline** — Theorem 1 guarantees that bisection converges to $λ^*$ achieving the desired sparsity within tolerance $ε$. However, this requires the mapping $s(λ)$ from regularization to sparsity to be continuous. The hard thresholding step (Section 3.4, acknowledged in the paper) introduces discrete jumps in sparsity that violate this assumption. While this does not affect the practical functioning of the algorithm, the theoretical overclaim should be toned down.

### Trivial

- Section 5 acknowledges pruning takes ~12 hours for LLaMA-3-70B on a single A100, which is significantly slower than SparseGPT/Wanda. While the paper notes this is an offline process, a wall-clock comparison table would better contextualize the practical trade-off.

## Nice-to-Haves

- Report the sparsity distribution of weights after FISTA convergence but before hard thresholding to clarify how much sparsity the ℓ1 penalty actually drives vs. what the hard thresholding adds.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Critic's claim that "the claimed architectural novelty is false"** — While overclaiming is a valid concern, the empirical ablation (Figure 4a) does demonstrate a real benefit from sequential processing, so the complete dismissal is too strong. Downgraded from fatal to major with nuance.

- **Critic's claim about "mischaracterization of parallel pruning"** — The paper's parallel pruning claim (Section 3.5) is at the inter-layer level, which is consistent with the design. This is not a contradiction. Removed as reviewer misunderstanding.

- **Demands for sequential SparseGPT baseline** — This is a reasonable ablation request but not a flaw in the current paper. Moved to nice-to-have.

- **Typos/formatting-related criticisms** — Removed per hard rules.

- **Complaints about missing appendix content** — The parser strips appendices, so these are not valid concerns. Removed.

## Novel Insights

The paper's most interesting contribution is arguably not the convex optimization framing, but rather the demonstration that an iterative, optimization-based weight-adjustment procedure (FISTA with ℓ1 regularization) can consistently outperform one-shot methods like SparseGPT and Wanda when combined with sequential within-layer processing. The rounding error decomposition (Eq. 8) for tuning λ is a practical engineering insight—if rounding error dominates, increase λ. However, this is undermined by the fact that the mask itself is never actually selected by the optimizer, reducing the method's effective contribution to iterative weight refinement given a magnitude-based mask.

## Suggestions

- **Reframe the contribution**: Instead of claiming "convex optimization for mask determination," position the paper as an iterative weight-refinement procedure using convex optimization, with magnitude-based mask selection. This is honest and still meaningful.
- **Add a compute-matched baseline**: Run SparseGPT or Wanda with the same total calibration token exposure and optimizer-step budget (e.g., by running them multiple times or with more calibration data) to isolate whether the convex formulation adds value per unit of compute.
- **Clarify the role of sparseGPT's sequential processing**: Acknowledge that sequential intra-layer reconstruction is standard and position intra-layer error correction as a standard component rather than a novel invention.

## Score and Decision

**Calibration anchors:**
- **High-scoring (avg 6–8)**: DSNT (6,6,6 avg 6.0) — accepted for a training-free fine-tuning approach that complemented existing methods with clear practical benefit. FISTAPruner has stronger empirical results than DSNT but similar framing issues.
- **Medium-scoring (avg 4.5–6)**: GBLM-Pruner (5,5,3,5 avg 4.5) — rejected for overclaiming novelty (gradient-based pruning is well-established) despite strong experiments. FISTAPruner is in a similar position but with more thorough evaluation. RotPruner (6,5,5 avg 5.33) — rejected due to experimental gaps despite decent results.
- **Low-scoring (<4)**: LAMP (5,3,5,1 avg 3.5) — rejected for low novelty, unfair baselines, and incomplete experiments. FISTAPruner is clearly above this tier.

FISTAPruner sits between GBLM-Pruner (rejected for overclaiming novelty with strong experiments) and DSNT (accepted for practical benefit with solid experiments). The empirical results are genuinely strong and comprehensive across model families. However, the novelty overclaiming—framing standard sequential processing as "intra-layer error correction" and overstating the convex optimizer's role in mask determination—is a real concern that mirrors GBLM-Pruner's reception.

The paper's core method is sound and produces good results, but the claimed contributions are narrower than presented. This places it in the borderline-to-weak-accept range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>