Now I have enough context to write the final review. Let me synthesize, carefully verifying each claimed weakness against the paper.

Key verification tasks:
1. Does the paper actually say α=0 gives worse results? YES - Section 2.3 explicitly states this.
2. Is the theory for the full FW, not the constrained version? YES - Lemma 1 and Section 4 address the vanilla FW on C_k, not the constrained version with frozen weights.
3. Does the paper's Table 1 show mixed/inconsistent results? YES - at 50% sparsity, several entries show SparseFW worsening perplexity.
4. Is the theoretical bound vacuous? Let me verify - it includes λ_max(Q) · (k + √(d_in · d_out · k)), which for LLM scales would indeed be enormous.
5. Is the SparseGPT comparison omission justified? The paper says they don't compare to methods with reconstruction steps. This is debatable but the paper explicitly scopes this.
6. Is there a structural mismatch between the story and method? YES, this is the most fundamental issue.

Let me write the review now.</think>## Summary

The paper proposes SparseFW, a layer-wise LLM pruning method that relaxes the combinatorial mask selection problem into a convex program (continuous masks in \[0,1\] with an L1 budget) and solves it using the Frank-Wolfe (FW) algorithm. The key insight is that existing methods like Wanda and RIA are greedy approximations to this quadratic mask selection objective; by solving the convex relaxation, weight interactions are accounted for jointly. The method achieves strong reductions in per-layer pruning error and improves zero-shot accuracy and perplexity over Wanda/RIA across several modern LLM architectures, particularly at higher sparsity.

## Strengths

- **Clean conceptual unification of greedy methods.** The paper provides an insightful reformulation showing that SparseGPT, Wanda, and RIA all solve single-weight greedy approximations of the same quadratic mask selection objective (Eq. MASK SELECTION). This reframing is a genuine conceptual contribution that clarifies the landscape of layer-wise pruning methods.

- **Strong per-layer pruning error reductions.** Figure 2 shows reductions of up to 80% in the local pruning objective relative to Wanda/RIA warmstarts. The analysis of continuous vs. thresholded iterate behavior (Figure 4) is informative about the optimization dynamics.

- **Principled and efficient algorithmic design.** The FW-based LMO for the constraint set C_k has an efficient closed-form solution (top-k of negative gradient entries), and the precomputation of G=XX^T and H=WG decouples per-iteration cost from sequence length and sample count. The method also naturally supports semi-structured sparsity patterns.

- **Theoretical approximation guarantee.** Lemma 1 provides a data-dependent bound decomposing error into optimization error (converging as O(1/T)) and thresholding error—a principled advantage over greedy heuristics that lack any such guarantee. Even if the bound is loose, the decomposition itself is conceptually useful.

- **Consistent accuracy improvements at high sparsity.** Table 1 shows genuine improvements in zero-shot accuracy at 60% sparsity and 2:4 semi-structured sparsity across multiple architectures (e.g., Qwen2.5 from 63.72→65.34 with SparseFW(RIA), LLaMA-3 from 58.78→60.17).

## Weaknesses

### Major:

- **Structural mismatch between narrative and the method that actually works.** The paper's central framing is that convex relaxation + FW solves the mask selection problem better than greedy heuristics by "fully accounting for weight interactions." However, the version that achieves the reported gains is not "pure" FW on the relaxed problem—it freezes α=0.9 (90%) of the highest-saliency weights identified by Wanda/RIA and optimizes only over the remaining 10%. The paper itself states (§2.3): "without fixing part of the mask, it tends to prune weights crucial for overall performance" and "α = 0.0 (full FW without any fixed weights) consistently yields worse results than the baselines." This means the practical algorithm is effectively "Wanda/RIA warmstart + FW fine-tuning of the lowest-saliency tail"—a far more modest contribution than the paper's introduction and abstract suggest. The paper's title "Don't Be Greedy, Just Relax!" and abstract claim of "outperforming strong baselines" obscures that the winning variant is deeply greedy-dependent. This does not invalidate the contribution, but it substantially reshapes what the contribution actually is: an effective refinement procedure for existing saliency-based masks rather than a replacement for greedy methods.

- **Theory does not cover the empirically effective algorithm.** Lemma 1 and the convergence guarantees apply to vanilla FW on the full constraint set C_k over all entries. The method actually used in experiments restricts the feasible set by freezing 90% of coordinates according to Wanda saliency—a strictly smaller feasible set for which no approximation guarantee is provided. The theoretical contribution, while correct for the unrestricted problem, is disconnected from the algorithm deployed in practice. This gap is acknowledged in the conclusion ("the local–global objective mismatch persists") but is not reflected in the framing of the theoretical contribution as a "key benefit of SparseFW over greedy heuristics" or in the "strong theoretical justification" claimed in the contributions.

- **The theoretical bound is effectively vacuous at LLM scale.** The thresholding error term in Lemma 1 scales as λ_max(Q) · (k + √(2 · d_in · d_out · k)). For modern LLMs where d_in, d_out ∈ {4096, 11008} and k is proportional to d_in·d_out, this term is astronomically large. The paper provides no discussion of whether this bound can be tightened or what its practical implications are. Presenting this as "strong theoretical justification" without acknowledging its vacuity at relevant scales is misleading.

### Minor:

- **Inconsistent perplexity improvements at moderate sparsity.** At 50% sparsity, several entries in Table 1 show SparseFW worsening perplexity relative to the baseline: e.g., LLaMA-3 8B worsens from 10.09 (Wanda) to 10.21 (SparseFW(Wanda)), DeepSeek-7B from 7.79 (Wanda) to 7.89 (SparseFW(Wanda)), and Qwen2.5 improves only marginally. The paper's claim of "consistent gains in final WikiText perplexity" is overstated—gains are consistent at 60% and 2:4 sparsity but not at 50%. The text softens this to "generally performs on par with or better," which is more accurate but still papers over notable regressions.

- **No comparison with SparseGPT, the de facto strong baseline.** The paper excludes SparseGPT from comparison because it involves a weight reconstruction step, making it a formally different problem. While this distinction is defensible in principle—the paper solves mask selection, not joint mask+reconstruction—SparseGPT is the most widely used and competitive LLM pruning method in practice. Practitioners will want to know how SparseFW + (no weight update) compares to SparseGPT (which does update weights). Even an apples-to-oranges comparison would be informative, as the Spark reviewer also notes. This is a gap in empirical coverage that limits practical relevance.

- **No wall-clock time or compute cost comparison.** The authors acknowledge SparseFW is "clearly more compute-intensive" but provide no quantitative comparison. SparseFW requires 2000 FW iterations per layer (Figure 3), which is substantially more than Wanda's single-pass calculation. Given that the practical value of one-shot pruning lies in efficiency, this omission limits assessment of the cost-benefit tradeoff.

### Trivial:

- **No standard deviations reported.** Table 1 omits standard deviations "for legibility," but several perplexity differences at 50% sparsity are small enough that statistical significance is unclear. This is a minor concern for a methods paper in this space, where single-run reporting is common.

## Nice-to-Haves

- A deeper analysis of *why* reducing per-layer reconstruction error does not reliably translate to better perplexity (the acknowledged local-global mismatch)—e.g., correlating per-layer error reductions with per-layer downstream impact or analyzing whether certain layers or weight types are responsible for the discrepancy.

- Experiments combining SparseFW with a weight reconstruction step (à la SparseGPT), which would naturally bridge the comparison gap and could yield further gains.

- Wall-clock time comparisons per model, and experiments at 70%+ sparsity where one-shot pruning methods degrade most severely.

- Exploration of layer-wise or adaptive α rather than a fixed α=0.9 across all layers.

## Removed Points

- **Inference speedup for unstructured sparsity.** The human finder raises this concern from other pruning papers, but this paper explicitly evaluates 2:4 semi-structured sparsity (which has hardware support) and frames unstructured sparsity results clearly. Complaining about lack of inference speedup for unstructured sparsity is scope creep for a mask selection paper.

- **Missing related works / concurrent methods.** Citing specific missing works is disallowed per the review rules (no external sources to confirm existence).

- **Formatting/style nitpicks.** Removed per rules.

- **Demanding confidence intervals for large-scale benchmarks.** Single-run evaluation is the norm in this area; requesting it is a nice-to-have at most.

- **Fairness of comparison claims favoring baselines (SparseGPT).** The paper explicitly explains why SparseGPT is not directly comparable (it solves a different, strictly larger problem). This is a reasonable methodological choice. However, not including SparseGPT results at all—even as a reference point—still limits practical relevance and is kept as a minor weakness above.

- **Claim that SparseFW also "changes weights" like SparseGPT.** This misunderstands the method. SparseFW produces a binary mask via thresholding; it does not modify the remaining weights. The continuous mask values during optimization are intermediate—they affect the optimization trajectory but the final output is a binary mask applied to the original weights. This is fundamentally different from SparseGPT's weight reconstruction. However, the distinction between mask selection and mask+reconstruction is a matter of problem scope, not a reason to exclude the comparison entirely.

## Novel Insights

The most striking empirical finding is that the convex relaxation (pure FW) *fails* to produce good perplexity despite dramatically improving the local objective—a vivid demonstration of the local-global objective mismatch in LLM pruning. The fact that 90% of weights must be frozen using Wanda saliency before FW can improve end-to-end performance reveals that most of the pruning decision is determined by simple per-weight saliency, and the interaction effects captured by the convex relaxation only matter for a small fraction of low-saliency "tail" weights. This reframes the contribution from "principled convex relaxation solves mask selection" to "FW is an effective refinement tool for the margin of a greedy mask." The paper would be stronger if it embraced this framing openly.

## Suggestions

- **Reframe the contribution honestly:** Center the paper on the hybrid method (Wanda/RIA warmstart + FW refinement of 10%), not on pure FW solving the convex relaxation. The title and abstract should reflect that FW refinement is the key practical contribution, not replacement of greedy methods.

- **Extend theory to the constrained variant:** Provide at least a qualitative argument or modified bound for the case where a subset of coordinates is frozen. This would bridge the gap between theory and practice.

- **Add SparseGPT results as a reference** in a table, even if not apples-to-apples, so practitioners can position SparseFW relative to the strongest available baseline.

- **Report wall-clock times** for pruning each model.

## Score and Decision

**Calibration.** I compared this paper against: Wanda (scores 5-8, accepted poster), RIA (scores 6-8, accepted poster), DSNT (scores 6, accepted poster), GBLM-Pruner (scores 3-5, rejected), and OWL (scores 3-8, rejected). Wanda and RIA are the most directly comparable: they are simple, effective pruning methods accepted with scores around 6. GBLM-Pruner, which proposed an incremental improvement over Wanda/SparseGPT with modest empirical gains, was rejected with scores 3-5. The present paper has a genuine conceptual contribution (unifying greedy methods as approximations to the convex relaxation) and strong local objective improvements, but its practical gains over baselines are inconsistent at moderate sparsity, depend critically on a heuristic (α=0.9) that undermines the stated narrative, and the theoretical contribution is disconnected from the deployed algorithm. This places the paper below Wanda/RIA (which are simpler, standalone methods with clear narrative-practice alignment) and above GBLM-Pruner (which had weaker framing and less principled backing). The paper has real value but needs reframing and empirical strengthening to reach a clear accept.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>