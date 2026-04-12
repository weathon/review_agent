=== CALIBRATION EXAMPLE 6 ===

# Final Consolidated Review
## Summary
This paper proposes SUSI, a learnable semi-structured pruning method for LLMs that replaces a categorical distribution over all feasible \(N\!:\!M\) masks with differentiable subset sampling via weighted reservoir sampling and Gumbel-Top-\(K\). The main idea is technically interesting: parameterizing each \(N\!:\!M\) group with only \(M\) logits instead of \(\binom{M}{N}\) mask logits substantially reduces mask-optimization overhead, and the experiments show that this lighter parameterization can match or slightly outperform prior semi-structured pruning baselines on OPT models, especially in perplexity.

## Strengths
- **A specific and nontrivial reformulation of learnable \(N\!:\!M\) pruning.** The key contribution is not just “another pruning heuristic,” but a concrete reparameterization of mask learning from a categorical over \(\binom{M}{N}\) configurations to subset sampling with only \(M\) logits per group (Section 3.2.1). This is a real algorithmic simplification and is clearly motivated by the parameter-count blowup of categorical mask learning.
- **The method is backed by a coherent optimization pipeline.** The paper combines WRS, Gumbel-Top-\(K\), relaxation, and annealing in a logically consistent way. Theorem 1 usefully clarifies that optimizing over ordered sampled subsets induces the same expected objective as optimizing over masks formed from those subsets, so the reformulation is not merely heuristic bookkeeping.
- **Empirical results are consistently favorable on the reported setup.** On the main 2:4 experiments, SUSI achieves the best perplexity on WikiText-2 for all three OPT scales in Table 2 (50.24 vs 50.91 MaskLLM on OPT-125M; 54.14 vs 55.86 on OPT-350M; 28.05 vs 28.56 on OPT-1.3B), and slightly best average zero-shot accuracy in Table 1 across the same models.
- **The paper includes targeted analyses that are actually informative for this method.** The ablation on the power term \(p\) and annealing is useful because these are central to making the relaxation train stably; Figure 4 shows they matter substantially. The cross-seed mask-overlap analysis is also more informative than generic “we ran 3 seeds,” since it probes whether the learned sparse structure is stable.
- **The parameter-efficiency advantage becomes more compelling at larger group sizes.** The paper does not only claim a modest 2:4 win; it shows why the gap to categorical-mask methods widens sharply for patterns like 2:8 and 4:8, where \(\binom{M}{N}\) grows quickly. That makes the contribution more than a small constant-factor tweak.

## Weaknesses

### Major:
- **The empirical support for broad scalability/practicality claims is still limited.** The main body evaluates only OPT-125M/350M/1.3B, and although the appendix adds Qwen2.5-0.5B and Llama3.2-1B, the paper does not validate SUSI on genuinely larger contemporary models. For a paper framing itself as a practical solution for LLM compression and deployment, evidence capped at 1.3B in the main experiments is not fully convincing on scalability.
- **The paper claims efficiency primarily through parameter counts, not end-to-end resource measurements.** The central practical claim is that SUSI lowers computational/memory burden relative to MaskLLM, but the evidence shown is mostly trainable-parameter count (Figure 3a) rather than wall-clock training time, peak VRAM, or hardware throughput. Since the paper also emphasizes hardware-compatible deployment, the lack of actual latency/tokens-per-second measurements leaves the “practical deployment” claim only partially substantiated.
- **The reduced parameterization likely imposes an expressivity tradeoff, but this is not analyzed.** Section 3.2.1 correctly argues that SUSI reduces mask parameters from \(O(\binom{M}{N})\) to \(O(M)\), but that reduction comes from restricting the family of representable mask distributions to those induced by sequential subset sampling from per-weight logits. The paper proves equivalence of the expected objective under its subset representation (Theorem 1), but it does not discuss whether this family is strictly less expressive than a full categorical over masks, nor when that restriction may hurt. The small degradation versus MaskLLM in Table 6 at 2:8 (e.g., average 35.22 vs 35.91 on OPT-350M) is consistent with such a tradeoff and would benefit from explicit discussion.
- **Some headline superiority claims are stronger than the measured margins justify.** The reported improvements over MaskLLM are generally small, especially in zero-shot accuracy (Table 1), and there is limited repeated-run performance reporting for those final benchmark numbers. The paper does include a seed-stability analysis in Section 4.3.3, which partially addresses robustness, so this is not a fatal flaw; however, the language “consistently outperforms” and “robust and practical solution” should be tempered unless supported by broader repeated-run statistics on the main benchmark tables.

### Minor
- **The power term \(p\) is important empirically but under-motivated conceptually.** Equation 11 modifies the standard update with a power term and the ablation suggests it is crucial, yet the paper offers only a stability intuition. A clearer explanation of what behavior this induces in subset selection or gradients would strengthen the method section.
- **Generalization outside OPT is mixed and underexplained.** Appendix Table 8 is useful, but it shows noticeably larger degradation relative to dense baselines on Qwen2.5-0.5B and Llama3.2-1B than on OPT. This weakens the impression that SUSI transfers cleanly across architectures without further tuning, and the paper does not analyze why.
- **The comparisons to MaskLLM would be more convincing if compute-matched.** The same training budget is described for SUSI, but the paper does not clearly establish whether MaskLLM and SUSI are matched in actual optimization cost (FLOPs, time, or memory headroom), which matters when attributing gains to the algorithm rather than differing effective training budgets.

### Trivial
- **The ablations could isolate components more sharply.** The paper ablates annealing and the power term, and compares soft vs STE, which is helpful, but it still does not fully separate the roles of the WRS parameterization versus the particular relaxation/training recipe.

## Nice-to-Haves
- Add wall-clock training time, peak memory, and inference latency on hardware with native sparse kernels.
- Expand repeated-run reporting for the main benchmark tables, not only mask overlap/stability analysis.
- Include a short discussion or toy analysis of the expressivity gap between WRS-induced mask distributions and full categorical mask distributions.
- Analyze why the gains shrink or reverse slightly in some higher-sparsity or non-OPT settings.
- Provide convergence curves normalized by compute, not just by number of tokens.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“MaskLLM’s overhead is not a real issue because the extra parameters are only used during short calibration.”** This criticism is overstated. The paper’s claim is about training/calibration efficiency, not inference-time parameter count, and Section 4.3.1 explicitly frames the advantage as lowering optimization cost and memory during mask learning. That is a legitimate scope. One can still ask for stronger end-to-end measurements, but the motivation itself is not invalid.
- **Criticism based on inability to run MaskLLM for 4:8 due to infrastructure.** Per instruction, concerns questioning feasibility/availability of a cited method on the authors’ hardware should not be used as substantive criticism. The valid version of the point is already captured above: the paper should provide stronger resource measurements rather than relying on parameter-count arguments alone.
- **Pure complaints about outdated models or demands for unrelated additional workflows (e.g., instruction tuning, DPO recovery, quantization integration).** These are outside the paper’s stated scope. Testing larger or more modern models is a reasonable request and retained above, but demanding additional adaptation pipelines is scope creep.
- **Reproducibility concerns about missing Monte Carlo sample count / locked-seed script / minor implementation details.** The paper includes code, hyperparameters, and a reproducibility statement; these details would be helpful but are not core review weaknesses under the stated rules.
- **Formatting/parser issues in equations/tables.** These are extraction artifacts and not paper flaws.

## Novel Insights
The most interesting synthesis across the reviews is that SUSI’s main contribution is best understood not simply as “more efficient mask learning,” but as an explicit **efficiency–expressivity tradeoff** for learnable semi-structured pruning. The paper makes a compelling case that a full categorical over \(N\!:\!M\) masks is unnecessarily expensive for practical learning, and the results suggest that a much smaller WRS-based family is often sufficient—especially at 2:4 sparsity. At the same time, the slightly weaker results in some more aggressive sparsity settings hint that the categorical parameterization may occasionally buy useful extra flexibility. Framing the paper in these terms would make the contribution sharper and more intellectually honest than presenting SUSI as uniformly superior.

## Suggestions
- Reframe the core claim as: SUSI offers a better efficiency/performance tradeoff than categorical-mask learning, rather than implying unconditional superiority.
- Add direct measurements of training wall-clock time, peak VRAM, and sparse inference throughput on supported hardware.
- Report repeated-run benchmark means/std for the main Table 1 and Table 2 results, especially against MaskLLM.
- Add a short theoretical or empirical discussion of the expressivity limitation of WRS-induced mask distributions relative to full categorical mask learning.
- Expand the architecture-scaling section with at least one stronger modern model and discuss why Appendix Table 8 shows larger dense-to-sparse degradation outside OPT.
- Improve the explanation of the power term \(p\): what it changes in practice, why \(p=3\) helps, and whether its optimal value is architecture- or sparsity-dependent.



# Actual Human Scores
Individual reviewer scores: [2.0, 2.0, 2.0, 2.0]
Average score: 2.0
Binary outcome: Reject
