Now I have sufficient calibration. Let me write the final review.

**Calibration Summary:**
- Papers with strong training-free compression at 50% sparsity (DSF, DSNT) scored 5-8, typically accepted as posters
- Papers with overclaiming/scope mismatch issues (UU9Icwbhin, cSd8Eom8Zt, XXpH3D0TVP) received major weakness flags and lower scores
- Papers with unfair baseline comparisons (9aZ2ixiYGd) had split scores (3,3,6,8) but still accepted due to novelty
- The ReLU sparsity paper (osoWxY8q2E) with similar efficiency claims scored 6,8,8 and was accepted as oral

The R-Sparse paper has:
- **Strengths**: Genuine novelty in extending to attention blocks, comprehensive experiments across 3 model families and 10 tasks, training-free at 50% sparsity with near-lossless performance, reasonable search overhead
- **Major issues**: Attribution problem (headline gains driven by scope mismatch, not rank-aware mechanism), theoretical disconnect between motivation and implementation, FP32-only efficiency evaluation
- **Not fatal**: The method still works and provides value; the issues are about claim precision rather than fundamental invalidity

Relative to anchors:
- DSF (5,8,6): Similar training-free 50% sparsity, but DSF's weakness was about hardware realization claims, not attribution
- DSNT (6,6,6): Training-free fine-tuning, clearer contribution attribution
- ReLU sparsity oral (6,8,8): Had efficiency claims but fewer attribution issues

R-Sparse's attribution problem is more serious than DSF's clarity issues but less severe than papers that were rejected for scope mismatch. The method genuinely works (Table 1 shows strong results), but the claim about what drives the improvement needs refinement.

**Estimated Score: 5.5-6.5 range** - Borderline accept, similar to DSF and DSNT, but with more significant weaknesses about claim attribution that prevent it from reaching the 7-8 range.

## Summary
This paper proposes R-Sparse, a training-free activation sparsity method that decomposes linear layers into sparse (large-magnitude input channels) and low-rank (SVD-based residual) components, applicable to both attention and MLP blocks. The method achieves 50% model-level sparsity with near-lossless performance across three LLM families and ten tasks, with demonstrated end-to-end speedups using customized kernels.

## Strengths
- **Extension to attention blocks enables higher model-level sparsity**: Unlike CATS and GRIFFIN which target only MLP blocks (limiting model-level sparsity to ~22-33%), R-Sparse applies to all seven linear layers per transformer block, enabling the reported 50% model-level sparsity (Section 3.4, Table 1). This is a genuine technical contribution that expands the achievable compression envelope.
- **Strong empirical performance at 50% sparsity without retraining**: Table 1 shows R-Sparse@50% maintains performance within ~2 points of full models across Llama-2/3 and Mistral families (e.g., Llama-2-7B: 64.06 vs. 65.88 full; Mistral-7B: 68.39 vs. 69.89 full), substantially outperforming baselines at matched model-level sparsity while requiring no fine-tuning.
- **The S×SVD contribution visualization provides actionable insight**: Figure 3's heatmap showing concentration in the bottom-right corner (high input channels × high singular values) offers a clear, empirically-grounded justification for the sparse-plus-low-rank decomposition strategy, distinguishing this from prior input-sparsity-only work.
- **Evolutionary search yields measurable gains with reasonable overhead**: Table 4 demonstrates adaptive per-layer ρ* search improves over uniform recipes by 0.89-1.95 points across tasks at various sparsity levels, with search completing in ~1 hour on a single A6000 for 7B models (Section 3.5).
- **Compatibility with weight quantization**: Table 2 shows R-Sparse@40% combined with INT4 achieves 66.41 average accuracy, only ~1 point below INT4-only (67.32), demonstrating the two compression axes can stack effectively.

## Weaknesses

### Fatal
None

### Major
- **Headline improvement claims conflate scope expansion with rank-aware mechanism**: The paper attributes the "18.74% improvement over CATS and 18.15% over GRIFFIN" (Section 4.2) primarily to the rank-aware framework, but this gain is largely driven by comparing R-Sparse (which sparsifies attention + MLP) against baselines forced to operate at model-level sparsity levels far beyond their design regime. At CATS's native 22% model-level sparsity, CATS achieves 64.32% average (Llama-2-7B) vs. R-Sparse@50% at 64.06%—essentially equal performance at different total sparsity budgets. The paper explicitly acknowledges factor ❶ ("while CATS and GRIFFIN only sparsify the MLP block, R-Sparse can be applied to both the attention and MLP blocks") as a source of gain, but does not disentangle how much comes from expanded scope vs. the rank-aware decomposition itself. Table 3 shows R-Sparse improves over Sparse-only by just 1.25 points (66.25 → 67.50), suggesting the rank-aware component contributes modestly relative to the scope expansion. This attribution problem undermines the central claim that rank-aware sparsity is the primary driver of improvement.

- **Theoretical motivation does not connect to the implemented method**: Section 3.2 derives a "bias" interpretation where non-sparse channels contribute $B_j = \sum_{k \in \mathcal{U}_j} \mathbf{W}_{down}^T[:, k]$—a sum of weight columns selected by input magnitude bins. However, Section 3.4 implements the residual as $Y_r = (\bar{X} - \sigma_{t(s)}(X))(A_r B_r)^T$, using actual residual activation values multiplied by a global low-rank SVD approximation. These are fundamentally different operations: the former discretizes inputs into bins and sums selected columns, while the latter uses continuous residual values with precomputed SVD factors. The paper bridges this with an empirical observation about stable rank ≈ 400 (Section 3.3), but this does not constitute a derivation showing the low-rank approximation captures the "bias" structure identified earlier. The theoretical arc (multi-phase ReLU → biases → low-rank) reads as post-hoc rationalization rather than principled derivation, weakening the claim that this is a theoretically-grounded framework rather than an engineered heuristic.

### Minor
- **Efficiency evaluation uses non-standard FP32 precision without FP16/BF16 validation**: Section 4.3 states "our implementation is based on the Hugging Face library with FP32 precision data format," and reports 42% speedup over dense. However, modern LLM deployments universally use FP16 or BF16, where memory bandwidth is more efficiently utilized and the relative benefit of weight sparsity diminishes. The fixed overhead of index computation and kernel dispatch becomes proportionally larger in lower precision, potentially reducing the practical speedup. Without FP16/BF16 results, the efficiency claims lack direct relevance to actual deployment scenarios.

- **Missing ablation isolating attention vs. MLP sparsification contribution**: The paper claims gains from three factors (attention coverage, rank-aware sparsity, adaptive ρ search), but only the adaptive search has a direct ablation (Table 4). There is no experiment comparing R-Sparse applied to MLP-only vs. MLP+attention at matched total sparsity, which would directly quantify how much of the gain at 40-50% model-level sparsity comes from sparsifying attention blocks versus the rank-aware mechanism. This limits understanding of what drives the improvement.

- **No comparison with concurrent input-channel sparsity work (Liu et al., 2024a)**: Section 2.2 acknowledges Liu et al. (2024a) "shares similar intuition but focuses solely on input channels," making it a directly competing approach for establishing what the rank-aware (low-rank) component uniquely contributes. An experimental comparison would clarify whether adding the low-rank term on top of input channel sparsity provides meaningful gains over input-sparsity-only methods.

### Trivial
- **Abstract cites "43% end-to-end efficiency improvements" but Figure 6 shows 36-42%**: The 43% figure in the abstract does not appear in Figure 6, which reports improvements of 38-42% for Llama-2-7B and 36-40% for Llama-3-8B. The discrepancy (possibly from a specific unreported configuration) should be clarified or the abstract claim adjusted.

- **Section 3.5 uses only 16 C4 samples for evolutionary search evaluation**: The search optimizes per-layer ρ* values using perplexity on 16 randomly selected C4 samples. No sensitivity analysis shows whether the found recipes are stable across different calibration sets or transfer robustly to domains beyond C4, given the search objective is C4 perplexity but evaluation is on downstream reasoning tasks.

## Nice-to-Haves
- Report efficiency results in FP16 or BF16 precision to align with standard deployment practices and provide practitioners with realistic speedup expectations.
- Add an ablation comparing R-Sparse applied to MLP-only vs. full model at matched total sparsity to isolate the contribution of attention block sparsification.
- Include a comparison with Liu et al. (2024a) on shared tasks to establish the unique value of the rank-aware low-rank component over input-channel sparsity alone.
- Show the distribution of found ρ* values across layers (e.g., histogram by layer type) to validate that the adaptive search discovers meaningful variation rather than converging to uniform values.
- Evaluate calibration stability by testing whether ρ* recipes found on different 16-sample subsets of C4 (or different corpora) yield consistent downstream performance.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **"Unfair baseline comparison" framed as making results invalid**: The harsh critic argued the comparison is "structurally unfair" and "forced to operate far outside their design regime." However, comparing methods at matched **model-level** sparsity is a valid practical comparison for deployment scenarios where total memory/compute budget is fixed. CATS and GRIFFIN could theoretically be extended to attention blocks but were not designed for this; the paper's comparison shows what happens when you push MLP-only methods to extreme per-layer sparsity vs. distributing sparsity across more modules. This is a legitimate comparison point, though the **attribution** of gains (scope vs. mechanism) is problematic. The weakness is retained but reframed as an attribution/claim issue rather than an invalid comparison.

- **"ReLUfication without retraining is a strawman baseline"**: The paper explicitly acknowledges in Section 2.2 that ReLUfication requires extensive continual training, and Table 1 shows ReLUfication collapsing to ~32-35% without retraining. This is presented as a known failure mode, not a competitive baseline. Including it demonstrates the training-free constraint is meaningful, not that ReLUfication-without-retraining is viable. This is not a strawman but a sanity check; removed.

- **"Low-Rank baseline collapse is an artifact of unreasonable operating point"**: Table 3 shows Low-Rank-only achieving 33.05% at 50% model-level sparsity. The critic argues this corresponds to "very low rank" and is unreasonable. However, 50% model-level sparsity applied to pure low-rank compression does require aggressive rank reduction. This is a valid operating point for the ablation—it shows low-rank alone cannot achieve this sparsity while R-Sparse's combination can. The ablation serves its purpose; the weakness is about interpretation, not the experiment本身的 validity. Softened to minor.

- **Nitpick about "not a typical edge device" (A6000)**: While the paper motivates edge deployment, testing on A6000 is reasonable for initial validation. The actual weakness is the FP32 precision, not the GPU choice itself. Removed the device criticism, retained the precision concern.

## Novel Insights
The paper's most valuable contribution is the empirical observation that extending activation sparsity to attention blocks—previously considered difficult due to different activation properties than MLPs—is feasible and enables substantially higher model-level sparsity without retraining. The S×SVD contribution heatmap (Figure 3) provides a clean visualization distinguishing this work: prior input-sparsity methods would select only the rightmost columns, low-rank methods would select only the bottom rows, but R-Sparse targets the bottom-right corner where both high-magnitude input channels and high singular values concentrate. This framing offers a principled way to think about combining sparse and low-rank compression rather than treating them as separate axes. However, the theoretical motivation (Section 3.2's "bias" derivation) does not cleanly connect to the implementation, suggesting the method was discovered empirically and rationalized post-hoc rather than derived from first principles.

## Suggestions
1. **Reframe claims to accurately attribute sources of improvement**: The abstract and Section 4.2 should clarify that the headline gains over CATS/GRIFFIN at matched model-level sparsity come primarily from extending sparsification to attention blocks (factor ❶), with the rank-aware decomposition (factor ❷) and adaptive search (factor ❸) providing additional but more modest gains (~1-2 points as shown in Tables 3 and 4). This is a paper about scope expansion enabled by input-side sparsity, not primarily about rank-aware mechanisms.

2. **Add an ablation comparing full-model vs. MLP-only R-Sparse**: At matched total sparsity (e.g., 40%), compare R-Sparse applied to all layers vs. only MLP blocks. This directly quantifies the contribution of attention sparsification and disentangles factor ❶ from factors ❷ and ❸.

3. **Report FP16/BF16 efficiency results**: Re-run the Figure 6 speedup experiment in FP16 precision to provide practitioners with realistic deployment expectations. If speedups diminish significantly, report both and discuss the precision trade-off honestly.

4. **Compare with Liu et al. (2024a)**: Run the concurrent input-channel sparsity method on the same tasks/models to establish what the low-rank component uniquely adds. If Liu et al. achieves similar results without the low-rank term, this would significantly weaken the rank-aware contribution claim.

5. **Clarify or remove the "bias" theoretical framing**: Either revise Section 3.2-3.4 to show a coherent derivation connecting the multi-phase ReLU observation to the SVD implementation, or reframe the motivation as purely empirical observations that inspired the method, avoiding the appearance of a theoretical derivation that doesn't hold.

## Score and Decision

**Calibration Reasoning:**
Compared against several anchor papers:
- **DSF** (DwiwOcK1B7, scores 5,8,6, accepted poster): Training-free 50% sparsity with strong results, weaknesses about hardware realization claims and missing ablations. Similar empirical strength to R-Sparse.
- **DSNT** (1ndDmZdT4g, scores 6,6,6, accepted poster): Training-free fine-tuning for sparse LLMs at 70% sparsity, clearer contribution attribution.
- **ReLU Sparsity Oral** (osoWxY8q2E, scores 6,8,8): Training-free activation sparsity with efficiency claims, fewer attribution issues than R-Sparse.
- **Overclaiming papers** (UU9Icwbhin, cSd8Eom8Zt): Rejected or low-scored due to scope mismatch between claims and experiments.

R-Sparse's core issues are:
1. Attribution problem: Headline "18% improvement" claims conflate scope expansion (attention + MLP) with mechanism quality (rank-aware decomposition). This is a significant claim-over-evidence issue similar to overclaiming anchors but less severe since the method genuinely works.
2. Theoretical disconnect: Motivation doesn't derive the method, reading as post-hoc rationalization.
3. FP32-only efficiency: Limits practical relevance but doesn't invalidate the accuracy results.

Relative to DSF (5,8,6) which had similar training-free 50% sparsity success but weaker hardware claims, R-Sparse has a more serious attribution problem but stronger empirical coverage (3 model families, 10 tasks vs. DSF's narrower evaluation). Relative to the ReLU sparsity oral (6,8,8), R-Sparse has murkier claim attribution.

The method is genuinely useful and the experiments are comprehensive, but the claim precision issues prevent this from reaching the 7+ range. This is a borderline paper where the core contribution (attention block sparsification at 50% model-level, training-free) is solid, but the framing overstates what the rank-aware mechanism contributes.

**Final Score: 5.5** — Marginally above rejection threshold. The paper provides real value (training-free 50% sparsity across attention + MLP is useful), but the claim attribution problems and theoretical disconnect are significant enough that acceptance would require substantial revision to reframe contributions accurately. This scores slightly below DSF (which had clearer claims despite similar empirical strength) due to the attribution issue being more central to R-Sparse's narrative.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>