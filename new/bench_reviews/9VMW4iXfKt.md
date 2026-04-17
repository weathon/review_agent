## Summary

R-Sparse proposes a training-free activation sparsity method for LLM inference that decomposes each linear layer's computation into a sparse component (input channels with large magnitudes) and a low-rank component (offline SVD of weights for residual channels). An evolutionary search determines per-layer sparse-to-rank ratios. Experiments on Llama-2/3 and Mistral-7B/8B across ten tasks show ~50% model-level sparsity with modest accuracy loss and up to 43% end-to-end speedup with custom Triton kernels.

## Strengths

1. **Principled combination of input sparsity and low-rank approximation.** The observation (Figure 1/3) that contributions concentrate in the bottom-right of the channel×SVD importance matrix is genuinely insightful and directly motivates the sparse-plus-low-rank decomposition—removing the top-left quadrant rather than just rows or columns. This goes beyond prior activation sparsity methods that only exploit output sparsity patterns.

2. **Training-free and broadly applicable.** Unlike ReLUfication approaches that require hundreds of billions of tokens for continual pre-training, R-Sparse requires only an offline SVD and a lightweight evolutionary search (~1 hour on a single A6000). The method applies to both attention and MLP blocks, unlike CATS and GRIFFIN which target only MLP layers.

3. **Consistent empirical improvements over training-free baselines.** Table 1 demonstrates substantial gains: at 50% model-level sparsity on Llama-2-7B, R-Sparse achieves 64.06% average accuracy vs. 45.91% (GRIFFIN) and 46.26% (CATS). The gap is consistent across all three model families.

4. **Real measured speedups.** Figure 6 shows up to 43% generation speed improvement with custom Triton kernels, going beyond theoretical FLOP counts. Table 2 confirms compatibility with INT4 quantization (GPTQ).

5. **Useful ablation structure.** Table 3 shows R-Sparse outperforms sparse-only or low-rank-only baselines, and Table 4 shows adaptive per-layer recipes improve over uniform settings (up to 1.95% average improvement).

## Weaknesses

### Major:

1. **Overclaiming of "no performance loss" at 50% sparsity.** The conclusion states "high levels of sparsity can be achieved … without any performance loss," and the abstract claims "comparable performance." The data shows clear degradation at 50% model-level sparsity: Llama-2-7B average drops 1.82 points (65.88→64.06), BoolQ drops 4.87 points (77.71→72.84), ARC-C drops 2.65 points, and Llama-3-8B average drops 3.24 points (69.44→66.20). The word "comparable" can be defended for some tasks, but "without any performance loss" directly contradicts the reported numbers. This matters because the core contribution is the efficiency-accuracy tradeoff, and overstating it misleads about the real cost.

2. **Comparison with baselines at matched model-level sparsity is not truly apples-to-apples.** R-Sparse sparsifies all linear layers (attention + MLP) while CATS and GRIFFIN only sparsify MLP blocks. At the same model-level sparsity percentage, the baselines are forced into extreme MLP pruning (since attention remains dense), which disproportionately hurts them. The paper does not report FLOP-equivalent or I/O-equivalent comparisons, making it unclear whether R-Sparse's advantage comes from better methodology or from distributing the sparsity budget across more layers. A "R-Sparse on MLP-only" ablation or FLOP-matched comparison would address this.

3. **The motivational story (Sections 3.2–3.3) is loosely connected to the actual method.** Section 3.2 introduces multi-phase ReLU and shows non-sparse components can be approximated as a few bias terms; Section 3.3 analyzes the "score matrix" S showing concentration patterns. However, the final R-Sparse algorithm (Section 3.4) uses simple magnitude thresholding on input X and standard top-r SVD decomposition of weights—neither of which directly uses the multi-phase ReLU discretization or the S-based importance scoring. The bias-to-low-rank argument is suggestive but not validated experimentally (no ablation comparing S-based rank selection vs. top-σ selection, nor comparison between the multi-phase ReLU approximation and the actual low-rank path). The "rank-aware" label is thus aspirational rather than validated.

4. **Storage and memory accounting for low-rank components is incomplete.** The paper reports memory I/O reduction but does not transparently account for the total storage overhead of storing both the sparse weight columns and the low-rank factors A_r and B_r. Section 3.4 gives a relative I/O formula but does not state whether the "50% model-level sparsity" figure includes or excludes the low-rank parameter overhead. For deployment on edge devices (the stated motivation), total parameter memory matters as much as I/O, making this a relevant gap.

### Minor:

- **Efficiency evaluation uses FP32 on HuggingFace**, which is not representative of typical deployment (FP16/BF16, vLLM/TensorRT-LLM). The 43% speedup may not transfer directly to production settings where dense inference is already much faster.
- **No evaluation on models larger than 8B**, which is where memory-bounded inference matters most and where the method's value proposition is strongest.
- **No comparison with concurrent Liu et al. (2024a)**, which is described as "a special case" of this framework but without empirical evidence.
- **The evolutionary search uses only 16 C4 samples**. The sensitivity of the searched recipe to calibration data selection, size, and domain shift is not analyzed.
- **Batched inference concerns**: The method relies on per-token, per-layer dynamic sparsity patterns. Under batched inference, different tokens activate different channels, reducing effective sparsity; this limitation is not discussed.

### Trivial:

- Notation choice: the hat in $\bar{X} - \sigma_{t(s)}(X)$ for the residual path is introduced without clear definition (it appears to be the original X but should be explicitly stated).

## Nice-to-Haves

- Comparison with ReLUfication *with* retraining to position R-Sparse on the full accuracy-cost frontier, even if R-Sparse's advantage is being training-free.
- Evaluation on harder benchmarks (GSM8K, MATH, HumanEval) where sparsity may interact more adversely with reasoning.
- Per-layer breakdown of searched ρ values to validate the "patterns vary across layers" claim.
- Runtime breakdown showing time in sparse path vs. low-rank path vs. threshold computation.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"ReLUfication without retraining is a strawman baseline"**: The paper explicitly scopes itself as training-free, so comparing against without-retraining ReLUfication is the appropriate comparison for that scope. Including trained baselines would be informative but is not required for the stated contribution.

- **"No comparison with weight pruning methods (SparseGPT, Wanda)"**: These are weight pruning methods with different goals (permanent parameter removal vs. dynamic activation sparsity). Their inclusion would be nice but is not within the paper's scope.

- **"Missing related work references"**: Removed per instructions—no external sources available to verify existence.

- **"Error accumulation across layers is not bounded"**: This is a generic concern for any compression method and is empirically addressed by the end-to-end evaluation. The paper shows the method works across all layers simultaneously; theoretical bounds would be a nice addition but are not standard in this area.

- **"The multi-phase ReLU is dropped in the actual method—this is confusing"**: While the motivation could be clearer, Section 3.2 explicitly motivates the qualitative observation (non-sparse components have simple structure) which is operationalized via the low-rank path. The connection exists, albeit loosely.

- **"FP32 speedups are not representative"**: While true, this is a common practice in the field and the paper does demonstrate compatibility with INT4 quantization separately. The FP32 evaluation is a proof-of-concept, not a deployment-level claim.

## Novel Insights

The decomposition of the channel×SVD importance matrix into a sparse (large-magnitude channels) plus low-rank (dominant singular values) structure, where the critical region is the bottom-right corner (combining both), is a useful observation that goes beyond naive activation sparsity or low-rank compression alone. This framing unifies two compression paradigms in a way that is synergistic rather than additive—the sparse part and the low-rank part cover different regions of the importance heatmap, and their combination yields a better approximation than either alone (as confirmed by Table 3). However, the method's current operationalization (simple magnitude thresholding + top-r SVD) does not fully exploit this insight, and more targeted selection mechanisms (e.g., using the S matrix to choose which singular values to retain) could yield further gains.

## Suggestions

1. **Reframe the core claim** from "without any performance loss" to "with modest average degradation (~2 points at 50% sparsity)" and report per-task deltas in a summary table. This is the single most impactful change.

2. **Add an MLP-only ablation for R-Sparse** at the same FLOP/IO budget as CATS/GRIFFIN to fairly separate the contribution of "sparsifying attention too" from "better methodology."

3. **Report total parameter storage** (sparse weights + low-rank factors) alongside the I/O-based sparsity metric, especially for the edge-device deployment motivation.

4. **Run at least one experiment on a 13B–70B model** to validate scalability claims.

## Score and Decision

**Calibration**: TEAL (similar training-free activation sparsity paper) received scores of 8/8/8/6 and was accepted as Spotlight. OATS (sparse+low-rank decomposition for LLM compression, training-free) received 6/8/3/8 and was accepted as Poster. ASVD (training-free SVD-based LLM compression) received 6/8/6/5 and was rejected. Q-Sparse (activation sparsity for LLMs) received 6/5/3/5 and was rejected.

R-Sparse is comparable in methodology quality to OATS (similar sparse+low-rank idea applied to LLMs) but has weaker evaluation scope (smaller models only, FP32-only speedups) and more significant overclaiming issues. It is noticeably weaker than TEAL, which had simpler methodology, broader evaluation including 70B models, and more careful claims. The overclaiming and comparison fairness issues are more severe than what appeared in OATS, placing R-Sparse below it but above papers like ASVD or Q-Sparse that had more fundamental issues.

**Score: 5.5** — The method is sound and the sparse+low-rank decomposition idea is a genuine contribution, but overclaiming ("without any performance loss"), imperfect baseline comparisons, loosely connected motivation, and limited evaluation scope prevent a higher score.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>