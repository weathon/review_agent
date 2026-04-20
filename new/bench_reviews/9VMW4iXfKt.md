## Summary

R-Sparse proposes a training-free activation sparsity method for LLMs that decomposes linear layers into a sparse component (from large-magnitude input channels) and a low-rank component (from weight SVD), enabling sparsification of all linear layers including attention blocks. Experiments across Llama-2/3 and Mistral models show R-Sparse achieves 50% model-level sparsity with ~2-3% absolute accuracy loss, significantly outperforming MLP-only baselines, and demonstrates compatibility with INT4 quantization.

## Strengths

- **Novel input-side sparsity paradigm eliminating channel prediction error:** Unlike CATS and GRIFFIN which require predicting active output channels before computation (introducing prediction errors), R-Sparse identifies sparse channels directly from input magnitudes (Section 3.4, Eq. 3). This removes a known failure mode of prior activation sparsity methods.

- **Rank-aware decomposition motivated by empirical heatmap analysis:** The contribution heatmaps (Figures 1 and 3) showing concentration in the bottom-right corner (large input channels + large singular values) provide non-trivial, empirically-grounded justification for combining input sparsity with low-rank SVD. This is not an ad-hoc combination but a principled decomposition.

- **Substantially higher sparsity with competitive accuracy across model families:** Table 1 and Figure 5 demonstrate that R-Sparse at 50% model-level sparsity degrades ~3 accuracy points for Llama-3-8B (69.44% → 66.20%; 64.06 for Llama-2-7B (65.88 → 64.06), while CATS_40% and GRIFFIN_50% collapse to 35-46% accuracy ranges. This holds across three model families and ten tasks.

- **Orthogonal compatibility with weight quantization:** Table 2 shows R-Sparse@40% + INT4 achieves 66.41 average accuracy—only 1.69 points below full FP16 (68.10)—valuable for practitioners stacking compression techniques.

- **Training-free with minimal search overhead:** The evolutionary search for layer-wise sparsity ratios takes ~1 hour on a single A6000 GPU (Section 3.5), unlike ReLUfication requiring ~150B tokens of continual pre-training.

- **Comprehensive ablation validating complementarity:** Table 3 shows that vanilla sparsity alone (66.25) and low-rank alone (33.05) both fall short of the combined R-Sparse (67.50), confirming the decomposition components capture distinct aspects.

## Weaknesses

### Fatal

None

### Major

- **Speedup evaluation uses an artificially weak baseline, undermining the efficiency claim.** Section 4.3 explicitly states: *"Without losing generality, our implementation is based on the Hugging Face library with FP32 precision data format."* Modern LLM inference runs optimized FP16/BF16 kernels through frameworks like vLLM, TensorRT-LLM, or FlashDecoding. Comparing a custom Triton sparse kernel against the unoptimized HuggingFace FP32 baseline inflates the relative speedup significantly. The 42-43% generation speed improvement (Figure 6) likely stems largely from the precision difference and lack of dense kernel optimization rather than from the sparsity mechanism itself. The paper needs to compare against an optimized dense baseline to validate the true efficiency gain.

- **Baseline comparison design is structurally unfair for accuracy.** In Section 4.1 and Table 1, the paper compares R-Sparse (which sparsifies both attention and MLP blocks) against CATS and GRIFFIN (which sparsify only MLP blocks). To achieve equal model-level sparsity (e.g., 50%), the baselines are forced to concentrate all sparsity into the 3 MLP layers out of 7 total linear layers per block. This requires extreme intra-block sparsity (~88%+) for CATS/GRIFFIN while R-Sparse distributes the same budget across all 7 layers (~50% per layer). Concentrating sparsity into fewer components is inherently more destructive to performance. The 18+ point average gains over CATS_40% and GRIFFIN_50% are largely attributable to this distribution advantage, not solely to the "rank-aware" mechanism. A valid evaluation would include a baseline that distributes sparsity equally across all layers using a naive approach (e.g., magnitude pruning applied uniformly) to isolate whether the gain comes from distributing sparsity or from the rank-aware mechanism.

- **"Comparable performance" claim overstates accuracy retention at 50% sparsity.** The abstract claims "comparable performance at 50% model-level sparsity," but Table 1 shows a 3.24-point absolute drop for Llama-3-8B (69.44% → 66.20%) and 1.82-point drop for Llama-2-7B. On harder reasoning tasks like ARC-Challenge, the drop is ~5.8 points for Llama-3-8B (50.51% → 44.71%). While these drops are reasonable for compression, framing them as "comparable" is misleading and obscures the accuracy-efficiency trade-off.

### Minor

- **Mask generation overhead is unquantified.** Section 3.4 applies threshold-based masking ($\sigma_{t(s)}(X)$) per-token during decoding. Computing this mask requires a full pass over the dense input vector $X$, introducing memory bandwidth and compute overhead. For small batch sizes or when sparsity ratios are moderate, this overhead can negate sparsity gains. The paper should quantify the latency cost of mask computation.

- **Evolutionary search may overfit to the 16-sample C4 calibration set.** Section 3.5 optimizes the sparse-to-rank ratio $\rho$ using only 16 random samples from C4. With a population of 32 over 5 generations, there is no analysis of whether the recipe generalizes to held-out calibration data or whether different random samples produce significantly different recipes. Without variance reporting across seeds, the adaptive search may simply memorize the calibration distribution noise.

- **Theoretical motivation for low-rank decomposition on residuals is hand-wavy.** Section 3.2 establishes that non-sparse components can be approximated as biases, but Section 3.3 jumps to top-$r$ SVD of the weight matrix without rigorously connecting why the top singular values of $W$ (which capture global variance directions) specifically reconstruct the residual activation space. The method likely works because multiplying small inputs by any weights yields low absolute error, and low-rank matrices reduce FLOPs—not because of the specific SVD-residual alignment.

### Trivial

None

## Nice-to-Haves

- Visualizing the angle or projection overlap between residual activations ($X - \sigma(X)$) and the subspaces captured by the top-$r$ singular vectors would strengthen the theoretical justification for the low-rank approximation.
- An analysis of error localization (which token types or positions suffer most) would clarify if the low-rank approximation introduces systematic biases or uniform degradation.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"Bias approximation to low-rank SVD is theoretically disjoint":** The paper does explicitly connect these through the bias matrix M construction (Section 3.3, lines 80-82), showing that the bias matrix formed from residuals has stable rank ~400, motivating the SVD approach. While not a tight theoretical proof, the connection is present.

- **"Comparison against uniform ρ=0.95 is weak":** The authors do provide this comparison in Table 4. While a stronger baseline would compare same (r,s) pairs distributed uniformly, the current comparison at least demonstrates the search adds value over a single tuned uniform parameter.

- **"Missing appendix proofs and details":** These exist in the original submission but are stripped by the PDF parser.

- **Various formatting/typo nitpicks:** These are parser artifacts, not author errors.

## Novel Insights

The paper's core contribution—input-side activation sparsity combined with rank-aware low-rank decomposition—is genuinely novel in the activation sparsity literature. Prior works focus on output-side sparsity requiring channel prediction (CATS, GRIFFIN, ReLUfication), which introduces prediction errors and limits sparsity to MLP blocks. By shifting to input-side, R-Sparse eliminates the prediction failure mode and unlocks attention block sparsification. The heatmap analysis (Figures 1 and 3) showing that computation concentrates in specific input channel × singular value combinations is empirically grounded and provides a non-trivial justification for the sparse+low-rank decomposition. However, the evaluation methodology has real flaws that weaken the evidence for these claims: the speedup comparison against an unoptimized FP32 baseline inflates efficiency claims, and the accuracy comparison forces baseline methods into an unnatural operating regime by concentrating sparsity into fewer components. The method is promising but the evidence base needs strengthening.

## Suggestions

1. **Report speedup against an optimized dense baseline** (e.g., FP16 vLLM or TensorRT-LLM) to validate the true efficiency gain from sparsity alone. Even if the improvement is smaller, it strengthens the paper's credibility.

2. **Add a naive distributed baseline** that applies uniform magnitude-based sparsity across all 7 linear layers per block at the same model-level sparsity budget. This would isolate whether the improvement comes from distributing sparsity or from the rank-aware mechanism specifically.

3. **Quantify the mask computation latency** during decoding and report it as a fraction of total generation time across different batch sizes and sequence lengths.

4. **Tone down the "comparable" language** in the abstract/introduction to accurately reflect the 2-3% absolute accuracy degradation on harder tasks.

## Score and Decision

I calibrated this paper against several human-reviewed papers in similar spaces:

- **Dobi-SVD** (scores 8,5,6,6,6, accepted poster): Similar post-training SVD compression with strong experiments. R-Sparse has comparable methodological novelty but weaker evaluation rigor (no optimized speedup baseline).

- **Double Sparse Factorization** (scores 5,8,6, accepted poster): Also factorization-based LLM compression with strong experiments. R-Sparse provides broader model family evaluation but has similar fairness questions in baselines.

- **Model Folding** (scores 6,5,6,6, accepted poster): Training-free compression. R-Sparse has stronger novelty but comparable evaluation issues.

- **S2-Attention** (scores 5,3,6,3, rejected): Questioned speedup claims against FlashAttention baseline and unfair comparisons. R-Sparse shares this pattern but has stronger empirical results overall.

R-Sparse sits between the strong accepted papers (Dobi-SVD, Double Sparse) and the rejected papers with evaluation issues (S2-Attention). The empirical results are genuinely strong across multiple models and tasks, and the methodological contribution is novel. However, the speedup baseline issue and baseline comparison unfairness are significant weaknesses that prevent a higher score. I place it at the same tier as the 5.5-6 borderline papers.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>