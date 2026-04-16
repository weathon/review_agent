Now I have sufficient calibration information. Let me write the final consolidated review.

## Summary

R-Sparse proposes a training-free activation sparsity method for LLM inference that decomposes each linear layer into a sparse component (based on input activation magnitude thresholding) and a low-rank component (based on offline SVD of weights). Unlike prior output-sparsity methods that require predicting active channels, R-Sparse leverages input-side sparsity, enabling application to all linear layers (attention and MLP) across Llama-2/3 and Mistral while achieving 50% model-level sparsity. The method includes an evolutionary search for per-layer sparsity-rank ratios and demonstrates up to 43% wall-clock speedup with custom Triton kernels.

## Strengths

- **Novel and practical perspective on activation sparsity**: Shifting from output activation sparsity (which requires channel prediction) to input activation sparsity combined with rank-aware decomposition is a clean and well-motivated idea that eliminates a key bottleneck of prior methods (CATS, GRIFFIN). This allows sparsification of all linear layers—not just MLP blocks—enabling higher model-level sparsity.

- **Strong empirical performance against training-free baselines**: At 50% model-level sparsity, R-Sparse averages 64.06% on Llama-2-7B commonsense reasoning vs. GRIFFIN's 45.91% and CATS's 46.26% (Table 1). The improvement is substantial and consistent across models and tasks, and the method also shows promising results on language modeling (WikiText-2) and summarization (XSUM).

- **Training-free and model-agnostic**: No retraining or fine-tuning is needed, and the method is validated on three model families (Llama-2, Llama-3, Mistral). This is a significant practical advantage over ReLUfication approaches that require 150B token retraining.

- **Real efficiency gains**: Custom Triton kernels achieve 40-43% end-to-end speedups, going beyond theoretical FLOP reductions. Compatibility with INT4 quantization (Table 2) demonstrates further practical gains.

- **Meaningful ablations**: The comparison of sparse vs. low-rank vs. combined (Table 3) and uniform vs. adaptive recipes (Table 4) provide useful insight into where the gains come from.

## Weaknesses

### Major:

- **The "no performance loss" claim is overstated**: The abstract says "comparable performance at 50% model-level sparsity" and the conclusion claims "without any performance loss," but Table 1 shows measurable degradations at 50% sparsity—e.g., Llama-2-7B drops from 65.88 to 64.06 (average), with BoolQ dropping from 77.71 to 72.84 and HellaSwag from 57.13 to 54.26. The claim should be tempered to "minor degradation" or "within X points." This is not a fatal flaw but an overclaim that weakens the paper's credibility.

- **Disconnect between motivation (Case I) and actual method**: Section 3.2 introduces a multi-phase ReLU that converts non-sparse components to data-dependent biases (Motivation Case I), and Section 3.3 observes a low-rank structure in these biases. However, the final R-Sparse method (Section 3.4) does not use the multi-phase ReLU or any data-dependent bias approximation at all—it uses hard input thresholding plus offline weight SVD. The key insight from Case I (that 90% sparsity is recoverable with just a few bias terms) is not operationalized. Case I and its experiments (Figure 2) effectively serve as motivation only, but the paper presents them as if they directly support the deployed algorithm. This creates a narrative gap: the strongest-looking empirical evidence (Figure 2's near-perfect recovery at 90% sparsity with l=2) is not what the method actually uses.

- **Limited evaluation scope for the strength of claims**: The evaluation covers commonsense reasoning (8 short benchmarks), WikiText-2 perplexity, and XSUM. There are no instruction-following, coding, math, or long-form generation evaluations. Given the claim of "comparable performance" at 50% model-level sparsity (applied to both attention and MLP), this is a thin slice of model behavior to justify broad claims, especially since low-rank approximation of attention layers could have outsized effects on tasks requiring precise attention patterns.

- **Baseline fairness concerns for CATS and GRIFFIN**: R-Sparse sparsifies all linear layers (attention + MLP), while CATS and GRIFFIN were designed only for MLP blocks. The paper scales up their MLP sparsity to match model-level budgets, potentially pushing them outside their intended operating regime. The paper acknowledges this ("different from CATS and GRIFFIN, which focus only on the MLP blocks") but does not provide an ablation showing R-Sparse applied only to MLP blocks—a comparison that would isolate the method's algorithmic advantage from the scope advantage.

### Minor:

- **Underspecified implementation details**: The threshold t(s) is defined as a percentile of |X|, but it is unclear whether this is computed per-token, per-layer, per-head, or from calibration data. The notation X̄ in Y_r = (X̄ − σ_{t(s)}(X))(A_r B_r)^T is never clearly defined. The memory I/O formula r(m+n)/(mn) + s is stated without derivation. These gaps affect reproducibility and the ability to assess memory/compute tradeoffs.

- **Dynamic thresholding overhead**: Computing the s-th percentile of input magnitudes for every token and every linear layer at inference time has non-trivial cost. No analysis or measurement of this overhead is provided.

- **Speedup evaluation limitations**: The 43% speedup uses FP32 on a single A6000 with custom Triton kernels. Modern LLM inference uses FP16/BF16 with highly optimized implementations (FlashAttention, cuBLAS). The comparison to a HuggingFace dense baseline rather than an optimized inference framework makes the speedup numbers hard to contextualize. Also, Figure 6 shows speedups decrease with longer generation lengths, presumably due to KV cache overhead, but this limitation is not discussed.

- **Calibration sensitivity and search cost**: The evolutionary search uses 16 C4 samples and takes ~1 hour per model, but no analysis of robustness to calibration data or search hyperparameters is provided. It is unclear whether 16 samples are sufficient across diverse tasks.

- **Sorted visualization may overstate structure**: Figures 1 and 3 sort rows and columns independently to visualize the "top-left removable" structure. Showing an unsorted heatmap would validate that this structure exists in the actual data flow.

- **Only 7B/8B models tested**: Whether the rank-aware structure observations generalize to larger models (13B+), or whether the sparsity recipes transfer, is unaddressed despite the paper framing the work for "advanced LLMs" broadly.

## Nice-to-Haves

- Comparison with weight-side pruning methods (e.g., SparseGPT, Wanda) at similar model-level sparsity to contextualize R-Sparse's advantage relative to the full compression landscape.
- An ablation where R-Sparse is applied only to MLP blocks, isolating the algorithmic contribution of the rank-aware decomposition from the scope advantage of also sparsifying attention layers.
- Evaluation on at least one generative benchmark (e.g., MT-Bench, HumanEval) to validate quality preservation beyond multiple-choice tasks.
- Per-layer breakdown of the searched ρ values and contribution to total sparsity/speedup, which would illuminate which layers most benefit from the low-rank component.
- Testing on a 13B+ model to assess scalability.

## Removed Points

- **"ReLUfication without retraining is a strawman" / "should compare with ReLUfication + retraining"**: The paper explicitly frames its contribution as *training-free* sparsity. Comparing against ReLUfication without retraining is the appropriate training-free baseline. Including a trained baseline would be a nice-to-have but is not required for the paper's stated scope. This critique overreaches.

- **"Missing comparison with weight pruning methods (SparseGPT, Wanda)"**: These methods target a fundamentally different compression axis (weight sparsity vs. activation sparsity). They are not direct baselines. Their inclusion would be informative but is not a necessity.

- **"The combination of sparsity + low-rank is not novel"**: While sparse+low-rank decomposition is well-established in signal processing, the specific application to *input activation sparsity + weight SVD decomposition in LLM inference* is the paper's contribution. The human finder's comparison to OATS/SLiM is noted—those are *weight* pruning methods. The novelty claim is about the domain (activation sparsity) and the specific design (no prediction needed, input-side thresholding + rank-aware complement).

- **"No variance/uncertainty estimates"**: Single-run evaluation is standard practice in the LLM efficiency literature (TEAL, CATS, GRIFFIN all do the same). While confidence intervals would strengthen the paper, their absence is not a substantive flaw by community standards.

- **"FP32 evaluation instead of FP16/BF16"**: The paper states their implementation is based on HuggingFace with FP32. While a more realistic precision setting would be better, this is an implementation choice, not a fundamental limitation.

## Novel Insights

The key insight—that LLM linear layers have a joint sparse+low-rank structure in the *input activation × weight singular value* space, enabling the removal of the "top-left" corner of the contribution matrix—provides a clean geometric interpretation for why combining input activation sparsity with low-rank weight approximation is more effective than either alone. The observation that different layers have different optimal sparse-vs-lowrank ratios (attention projections like o_proj rely more on singular value components, while q/k projections are more compressible) is a useful architectural insight that could guide future work on adaptive compression strategies.

## Suggestions

- **Temper claims**: Replace "comparable performance" and "no performance loss" with precise numbers (e.g., "within 1-3 points on commonsense benchmarks at 50% sparsity").
- **Add an MLP-only ablation**: Show R-Sparse applied only to MLP blocks at 33% model-level sparsity to isolate the algorithmic contribution from the scope advantage.
- **Clarify X̄ notation and threshold computation**: Explicitly state X̄ = X (the original dense input) and specify whether t(s) is computed per-token or from calibration data.
- **Provide a latency/memory breakdown**: Decompose the end-to-end speedup into contributions from the sparse path, low-rank path, and thresholding/overhead, and report on FP16/BF16 if possible.

## Score and Decision

**Calibration**: I compared against several related papers:
- **TEAL** (Accept/Spotlight, scores 6-8): Training-free activation sparsity, 40-50% model-wide sparsity, wall-clock speedups. Very similar approach but simpler (magnitude thresholding only, no low-rank component). TEAL evaluated on more model sizes including 70B, had stronger baselines, and was well-polished.
- **OATS** (Accept/Poster, scores 3-8): Sparse+low-rank decomposition for weight pruning. Related technique but different domain (weight-side). The novelty concern around sparse+low-rank was raised but the paper still accepted.
- **ReLU Strikes Back** (Accept/Oral, scores 6-8): Required retraining, but addressed a broader question and evaluated on more tasks including HELM.
- **Q-Sparse** (Reject, scores 3-6): Required training, no wall-clock speedups, weak evaluation.
- **Targeted Low-rank Refinement** (Reject, scores 3-5): Sparse+low-rank for weight pruning, limited novelty, rejected on novelty grounds.
- **SLiM** (Reject, scores 3-5): Quantized sparse+low-rank, weak evaluation, rejected.

R-Sparse sits above Q-Sparse, SLiM, and Targeted Low-rank Refinement (it has real wall-clock speedups, solid baselines, and a novel activation-domain application). It sits below TEAL (which has broader evaluation, more model sizes, and is more complete). The main issues are: (1) overclaimed results, (2) disconnection between motivation Case I and the actual method, (3) limited evaluation scope, and (4) MLP-only ablation missing for fair baseline comparison. These are significant but not fatal—the core contribution of input-side activation sparsity + rank-aware decomposition genuinely works and outperforms baselines by large margins.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>