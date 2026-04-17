Now I have a good calibration picture. Let me synthesize the final review.

## Summary

R-Sparse proposes a training-free activation sparsity method for LLM inference that combines input-side activation sparsity with offline low-rank decomposition of weight matrices. By sparsifying large-magnitude input channels (sparse path) and routing the remaining channels through a low-rank SVD approximation of the weights (low-rank path), the method achieves 50% model-level sparsity across both attention and MLP blocks without retraining or predicting active output channels, yielding up to 43% speedup on a custom kernel.

## Strengths

- **Novel and well-motivated approach to activation sparsity.** Shifting from output-side sparsity (which requires expensive channel prediction) to input-side sparsity eliminates a key practical bottleneck, and combining it with low-rank decomposition to handle the non-sparse residual is a natural and effective idea. This conceptual contribution is substantive and distinct from prior work like CATS and GRIFFIN.

- **Strong empirical improvements over training-free baselines.** At matched model-level sparsity, R-Sparse substantially outperforms CATS (by ~18% average on Llama-2-7B at 40% sparsity) and GRIFFIN (by ~18% at 50% sparsity). The method also applies to all linear layers—including attention—unlike baselines restricted to MLP blocks.

- **Demonstrated practical speedup.** The custom Triton kernel achieves 40–43% generation speed improvements on A6000 GPUs (Figure 6), and compatibility with INT4 quantization (Table 2) adds practical value for compression stacks.

- **Comprehensive evaluation across model families.** Results on three LLM families (Llama-2, Llama-3, Mistral) and ten tasks (commonsense reasoning, language modeling, summarization) provide reasonable breadth of evidence.

## Weaknesses

### Fatal
None.

### Major

- **Methodological disconnect between the motivational analysis (§3.2) and the deployed method (§3.4).** Section 3.2 introduces a multi-phase ReLU $\sigma_T$ and demonstrates that non-sparse components can be expressed as data-dependent biases, claiming "we will show later how these data-dependent biases can be converted into static biases and being pre-computed." This promissory note is never fulfilled: the actual R-Sparse method uses hard thresholding $\sigma_{t(s)}$ and SVD low-rank approximation, not the multi-phase mechanism or explicit bias construction from §3.2. The empirical 90% sparsity results in Figure 2 are under the multi-phase ReLU setting, not under the actual deployed method. While the observations in §3.2 and §3.3 are individually interesting, the narrative that they jointly motivate and justify the specific R-Sparse decomposition is not established. This weakens the paper's conceptual coherence without invalidating the empirical results.

- **Under-specified core decomposition formula.** The central formula $Y_r = (\bar{X} - \sigma_{t(s)}(X))(A_r B_r)^T$ contains an undefined symbol $\bar{X}$. In context, $\bar{X}$ likely denotes the original input $X$, making this the residual after masking: $Y = \sigma_{t(s)}(X)W^T + (X - \sigma_{t(s)}(X))W^T \approx \sigma_{t(s)}(X)W^T + (X - \sigma_{t(s)}(X))(A_r B_r)^T$. But this approximation—replacing $W^T$ with $(A_r B_r)^T$ only for the residual term—is where all the modeling decisions live, and it receives no error analysis or theoretical justification. The paper asserts it is motivated by the SVD importance heatmap (Figure 3) but does not derive or empirically validate the approximation quality of $(X - \sigma_{t(s)}(X))W^T \approx (X - \sigma_{t(s)}(X))(A_r B_r)^T$ specifically.

- **Limited evaluation scope.** Experiments are restricted to 7B/8B parameter models on short-context commonsense reasoning tasks, WikiText-2 perplexity, and XSUM summarization. There are no results on larger models (13B+, 70B), no evaluation on more demanding benchmarks (MMLU, GSM8K, code), and no analysis of behavior under longer contexts where approximation errors may accumulate. The efficiency evaluation uses only FP32, single-batch, single-GPU (A6000) with short generations (128–2048 tokens); it is unclear how speedups transfer to realistic FP16/BF16 deployments with batching, different GPUs, or longer contexts.

### Minor

- **Overstated "comparable performance" and "no performance loss" claims.** At 50% model-level sparsity, average accuracy drops by 1.8–1.7 points across models (e.g., Llama-2-7B: 65.88→64.06), with larger per-task drops on BoolQ (77.71→72.84) and ARC-C (43.43→40.78). While close, "comparable" should be qualified; "without any performance loss" (Conclusion) is not supported.

- **Evolutionary search robustness not analyzed.** The sparsification recipe is optimized on 16 C4 samples using evolutionary search. No analysis is provided on sensitivity to calibration data, transferability across domains, or how search overhead scales with model size.

- **The memory I/O cost formula** ($r\frac{m+n}{mn} + s$ relative to a full layer) is given without derivation, and overhead from index computation for sparse access and storage of per-layer $A_r$, $B_r$ matrices is not accounted for.

### Trivial
- The term $\bar{X}$ is undefined (likely $X$); this should be explicitly defined.

## Nice-to-Haves

- Compare with combined sparse+low-rank methods like OATS at matched parameter budgets (not just matched sparsity), which would better contextualize the contribution.
- Results on larger models (Llama-2-70B or Llama-3-70B) and more demanding benchmarks (MMLU, GSM8K).
- Breakdown of speedup into decoding-phase vs. prefill-phase contributions.
- Analysis of per-layer $\rho$ values from the evolutionary search to provide insight into which layers benefit more from sparse vs. low-rank components.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **Unfair comparison with ReLUfication baseline:** The harsh reviewer flags that comparing against ReLUfication without retraining disadvantages the baseline. However, R-Sparse's claim is specifically "training-free" efficiency, so comparing against a method that would require extensive retraining to achieve comparable sparsity is the appropriate comparison point for the targeted use case. This asymmetry favors R-Sparse but is methodologically sound for evaluating training-free methods. Removed as it is not a genuine unfairness.

- **Missing comparison with weight pruning (SparseGPT, Wanda):** Weight pruning and activation sparsity are fundamentally different compression paradigms—one modifies weights permanently while the other exploits dynamic runtime sparsity in activations. Direct comparison at matched sparsity ratios conflates different mechanisms. The paper compares against the appropriate activation sparsity baselines for its stated scope.

- **Batched inference concerns about varying sparsity patterns:** This is a valid concern for deployment but outside the paper's stated scope of single-batch on-device inference. The introduction explicitly targets "small-batch on-device applications." Mentioned as a nice-to-have but not a core weakness.

- **Formatting/style nitpicks:** Removed per rules.

## Novel Insights

The paper's key insight—that the contribution heatmap $\mathbf{S}_{i,j}$ of input channels and SVD components in a linear layer concentrates in a small region (allowing simultaneous sparsification of input channels and low-rank approximation of weights for the residual)—is genuinely novel and empirically validated. However, the promising multi-phase ReLU bias formulation (Observation I) deserves deeper investigation; the paper's own Figure 2 shows dramatic performance recovery going from $l=1$ to $l=2$, suggesting that even simple data-independent bias terms could recover significant capacity under high sparsity. This avenue is left largely unexplored in favor of the SVD approach, and future work might profitably combine explicit bias-correction terms with the low-rank path.

## Suggestions

- **Bridge Observation I and the deployed method.** Either integrate the multi-phase bias mechanism into the actual inference scheme, or empirically demonstrate that the bias/low-rank equivalence holds under the real thresholding + SVD setup (e.g., show that the stable rank of the residual bias matrix $\mathbf{M}$ is low when constructed from actual R-Sparse residuals, not from the multi-phase ReLU).

- **Define $\bar{X}$ and provide error analysis.** Explicitly state that $\bar{X} = X$ (or whatever it represents), and empirically measure the approximation error $\|(X - \sigma_{t(s)}(X))W^T - (X - \sigma_{t(s)}(X))(A_rB_r)^T\|$ across layers and tokens to justify the decomposition.

- **Temper claims.** Replace "without any performance loss" with "with small performance degradation" and "comparable performance" with "near-comparable performance" to match the actual experimental evidence.

- **Evaluate on at least one larger model** and include MMLU or a more challenging reasoning benchmark to strengthen generalizability claims.

## Score and Decision

Calibration anchors:
- **TEAL** (training-free activation sparsity, strong evaluation up to 70B, clear methodology): Accept (Spotlight), avg score ~7.5
- **ReLU Strikes Back** (ReLU activation sparsity, requires training, good speedups): Accept (oral), avg score ~7.3
- **OATS** (sparse+low-rank decomposition, weight pruning): Accept (Poster), avg score ~6.25
- **ASVD** (activation-aware SVD, limited evaluation, methodological issues): Reject, avg score ~6.25 (but with a 5 and clear weaknesses)
- **Q-Sparse** (activation sparsity with training, unclear methodology): Reject, avg score ~4.75

R-Sparse makes a genuine and practical contribution: combining input-side activation sparsity with low-rank decomposition is a good idea, and the empirical results are meaningfully better than prior training-free baselines. However, the paper has substantive presentation issues (undefined notation, unmotivated decomposition, motivational disconnect) and limited evaluation scope (7B/8B only, FP32 speedups only). It falls below TEAL (which has cleaner methodology and 70B results) but above the rejected papers. The core contribution is empirical rather than theoretical.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>