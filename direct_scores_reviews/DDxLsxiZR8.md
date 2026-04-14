## Summary

CAT Pruning proposes an acceleration strategy for DiT-based text-to-image diffusion models (Stable Diffusion 3, Pixart-Σ) that combines three components: (1) relative noise magnitude as a per-token importance criterion, (2) EWMA-based staleness tracking to enforce distributional balance across timesteps, and (3) KMeans clustering with positional encoding to enforce spatial coherence in token selection. Unselected tokens reuse cached hidden states from the prior step. The method claims 50–60% MACs reduction and up to 2.15× throughput improvement while maintaining CLIP scores comparable to the baseline.

---

## Strengths

- **Per-component visual ablation is genuinely informative.** Figures 4, 5, and 6 sequentially and clearly isolate the failure modes of (a) noise-only selection (background artifacts, teddy bear case), (b) missing staleness (noise inconsistency), and (c) missing clustering (absent windows, blurry details). Very few acceleration papers provide this level of incremental visualization; it makes the necessity of each module concrete rather than asserted.

- **EWMA staleness framing is the right abstraction.** Framing unbounded token neglect as a "staleness" problem analogous to async-SGD is an intellectually clean insight. The connection to exploration-exploitation trade-offs correctly identifies that pure greedy selection of high-noise tokens collapses the distribution, and the EWMA fix (Equations 1–2) is parsimonious — one hyperparameter, no retraining.

- **Spatial clustering with normalized positional encoding is simple and effective.** Concatenating (i/h, j/w) coordinates to the noise magnitude vector before KMeans is a minimalist way to enforce spatial contiguity without a trainable component. The cluster visualizations (Figure 7) show semantically meaningful groupings aligned to image content, validating the approach on modern 1024×1024 DiT architectures.

- **The method is retraining-free and architecture-agnostic within the DiT family.** Achieving 1.82×–2.15× speedup without any fine-tuning on SD3 and Pixart-Σ at 1024×1024 resolution is practically significant for production deployment, distinguishing CAT Pruning from distillation or pruning-with-finetuning approaches.

---

## Weaknesses

- **CLIP score is the sole quality metric, and this is a critical evaluation gap.** CLIP measures text-image alignment but does not capture image fidelity, texture consistency, or distributional quality. A method that blurs background details or drops structural elements (as seen in Figure 9 at α=0.2: missing eye, reduced windows) could maintain or even improve CLIP score while producing visibly degraded outputs. For an ICLR submission claiming "performance preservation" under 50–60% compute reduction, FID or equivalent distributional metrics are standard and necessary. The absence of FID means the core preservation claim is unsupported.

- **The AT-EDM baseline appears to be a broken re-implementation, undermining quality comparisons.** The paper explicitly modifies AT-EDM by replacing its similarity-based copy with a cache-and-reuse mechanism, because the original design "is not suitable" at 30% token budget. The resulting CLIP scores are catastrophic: 14.66 (Pixart-Σ COCO 28-step), 11.00 (Pixart-Σ COCO 50-step), and 17.08 (Pixart-Σ PartiPrompts 50-step), versus a baseline of ~31. The variability between datasets for the same model (24.30 vs. 14.66 on Pixart-28) further suggests implementation defects rather than genuine method failure. When the primary competitor's numbers are this anomalous, superiority claims over AT-EDM in quality metrics cannot be trusted, and speedup advantages should be reported with appropriate caveats.

- **Graph Pooling is described but not defined.** Section 3.4 introduces "1 light-weighted Graph Pooling Layer, which is not trainable" and Algorithm 2 calls `pool(clusters, ...)`, but neither the graph structure (how are edges defined between cluster tokens?), the pooling operation (mean, max, or attention-weighted?), nor the computational cost are specified anywhere in the paper. This is a reproducibility-critical gap for a component that directly determines cluster score assignment and thus which tokens are selected.

- **Algorithm 1 leaves the attention scope ambiguous in a computationally significant way.** Line 1 computes Q, K, V from `Update(T_s)` (only selected tokens), and line 2 applies full attention over these Q, K, V. It is unclear whether K and V come exclusively from selected tokens (changing the attention distribution relative to baseline) or from all tokens (limiting savings to the MLP only). This distinction changes both the correctness interpretation of the approximation and the MACs reduction accounting for the attention module.

- **Key hyperparameters are neither ablated nor specified.** The EWMA decay parameter `a` appears in Equations 1–2 but is never stated, justified, or ablated. The number of clusters `n=20` is fixed without sensitivity analysis. The pruning start step `t_0=8` and the sparsity `α=0.3` are chosen empirically with no study of robustness. Given that these parameters collectively define the method's behavior, their omission prevents reproducibility and raises concerns about cherry-picking.

- **The ablation study (Figure 6) is purely qualitative.** The paper's ablation — comparing {noise only}, {noise+balance}, {cluster+noise+balance}, and {consecutive rows} — is presented only as qualitative image pairs. No CLIP scores or MACs numbers are reported per ablation condition, making it impossible to quantify the individual contribution of clustering versus staleness versus noise magnitude. This is a standard expectation for a method paper.

- **Classifier-free guidance interaction is unaddressed.** The paper uses CFG with two forward passes (conditional + unconditional). It is not discussed whether the same token mask is applied to both passes, or how the token selection criterion (based on conditional noise) interacts with the unconditional pass. Different masks could cause artifacts; the same mask may not be appropriate for the unconditional pass. This is a non-trivial implementation detail for all tested models.

- **No limitations section.** The paper does not acknowledge failure modes such as high-frequency detail tasks (text rendering, fine facial geometry), behavior at very high sparsity, or the restriction to DiT architectures.

---

## Nice-to-Haves

- Quantitative ablation table reporting CLIP (and ideally FID) for each method component to complement Figure 6's qualitative visualization.
- Sensitivity plots for `t_0`, `α`, and EWMA decay `a` — even a simple 2D grid would help establish robustness.
- Explicit latency breakdown separating KMeans clustering overhead, EWMA tracking, topk operations, and denoising step cost, to confirm that selection overhead does not significantly offset MACs savings.
- Comparison with token merging methods (e.g., ToMe) to situate pruning vs. merging tradeoffs in this context.
- Evaluation on a model with a publicly available FID evaluation pipeline (e.g., SDXL on MS-COCO) to enable standard benchmarking.
- Token mask evolution visualization across timesteps to confirm that cluster-aware selection maintains spatial coherence in practice.
- Discussion of interaction with block-level caching methods (DeepCache, FORA) and any combined experiments, since the paper claims compatibility.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Throughput discrepancy between datasets (SD3-28: 1.82× vs. 1.87×):** The harsh critic raises concern that identical MACs should produce identical speedup. However, throughput can legitimately vary between benchmark runs due to memory access patterns, batch effects, and GPU state; this difference (<3%) is well within normal variance and is not a substantive concern.

- **Grammar errors and awkward phrasing (e.g., "demonstrate reveal"):** Pure stylistic critique with no bearing on technical content. Removed per formatting/style nitpick rule.

- **Staleness exploration absent at step t_0+1:** The harsh critic notes that staleness-based exploration (Algorithm 2 line 17) is absent at the first pruning step. This is trivially correct — there is no staleness history before the first step — and is an expected, reasonable design choice rather than an inconsistency.

- **Claim that "little attention has been given to intra-kernel optimization" is overstated:** This is a framing/positioning critique about related work coverage, not a technical flaw in the method.

- **R² = 0.67–0.79 leaving 21–33% unexplained variance:** The harsh critic uses this to argue the selection criterion is unreliable. However, the method does not claim perfect prediction — it claims a probabilistic improvement in selection quality, which Proposition 1 correctly states as "increases the likelihood." A Pearson r of 0.82–0.89 is a strong positive correlation sufficient to justify using prior-step noise as a proxy. The imperfect fit does not invalidate the approach; it motivates the complementary staleness mechanism.

- **Proposition 1 proof is in a missing appendix:** The appendix is noted as removed in the submitted text. The proposition itself is a mild theoretical support for an empirically observed correlation; its absence is a transparency gap but not a reason to reject the empirical results shown in Figure 3.

- **No comparison with DeepCache, FORA, TGATE:** The paper explicitly scopes CAT Pruning as complementary to (not competing with) block-level caching methods. Demanding a direct comparison is partial scope creep. That said, a combined experiment would strengthen the paper significantly (retained as nice-to-have).

- **Requesting human preference studies:** Not standard for algorithmic acceleration papers at ICLR; removed as non-standard practice requirement.

- **Demanding custom sparse CUDA kernels:** The paper achieves real wall-clock speedups measured on hardware. The critic's demand for custom CUDA kernels goes beyond the paper's scope and the evaluation standard in this community.

---

## Novel Insights

The most genuinely novel observation in the sub-reviews is the connection between spatial coherence enforcement and cluster-aware selection: the paper's sequential-row baseline (Figure 6, column 3) produces surprisingly strong results at 70% token pruning, suggesting that spatial contiguity is the dominant factor in quality preservation, and that cluster awareness improves over sequential selection primarily by adapting to image-content-dependent spatial structure rather than imposing uniform spatial grids. This has implications beyond this paper — it suggests that many token pruning methods may implicitly under-exploit spatial locality, and that lightweight content-adaptive spatial grouping may generalize as a broadly useful inductive bias for diffusion acceleration.

---

## Suggestions

1. **Add FID evaluation** (e.g., on COCO 30K or an equivalent split) as the primary quality metric. CLIP scores should be reported alongside FID, not as a replacement.
2. **Fix or replace the AT-EDM baseline** — either use AT-EDM on its native SD-XL environment, or implement a cleaner adaptation and verify that the resulting images are visually coherent before reporting CLIP scores.
3. **Define graph pooling mathematically** — at minimum, specify the edge construction rule and the aggregation function used in `pool(clusters, ...)` in Algorithm 2.
4. **Report the EWMA decay value `a`** used in all experiments, and provide a sensitivity ablation over at least 3 values.
5. **Provide a quantitative ablation table** alongside Figure 6, reporting CLIP and ideally FID per method variant.
6. **Clarify the attention scope in Algorithm 1** — explicitly state whether K and V include all tokens or only T_s, and update the MACs accounting accordingly.
7. **Add a brief CFG discussion** — one paragraph explaining whether the same or different masks are used for conditional and unconditional passes and why.

---

**Novelty:** Moderate. The three individual components (noise-magnitude pruning, staleness tracking, spatial clustering) each have precedents; the novelty lies in their specific combination and adaptation to DiT-based diffusion models. The spatial-clustering-with-positional-encoding is the most distinctive element.

**Technical Soundness:** Weak-to-moderate. Algorithm 1's attention scope is ambiguous. Graph pooling is undefined. The theoretical foundation (Proposition 1) is relegated to a missing appendix. The method appears to work empirically, but the mechanical description leaves critical implementation choices unspecified.

**Empirical Support:** Weak. CLIP-only evaluation is insufficient for quality claims. The sole competitor (AT-EDM) appears broken in the reported experiments. No quantitative ablation is provided.

**Significance:** Moderate. A retraining-free 2× speedup on production-scale SD3 and Pixart-Σ is practically valuable. But the significance is hard to assess without FID results and a credible competitive comparison.

**Clarity:** Below expectation for ICLR. The method overview and ablation figures are effective, but graph pooling, attention scope, and hyperparameter choices are under-specified to the point of hindering reproducibility.

MY FINAL SCORE: <pineapple>4.3</pineapple>