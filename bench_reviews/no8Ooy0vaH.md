## Summary

A³ proposes a post-training low-rank approximation framework that decomposes Transformer layers into three functional components (QK, OV, MLP) and reduces hidden dimensions within each component by minimizing component-specific functional losses (attention score error, attention output error, MLP output error). Unlike prior low-rank methods that decompose weight matrices into two smaller matrices, A³ directly reduces hidden dimensions ($d_{qk}$, $d_{vo}$, $d_{inter}$), eliminating the runtime overhead of extra GEMM kernel launches. The paper provides closed-form SVD-based solutions for QK and OV components, and a CUR-based solution for MLP. Empirical results demonstrate significant perplexity improvements over SVD-LLM across multiple LLM families (LLaMA, MPT, Phi).

## Strengths

- **Functional optimization formulation:** The paper correctly identifies that prior low-rank methods optimize layer-wise output error without considering Transformer architectural structure. By minimizing attention score error (QK) and attention output error (OV) rather than generic linear layer output error, the method aligns local optimization objectives with end-to-end model performance. Table 1 shows dramatic perplexity improvements (e.g., 4.69 vs. 7.87 on LLaMA-3.1-70B WikiText-2 at 10% compression).

- **Hardware-efficient design:** By reducing hidden dimensions directly rather than factorizing weights into separate matrices, A³ achieves inference speedups without additional kernel launches. Figure 3 and Table 11 demonstrate consistent throughput improvements over SVD-LLM across compression ratios, with speedups of 11-43% depending on configuration.

- **Architectural breadth:** The framework extends beyond vanilla MHA to support GQA (Equation 22-23) and RoPE (Equation 48), enabling application to modern architectures like LLaMA-3.1. The ablation in Figure 4 and Appendix G validates performance across MPT-7B/30B, LLaMA-2-7B/13B, LLaMA-3.1-8B/70B, and Phi-3.

- **Strong empirical results against the best available baseline:** The comparison against SVD-LLM shows consistent and substantial improvements. Table 3 shows A³ maintains reasonable perplexity even at 80% compression on MPT-30B (37.09 vs. baseline collapse), while competing methods like CLOVER exceed perplexity of 1000+.

## Weaknesses

- **Submission integrity concern:** The manuscript contains visible inline reviewer response tags (e.g., "@Reviewer gBeN", "@Reviewer n81d") embedded throughout the text, including in section headings (e.g., "E.2 @REVIEWER N81D A³ THROUGHPUT EVIDENCE AT SCALE"). This suggests the paper is a post-rebuttal revision with review artifacts left in the text, which compromises the ability to evaluate it as a standalone contribution.

- **Calibration dataset asymmetry in headline results:** The primary comparison in Table 1 uses different calibration datasets: A³ is calibrated on SlimPajama while SVD-LLM uses WikiText-2. Table 7 demonstrates that SVD-LLM calibrated on WikiText-2 overfits (lower WikiText-2 perplexity but higher C4), inflating A³'s margin. The headline LLaMA-3.1-70B result (4.69 vs. 7.87) should be validated with matched calibration sets.

- **Incomplete baseline coverage:** ESPACE (NeurIPS 2024), CALDERA (NeurIPS 2024), and SLiM are mentioned in related work as competitive post-training compression methods but are excluded from empirical comparison. Without head-to-head evaluation, the claim of "state-of-the-art" is not fully substantiated.

- **RoPE adaptation requires custom kernels:** Section 3.4 states that achieving full throughput for RoPE models requires a custom kernel to "fuse indexing and rotation together, which is out of the scope of this paper." Since RoPE is used in most evaluated models (LLaMA-2, LLaMA-3.1, Phi-3), this limitation undermines the "no runtime overhead" claim for the primary use case.

- **Softmax scaling after dimension reduction:** When reducing $d_{qk}$ to $r < d_{qk}$, the softmax scaling factor $\sqrt{d_{qk}}$ in Equation 1 should theoretically be adjusted to $\sqrt{r}$. The paper does not discuss whether this adjustment is made, which could affect attention temperature and model behavior.

- **CUR approximation lacks theoretical justification:** The MLP solution uses deterministic top-k selection by $\lambda_i = \|r_i\|^2 \cdot \|w_i\|^2$ (Equations 20-21). While this is inspired by Drineas et al. (2006), the original paper provides guarantees for randomized leverage-score sampling, not deterministic top-k selection. No approximation bound is provided for the greedy variant used.

## Nice-to-Haves

- **Ablation of per-head vs. joint OV optimization:** Section B.2.3 presents the globally optimal joint solution (Theorem 5) but notes it increases KV-cache size. The paper never quantifies the performance gap between the per-head approximation and joint optimization at compression ratios where KV-cache overhead is acceptable.

- **Condition number analysis:** The analytical solutions require inverting autocorrelation matrices. An analysis of condition numbers across layers would clarify numerical stability properties, especially for layers near redundancy.

- **Decoding throughput measurement:** The runtime analysis focuses on prefill throughput. For deployment, decoding throughput under KV-cache reduction is equally important.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **Title precision concern ("for Attention" vs "for Transformers"):** The title accurately reflects the main focus on attention components; MLP is included but secondary. This is not a substantive concern.

- **Runtime overhead contradiction claim:** The reviewer misread Table 11. The peak memory for 20% A³ with SDPA (26,037 MB) is actually *lower* than original SDPA (32,917 MB), consistent with the paper's claims. The throughput improvements are confirmed.

- **"Mechanistic interoperability" typo:** This is indeed a typo for "mechanistic interpretability" but does not affect technical content. Minor issue.

- **Demand for confidence intervals:** Standard practice in this field is single-run evaluations for large model perplexity, where differences like 4.69 vs 7.87 are well beyond typical variance. Not a substantive concern.

- **Comparison to structured pruning:** This is scope creep. The paper explicitly targets low-rank approximation methods, and structured pruning is a different compression paradigm with different trade-offs.

## Novel Insights

The decomposition of Transformer layers into QK, OV, and MLP functional components—grounded in mechanistic interpretability work (Elhage et al., 2021)—provides a principled framework for component-aware compression. The insight that minimizing attention *score* error (QK) and attention *output* error (OV) rather than generic linear layer output error yields better end-to-end performance is well-motivated and empirically validated. The approach of reducing hidden dimensions in-place rather than factorizing weights represents a genuine shift in how to think about low-rank approximation for Transformers. The observation that Table 10 shows non-compositional interaction between QK and OV compression (joint effect ≈ sum at low compression but diverges significantly at 40%) suggests the local objectives are well-aligned for practical compression ratios but accumulate error at extreme compression.

## Suggestions

1. **Remove all `@Reviewer` inline tags** before any subsequent submission. These artifacts should have been placed in a response document, not embedded in the paper text.

2. **Provide matched calibration comparisons** for all headline results. Run SVD-LLM with SlimPajama calibration to enable fair comparison on WikiText-2 perplexity metrics.

3. **Add ESPACE and CALDERA baselines** to Table 1, or clearly state the scope limitation in the baseline selection and provide reasoning for why SVD-LLM is the primary comparison target.

4. **Clarify softmax scaling handling:** Explicitly state whether the softmax division factor is kept as $\sqrt{d_{qk}}$ or adjusted to $\sqrt{r}$ after compression, and justify the design choice.

5. **Implement and benchmark the RoPE custom kernel** for at least one model, or revise the claims to acknowledge that "no runtime overhead" applies to MHA-NoPE models specifically, while RoPE models require additional implementation effort to achieve full efficiency.