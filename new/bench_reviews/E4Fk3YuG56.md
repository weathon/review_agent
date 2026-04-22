Now I have thoroughly read the paper. Let me write the final consolidated review.

## Summary

The paper proposes Cut Cross-Entropy (CCE), a method that reformulates cross-entropy loss computation to avoid materializing the full O(N×|V|) logit matrix in GPU global memory. CCE decomposes the loss into an indexed matrix multiplication (requiring only O(ND) memory) and an online log-sum-exp (requiring O(N) memory), both computed in on-chip SRAM via custom kernels. Gradient filtering exploits the inherent numerical sparsity of softmax (values below bf16 precision are skipped), yielding a 3.5× backward speedup. On Gemma 2 (2B), CCE reduces loss+gradient memory from 28 GB to 1.164 MB while matching `torch.compile` speed, enabling batch size increases of 1.5–10× across 11 frontier models.

## Strengths

- **Dramatic memory reduction with no speed penalty.** Table 1 shows CCE reducing loss+gradient memory from 28,000 MB to 1,164 MB (24×) on Gemma 2 (2B), while the total loss+gradient time (145 ms) essentially matches `torch.compile` (143 ms). Figure 1 demonstrates concrete batch size increases from 1.5× to 10× across 11 models on a 16-GPU setup.

- **Clean algorithmic decomposition.** The reformulation of cross-entropy into an indexed matrix multiplication (Algorithm 1) and a linear-log-sum-exp (Algorithm 2) both computed in flash memory is natural and well-executed, directly analogous to FlashAttention's tiling strategy. The mathematical derivation from Eq. 4 is clear and the connection to the FlashAttention approach is well-motivated.

- **Gradient filtering exploits real mathematical structure.** The insight that softmax values below bf16 precision (ε = 2^{-12}) contribute nothing to the gradient is well-justified and empirically validated. Figure 3 shows fewer than 50 out of 256K vocabulary entries have probability above the precision cutoff, and the 3.5× backward speedup (Table 1 row 1 vs. 7) confirms practical effectiveness.

- **Exemplary ablation design.** Table 1 provides 10 configurations systematically isolating the impact of vocabulary sorting (row 6), gradient filtering (row 7), Kahan summation (rows 8–10), and their combinations, making each component's contribution transparent.

- **Scientific integrity in reporting pretraining failures.** Section 5.3 honestly discloses that naive CCE causes validation perplexity degradation during pretraining, diagnoses two specific failure modes (gradient filtering starving low-support tokens and bf16 summation precision loss), and introduces CCE-Kahan-FullC as a remedy — then validates it matches `torch.compile` across four models (Figure 5).

- **Comprehensive memory analysis across 11 models.** Figure 1's analysis of cross-entropy memory dominance across frontier models is a valuable contribution independent of the proposed method, clearly demonstrating the growing bottleneck.

## Weaknesses

### Fatal
None.

### Major

- **The abstract's headline "28 GB to 1 GB" claim applies only to the fine-tuning variant, not the pretraining variant that the paper's motivation most urgently targets.** The 1,164 MB figure (Table 1, row 1) is for CCE, which works for fine-tuning but causes validation perplexity degradation during pretraining (Section 5.3). The pretraining-suitable variant, CCE-Kahan-FullC, uses 2,326 MB (2.3 GB) and has 2.2× backward time (313 ms vs. 143 ms for `torch.compile`). While still a 12× memory reduction, the abstract's framing presents the most favorable numbers without this qualification. This is not a methodological flaw — the paper is transparent in the body — but creates a misleading first impression for the primary use case.

- **Pretraining convergence validation uses pretrained models as starting points on only 5% of OpenWebText for ~1500 steps.** The "pretraining" experiments start from already-initialized models (Gemma 2 2B Instruct, Qwen 2.5 7B Instruct, etc.) rather than randomly initialized classifiers. This is a continued pretraining / fine-tuning scenario, not from-scratch pretraining. A randomly initialized classifier would produce near-uniform softmax distributions, substantially reducing gradient filtering effectiveness and potentially exposing numerical issues not observed with already-sharp distributions. Since the paper's core motivation is enabling large-scale pretraining with large vocabularies, and the gradient filtering speedup (3.5×) depends on sparsity patterns that emerge only during later training, the absence of from-scratch pretraining validation leaves a meaningful evidential gap for the primary use case.

### Minor

- **Block-level sparsity analysis is missing for gradient filtering.** Gradient filtering in Algorithm 3 operates at block granularity (V_B × N_B blocks are skipped only if ALL elements fall below ε). Partially sparse blocks — containing a mix of above-ε and below-ε elements — must still be fully computed. Figure 3 shows element-level sparsity, but the actual filtering efficiency depends on how vocabulary sorting clusters these into uniformly dense/sparse blocks. A block-level sparsity analysis would clarify how well the 3.5× speedup generalizes across different models and training stages.

- **The gradient filtering "below numerical precision" claim is stronger than formally established.** Section 4.3 argues that softmax values below ε = 2^{-12} are negligible in bf16. While the footnote provides a reasonable justification (7-bit fraction plus 5 extra bits for rounding), and the empirical results support it, the claim that filtering is truly "below numerical precision" depends on the accumulation format and rounding behavior. Many individually sub-threshold elements across a batch could collectively contribute to a gradient update in fp32 accumulation. The paper partially addresses this by showing matching convergence, but a quantification of the maximum gradient error would strengthen the argument.

- **No convergence comparison with Liger Kernels.** Table 1 compares CCE and Liger on speed and memory, but Liger is the most directly competing method and its convergence behavior is not evaluated. Since Liger also modifies the computation order, verifying that Liger also converges identically to `torch.compile` would contextualize CCE's advantage as purely speed+memory rather than also convergence.

### Trivial
None.

## Nice-to-Haves

- A from-scratch pretraining experiment, even at smaller scale, testing gradient filtering effectiveness with a randomly initialized classifier, would substantially strengthen the pretraining claims.

- Final validation perplexity values with means and standard deviations (rather than visual curve comparison alone) for the training experiments in Figures 4 and 5 would make the convergence equivalence claim more rigorous.

- Analysis of gradient filtering rate (fraction of blocks skipped) over training steps would clarify when the 3.5× speedup materializes and whether there's a "cold start" phase.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic's claim that convergence evidence is "insufficient for the pretraining claim" because short runs cannot establish long-run quality.** While the pretraining runs are short, the paper tests numerical equivalence of CCE-Kahan-FullC against `torch.compile` — not convergence to a specific final model. If gradients are numerically equivalent (matching curves across 5 seeds), training longer would produce equivalent results. The short run is sufficient evidence for numerical correctness. The from-scratch initialization concern is kept as a major weakness above.

- **Harsh critic's claim that "many individually sub-threshold contributions could collectively exceed the precision threshold in a fp32 accumulator."** The paper explicitly states it accumulates in bf16 (the model's dtype), not fp32. CCE-Kahan-FullC adds Kahan summation for further precision in pretraining. The fp32 accumulation concern is addressed by the design.

- **Harsh critic's request for downstream benchmarks (MMLU, HellaSwag).** For a systems paper evaluating a loss implementation, showing matching training loss / validation perplexity is the appropriate metric. Downstream benchmarks measure model quality, not loss implementation quality, and would add noise without relevant signal.

- **Harsh critic's concern about "spin-lock contention" in the LSE reduction.** The paper states this incurs "little overhead" and the empirical results confirm competitive throughput. Analyzing lock contention is an implementation-level optimization concern that doesn't affect the paper's claims.

- **Harsh critic's nitpick about Figure 1 not specifying CCE vs CCE-Kahan-FullC.** Figure 1's caption references Table A4 for exact values (the appendix, stripped by the parser). The figure is clearly about memory/batch size benefits, where even CCE-Kahan-FullC still provides 12× reduction.

- **Strength finder's claim about "open-source implementation."** The paper references a GitHub repository, but I cannot verify its availability or contents. This is not a core strength of the paper.

## Novel Insights

The paper identifies a fundamental and growing asymmetry in LLM training: as vocabularies expand from 32K to 256K+ tokens, the cross-entropy layer's memory consumption grows from 40% to 89% of total training memory, making it the dominant bottleneck — far exceeding attention, which has already received extensive optimization (FlashAttention). The insight that gradient filtering is effectively "free" because softmax values below bf16 precision carry zero information is particularly elegant: rather than approximating the gradient, CCE skips terms that the numerical representation would discard anyway. The discovery that naive gradient filtering harms pretraining (by starving low-support tokens of gradient updates) but is safe for fine-tuning is an important practical finding that differentiates the two regimes and justifies the CCE vs. CCE-Kahan-FullC design choice.

## Suggestions

- Qualify the "1 GB" headline in the abstract (or add the CCE-Kahan-FullC's 2.3 GB alongside it) so readers immediately understand the pretraining variant's footprint.

- Run a short from-scratch pretraining experiment (even 10K steps with a small model like GPT-2) to validate that CCE-Kahan-FullC matches `torch.compile` from random initialization, particularly during the early training phase when softmax distributions are flatter.

- Report block-level skip rates for gradient filtering (fraction of V_B × N_B blocks fully skipped) alongside the element-level sparsity in Figure 3, to clarify the relationship between softmax sparsity and the achieved speedup.

## Score and Decision

**Calibration anchors:**

- FlashAttention-2 (avg score 7.25): Directly analogous "Flash-style" GPU kernel optimization for LLMs. CCE addresses a more severe bottleneck (cross-entropy at 89% memory vs attention's quadratic scaling) and achieves more dramatic savings (24× memory). FlashAttention-2 had more mature validation (end-to-end GPT training). CCE is comparable in algorithmic cleanliness and engineering execution.

- Scaling FP8 training (avg score 7.5): Memory optimization for LLM training via precision. CCE is more narrowly scoped but solves a cleaner problem with a more dramatic improvement.

- FlashSampling (avg score 2.5): Also targets softmax/cross-entropy memory but was rejected for weak experiments and poor motivation. CCE is far superior in every dimension.

- DeMo (avg score 2.6): Optimizer with convergence concerns. CCE has stronger convergence evidence and a more compelling use case.

- LUT-GEMM (avg score 7.0): Efficient kernel for quantized inference. CCE has broader impact (affects all LLM training with large vocabularies) and cleaner algorithmic contribution.

CCE is a strong, well-executed systems paper addressing a real and growing bottleneck. The core idea is clean, the results are dramatic, and the evaluation is thorough for a systems paper. The main gap is the from-scratch pretraining validation, which weakens but does not invalidate the claims — matched curves across 5 seeds with 4 models for both fine-tuning and continued pretraining provide strong evidence of numerical correctness. The abstract's framing of the pretraining variant's memory cost is the other significant issue. Relative to the anchors, this paper is in the 7.0–7.5 range: clearly above medium/borderline papers, but with real (though not fatal) evidential gaps.

MY FINAL SCORE: <pineapple>7.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>