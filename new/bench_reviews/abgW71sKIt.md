The paper text is provided in the user message. Let me carefully go through the key issues.

**On the "1-bit" claim (Critic Issue 1):** Looking at Tables 1 and 2, the paper uses "Weight Bits 1.11" and "1.06" - these are the same effective bit-widths as ARB-RC. The comparison is fair since all methods compared (PB-LLM, BiLLM, ARB-RC, ARB-X) use similar or same bit budgets. The paper does not claim to be a strictly binary-only method; it uses the same parameterization as ARB-RC (α_r, α_c, B). The framing as "1-bit PTQ" is consistent with the field's convention where methods with scaling factors but binary weight values are considered "1-bit" (as in XNOR-Net, BiLLM, ARB-LLM). This is not a misleading claim.

**On AMP (Critic Issue 2):** The AMP mechanism is indeed somewhat heuristic - it uses sign of gradients to create binary masks for selective parameter updates. This is a valid concern.

**On derivations (Critic Issue 4):** The derivations do have some notational issues and the paper promises full derivations in Appendix B. The α_c* formula does look plausible for the Output Error objective since it involves S = X̂^T X (which captures the cross-term between full-precision and quantized inputs). The concern about the derivation being opaque is legitimate but Appendix-based derivations are common in the field.

**On calibration data (Critic Issues 5 & Spark 1):** The paper's title and abstract mention "data-aware" and examining "the influence of calibration data" but doesn't actually vary calibration data. This is a real gap - the claimed contribution about calibration data analysis is not delivered.

**On the LLaMA-2-7B PTB result (3166):** Looking at the paper, the proposed method achieves 3166.0 on PTB for LLaMA-2-7B, which is worse than ARB-RC (763.19) and ARB-X (681.24). This directly contradicts the "consistently outperforms" claim. The authors dismiss it by saying "large perplexity indicates that the metric cannot provide a meaningful evaluation" but this is a significant regression.

**On selective layer-wise alignment:** The paper claims only the last FC layer per block gets output alignment. The ablation in Table 4 compares Output Error vs Activation-conditioned Error, but doesn't compare selective vs all-layer output alignment. This is a missing ablation.

Now let me calibrate against similar papers:
- ARB-LLM: Accepted poster with scores 8, 5, 8 (avg ~7)
- PB-LLM: Accepted poster with scores 8, 6, 5, 8 (avg ~6.75)
- STBLLM: Accepted poster with scores 5, 8, 5, 6 (avg ~6)
- LLM-QAT: Rejected with scores 5, 5, 5, 5

This paper has real contributions (insightful analysis, consistent improvements) but also real weaknesses (PTB regression, missing ablations, opaque AMP, unfulfilled "calibration data" claim). It's comparable to ARB-LLM and STBLLM in quality - similar incremental gains over baselines. I'd place it slightly below ARB-LLM (which had cleaner analysis and more comprehensive ablations) but similar to STBLLM.

Now let me write the final review.</think>Now I have sufficient information to calibrate and write the final review.

## Summary

This paper investigates why naive output alignment fails for 1-bit post-training quantization (PTQ) of LLMs and proposes a data-aware approach addressing three identified issues: (1) layer-wise output matching does not necessarily reduce block-level loss, (2) quantization errors accumulate across layers, and (3) indiscriminate output matching disrupts token interactions in attention. The method introduces selective block-level output alignment (applied only to the last FC layer per block), a modified optimization objective that accounts for accumulated errors using full-precision inputs, and an Attention Matrix Preservation (AMP) mechanism.

## Strengths

- **Insightful preliminary analysis (Section 3):** The identification of three failure modes of naive output alignment—block-level loss inconsistency, error accumulation, and attention degradation—is systematic and provides genuine understanding. The observation that ARB-X can reduce layer-level loss while increasing block-level loss (Figure 1) and that token similarity matrices diverge in deeper layers (Figure 2) are concrete findings that advance the field's understanding.

- **Well-motivated design:** Each proposed component directly addresses an identified problem: the Output Error objective addresses error accumulation; selective application addresses block-level mismatch; AMP addresses attention degradation. The coherency from diagnosis to solution is commendable.

- **Consistent empirical improvements where effective:** On OPT models especially, perplexity improvements are substantial (e.g., 4.85 improvement on OPT-1.3B/C4, ~3.0 on OPT-2.7B). Zero-shot QA improvements, while modest (≤0.78%), are consistent across model sizes.

## Weaknesses

### Major

- **LLaMA-2-7B PTB result contradicts the "consistently outperforms" claim:** Table 2 shows the proposed method achieves PPL of 3166.0 on PTB for LLaMA-2-7B, which is dramatically worse than ARB-RC (763.19) and ARB-X (681.24)—a 4×+ regression over the closest baseline. The paper dismisses this with "the large perplexity indicates that the metric cannot provide a meaningful evaluation," but this is inadequate: (a) similar large PPL values appear for other methods on PTB (e.g., BiLLM at 5243), so PTB is not uniquely broken; (b) a 4× regression over the most competitive baseline on a standard benchmark deserves serious investigation; (c) if PTB is truly unreliable, it should not be used as a primary evaluation metric at all. This anomaly directly undermines the "consistently outperforms" narrative.

- **Claimed contribution about "role of calibration data" is unfulfilled:** The abstract states the paper investigates "the influence of calibration data on 1-bit post-training quantization" and the contributions list "systematically examining the influence of calibration data." However, no experiment varies calibration data source, size, or composition. The analysis only measures depth-wise error accumulation and attention degradation on a single calibration set (C4). This disconnect between claimed and actual contribution is misleading; what the paper actually studies is the impact of *output alignment objectives*, not calibration data.

- **AMP mechanism is mathematically under-specified:** The AMP masks (Eq. 10) are defined as sign(gradient of L_AMP), then used as hard gates (Eq. 11). There is no derivation of these gradients (promised in Appendix B, which is not available), no justification for why sign-based binary masking is preferred over soft regularisation (e.g., adding L_AMP as a loss term), and no convergence analysis. Since AMP produces the largest gains on LLaMA models (~10 PPL improvement in Table 3), its opacity is a significant concern.

- **Missing critical ablation on selective strategy:** The central design choice—applying output alignment only to "the last fully connected layer of each block"—is justified only by the claim that it "has the most direct impact on the block loss," with no theoretical or empirical evidence. No ablation compares selective vs. all-layer output alignment under the same objective and AMP settings. This is arguably the most important missing experiment, as the gains could potentially come entirely from the modified objective and AMP rather than the selectivity criterion.

### Minor

- **Notation inconsistencies:** The paper alternates between W_c, Ŵ, and Ŵ for quantized weights, and between X_b and X̂ for quantized activations. Section 4 uses different notation than Section 3, making the derivations harder to follow than necessary.

- **Limited evaluation on newer architectures:** LLaMA-3-8B results show substantially higher perplexity than LLaMA-2-13B (C4 PPL: 35.14 vs 13.80), suggesting the method may not generalize as well to newer architectures. No results on other recent models (Mistral, Gemma, Qwen).

- **No inference overhead or calibration cost analysis in the main text:** The paper claims "minimal overhead" but defers overhead analysis to Appendix D. Since the output alignment objective involves full-precision inputs and the α_r update requires solving a least-squares system involving the Gram matrix, the practical computational cost is relevant for the PTQ community.

## Nice-to-Haves

- Comparison with 2-bit or mixed-precision methods at comparable compression ratios (e.g., GPTQ W2A16) would contextualize whether 1-bit binarization is competitive at similar memory footprints.

- Per-layer/per-block error trajectory visualizations comparing the proposed method against ARB-RC would strengthen the claim about block-level error reduction.

- Generated text samples from quantized models would help evaluate practical quality, given that even the best PPL values remain far from full-precision.

## Removed Points

- **"1-bit" claim is misleading / unfair comparison:** The paper uses "Weight Bits 1.11" and "1.06" identical to ARB-RC and comparable to BiLLM. All compared baselines use the same convention of binary weights with scaling factors. The "1-bit" framing is standard in this field (XNOR-Net, BiLLM, ARB-LLM all call their methods "1-bit"). This is not a structural unfairness issue.

- **PTB on LLaMA-2-7B shows all methods produce impractically high PPL, so the metric is meaningless:** While partially acknowledged by the paper, this doesn't excuse the 4× regression over ARB-RC. However, the reviewer's broader claim that the entire evaluation is invalid because PPL values are too high is too sweeping—perplexity improvements are meaningful on other benchmarks.

- **No evaluation on chat models:** This is outside the stated scope of the paper, which focuses on base model quantization. Standard in the 1-bit PTQ literature (BiLLM, ARB-LLM, PB-LLM, STBLLM) evaluates on base models.

- **Comparison with STB-LLM or QuIP#:** The paper cites STB-LLM but focuses on the same comparison set as ARB-LLM. STB-LLM achieves sub-1-bit average precision through different mechanisms (sparsity + binarization), making direct comparison at the same bit budget difficult. This is a nice-to-have, not a requirement.

- **Formatting issues / OCR-like errors in equations:** Some notational inconsistencies exist (e.g., NF = N − d), but these appear to be minor presentation issues, not fundamental mathematical errors.

- **Variance/robustness across seeds:** While always desirable, single-run evaluation is standard practice in the 1-bit PTQ literature (BiLLM, ARB-LLM, PB-LLM all report single runs). Requesting multiple runs here imposes a standard not applied to competing work.

## Novel Insights

The analysis distinguishing Activation-conditioned Error (∥XW_b − X_bW_c∥) from Output Error (∥XW − X_bW_c∥) is a meaningful conceptual advance. While the difference seems small—replacing the quantized input in the target with the full-precision input—this captures the accumulation of quantization error through prior layers. The empirical finding that naive layer-wise output alignment can *increase* block-level loss (despite decreasing layer-level loss) is a non-obvious and practically important observation that should inform future PTQ design. The RMSNorm-sensitivity hypothesis for LLaMA models is also an interesting architectural insight, though it deserves more rigorous testing.

## Suggestions

- **Most critical:** Run an ablation comparing selective output alignment (last FC only) vs. all-layer output alignment under identical objective and AMP settings. This is essential to validate the paper's central design choice.

- Investigate and explain the LLaMA-2-7B PTB regression (3166 vs 763 for ARB-RC). At minimum, analyze whether this is an artifact of PTB's small vocabulary or whether it indicates a failure mode of the method on short-sequence data.

- Either (a) add experiments varying calibration data (size, source, composition) to support the claimed contribution, or (b) remove/rephrase the "influence of calibration data" framing from the abstract and contributions.

- Provide the AMP gradient derivations and justify why binary sign-based gating is preferred over adding L_AMP as a differentiable regularizer.

## Score and Decision

Calibration anchors:
- ARB-LLM: Accepted poster, scores 8/5/8 (avg ~7). Clean incremental improvement over BiLLM with comprehensive ablations, similar evaluation scope.
- STBLLM: Accepted poster, scores 5/8/5/6 (avg ~6). Novel sparsity-based sub-1-bit approach, but with incremental novelty concerns and confusing ablation settings.
- PB-LLM: Accepted poster, scores 8/6/5/8 (avg ~6.75). Limited to LLaMA-7B only, partial binarization concept.
- LLM-QAT: Rejected, scores 5/5/5/5. Limited evaluation, insufficient novelty.

This paper has real contributions—the preliminary analysis is insightful and the method produces genuine improvements on OPT models. However, the PTB regression on LLaMA-2-7B, the unfulfilled "calibration data" claim, the opaque AMP mechanism, and the missing ablation on the selective strategy are significant weaknesses. It sits below ARB-LLM (which had cleaner analysis and ablations with similarly strong results) and roughly comparable to STBLLM (which had novelty concerns but real practical advances). Given the weaknesses outweigh the comparable papers' but the empirical contributions are real, I place it in the marginal range.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>