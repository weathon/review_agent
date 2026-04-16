Now I have enough context from calibration papers. Let me write the final review.

## Summary
This paper discovers that a tiny number of individual scalar weights in LLMs (termed "super weights") are disproportionately important: pruning a single super weight in Llama-7B dramatically increases perplexity and severely degrades zero-shot accuracy. The authors propose a data-free method to identify super weights via activation spikes in a single forward pass, link them to persistent "super activations" that suppress stopword likelihood, and show that preserving these super outliers improves round-to-nearest quantization for both activations and weights.

## Strengths
- **Striking and novel core finding.** The result that pruning one scalar out of billions can catastrophically degrade quality—while pruning 7,000 other outlier weights barely matters (Table 1)—is genuinely surprising and easily communicable. This is a meaningful refinement beyond prior work that identified important outlier *channels* or *groups*.
- **Simple, data-free identification procedure.** Detecting super weights through activation spikes in `down_proj` (Figure 3) requires only a single forward pass with any prompt, contrasting favorably with methods like SmoothQuant that need calibration data. The provided directory of coordinates (Table 2) is a concrete, reusable contribution.
- **Well-designed ablation.** The "Prune SW + Restore SA" experiment (Table 1) isolating the effect of the super activation is informative, showing that restoring it recovers ~42% of lost quality—demonstrating that super weights have effects beyond just super activations.
- **Consistent structural pattern.** The finding that super weights consistently appear in early-layer `down_proj` weights across model families (Llama, Llama2, Mistral, OLMo, Phi-3) suggests a genuine architectural phenomenon rather than an artifact of one model.

## Weaknesses

### Major:

1. **Core claim is overstated relative to evidence.** The abstract claims pruning a super weight "reduces zero-shot accuracy to guessing," but Table 1 shows post-pruning accuracies of 59.90 on PIQA (vs. random ~50), 56.12 on Winogrande (vs. random ~50), and 30.68 on HellaSwag (vs. random ~25). While these are severe degradations, they are not "guessing" levels on most tasks. Similarly, "increasing perplexity by 3 orders of magnitude" overstates the actual ~100-200× increases observed (C4: 7.08→763.65 ≈ 108×; Wiki-2: 5.67→1211.11 ≈ 213×). These are approximately 2 orders of magnitude, not 3. For a paper whose core contribution rests on the extraordinary importance of one scalar, precise characterization matters.

2. **Super weight identification procedure is underspecified.** Section 3.1 describes the method as "detecting spikes" and "removing weights until magnitudes are greatly suppressed," but critical details are missing: (a) what constitutes a "spike" (threshold relative to what baseline?), (b) whether identification is stable across prompts and domains, (c) what the stopping criterion is for iteration, and (d) whether there is a unique super weight per model or whether the procedure might find different weights with different prompts. The paper states "this detection only requires a single input prompt" but does not demonstrate that *any* prompt yields the same result or characterize what properties a prompt needs. Without these details, the claim of a "data-free method" is asserted rather than validated.

3. **Quantization evaluation lacks meaningful baselines for weight quantization.** The weight quantization experiments (Figure 7, Section 5.2) compare only against vanilla round-to-nearest (RTN) at different block sizes. There are no comparisons against AWQ, GPTQ, SqueezeLLM, or SpQR—methods that are the standard baselines for weight-only 4-bit quantization. The paper notes in Section 5.2.1 that AWQ and SqueezeLLM implicitly preserve super weights, which makes the absence of direct comparison conspicuous. Without it, the practical significance of the weight quantization contribution cannot be assessed.

4. **The mechanistic story is incomplete.** The "Prune SW + SA" experiment shows that restoring the super activation recovers only 42% of the quality loss (Table 1), leaving 58% of the super weight's effect unexplained. The paper acknowledges this ("super activations only partially explain how super weights operate") but does not investigate what drives the remaining 58%. The stopword suppression analysis (Figure 5) is suggestive but not mechanically traced from the super activation channel to specific logits. This gap between the clean narrative in Figure 2 and the empirical findings weakens the conceptual contribution.

5. **"Data-free" claim for weight quantization is inconsistent.** The abstract and introduction emphasize a "data-free method," but Section 4.2 reveals that the z-score clipping threshold is tuned on 500 examples from Wikitext-2. This is a calibration dependency that contradicts the data-free framing. The activation quantization method is genuinely data-free (given known super weight coordinates), but the weight quantization method is not.

### Minor:

- **Limited evaluation of the pruning phenomenon across models.** Table 1 (the central evidence for the super weight's importance) is shown only for Llama-7B. Equivalent full tables—not just the partial Figure 5/6 plots—are not provided for Llama-13B, Llama-30B, Mistral, OLMo, or Phi-3, despite the claim that super weights "behave similarly across model families."
- **Quantization gains on Mistral-7B are marginal.** Table 4 shows only 14-25% of SmoothQuant's improvement on Mistral-7B, and the paper's hypothesis about LayerNorm weights is speculative and untested.
- **No actual latency or hardware implementation results.** The paper claims the method is "hardware-friendly" but provides no inference benchmarks or kernel implementations, leaving the practical deployment advantage unverified.

### Trivial:
- None significant.

## Nice-to-Haves
- Investigating training dynamics: analyzing when super weights emerge during pre-training and whether they can be regularized away would transform this from a descriptive observation into a deeper scientific contribution.
- Testing identification robustness across diverse prompts, tokenizers, and architectural variants.
- Adding perplexity comparisons for weight quantization against AWQ, GPTQ, and SqueezeLLM at comparable bitrates.
- Evaluating on generation-heavy benchmarks (MT-Bench, AlpacaEval) beyond short-context classification tasks.

## Removed Points
*These points are flagged to be removed, treated with caution:*
- **"No variance or robustness analyses"**: Single-run evaluation without confidence intervals is standard practice for large-scale LLM benchmarking. Demanding error bars for benchmarks like HellaSwag/PIQA goes beyond community norms.
- **"Missing related works"**: No external knowledge is available to verify whether specific related works exist that should have been cited.
- **Demanding theoretical proofs for why super weights emerge**: This is an empirical discovery paper; requiring a theoretical training dynamics analysis is scope creep. It would strengthen the paper but is not a flaw of the current submission.
- **Format/style nitpicks**: These do not affect substance.
- **Questioning existence of cited models or benchmarks**: All cited models (Llama, Mistral, OLMo, Phi-3) and benchmarks are real and available.

## Novel Insights
The paper's most striking insight—that a single scalar in an MLP down-projection channel creates a persistent, input-independent super activation that propagates through skip connections and functions as a global bias suppressing stopword likelihood—is a genuinely new way to understand how LLMs prevent degenerate output distributions. The finding that the super weight's position is invariant to instruction fine-tuning (noted in Section 3.1) suggests this is not a learned idiosyncrasy but a structural property of the transformer+GLU architecture. However, the 58% of the effect left unexplained by the super activation pathway remains an important open question.

## Suggestions
- Tone down the abstract and introduction: replace "reduces accuracy to guessing" with "severely degrades accuracy" and "3 orders of magnitude" with "2 orders of magnitude" (or report the exact multipliers).
- Add a complete pruning comparison table for at least one additional model (e.g., Mistral-7B) to support the cross-model generality claim.
- Run 4-bit weight quantization comparisons against AWQ and GPTQ at group size 128 on C4/WikiText-2 perplexity and zero-shot accuracy.
- Clearly separate the "data-free" claim: state that identification and activation quantization are data-free, while weight quantization requires calibration for the z-score threshold.

## Evaluation

**Originality**: The discovery that individual scalar weights are catastrophically important is novel and distinguishes itself from prior work on channel-level or group-level outliers. The connection to super activations and stopword suppression adds genuinely new mechanistic insight. However, the quantization application (hold out outliers + round-to-nearest) is incremental relative to methods that do the same at larger scale (SpQR, SqueezeLLM). **Score: 6/10**

**Importance of research question**: Understanding what makes specific parameters disproportionately important is a fundamental question for LLM science and directly relevant to compression. **Score: 7/10**

**Claims well supported**: The core pruning claim is backed by strong evidence for Llama-7B, but overstated ("guessing," "3 orders"). The mechanistic claim is only partially supported (42% recovery). The quantization claims are under-backed by baseline comparisons. **Score: 4/10**

**Soundness of experiments**: The experimental design for the pruning phenomenon is sound, but narrow (one model fully detailed). Quantization experiments lack strong baselines for weight quantization. **Score: 5/10**

**Clarity**: The paper is well-structured and clearly written. The figures are informative. The overclaiming in the abstract/intro detracts. **Score: 6/10**

**Value to research community**: The super weight directory and the identification method are practical contributions. The empirical phenomenon will likely inspire follow-up work. **Score: 7/10**

**Calibration**: The "House of Cards: Massive Weights in LLMs" paper (rejected, scores 3/5/8/5, avg ~5.25) is the closest comparison: similar discovery of important weights, but with novelty concerns, limited generalizability, and an incremental application method. The current paper has a stronger core finding (individual scalars vs. groups) and a cleaner identification method, but similarly has an incomplete mechanistic story and quantization contributions that don't match state-of-the-art. The "Systematic Outliers" paper (accepted poster, scores 3-8, avg ~6) had similar empirical analysis depth but more breadth. SpQR (accepted poster, scores 6/8/6/6, avg 6.5) had stronger quantization results with proper baselines. Given these anchors, this paper sits at approximately **5.5**: its empirical discovery is genuinely interesting and novel, but the overclaiming, incomplete mechanism, and weak quantization baselines hold it back from a clear accept.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>