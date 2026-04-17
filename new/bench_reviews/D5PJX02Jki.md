Now I have good calibration information. Let me compile the final review. The relevant comparison papers are:

1. **Scaling Laws of RoPE Extrapolation** - scores 8, 6, 10 (average ~8), accepted poster. Strong paper with good theory, tested at 7B/13B scale.
2. **Round and Round We Go (RoPE analysis)** - scores 5, 5, 8, 5, 8 (average ~6.2), accepted poster. Analysis paper with novel insight about RoPE but questionable causal claims.
3. **Wavelet PE** - scores 5, 6, 5, 5 (average ~5.25), accepted poster. Novel PE method but limited scale.
4. **FIRE** - scores 6, 6, 8 (average ~6.7), accepted poster. Novel PE method with reasonable experiments.
5. **Periodic Extension** - scores 1, 3, 3, 3 (average ~2.5), rejected/withdrawn. Weak paper with poor evaluation.

This paper is in a similar space to these PE improvement papers. Its key strengths are: (1) genuinely novel insight about imaginary component of RoPE, (2) two practical configurations with real efficiency benefits, (3) solid empirical results at 376M/776M scale, (4) compatibility with other methods. Its key weaknesses are: (1) limited model scale, (2) head-count ablation is missing for EC variant, (3) the "dominance" claim for imaginary attention is overclaimed, (4) EH variant shows inconsistent results at shorter contexts.

Compared to Round and Round We Go (avg 6.2, accepted poster), this paper has a more concrete method and better empirical validation, but similar issues with overclaiming causal mechanisms. Compared to FIRE (avg 6.7, accepted poster), this paper has comparable scale experiments but with a more novel theoretical insight. Compared to Scaling Laws of RoPE (avg 8), this paper tests at smaller scale.

I'll give this paper a score around 5.5-6.</think>## Summary
This paper identifies that standard RoPE implementations discard the imaginary component of the complex-valued dot product when computing attention scores, and proposes RoPE++ to re-incorporate this component as additional "imaginary attention" heads. The imaginary attention requires only a −π/2 rotation of query vectors before applying standard RoPE, enabling efficient implementation. Two configurations are introduced: RoPE++EH (equal head count, half KV cache) and RoPE++EC (equal cache, doubled heads). Experiments at 376M and 776M scales show improvements on long-context synthetic benchmarks (RULER, BABILong), and theoretical analysis shows the imaginary component's characteristic curve decays more slowly, benefiting long-range dependencies.

## Strengths
- **Genuinely novel and insightful observation**: The identification that standard RoPE discards the imaginary component—and that this discarded component has distinct attention properties (slower decay via sine integral characteristic curve)—is a real conceptual contribution. This "blind spot" in a widely-used technique has been largely overlooked, and the mathematical derivation connecting imaginary attention to a Si(·) decay profile is elegant and well-motivated.

- **Efficient and practical design**: The key insight that imaginary attention only requires rotating q by −π/2 before applying the same RoPE, sharing KV cache, is clean and allows a single FlashAttention pass. The two configurations (EH for cache savings, EC for performance) address different deployment scenarios, and Figure 4 concretely demonstrates that RoPE++EH reduces memory and TPOT with benefits growing at longer contexts.

- **Solid empirical coverage at moderate scale**: Pre-training from scratch at 376M and 776M with 50B+10B tokens for each variant, comparing against four position encoding baselines (RoPE, FoPE, Pythia, ALiBi), evaluating on short-context (10 benchmarks) and long-context (RULER, BABILong at multiple lengths), plus combinations with PI and YaRN—this is a substantive experimental effort.

- **Compatibility with existing long-context techniques**: Table 3 showing that RoPE++ works on top of YaRN and Linear PI strengthens the practical relevance claim.

- **Creative noise-injection ablation (Section 5.2)**: Adding Gaussian noise separately to real vs. imaginary attention components is a reasonable approach to probe their relative importance, and the finding that corrupting imaginary attention causes more degradation on long-context tasks is suggestive.

## Weaknesses

### Major

- **Missing head-count controlled ablation for RoPE++EC**: RoPE++EC doubles the number of attention heads at constant cache. Without a baseline that doubles heads under vanilla RoPE (same head configuration, no imaginary rotation), it is impossible to determine whether the improvements come from the imaginary extension itself or simply from having more attention heads with smaller per-head dimensions. This is a confound that directly undermines the paper's core attribution of gains to the imaginary component. The paper should include at minimum one "vanilla RoPE with 2× heads (halved per-head dim)" baseline to isolate the imaginary rotation's contribution.

- **Limited model scale**: All experiments are at 376M and 776M parameters with 50B training tokens. It is unclear whether observed benefits—especially the claimed dominance of imaginary attention for long context—persist at scales relevant to practical LLMs (7B+). Both reviewers and the authors' own Appendix C reference scaling analysis, but it is not presented in the main text. Long-context capability at 376M is extremely limited (e.g., RULER scores ~10 at 64k), making it hard to assess meaningful improvements in realistic settings.

- **Overclaimed dominance of imaginary attention in long-context modeling**: The claim that "imaginary attentions play a dominant role in modeling long-context dependencies" goes beyond the evidence. The noise-injection experiment (Section 5.2) compares corrupting real vs. imaginary attention at equal σ, but this does not account for potential differences in magnitude, variance, or number of heads between the two components. A cleaner ablation—e.g., zeroing out the imaginary output entirely vs. zeroing out the real output—would be more conclusive. The qualitative attention-pattern evidence ("odd-index imaginary attention highlights initial positions more strongly") also does not establish causality.

- **RoPE++EH shows inconsistent short-context results**: In Table 2 (376M Long), RoPE++EH underperforms vanilla RoPE on RULER-4k (29.9 vs. 31.6), RULER-16k (17.6 vs. 22.0), and on BABILong at several context lengths (2k: 14.1 vs. 17.7; 4k: 15.6 vs. 16.1). In Table 1 (376M Short), RoPE++EH has worse WikiText perplexity (20.8 vs. 19.9) and lower average (40.3 vs. 40.1). The paper's claim that "RoPE++EH achieves comparable performance with vanilla RoPE using half the KV-cache" is sometimes contradicted by the data, particularly at shorter contexts. This should be acknowledged more transparently.

### Minor

- **Theoretical derivation of characteristic curve (Eq. 5) is informal**: The averaging over frequency components and the leap from average sin(θ∆t) to the Si(·) function involves assumptions about frequency weight distributions that are not rigorously justified. The "semantic aggregation" claim ("when q_t, k_s are similar, attention is on average larger regardless of ∆t") is asserted without proving conditions under which it holds given the sign structure of sin. This weakens the theoretical motivation even though the empirical results stand on their own.

- **Long-context evaluation limited to synthetic benchmarks**: All long-context results are on RULER and BABILong (needle-in-haystack-style tasks). While these are widely used, they do not fully reflect real long-context comprehension (e.g., long document QA, multi-hop reasoning over books). The paper would be strengthened by at least one realistic long-context benchmark.

- **The parameter-sharing constraint is justified imprecisely (Section 3.3)**: The statement "rotating q_t in imaginary attention by π/2 yields real attention, with no architecture modification" is an oversimplification. Adding a π/2 rotation to an imaginary head produces a different attention pattern (the negative imaginary component), not the real attention pattern—these are mathematically distinct. The constraint that real and imaginary heads must share Wq could be justified more carefully (e.g., via efficiency arguments or empirical comparisons), rather than the current argument which conflates parameterization with computation.

### Trivial
- The paper's notation in Eq. 2–4 suffers from OCR/parsing issues that can make the mathematical details harder to follow, though the intended meaning is recoverable.

## Nice-to-Haves
- Perplexity-vs-context-length curves beyond the 32k training window would directly validate the length-extrapolation argument from Section 3.4.
- Training loss curves comparing RoPE++ vs. vanilla RoPE would clarify whether the efficiency claim for RoPE++EH (half cache, comparable performance) involves any convergence penalty.
- Experiments at 1.5B–7B scale, even with shorter training runs, would substantially increase confidence in scaling.
- A simple learnable mixing coefficient between real and imaginary attention, rather than a fixed 50/50 split, could potentially improve performance and is architecturally trivial to implement.

## Removed Points
These points are flagged to be removed, treat them with caution:
- **"Ambiguous/incomplete definition of the core mechanism"** (Harsh Critic Issue 1): The paper actually specifies the mechanism quite clearly in Eq. 3–4 and Section 3.3. The real+imaginary attention computation is well-defined: q is rotated by −π/2 for imaginary heads, K and V are shared, and the outputs are concatenated through W_o. The paper describes how the tensors are interleaved in FlashAttention. While more pseudocode would help, the mechanism is sufficiently specified for reproduction.
- **"Overclaiming about cache/parameter efficiency"** (Harsh Critic Issue 2, related to W_o change): The paper explicitly states that W_o in RoPE++EH equals original RoPE size and W_o in RoPE++EC is double-sized. It acknowledges that the number of output heads doubles. The accounting is transparent in Section 3.3—the information needed to compare is present.
- **"The claim that configurations like 75% imaginary are impossible is wrong"**: The paper's argument is that with shared Wq, you cannot independently control the ratio of real to imaginary heads because they are derived from the same parameterization. This is a design constraint, not a mathematical impossibility claim. The paper is explaining why their specific architecture works the way it does.
- **"Missing comparison with CoPE, FIRE, wavelet-based PE"** (Human Finder): The paper already compares with four PE methods including FoPE and ALiBi. Demanding additional baselines from potentially non-overlapping work is scope creep. FIRE and CoPE address different problems (learned interpolation, contextual counting).
- **"Reproducibility concerns about FlashAttention implementation"**: The paper provides a GitHub link (https://github.com/OpenMOSS/rope_pp). Implementation details for FlashAttention interleaving are standard engineering, not a scientific gap.
- **"Formatting/style nits"**: Removed per rules.

## Novel Insights
The observation that imaginary attention in RoPE++ has a characteristic curve approximating a sine integral (slower decay than the real attention's exponential decay) is a genuine analytical insight that has not appeared in prior RoPE literature. However, the causal attribution of long-context gains to this specific property remains under-justified by the current experiments—the head-count confound and noise-injection methodology leave the door open for alternative explanations.

## Suggestions
- **Add a head-count-controlled baseline**: Train vanilla RoPE with 2× heads and halved per-head dimension (matching RoPE++EC's architecture but without the imaginary rotation) to directly measure the imaginary component's contribution. This is the single most impactful experiment the paper could add.
- **Transparently acknowledge RoPE++EH's short-context trade-offs**: Reframe the EH results as a Pareto trade-off (cache savings at modest short-context cost) rather than claiming "comparable" performance, which is misleading given the data.
- **Add a zeroing-out ablation**: Instead of (or in addition to) noise injection, completely ablate the imaginary attention output and report the resulting performance delta. This cleanly establishes the imaginary component's contribution without confounds from noise scaling.
- **Report perplexity-at-length curves** for extrapolation beyond 32k to validate Section 3.4's claims empirically.

## Score and Decision

Calibration against similar papers:
- **Scaling Laws of RoPE Extrapolation** (ICLR, avg ~8): Strong theoretical framework, tested at 7B/13B, accepted poster.
- **Round and Round We Go** (ICLR, avg 6.2): Novel analysis of RoPE internals, some overclaimed causal conclusions, accepted poster.
- **FIRE** (NeurIPS, avg 6.7): Novel PE method, moderate scale experiments, accepted poster.
- **Wavelet PE** (NeurIPS, avg 5.25): Novel PE method, limited scale, weak math rigor, accepted poster.
- **Periodic Extension** (-withdrawn, avg 2.5): Weak paper, rejected.

This paper sits above Wavelet PE (stronger insight and clearer method) and Periodic Extension (much stronger), below FIRE (more complete ablations and larger scale), and comparable to Round and Round We Go (similar novelty in analysis but with a more concrete method). The main gaps relative to the higher-scoring papers are: (1) the missing head-count ablation which creates a serious confound for attributing gains to the imaginary component, (2) limited model scale, and (3) overclaiming of the "dominance" result for imaginary attention. These are addressable but meaningful weaknesses.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Borderline Reject</orange>