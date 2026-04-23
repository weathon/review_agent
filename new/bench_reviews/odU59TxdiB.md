Now I have all the information I need to write the final review.

## Summary

SSLAM introduces audio mixture-based self-supervised pre-training to improve transformer models' ability to handle polyphonic soundscapes. The method combines three novel components: (1) element-wise max mixing of log-mel spectrograms as a training signal, (2) a partial mixing strategy that only mixes t/2 of the audio duration, and (3) a Source Retention Loss (SRL) that encourages the student to match the average of teacher representations from individual sources. Evaluated on AudioSet-2M, the method achieves 50.2 mAP (a claimed 3.9% improvement over prior SOTA), with consistent improvements on polyphonic benchmarks particularly in linear evaluation.

## Strengths

- **Well-motivated problem identification with evidence.** The paper makes a compelling case that AudioSet's multi-label samples are often not truly polyphonic — e.g., "Carnatic music," "Music," "Musical instrument," and "Classical music" labels describe facets of the same event (Section 1, Appendix B.1). This motivates explicit polyphonic training beyond what standard AudioSet pre-training provides.

- **Comprehensive polyphonic evaluation protocol.** The inclusion of SPASS (5 sub-environments), IDMT-DESED-FL, and URBAN-SED, plus the degrees-of-polyphony breakdown (Table 3: {2,3} through {14+}), provides a valuable evaluation template that goes well beyond standard audio SSL benchmarks.

- **Consistent linear evaluation improvements across all polyphonic datasets.** Table 2 shows SSLAM outperforms the baseline (MB-UA) across all six polyphonic settings in linear evaluation, with improvements up to +5.7 mAP on SPASS-Market (62.8→68.5). The incremental ablation (MB-UA→MB-PMA→MB-UA-PMA→SSLAM) cleanly demonstrates that each component contributes.

- **Advantage scales with polyphony degree.** Table 3 is a particularly compelling evaluation — SSLAM's advantage over the baseline grows from marginal at {2,3} to +3.0 mAP at {14+} in linear evaluation, directly validating that the method targets the intended challenge.

- **Maintains monophonic performance while achieving SOTA on AS-2M.** Table 1 shows SSLAM achieves 50.2 mAP on AS-2M (improving over the prior best 48.6) while remaining competitive on ESC-50 (96.2%) and KS1 (98.8%), addressing the natural concern that polyphonic training could hurt monophonic generalization.

## Weaknesses

### Fatal
None

### Major

- **The core methodological choice — element-wise max mixing — is not compared against the most natural alternative (additive mixing in the spectrogram domain).** The paper uses max(S₁, S₂) in log-mel space (Eq. 3), which produces a spectrogram that corresponds to no physical audio signal. In real polyphonic audio, sources combine additively: the correct spectrogram-domain mixture would be approximately log(exp(S₁) + exp(S₂)). The paper states "we found that performing the mix in the spectrogram domain yielded superior performance (refer to Appendix E.0.1)" (Section 3.2.1), but this comparison is spectrogram-domain vs. waveform-domain mixing, *not* max vs. additive. The IBM/CASA justification is loose — IBM is a separation target for when you know which source to extract, not a strategy for creating training data. Without a direct comparison to additive mixing, the central design decision remains empirically unjustified against the physically correct alternative.

- **The ablation study never reports results on AS-2M, the benchmark carrying the headline number.** The paper's most prominent claim is 50.2 mAP on AS-2M (Table 1), a 1.6-point improvement over prior work. Tables 2–6 evaluate on SPASS variants, IDMT, URBAN-SED, and AS-20K — but never on AS-2M. Table 4 shows Stage 1 with unmixed audio achieves 40.2 on AS-20K and the full SSLAM achieves 40.9, suggesting ~0.7 mAP improvement from the proposed components on AS-20K. Without the corresponding AS-2M numbers, we cannot determine how much of the 50.2 headline result comes from the baseline architecture/training versus the proposed mixing and SRL contributions. If Stage 1 already achieves ~49.5 on AS-2M, the proposed contributions add very little to the headline result.

- **The abstract overclaims polyphonic improvements by attributing the "up to 9.1% (mAP)" figure to "both linear evaluation and fine-tuning regimes."** The abstract states: "SSLAM sets new SOTA in both linear evaluation and fine-tuning regimes with performance improvements of up to 9.1%(mAP)." The 9.1% figure (5.7 mAP absolute on SPASS-Market) comes exclusively from linear evaluation. In fine-tuning on the same dataset, the improvement is only 0.5 mAP (89.7→90.2, Table 2). Across all polyphonic datasets in fine-tuning, the largest improvement is 2.5 mAP (SPASS-Waterfront). While the paper acknowledges "fine-tuning improvements were marginal" in Section 5 (line 248), the abstract framing is misleading and the paper's claim of "substantial improvements in handling real-world polyphonic audio" (line 49, contributions point 4) is not supported by the fine-tuning evidence.

### Minor

- **SRL global loss hurts performance (Table 5, row 4) without explanation.** Table 5 shows that adding SRL global loss drops AS-20K fine-tuning from 40.9 to 40.6. The paper acknowledges this ("everywhere except SRL, the global loss showed performance improvement," line 250) but does not explain why. This suggests SRL's benefit is limited to the local loss only, which constrains the scope of this claimed contribution.

- **SSLAM underperforms the baseline at low polyphony in linear evaluation.** Table 3 shows SSLAM achieves 60.6 mAP at the {2,3} level vs. MB-UA's 61.5 in linear evaluation. Since low-degree polyphony is the most common real-world scenario, this is notable, though the paper does acknowledge it (line 248). In fine-tuning, SSLAM recovers (87.7 vs 87.3).

- **No comparison of single-stage training (all losses from scratch) vs. the two-stage curriculum.** The two-stage approach adds training complexity, and without testing whether a single-stage approach achieves similar results, we don't know if the curriculum is necessary.

### Trivial
None

## Nice-to-Haves

- A direct comparison of element-wise max vs. additive mixing (log(exp(S₁) + exp(S₂))) with all other components fixed would resolve the major design justification gap.
- Reporting the Stage 1 model's performance on AS-2M would clarify the contribution of the proposed components to the headline result.
- Multiple seeds with standard deviations would strengthen confidence in small-margin improvements, though single-run evaluation is common in large-scale audio SSL.
- Visualization of max-mixed spectrograms vs. additive-mixed spectrograms vs. real polyphonic recordings would help readers assess the distribution shift.
- Testing on real-world polyphonic field recordings beyond synthetic mixtures would validate transfer.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The paper does not discuss the negative SRL global loss result (Table 5)."** — Removed because the paper explicitly discusses this at line 250: "Our experiment showed that everywhere except SRL, the global loss showed performance improvement (refer to Table 5)." While the paper doesn't explain *why*, it does acknowledge the result.

- **"No error bars / standard deviations reported, making small improvements statistically unverifiable."** — Weakened to minor/nice-to-have. Single-run evaluation without error bars is standard practice in large-scale audio SSL pre-training papers (e.g., MERT, BEATs, Audio-MAE also report single numbers). Requesting multiple runs of full AudioSet-2M pre-training is impractical.

- **"IBM is a separation target, not a data creation strategy — the analogy is misleading."** — The paper says "inspired by principles from CASA, particularly the IBM" (Section 3.2.1), not that they are implementing IBM. While the analogy is loose, the paper is careful with its language. Downgraded from a standalone weakness; the substantive concern is already captured in the max-vs-additive mixing issue.

- **"SRL target of averaging teacher representations has no theoretical basis."** — In SSL, many design choices (including the EMA teacher itself) lack theoretical grounding and are justified empirically. The paper shows SRL helps consistently in Table 2 ablations. This is trivial-level.

- **"Partial mixing strategy (3 regions covering t/2) is ad hoc."** — Table 4 provides empirical justification (partial mixing 39.9 vs. full mixing 39.0 in Stage 1). The specific parameter choices could benefit from sensitivity analysis but are empirically grounded.

- **"Ablation variants all start from Stage 1 pretrained weights, so comparisons don't isolate mixing effects."** — The paper is transparent about this experimental design (Section 4.3). Comparing Stage 2 strategies given the same initialization is a reasonable ablation approach, and the paper is explicit about what is being compared.

- **"No total training FLOPs/epochs comparison vs. baselines."** — This is a minor computational comparison concern; most audio SSL papers don't report FLOPs comparisons. The paper does report GPU hours (4×3090, 7h/epoch Stage 1, 7.5h/epoch Stage 2).

- **"Missing comparison with mixup/SpecAugment mix from speech."** — These are supervised augmentation strategies, not self-supervised pre-training objectives. The comparison would be out of scope.

## Novel Insights

The most insightful observation from the reviews is the tension between linear evaluation and fine-tuning as proxies for practical impact. SSLAM's dramatic linear-evaluation improvements (up to 9.1%) but marginal fine-tuning improvements (0.5-2.5 mAP) suggest the mixing-based training alters representation structure in ways that are easily overcome by fine-tuning. Since the paper motivates its work partly by noting that pre-trained encoders are often "used in a frozen state" with only a projection layer added (Section 1, line 25-29), the linear evaluation improvements *are* practically relevant for that use case — but the paper does not explicitly make this argument to justify why linear evaluation improvements matter despite marginal fine-tuning gains. This framing would significantly strengthen the paper's practical contribution story.

## Suggestions

- Add one row to Table 2 showing AS-2M results for each ablation variant (MB-UA, MB-PMA, MB-UA-PMA, SSLAM). This is the single most impactful experiment the authors could add to support the headline claim.
- Add one ablation comparing element-wise max vs. additive mixing in the spectrogram domain (e.g., log(exp(S₁) + exp(S₂))), keeping all other components fixed. This directly addresses the core methodological question.
- Qualify the abstract's "up to 9.1% (mAP)" claim to specify it applies to linear evaluation, or provide the corresponding fine-tuning maximum improvement alongside it.

## Score and Decision

**Calibration anchors compared:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| MERT (w3YZ9MSlBu) | 7.50 | Accept (poster) | Audio SSL with novel objectives, comprehensive evaluation, clearer attribution of contributions. Stronger than SSLAM. |
| Multi-resolution HuBERT (kUuKFW7DIF) | 8.00 | Accept (spotlight) | Novel SSL objective with strong, well-attributed results. Much stronger than SSLAM. |
| MW-MAE (Q53QLftNkA) | 5.25 | Accept (poster) | Audio SSL with marginal gains but complete ablations. Weaker problem motivation than SSLAM but more complete evidence. |
| Matrix-SSL (e1IMBXiDhW) | 5.75 | Reject | SSL with overclaimed SOTA and missing key comparisons. Similar weakness pattern (overclaiming, incomplete ablations) but less genuine novelty. |
| DeCUR (TQsrRW9mq9) | 5.25 | Reject | Strong results but missing comparisons. Similar pattern. |
| DnfPX10Etk (DnfPX10Etk) | 3.50 | Reject | SOTA claims with limited baselines. Weaker than SSLAM which has better problem formulation. |
| HarmonyLM (mp8ZgMZ1RG) | 1.67 | Reject | Fundamental issues with novelty. Far weaker than SSLAM. |

SSLAM is stronger than the clearly rejected low-scoring papers (HarmonyLM, DnfPX10Etk) because it identifies a genuine, important gap and provides a comprehensive polyphonic evaluation. It is comparable to the mid-range rejected papers (Matrix-SSL, DeCUR) in having overclaimed results with incomplete ablations, but has a more compelling and novel problem formulation. It sits below MW-MAE (5.25, accepted) because MW-MAE, despite marginal gains, provided complete ablation evidence. The missing AS-2M ablation is a significant evidential gap for a paper whose headline claim is the AS-2M SOTA, and the max-vs-additive mixing gap leaves the core design choice unjustified. These are not issues that can be fully resolved in rebuttal (they require new experiments), placing the paper in borderline-reject territory.

**Originality:** Moderate — the problem identification is novel for audio SSL, but the method builds incrementally on masked latent bootstrapping with a mixing strategy whose physical motivation is weak.

**Importance of research question:** High — polyphonic audio handling is genuinely underexplored and practically important.

**Claims well supported:** Moderate — the polyphonic evaluation evidence is strong in linear evaluation but the headline AS-2M claim lacks ablation support, and fine-tuning improvements are marginal.

**Soundness of experiments:** Moderate — comprehensive polyphonic evaluation but missing the most critical ablation (AS-2M) and the core design comparison (max vs. additive).

**Clarity:** Good — well-structured presentation with clear problem motivation.

**Value to community:** Good — the polyphonic evaluation protocol and degrees-of-polyphony analysis are useful contributions regardless of the method's merits.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>