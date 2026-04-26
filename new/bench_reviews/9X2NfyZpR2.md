Now I have a thorough understanding of the paper. Let me synthesize the review, carefully filtering the harsh critic's points against the actual paper content.

Let me now assess each claim from the harsh critic:

**1. "Claimed competitiveness is overstated and selective"** — This is a valid concern. The paper claims "competitive with, and occasionally superior to, fully supervised approaches." On Breakfast 30% obs, deterministic TbLTA gets 29.03 vs ActFusion's 28.45 — genuinely competitive/superior. But on 50Salads, TbLTA deterministic avg 20.92 vs ActFusion 28.39 — a 7.5 point gap. On EGTEA, 65.37 vs 76.80 — an 11.4 point gap. However, the paper DOES acknowledge this: "Performance on 50Salads paints a complementary picture. Here, long videos, denser action distributions, and frequent transitions yield weaker temporal regularities, amplifying the impact of imprecise temporal alignment in the absence of boundary annotations." And for EGTEA: "supervised models retain a clear edge overall." And in the conclusion: "While fully-supervised models still dominate the paradigm." So the paper tempers its claims in the body, though the abstract and conclusion still make broader claims. This is a legitimate issue about claim calibration, but the paper IS partly addressing it.

**2. "Stochastic Top-1 vs deterministic baselines comparison is misleading"** — The paper is careful about presentation: the table uses distinct formatting (bold for deterministic, gray for probabilistic) and explicitly states "* means stochastic protocol" and "The highest accuracy under a probabilistic framework is indicated in gray." The text does describe stochastic results as achieving "substantially higher accuracy by capturing multiple plausible futures" but also calls this a "dual view" that "illustrates both the flexibility and the limits of our approach." While the distinction could be made more prominent, the paper does separate the results clearly in the table. This is somewhat valid but less severe than claimed.

**3. "No evaluation of pseudo-label quality"** — This is a valid concern. The framework hinges on ATBA-generated pseudo-labels, but there's no frame-level accuracy analysis of these pseudo-labels. This makes it hard to understand whether the framework works because of or despite the alignment quality.

**4. "Circular dependency in cross-modal attention"** — The binary mask M is derived from pseudo-labels Ŷ, and these pseudo-labels also supervise the features that produce them. This is inherent to self-training/bootstrapping methods and is addressed by the progressive training scheme (first training with Lvid, then with alignment losses). The paper notes: "Since pseudo-labeling requires a reliable initialization, we adopt a progressive training scheme." This is a reasonable design consideration, not a fatal flaw, though a discussion of potential instability would strengthen the paper.

**5. "Self-referential duration loss"** — This is inherent to the self-supervised nature of the method. The paper explicitly acknowledges this: "Since it is trained without temporal ground truth and relies on momentum-based class-wise priors, we use this term only as a weak duration prior rather than a precise per-instance predictor." This is a known limitation discussed in the paper.

**6. "Progressive training receives no ablation"** — Valid concern. The three-stage training could be doing important engineering work.

**7. "No standard deviations"** — True, but this is standard practice in this field. The MoC metric averaged over splits typically doesn't report std in the LTA literature. Trivial concern.

**8. "Ablation tables use Top-1 MoC vs main results"** — The paper explains this choice: "we report results using the Top-1 MoC metric. For clarity, we adopt this choice Top-1 MoC for ablations as it provides a stable reference point." Minor issue.

**9. "EGTEA rare class competitiveness is selective"** — Valid but the paper does acknowledge "supervised models retain a clear edge overall" on EGTEA. The rare class result is an interesting finding worth noting.

**10. "Conclusion overreach: 'dense LTA does not need to rely on exhaustive frame-level annotation'"** — This is indeed overclaimed given the 50Salads and EGTEA gaps. The text says "does not needs" (grammar issue, but that's parser). Valid concern about overclaiming.

Now let me finalize the calibration anchors:

**High anchors (avg ≥ 6):**
- RASO (avg 7.5, Accept Spotlight): Weakly-supervised framework, genuinely strong results across multiple benchmarks, well-designed.
- Norton (avg 8.0, Accept Oral): Multi-granularity correspondence learning for noisy video, unified OT framework, very strong.
- PointOBB-v2 (avg 7.0, Accept Poster): Weakly supervised detection, strong competitive results vs fully supervised.

**Medium anchors (avg ~5):**
- InterAct (avg 5.0, Reject): Action segmentation with unsupported claims and over-smoothing issue.
- EVA (avg 5.75, Reject): Video anticipation with overclaimed novelty.

**Low anchors (avg ≤ 4):**
- ShuffleMamba (avg 3.5, Withdrawn/Reject): Claims competitive with supervised methods but actually underperforms baselines.
- XCg9YcSKCZ (avg 3.5, Reject): Weakly supervised with overclaimed results.
- PageLgQlXz (avg 4.0, Withdrawn/Reject): Dual-level prototypes for temporal action localization.

This paper is genuinely novel — it introduces the first weakly-supervised (transcript-only) LTA framework. It has real contributions and real results. However, it overclaims competitiveness. On Breakfast it's genuinely competitive; on 50Salads and EGTEA it's substantially behind. The overclaim in abstract/conclusion is a significant concern but not fatal — the body text does discuss limitations. The stochastic vs deterministic comparison is clearly separated in the table, reducing confusion.

This paper falls somewhere between medium and high. The novelty is significant (first transcript-only LTA), the architecture is well-designed, the ablations are comprehensive, and the Breakfast results are genuinely competitive. The main issues are: (1) overclaimed competitiveness that doesn't hold on 2/3 benchmarks, (2) no pseudo-label quality analysis, (3) the stochastic vs deterministic comparison can be misleading even though separated in the table.

Compared to RASO (7.5, Spotlight) which had genuinely strong results with weak supervision across benchmarks, this paper is weaker because the results are not uniformly strong. Compared to InterAct (5.0, Reject) which had unsupported claims, this paper is stronger because it has genuine novelty and competitive results on at least one benchmark and acknowledges limitations.

I'd estimate this paper at around 5.5-6.0. It's a solid contribution with real novelty but overclaimed results that undermine the paper's claims. The method itself is sound and the architecture is well-designed.</think>Now I have thoroughly read the paper and verified the reviewer claims. Let me write the consolidated review.

## Summary

TbLTA introduces the first weakly-supervised framework for dense long-term action anticipation (LTA), trained exclusively from video transcripts (ordered action lists without timing information). The method uses an ATBA-based temporal alignment module to generate pseudo-labels from transcripts, combines these with cross-modal attention, CTC-based global consistency, CRF-enforced temporal coherence, and a self-supervised duration loss in a unified encoder-decoder architecture. Results on Breakfast, 50Salads, and EGTEA establish the first transcript-only LTA baselines, with deterministic performance competitive on Breakfast but substantially behind supervised methods on the other datasets.

## Strengths

- **Genuinely novel problem formulation**: TbLTA is the first method to tackle dense LTA using only transcript-level supervision, eliminating all frame-level annotations. This is a meaningful shift in the LTA paradigm, and the argument that LTA (which depends on logical progression of steps) is particularly suited to transcript supervision is well-motivated.

- **Sound and well-validated architecture**: Each major component (CTC loss, cross-modal attention with local masking, CRF loss, duration loss) is ablated in Tables 3–4 with clear quantitative evidence. The cross-modal attention is the most impactful component (~5.7 points on Breakfast), and the CRF loss is crucial for long-horizon predictions (~5.3 points on 50Salads, ~4.1 on Breakfast). The modular design is thoughtful and the ablations are thorough.

- **Competitive Breakfast results are real**: Under deterministic evaluation, TbLTA achieves 29.03 avg MoC on Breakfast at 30% observation, outperforming all supervised baselines including ActFusion (28.45). This is a meaningful result that demonstrates transcript-only supervision can capture procedural regularities effectively on this benchmark.

- **Comprehensive architectural contributions**: The locally-masked cross-modal attention mechanism (Eqs. 1–2), CTC-based global consistency extended to anticipation, and the self-supervised duration prior (Eq. 7) are each non-trivial and well-designed technical contributions that are individually validated.

## Weaknesses

### Fatal

None.

### Major

- **Overclaimed competitiveness across benchmarks**: The abstract states the method is "competitive with, and occasionally superior to, fully supervised approaches," and the conclusion states it "achieves results that are competitive with, and in certain settings even superior to, fully supervised methods." While this holds on Breakfast (deterministic avg 29.03 vs. ActFusion 28.45), it clearly fails on 50Salads (20.92 vs. 28.39, a 27% relative gap) and EGTEA (65.37 vs. 76.80, an 11.4-point gap). The body text partially acknowledges this ("Performance on 50Salads paints a complementary picture"; "supervised models retain a clear edge overall" on EGTEA), but the abstract and conclusion present a selective picture. The claim that "dense LTA does not need to rely on exhaustive frame-level annotation" in the conclusion is not supported by two of three benchmarks. The overclaim matters because it shapes initial reader impressions and could mislead about the method's generality.

- **Stochastic (Top-1) results are presented alongside deterministic baselines without stochastic baselines**: Table 1 presents TbLTA* Top-1 results (37.15 Breakfast avg, 28.51 50Salads avg) that are visually prominent and produce the paper's most eye-catching numbers, but these come from a fundamentally different evaluation protocol (sampling multiple futures and selecting the best) compared to deterministic baselines. While the table uses distinct formatting and a footnote, the text's statement that TbLTA achieves "substantially higher accuracy by capturing multiple plausible futures" could mislead readers into direct comparison. Without stochastic versions of supervised baselines, these numbers cannot be meaningfully contextualized. The ablation tables additionally use Top-1 MoC exclusively, making it harder to connect ablations to the main deterministic results.

- **No evaluation of pseudo-label quality**: The entire framework depends on pseudo-labels generated by the ATBA temporal alignment module, which supervises both segmentation and anticipation. The paper provides no analysis of these pseudo-labels' accuracy — no frame-level segmentation accuracy against ground truth, no analysis of alignment failure modes, and no correlation between pseudo-label quality and downstream performance. This is a critical gap because it is impossible to assess whether the framework succeeds because of or despite the alignment quality, and the weak performance on 50Salads/EGTEA may be directly attributable to alignment failures.

### Minor

- **Progressive training scheme is not ablated**: The three-stage training (10 epochs Lvid → 30 epochs LA+LTAS → full objective) with optimizer resets is described as critical for initialization quality (Section 4.1: "Since pseudo-labeling requires a reliable initialization, we adopt a progressive training scheme"), yet receives no ablation. This engineering choice may be load-bearing for the method's success.

- **No sensitivity analysis to transcript noise**: Real-world transcripts may contain ordering errors, omissions, or incorrect labels. The paper assumes transcripts are clean and perfectly ordered, which may limit practical applicability.

### Trivial

- None.

## Nice-to-Haves

- Report frame-level accuracy of ATBA pseudo-labels against ground truth (even in an appendix or supplementary), and analyze the correlation between alignment quality and downstream LTA performance.
- Add stochastic evaluations of at least one supervised baseline (e.g., ActFusion) to contextualize the Top-1 numbers.
- Provide an ablation of the progressive training scheme to demonstrate its necessity.
- Include error analysis by prediction horizon and action type to clarify when transcripts suffice and when dense annotations remain necessary.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Circular dependency in cross-modal attention creates instability or confirmation bias"** — The progressive training scheme (Lvid first, then alignment losses) explicitly addresses initialization concerns, and the paper notes the necessity of this design. This is a standard bootstrapping concern in self-training methods, not a novel criticism specific to this architecture.

- **"Self-referential duration loss entrenches early biases"** — The paper explicitly acknowledges the self-supervised nature of the duration loss and positions it as "a weak duration prior rather than a precise per-instance predictor," discussing its limitations regarding classes with high intra-class variability. This is already addressed in the paper.

- **"No standard deviations reported"** — Reporting averaged MoC over splits without standard deviations is standard practice in the LTA literature (all baselines in Table 1 also omit std). This is a field norm, not a paper-specific flaw.

- **"EGTEA rare-class competitiveness is selective"** — The paper explicitly states "supervised models retain a clear edge overall" on EGTEA. Discussing rare-class performance is a genuine insight about where transcript supervision helps, not a selective claim.

- **"Ablation uses Top-1 MoC while main results use deterministic metrics"** — The paper provides a clear justification: "For clarity, we adopt this choice Top-1 MoC for ablations as it provides a stable reference point." The Top-1 metric is more stable for component-wise comparisons; this is a reasonable and explained methodological choice.

- **"Missing related works"** — Per the instructions, I cannot verify completeness of the related work section.

- **"Formatting/grammar issues"** — Removed per parser artifact rule.

## Novel Insights

The paper reveals an interesting asymmetry: transcript-only supervision works surprisingly well for LTA on procedurally structured activities (Breakfast) where the logical ordering of steps carries substantial predictive power, but degrades significantly on datasets with weaker temporal regularities (50Salads, EGTEA). This suggests that the viability of weak supervision for anticipation is fundamentally task-dependent — not just on the supervision paradigm but on the inherent predictability of the activity structure. The rare-class advantage on EGTEA (60.11 vs. 55.10) also suggests that semantic transcript information can partially compensate for data imbalance, a finding that could inform future weakly-supervised approaches.

## Suggestions

- Temper the competitiveness claims: clearly state in the abstract and conclusion that results are competitive on Breakfast but substantially behind on 50Salads and EGTEA, and discuss the dataset properties that explain this discrepancy.
- Add a pseudo-label quality analysis: even a simple frame-level accuracy metric for the ATBA-generated pseudo-labels would clarify the framework's operating conditions.
- If stochastic Top-1 results are reported, include at least one stochastic supervised baseline to make the comparison interpretable.

## Score and Decision

**Calibration anchors:**

1. **Norton** (avg 8.0, Accept Oral) — Multi-granularity correspondence learning for noisy videos; very strong, unified OT framework with consistently strong results. TbLTA is less strong due to inconsistent competitiveness across benchmarks.

2. **RASO** (avg 7.5, Accept Spotlight) — Weakly-supervised surgical model that genuinely outperforms supervised methods across multiple benchmarks. TbLTA is weaker because its competitiveness is selective (only 1/3 benchmarks).

3. **PointOBB-v2** (avg 7.0, Accept Poster) — Single-point supervised detection with competitive vs. supervised results; reviewers noted limited theoretical justification but sound results. TbLTA is comparable in novelty but has more overclaiming issues.

4. **InterAct** (avg 5.0, Reject) — Action segmentation with unsupported claims and over-smoothing; overclaimed results. TbLTA is clearly stronger — genuine novelty, real competitive results on Breakfast, and proper ablations.

5. **EVA** (avg 5.75, Reject) — Video anticipation with overclaimed novelty. TbLTA has stronger novelty and more comprehensive evaluation.

6. **ShuffleMamba** (avg 3.5, Withdrawn/Reject) — Claims competitive with supervised ViTs but actually underperforms baselines on multiple benchmarks. TbLTA is significantly stronger — it genuinely outperforms supervised methods on Breakfast and has sound architectural design.

TbLTA sits between InterAct/EVA (rejected, ~5-6) and PointOBB-v2 (accepted poster, ~7). Its genuine novelty as the first transcript-only LTA method with competitive Breakfast results pushes it above 5, but the selective competitiveness claims and missing pseudo-label analysis keep it below 6.5.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>