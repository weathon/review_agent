Now I have sufficient information. Let me compile the final review.

## Summary

This paper investigates whether the performance gap between Transformers and modern recurrent models (SSMs) on associative recall and copying tasks is due to expressivity limitations or optimization difficulties. Through over 3,000 experimental runs, the authors demonstrate that SSMs succeed only within an extremely narrow learning rate window—a confound missed by prior work—leading them to argue that optimization instability, not expressivity, is the primary differentiator. The paper also documents contrasting scaling behaviors (width for SSMs, depth for Transformers), analyzes single-layer training dynamics, and identifies the 1D convolution as the critical architectural component enabling Mamba's single-layer success on MQAR.

## Strengths

- **The LR sensitivity finding is a genuine and important methodological contribution.** Figure 1 compellingly shows that Mamba and Hyena achieve near-zero accuracy for most learning rates but spike to success at specific values—values that prior work's sparse grids (Arora et al., 2023, dashed lines) systematically missed. This directly recontextualizes previous expressivity conclusions as potentially confounded by suboptimal tuning. Figure 5 replicates this finding on the copying task, confirming it is not task-specific. This should change how future architecture comparisons handle hyperparameter search.

- **The convolution ablation (Table 2) is a clean, symmetric mechanistic finding.** Removing the 1D convolution from 1-layer Mamba drops accuracy from 99% to 2% (matching 1-layer Attention's 2%), while adding a convolution to 1-layer Attention raises accuracy from 2% to 99%. This cleanly identifies the convolution as the critical component and provides a falsifiable architectural insight for shallow sequence model design.

- **The scale of the empirical effort (3,000+ runs, ~20,000 GPU hours, 5 seeds with error bars) provides high credibility** that the LR sensitivity finding is not an artifact of a few bad seeds or narrow configurations.

- **The paper correctly identifies a real methodological gap in prior work**: the tendency to evaluate SSMs with LR grids designed for Transformers, which can systematically understate SSM capabilities. Figure 2's overlay of Arora et al.'s results vs. the fine-tuned results directly illustrates this problem, with Mamba becoming solvable at high sequence lengths where it was previously thought to fail.

## Weaknesses

### Fatal

None.

### Major

- **The central thesis overclaims what the evidence supports.** The introduction states: *"Transformers differ from SSMs not in terms of expressive power but mainly because of their optimization dynamics."* This strong formulation is not sustained by the paper's own results. (1) The 1-layer Transformer's failure on MQAR—regardless of width or learning rate (Figure 3)—is an expressivity limitation, not an optimization one, which the paper itself acknowledges: *"a single-layer transformer lacks the expressivity needed to effectively leverage this mechanism"* (Section 6). (2) Even with careful LR tuning, Hyena still shows sizable gaps at low widths (Section 4: *"we confirm that a sizable gap with Transformers can still be observed at low widths (e.g. Hyena)"*). (3) The paper's own data in Figure 2 shows that while fine-tuned Mamba improves dramatically, it does not fully close the gap at all settings. The evidence supports a more nuanced claim—that optimization instability is a significant and underappreciated confound in prior evaluations—but not the claim that expressivity differences are secondary. The abstract and conclusion use more moderate language ("not just in their expressivity"), yet the introduction's strong framing dominates the narrative. This matters because the overclaim could lead practitioners to underestimate genuine expressivity differences.

### Minor

- **The narrow LR window is not quantified.** Figure 1 is the paper's most important result, yet the viable range's width, the fraction of the searched range that yields success, and how it scales with model size or sequence length are never reported. Without quantification, the reader cannot assess severity or predict behavior at scale. This would strengthen—but does not invalidate—the finding.

- **The "opposite scaling behavior" finding is likely a consequence of the same optimization brittleness rather than an independent architectural insight.** The paper acknowledges that *"brittle optimization has a direct impact on scaling, causing SSMs to favor width over depth"* (abstract), so the link is recognized. However, the presentation in Table 1 and Figure 4 frames scaling preference as an independent finding, when the deeper-but-narrower Mamba failing (24-layer at 16%) is more naturally explained by depth-compounded optimization difficulty—essentially the same problem documented in Figure 1. The framing implies wider SSMs are architecturally superior, when the real lesson may simply be "deeper SSMs are harder to optimize," which is not novel.

- **The induction head interpretation for 1-layer models is imprecise.** The paper describes a loss bump in 1-layer Attention as "resembling the formation of an induction head circuit" (Section 6). However, induction heads are by definition a two-layer circuit (Olsson et al., 2022, as cited by the paper in Section 2). The paper hedges with "resembles" and "attempts to form," which is appropriate, but the repeated invocation of "induction heads" for a single-layer model risks misleading readers. Similarly, the Mamba loss bump is described as "reinforcing the connection between Mamba and Attention mechanisms"—a loss bump during training can have many causes, and this connection is speculative without mechanistic evidence.

- **The DeltaNet stability hypothesis is plausible but unverified.** The paper hypothesizes that Householder-based updates prevent vanishing off-diagonal gradients, distinguishing DeltaNet from Mamba/Mamba2. This is presented as a hypothesis (which is fine), but no controlled ablation tests this specific mechanism (e.g., replacing DeltaNet's update rule with a Mamba-style decay while holding other components fixed).

- **Validation is limited to synthetic benchmarks.** The authors acknowledge this in Section 8, and the jump from MQAR/copying to language modeling is large. Even a small-scale language modeling experiment would substantially strengthen the practical relevance of the LR sensitivity finding.

### Trivial

None.

## Nice-to-Haves

- Quantify the viable LR window (e.g., fraction of searched range achieving >90% accuracy) and test whether standard training recipes (warmup, cosine scheduling, gradient clipping) expand it—this would directly address the most practically relevant question for practitioners.
- A controlled ablation for the DeltaNet stability hypothesis (e.g., replacing DeltaNet's Householder updates with Mamba-style decay while holding other components fixed) would transform the hypothesis into a verified mechanistic insight.
- Even a small-scale language modeling experiment (e.g., WikiText-103) would substantially strengthen claims about practical relevance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh critic: "Table 1 parameter matching is unfair (80M vs 150M)"** — The paper is NOT comparing 80M Mamba to 150M Transformer as a fair match. The table shows a progression: 80M (12-layer, width 1024) fails, 150M achieved via depth (24-layer, width 1024) barely succeeds at 16%, and 150M achieved via width (12-layer, width 1408) succeeds at 100%. The argument relies on the two 150M configurations, not the 80M vs 150M comparison. Partial misread by the harsh critic.

- **Harsh critic: "The 'opposite scaling behavior' finding is not independent"** — The paper itself acknowledges the causal link between optimization brittleness and scaling preference (abstract: "brittle optimization has a direct impact on scaling"). The harsh critic treats this as a criticism, but the paper already makes this connection. The minor weakness is about framing, not about the paper missing the connection.

- **Harsh critic: demand for testing standard training recipes (warmup, cosine scheduling, gradient clipping)** — This is a reasonable suggestion but goes beyond the paper's stated scope. The paper's contribution is identifying the LR sensitivity confound; whether standard recipes mitigate it is a follow-up question. Moved to Nice-to-Have.

- **Harsh critic: "Show the actual learned attention/SSM matrices for successful vs. failed runs"** — This is a reasonable suggestion for future mechanistic analysis but is beyond the paper's scope and would add considerable complexity. Moved to Nice-to-Have.

- **Strength finder: "Novel observation of induction-head-like dynamics in single-layer Transformers"** — This is listed as a strength but the induction head interpretation for a single layer is imprecise (as noted in the weaknesses). The observation of a loss bump is genuine, but calling it "induction-head-like" over-interprets it. Downgraded from a strength to part of the minor weakness.

- **Strength finder: "Hypothesis for DeltaNet's superior stability"** — A hypothesis is not a contribution unless verified. The observation that DeltaNet is more stable IS a contribution (Figure 7), but the explanation remains speculative. Downgraded.

## Novel Insights

The paper's most important insight—that the SSM vs. Transformer gap on associative recall was significantly confounded by optimization hyperparameter selection—is both genuine and actionable. However, a subtler insight emerges from the tension in the paper's own evidence: the relationship between expressivity and optimization is not zero-sum but interacting. The 1-layer Attention failure is purely expressivity-limited, the 2-layer gap is largely optimization-limited, and the scaling behavior is optimization-limited-expressivity (wider models are both more expressive and easier to optimize). The paper's binary framing ("expressivity vs. optimization") misses this interaction, which is arguably more interesting than either factor alone.

## Suggestions

- Revise the central thesis in the introduction from "not in terms of expressive power but mainly because of their optimization dynamics" to the more accurate "both expressivity and optimization contribute, but optimization instability has been a significant and underappreciated confound in prior evaluations." This would align the framing with the paper's own evidence while preserving the genuine contribution.
- Quantify the viable LR window across model sizes and sequence lengths (even as a simple table reporting the fraction of the grid achieving >90% accuracy), transforming Figure 1 from a qualitative observation into a rigorous finding.
- Replace "induction head" language for 1-layer dynamics with more precise terminology (e.g., "loss phase transition reminiscent of multi-layer induction head formation") to avoid the definitional conflict.

## Score and Decision

**Calibration anchors:**

1. **zj2mI9TSF7** (Mechanistic comparison of Transformers vs SSMs on AR, avg 4.0, Reject) — Most topically similar; this paper's LR sensitivity finding and convolution ablation are more concrete and actionable contributions, placing our paper above this anchor.

2. **C0AQNXhTiY** (Memory determines learning direction in SSMs, avg 4.5, Reject) — Similar SSM optimization topic; our paper has much more extensive experiments and a clearer practical finding, placing it above this anchor.

3. **8cDoHzqDXP** (Recall scaling laws in Mamba, avg 3.33, Reject) — Similar topic but weaker and less generalizable; our paper is clearly above this.

4. **D0u0glT060** (Deconstructing Positional Information, avg 7.2, Accept Poster) — Synthetic task analysis with overclaimed generality, but with theoretical backing. Our paper has weaker theoretical grounding and similar synthetic-only scope, placing it below this anchor.

5. **A4Us8jxVGq** (How Transformers Learn to Associate Tokens, avg 7.2, Accept Oral) — Strong theory + real-world validation; our paper is well below this.

6. **NX6oMuoLxu** (Normalization in Mamba, avg 2.67, Reject) — Much weaker, poorly grounded; our paper is clearly above this.

7. **clCsSQ5rKg** (NeuMa, avg 2.0, Reject) — Much weaker; our paper is clearly above this.

The paper sits above the 4.0-4.5 range (stronger than topically similar rejected papers) but below the 7.0+ range (weaker than accepted papers with similar synthetic-task profiles that had theoretical backing). The genuine contribution (LR sensitivity confound, convolution ablation) is significant, but the overclaiming in the introduction and the lack of quantification/practical validation keep it from the accept range. A score of 5.5 reflects a paper with real contributions that need more honest framing and additional rigor to reach acceptance.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>