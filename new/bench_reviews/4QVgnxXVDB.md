Now I have enough calibration data. Let me write the final review.

## Summary
This paper proposes 3CIL, a causality-inspired imitation learning method for autonomous driving that combines contrastive learning, action residual prediction, and divergence-based sample weighting to mitigate spurious correlations. The method achieves highest accumulated rewards in 5 of 6 CARLA evaluation scenarios, including unseen towns, but shows a significant performance drop (>50% compared to its ablation baseline) in one out-of-distribution setting.

## Strengths
- **Structured trait-to-loss mapping grounded in causal analysis**: The paper derives three robustness traits (T1-T3) from a causal diagram of the driving task and maps each to specific optimization objectives ($\mathcal{L}_{fo}$ for future reconstruction, $\mathcal{L}_{ar}$ for action residual prediction, $\mathcal{L}_{RNC}$ for contrastive learning), providing clearer theoretical motivation than heuristic causal applications (Section 3.1, Equations 1-3).

- **Action residual prediction as training regularizer**: By optimizing the representation model to predict action residuals $\Delta a_t$ rather than absolute actions during training (Equation 2), the method forces the latent state to capture state dynamics instead of relying on action-history shortcuts, directly addressing the inertia and copycat failure modes identified in Section 2.1.

- **Divergence-based sample weighting**: The policy learning stage assigns sample weights based on action residual prediction errors from the representation model (Equation 4), enabling identification of scenarios where the representation diverges from expert dynamics rather than relying on policy prediction errors used in prior work like Keyframe (Section 3.3).

- **Strong empirical performance in majority of scenarios**: 3CIL achieves highest accumulated rewards in 5 of 6 evaluation scenarios including unseen towns (Scenario 5: 538.50 vs. 516.70 for second-best; Scenario 1-4 all best), demonstrating the combined components improve robustness compared to baselines like CIL, DIGIC, and PALR (Table 1).

## Weaknesses

### Fatal
None

### Major
- **Contradictory generalization results undermine robustness claims**: The central claim is improved robustness and generalization to unfamiliar scenarios (Abstract, Introduction). However, in Scenario 6 (unseen Town05), 3CIL achieves an accumulated reward of 195.53, significantly lower than both the ablation baseline RAP (447.44) and DIGIC (409.88)—a >50% performance drop relative to RAP (Table 1). This result directly contradicts the conclusion that "3CIL still maintains a robust driving strategy" (Section 4.2) and suggests the proposed causal components may actively harm out-of-distribution generalization in certain settings. The paper does not analyze why 3CIL fails specifically in Town05 while RAP (which lacks the sample-weighting term) succeeds, leaving a critical gap in understanding the method's limitations.

- **Overclaimed causal framing without structural enforcement**: The paper's narrative heavily emphasizes "causal reasoning" and positions the method as addressing "causal confusion" through causal mechanisms. However, the method implements standard deep learning components (VAE reconstruction loss, supervised contrastive loss, residual MSE) without structural interventions that actually prevent use of spurious features (e.g., invariant risk minimization, gradient reversal, or backdoor adjustment). The causal graph in Figure 1b is used for post-hoc justification and trait derivation rather than as a structural constraint on the architecture. While the paper uses "causality-inspired" in the title, the text creates an expectation gap by implying causal mechanisms where only causal interpretations of standard losses exist.

### Minor
- **No statistical significance reporting**: Table 1 reports single-run metrics without variance or standard deviation. CARLA evaluations are known to be high-variance, and without multiple seeds, the superiority of 3CIL over RAP in Scenarios 1-5 and its inferiority in Scenario 6 cannot be statistically validated. This is particularly concerning given the Scenario 6 discrepancy.

- **Ambiguous positioning of ablation baselines**: Section 4.1 describes RNC and RAP as methods that "can be seen as ablation experiments of 3CIL" while citing them as external works (Zha et al., 2023; Chuang et al., 2022). This conflates author-implemented ablations with independent external baselines, potentially obscuring that RAP (the strongest comparison in Scenario 6) is essentially 3CIL minus the weighting term. Clearer distinction between reproduced baselines and ablation variants would improve transparency.

### Trivial
None

## Nice-to-Haves
- **Analysis of Scenario 6 failure mode**: Investigating why 3CIL underperforms in Town05 compared to RAP would strengthen the paper—whether the contrastive loss overfits to source towns, the weighting term amplifies noise in unseen domains, or other factors. This analysis is crucial for the robustness claim.

- **Input ablation for speed dependency**: Conducting an ablation removing $v_{speed}$ from the input vector would test whether 3CIL is truly robust to spurious correlations. If performance degrades severely, it would confirm reliance on speed as a shortcut feature despite the causal framing.

- **Formal derivation linking losses to causal objectives**: If the method is positioned as causality-inspired, providing a formal derivation linking the loss functions to a causal objective (e.g., minimizing causal risk rather than empirical risk) would strengthen the theoretical grounding beyond heuristic justification.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Harsh Critic Point 2 (Ambiguous Inference Procedure)**: The critic claims ambiguity about whether the model predicts absolute actions or residuals at inference. However, Section 3.3, line 186 explicitly states: "For a certain sample $h_i$, the imitator's prediction is made as: $\hat{a}_i \sim J(\hat{a}_i|\hat{s}_i)$, $\hat{s}_i \sim G(\hat{s}_i|h_i)$." This clearly indicates absolute action prediction at inference. The action residual prediction is ONLY used during training (for the representation model) and for computing sample weights. This is a valid design choice where the residual branch serves as a training regularizer, not a deployment mechanism. The critic misread the inference procedure.

- **Harsh Critic Point 4 (Baseline Conflation as inflating competitiveness)**: While the positioning of RNC/RAP as both ablations and external works is a minor presentation issue, the critic's claim that this "inflates the perceived competitiveness" is overstated. The paper is transparent that these are implemented as ablation variants based on prior work, and RAP actually outperforms 3CIL in Scenario 6, which would not inflate 3CIL's competitiveness if anything.

- **Strength Finder claim about "Superior Generalization in Distribution Shifts"**: This strength claims 3CIL achieves highest rewards in 5 of 6 scenarios "including unseen towns (Scenario 5, 6)." However, Table 1 shows 3CIL does NOT achieve the highest reward in Scenario 6 (195.53 vs. RAP's 447.44). This strength is factually incorrect and conflicts with the verified Major weakness about Scenario 6 failure.

- **Generic strengths about problem importance**: Any strengths merely stating the paper "addresses an important problem" or "targets an interesting question" without concrete evidence are removed as superficial.

## Novel Insights
The paper's core tension reveals an important observation about causality-inspired methods: combining causal and anti-causal supervisions (residual prediction + contrastive learning) with adaptive sample weighting improves performance in most distribution shift settings, but may create fragility in specific OOD scenarios where the weighting mechanism amplifies domain-specific noise. The fact that RAP (3CIL without sample weighting) significantly outperforms 3CIL in Town05 suggests the sample-weighting term, while beneficial in 5/6 scenarios, introduces a failure mode in certain unseen environments. This trade-off between adaptive emphasis and OOD stability warrants deeper investigation and is not adequately discussed in the paper.

## Suggestions
1. **Report multi-seed results with confidence intervals**: Re-run experiments with at least 5 seeds and report mean ± std for all metrics in Table 1 to enable statistical validation of claims.

2. **Analyze the Scenario 6 failure**: Provide a detailed analysis of why 3CIL fails in Town05 compared to RAP. Is it the contrastive loss overfitting? Does the weighting term amplify noise? Include visualization of attention maps or representation space to diagnose the failure mode.

3. **Clarify causal framing**: Soften claims about "causal reasoning" driving the design. Explicitly position the method as "causality-inspired" (as in the title) with the causal diagram serving as motivational framework rather than structural constraint. Consider adding a limitation discussing that standard losses are used without formal causal intervention mechanisms.

4. **Distinguish ablation variants from external baselines**: Clearly separate author-implemented ablation variants (RNC, RAP as implemented in this work) from independently developed external baselines (CIL, DIGIC, PALR, etc.) in Section 4.1 to avoid confusion.

5. **Add input ablation for speed**: Test performance when $v_{speed}$ is removed from the input to verify whether 3CIL truly reduces reliance on spurious speed-action correlations.

## Score and Decision

**Calibration Process:**

I retrieved anchors across three score bands:

**High-scoring anchors (≥6):**
- `/home/wg25r/review_agent/human_reviews_2026/a9bOgeqbdB.md` (6.67): RAP paper with exceptional experimental validation across 4 benchmarks, thorough ablations, and no contradictory results.
- `/home/wg25r/review_agent/human_reviews_2026/lTaPtGiUUc.md` (7.33): World model with thorough ablations but reviewers noted marginal gains and missing confidence intervals.
- `/home/wg25r/review_agent/human_reviews_2026/sFjxg8cyJS.md` (6.00): Causal confusion analysis with theoretical contributions and solid experiments.
- `/home/wg25r/review_agent/human_reviews_2026/5d7prMWHNF.md` (6.00): Causal delta embeddings with strong OOD performance and clear ablations.

**Medium-scoring anchors (around 5):**
- `/home/wg25r/review_agent/human_reviews_2026/ZMDoV1RaXC.md` (5.00): Curriculum learning for driving, moderate novelty, engineering-oriented.
- `/home/wg25r/review_agent/human_reviews_2026/WSCN3Jkebv.md` (4.50): Causal IL framework where reviewers noted "claims outpace evidence" with limited experiments (40 trajectories).

**Low-scoring anchors (≤4):**
- `/home/wg25r/review_agent/human_reviews_2026/9SUbx81pko.md` (3.33): Dual-policy IL+RL with marginal improvements and added complexity.
- `/home/wg25r/review_agent/human_reviews_2026/zYQ9o4e3Pe.md` (4.00): Causal confusion in offline RL with conceptual errors in modeling exogenous variables.

**Positioning:**
This paper is stronger than low-scoring anchors (no fundamental conceptual errors, strong empirical results in 5/6 scenarios) but weaker than high-scoring anchors due to: (1) the Scenario 6 failure that directly contradicts the robustness claim, (2) overclaimed causal framing without structural enforcement, and (3) lack of variance reporting. It aligns most closely with the 4.50 anchor (WSCN3Jkebv) where "claims outpace evidence," though 3CIL has more extensive experiments. The Scenario 6 failure is a notable weakness not present in the 6.0+ anchors, which either have consistent results or acknowledge limitations.

Compared to the 5.00 anchor (ZMDoV1RaXC), 3CIL has stronger empirical results but similar issues with claims exceeding what the data supports. The key differentiator is that 3CIL's failure in Scenario 6 is more damaging to its core claim than the moderate novelty concerns in the 5.00 paper.

**Final Score:** 5.0 — The paper makes real contributions (strong performance in 5/6 scenarios, well-motivated design) but has a significant weakness (Scenario 6 failure contradicting robustness claims) and overclaims the causal nature of the method. This is a borderline paper that would benefit from additional analysis and tempered claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>