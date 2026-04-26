Now I have sufficient calibration data. Let me now synthesize the final review.

**Calibration summary:**

High-scoring anchors:
- MeteoRA (6.2, Accept Poster): LoRA MoE for multi-task, standard formulation, practical contribution, limited deep theory
- ReMoE (6.6, Accept Poster): Novel routing for MoE, strong experiments, theoretical framing somewhat overclaimed but method sound
- NoRM (7.5, Accept Spotlight): Strong PEFT method with solid theory
- VeRA (7.25, Accept Poster): Clean PEFT contribution, strong empirical results

Medium-scoring anchors:
- ELREA (5.8, Accept Poster): LoRA ensemble method, empirical but limited theory
- LoraHub (5.33, Reject): LoRA composition, gradient-free, limited optimization

Low-scoring anchors:
- MoRE (4.0, Reject): LoRA MoE, marginal gains, weak novelty
- MoIN (4.5, Reject): LoRA MoE with clustering, unfair comparisons
- Smoothness/Sparsity MoE paper (3.0, Reject): Questionable theory with math errors
- Zap3nZhRIQ (3.0, Reject): Theory/equation inconsistency

The paper under review (LoRA-Mixer) has:
- A practical and well-evaluated framework with 15 benchmarks and 3 base models
- A real sign inconsistency in the RSL formulation (theoretical claim contradicts the formula)
- However, the gradient derivation is mathematically correct and the method works empirically
- The 48% parameter comparison confounds placement and routing, but this is transparent
- Comparisons against routing-loss baselines (Table 8) are fair (same architecture, same data)
- SSM compatibility is a genuine plus
- Marginal gains over single LoRA on some tasks

This seems like a paper with a genuine technical contribution (LoRA-Mixer framework, RSL loss) and strong empirical results, but with a clear theoretical inconsistency that undermines part of the claimed mechanism. The inconsistency is not fatal — the method works, and the gradient analysis is correct — but the textual claim that "minimizing H(p(x)) promotes specialization" contradicts the formula. This is a significant theoretical presentation issue.

Compared to anchors:
- Better than MoRE (4.0) because the empirical evaluation is more thorough and the framework more complete
- Better than the rejected theory papers (3.0-3.5) because empirical results are solid
- Similar to or weaker than MeteoRA (6.2) — MeteoRA has clean formulation without theory problems, but LoRA-Mixer has more evaluation breadth and SSM compatibility
- Weaker than ReMoE (6.6) because ReMoE has a theoretically cleaner contribution

I'd place this around 5.0-5.5 — the empirical contribution and framework are solid, but the theoretical inconsistency in the core claimed mechanism is a real problem. It's borderline: the paper has enough going for it that it could be a poster, but the theory issue brings it down.

Let me settle on 5.5 — the paper is above average for this venue but has a clear flaw that needs addressing.</think>## Summary

The paper introduces LoRA-Mixer, a framework that routes task-specific LoRA experts through the linear projection layers of attention modules (rather than FFN blocks), paired with a Routing Specialization Loss (RSL) that adds entropy regularization to the standard auxiliary load-balancing loss. LoRA-Mixer is evaluated across 15 benchmarks, 3 base models (including a state-space model), and supports both joint training and plug-and-play composition of externally sourced LoRA modules with as little as 2K routing data.

## Strengths

- **Comprehensive empirical evaluation across architectures**: Testing on Falcon-Mamba-7B (a state-space model) alongside Mistral-7B and LLaMA3-8B provides genuine evidence for the architecture-agnostic claim — a dimension where most LoRA-MoE methods offer no coverage. Gains against baselines like MoLE and MixLoRA are consistent, e.g., +3.79% on GSM8K, +3.95% on ARC-C (LLaMA3-8B).
- **Fair routing loss comparison (Table 8)**: RSL is compared against GMoE, DS-MoE, and AESL under identical conditions (same 2K training data, same architecture). The consistent advantage (+2.77 on CoLA over AESL, +6.86 on HumanEval over GMoE) isolates the contribution of the loss formulation.
- **Practical plug-and-play capability**: The experiment composing externally sourced LoRA modules from LoRAHub with only 2K routing data (Table 3) demonstrates a genuinely useful feature for production deployment, outperforming single LoRA on all five tested GLUE tasks.
- **Clear qualitative evidence of specialization**: Figure 4 shows RSL-optimized routing produces task-differentiated expert loads, whereas auxiliary-loss routing yields flat near-uniform distributions. This directly visualizes the claimed specialization effect.

## Weaknesses

### Fatal
None.

### Major

- **RSL sign inconsistency contradicts stated mechanism**: The RSL loss is defined as $\mathcal{L}_{\mathrm{RSL}} = \alpha \sum_i \bar{p}_i \bar{f}_i - \lambda \mathbb{E}[\mathcal{H}(p(x))]$ (Eq. 5). When minimized as part of the total loss, the $-\lambda \mathbb{E}[\mathcal{H}]$ term *maximizes* entropy — pushing routing distributions toward uniform. Yet the text explicitly states "minimizing H(p(x)) reduces token-conditional uncertainty...directly promoting specialization" (Section 3.3). This is a direct contradiction: the formula does the opposite of what the text claims. The gradient derivation (Eq. 8: $\nabla \mathcal{L}_{RSL} = \alpha \frac{\partial \bar{p}_i}{\partial p_i(x)} \bar{f}_i + \lambda(\log p_i(x) + 1 - \mu)$) is mathematically consistent with the formula and does amplify token-level differences, so the method likely works empirically through this gradient effect. However, the *explanation* of the mechanism (entropy minimization → specialization) is wrong. The correct explanation would be that the entropy term provides gradient curvature that amplifies token-aware variance, not that it minimizes entropy. The information bottleneck claim also rests on the incorrect "minimizing entropy" framing. This undermines the paper's theoretical contribution without invalidating the empirical results.

- **The "48% of parameters" comparison conflates architectural placement with routing mechanism**: LoRA-Mixer applies experts only to attention projection layers, while baselines like MixLoRA and MoLE apply them to FFN layers (or both). The parameter saving comes primarily from the choice of *where* to place experts, not from the routing mechanism itself. The paper presents this comparison as evidence for efficiency of the overall approach but provides no controlled ablation comparing LoRA-Mixer on projection layers vs. FFN layers with matched parameter budgets, making it impossible to attribute performance gains to the routing mechanism versus the placement strategy.

### Minor

- **RSL underperforms at 4K training data**: Table 9 shows RSL worsens performance at 4K data (78.77 vs. 79.14 without RSL), but recovers at 6K+. This non-monotonic pattern is acknowledged only via a reference to Appendix A.16 and weakens the claim that RSL is universally beneficial across data regimes.

- **Marginal improvements over single LoRA baseline**: On LLaMA3-8B, gains over a well-tuned single LoRA are modest (+0.11 on SST2, +0.29 on ARC-E, +0.39 on GSM8K). The larger headline numbers (+3.79% on GSM8K) are comparisons against MoLE/MixLoRA, not against single LoRA. No standard deviations are reported across the 3 runs.

- **Cross-model transfer claim is overstated**: The paper claims Mistral-7B and LLaMA3-8B "have the same architecture," but they differ in vocabulary size (32000 vs. 128256), number of layers, and other details. On ARC-E, transfer performance *decreases* (88.45 → 85.89). The paper frames this as a success ("we outperform the LLaMA3-8B on two of the three tasks") without acknowledging the degradation on the third.

### Trivial
None.

## Nice-to-Haves

- A controlled ablation comparing LoRA-Mixer applied to (a) projection layers, (b) FFN layers, (c) both, with matched total parameters would clarify the contribution of the placement strategy versus the routing loss.
- Side-by-side routing distribution visualizations (with vs. without RSL) on identical inputs, or histograms of per-token routing entropy, would provide direct evidence that the gradient effect produces peaked distributions.
- Running the $+\lambda \mathcal{H}$ vs. $-\lambda \mathcal{H}$ ablation would definitively resolve the sign ambiguity and clarify whether the claimed "minimizing entropy" mechanism or the gradient amplification effect drives performance.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Harsh Critic's claim that "the theoretical analysis is invalid" if the sign is flipped**: As verified above, the gradient derivation (Eq. 8) yields $\lambda(\log p_i(x) + 1 - \mu)$ regardless of whether the entropy term is $+\lambda \mathcal{H}$ or $-\lambda \mathcal{H}$. Both formulations produce the same gradient. The gradient analysis is therefore *not* invalidated by the sign issue — only the textual interpretation of the mechanism is wrong.

- **Harsh Critic's claim that RSL comparison against GMoE/DS-MoE/AESL is unfair because they have "different architectures"**: Table 8 explicitly states "all experiments are conducted with the same training data (2k), and the only difference is the routing loss." The comparison is specifically designed to isolate the routing loss contribution and is fair as described.

- **Harsh Critic's DeepSeek-R1 evaluation critique**: Using an LLM to evaluate MedicalQA free-form answers is a reasonable choice for this domain. This is not a fundamental methodological flaw.

- **Strength Finder's claim about "strong convexity and well-conditioned optimization"**: The strong convexity argument relies on the entropy term, but the Appendix A.1 is inaccessible and the main text only asserts this without proof. Additionally, $-\mathcal{H}$ (convex on simplex) only provides convexity to the *balance* part of the loss, not to the task loss. This strength is weakened by the sign inconsistency.

- **Harsh Critic's "information bottleneck perspective is asserted but not formally developed"**: This is partially addressed by the Appendix references (A.1, A.2), which are in the original submission but stripped by the parser. While the main text should be more self-contained, dismissing the theoretical contributions entirely would be unfair.

- **Nitpicks about formatting and garbled text** (e.g., "the of the null is a result to $\alpha$" on line 82, or "$\mathbf{\Pi}(T)$" on line 94): These are parser artifacts.

## Novel Insights

The paper's most interesting empirical finding is not the headline number but the data efficiency result: RSL with 2K samples (avg 79.26) approaches the performance of auxiliary-loss routing with 8K samples (avg 79.48). If robust, this has practical significance — routing data is often the bottleneck in plug-and-play LoRA composition. The gradient amplification interpretation (which differs from the paper's stated mechanism) suggests a useful design principle: adding an entropy-derived curvature term to routing losses can substitute for larger data budgets, which could inform future MoE routing designs beyond the specific LoRA-Mixer context.

## Suggestions

- **Fix or clarify the RSL sign**: Either correct Eq. 5 to $+\lambda \mathcal{H}$ (matching the "minimizing entropy" claim) and update the gradient derivation accordingly, or revise the text to accurately describe the mechanism as gradient amplification rather than entropy minimization. The current presentation has the formula and the explanation in direct contradiction.
- **Add a placement ablation**: Compare LoRA-Mixer on projection layers vs. FFN layers vs. both, with matched parameters, to isolate the routing loss contribution from the architectural placement effect.
- **Address the 4K anomaly**: Discuss why RSL underperforms at 4K data in the main text rather than deferring to the appendix.

## Calibration Anchors

| Paper | Score | Comparison |
|-------|-------|------------|
| MeteoRA (yOOJwR15xg) | 6.2 | Similar LoRA-MoE multi-task framework; MeteoRA has cleaner formulation without theory errors but less evaluation breadth. LoRA-Mixer has better SSM coverage and routing loss comparison but the RSL sign issue is a clear weakness. |
| ReMoE (4D0f16Vwc3) | 6.6 | Novel MoE routing with strong experiments; ReMoE's contribution is theoretically cleaner. LoRA-Mixer is less novel in routing design but broader in evaluation. |
| MoRE (LWvgajBmNH) | 4.0 | LoRA-MoE for multi-task PEFT, rejected for marginal gains and weak novelty. LoRA-Mixer is clearly stronger — better baselines, broader evaluation, clear routing interpretability. |
| Smoothness/Sparsity MoE (hAyw43h0MH) | 3.0 | Questionable theoretical assumptions; rejected. LoRA-Mixer has actual empirical gains and the theory issue is about textual interpretation, not a fundamental math error. |
| Zap3nZhRIQ (theory/equation inconsistency) | 3.0 | Theory/equation inconsistency led to rejection. LoRA-Mixer's issue is less severe — the gradient math is correct, only the textual mechanism description contradicts the formula. |
| ELREA (l0gZS0sAlf) | 5.8 | LoRA ensemble with empirical gains but limited theory. LoRA-Mixer is comparable in scope but with a clearer routing loss contribution. |

**Score reasoning**: LoRA-Mixer sits between MeteoRA (6.2) and the rejected theory-inconsistent papers (3.0–3.5). It has solid empirical results and a practical framework, but the RSL sign inconsistency is a real theoretical flaw that undermines the claimed mechanism. The gradient analysis is correct, so the method presumably works as implemented, but the paper presents an incorrect explanation of why. This is worse than a minor presentation issue (the claimed mechanism IS the core contribution) but not fatal (the method works and the math is salvageable). Combined with the architectural confound in the 48% parameter comparison, this places the paper below MeteoRA but above the rejected papers. The score should reflect a paper that has genuine empirical value but needs significant corrections to its theoretical presentation.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>