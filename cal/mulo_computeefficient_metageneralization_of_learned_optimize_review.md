=== CALIBRATION EXAMPLE 15 ===

# Final Consolidated Review
## Summary
This paper applies maximal update parameterization (µP) to two learned optimizer architectures, derives the corresponding update rules, and proposes a simple multi-width meta-training recipe. The main empirical finding is convincing: learned optimizers trained under the proposed µ-parameterization are dramatically more stable and generalize much better than standard-parameterized learned optimizers to wider unseen networks, and they also show intriguing empirical gains on deeper networks and much longer horizons.

## Strengths
- **Clear and specific technical contribution: deriving µ-parameterization for learned optimizers rather than only for hand-designed optimizers.** Section 4 does more than simply re-run prior experiments under µP: it adapts the optimizee initialization, pre-activation multipliers, and the LO update scaling for two concrete architectures (small_fc_lopt and VeLO), and states explicit propositions for when these satisfy µP desiderata.
- **The empirical effect on width generalization is large and practically meaningful.** The paper’s strongest evidence is in Figures 3–4 and Table 1: µLOs remain stable and continue improving on widths far beyond meta-training, while SP learned optimizers often diverge or stop making progress. This is not a subtle gain.
- **The evaluation probes several axes of distribution shift rather than only one proxy task family.** Although meta-training is only on MLPs, evaluation spans 35 tasks including MLPs, ViTs, and decoder-only LMs, with width, depth, and horizon shifts. That breadth makes the observed cross-task robustness more compelling than a narrow in-family study.
- **The paper includes a concrete mechanistic check rather than only end-task losses.** Figure 2 tests pre-activation stability across widths and shows that µ-parameterized setups behave harmoniously while SP baselines blow up, which is directly relevant to the paper’s stated rationale.
- **The baseline setup is stronger than what many LO papers use.** The paper compares against both SP learned optimizers and extensively tuned hand-designed baselines (AdamW and µAdam, with 500+ configs per task), which helps establish that the method is not merely beating weak baselines.

## Weaknesses

### Major:
- **The paper does not fully disentangle the benefit of µ-parameterization from the benefit of the proposed multi-width meta-training recipe.**  
  The method as evaluated combines two ingredients: (i) µ-parameterization and (ii) a multiple-width meta-training distribution. Section 5.2.1 compares single-width vs multi-width training *within* µP, and later sections compare µLO_M against SP LO_M, but there is no clean component-wise ablation of the individual µP modifications or a more systematic decomposition of how much each ingredient contributes. As a result, the headline gains are real, but the causal attribution to the full proposed recipe versus specific components remains somewhat under-resolved.
- **The claims around deeper-network and longer-horizon generalization are interesting but mechanistically under-supported.**  
  The paper is careful to say these findings are “purely empirical,” which is appropriate, but Section 5.2.4 still makes these results sound more explanatory than the evidence justifies. The evidence consists mainly of loss curves; there is no deeper analysis of optimizer state magnitudes, gradient norms, or controlled tests of the hypothesis that improved pre-activation stability is what drives the depth/horizon benefits. These results are promising, but they should be presented more clearly as empirical observations rather than as an understood extension of the core method.
- **Cross-architecture generalization is impressive but still rests on a substantial train-test domain gap that is not deeply analyzed.**  
  The paper explicitly meta-trains learned optimizers only on MLP image-classification tasks (“these tasks only include MLPs”) and then evaluates on ViTs and LMs. That is a strength in one sense, but it also leaves an unanswered question: which parts of the learned policy transfer across architectures, and how much of the observed robustness is due to generic scale stabilization versus architecture-aware optimization behavior? A small study including non-MLP meta-training tasks would have materially strengthened the generalization claim.

### Minor
- **The theoretical treatment relies on strong assumptions and is narrower than the framing may suggest.**  
  Propositions 4.1 and 4.2 assume LLN-style alignment conditions and establish sufficiency under those assumptions. This is acceptable, but it means the theory is not a full characterization of learned optimizer dynamics; in particular, the surprising depth and horizon results are outside the theory’s scope. The paper mostly acknowledges this, but the overall framing occasionally reads more broadly than what is formally established.
- **The “compute-efficient” / “zero extra computational cost” framing needs sharper qualification.**  
  The paper’s intended meaning appears to be “no extra cost relative to SP learned optimizers under the same meta-training budget,” and the abstract/body repeatedly compare FLOP-matched learned optimizers. That said, the wording can still be read too broadly. Since the method still requires substantial meta-training and uses a nontrivial recipe, the claim should be phrased more precisely as *no additional cost relative to matched SP-LO meta-training*, not as zero cost in an absolute sense.
- **Table 1’s average-rank summary is useful but hides effect sizes.**  
  The paper does provide loss curves elsewhere, so this is not a fatal issue, but rank aggregation can compress differences and make some margins look more decisive than they are. A complementary aggregate based on normalized loss gaps would make the cross-task summary stronger.
- **The evaluation scale, while substantial for an academic study, still stops short of the largest modern regimes.**  
  The paper itself acknowledges this in Section 6. Given the paper’s framing around scaling and deployment relevance, the absence of tests beyond the reported width ranges modestly limits how far one can extrapolate.

### Trivial
- **Validation/test metrics would complement the training-loss results.**  
  The paper optimizes and reports training loss throughout, which matches the learned optimization objective, but reporting downstream accuracy/perplexity where applicable would help assess whether the gains reflect better useful optimization rather than only lower training loss.

## Nice-to-Haves
- Add a **component-wise ablation** isolating optimizee initialization scaling, pre-activation multiplier scaling, and optimizer update scaling.
- Add analyses of **gradient norms / optimizer-state magnitudes / update norms vs width** to better support the proposed mechanism.
- Include a **small mixed-architecture meta-training study** (e.g., adding one ViT or LM family) to test whether the cross-architecture gains are robust or incidental.
- Clarify the compute claim by explicitly tabulating the **meta-training FLOP budget** for SP LOs vs µLOs in the main paper.
- Report **validation/test performance** in addition to training loss on the main OOD tasks.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Unfair comparison because hand-designed baselines are per-task tuned while learned optimizers are trained only on MLPs.”**  
  Removed as a weakness under the stated rules. The asymmetry favors the baselines, not the proposed method, so it actually strengthens the paper’s case rather than weakening it.
- **Claims that the paper is under-specified because it does not detail the exact ES variant or trivial implementation details in the main text.**  
  Removed as a reproducibility nitpick. The paper points to appendix sections for meta-training details, which is standard.
- **Criticism that the paper should compare to additional external methods/related work such as other parameterization families.**  
  Removed because missing related-work comparisons cannot be verified here and are not required to assess the paper on its own stated contribution.
- **Doubt about existence/release/availability or verifiability of cited tools/models/references.**  
  Removed by rule.
- **Formatting/parser issues in the extracted text.**  
  Removed as non-paper artifacts.

## Novel Insights
The most interesting synthesis across the paper and reviews is that the strongest contribution is not merely “µP helps learned optimizers,” but that **µP appears to convert a learned optimizer from a scale-fragile controller into one whose policy transfers across substantial optimizee shifts without changing the learned architecture itself**. The results suggest that a large part of learned-optimizer meta-generalization may be bottlenecked by the parameterization of the optimizee-update interface rather than only by the expressive power or scale of meta-training. At the same time, the paper does not yet fully resolve whether the surprising gains on depth and long horizons arise from the same mechanism or from a broader regularizing effect of the µ-parameterized training regime.

## Suggestions
- Add a targeted ablation that isolates each µP ingredient and separates it from the multi-width meta-training recipe.
- Reframe the depth and horizon results explicitly as **strong empirical observations** rather than partially explained consequences of the theory.
- Include one or two diagnostic plots on **update magnitude scaling, gradient norms, and optimizer-state statistics** across widths to substantiate the stabilization mechanism.
- Strengthen the cross-architecture claim with a small experiment meta-training on a mixed task distribution including at least one non-MLP family.
- Tighten the wording around compute: say **“no extra meta-training cost relative to FLOP-matched SP learned optimizers”** rather than “zero extra cost.”



# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 2.0]
Average score: 5.0
Binary outcome: Accept
