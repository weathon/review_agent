Now I have all the calibration anchors and paper details needed. Let me compile the final review.

## Summary

The paper identifies and mitigates Specificity Failure in knowledge editing of LLMs — a severe failure mode where edited models incorrectly output the edited object when the edited subject appears in unrelated contexts. Through causal tracing and correlation analysis, the paper attributes this failure to an "Attention Drift" phenomenon, where attention heads excessively focus on edited-entity tokens. The authors propose Selective Attention Drift Restriction (SADR), a lightweight regularization term added during the v\* optimization of locate-then-edit methods, which selectively constrains over-attending heads. SADR achieves substantial improvements in specificity metrics (RS, DNS) while preserving edit success rates above 99% across multiple models.

## Strengths

- **The problem identification is genuinely important and quantitatively striking.** Table 1 shows that existing editing methods (ROME, MEMIT, PMET) cause RS to plummet from ~80% to ~12–28% on GPT-J, while NS remains at ~80%. This reveals a severe, previously under-measured failure mode. The gap between NS and RS/DNS demonstrates that standard evaluation systematically misses this problem.

- **The new evaluation metrics (RS, DNS) are a concrete, adoptable contribution.** Section 2.1 defines Relation Score and Distract Neighborhood Score that explicitly test specificity when the edited subject appears in context. Figure 2 provides clear examples. These metrics capture a real failure mode that NS alone misses, and their adoption would improve future knowledge editing evaluation.

- **The Contaminating Substitution experiment (Section 3.2, Figure 3) provides non-obvious evidence.** Replacing attention activations from the edited model into the vanilla model decreases correct-answer probability by up to 3.74%, comparable to the 4.59% from directly edited MLP activations (vs. 5.26% total). This is a non-trivial finding since editing methods only modify MLP parameters, yet attention emerges as a near-equal contributor to failure.

- **SADR is conceptually simple and practically effective.** Adding a selective KL-divergence regularization term to the existing optimization (Equation 2) requires no architectural changes. Table 3 shows consistent improvements: ROME+SADR on GPT-J improves RS from 11.94 to 57.07; on Llama3, RS improves from 29.38 to 65.88 with ES above 99%.

- **Selective head restriction is validated by ablation (Figure 6).** Restraining only over-attending heads consistently outperforms restraining all heads across γ values on both Edit Success and Specificity, confirming that not all attention drift is harmful and the selection criterion is well-founded.

## Weaknesses

### Fatal

None.

### Major

- **The causal claim that attention drift is "the primary trigger" for specificity failure is overstated relative to the evidence.** Section 3.5 concludes that "the max attention drift at the edited token position among heads is a primary trigger," and Section 3.2 states "the primary cause of specificity failure is the attention module mishandling the information." However, the correlational evidence (Table 2) shows ρ=0.49–0.62, explaining only 24–38% of variance — over 60% remains unaccounted for. The Contaminating Substitution experiment (Section 3.2) shows attention carries contaminated information, but since attention is downstream of the edited MLP, this contamination could be a passive conduit rather than an independent cause. The patching experiment (Figure 5) provides more direct causal evidence (replacing 10 layers of attention weights yields large recovery), but patching 10 consecutive layers is a very aggressive intervention that could mask other contributing factors. The evidence establishes attention drift as *a significant factor*, not clearly *the primary trigger*. This matters because the method's effectiveness ceiling depends on whether attention drift is the root cause or merely a symptom of MLP weight changes.

- **The claim of broad applicability across all three editing paradigms is unsupported in the main text.** The abstract states SADR works on "five editing methods covering all three categories" (locate-then-edit, parameter-preserving, meta-learning). However, SADR is defined as a modification to the v\* optimization objective (Equation 2), which is specific to the locate-then-edit framework. The main results (Table 3) only cover ROME, MEMIT, and PMET — all locate-then-edit methods. The paper acknowledges on line 107 that "We primarily focus on locate-then-edit knowledge editing," and refers to Appendix E.2 for other paradigms, but never explains in the main text how SADR adapts to methods that lack the v\* optimization step. If the adaptation requires fundamentally different implementation, the claim of "flexible" applicability across categories is misleading as stated.

### Minor

- **SADR slightly worsens DNS in some configurations.** In Table 3, MEMIT+SADR on GPT-J decreases DNS from 49.35 to 47.35, while significantly improving RS (27.75 → 81.44). GPT-NeoX DNS values remain very low even with SADR (8.84 for ROME, 16.22 for MEMIT). The paper does not discuss or analyze these cases where SADR fails to improve or slightly degrades a metric, which would help readers understand the method's limitations.

- **The relative improvement framing in the abstract can be misleading.** The abstract reports "improvements of up to 130.9% and 295.8%," which are relative improvements on metrics with very low baselines (e.g., RS=11.94 → 57.07 is a ~378% relative improvement but a ~45-point absolute improvement on a 0–100 scale). While technically accurate, this framing inflates perceived impact; reporting both relative and absolute improvements would be more transparent.

- **The patching experiment uses a large intervention window.** Figure 5 patches 10 consecutive layers of attention weights, which is a substantial intervention. A more fine-grained patching analysis (e.g., per-layer or with smaller windows) would strengthen the causal claim and help identify which specific layers matter most.

- **The Contaminating Substitution analysis does not fully disentangle cause from conduit.** Section 3.2 shows attention activations carry contamination comparable to MLP activations. However, since attention receives inputs from the edited MLP layer, the attention contamination could be a downstream effect rather than an independent cause. A control experiment — e.g., substituting attention outputs while keeping MLP outputs from the vanilla model — would better isolate attention's independent contribution.

### Trivial

- None.

## Nice-to-Haves

- Multi-edit evaluation to test whether attention drift compounds with sequential edits — the paper explicitly scopes to single-edit (Section 2.1), but the practical relevance of knowledge editing demands sequential/batch settings.
- Analysis of SADR failure cases: characterizing when DNS worsens or RS improves only modestly (e.g., by relation type or subject frequency) would reveal whether SADR addresses a subset of the problem.
- Qualitative examples showing edited model outputs with and without SADR, beyond probability metrics, to make the failure mode and mitigation tangible.
- Investigation of optimization dynamics under dynamic head selection — tracking how H_l(S_j) changes across steps would verify stability.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Dynamic head selection instability (Harsh Critic, Evidential):** The reviewer speculates that dynamic head selection could cause optimization instability or oscillation. However, the method works consistently across 9 model-method combinations in Table 3, and no evidence of instability is observed. This is a speculative concern without empirical support — removed as a Major weakness, demoted to Nice-to-Have.

- **KL divergence direction not ablated (Harsh Critic):** The paper uses D_KL(W || W\*) which penalizes the edited model attending where the vanilla model did not — the direction that directly serves the stated goal of preventing excessive attention. While an ablation would be ideal, the intuitive correctness of the choice and the method's empirical success make this a trivial concern.

- **Aggregation methods in Table 2 not controlled (Harsh Critic):** The reviewer notes that different factors use different aggregation (max vs. sum). While technically true, the direction of results is consistent across factors, and the paper's key finding — that max-head drift on subject tokens (Factor 1) has higher ρ than cumulative drift (Factor 3) — is robust. This is a minor presentation issue.

- **Formatting of Table 3 (Harsh Critic):** The reviewer notes the table is difficult to parse. This is a parser artifact from the extracted text, not a paper quality issue.

- **Missing related works (Harsh Critic):** Per rules, we do not flag missing related works without external confirmation.

- **Computational overhead not reported (Harsh Critic):** SADR adds a forward pass through the model at each optimization step to compute attention weights. This is trivial implementation overhead and not a substantive concern.

- **Missing appendix proofs/results (Harsh Critic):** The parser strips appendix sections from all papers; they exist in the original submission.

- **No comparison with aggressive ω tuning (Harsh Critic):** Section 6.2 actually does compare SADR's trade-off against varying ω, showing SADR provides a better curve. The reviewer's concern about "limited range" is not substantiated — the comparison exists and is favorable.

- **Overclaiming "five editing methods" when only locate-then-edit shown (Strength Finder filter):** The strength finder lists "Comprehensive cross-method and cross-model evaluation" as a supporting strength, but since the main text only shows locate-then-edit results, this strength is partially inflated — the "five editing methods" claim relies on appendix content not visible in the main paper. I've kept the strength about cross-model evaluation but weakened the cross-method claim.

## Novel Insights

The paper reveals an underappreciated asymmetry in knowledge editing failures: while the MLP module is the one being directly edited, the attention module emerges as a near-equal contributor to specificity degradation. This finding — that editing one module (MLP) causes cascading failure in another (attention) that is arguably more responsible for the downstream error — has implications beyond this paper. It suggests that future knowledge editing methods should not only regularize the edited module but also monitor and constrain the attention module's response to parameter changes, which is a distinct design principle from simply constraining the magnitude of the edit itself.

## Suggestions

- Tone down the "primary trigger" language to "a significant contributing factor" throughout the paper, or provide additional evidence (e.g., per-layer patching, controlled experiments disentangling attention's independent contribution from MLP-driven contamination) to support the stronger claim.
- Include a brief explanation in Section 4 or 5 of how SADR adapts to non-locate-then-edit methods, or move the key appendix results (E.2) into the main text to support the three-category claim.
- Add a paragraph analyzing cases where SADR underperforms or worsens metrics (e.g., GPT-J MEMIT DNS, GPT-NeoX DNS), discussing whether these are tied to specific model architectures, edit layers, or relation types.

## Score and Decision

**Calibration anchors:**

- AlphaEdit (avg 8.0, Oral): Null-space projection for knowledge editing specificity, with formal theoretical guarantees and single-line implementation. Much stronger theoretical contribution than the current paper.
- Editing Overfit / EVOKE (avg 7.33, Spotlight): Identifies editing overfit problem, proposes LTI solution, comprehensive experiments. Similar problem-solution structure, slightly stronger experimental coverage.
- Attention Satisfies (avg 6.0, Poster): Attention-based probing of factual errors with correlation-based claims, but practicality concerns and overclaiming about causal mechanism. Similar overclaiming pattern to the current paper, but the current paper is stronger because it proposes a working mitigation rather than just a diagnostic.
- Representation Shattering (avg 4.6, Reject): Shows KE distorts representations beyond targeted facts, but limited models and mostly qualitative analysis. Weaker than the current paper in experimental rigor.
- IRIS (avg 2.5, Withdrawn): Overclaims causal relationships from co-occurrence with no mechanism validation. Far weaker than the current paper.

The current paper sits between "Attention Satisfies" (6.0) and "Editing Overfit" (7.33). It is stronger than "Attention Satisfies" because it proposes and validates a working solution, not just a diagnostic, and the problem is more severe and clearly demonstrated. It is weaker than "Editing Overfit" because the causal mechanism claim is less rigorously supported and the applicability across editing paradigms is not demonstrated in the main text. The overclaiming of "primary trigger" based on moderate correlations is the main factor keeping this below 7.

MY FINAL SCORE: <pineapple>6.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>