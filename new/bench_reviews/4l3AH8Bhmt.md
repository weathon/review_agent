Now let me search for calibration papers in the human review dataset.Now I have enough information to write the consolidated review. Let me synthesize everything.Now I have sufficient grounding to write the consolidated review.

---

## Summary

This paper investigates *Specificity Failure* in knowledge-editing for LLMs — a phenomenon where editing a single fact causes the model to erroneously apply the edited object to unrelated queries whenever the edited subject appears in context. The authors identify *Attention Drift* (excessive post-edit attention to the edited subject's tokens) as a key trigger, support this via causal-tracing contamination analysis, Pearson correlation, and attention-weight patching, then propose **SADR** (Selective Attention Drift Restriction): a lightweight regularization term added to the editing objective that penalizes attention-head changes that exceed the vanilla model's maximum attention to the subject token. Two new evaluation metrics — Relation Score (RS) and Distract Neighborhood Score (DNS) — are introduced to capture this contextual failure mode, filling a real gap in standard evaluation. Experiments on GPT-J, Llama3, and GPT-NeoX with ROME/MEMIT/PMET show RS/DNS improvements of up to 130.9%/295.8% with modest tradeoffs in generalization.

---

## Strengths

- **Novel evaluation metrics.** RS and DNS expose a genuine, previously under-measured blind spot: existing specificity (NS) metrics do not test what happens when the edited subject or edited sentence appears in context. Table 1 clearly demonstrates catastrophic collapse (RS: 79.73→11.94 for ROME on GPT-J), and these metrics will likely be adopted by the community.
- **Empirical impact is substantial.** Table 3 shows consistent, often large RS/DNS improvements across all three models and editors tested in the main paper, with 95% confidence intervals confirming statistical reliability. In over half of the setups the improvement exceeds 50%.
- **Mechanistic analysis goes beyond pure empiricism.** The paper employs contaminating substitution (Figure 3), Pearson correlation (Figure 4/Table 2), and patching experiments (Figure 5) — three distinct probes — to build a coherent picture of attention drift's role, rather than just proposing a regularizer.
- **Simple, modular method.** SADR plugs into the gradient-based optimization shared by ROME, MEMIT, and PMET without changing the editing architecture, and the ablation (Figure 6) validates the design choice of selective vs. all-head constraint.
- **Trade-off curve is principled.** Figure 7 demonstrates that varying γ in SADR achieves a Pareto-better frontier than adjusting other ROME hyperparameters (steps, LR, ω), giving practitioners a useful control knob.

---

## Weaknesses

### Fatal
*None identified.* The empirical contributions are real and the method is sound; no weakness individually invalidates the paper's core value.

### Major

1. **Causal claim is overstated relative to evidence.** The Abstract, Section 3.5, and Conclusion state that Specificity Failure *"primarily stems from"* Attention Drift. However, Figure 3 shows that replacing 6-layer windows of MLP activations reduces P(o_true) by **4.59%**, while replacing attention activations reduces it by **3.74%**, out of a total drop of **5.26%**. MLP contamination is numerically larger, which the paper acknowledges for the edited layer, yet still concludes attention is the "primary cause." The correlation evidence (ρ ≈ 0.49–0.62 in Table 2) and patching results (Figure 5) establish that attention drift is **functionally important and a meaningful trigger**, but not that it is the *dominant* root cause rather than a downstream consequence of the MLP edit propagating to attention. The paper should replace "primarily stems from" with language like "is strongly mediated by" or "is triggered significantly by," and acknowledge that MLP edits likely cause attention drift, not that attention drift exists independently of the MLP change.

2. **The mechanism linking MLP edits → attention drift is not explained.** The paper establishes that MLP edits correlate with subsequent attention drift and that suppressing attention drift helps, but never explains *why* modifying MLP parameters distorts attention weight distributions. This gap means attention drift could be a symptom — suppressing it addresses the manifestation, not the root. A brief causal analysis or circuit-level hypothesis would substantially strengthen the mechanistic contribution.

3. **Cross-method generality claim is partially deferred.** The introduction and abstract state SADR works "across five editing methods covering all three categories," but Table 3 (the main result) includes only three locate-then-edit methods. Parameter-preserving and meta-learning results are relegated to Appendix E.1 and E.2. While the scope is clearly stated in Section 5.1, it weakens the headline claim in the main body.

### Minor

4. **PS drop in some configurations exceeds the "minimal impact" framing.** The paper claims "less than 3% decrease" in Rewrite/Generation tasks, but ROME on GPT-J shows PS 99.58→96.36 (3.22 pp drop) and ROME on GPT-NeoX shows 98.75→96.13 (2.62 pp). The paper should either tighten the framing or provide aggregate statistics rather than the most favorable reading.

5. **No direct attention visualization before/after SADR.** For a paper whose central claim is that SADR reduces attention drift, there is no heatmap or distribution plot showing the attention patterns improving. Adding such a visualization (even on one example) would make the mechanism more tangible and directly confirmable.

6. **Dependency on the vanilla model during editing is not discussed.** SADR requires a forward pass through the unedited model at each optimization step (to compute M_l(S_j) and the KL target). This is a practical constraint that may not hold for all deployment settings and adds computational overhead. Neither the compute cost nor the assumption of vanilla model availability is discussed.

7. **Head-selection threshold is heuristic and not sensitivity-tested.** The rule "select a head if its post-edit attention to the subject exceeds the maximum over all vanilla-model heads" is reasonable but unexplained. The paper does not report how many heads are selected on average, whether results are sensitive to alternative thresholds, or what the distribution looks like across layers and examples.

### Trivial

8. **Hyperparameter γ guidance is absent.** Figure 6 shows γ effects on ROME/GPT-J but offers no heuristic for choosing γ across different models or editors, which would help practitioners.

---

## Nice-to-Haves

- Attention heatmaps showing drift before editing, after editing (failing case), and after SADR (fixed case) would be highly impactful for reader intuition.
- Per-layer statistics of which heads are selected and how consistently, to assess whether SADR targets a stable set or varies unpredictably.
- An experiment that also applies naive uniform attention KL (across all heads, not subject-selective) as a baseline beyond the with/without head selection in Figure 6, to further justify the selective design.
- Sequential editing evaluation to understand if attention drift compounds with multiple edits (explicitly scoped out in this paper, so this is a future direction, not a requirement).
- Reporting computational overhead of SADR relative to the base editing method.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

**[Harsh Critic, Point 2 — Scope Creep]** The criticism that SADR is "evaluated entirely on the same bespoke failure mode it is designed around" and therefore over-generalizes "specificity" is partially valid regarding abstract framing (retained as part of Major weakness #1 above). However, the deeper structural charge that RS/DNS are not "real" specificity metrics is removed: the paper explicitly defines Specificity Failure as a contextual phenomenon in Section 2.1 and is transparent that NS (standard specificity) is largely unaffected. NS remaining stable when the editing subject doesn't appear in context is *expected behavior*, not a flaw. The paper's narrow scope is honest, even if the abstract framings is mildly overreaching.

**[Harsh Critic, Point 3 — Fairness of cross-method comparison]** The hard-rule about unfair comparisons favoring baselines does not apply here (this is a fairness-to-SADR concern, not favoring baselines). However, the claim that "there is little detail on whether each baseline was retuned comparably" is removed as a reproductibility nitpick: SADR adds only one regularization term γ on top of the existing optimizer, so the comparison is structurally fair by design.

**[Spark, Point 2 — No sequential/batch evaluation]** Removed per the soft rule on scope: Section 2.1 explicitly scopes to single-edit scenarios with clear justification. Sequential editing results would be a welcome extension but are not a core flaw.

**[Spark, Point 3 — No general NLP benchmark evaluation]** Moved to Nice-to-Have: the paper does report commonsense reasoning and perplexity in Appendix E.1. Not reporting these in the main body is a presentation choice, not a missing result.

**[Human Finder, Point 4 — Limited exploration to single edits]** See above (scoped explicitly).

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the following: SADR's core design insight — that the same MLP-parameter edit that writes new knowledge simultaneously distorts attention weight distributions, and that this distortion (not the MLP update itself) is the proximate cause of contextual specificity failures — suggests that the attention module acts as an **amplifier and context-spreader** for the MLP's local change. This implies a broader principle: any editing method that changes hidden-state distributions at the edited token position risks "leaking" that change into the attention mechanism and thereby contaminating unrelated but co-occurring queries. This framing, if stated precisely, could generalize beyond ROME-style editing and motivate attention auditing as a routine diagnostic in editing pipelines. The paper hints at this but does not fully articulate the mechanistic story.

---

## Suggestions

1. **Revise causal language throughout.** Replace "primarily stems from Attention Drift" with claims like "is substantially mediated by Attention Drift, which emerges as a downstream effect of MLP parameter changes." This is more accurate to the evidence and harder to dispute.
2. **Add one paragraph in Section 3 or 4 proposing a mechanistic account of why MLP edits propagate to attention drift.** Even a speculative circuit-level account (e.g., "the edited MLP hidden state is read by attention keys across layers, creating an information feedback loop") would elevate the analytical contribution.
3. **Move Table 3's partial results for parameter-preserving and meta-learning methods into the main body,** even as a summary row, to directly support the cross-paradigm claim in the introduction.
4. **Add PS aggregate framing.** Provide the harmonic mean of ES, PS, NS, RS, and DNS as Avg.S prominently (this is already in Table 3) and use it as the headline metric for the "minimal impact" claim rather than quoting only the favorable cases.
5. **Include a vanilla-model-forward-pass compute table** showing the wall-clock overhead of SADR relative to base editing.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Score | Decision |
|---|---|---|---|
| FiNE (5xP1HDvpXI) | Knowledge editing locality, causal localization | 6, 8, 6, 6 (avg 6.5) | Accept Poster |
| PRUNE (bfI8cp8qmk) | Sequential model editing, regularization | 5, 6, 6, 8 (avg 6.25) | Accept Poster |
| IFMET (INFfvQArFY) | Multi-hop editing, mechanistic analysis | 6, 8, 5, 6 (avg 6.25) | Reject |
| FAI (W6yIKliMot) | Attention intervention to fix LLM failure mode | 8, 6, 6, 6 (avg 6.5) | Accept Poster |
| Mech. Unlearn (vsU2veUpiR) | Mechanistic editing/unlearning, presentation issues | 8, 3, 5, 5 (avg 5.25) | Reject |

The paper under review is most comparable to FiNE and FAI (both accepted at ~6.5): it identifies a targeted failure mode, proposes a mechanistic analysis, and delivers a simple effective method with strong empirical results. Its weaknesses — overstated causal claim, partial cross-method evidence in main body, missing attention visualizations — are moderate and addressable. Unlike the rejected Mechanistic Unlearning paper (which suffered from severe presentation failures) or IFMET (which had experimental gaps), this paper is well-organized and the empirical contributions are clear and strong. The key differentiator from a higher-scoring paper is the imprecision of the causal claim (where Figure 3 actually shows MLP contamination > attention contamination, contradicting "primarily stems from attention"), and the lack of an explanatory mechanism linking the MLP edit to the observed attention drift. On balance, this positions the paper slightly below FiNE/FAI in analytical rigor, but the practical contributions (new evaluation metrics RS/DNS, substantial improvements on a documented real problem) justify acceptance at poster level.

**Originality:** Good — new evaluation metrics and mechanistic framing for contextual specificity failure are genuine novelties.  
**Importance:** High — contextual specificity failure is a real barrier to practical knowledge editing.  
**Claim support:** Moderate — empirical results are strong; causal story is partially oversold relative to evidence.  
**Experimental soundness:** Good — multiple models, methods, ablations, confidence intervals; some gaps in alternative baselines and visualizations.  
**Clarity:** Good — paper is readable and well-organized.  
**Community value:** High — RS/DNS metrics and SADR regularizer are directly adoptable by knowledge editing practitioners.

**Score: 6.0**

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>