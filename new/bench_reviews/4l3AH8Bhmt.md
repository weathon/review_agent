Now I have enough context to write a comprehensive meta-review. Let me consolidate my analysis.

The paper under review identifies "Attention Drift" as a mechanism for specificity failure in knowledge editing, and proposes SADR as a mitigation. Let me now carefully assess the key claims and weaknesses.

**Key assessment:**

1. **The paper identifies a real, important problem** — specificity failure when edited subjects appear in context. Table 1 is quite striking: RS drops from ~80 to ~12-28, DNS drops from ~62 to ~30-40. This is a real and severe issue.

2. **The causal story is partially supported but overstated** — Correlation (ρ=0.49, 0.62) is moderate, not strong. Patching attention helps, but doesn't cleanly separate attention drift from hidden-state drift. The paper claims "primary cause" but evidence is more consistent with "attention is an important lever."

3. **The evaluation metrics (RS, DNS) are legitimate stress tests** but the paper frames them as measuring broad "specificity failure" when they are more adversarial/narrow. NS stays stable around 80%. This isn't misleading per se — the RS/DNS scenarios are valid practical concerns — but the "over 50% failure" framing leans heavily on these new metrics.

4. **SADR is simple but effective** — The method adds a KL regularizer on selected attention heads during editing. It substantially improves RS and DNS while keeping ES/PS high. This is a genuine contribution.

5. **The absolute performance remains far from vanilla** — Even with SADR, RS often stays well below the unedited model's levels (e.g., ROME on Llama3: 92.66 → 41.06 with SADR vs 29.38 without). The problem is only partially solved.

Let me now formulate the final review.

**Calibration:**
- FiNE (knowledge editing, neuron-level): Scores 6,8,6,6 → Accept (Poster). Similar: mechanistic analysis of KE, proposes targeted intervention, moderate novelty.
- PRUNE (knowledge editing, sequential): Scores 5,6,6,8 → Accept (Poster). Plugin method for existing KE, good empirical results.
- BaFT (knowledge editing, representation fine-tuning): Scores 5,8,3,6 → Accept (Poster). Novel method with editing-locality tradeoff analysis.
- iReVa (knowledge editing, key-value adaptors): Scores 3,6,5,6 → Reject. Flawed experimental setup, methodological issues.
- "How new data permeates" (priming in LLM learning): Scores 8,8,6,8 → Accept (Spotlight). Novel phenomenon, strong analysis, practical mitigation.

The paper under review is similar in quality to FiNE and PRUNE — it identifies a real phenomenon, provides mechanistic analysis, proposes a practical fix, and evaluates across multiple models. Its weaknesses are: (1) the causal mechanism claim is overstated relative to evidence, (2) RS/DNS improvements are real but remaining gaps are large. It's stronger than iReVa (rejected) but not at spotlight level like "How new data permeates."

I'd place this in the 5.5-6.5 range, leaning toward 6.</think>## Summary

This paper identifies "Specificity Failure" in knowledge editing—the phenomenon where edited LLMs inappropriately apply edited knowledge to unrelated contexts containing the edited subject—and traces it to an "Attention Drift" mechanism where certain attention heads assign excessive scores to the edited subject token. Through causal tracing, correlation analysis, and patching experiments, the authors demonstrate the role of attention drift and propose Selective Attention Drift Restriction (SADR), which adds a KL-divergence regularization term during editing to constrain attention heads exhibiting excessive drift. Experiments across five LLMs (1.1B–20B) and five editing methods show substantial improvements in Relation and Distract Neighborhood specificity metrics with minimal degradation of edit success.

## Strengths

- **Clear identification and formalization of a real, underappreciated problem.** Table 1 makes a compelling case: when the edited subject appears in context (RS/DNS metrics), performance collapses to near-random levels (RS around 12–28% after editing), even though standard neighborhood specificity (NS) remains stable. This is a practically important stress test that most prior KE evaluations miss.

- **Multi-pronged mechanistic analysis.** Sections 3.2–3.4 use complementary methods—contaminating substitution (Fig. 3), correlation analysis (Fig. 4, Table 2), and patching experiments (Fig. 5)—to build a case that attention activations at the last token position are important levers for specificity behavior. The patching result (739% relative improvement on Relation) is particularly striking.

- **Simple, interpretable, and broadly applicable method.** SADR adds a KL penalty only on heads whose attention to the subject exceeds the maximum baseline attention, requiring one hyperparameter (γ). It can be layered on any locate-then-edit method and is evaluated across three editing paradigms and five model architectures, showing consistent improvements.

- **Strong experimental coverage.** Table 3 provides results for 9 model×method combinations with confidence intervals. The ablation in Fig. 6 validates selective vs. all-head restriction, and Fig. 7 shows the generalization–specificity trade-off is more favorable with SADR than with alternative hyperparameter tuning.

- **Honest reporting of trade-offs.** The paper acknowledges that PS drops by a few points (e.g., GPT-J/ROME: 99.58→96.36) and argues that the specificity gains justify this. The trade-off discussion in Section 6.2 and Appendix is a responsible inclusion.

## Weaknesses

### Fatal

None.

### Major

- **The causal mechanism claim is overstated relative to the evidence.** The paper repeatedly frames attention drift as *the* "primary cause" or "trigger" of specificity failure (Abstract: "primarily stems from"; §3.5: "primary cause"; Conclusion: "stem from the Attention Drift phenomenon"). However, the evidence is primarily correlational and interventional in a way that doesn't cleanly isolate attention drift from hidden-state/MLP drift. The contaminating substitution (Fig. 3) shows that both MLP and attention activations matter; the correlation analysis (ρ=0.49, 0.62) is moderate; and the patching experiment restores vanilla attention patterns but doesn't rule out that the same improvement could be achieved by patching hidden states or value vectors. The paper would be more accurate—and its interpretability contribution stronger—if it claimed that attention drift is a *key lever* for specificity rather than asserting it as the primary mechanistic cause. This overclaim doesn't invalidate the empirical method, but it undermines the mechanistic interpretability narrative that positions the paper.

- **Absolute specificity performance after SADR remains far from the unedited baseline, and the paper tends to report relative improvements without adequate acknowledgment of remaining gaps.** For example, ROME+SADR on Llama3 raises RS from 29.38 to 41.06 (a ~40% relative gain), but the unedited baseline is 92.66—still less than half recovered. GPT-NeoX/ROME DNS goes from 8.84 to 19.45 vs. a vanilla model's 58.34. While SADR substantially mitigates the problem, claiming it "significantly mitigates Specificity Failure" (as the abstract does) overstates the practical resolution. The remaining gap is large enough that the problem is far from solved.

### Minor

- **No comparison to alternative regularization strategies that target non-attention components.** The ablation in §6.1 compares selective vs. all-head attention restriction, and §6.2 compares varying γ with varying other ROME hyperparameters. But there is no comparison against, e.g., penalizing hidden-state drift at the last token, or penalizing logit drift on specificity prompts—alternatives that might achieve similar gains via different mechanisms. This makes it harder to assess whether SADR's success reflects a mechanistically insightful design choice versus the general benefit of "don't change too much near the subject."

- **The paper lacks a direct comparison against an unedited model's behavior on analogous distractor constructions.** The DNS and RS metrics explicitly include the edited fact or subject. It is plausible (and not tested) that unedited models also show degraded performance when a strongly associated prior sentence is prepended (recency bias). Knowing how much of the measured failure is genuinely edit-induced versus ordinary context sensitivity would clarify the practical severity.

- **Ablations are limited to GPT-J/ROME.** Fig. 6 and Fig. 7 only report ablations on one model–method pair. The claim that selective head restriction is universally beneficial (generalized from one setup) is not substantiated across the full range of models and editors.

- **The SADR head selection threshold is heuristic.** A head is selected if its post-edit attention to the subject exceeds the maximum pre-edit attention across heads at that layer. No sensitivity analysis or justification for this specific criterion is provided, leaving it unclear whether other reasonable thresholds would perform similarly.

### Trivial

- **Table 3 formatting is confusing**: two lines per method (baseline vs. +ours) are rendered with `<br>` tags, making the exact deltas hard to read at a glance.

## Nice-to-Haves

- Evaluation on multi-hop or more naturalistic QA benchmarks (e.g., MQuAKE) to verify SADR's behavior in compositional reasoning scenarios.
- Sequential/batch editing evaluation, since real-world use requires multiple edits.
- Attention heatmap visualizations before/after SADR to directly confirm the mechanism works as described.
- Comparison with increasing the existing KL weight ω as a baseline, to isolate whether SADR's value comes from targeting attention specifically versus simply adding more regularization.

## Removed Points

- *Harsh critic's point about RS/DNS being "self-referential" or "narrow" metrics that don't capture true specificity.* RS and DNS are standard in recent KE literature (Hoelscher-Obermaier et al., 2023; Yao et al., 2023); they test a real failure mode where edited models incorrectly apply edited knowledge. This is not self-referential—it captures a genuine, practically relevant issue. **Weakness downgraded to minor** since the severity framing is somewhat overstated in the paper but the metrics themselves are valid.

- *Harsh critic's claim that the "over 50%" figure is misleading because it's only from RS/DNS, not NS.* The paper is clear about what the 50% figure refers to (Section 1: "a 6B GPT-J model after the knowledge editing can exhibit severe Specificity Failure in over 50% of cases regarding factual statements"), and Table 1 shows the RS/DNS numbers transparently. The "over 50%" is accurate for RS, even if NS is less affected. However, the broad claim that editing "severely degrades existing knowledge and capabilities" is somewhat overstated given NS stability. **Partial weakness retained above in Major point.**

- *Human finder's point about outdated models (GPT-J, GPT-NeoX).* The paper tests on Llama3-8B (a current model) alongside GPT-J and GPT-NeoX. Testing on 70B+ models would strengthen the paper, but the 1.1B–20B range is reasonable for KE research and includes a recent model. This is a nice-to-have, not a major weakness.

- *Human finder's point about lack of downstream task evaluation.* The paper does report commonsense reasoning and perplexity results in Appendix E.1 and includes fluency scores in Table 3. While broader evaluation would be welcome, the paper already goes beyond pure KE metrics. This is a nice-to-have.

- *Neutral reviewer's claim that the method is "somewhat incremental."* SADR is indeed a straightforward application of selective KL regularization, but the key contributions are the identification/diagnosis of the attention drift phenomenon and the head selection criterion. The method's simplicity is arguably a strength rather than a weakness in this context. **Removed.**

- *Spark's suggestion to compare with increasing ω.* This is a sensible request but goes beyond what's standard for an ablation in this area. The trade-off comparison in Fig. 7 partially addresses this. Listed as a nice-to-have.

## Novel Insights

The identification of a specific, measurable attention drift pattern—where post-edit attention heads over-focus on the subject's last token—that correlates with specificity failure (ρ~0.5–0.6) and can be partially reversed by patching (Fig. 5) is a genuinely novel diagnostic insight. The finding that selective head restriction outperforms all-head restriction (Fig. 6) is particularly interesting, as it suggests that not all attention drift is harmful—some may reflect legitimate information routing for the new fact—which nuance goes beyond a simple "less change = better" story.

## Suggestions

- Temper the causal language throughout: replace "primary cause" with "key factor" or "important mechanism," especially in the Abstract and Conclusion.
- Report absolute performance alongside relative improvements. For each model×method pair, show how far SADR still falls short of the vanilla baseline on RS/DNS, to give readers an honest picture of the remaining gap.
- Add a comparison with simply increasing the existing KL weight (ω) or a generic hidden-state regularization, to demonstrate that targeting attention specifically provides benefit beyond "constrain changes generally."
- Include at least one ablation (head selection or γ sensitivity) on a second model–method pair to support generalizability claims.

## Score and Decision

**Calibration anchor papers:**

- **FiNE** (knowledge editing, neuron-level, mechanistic analysis): Scores 6,8,6,6 → Accept (Poster). Similar profile: identifies a real KE problem, proposes a targeted intervention, moderate novelty, some weaknesses in evaluation scope.
- **PRUNE** (knowledge editing, plug-in method for sequential editing): Scores 5,6,6,8 → Accept (Poster). Plugin method with theoretical motivation, practical improvements, but some methodological questions.
- **"How new data permeates LLM knowledge"** (priming phenomenon, mitigation technique): Scores 8,8,6,8 → Accept (Spotlight). Novel phenomenon with strong analysis and effective mitigation. A stronger version of what this paper aspires to be.
- **iReVa** (knowledge editing, key-value adaptors): Scores 3,6,5,6 → Reject. Flawed evaluation, methodological issues, overclaimed results.
- **BaFT** (knowledge editing, representation fine-tuning): Scores 5,8,3,6 → Accept (Poster). New method, theoretical support, editing-locality tradeoff.

This paper is stronger than iReVa in evaluation rigor and mechanistic depth. It is roughly comparable to FiNE and PRUNE: identifies a real phenomenon, provides useful (if not air-tight) mechanistic analysis, and offers a practical mitigation. The main weaknesses relative to FiNE/PRUNE are: (1) the causal overclaim about attention drift, and (2) the remaining large gap to vanilla performance even after SADR. These are significant but do not negate the genuine contributions. The paper is not at the level of "How new data permeates" (spotlight quality) because the mechanism analysis is less airtight and the evaluation is narrower.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>