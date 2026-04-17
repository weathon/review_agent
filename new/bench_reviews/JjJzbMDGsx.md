The paper content was provided in the user prompt. Let me verify the key claims from the harsh critic against the actual paper content I already have.

Going through the harsh critic's "Critical Issues":

1. **Evaluation metric conflates any Latin with "confusion"** — The paper explicitly addresses this in §5.2 by creating FLORES-NO-LATIN and FLORES-WITH-LATIN partitions precisely because Latin confusion is hard to measure. They acknowledge that "Latin confusion presents a more nuanced challenge" and that's why they restrict evaluation of Latin confusion to FLORES-NO-LATIN where any Latin is erroneous. They also separately evaluate code-switch preservation on FLORES-WITH-LATIN. This is a reasonable evaluation design. However, it remains true that on FLORES-NO-LATIN, any Latin character is counted as confusion even though some Latin could be legitimate (e.g., named entities). The paper is partially transparent about this. This is a valid but minor concern, not a structural issue that invalidates the results.

2. **Pseudo-labeling procedure may not reflect actual user-desired language families** — This is a valid concern. The self-distillation creates pseudo-targets from the model's own norm-adjusted distribution, which is heuristic. However, the paper does validate this through the norm-adjustment ablation (LCG-adjusted vs. LCG-unadjusted) and through the 86.7% code-switch preservation rate. The concern about circularity is partially addressed by the empirical validation.

3. **Script-based grouping prevents addressing same-script confusion** — The paper explicitly acknowledges this in §6 as a limitation. The question is whether this invalidates the paper's claims. The paper is about *language confusion* specifically in the form of cross-script mixing, which is the most visible and impactful form. Same-script confusion (e.g., English in Spanish) is a related but different problem. The claim language in the abstract ("language confusion") does overclaim relative to the actual scope (cross-script confusion), but the paper is honest about this limitation. This is a valid criticism of claim scope.

4. **Evaluation of impact on task quality is shallow** — Valid concern about lack of error bars and per-language breakdowns. However, the changes in task metrics are generally very small or slightly positive, which is reassuring. The claim "without negatively impacting task performance" is somewhat strong given the lack of statistical significance testing, but the evidence directionally supports it.

5. **Lack of qualitative or human-level analysis** — Valid. The code-switch evaluation has some human validation but is limited.

From the Spark reviewer:
- **Missing general-purpose benchmarks (MMLU, GSM8K, etc.)** — This is scope creep. The paper is about multilingual generation, not about general LLM capabilities. Including INCLUDE (knowledge/reasoning) and Humaneval-XL (coding) already provides reasonable coverage.

- **Simple rule-based baseline** — The paper does compare against greedy decoding (which is a kind of deterministic baseline) and in-context learning. A "detect language of previous token, restrict to that family" baseline would be informative but is arguably close to what the "persistence of previous token's language" rule already does. Still, this is a reasonable suggestion.

- **Evaluation on truly low-resource languages** — Valid point. The evaluation languages (Arabic, Hebrew, Korean, Thai) are not truly low-resource, yet the method is supposed to help with low-resource confusion.

Now let me synthesize the final review.

## Summary

The paper introduces the Language Confusion Gate (LCG), a lightweight two-layer MLP trained via norm-adjusted self-distillation that dynamically masks tokens from inappropriate language families during decoding to mitigate language confusion in multilingual LLMs. Key insight: correct-language tokens are usually among top predictions, and output token embedding norms bias sampling toward high-resource languages. LCG reduces cross-script confusion substantially across multiple models with minimal computational overhead.

## Strengths

- **Novel mechanistic insight**: The norm imbalance analysis (§3.2, Table 1) demonstrating that high-resource language tokens have systematically larger output embedding norms, creating a sampling bias, is a genuine contribution. The logit decomposition into norm × cosine similarity is standard but the empirical finding and its application via norm-adjusted self-distillation is novel and well-motivated. Figure 2 effectively illustrates how norm adjustment eliminates confusion tokens from the top candidates.

- **Practical, deployable solution**: The LCG adds only 0.4% latency overhead (§6), requires no retraining of the base LLM, and intervenes on only 0.33–0.38% of tokens. This is a practically significant contribution for production systems.

- **Strong empirical results on CJ confusion**: The reductions in Chinese/Japanese character confusion are dramatic and clearly meaningful — e.g., Qwen3-8B CJ confusion from 4.5% to 0.1% on FLORES-NO-LATIN, with task performance stable or slightly improved. These are unlikely to be artifacts since CJ characters in Arabic/Hebrew/Korean/Thai outputs are almost always genuine confusion.

- **Well-designed ablations**: The norm-adjustment ablation (LCG-adjusted vs. LCG-unadjusted) credibly demonstrates the value of the norm debiasing mechanism. The comparison against greedy decoding, ICL, and ORPO baselines (Figure 3) provides meaningful context, with ORPO showing degraded INCLUDE accuracy on Qwen3-8B (61.4 → 57.3).

- **Comprehensive model coverage**: Evaluation across Qwen3, Llama3.1, Gemma3, GPT-OSS, and both thinking and no-think modes provides strong evidence of generalizability.

## Weaknesses

### Major

- **The method's scope is narrower than claimed, and the paper overstates generality**: The paper uses "language confusion" throughout (abstract, introduction, title) as if addressing the full problem, but LCG operates on four coarse script families (CJ, Latin, Symbols, Low-Res) and cannot handle same-script confusion (e.g., English fragments in Spanish/German, or confusion between Cyrillic languages). The acknowledgment in §6 understates this: it's not a minor future extension — it's a fundamental architectural limitation. The paper would be more honest if framed as addressing *cross-script* confusion specifically. The evaluation also only covers 5 target languages (Arabic, Hebrew, Korean, Thai, Chinese) that are all relatively well-served by LLMs, leaving truly low-resource settings untested.

- **The self-distillation pseudo-labels lack ground-truth validation, and the relationship between learned gate vs. heuristic rules is unclear**: The gate is trained on pseudo-targets derived from the model's own norm-adjusted predictions, which creates a circular dependency: if the base model's corrected predictions are still wrong, the gate learns incorrect behavior. No analysis is provided of pseudo-label accuracy (e.g., precision/recall of the norm-adjusted top-k predictions against human annotations of correct language family). Furthermore, the three hand-crafted intervention rules (§4.3) — especially rule (2) that overrides the gate when contradicted by high-confidence model output — may be doing substantial work. The "No Rule" ablation mentioned briefly in §5.3 is only discussed qualitatively ("LCG can still reduce language confusion without the additional rules, but the combination of rules and LCG further reduces language confusion rate"), without quantitative decomposition. It is unclear whether the learned gate alone is sufficient or whether the heuristic rules are essential.

- **Over-suppression of legitimate code-switching is not adequately addressed**: In Table 5, Qwen3-8B's code-switch rate drops from 46.34% (no LCG) to 25.90% (LCG-adjusted), which is *12.5 percentage points below* the ground-truth answer rate (38.36%). This suggests meaningful over-suppression of legitimate Latin usage. The 86.7% preservation rate on human-validated examples sounds reasonable but is measured only on Qwen3-8B and only on a subset of pre-selected "good" code-switch cases from outputs *without* LCG — a methodology that may favor easier, more obvious cases. The claim that LCG "preserves legitimate code-switching" needs stronger support given this magnitude of over-suppression.

### Minor

- **Latin confusion metric may count some legitimate Latin as erroneous**: While FLORES-NO-LATIN is a reasonable filtering strategy, some sentences naturally contain Latin-script terms (e.g., named entities, abbreviations) even when the reference doesn't. The paper acknowledges this difficulty but does not estimate how many "confusion" instances are actually legitimate uses. This concern primarily affects Latin% numbers; CJ% numbers are more trustworthy.

- **Task performance evaluation lacks statistical rigor**: No error bars, confidence intervals, or statistical significance tests are provided. Some task metric changes are small (e.g., Qwen3-30B INCLUDE accuracy: 71.12 → 70.83). While the direction is generally positive, the claim "without negatively impacting task performance" would be stronger with variance estimates.

- **Training data details are sparse**: The ~78K sample training dataset composition is listed but distribution across language families and tasks is not reported, making it hard to assess whether the gate has sufficient exposure to diverse linguistic contexts.

- **The analysis in §3.1 is conducted only on Qwen3-8B on FLORES-NO-LATIN**: The claim that "correct-language tokens appear within top-3 99.29% of the time" is a key motivator for a logits-based intervention, but its generality across models and tasks is not established.

### Trivial

- The speculative decoding compatibility (Appendix F) is mentioned but not experimentally validated.

- ORPO baseline details (training cost, hyperparameters, data scale) are not provided.

## Nice-to-Haves

- Evaluate on at least a few genuinely low-resource languages to validate the Low-Res language family behavior.
- Report gate prediction accuracy (precision/recall/F1) against human-annotated language family labels on held-out data.
- Provide per-language breakdown of confusion rates to show uniformity of improvements.
- Ablate each intervention rule independently (not just all-or-nothing) to isolate contributions.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"LCG doesn't work on same-script confusion"** — The paper explicitly acknowledges this limitation in §6. While the framing could be improved (see Major weakness above), the limitation itself is disclosed. Removed as a standalone fatal criticism; kept as a scope-overclaim issue.

- **"Missing experiments on MMLU, GSM8K, etc."** — This is scope creep. The paper is about multilingual generation quality, not general LLM capabilities. INCLUDE already tests reasoning/knowledge in multiple languages. Removed.

- **"No confidence intervals for task performance"** — For large-scale benchmark evaluation, single-run reporting is the community norm. While variance estimates would be nice, their absence doesn't undermine the paper's core claims. Moved to Minor.

- **"The pseudo-labeling is heuristic and not grounded in user preference"** — While a valid conceptual concern, the paper validates the approach empirically (significant confusion reduction with preserved task performance and partial code-switch preservation). The concern is partially mitigated by results. Kept as part of Major weakness but softened.

- **"FLORES-NO-LATIN conflates all Latin with confusion"** — The paper explicitly creates this partition to handle exactly the nuance the critic raises, and separately evaluates FLORES-WITH-LATIN. While not perfect, this is a responsible design. Kept as a Minor concern about edge cases, not a structural flaw.

## Novel Insights

The embedding norm imbalance insight — that high-resource language tokens systematically have larger output embedding norms, which mechanically biases sampling — is the paper's most novel contribution. This provides a concrete mechanistic explanation for why language confusion disproportionately affects low-resource languages, and the norm-adjusted self-distillation elegantly leverages this insight to create better pseudo-targets. The general principle (that logit scale effects beyond semantic similarity can systematically distort sampling) likely generalizes beyond multilingual settings.

## Suggestions

1. Reframe title and abstract to specify "cross-script" confusion rather than the broader "language confusion," which would better match the method's actual scope.
2. Quantify the contribution of each intervention rule (§4.3) independently through per-rule ablation.
3. Report pseudo-label quality: what percentage of norm-adjusted top-k predictions correctly identify the language family at each position?
4. Add error analysis on the 13.3% of legitimate code-switch cases that LCG blocks, to characterize when over-suppression occurs and whether it has practical impact.

## Score and Decision

Calibration: SADI (similar inference-time intervention paper, accepted poster) scored 8/6/6/6/6 ≈ 6.4 average. DeCo (similar decoding-time intervention, accepted poster) scored 6/6/6/6. VTI (latent space steering for hallucination, accepted spotlight) scored 6/8/8. The "When Is Multilinguality a Curse?" paper (multilingual analysis, rejected) scored 6/6/5/8. Crosslingual capabilities paper (rejected) scored 3/6/8.

This paper has genuine novelty in the norm-imbalance insight, strong practical results on the uncontroversial CJ confusion metric, excellent efficiency (0.4% overhead), and broad model coverage. However, it overclaims scope (cross-script vs. full language confusion), has an unaddressed over-suppression problem for legitimate code-switching (12.5pp below ground truth), and the core evaluation metric for Latin confusion has known edge cases. The self-distillation circularity and lack of gate accuracy analysis are real but partially mitigated by empirical results. These are significant but not fatal weaknesses — the core method works well for its actual scope. The paper is comparable in quality to DeCo (accepted poster, all 6s) but with slightly more substantial contributions (the mechanistic insight and broader evaluation).

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>