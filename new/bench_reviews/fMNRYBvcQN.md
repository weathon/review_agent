Now let me search for calibration anchors.Now I have sufficient calibration data. Let me write the final review.Now I have everything I need to write the final review.

---

## Summary

This paper studies *benign relearning attacks* against approximate LLM unlearning methods. The central finding is that finetuning an unlearned model on a small set of loosely related, publicly available data — data that does not directly answer the evaluation queries — is sufficient to recover supposedly forgotten knowledge. The paper evaluates this phenomenon across three benchmarks (WMDP, TOFU, WHP), five gradient-based unlearning methods, and two threat model scenarios, and provides mechanistic intuition via a controlled synthetic experiment. The core claim is that existing approximate unlearning methods "obfuscate" rather than truly "forget" target knowledge.

---

## Strengths

- **Novel, stricter threat model** (Section 2.2): Unlike prior work (Lynch et al., 2024; Tamirisa et al., 2024) that uses relearn data which may directly answer evaluation queries (e.g., the first three HP books contain answers like "Harry Potter's friends are..."), this paper carefully constructs relearn sets that are *insufficient to directly answer evaluation queries*. This is a meaningful methodological refinement that strengthens the "obfuscation" interpretation.

- **Strong multi-benchmark empirical coverage** (Sections 3–4): Results span three diverse unlearning tasks — hazardous knowledge suppression (WMDP), verbatim copyright recovery (WHP), and fictitious fact retention (TOFU) — across five unlearning objectives (GA, GD, KL, NPO, SCRUB) and two models (Zephyr-7b-beta, Llama-3-8b). The consistency of the attack across methods and settings is notable.

- **Compelling WHP result** (Section 4.2, Figure 5): Rouge-L scores jump from 0.03–0.23 (unlearned) to 0.44–0.78 (relearned) by finetuning on GPT-generated generic facts about Harry Potter characters — data that unambiguously does not contain the verbatim copyrighted text. This is the paper's cleanest piece of evidence that memorized information persists internally.

- **Mechanistic synthetic experiment** (Section 5, Figure 6, Table 4): The experiment demonstrates that during relearning on *Anthony* alone, the NLL of the unlearned *Mark* token drops in tandem (Figure 6), and that relearning success rate scales with the frequency of co-occurrence between the token pair in pretraining (Table 4). This provides a principled, controlled illustration of the latent-memory mechanism.

- **LoRA susceptibility finding** (Table 2): The result that parameter-efficient unlearning (LoRA) makes models significantly more susceptible to relearning attacks (scores recovering from 1–1.67 → 5.08–6.2 with LoRA vs. similarly large recovery with full unlearning) is a practical cautionary finding for the community.

- **Relevance gradient characterization** (Section 6, Table 5): The paper establishes that completely unrelated data (gibberish) yields only a marginal ASR boost (9%), while correlated English text achieves 100%. This operationalizes the degree to which the relearn data must be related.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing naive baseline control for the "obfuscation vs. re-teaching" distinction** — The paper's central thesis is that approximate unlearning *obfuscates* (retains latent knowledge) rather than *forgets* (removes associations). However, the experiments never test whether a **model that was never trained on the forget data** could reach similar relearn scores with the same relearn data. Without this control, an alternative explanation — that the relearn data is simply informative enough to re-teach the content from scratch — is not ruled out. The WHP result is the strongest counter-evidence to this alternative (generic character facts should not re-teach verbatim text), and the TOFU experiment has natural protection (target keyword never appears in relearn set), but for WMDP the relearn data is explicitly constructed to be *topically specific* to each evaluation question (GPT-generating "general knowledge relevant to q"), making the alternative non-trivial. The "obfuscation" framing is plausible and likely correct, but the single most important control experiment — testing a blank model on the same relearn set — is absent.

### Minor

- **RMU (the WMDP benchmark's own method) is discussed only in the appendix and Section 7** — RMU is the method specifically designed and evaluated for WMDP. The paper reveals an important nuance: the relearning attack succeeds on MCQ evaluation but fails on sentence completions for RMU. This metric-dependence is a meaningful caveat to the broad claim that "regardless of the objective used for unlearning, vanilla finetuning… is sufficient to recover the unlearned content" (Section 4.1). This should be prominently presented alongside Figure 4, not deferred to a discussion section and appendix.

- **Section 3 evaluates only gradient ascent as the unlearning method** — Sections 3 and 4 serve different threat model scenarios (partial unlearn set vs. public data), but Section 3 tests only GA, while Section 4 tests five methods. Given that Section 4 shows that methods differ in susceptibility (e.g., SCRUB vs. GA), the Section 3 results may not generalize to other unlearning methods.

- **Small evaluation set in WHP Section 4.2** (15 samples) — While the result is striking, 15 completions is a small evaluation set and the variance is uncharacterized. The pattern holds qualitatively (Figure 5) but quantitative confidence is limited.

- **Gap between synthetic experiment and real settings** — The synthetic mechanism (token-level co-occurrence frequency → relearning susceptibility) is clearly illustrated in the toy name-pair setting, but its connection to the WMDP setting (multi-step semantic relationships between virology concepts and bioweapon synthesis) is asserted rather than bridged. The paper appropriately frames Section 5 as providing "intuition," but a brief bridge argument would strengthen the flow.

### Trivial
None.

---

## Nice-to-Haves

- Adding a control experiment where a model *not* pretrained on the forget content is finetuned on the same relearn data would be a clean way to definitively substantiate the "obfuscation vs. re-teaching" distinction.
- Reporting LLM-as-Judge scores with error bars (e.g., standard deviation across the 70 WMDP evaluation questions) and Rouge-L across the 15 WHP completions would strengthen confidence in the quantitative results.
- A concrete proposed metric — e.g., a "relearning robustness score" that accompanies standard forget/retain evaluations — would sharpen the contribution from empirical finding to actionable recommendation.

---

## Removed Points

*These points were reviewed and removed; treat them with caution.*

1. **Harsh Critic — Figure 4 "relearn scores exactly match original scores"**: The critic noted that the OCR-extracted figure description shows identical Original and Relearn values and uses this to imply the relearn data may be highly informative. This is explicitly a PDF parsing artifact — the bar chart caption describes stacked bar values and the parser cannot differentiate the Original vs. Relearn bars. **Removed per hard rule on formatting artifacts.**

2. **Harsh Critic — relearn set appendix construction details**: The critic suggests the key WMDP relearn set construction details are in a "parsed away" appendix. Since the paper explicitly states the construction rule in Section 4.1 ("we find public online articles related to q and use GPT to generate paragraphs about general knowledge relevant to q… We ensure that this resulting relearn set does not contain direct answers") and cites Appendix C.2, the appendix is not "absent" — it is stripped by the parser. **Removed per hard rule on absent appendix sections.**

3. **Harsh Critic — abstract slightly oversells generality**: The claim that "the abstract's framing slightly oversells the generality" based on the fact that Section 6 shows unrelated data only yields 9% ASR is a nitpick — the abstract says "potentially loosely related," not "completely unrelated," which is consistent with Section 6's findings. **Removed as strawman.**

4. **Strength Finder — "effective visualization of the threat pipeline" (Figure 1, Figure 2)**: Generic presentation strength without citation to specific metrics or comparative evidence. **Removed as too generic.**

5. **Strength Finder — importance of the problem**: Framing unlearning evaluation as important is generic and applies to all papers in this area. **Removed as non-specific.**

---

## Novel Insights

The paper's most genuinely novel observation is the demonstration — via the synthetic token co-occurrence experiment — that approximate unlearning methods operating on full-token sequences fail to remove *associations between correlated parts* of the forget data, such that relearning on one correlated token passively restores prediction probability for an unseen partner token. This mechanism-first framing is more precise than the general "finetuning reverses unlearning" message of prior work and offers a testable prediction: unlearning methods that explicitly target the internal associations (rather than just the output distribution) should be more robust to benign relearning. The PEFT/LoRA susceptibility finding also contributes a practical warning: efficiency-motivated unlearning via low-rank adapters is qualitatively more fragile than full-parameter unlearning.

---

## Suggestions

1. Add a control experiment: take the same LLM *before* exposure to the WMDP bio-corpus (or a model that was never finetuned on it), apply the same relearn data, and measure evaluation scores. This single experiment would either confirm or refute the "latent memory" interpretation.
2. Move RMU results from Appendix E to the main body alongside Figure 4, reporting both MCQ and sentence completion accuracy for all methods. Explicitly state where the attack does and does not succeed.
3. For WMDP, show at least one complete relearn article alongside the corresponding test question and model output, so readers can independently judge whether the relearn content provides "general" vs. "functionally sufficient" background.

---

## Score and Decision

**Calibration anchors consulted:**

| Path | Avg Score | Comparison |
|------|-----------|------------|
| `/home/wg25r/review_agent/human_reviews/fXJCqdUSVG.md` | 6.5 (Accept, Poster) | Most comparable: evaluates durability limitations of LLM safety defenses through empirical case studies. The paper under review tests more methods (5 vs 2) across more benchmarks (3 vs 2) and adds a mechanistic experiment, but that paper had cleaner evaluation metrics. |
| `/home/wg25r/review_agent/human_reviews/MGKDBuyv4p.md` | 7.33 (Accept, Spotlight) | Stronger than this paper: introduces novel methods + benchmark (TinyMem) in addition to evaluation. |
| `/home/wg25r/review_agent/human_reviews/51WraMid8K.md` | 8.0 (Accept, Oral) | Stronger: formal probabilistic framework with guarantees, more theoretical depth. |
| `/home/wg25r/review_agent/human_reviews/iQIQT88prm.md` | 5.33 (Reject) | Below this paper: limited experimental scope (CIFAR only, no LLM benchmarks), unclear method details. |
| `/home/wg25r/review_agent/human_reviews/hwXUmwJAq5.md` | 3.0 (Reject) | Well below this paper: minimal experiments, no LLM validation, no novel insight. |

**Reasoning:** The paper under review is most similar in character to fXJCqdUSVG (6.5): empirical evaluation of LLM safety robustness limitations through case studies, with practical implications. The paper under review has broader methodological coverage (5 unlearning methods, 3 benchmarks vs 2 methods, 2 settings) and a more principled mechanistic experiment, but its core mechanistic claim — "obfuscation not forgetting" — is somewhat undercut by the missing naive baseline control. The WHP verbatim recovery result is a particularly strong piece of evidence, and the LoRA susceptibility finding is practically useful. The paper is solidly above the medium anchors (5.33 reject) and comparable to the 6.5 accepted-poster anchor. The missing control experiment keeps it below spotlight-quality work. Score: **6.0**.

**Axis evaluations:**
- *Originality*: Moderate-to-good. The key novelty (relearn data that doesn't directly answer queries) is meaningful; the attack mechanism itself builds on prior relearning work.
- *Importance*: High. LLM unlearning for safety/copyright is actively deployed; showing it's brittle to this simple attack has direct practical consequences.
- *Claims well-supported*: Mostly yes for the empirical finding; the mechanistic "obfuscation" interpretation is well-supported for WHP/TOFU but less conclusive for WMDP.
- *Soundness of experiments*: Good. Multiple methods, models, metrics; appropriate ablations on relearn data relevance.
- *Clarity*: Good. The pipeline is well-explained; the placement of RMU in the appendix is the main structural flaw.
- *Value to community*: High. The practical implication — that relearning robustness should be a standard unlearning evaluation criterion — is timely and actionable.

**Decision: Accept (Poster)**

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>