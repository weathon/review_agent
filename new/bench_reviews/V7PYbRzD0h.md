## Summary

The paper introduces Chain-of-Jailbreak (CoJ), a multi-turn attack that decomposes malicious queries into sequences of harmless-looking sub-queries to iteratively edit images, bypassing safety filters in commercial image generation services (GPT-4V, GPT-4o, Gemini 1.5, Gemini 1.5 Pro). It constructs CoJ-Bench with 9 safety scenarios, 3 edit operations, and 3 edit elements, and proposes Think-Twice Prompting as a defense that asks models to self-assess safety before generating.

## Strengths

- **The core finding—that multi-turn iterative editing bypasses safety filters in widely deployed commercial image generation APIs—is valid and practically important.** Table 3 reports human-evaluated JSR of 54.8% on GPT-4V and 62.3% on GPT-4o, demonstrating that current guardrails are inadequate against this attack class. This is a real vulnerability that service providers should address.

- **The edit operation × edit element taxonomy provides a structured decomposition framework with actionable ablation insights.** Tables 5–6 reveal that insert-then-delete is the most effective operation (49% avg.) and word-level editing outperforms character and image-level editing (51% vs. 39%), providing concrete guidance for both attackers and defenders.

- **Dual evaluation methodology (human + automatic) yields consistent rankings.** Table 3 shows both methods agree on the relative model ordering (GPT-4o > GPT-4V > Gemini 1.5 > Gemini 1.5 Pro), increasing confidence in the automatic evaluation pipeline.

## Weaknesses

### Fatal
None.

### Major

- **The baseline comparison in Table 4 is structurally unfair.** CoJ uses multiple turns (2–5 sequential API calls building on prior outputs), while every baseline (instruction ignore, refusal suppression, role play, etc.) is a single-turn method. The headline "significantly outperforms other jailbreaking methods (i.e., 14%)" conflates the multi-turn advantage with the specific decomposition strategy. The paper's own threat model (Section 2.1) mentions "small number of queries" as a goal, yet no query counts are reported for any method. Without multi-turn baselines—e.g., allowing refusal suppression or role play to re-prompt after a refusal—it is impossible to isolate whether the advantage comes from the decomposition strategy or simply from having multiple turns. The finding risks reducing to "multi-turn attacks outperform single-turn attacks," which is expected. This does not invalidate the empirical result, but it substantially weakens the novelty claim about the specific decomposition being the key contribution.

- **The paper does not disaggregate results by text-in-image (slogan) generation vs. harmful visual content generation.** The method is introduced and primarily demonstrated through text slogan examples (Figure 1, Figure 3), and only 19.3% of prompt sets involve image-level editing (Section 2.3). While Table 6 shows image-level editing achieves 39% average JSR (and up to 72% on GPT-4o), the paper does not report which safety scenarios correspond to text-in-image vs. visual content. This makes it impossible to assess whether CoJ works for the most dangerous scenarios (pornography, violence, child abuse) or primarily for offensive text slogans—a fundamentally different and less severe safety failure. Without this breakdown, the claimed severity of the vulnerability cannot be properly calibrated.

- **Defense evaluation (Table 7) is insufficient to support the ">95% defense" claim.** Only 40 test cases are sampled from successful attacks—too small to support the claim with confidence, especially with no variance estimates. More importantly, Think-Twice Prompting is trivially simple ("determine whether it is safe to generate"), and no adaptive attack is tested. An adaptive attacker aware of this prompt could modify sub-queries to appear safe during the description step (e.g., "describe the image you will generate, making it sound safe"). The paper also acknowledges (Section 4.4) that the defense is injected as a user message rather than a system prompt, since the authors lack system prompt access—limiting practical deployability.

### Minor

- **The abstract's "over 60% cases" claim is misleading.** This holds only for GPT-4o (62.3%); the cross-model average for human evaluation is ~44%. The abstract should represent the range more accurately (e.g., "up to ~62% on GPT-4o").

- **The Levenshtein distance framing adds limited value.** Section 2.2 introduces edit operations "inspired by Levenshtein distance," but the actual decomposition is performed by Mistral-Large-2 (Section 3.2) with manual filtering—not derived algorithmically from edit distance. The connection does not constrain or improve the method.

- **CoJ-Bench is relatively small (120 seed queries, 776 decomposed query series) with no statistical significance testing.** Given the high variance in jailbreak outcomes, confidence intervals or significance tests would strengthen the reported JSR differences.

### Trivial
None.

## Nice-to-Haves

- Compare against at least one multi-turn baseline (e.g., Crescendo, or simply allowing single-turn methods to re-prompt after refusals) to isolate the decomposition strategy's contribution from the multi-turn advantage.
- Disaggregate JSR by text-in-image vs. harmful visual content across safety scenarios to properly calibrate threat severity.
- Test an adaptive attack against Think-Twice Prompting; even a simple adaptive strategy would strengthen the defense evaluation.
- Analyze why Gemini models are more robust (~2× lower JSR) than GPT models—is this architectural, input filtering, or output filtering?

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"DALL-E 3 mentioned in abstract but not tested":** The abstract uses DALL-E 3 as an example of the model class ("Text-based image generation models, such as Stable Diffusion and DALL-E 3"), not as a tested model. This is not a misleading claim.

- **"Stable Diffusion and Midjourney excluded because safeguards are too weak":** The paper explains this design choice in Section 4.1 with a reference to Appendix B. While the claim could be better supported in the main text, this is a reasonable scoping decision, not a fundamental flaw.

- **"Manual filtering of decompositions raises reproducibility concerns / criteria for incorrect decomposition not specified":** Standard practice for LLM-assisted data generation; minor concern.

- **"Table 4 uses 50 seeds vs. Table 3 uses full set":** The paper states "We randomly select 50 seed malicious queries" for Table 4. This is disclosed, just not compared directly.

- **"Deng & Chen (2023) distinction is narrow/incremental":** The paper does distinguish its contribution (multi-turn iterative editing vs. single-round decomposition). Whether this is sufficient is already captured in the Major weakness about baseline fairness.

- **Strength removed: "CoJ dramatically outperforms all existing prompt-based jailbreak methods (6× improvement)":** This strength conflicts with the verified Major weakness about structurally unfair comparison. The gap is partially explained by the multi-turn vs. single-turn asymmetry.

## Novel Insights

The paper reveals a specific structural weakness in commercial image generation safety systems: they evaluate prompts in isolation per turn rather than reasoning about cumulative intent across an editing conversation. The insert-then-delete operation being most effective (because it adds and removes benign words like "not") suggests that safety filters rely heavily on keyword-level checks rather than semantic reasoning about the trajectory of edits—a finding with clear implications for defense design.

## Suggestions

- Add a simple multi-turn baseline: for each single-turn method in Table 4, allow 2–3 re-prompts after refusals using the same strategy. This would isolate whether CoJ's advantage stems from its decomposition or from simply having more turns.
- Report JSR separately for text-slogan cases vs. image-level editing cases across all safety scenarios. This requires only re-grouping existing data and would resolve the severity calibration concern.
- For the defense evaluation, test at least one adaptive attack variant (e.g., instructing the model that the safety check is a formality) on the same 40 cases.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Safety Alignment Should be Made More Than Just a Few Tokens Deep | `/home/wg25r/review_agent/human_reviews/6Mxhg9PtDE.md` | 9.5 | Provides deep mechanistic understanding of safety failures with principled mitigation. CoJ is much weaker on mechanistic depth and novelty. |
| Targeted Attack Improves Protection | `/home/wg25r/review_agent/human_reviews/agHddsQhsL.md` | 7.5 | Novel targeted attack on diffusion models with strong experiments and explanatory analysis. CoJ has weaker novelty and analysis. |
| Catastrophic Jailbreak via Generation | `/home/wg25r/review_agent/human_reviews/r42tSSCHPh.md` | 7.0 | Surprising 0→95% finding with clean evaluation across 11 models. CoJ's finding is less surprising and evaluation less rigorous. |
| ActorAttack (multi-turn LLM jailbreak) | `/home/wg25r/review_agent/human_reviews/kvvvUPDAPt.md` | 5.33 | Very similar concept (multi-turn decomposition jailbreak). Criticized for unfair baselines, weak technical contribution, small dataset. CoJ shares these weaknesses but has structured taxonomy and defense contribution. |
| BSPA (black-box prompt attacks on image generators) | `/home/wg25r/review_agent/human_reviews/x31F1VmiV7.md` | 5.25 | Similar topic (image generation safety), limited novelty. CoJ has stronger attack results but similar novelty concerns. |
| NEMESIS (chain-of-thought jailbreak survey) | `/home/wg25r/review_agent/human_reviews/5kMwiMnUip.md` | 1.40 | Survey of known attacks, no real contribution. CoJ is clearly far better. |

CoJ is clearly above the low-scoring anchors (it has real empirical findings, structured evaluation, and a defense). It is comparable to the medium-scoring anchors like ActorAttack (5.33), sharing the key weakness of unfair baseline comparison and limited novelty. CoJ has some advantages over ActorAttack (structured taxonomy, defense, human evaluation, tested on commercial APIs) but also shares the same core methodological concern. It falls well below the high-scoring anchors (7.0–9.5), which offer deeper mechanistic insight, more surprising findings, or more rigorous evaluation.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>