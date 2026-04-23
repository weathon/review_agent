Now I have all the information needed to write the final review. Let me synthesize everything.

## Summary

The paper introduces Ambig-SWE, an underspecified variant of SWE-Bench Verified, and proposes a structured evaluation framework that decomposes agent performance under underspecificity into three capacities: detecting missing information, asking targeted clarification questions, and leveraging interaction to improve task completion. Using a paired experimental design (Full/Hidden/Interaction settings on 500 issues), the paper evaluates six LLMs and finds that interaction can recover substantial performance (up to 89% of fully-specified performance for Claude Sonnet 4), but most models default to non-interactive behavior and struggle with detection even under explicit prompting.

## Strengths

- **The three-step decomposition (detect → ask → leverage) is a genuine conceptual contribution.** Prior work evaluates underspecificity end-to-end; breaking it into these capacities provides a reusable evaluation framework and directly enables the targeted analyses in RQ1–RQ3. This decomposition reveals differential model failures (e.g., Qwen 3 Coder solves tasks competitively but has 100% FNR on detection in Table 2) that would be invisible in monolithic evaluation.

- **The paired Full/Hidden/Interaction design enables causal attribution.** Using the same 500 issues across all three settings eliminates problem-difficulty confounds, allowing legitimate causal claims about the effect of interaction. The paper explicitly justifies this over naturally underspecified issues (Section 2.1): "we did not evaluate on naturally underspecified SWE-Bench examples because they lack the paired ground truth necessary for causal measurement."

- **The exploration-first vs. immediate-asking distinction (Section 5.3) is a practically relevant and non-obvious finding.** Claude Sonnet models explore the codebase before asking, producing fewer but more targeted questions (4.03 avg.) while achieving comparable information gain to Qwen (6.02 avg. questions), which asks immediately. This has direct implications for agent design.

- **The finding that interaction improves effectiveness but not efficiency is important and counterintuitive.** Claude Sonnet 4 increases steps from 65→75 when interaction is enabled (Section 3.2), challenging the assumption that more information should make agents faster, and suggesting current training paradigms don't optimize for efficient information integration.

- **Structured detection analysis across prompt conditions (Table 2) reveals that prompt engineering is insufficient.** The systematic evaluation across Neutral/Moderate/Strong Encouragement shows that even strong prompting cannot make most models reliably detect underspecificity, with the notable exception of Claude Sonnet 4 (89% accuracy under Strong Encouragement).

## Weaknesses

### Fatal
None.

### Major

- **Unequal turn budgets confound cross-model comparisons, yet the paper makes definitive comparative claims.** Claude Sonnet 4 and Qwen 3 Coder receive 100 interaction turns while all other models receive only 30 (Section 3.1), justified as accounting for "greater reasoning and planning capacity." This means their Hidden-setting performance is inflated relative to others (more steps to explore/attempt solutions), and their Interaction-setting performance is similarly inflated. The paper then draws cross-model conclusions such as "proprietary models generally demonstrating greater effectiveness in utilizing interaction compared to open-weight models" (Section 3.1) and compares Qwen 3 Coder's step count (65) against other models' entire 30-turn budget for efficiency analysis. These comparative claims cannot be supported under systematically different resource budgets. The within-model comparisons (Hidden vs. Interaction for each model individually) remain valid, but the cross-model narrative that forms a substantial portion of the results is undermined.

- **The headline "up to 74% improvement" claim in the abstract is misleading about where interaction matters most.** This figure represents relative improvement from very low bases (e.g., a model going from ~4% to ~7% Hidden→Interaction), while models with the most meaningful absolute gains (Claude Sonnet 4: tens of percentage points) show lower relative improvements. The "up to 74%" framing makes interaction sound transformative for the weakest models when the evidence actually shows it is transformative for the strongest ones—the models where interaction yields the largest absolute gains are the ones that were already most capable. The paper provides more informative metrics in the body ("recover up to 80% of the Full setting performance," "relative performance of 89%"), but the abstract's most prominent quantitative claim distorts the core finding. This matters because it shapes how readers interpret the paper's contribution and the practical value of interaction.

### Minor

- **The detection experiment (RQ2) cannot distinguish detection failure from instruction-following failure for non-interacting models.** Qwen 3 Coder has 100% FNR across all prompt conditions (Table 2). The paper interprets this as "complete non-responsiveness to interaction prompts, suggesting fundamental limitations in certain training approaches" (Section 4.3), which is more careful than claiming detection failure per se. However, RQ2 is framed as "Detection of Incomplete Task Specifications" and uses detection metrics (FPR/FNR/Accuracy); for a model that never interacts, these metrics reveal only that it won't follow interaction instructions, not whether it could detect underspecificity if it were willing to interact. The same concern applies partially to Claude Haiku 3.5 (FNR of 0.97 under Neutral prompting).

- **Table 1's comparison of resolve rates with vs. without navigational information is subject to selection bias.** Models self-select into asking for navigational details based on their own assessment of task difficulty, which correlates with problem characteristics. The paper's interpretation that "requesting navigational details improves performance" assumes these groups are comparable, but they are not randomly assigned. This is particularly relevant for the Qwen 3 Coder finding (performance worsens with navigational info: 52.38% vs. 55.43%), which likely reflects that the model asks for navigation on harder problems rather than navigation causing worse performance. The paper provides a thoughtful qualitative explanation for Qwen's case but does not acknowledge the general selection bias.

- **The "relative performance" metric (e.g., "recover up to 80% of the Full setting performance," "relative performance of 89%") is never explicitly defined.** The natural interpretation is (Interaction − Hidden) / (Full − Hidden), but this formula is not stated, and the interpretation changes substantially if it means something else (e.g., Interaction / Full). This should be defined explicitly for clarity.

- **There is a disconnect between the motivating claim and the experimental setup regarding "emerging" underspecificity.** The introduction states that "real-world agentic tasks often involve multiple, interdependent gaps in specification that emerge over the course of a trajectory" (Section 1), but the experimental design generates underspecification by stripping information from the issue description upfront. The underspecification does not "emerge" during the trajectory—it is present from the start. While the controlled design is methodologically defensible, the motivation oversells what the experiment actually tests.

### Trivial
None.

## Nice-to-Haves

- Running Claude Sonnet 4 and Qwen 3 Coder with the standard 30-turn budget would enable fair cross-model comparisons and substantially strengthen the paper's comparative claims.
- A per-model improvement distribution (e.g., waterfall plots of Interaction−Hidden differences across individual issues) would reveal whether gains are consistent or driven by a few dramatic recoveries, providing more nuance than aggregate resolve rates.
- Adding a forced-interaction condition for RQ2 (where non-interacting models are compelled to ask questions, as in RQ1) would separate willingness to interact from ability to detect, making the detection evaluation meaningful for all models.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Wilcoxon signed-rank test with n=500 detects any non-zero difference as significant.** While technically true, the paper reports effect sizes through resolve rate differences, and the statistical tests serve to confirm directionality rather than establish practical significance. This is a minor statistical note, not a substantive weakness.

- **Data leakage hypothesis for Qwen 3 Coder.** The paper itself mentions this possibility ("Some models achieve higher resolve rates in the Hidden setting likely due to their superior programming acumen, or data leakage," Section 3.2). While investigating it would be ideal, verifying training data contamination is impractical in a standard submission, and the paper appropriately flags it as a potential factor.

- **Cosine distance measures change in embedding space, not task-relevant information gain.** The paper explicitly acknowledges this in the Limitations section: "Question quality is approximated via latent vector changes that weigh all information equally, though models may prioritize details differently." The paper also uses LLM-as-judge as a complementary metric.

- **LLM-as-judge scores converge around 4/5 for all capable models (ceiling effects).** The paper notes this and uses cosine distance precisely because it provides the granularity that the judge metric lacks. This is a known limitation that the paper addresses by using dual metrics.

- **User proxy (GPT-4o) may be more cooperative than real users.** The paper acknowledges this in limitations and justifies the design choice: "The goal is not to simulate real users but provide the information injection to the trajectory and analyze model behaviors" (Section 2.2). This is a deliberate design choice for controlled evaluation, not an oversight.

- **Reproducibility concerns about undisclosed hyperparameters or implementation details.** The paper commits to releasing code and data, uses the standard OpenHands framework, and provides full prompts in the appendix. Standard reproducibility expectations are met.

- **Missing related works.** Per instructions, I do not flag missing citations.

- **The synthetic underspecification differs from natural underspecification.** The paper devotes significant attention to this (Section 2.1), conducts a distributional difference analysis, and argues the differences may not impact agent performance. This is a reasonable and acknowledged design tradeoff for controlled evaluation.

## Novel Insights

The most insightful observation across the reviews is that interaction capability appears to be orthogonal to task-solving capability: Claude Haiku 3.5 achieves similar relative performance recovery (~80%) as Claude Sonnet 3.5 despite vastly inferior coding ability, while Qwen 3 Coder—competitive with Claude Sonnet 4 on SWE-Bench—is completely unable to interact. This suggests that the training paradigms that produce strong coding models may not be the same ones that produce effective interactive agents, and that interaction capability may need to be specifically targeted rather than emerging as a side effect of scale.

## Suggestions

- Replace or contextualize the "up to 74%" claim in the abstract with the more informative "relative performance recovery" metric (e.g., "interaction recovers up to 89% of fully-specified performance for the strongest models"), or at minimum report both absolute and relative improvements so readers can assess practical significance.
- Either run equal-turn-budget experiments for the two models with 100 turns, or explicitly qualify all cross-model comparative claims by noting the turn-budget confound. The current text acknowledges the different budgets but does not treat them as a limitation.
- Add an explicit formula defining the "relative performance" metric (presumably (Interaction − Hidden)/(Full − Hidden)) to avoid ambiguity.

## Calibration

**Anchors examined:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| MACEval | /home/wg25r/review_agent/human_reviews_2026/gOQ4x4Ykyg.md | 2.0 | Far weaker: unbelievable claims, no released tasks. Ambig-SWE is clearly superior. |
| InteractComp | /home/wg25r/review_agent/human_reviews_2026/psEcdvhOJx.md | 3.5 | Topically closest (interaction + ambiguity). Ambig-SWE is stronger: larger scale (500 vs 210 issues), better decomposition, paired design for causal claims. |
| SLN | /home/wg25r/review_agent/human_reviews_2026/cGjTMuhqz3.md | 5.0 | Similar pattern of misleading headline claim from low bases. Ambig-SWE has more substantive analytical contribution. |
| SWE-Bench Pro | /home/wg25r/review_agent/human_reviews_2026/9R2iUHhVfr.md | 4.5 | Same domain. Ambig-SWE has more analytical depth (three-step decomposition) but similar methodological concerns. |
| HAL | /home/wg25r/review_agent/human_reviews_2026/vUaY1t64ZZ.md | 5.2 | Similar confounded-comparison issue, accepted as poster. Ambig-SWE has comparable analytical contribution. |
| Terminal-Bench | /home/wg25r/review_agent/human_reviews_2026/a7Qa4CcHak.md | 7.33 | Stronger: better curation, human-written solutions, many agents tested. Ambig-SWE has more analytical framework but weaker methodology. |
| LLMs Get Lost | /home/wg25r/review_agent/human_reviews_2026/VKGTGGcwl6.md | 8.0 | Far stronger: 200K+ conversations, rigorous methodology, clear decomposition. Ambig-SWE is significantly below this. |
| DSA | /home/wg25r/review_agent/human_reviews_2026/LwzBNQUmAi.md | 4.5 | Similar pattern of inflated relative gains from weak baselines, rejected. Ambig-SWE has more genuine contribution beyond the headline. |

Ambig-SWE sits above the papers rejected for inflated claims (SLN at 5.0, DSA at 4.5) because its conceptual framework and analytical insights are genuine contributions beyond the headline number. It sits around HAL (5.2, accepted poster) which had a similar confounded-comparison issue but enough contribution for a poster. It is clearly below the well-executed benchmarks (Terminal-Bench at 7.33, LLMs Get Lost at 8.0).

**Evaluation axes:**
- **Originality:** The three-step decomposition and paired evaluation design are original and useful. The benchmark itself (synthetic underspecification on SWE-Bench) is a reasonable but incremental extension.
- **Importance of research question:** High. Understanding how agents handle underspecificity is critical for real-world deployment.
- **Claims supported:** Partially. Within-model claims are well-supported; cross-model claims are confounded by unequal turn budgets. The headline "74%" claim is misleading.
- **Soundness of experiments:** The paired design is methodologically sound for within-model analysis. Cross-model comparisons are confounded. Detection evaluation has limitations for non-interacting models.
- **Clarity:** Generally well-written with clear takeaway boxes. The "relative performance" metric could be defined more explicitly.
- **Value to community:** The structured evaluation framework and empirical insights (exploration-first, effectiveness≠efficiency) are valuable for agent design.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>