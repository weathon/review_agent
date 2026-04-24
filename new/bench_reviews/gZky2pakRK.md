## Summary

HAICOSYSTEM is a framework for evaluating AI agent safety in holistic, multi-turn interactions among simulated human users, AI agents, and tool-equipped environments. The paper introduces 132 scenarios across seven domains, a multi-dimensional evaluation framework (HAICOSYSTEM-EVAL), and reports results from 8,700 simulated episodes across 12 models, arguing that evaluating the full user→agent→environment ecosystem surfaces risks invisible to isolated benchmarks.

## Strengths

- **Holistic scope (Table 1, §2):** HAICOSYSTEM is the only compared framework to jointly support multi-turn human→AI interaction, multi-turn AI→environment tool use, and both benign and malicious user intents in a single evaluation, addressing a clear gap in prior work.
- **Scale and breadth (§3.1, §5.1):** 132 scenarios spanning seven domains at three realism levels, 8,700 episodes, and 12 evaluated models provide a substantial empirical foundation for stress-testing agents across diverse settings.
- **Multi-dimensional evaluation (§4, Figure 2):** The seven metric groups (targeted, system, content, societal, legal, efficiency, goal) enable granular assessment of safety and performance dynamics across entire interaction trajectories, going beyond binary harm detection.
- **Organizational contexts:** Explicit inclusion of organizational scenarios (not just personal assistants) reflects a realistic and underexplored deployment setting.
- **Release commitment:** The pledge to release the simulation platform meaningfully lowers the barrier for follow-up research.

## Weaknesses

### Fatal
None.

### Major

- **Overclaiming in abstract regarding tool use and malicious intent:** The abstract states that models show "higher risks when interacting with malicious users and using tools simultaneously." However, this pattern is clearly contradicted by Llama3.1-405B (0.59 with tools vs. 0.63 without; §5.3, Figure 5) and absent in Llama3.1-70B (0.62 vs. 0.62). With only four representative models tested and no variance estimates, generalizing from two supportive models while the other half contradict or nullify the pattern overstates the robustness of the finding. While §5.3 appropriately notes Llama3.1-405B as an exception, the abstract's phrasing presents a model-dependent observation as a general tendency.
- **Binary risk threshold lacks calibration:** The headline claim that models "exhibit safety risks in 62% of cases" (Abstract, §5.2, Figure 3) rests on a binary threshold where any negative score on any dimension flags an episode as risky. The paper provides no score distributions (e.g., whether most "risky" episodes cluster near –1 or –10), sensitivity analyses under alternative thresholds, or evidence that this cutoff distinguishes meaningful failures from minor infractions. The manual validation (100 episodes, 90% accuracy, 0.8 Pearson correlation) assesses agreement on continuous or coarse-grained scores, not the validity of the binary threshold itself. Without this calibration, the strong quantitative prevalence claim is inadequately supported.

### Minor

- **Confounded goal-completion vs. safety analysis:** The reported positive correlations between goal completion and targeted safety (e.g., *r* = 0.71 for GPT-4-turbo, §5.2) pool benign and malicious scenarios. Because goal alignment and safety are aligned in benign scenarios but may be in tension in malicious ones, the pooled correlation is confounded by scenario type and is uninterpretable without disaggregation.
- **Limited generalizability of single-turn comparison:** The multi-turn vs. single-turn analysis (Figure 6, §5.3) uses only GPT-4-turbo as the agent and adapts existing jailbreak datasets (DAN, PAP, WildTeaming). While the comparison is suggestive, broader agent coverage and native multi-turn benchmarks would strengthen the claim that multi-turn dynamics uniquely amplify risk.
- **Concentration of GPT-4o roles:** GPT-4o serves as the user simulator, environment engine, and safety evaluator, creating potential circularity. Although the evaluator is validated against human judgments (100 samples, 0.8 Pearson), the paper does not assess whether GPT-4o systematically elicits behaviors that it subsequently flags. This is a standard limitation of LLM-based simulation paradigms but warrants more explicit discussion of external validity.
- **Risk dimension separability:** The six safety dimensions (TARG, SYST, CONT, SOC, LEGAL) are defined conceptually but the paper provides no empirical evidence (e.g., inter-dimensional correlation matrix or factor analysis) that they are separable in practice. If dimensions are highly correlated, reporting them separately may be misleading.

### Trivial

- **Figure 2 color-coding inconsistency:** The text defines LEGAL as a risk dimension on [–10, 0] (§4), but Figure 2 groups it under a green [0, 10] bar. This appears to be a figure-caption error and should be corrected.

## Nice-to-Haves

- Histograms or density plots of continuous dimensional scores to demonstrate the severity distribution underlying the binary risk ratios.
- A "benign without tools" condition in Figure 5 to isolate the main effect of tools from the interaction with malicious intent.
- Disaggregated correlation analysis between goal completion and safety by user intent and scenario type.
- A correlation matrix or factor analysis of the six risk dimensions to empirically justify the multi-dimensional framework.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"Structurally flawed" multi-turn comparison:** The harsh reviewer's claim that the single-turn vs. multi-turn comparison is "invalid" overstates the case. The paper compares the same scenarios under single-turn and multi-turn conditions, which is a reasonable experimental manipulation. The null result for WildTeaming is explained by the dataset's release date post-dating GPT-4-turbo's training cutoff—a factual observation, not an "unsupported post-hoc memorization hypothesis."
- **"Logically confounded" tool+malicious claim:** The absence of a "benign without tools" condition does not invalidate the comparison between malicious with-tools and malicious without-tools, which is the appropriate contrast for assessing whether tools increase risk specifically in malicious contexts. A benign without-tools condition would test the main effect of tools, not the interaction.
- **Thin validation criticism:** While 100 samples is small for 8,700 episodes, it is a reasonable initial validation effort for an automated framework and not a fatal flaw.
- **Missing appendix or proofs:** The parser strips appendix sections; they exist in the original submission.

## Novel Insights

None beyond the paper's own contributions.

## Suggestions

- Revise the abstract to tone down the generalization about simultaneous tool use and malicious intent, or explicitly qualify it as model-dependent (e.g., "some models show higher risks...").
- Report the distribution of continuous scores and test whether the 62% prevalence claim holds under stricter thresholds (e.g., any dimension < –5).
- Disaggregate the goal-completion vs. safety correlation by benign/malicious intent in the main text or appendix to clarify whether the pooled correlation is driven by scenario-type confounding.

## Score and Decision

**Calibration reasoning:** I compared this paper against several calibration anchors. *High-scoring* benchmarks include Cybench (avg 8.67), which used real professional CTF tasks with human difficulty metrics and granular subtask evaluation, and AgentHarm (avg 6.75), a narrower but useful agent misuse benchmark. HAICOSYSTEM has broader, more realistic scenario coverage than AgentHarm but less rigorous experimental controls than Cybench. *Medium-scoring* rejected papers include CASE-Bench (5.25) and 3CB (5.33), which had weaker motivation or more superficial results; HAICOSYSTEM substantially exceeds these in empirical depth and conceptual clarity. *Low-scoring* rejected papers like XSafety (3.5) had fundamental methodological weaknesses (e.g., small human evaluation, questionable dataset construction) that HAICOSYSTEM does not share. The paper's core contribution—a holistic simulation framework—is valuable and well-motivated, but the overclaiming in the abstract and the uncalibrated binary threshold are genuine issues that prevent a higher score.

**Score:** 6.0

**Decision:** Accept (Poster)

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>