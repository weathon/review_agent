## Summary
VARP is a VLM-based agent framework for playing action role-playing games (ARPGs) using only screen-level visual input, evaluated on *Black Myth: Wukong* (BMW). The paper defines a 13-task benchmark, releases a human gameplay dataset of 1,000 records with synchronized keyboard/mouse logs, and proposes two novel modules: Self-Optimizable Action Generation (SOAG), which synthesizes dodge/attack counter-macros by observing enemy attack animations; and Decomposable Task-Specific Auxiliary modules (DTSA), which routes complex combat decisions through parallel specialized sub-modules to mitigate VLM attention dilution. The system is compared against a Cradle-adapted baseline and human novice players.

---

## Strengths

- **BMW as a genuinely hard, underexplored testbed.** Unlike Minecraft or ALFWorld, BMW provides a high-fidelity AAA ARPG with near-zero in-game text cues for action guidance and complex continuous combat mechanics. Establishing a structured benchmark in this domain—with difficulty ranging from trivially easy to humanly challenging—fills a concrete gap that existing game-agent benchmarks do not cover.

- **DTSA's empirically grounded insight on VLM attention dilution.** The paper makes a specific, testable design claim: decomposing a large decision-making prompt into parallel specialized sub-modules (enemy status, combat mode, health monitoring, spell-skill timing, integration) reduces hallucination and forgetting compared to monolithic prompting. The ablation in Figure 4 supports this: removing DTSA degrades performance on tasks that were previously solved reliably (e.g., Defeat Wolf/Soldier drops from 100% to 60%, Defeat Wolf/Swordsword from 80% to 60%). This is a non-trivial, actionable design insight for VLM-based agents beyond gaming.

- **SOAG's data-driven combat macro synthesis.** Generating new action macros by having the VLM categorize enemy attack animations and encode counter-responses as annotated Python functions is a distinctive idea that requires no predefined reward signal. Figure 6 documents macros being refined across combat iterations—from "dodge 4 times then attack 5 times" to interleaved dodge-counterattack patterns—providing qualitative evidence of iterative tactical adaptation.

- **Human gameplay dataset with synchronized logs.** 1,000 records from 200 volunteers with paired screenshot footage and keyboard/mouse logs is a tangible, reusable resource. The empirical note that >90% of task 11/12 attempts were discarded provides honest evidence of task difficulty.

---

## Weaknesses

### Fatal
None that fully invalidate the work, but the major issues below collectively undermine the paper's empirical reliability to a degree that requires substantial revision before the claims can be trusted.

### Major

- **Critical numerical inconsistencies between Figure 3, Table 3, and the narrative text.** Section 4.4 states "In task 9, the VARP agent's average success rate was 40%" and "task 10... average success rate was 20%." Yet Figure 3's table shows GPT-4o at 100% for both Defeat Crow Diviner (task 9) and Defeat Battleguard (task 10). These figures are irreconcilable: Table 3 (VARP GPT-4o: 60% on task 9, 40% on task 10) and the ablation table Figure 4 (full pipeline: 60% on task 9, 40% on task 10) independently agree with the text, but contradict Figure 3. Similarly, human novice success on task 10 is described as 15.63% in the text but shown as 100% in Figure 3's table. These are not rounding or parser artifacts—they indicate fundamentally different numbers for the same experimental condition, which severely undermines confidence in all reported results.

- **Task 12 (Defeat Wandering Wight) is absent from Figure 3.** Table 2 defines 13 tasks, including task 12 as "Very Hard," but Figure 3 lists only 12 entries, skipping directly from Guangzhi (task 11) to Autonomous Navigation (task 13). No explanation is given. This omission means the performance evaluation is incomplete for a full third of the "very hard" combat tasks.

- **Photo mode pause converts a real-time ARPG to a turn-based problem, but is not treated as a core limitation.** The paper converts real-time combat into quasi-turn-based interaction by pausing the game during every VLM inference (Section 4.3). The defining challenge of the ARPG genre is the perception-action loop under real-time constraints; this setup sidesteps that challenge entirely. The paper's framing—"Can VLMs play ARPGs?"—implies real-time play, but the evaluation answers a weaker question. No latency measurements are reported, so readers cannot assess how far the system is from real-time viability. This limitation should be foregrounded prominently rather than disclosed as a one-line implementation detail.

- **The abstract's "90% of easy and medium-level combat scenarios" claim is not supported by the data.** The only "Middle" difficulty task is task 9 (Crow Diviner). VARP achieves 60% on this task (Table 3, GPT-4o) or 40% as reported in the Section 4.4 text. Neither value—nor any average combining easy and middle tasks—justifies a "90%" headline claim. This inflation of the primary result is misleading.

- **The human-guided trajectory system—the second major architectural contribution—has no ablation.** Figure 4 ablates only SOAG and DTSA within the action planning system. The human-guided component is evaluated only as a case study for task 13 (40% success vs. 0% without it), with no systematic comparison across combat tasks. Without a proper ablation, the human-guided system's independent contribution to task performance is unestablished.

- **Five trials per task is insufficient for statistically reliable conclusions.** With 5 trials, a difference of one success separates 40% from 60%. Key comparisons (task 9: VARP 60% vs. Cradle 20%; task 10: 40% vs. 0%) rest on 3 vs. 1 and 2 vs. 0 successes, respectively. No variance estimates or confidence intervals are reported anywhere in the paper.

### Minor

- **Task count inconsistency across sections.** The abstract states 13 tasks (76.9% combat), the introduction states 12 tasks (75% combat), and Section 4.2 defines "10 basic + 3 challenging = 13 tasks." This is never reconciled and affects interpretation of all task-based claims.

- **"Data entry" is undefined in the dataset.** The paper reports "1,000 valid data entries" but never defines whether an entry is a task episode, trajectory segment, screenshot-operation pair, or other granularity. This omission makes it impossible to assess the dataset's scale and density.

- **Incorrect cross-reference in Section 3.3.** The text directs readers to "Sec. 3.1" for dataset collection details; this content actually appears in Section 4.1.

- **SOAG's restriction to dodge + light attack combinations is unexplained.** The paper limits generated action macros to these two atomic operations, excluding heavy attacks, spell skills, and other mechanics. The justification for this restriction is absent. It raises questions about whether SOAG can learn truly novel combat behaviors or is effectively generating a small family of predetermined patterns.

- **Skill curation and human-guided retrieval are underspecified.** The paper uses text-embedding-ada-002 for similarity but does not report the top-k retrieved, whether visual embeddings factor into human-guided retrieval, the n-frame lookahead size, or the conditions that trigger action library updates.

### Tiny

- The DTSA architecture is described as "similar to an MLP," which is an inapt analogy (it is prompt decomposition with a final integration call, not differentiable computation).
- Task difficulty levels are manually assigned without operational criteria (e.g., empirical human success rates or average episode length), making reproducibility of the difficulty taxonomy difficult.

---

## Nice-to-Haves

- **Attempt real-time operation and report results honestly.** Even a failed attempt at unpaused gameplay with latency breakdown (per-module inference time) would substantially strengthen the paper—either showing that photo mode is a temporary engineering limitation with a plausible path forward, or that VLMs are fundamentally too slow for this setting.
- **RL baseline for at least one task.** The paper positions VARP against RL but provides no experimental comparison. The authors cite existing RL projects on BMW; even partial results from one such system on a shared task would substantiate the positioning claim.
- **Failure case analysis with screenshots.** Showing 2–3 failure cases with the agent's input screenshot, its decision, and a post-hoc analysis would reveal whether failure bottlenecks are perceptual, reasoning, or execution—critical for understanding whether better VLMs would help.
- **Cost and token efficiency reporting.** Running GPT-4o with 5+ parallel sub-modules per decision is expensive. Reporting API cost per task or per encounter would allow the community to assess practical viability relative to RL training costs.

---

## Removed Points
*These points are flagged as removed; treat them with caution.*

- **Missing related work criticisms (Harsh Critic §2):** Per review protocol, missing references are not cited as weaknesses since their existence cannot be independently verified.
- **"Table 1 is not rigorous enough as a scientific comparison" (Harsh Critic §1.3):** Table 1 is a qualitative positioning table, not a quantitative evaluation. Critiquing it as if it were a controlled experiment is scope creep.
- **"Paper motivation not tied to ICLR-level scientific questions" (Harsh Critic §1.1):** This is a systems and benchmark paper exploring VLM capability boundaries in a new domain. Demanding a formal hypothesis-testing structure imposes a different paper type's standards on a systems contribution.
- **"Only GPT-4o used for Cradle comparison" (Harsh Critic §4.5):** The paper explicitly frames this as an architecture comparison, and using the same model for both sides is the correct control. This is not a weakness.
- **"Comparison with Cradle limited to tasks 1–10" (Harsh Critic §4.5):** Cradle was adapted for combat tasks and was not designed for very-hard tasks requiring human guidance. Excluding tasks outside Cradle's adapted scope is reasonable, not a flaw in the experimental design.
- **"Unfair human comparison because humans play real-time while agents pause" (Harsh Critic §4.4):** This asymmetry favors humans (who play in real time under real pressure), not the agent. Per review guidelines, comparisons that are asymmetrically biased in favor of the baseline should not be counted as weaknesses for the agent method.
- **No ethics/consent statement, no broader-impact section (Harsh Critic §4.1, §5.3):** These are not scientific weaknesses at ICLR for this type of paper.
- **Generic strengths ("well-written," "topic is important," "experiments are extensive"):** These apply to any competent submission and are not specific strengths of this paper.

---

## Novel Insights
The most transferable insight from this paper is the DTSA decomposition principle: for multi-criteria sequential decisions in partially observable environments, routing each decision criterion to an independent VLM sub-module with bounded context outperforms presenting all criteria in a single monolithic prompt. The ablation provides concrete evidence that attention dilution causes qualitative failures on *easy tasks that were previously solved*, not just difficult ones—suggesting the issue is not simply harder reasoning but active interference between criteria in long prompts. This is an actionable design principle for any VLM agent facing structured multi-factor decisions. The SOAG concept points toward an interesting direction—VLMs synthesizing rule-based behavioral responses from visual pattern recognition of opponent animations without any reward signal—though the current implementation is narrow (dodge + light attack only) and the optimization is one-shot per encounter rather than iterative in a rigorous sense.

---

## Evaluation Summary

**Originality:** Moderate. SOAG and DTSA are novel modules with clear design rationale. The domain (AAA ARPG, zero text guidance) is genuinely unexplored for VLM agents. The overall pipeline extends Cradle-style modular prompting rather than representing a fundamental methodological departure.

**Importance of the research question:** High in principle. Understanding VLM capability boundaries in action-intensive visual environments is timely. However, the photo-mode conversion substantially weakens the practical significance of the findings.

**Claims well-supported:** Weak. The paper's core empirical claims are undermined by mutually inconsistent numbers across Figure 3, Table 3, and the narrative text; insufficient trial counts; a missing ablation of the human-guided system; and a missing task in the evaluation table. The ablation for DTSA and SOAG is directionally coherent, but the reliability of specific percentages is in doubt given the contradictions elsewhere.

**Soundness of experiments:** Poor. The combination of photo-mode conversion, 5 trials per task, missing task 12 in Figure 3, and mutually contradictory success rates across sections indicates the experimental setup was not executed or reported with sufficient care for an ICLR submission.

**Clarity of writing:** Below average. The high-level narrative is readable, but task count discrepancies across sections, an incorrect cross-reference, and especially the contradictions between figure tables and prose text indicate insufficient manuscript preparation.

**Value to the research community:** Moderate, contingent on corrections. The dataset and benchmark have genuine utility if the reported results are reconciled. The DTSA design principle is the most community-valuable takeaway.

**Contextualization relative to prior work:** Adequate for the broad framing (screen-only vs. API agents, VLM vs. RL) and for the direct Cradle comparison, which is appropriately adapted and fairly evaluated.