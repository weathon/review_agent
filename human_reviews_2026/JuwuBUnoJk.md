# SABER: Small Actions, Big Errors — Safeguarding Mutating Steps in LLM Agents

- Decision: Reject
- Scores: 6, 6, 6, 2

## Abstract
Despite rapid progress in LLM agents, performance on long-horizon, tool-using tasks remains fragile. To better understand this fragility, we ask a simple question: \emph{do all actions contribute equally to failure?} Analyzing execution traces on $\tau$-Bench (Airline/Retail) and SWE-Bench Verified, we decompose trajectories into \emph{mutating} (environment-changing) vs.\ non-mutating steps and formalize \emph{decisive deviations}—earliest action-level divergences that flip success to failure. A logistic regression reveals that each additional deviation in a mutating action reduces the odds of success by upto $92\%$ on Airline and upto $96\%$ on Retail for SoTA models. In contrast, deviations in non-mutating actions have little to no effect. Errors also grow with context length as agents drift from role and act on stale constraints. Motivated by these observations, we introduce \cm{}, a model-agnostic, gradient-free, test-time safeguard that (i) adds mutation-gated verification, (ii) injects \emph{Targeted Reflection} before mutating steps, and (iii) performs block-based context cleaning. \cm{} delivers consistent gains—e.g., Qwen3-Thinking: +28\% \emph{relative} on Airline, +11\% on Retail, and +7\% on SWE-Bench Verified; Claude: +9\%/+7\%. We further identify ceiling effects in $\tau$-Bench, where annotation errors and underspecified tasks artificially cap model performance. To address this, we release $\tau$-Bench Verified, which restores benchmark headroom through targeted revisions. Our results argue for action-level analysis, targeted safeguards, and reliable evaluations as prerequisites for robust multi-turn agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates failure modes in long-horizon LLM agents, specifically focusing on tool-using settings. By analyzing execution traces in $\tau$-Bench and SWE-Bench Verified, the authors distinguish mutating actions—those that change the external environment—from non-mutating ones, and show through logistic regression that deviations in mutating steps overwhelmingly predict task failure, while deviations in non-mutating steps have negligible effect. Motivated by this finding, they propose SABER, a lightweight, model-agnostic test-time safeguard incorporating (i) mutation-gated verification, (ii) targeted reflection, and (iii) block-based context cleaning. SABER significantly improves success rates for state-of-the-art models across multiple benchmarks. The paper additionally identifies annotation and underspecification issues in $\tau$-Bench and releases a corrected $\tau$-Bench Verified to restore useful headroom.

### Strengths
1. Clear motivation from empirical insight, where a logistic regression model is used to show that mutation actions are the primary source of task failure.

2. Novel and practical safeguard approach that does not require model retraining and applies to any models.

3. Performance improvements across diverse models and benchmarks demonstrate method effectiveness.

4. $\tau$-Bench Verfied: they identify flaws and offer a cleaned version, which helps the community.

### Weaknesses
1. The proposed method, SABER, requires a human in the loop, where a confirmation message is sent to the user and the user determines the next action. While the experiments are conducted in a simulated manner, it is non-trivial to assess the effectiveness in real-world settings.

2. The effectiveness of the block-based context filtering component is not evaluated.

3. The contributions of the paper feel somewhat disconnected. The proposals of SABER and $\tau$-Bench Verified are not clearly linked. From my perspective, I would like to see a more detailed analysis of SABER.

### Questions
Have you tried your method to smaller LLMs? I am curious of the capability of smaller models to complete the mutation-gated verification and reflection.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies an interesting problem of impact of different actions on the final task success and failure in context of LLM agents. Particularly, the paper divides trajectories into mutating and non-mutating steps. Mutating steps modify the environment e.g. editing a file, while non-mutating steps do not modify the environment e.g. reading a file. The paper studies the impact of these actions on the final task success and failure - and devises strategies to reduce the impact of mutating steps on task failure, especially as the context window grows. Finally, the paper also proposes a tau-bench refined which is a cleaned version of tau-bench with removed ambiguity and preserved task coverage.

### Strengths
* The paper studies an interesting problem of impact of different actions on the final task success and failure in context of LLM agents.

* The paper provides interesting analysis on the role of mutating steps especially as the context window grows.

* The authors show the efficacy of the proposed approach on tau-bench and swe-bench-verified.

### Weaknesses
* "Mutating steps having higher impact on task failure then non-mutating steps" is an interesting observation. However, I would imagine this will be very intuitive. E.g for swe-bench-verified, if the agent wrongly edits a line of code, then fixing it post-edit will require the agent to realize its mistake and undo the edit, before continuing to fix the problem.

* The authors mention L105: "7% on SWE-Bench Verified, while both GPT-5 and Claude 4 gain further headroom once benchmark flaws are removed." Do the authors also release a refined version of SWE-Bench Verified? If so it will be interesting to provide more details and examples of flawed problems.

* While I understand the motivation for user gated verification for mutating steps especially as the context window grows, I have two questions:
    * Can the new setup be compared against previously reported benchmark numbers e.g. on SWE-Bench Verified?
    * In practice, how does it differ from current implementations of cli agents e.g. claude code which have require permissions while changing the codebase but not while reading different files?

* Also as in Sec. 4, targeted reflection is mentioned as a concise high salience summary at point of mutating steps. Can the authors provide ablations to show the impact of targeted reflection on the performance of the agent?

* Also how does "context based filtering" differ from context management in opensource scaffolds like OpenHands?

*  It will be important to have ablations on the impact of each of these components in Sec. 4 on the final performance of the agent.

### Questions
Please see the weaknesses section for some additional questions.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies why LLM-based agents fail in long, tool-using tasks. It separates actions into mutating (changing the state, like canceling a booking) and non-mutating (information-seeking). Using logistic regression on τ-Bench and SWE-Bench Verified, it finds that mistakes in mutating actions strongly predict failure.
To address this, the authors introduce SABER, a simple, model-agnostic safeguard that adds user checks for risky actions, targeted reflection to reduce drift, and block-based context cleanup. 
The paper also fixes annotation issues in τ-Bench, releasing τ-Bench Verified, which restores valid upper bounds.

### Strengths
1. The decomposition into mutating vs. non-mutating actions and formalization of decisive deviations provides a fresh lens on agent fragility, substantiated by significant regression results. 
2. SABER's design, requiring verification only at mutating steps, minimizes overhead while delivering robust gains 
3. Identifying and fixing τ-Bench flaws via τ-Bench Verified exposes hidden model headroom. 
4. Evaluations span open (Qwen3) and closed (GPT-5, Claude-4) models across domains (Airline, Retail, SWE), demonstrating generalizability without retraining.
5. Section 8 acknowledges SABER's test-time nature and simulator reliance, fostering future work on internalized safeguards.

### Weaknesses
1. The work is empirically solid but lacks formal analysis. This limits generalization beyond observed datasets.
2. Findings rely mainly on τ-Bench and SWE-Bench, with improvements shown mostly on their “Verified” versions. No tests on other benchmarks or robustness checks for noise, which risks overfitting.
3. SABER depends on human or simulated confirmations, reducing autonomy and adding latency. The paper acknowledges this but doesn’t measure the trade-off or failure cases when users reject valid actions.

### Questions
1. Why no evaluations on additional benchmarks to validate generalization beyond τ-Bench/SWE-Bench?
2. Why does full SABER underperform components on Retail Verified?
3. Why not quantify inter-annotator agreement for τ-Bench Verified revisions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper analyzes the execution traces on Tao-Bench (Airline/Retail) and SWE-Bench Verified and categorizes the steps to mutating and non-mutating actions. The authors then show deviations in mutating actions are the decisive predictors of failure. In contrast, deviations in non-mutating actions have little to no effect. 

The paper then introduces SABER, a model-agnostic, gradient-free, test-time safeguard that (i) adds mutation-gated verification, (ii) injects targeted reflection before mutating steps, and (iii) performs block-based context cleaning. The authors experiments show consistent gains with SABER on Tao-Bench and Swe-Bench verified.

### Strengths
- Improving Existing Benchmarks: Correct benchmarks are the essential to improve agentic systems performance. Inconsistency and incorrectness in the benchmark can result into incorrect design decisions in agentic systems and misguide the agent developers. Through multiple experimentation on Tao-Bench the authors have identified potential issues in the benchmark and released Tao-Bench verified which can help future agent developer

### Weaknesses
- Lack of Novelty: The first finding in the paper which identifies mutating steps (or state changing steps) to be the main cause of failures has been noted in previous work where the most common failure modes are state changing (or mutation steps as defined by this paper). For example, the failure modes detected here: https://arxiv.org/pdf/2405.15793. The authors may want to further expand their categorization into multiple types of actions, understanding which types of actions or sequences are likely to cause failures can be helpful for agent developers. Also, when it comes to SABER design the "mutation-gated human verification" is currently used in many production systems where irreversible actions need confirmation from the user. For example, command line execution or applying edits to a file in coding agents need user verificaiton. Similarly, summarizing the trajectories is a well-known technique used in SOTA production agent systems to keep the agent focused on the task, and it has also been noted to be effective in previous work, for example: https://arxiv.org/pdf/2503.07832.

### Questions
- Why Swe-Bench results are not reported for Claude and GPT-5 models?
- What was the base agent or the evaluation prompt that was used to evaluate the models on the benchmark?

### Soundness
2

### Presentation
2

### Contribution
2
