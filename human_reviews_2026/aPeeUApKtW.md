# SWE-eval: Trajectory-Enhanced Evaluation for Agentic Issue Resolution

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Agents and Language Models (LMs) demonstrate significant advancements in software engineering, particularly in issue resolution. Current benchmarks can qualitatively assess the correctness of generated patches. However, they lack mechanisms for quantitatively evaluating the trajectory, which is important to reveal the point of improvement. To obtain understanding of issue-resolving agents' working processes, we propose SWE-eval, a trajectory-augmented evaluation framework. SWE-eval additionally assesses a coding agent's reasoning trajectory across three dimensions: (1) Efficiency, measured by resource consumption; (2) Logical Consistency, where Intra-turns measures the logical consistency within a single turn and Inter-turns measures logical consistency across multiple conversation turns; (3) Tool Utilization, for which we design a metric Info-gain to assess how much new information the tool provides for solving problems. Our experiments on three agents and nine LMs demonstrate that SWE-eval effectively reveals underlying interpretations of agent performance and can guide development of more effective agents. First, our evaluations show that elevating trajectory-aware metrics is crucial for enhancing the % Resolved. Second, we trace divergent agent behaviors to shallow exploration, missing backtracking, and loop entrapment. We also show that fine-tuning on agents risks overfitting and scaling LMs improves trajectories. Third, LLM-based evaluations align closely with expert judgments and exhibit consistent stability, serving as reliable proxies.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes SWE-eval, a trajectory-augmented evaluation for issue-resolving agents that scores (i) Efficiency (#tokens, #turns), (ii) Logical Consistency, and (iii) Tool Utilization. It also reports separability of resolved vs. unresolved cases, agent/LM case studies, and some inter-rater statistics.

### Strengths
The paper provides quantitative analyses for each metric. Reliability analysis is also conducted: reports stability and human alignment (e.g., ICC and mean-diffs) for LLM-based scoring.

### Weaknesses
Limited novelty; largely a packaging of known factors. Most components of the proposed “trajectory evaluation” (efficiency via #Tokens/#Turns, consistency via Inter-/Intra-turn, tool use via %Tool Success/Info-gain) are not new individually. The contribution reads as a consolidation of existing ideas applied to SWE-bench agent traces,.

Correlation to causation leap without interventions. The paper repeatedly infers “because success correlates strongly with trajectory-aware metrics, optimizing these metrics should improve %Resolved.” That is an unsubstantiated causal jump. There are no experiments, such as targeted modifications that improve a specific metric while holding others fixed to establish that moving the metric causes gains.

Unclear practical utility beyond diagnosis. The paper emphasizes that the metrics “diagnose failure modes,” but it is not shown how practitioners should act on them to improve agents or benchmarks. Do these metrics guide dataset curation, agent retraining, or guardrail design better than simpler baselines? The application and downstream decisions are underspecified.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SWE-eval, a trajectory-augmented evaluation framework that assesses coding agents across three dimensions: (1) Efficiency: resource consumption, (2) Logical Consistency: stuck-in-loop, intra-turn and inter-turn consistency, and (3) Tool Utilization: tool success rate and information gain. This paper evaluates three agents (SWE-agent, OpenHands, Moatless) with nine language models on SWE-bench-Lite and SWE-bench-Verified, demonstrating that trajectory metrics correlate with success rates and reveal distinct failure modes. LLM-based evaluators show strong alignment with human judgments.

### Strengths
**1. Originality**: The three-dimensional framework propose new metrics, such as Info-gain metric and systematic trajectory evaluation that presents meaningful contributions.

**2. Experiment Setup**: In addition to human validation, this work conducts experiments on three coding agents across nine models, supporting findings with experimental results. 

**3. Writing Clarity**: This work presents structured writing with clear motivation.

### Weaknesses
**1. Limitations In Methodology**

- **Info-gain validation**: The paper does not validate that Info-gain actually measures information gain vs. general response quality.
- **Prompt sensitivity**: No ablation studies on how prompt engineering affects LLM judge scores.
- **Human baseline insufficiency**: Only 3 experts, unclear sample size, and no raw inter-annotator agreement reported before consensus.

**2. Limitations In Experiments**

- **Insufficient case study**: Django-12700 may not be representative.
- **Missing baselines**: No comparison to simpler heuristics, such as edit distance for loop detection, TF-IDF for Info-gain.
- **Agent confounds**: Different agents use different tools, so trajectory metrics may conflate agent reasoning quality with tool choices, making it unclear whether observed differences (e.g., more turns, lower tool success) reflect inferior agent capability or merely architectural differences in how agents decompose tasks.
- **Missing analyses:** Some critical analyses are missing, such as the error analysis of when trajectory metrics fail to predict outcome, the discussion of computational overhead introduced by trajectory evaluation, and actionable guidance on how to use metrics to improve agents.

**3. Limitations In Generalizability:**

- **Limited Language/task specificity**: All experiments on Python, but unclear if findings can generalize and transfer to other programming languages and SWE tasks.
- **Benchmark contamination risk**: Models may have seen SWE-bench instances during training.
- **Cost analysis missing**: LLM-based evaluation can be very expensive, but no cost-benefit analysis and use alternatives are provided.

### Questions
My questions are following several aspects mentioned in weakness:

- Can you provide evidence that Info-gain measures information rather than correlating with simpler features (response length, confidence, tool success)?
- How sensitive are LLM judge scores to prompt variations?
- Can you provide human evaluation details to validate the support, such as how many trajectories were annotated by human experts, the raw inter-annotator agreement (before consensus), why three human experts can sufficiently support the conclusion?
- For the identified failure modes (shallow exploration, missing backtracking, loop entrapment), what percentage of failures does each account for?
- What is the cost (time/money) of trajectory evaluation vs. patch-only evaluation? Is it practical for large-scale use?
- Have you tested on non-Python repositories? How do your evaluation framework and findings transfer to other programming languages and SWE tasks?
- Does tool utilization unfairly favor agents with more tools? How do you control for this confound?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes SWE-eval, a trajectory-augmented evaluation framework for agentic software issue resolution that goes beyond patch correctness by measuring (i) Efficiency, (ii) Logical Consistency, and (iii) Tool Utilization, with experiments on SWE-bench-Lite/Verified showing diagnostic value and reasonable alignment with expert ratings.

### Strengths
- Clear, useful shift from outcome-only scoring to process-aware evaluation; the three axes and their concrete metrics form a coherent rubric that exposes failure modes such as shallow exploration, missing backtracking, and loop entrapment. 

- Compelling cross-agent/LM analyses on SWE-bench-Lite and Verified: associations between lower turns/tokens and higher %Resolved, and ICC-based comparisons that show LLM-judge scores correlate with experts (strong on Info-gain, moderate on Intra-turns). 

- Case studies and tables highlight concrete failure patterns (e.g., loop traps; oversized patches from mis-extracted files) and suggest design fixes for agents and benchmarks (loop breakers, backtracking, stricter patch extraction).

### Weaknesses
- Reliance on LLM-as-judge needs tighter validation. Alignment with experts is uneven (e.g., weaker for Inter-turns), and prompts/models are central to metric values. Please provide fuller prompt/aggregation details, rater independence (ensuring the judge is not the actor model), seed sensitivity, and calibration (e.g., z-score or temperature scaling against human anchors). Also report per-metric confidence intervals/bootstrap for each model/agent. 

- Loop detection is brittle. The exact-string hash for Stuck-in-Loop may miss semantically equivalent repetitions (minor rephrasings) or action-level loops (same tool invocations with different wording). Consider a semantic or action-trace criterion (normalized tool call + args; Levenshtein/embedding similarity thresholds). 

- Benchmark and reporting scope. Main results focus on SWE-bench-Lite/Verified with a 30-call cap; broader generalization (e.g., SWE-bench-Live, multilingual variants, industrial datasets) is mostly deferred to appendix. Include per-repo/domain stratification, cost/latency breakouts, and framework-robustness checks (same trajectories scored across different agent shells) in the main paper.

### Questions
Which exact models/prompts were used for Inter-/Intra-turns and Info-gain, and were judges ever the same family as the actors? Could you report cross-judge variability, and a cross-model triangulation (e.g., swap in a very different judge) to show the conclusions persist?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new evaluation framework for SWE task, termed SWE-eval. It is a trajectory-augmented evaluation framework. It considers three parts, 1) efficiency, 2) logical consistency, and 3) tool utilization. The experiments on 3 agents and 9 LLMs demonstrate that the proposed evaluation can effectively reveal underlying interpretations of agent performance.

### Strengths
1. The topic and direction are practical. 
2. The codes are provided, enabling reproducibility.
3. The motivation is clear, and the case studies are comprehensive.

### Weaknesses
1. The sub-capture and the content in Figure 1 are inconsistency. 
2. It is not a new evaluation of the SWE task or a new format of the SWE task. It seems a normal analysis of the trajectory of the agent during the researcher pay effort on the SWE bench. 
3. The token efficiency is easy to detect and is a normal metric. And the Stuck-in-Loop problem seems common and can be avoided during the rollout stage.
4. Instead of these common metrics, what about conducting evaluations on different sub-tasks of the SWE task, like file reading, bug localization, patch writing, etc.?
5. In Table 1 and Table 3, please explain the meaning of the number with a green background.
6. In Table 1, for Inter-turns, the difference between resolved and unresolved trajectories is not significant.

### Questions
See Weaknesses part.

### Soundness
3

### Presentation
4

### Contribution
3
