# Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
While Multi-Agent Systems (MAS) excel at complex tasks, their growing autonomy with operational complexity often leads to critical inefficiencies, such as excessive token consumption and failures arising from misinformation. Existing methods primarily focus on post-hoc failure attribution, lacking proactive, real-time interventions to enhance robustness and efficiency. To this end, we introduce SupervisorAgent, a lightweight and modular framework for runtime, adaptive supervision that operates without altering the base agent's architecture. Triggered by an LLM-free context filter, SupervisorAgent intervenes at critical junctures to proactively correct errors, guide inefficient behaviors, and purify observations. On the challenging GAIA benchmark, SupervisorAgent reduces the token consumption of the Smolagent framework by an average of 29.68% without compromising its success rate. Extensive experiments across five additional benchmarks (math reasoning, code generation, and question answering) and various SoTA foundation models validate the broad applicability and robustness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces SUPERVISORAGENT, a lightweight, runtime “supervisor” that sits on top of a multi-agent system (MAS). A non-LLM prioritized adaptive filter detects four trigger conditions (sub-agent completion, error occurrence, inefficient behavior, excessive observation length) and, when activated, a three-stage intervention pipeline decides among four actions: approve, provide_guidance, correct_observation, run_verification. The goal is to reduce token usage and improve robustness without modifying underlying agents. Experiments on GAIA (Smolagent) and several code/math/QA benchmarks report sizable token savings with roughly maintained—or sometimes improved—accuracy, and an ablation suggests observation “purification” is the main driver of efficiency, while guidance/correction help accuracy.

### Strengths
1. The paper addresses a critical and highly relevant challenge in the field of agentic AI. The prohibitive token cost and lack of operational robustness are major barriers to the real-world deployment of complex Multi-Agent Systems. The work's focus on runtime efficiency is a valuable contribution.

2. The empirical validation is extensive and rigorous. The authors test their method on six different benchmarks spanning multiple domains, which strongly supports the claim of general applicability.

3. The paper highlights reduced variance and a more compact per-problem token distribution after supervision—an under-discussed but practically important property for cost predictability. 

4. The prioritized LLM-free filter (sub-agent summary → error → inefficiency → length) is a neat engineering idea to avoid constant, costly oversight while still catching the most harmful failure modes. The concrete heuristics (e.g., length threshold ≈3,000 chars; loop/inefficiency checks) are spelled out sufficiently to reproduce.

### Weaknesses
1. The core idea—a meta-controller that detects errors/loops/long observations and either prunes context or issues guidance—resembles prior runtime oversight, routing, or budgeted-reasoning controllers. The paper would benefit from a sharper contrast to existing runtime monitors/filters (as opposed to offline compression or architectural agent redesign). As written, the conceptual delta may feel incremental.

2. The authors justify Smolagent because it relies less on “powerful external tools,” isolating the MAS. But in many real MAS workloads, tool calls dominate; it is unclear whether the same gains hold with tool-heavy agents, where tool output is the main token pressure and mis-tooling is the dominant failure mode. The paper acknowledges tokens meticulously but not end-to-end cost/latency including tool latencies and external compute.

3. The authors justify using Smolagent as the base framework because its capabilities stem from "internal agentic interactions rather than powerful external tools". While this provides a controlled environment, it also means the baseline might be inherently less efficient and robust. Applying SUPERVISORAGENT to a weaker baseline could inflate its perceived benefits.

4. The prioritized conditions (e.g., 3,000-char cutoff; loop detectors) appear hand-tuned. How sensitive are results to these thresholds across tasks and models? The paper claims model-agnosticism but does not deeply analyze failure cases where the supervisor fires too often/too rarely on different backbones.

### Questions
1. What fraction of ActionSteps trigger each condition on GAIA (by level) and on each external benchmark? Please report a table of invocations per action and supervisor token cost so readers can reason about net savings.

2. Why was the ablation study limited to the 30 most token-intensive tasks from the GAIA validation set? Do the individual contributions of the Correction, Guidance, and Purification modules remain consistent when evaluated across the entire dataset, including less token-intensive tasks?

3. The current filter is rule-based. Have you considered a learning-based approach for the supervisor? For example, could a policy be trained to decide when and how to intervene, potentially leading to a more adaptive and less heuristic-driven system?

4. Have you tested a tool-rich agent configuration (e.g., retrieval-augmented browsing with multi-source aggregation)? If the majority of cost/latency comes from tools rather than LLM tokens, does the supervisor still help?

### Soundness
3

### Presentation
2

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
This paper addresses critical challenges in MAS, such as excessive token consumption and error propagation, by introducing SUPERVISORAGENT, a lightweight, non-intrusive meta-agent framework for runtime supervision. The approach is innovative in its use of an LLM-free adaptive filter to trigger targeted interventions, enabling proactive error correction, inefficiency guidance, and observation purification without modifying the base agents' architecture. The experiments are comprehensive, spanning multiple benchmarks and models, and demonstrate efficiency gains on a specific baseline. However, the paper's claims of a "significant Pareto improvement" are undermined by unfavorable comparisons to SOTA baselines presented in its own results, raising concerns about the method's broader applicability and impact.

### Strengths
The paper effectively highlights underexplored pain points in MAS, such as runtime inefficiencies from error propagation and excessive observations, which lead to high token costs (for example, up to 2 million tokens on GAIA tasks). This framing is fresh and relevant, building on recent MAS advancements while addressing a critical paradox: increased autonomy often reduces robustness and economic viability.

SUPERVISORAGENT is a lightweight, non-intrusive framework that integrates seamlessly with existing MAS without architectural changes. Its LLM-free adaptive filter for triggering interventions (e.g., at agent-agent, agent-tool, or agent-memory points) is efficient and proactive, focusing on runtime process control rather than static optimizations. The action space (e.g., approve, provide_guidance, correct_observation, run_verification) is well-formalized, and the memory-augmented context window enables holistic oversight. This design is modular and broadly applicable, as validated across diverse benchmarks and models (e.g., GPT-4.1, Gemini-2.5-pro, Qwen3).

### Weaknesses
The paper claims a "significant Pareto improvement" (reduced tokens without compromising success rates), but this holds only against the weak Smolagent baseline (50.91% accuracy, 527.76K tokens reduced to 371.12K). In contrast, the paper's own Table 1 shows SOTA baseline AWorld achieving higher accuracy (60.00%) at much lower cost (128.27K tokens). The supervised system (Smolagent + SMAS) is thus both less accurate and more expensive than AWorld, contradicting the Pareto claim. While the authors justify Smolagent for isolating "agentic interactions" over external tools, this does not excuse avoiding direct SOTA integration or comparison. It makes the method's impact limited.

The LLM-free filter relies on hardcoded heuristics (e.g., detecting repetitive actions or observations exceeding 3,000 characters). These thresholds lack justification, sensitivity analysis, or ablation, raising concerns about brittleness and potential overfitting to Smolagent or GAIA. Without robustness tests (e.g., varying thresholds across datasets), the filter's generalizability is questionable.

Token reductions vary inexplicably between main results (29.68% on full GAIA) and ablation on 30 high-token tasks (50.13%). This suggests benefits are skewed toward long-tail cases, potentially masking uneven performance. The abstract and conclusions claim "average" savings (e.g., 29.45%) without clarifying this distribution, which could mislead readers and requires more transparent analysis.

### Questions
Why was SUPERVISORAGENT not applied to SOTA baselines like AWorld? If integrated, could it further reduce AWorld's token costs (e.g., from 128K to below 100K) while maintaining its 60% accuracy? If incompatible, please explicitly discuss this limitation in the paper.

How does SUPERVISORAGENT's own overhead (e.g., from context aggregation or LLM calls in interventions) factor into total token costs?

### Soundness
3

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
This paper addresses two critical and intertwined problems in modern Multi-Agent Systems (MAS): operational inefficiency (excessive token consumption) and lack of robustness (failures from error propagation). The authors argue that existing methods are either post-hoc (analyzing failures after they occur) or design-time (optimizing the agent architecture itself), with a lack of tools for proactive, real-time intervention .




To fill this gap, the authors propose SUPERVISORAGENT, a lightweight, modular meta-agent framework designed for runtime supervision. The framework's core is a lightweight, LLM-free adaptive filter that monitors agent interactions. This filter triggers interventions only at critical junctures, which it identifies as: 1) explicit error occurrences, 2) inefficient behaviors (like loops), or 3) excessive observation lengths.





When triggered, the SUPERVISORAGENT (which is an LLM) analyzes the system's state using a rich context window and performs one of four actions: approve, provide_guidance, correct_observation, or run_verification .


The authors demonstrate the system's effectiveness by applying it to the Smolagent framework on the GAIA benchmark, achieving a 29.45% reduction in token consumption without any loss in task success rate. The paper further validates this approach's generalizability across five other benchmarks (for math, code, and QA) and multiple state-of-the-art foundation models .

### Strengths
The paper's framing of MAS inefficiency as a runtime process control problem is a fresh and valuable perspective.


The hybrid "LLM-free filter + LLM supervisor"  is a very strong and practical design choice that balances cost and capability.


The 29.45% token reduction on GAIA with no accuracy loss is a headline-worthy result, strongly supported by the data.



The ablation in Table 3 is a highlight, perfectly justifying the three-part design of the supervisor's intervention strategies (Purification for efficiency, Correction/Guidance for robustness) .


The method is shown to be model-agnostic (Figure 4b) and benchmark-agnostic (Table 2), proving it is a general framework and not a one-off trick.

### Weaknesses
1. The paper's main claim is "Stop Wasting Your Tokens", and it reports a ~30% token reduction. However, it appears this 30% saving applies only to the base agents (Smolagent). The paper never states the **token cost of the SUPERVISORAGENT itself**. The supervisor is an LLM (e.g., GPT-4.1)  and is called every time the filter is triggered. To make a true claim of efficiency, the paper must report the net token savings (i.e., Baseline_Tokens - (SMAS_Agent_Tokens + Supervisor_Agent_Tokens)). Without this, the primary claim of token reduction is incomplete.

2. The paper focuses exclusively on token efficiency. However, the proposed framework introduces an extra LLM call into the execution loop every time an intervention occurs. This will "stop the world" and add wall-clock latency. It's possible the system saves tokens but becomes significantly slower to run. A comprehensive "efficiency" analysis should include latency, especially when the interventions are designed to cut down on inefficient loops (which also consume time).

### Questions
1. Do the token savings reported in Table 1 and 2 (e.g., the 29.68% on GAIA)  account for the tokens consumed by the SUPERVISORAGENT itself? If not, what is the net token savings for the entire system (base agents + supervisor)?

2. The ablation study (Table 3) shows that "Purification" is the main source of token savings, while "Correction" and "Guidance" are key for accuracy . This suggests that for tasks where the baseline is already robust (high accuracy), a "Purification-Only" supervisor might offer the best cost-benefit. Have you considered this variant?

3. The "Excessive Observation Length" trigger is a hardcoded "3,000 characters". How sensitive is the framework's performance (both token savings and accuracy) to this hyperparameter?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses two critical challenges in MASs, 1) Operational Inefficiency (measured through token usage here), 2) Robustness Failures (error propagation, misinformation, etc.). Unlike prior methods that focus on post-hoc error attribution and diagnosis, the paper introduces SupervisorAgent, a meta-agent, that monitors the interactions in an MAS (agent-agent, agent-tool, agent-memory), uses an LLM-free "adaptive-filter" to determine high-risk interactions, and if invoked, the SupervisorAgent intervenes through actions like proactively correcting errors, guiding agents away from inefficient behaviors (like loops), and purifying excessively long or noisy observations to reduce token load.

The authors implement the SupervisorAgent over the SmolAgents framework, and compare against strong baselines across several benchmarks, finding that the intervention reduces token consumption by average as much as 29.45% without losing accuracy. The paper concludes with an ablation study of the various SupervisorAgent interventions, finding that purification leads to efficiency gains, whereas correction and guidance are necessary for performance.

### Strengths
1. The paper introduces a novel, lightweight, non-intrusive monitoring of MAS. This could potentially become a major MAS development pattern going ahead.
2. The paper is well-to-read, and educational from the perspective of understanding MAS design, issues with current MAS, and nicely introduces the proposed intervention.
3. Evaluation: The authors validate the intervention across 6 benchmarks, and across 3 leading LLMs, including open and proprietary models.

### Weaknesses
1. The overhead introduced due to SupervisorAgent is not described in detail. 
2. The details about the working of adaptive filter are not described. Since one of the core features of the SupervisorAgent is "lightweight" monitoring, the authors should clearly describe how the adaptive filter works without LLMs, especially to identify "inefficient behavior" which seems to be a highly subjective criteria unlike the other 2 high-risk interactions identified.
3. Many MAS proposed in the past have included an additional agent that acts as a verifier or quality-assurance agent (for example, ChatDev includes a "Tester" agent). However, one issue with such interventions is that often the verifier can itself fail, introduce misinformation, or propagate errors. The authors do not discuss the failure modes or failure rate of complex SupervisorAgent actions like "correct_observation".

### Questions
1. Do the reported token costs for SMAS in all figures/tables take into account all supervisor token costs?
2. Can the authors describe the overhead introduced by SupervisorAgent, not only in terms of the token costs, but also in terms of wall-clock times, or number of additional LLM calls invoked (especially relevant since once a SupervisorAgent intervention will be invoked, the agent system's execution will need to be halted).
3. Can the authors' comment on the interaction of SupervisorAgent when implemented in other MAS frameworks, which could include significant interaction with external environments (for example, through MCP)?

### Soundness
4

### Presentation
3

### Contribution
3
