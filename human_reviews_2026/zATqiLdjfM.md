# MARVEL: Multi-Agent RTL Vulnerability Extraction using Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Hardware security verification is a challenging and time-consuming task. Design engineers may use formal verification, linting, and functional simulation tests, coupled with analysis and a deep understanding of the hardware design being inspected. Large Language Models (LLMs) have been used to assist during this task, either directly or in conjunction with existing tools. We improve the state of the art by proposing MARVEL, a multi-agent LLM framework for a unified approach to decision-making, tool use, and reasoning. MARVEL mimics the cognitive process of a designer looking for security vulnerabilities in RTL code. It consists of a supervisor agent that devises the security policy of the system-on-chips ( SoC s) using its security documentation. It delegates tasks to validate the security policy to individual executor agents. Each executor agent carries out its assigned task using a particular strategy. Each executor agent may use one or more tools to identify potential security bugs in the design and send
the results back to the supervisor agent for further analysis and confirmation. MARVEL includes executor agents that leverage formal tools, linters, simulation tests, LLM-based detection schemes, and static analysis-based checks. We test our approach on a known buggy SoC based on OpenTitan from the Hack@DATE competition. We find that of the 51 issues reported by MARVEL, 19 are valid security vulnerabilities, 14 are concrete warnings, and 18 are hallucinated reports.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces MARVEL, a multi-agent framework for detecting security vulnerabilities in RTL (Register-Transfer Level) designs. MARVEL follows a Supervisor-Executor architecture, where the supervisor agent orchestrates six executor agents (Linter, Assertion, CWE, Similar-Bug, Anomaly, Simulator), each paired with tools (e.g., Synopsys SpyGlass Lint, VC Formal, Verilator) and small RAG stores. The authors evaluate MARVEL on a buggy dataset from Hack@DATE 2025 using OpenTitan SoC, and MARVEL reports 51 issues, 19 of which are validated as true security vulnerabilities. The authors also provide ablations to compare single-agent vs. supervisor-executor and per-agent contributions.

### Strengths
1. The design of MARVEL is clear and intuitive. Agents and toolchains are concretely described, including their prompts, actions, and failure-recovery loops.
2. MARVEL is fully automated and useful for real RTL workflows and complementary to human verification.
3. Evaluation shows that MARVEL can detect security vulnerabilities with a 19/51 TP ratio.

### Weaknesses
1. The novelty of this work should be argued more precisely. For example, the paper claims that MARVEL is the "first" multi-agent framework for hardware bug detection. However, SV-LLM is also multi-agent, albeit with weaker agent integration. The authors should clarify the novelty of MARVEL relative to existing studies in terms of what is technically new.
2. The evaluation relies on a non-public SoC, and the full ground-truth bug list is not available. This limits community reproducibility and prevents reporting standard detection metrics (precision/recall/F1).
3. There is no comparison between MARVEL and other recent LLM-for-RTL systems (e.g., SV-LLM, Self-HWDebug) on the same target, leaving MARVEL's relative advantage unclear.

### Questions
1. What is the complete ground-truth bug list per IP? Can you report precision/recall/F1?
2. Can you provide a quantitative comparison vs. other recent systems (e.g., SV-LLM) on the same SoC?
3. How does MARVEL perform using only open-source tools, excluding all the proprietary tools?
4. Can you elaborate on the technical novelty of MARVEl compared to other recent studies?

### Soundness
3

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
3

### Summary
This paper introduces MARVEL, a multi-agent LLM framework for detecting security vulnerabilities in RTL hardware designs. MARVEL uses a supervisor-executor architecture where a supervisor agent coordinates six specialized executor agents (Linter, Assertion, CWE, Similar Bug, Anomaly, Simulator) that employ different bug detection strategies and integrate with industry EDA tools (VC SpyGlass, VC Formal, Verilator).

### Strengths
1. This work claims to be the first comprehensive multi-agent approach for RTL security verification.  

2. The paper addresses hardware security verification. The approach handles hardware-specific challenges including clocking, concurrency, hardware CWEs, and FSM properties, and is evaluated on a real-world Hack@DATE benchmark.

3. MARVEL demonstrates solid engineering through practical integration of industry-standard EDA tools (VC SpyGlass Lint, VC Formal, Verilator), iterative refinement for error correction, and logical specialization of six agents combining tool-based and LLM-based approaches.

### Weaknesses
1. Table 2 presents security issues classified as 'correct' or 'incorrect,' but the paper never specifies who made these determinations. If the authors themselves judged their system's outputs, this constitutes circular reasoning and lacks objectivity. The absence of inter-rater reliability metrics, independent expert validation, or comparison with the official Hack@DATE answer key fundamentally undermines the credibility of the paper's central claims.

2. The evaluation is fundamentally incomplete without a recall metric. Without knowing the ground truth bug count, it is impossible to assess whether finding 19 bugs represents excellent coverage (e.g., 95% if 20 bugs exist) or poor coverage (e.g., 9.5% if 200 bugs exist). While the authors note that 'ground truth was not disclosed,' they could have requested this information from organizers or conducted independent exhaustive analysis with domain experts. This omission violates standard practices in security research, which require reporting both precision and recall.

3. The paper provides no evidence of generalization beyond a single benchmark. All 12 evaluated IPs come from the OpenTitan family, sharing the same design style, coding conventions, and security policies. Without evaluation on diverse SoCs—such as different RISC-V implementations, commercial IPs, legacy designs, or other application domains—there is significant risk of overfitting to benchmark-specific patterns. This single-benchmark evaluation is insufficient to support the paper's claims of general applicability.

4. The paper's efficiency claims lack quantitative support due to the absence of a human expert baseline. The authors claim MARVEL 'saves hours' compared to manual analysis, but provide no empirical evidence. With 37% precision, experts must manually validate 32 false positives—potentially requiring 160 minutes at 5 minutes per issue, costing approximately `$400` in expert time plus `$3` in LLM costs. Without demonstrating that this total cost/time is lower than direct expert analysis, the efficiency claims remain speculative.

5. The paper completely lacks statistical significance testing. Despite GPT-5's inherent stochasticity (temperature=0.15), each IP was evaluated only once, with no reporting of variance, confidence intervals, or results from multiple runs with different random seeds. The reported results could represent lucky outliers rather than typical performance. 

6. The paper lacks essential comparisons with state-of-the-art methods. FLAG, a publicly available baseline, was not evaluated on the same benchmark. Additionally, the paper provides no comparison with: (1) pure tool usage (VC SpyGlass/VC Formal without LLM assistance), (2) single-agent LLM with tool access, or (3) other claimed baselines.

7. The ablation study is incomplete and potentially biased. It evaluates only three IPs—the same ones used for model selection in Section 3.2—creating risk of data leakage. The study is missing systematic evaluations of: supervisor-only configurations, optimal 2-agent and 3-agent combinations, and all-agents-except-worst configurations. Given that the Similar Bug Agent contributes minimally (1 D&L) and the Anomaly Agent primarily generates false positives (16 FI vs. 3 D&L), it remains unclear whether all six agents are necessary.

8. The supervisor's decision-making process remains opaque. The paper states that the supervisor may call agents in any order and multiple times' but provides no specification of selection criteria, validation of decision quality, or analysis of optimality. Figure 5 shows that approximately 35% of actions involve navigation (List Dir, Read File), suggesting potential inefficiency. The non-deterministic workflow raises reproducibility concerns, and while Figure 7 displays call frequency, it does not assess decision quality.

9. The analysis of false positives is inadequate. While the authors note that 'many false positives arose from security suggestions rather than concrete flaws,' this explanation is insufficient. The paper lacks a systematic taxonomy of the 32 false positives by type, root cause, or responsible agent. Notably, the Anomaly Agent produced 16 false identifications (84% FP rate) and the CWE Agent produced 14, but no explanation is provided for these high rates. Without pattern identification across file types, security objectives, or agent combinations, iterative improvement is hindered.

10. The formulation of security objectives is underspecified and unvalidated. The paper claims that 148 security properties were 'correctly formulated,' but the formulation process remains opaque—is it automatic, does it involve manual review, or was expert validation performed? Without comparison to expert-defined ground truth, it is impossible to assess completeness (are critical properties missing?) or relevance (are irrelevant properties included?). Poor property formulation directly leads to searching for incorrect bugs or missing important vulnerabilities

11. The anomaly detection methodology is questionable. The approach uses DBSCAN clustering to identify outliers as potential bugs, but outliers do not necessarily represent bugs—unique but valid design patterns may be incorrectly flagged. The selection of hyperparameters (eps, min_samples) is unjustified, and semantic embeddings may miss security context (e.g., similar syntactic structures with different security implications). The empirical results confirm this weakness: the Anomaly Agent achieves an 84% false positive rate (16 FI vs. 3 D&L). Better alternatives exist, including AST-based pattern matching, data flow analysis, and supervised learning approaches.

12. The paper provides insufficient analysis of the multi-agent architecture's benefits. The comparison with a single-agent baseline is limited to three IPs and lacks depth. Missing analyses include: the value of redundancy when multiple agents identify the same bug, complementarity in detecting different bug types, coordination overhead (approximately 35% spent on navigation), diminishing returns as agents are added, conflict resolution mechanisms, and optimal agent ordering. Without this analysis, it is impossible to determine whether the multi-agent architcture's complexity is justified compared to simpler alternatives, such as a single agent with multiple tools or a reduced agent set.

### Questions
- What is the core methodological contribution beyond adapting existing multi-agent frameworks (Lee et al., 2024) to a new domain? The supervisor-executor architecture, RAG-based retrieval, embeddings-based similarity, DBSCAN clustering, and tool-calling APIs are all standard techniques. Could the authors clarify what algorithmic innovations, theoretical insights, or generalizable methodological advances distinguish this from an engineering application of existing methods? 

- Without novel ML methods or strong empirical validation (currently limited by single-benchmark evaluation, missing baselines, and lack of statistical testing), how does this work meet the research contribution standards of a top-tier ML venue versus being more appropriate for hardware security conferences or industry tracks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Hardware security verification is complex and time-consuming, relying on formal verification, linting, and simulation tests. To enhance this process, the authors propose MARVEL, a multi-agent LLM framework that unifies reasoning, decision-making, and tool use. MARVEL imitates how a designer identifies security flaws in RTL code through a supervisor agent that defines security policies and delegates tasks to executor agents. These agents use formal tools, linters, simulations, LLM-based detection, and static analysis to find bugs and report results for validation. Tested on an OpenTitan-based SoC, MARVEL detected 51 issues, with 19 confirmed security vulnerabilities.

### Strengths
(1) This idea is straightforward and can be easily understood.

(2) The experiments explore multiple GPT models for detecting hardware vulnerabilities, which is interesting.

### Weaknesses
(1) This paper focuses on the use of Large Language Models (LLMs) for hardware security; however, it remains unclear how the authors specifically leverage these models for vulnerability detection. The methodology section lacks sufficient detail on how the LLMs are integrated into the detection pipeline, what kind of data or prompts are used, and how the outputs are analyzed or validated. 

(2) My another concern about this paper  is the novelty issue. The paper appears to directly apply existing LLMs (e.g., GPT-5) to the hardware security domain without providing new insights or deeper understanding. Moreover, there are many existing works that have explored similar topics using LLMs for hardware vulnerability detection. The paper fails to discuss how it relates to or differs from these prior studies, which makes its contribution seem quite limited.

(3) The experimental evaluation is also insufficient. For example, the paper only investigates GPT-based models, but does not consider other popular alternatives such as Meta’s LLaMA 3 or Google’s Gemini, which could provide a more comprehensive comparison for vulnerability detection.

(4) Finally, the writing quality requires significant improvement. In several critical sections, it is difficult to understand what the authors actually did in terms of experimentation and analysis, or what motivated their methodological choices. Clarifying these aspects would strengthen the paper considerably.

### Questions
Please refer to my comments for more details.

### Soundness
2

### Presentation
2

### Contribution
2
