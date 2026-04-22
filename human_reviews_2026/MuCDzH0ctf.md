# ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 8, 4, 6

## Abstract
Autonomous web agents solve complex browsing tasks, yet existing benchmarks measure only whether an agent finishes a task, ignoring whether it does so safely or in a way enterprises can trust. To integrate these agents into critical workflows, safety and trustworthiness (ST) are prerequisite conditions for adoption. We introduce ST-WebAgentBench, a configurable and extensible framework designed as a first step toward enterprise-grade evaluation. Each of its 375 tasks carries one or more ST policies (3,057 in total), concise rules encoding constraints, and is scored along six orthogonal dimensions (e.g., user consent, robustness). Tasks span three difficulty tiers for fine-grained capability profiling, and a “Modality Challenge” disentangles vision-only from DOM-only information retrieval, isolating the contribution of each perceptual modality to agent failures. Beyond raw task success, we propose the Completion Under Policy (CuP) metric, which credits only completions that respect all applicable policies, and the Risk Ratio, which quantifies ST breaches across dimensions. Evaluating three open state-of-the-art agents shows their average CuP is less than two-thirds of their nominal completion rate, revealing substantial safety gaps. To support growth and adaptation to new domains, ST-WebAgentBench provides modular code and extensible templates that enable new workflows to be incorporated with minimal effort, offering a practical foundation for advancing trustworthy web agents at scale.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces ST-WebAgentBench, a new benchmark designed to evaluate web agents not only on task completion but also on safety and trustworthiness in realistic enterprise scenarios. It includes 222 tasks with associated safety policies across six dimensions like user consent and error handling. The benchmark proposes new metrics such as Completion under Policy (CuP) that measure task success while respecting safety policies.

### Strengths
- Most agentic evaluations focus on performance and this paper addresses a critical gap by integrating safety and trustworthiness into web agent evaluation. 
- Provides a diverse set of policies that agents need to comply and a diverse set of task to test this compliance. 
- Open-source with extensible design and human-in-the-loop mechanisms for practical use.

### Weaknesses
- It seems important SoTa Agentic implementations are missing. Please see Questions section below.
- Threat model and details of ST evaluations are missing as well. Please see Questions section below.

### Questions
- What is the threat model? Please clearly define it clearly in the main section. Line 216 mentions prompt injection. How it was constructed? Is user benign? What is the state of environment;
- How adherence to policies is checked? Is it an LLM judge? Is there any study to check robustness of that judge (i.e. it may introduce a certain bias into evaluation)?
- What specific models were used in experiments? AWM, WebVoyager, WorkArena are all agentic scaffolding, and performance will vary depending on which backbone model was used. Is GPT-4 used in all experiments? In which case, this is an old model, not designed for agentic use-case. For example, recent GPT-5, GPT-o, Gemini-2.5 series incorporate Instruction Hierarchy, which should allow adherence to hierarchical policy and shown to be effective against prompt injection attacks. But one needs to use it with proper tool-usage API. Besides, 1) computer use agent from Claude; 2) UI-Tars; 3) Kimi operates differently than agents used in the paper. I'm not sure if authors are planning to evaluate those?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presnets ST-WebAgentBench, a benchmark suite designed to evaluate the safety and trustworthiness (ST) of autonomous web agents in enterprise scenarios. Unlike prior benchmarks that focus solely on task completion, the proposed benchmark pairs 222 realistic tasks with 646 policy instances across six safety dimensions. The paper also introduces new metrics (CuP, pCuP, Risk Ratio) to measure policy-compliant completions and violations. Empirical results on three state-of-the-art agents show significant gaps between solely task completion and policy adherence, highlighting critical risks for large-scale deployment.

### Strengths
The paper makes a strong contribution by introducing a benchmark that evaluates web agents not only on task completion but also on safety and trustworthiness. I appreciate the clear definition of six policy dimensions and the formal hierarchy that governs them, which provides a structured way to reason about permissible actions. The metrics such as Completion-under-Policy and Risk Ratio are well-motivated and transform qualitative policy adherence into quantitative measures. Integrating these checks into BrowserGym with minimal architectural changes is a practical design choice that supports reproducibility. The empirical finding that CuP is significantly lower than raw completion highlights an important gap in current evaluation practices.

The methodology is technically sound and well-documented. The benchmark includes a diverse set of tasks and policies, and the modular design using YAML and JSON templates makes it extensible. Evaluators like is_sequence_match and is_input_hallucination cover common failure modes effectively, and the inclusion of human-in-the-loop simulation for consent checks is clever. The placement of POLICY_CONTEXT before task instructions aligns with the policy hierarchy and likely reduces unintended overrides. Reporting experimental budgets and evaluation protocols adds transparency and strengthens replicability.

In summary,
1. The paper addresses a critical gap. The benchmark directly targets safety and trustworthiness, which are prerequisites for real-world enterprise adoption but largely ignored by existing web agent benchmarks.
2. The benchmark covers six orthogonal dimensions (user consent, boundary, strict execution, hierarchy, robustness, error handling), derived from both academic taxonomies and expert interviews, ensuring relevance and breadth
3. Introduction of Completion-under-Policy (CuP) and Risk Ratio provides actionable, fine-grained evaluation of agent behavior under policy constraints, moving beyond raw completion rates.
4. The paper benchmarks three open-source agents, showing that policy adherence dramatically reduces effective completion rates, and identifies which safety dimensions contribute most to failures.

### Weaknesses
I have several concerns. The benchmark’s 222 tasks, while enterprise-relevant, are restricted to three applications and English language, potentially limiting generalizability to broader workflows and multilingual settings. Another concern is the skew in policy distribution, which may bias results toward certain violation types while underrepresenting others like error handling. The reliance on prompt-level policy injection could conflate adherence with prompt compliance, missing violations that would persist under stricter enforcement. The benchmark’s application set, while useful, may not capture the complexity of enterprise environments with multi-tenant privacy and transactional workflows. Additionally, the binary nature of CuP might be too rigid, as it does not account for recoverable or low-severity violations. These factors could limit the generalizability of the findings.

To improve the paper, consider adding confidence intervals for reported metrics and sensitivity analyses for policy placement and evaluator thresholds. Introducing a graded CuP metric that weights violations by severity would provide a more nuanced view of agent performance. Expanding coverage to include robustness and error-handling scenarios through adversarial DOM changes or synthetic UI faults would strengthen the benchmark. Combining prompt-level injection with programmatic enforcement mechanisms could help distinguish compliance from true policy reasoning. Formalizing properties of CuP and including a proof sketch would elevate the metric from a practical tool to a theoretically grounded measure.

Future work could explore baselines that integrate policies architecturally, such as shielded planning or SMT-based validation, to test whether these approaches reduce the gap between CR and CuP. Adding tasks that require reasoning across multiple policy families would challenge agents to handle policy arbitration. Evaluating robustness against advanced attacks like data exfiltration or origin isolation breaches would broaden the security dimension. Human evaluation on a stratified sample could validate the alignment between automatic checks and expert judgment. Finally, releasing policy-violation traces would enable deeper root-cause analysis and foster research on learning-from-intervention strategies.


In summary,
1. The benchmark’s 222 tasks, while enterprise-relevant, are restricted to three applications and English language, potentially limiting generalizability to broader workflows and multilingual settings.
2. Only three agents are evaluated, and all experiments use pass@k runs due to API cost constraints. This limits the statistical robustness and breadth of the empirical findings.
3. The scalability analysis shows sharp degradation in CuP as policy count increases, but the benchmark does not yet model the full complexity of real-world, multi-organizational policy environments.

### Questions
1. How do you ensure that policy injection does not unintentionally alter task semantics or introduce bias in agent decision-making?
2. How did you validate that the six safety dimensions are truly orthogonal? Were there cases where policies overlapped or conflicted, and how were these resolved?
3. How were the 222 tasks and 646 policy instances selected and paired? Was there a systematic process to ensure coverage and avoid bias toward specific workflows?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces ST-WEBAGENTBENCH, the first benchmark dedicated to evaluating safety and trustworthiness in web agents across 222 enterprise-style tasks. The paper pairs each task with policy constraints spanning six ST dimensions (user consent, boundary enforcement, strict execution, hierarchy adherence, robustness, and error handling) and proposes Completion-under-Policy (CuP) as a new metric that credits only policy-compliant completions. Experiments on 3 SOTA open source agents reveal that average CuP (15.0%) is substantially lower than raw completion rate (24.3%), exposing critical safety gaps that worsen as policy load increases. Overall, I believe this is a valuable contribution, If the authors can resolve my concerns I would be happy to raise my score.

### Strengths
1. The paper addresses a critical gap in web agent evaluation. Current benchmarks only measure whether agents finish tasks, completely ignoring safety aspect. The paper makes a convincing case that this is a serious problem for actual enterprise deployment.
2. The scalability study (policy‑load vs. CuP) convincingly demonstrates that current agents do not gracefully handle even modest policy stacks, highlighting a concrete research bottleneck.
3. Table 1 gives a clear, simple comparison of the benchmarks.

### Weaknesses
1. The evaluation is limited to three SOTA open‑source agents; there is no comparison with the latest closed-source models baselines (e.g., GPT, Claude, Gemini), so the behavior of frontier models for policy-prompting remains untested.
2. Only 3 applications and 222 tasks may not capture the full heterogeneity of enterprise web workflows (e.g., finance, health, legacy systems etc.). The authors acknowledge this in the paper but don't really discuss implications.
3. The current “POLICY_CONTEXT” setup assumes the LLM will reliably follow text-only constraints. But prompt-based controls are fragile - prone to injections, getting lost in long contexts, and being ignored. So when a policy is violated, is that an architectural issue or just the LLM not following instructions?
4. Using pass@3 with only a single "sweep" per agent means each task was run 3 times total. If so, doesn’t the 2–5% all-pass@3 suggest high variance?

Minor Suggestions / Nitpicks -
In Figure 2, the legend and bar chart use "all-pass@2", whereas the main text in Section 4.2 refers to the "all-pass@3". Please make them consistent.

### Questions
1. The sharp decline in CuP as the policy count increases is a key finding. Do your results provide any insight into the root cause? Is it a failure of the LLM's reasoning capabilities when presented with many constraints, or is it an architectural issue where the agent's control loop fails to systematically check policies before acting?
2. Your policy injection method prepends the POLICY_CONTEXT to the system prompt. Did you experiment with alternative methods, such as providing policies just-in-time based on the current UI state or using a verifier model to approve actions? How sensitive do you believe the results are to this specific prompt engineering choice?
3. Can you also add confidence intervals or significance tests for the key results in Figure 2 and Table 3, especially given the small number of runs.
4. The POLICY_CONTEXT template (Appendix E.4) is very detailed. Has the impact of template length on agent performance been evaluated? 
5. Figure 3 shows CuP declining sharply with increasing policy count. At what point does the policy context exceed reasonable prompt lengths? Have you tested with even larger policy loads (10+, 20+) to see if there's a breaking point?
6. Could you provide human baselines (with and without policies shown) to calibrate difficulty and policy load effects? Comparing human completion rates with and without policies would help clarify whether the sharp CuP decline is unique to agents or if policy enforcement is generally challenging.
7. The results show that User-Consent and Strict-Execution are the dominant sources of risk. It would be interesting to add a brief qualitative analysis or an example from the logs showing a typical failure for each of these dimensions to give readers a more concrete understanding of how the agents are failing.

### Soundness
3

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
This paper introduces ST-WebAgentBench, a benchmark for evaluating safety and trustworthiness of web agents through 222 tasks with 646 policy constraints across six dimensions. The authors propose new metrics (CuP, pCuP, Risk Ratio) that jointly assess task completion and policy compliance. Experiments reveal a significant gap: agents achieve 24.3% completion rate but only 15.0% policy-compliant completion. While the problem is important and timely, the work suffers from limited scope, methodological weaknesses, and insufficient experimental depth that undermine its claimed contribution as a comprehensive enterprise readiness benchmark.

### Strengths
1. The paper addresses a genuine gap in web agent evaluation. Current benchmarks measure only task success, ignoring whether agents complete tasks safely or within policy constraints. This matters for real-world deployment where unsafe successes can cause serious harm.
2. The hierarchical policy framework (organizational > user > task) is well-motivated and reflects real enterprise governance structures. The formalization provides a principled approach to reasoning about conflicting constraints.
3. The central empirical finding is significant: fewer than two-thirds of successful completions respect all policies, and performance degrades sharply as policy count increases (18.2% to 7.1% CuP as policies grow from one to five+). This reveals critical scalability issues.
4. The benchmark infrastructure is extensible, building on BrowserGym with modular policy templates that enable community expansion.

### Weaknesses
1. Only 222 tasks across three applications (GitLab, ShoppingAdmin, SuiteCRM) in English cannot support claims about "enterprise readiness" or comprehensive safety evaluation. Where are tasks for email, document collaboration, financial systems, HR platforms, or communication tools? The authors position this as "the first benchmark" and standard for enterprise deployment, but it covers only a narrow slice of enterprise workflows. The generalization claims far exceed what this limited set can justify.

2. Policy violation detection uses LLM-based "fuzzy matching," introducing non-determinism without any inter-annotator agreement validation. How do we know automated evaluations are correct? The "simulated user-confirmation mechanism" is vaguely described—if it auto-approves requests, it cannot test whether agents appropriately seek permission. These methodological gaps undermine result credibility.

### Questions
1. Can you provide actionable guidance for building policy-aware agents? What architectural principles, design patterns, or failed approaches have you identified?
2. Among completed tasks, what fraction violate policies? This would isolate safety gaps from general capability limitations.

### Soundness
3

### Presentation
3

### Contribution
3
