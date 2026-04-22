# SOCK: A Benchmark for Measuring Self-Replication in Large Language Models

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
We introduce SOCK, a benchmark command line interface (CLI) that measures large language models’ (LLMs) ability to self-replicate without human intervention. In this benchmark, self-replication is defined not only as an LLM's ability to create a functioning and running copy of itself, but also the ability for that self-replication to persist and occur across different computational contexts. Accordingly, we’ve developed a system to categorize LLMs based on broad self-replication capabilities in two general classes, Replication-Capability Levels (RCL) and Persistence-Capability Levels (PCL). Using a five-task suite based on practically manipulable modern CLI utilities and computer processes, experiments are orchestrated in a controlled environment with an LLM acting agentically. The performance of the LLM on agent tasks is then computed to produce an R-score (a quantitative evaluation of overall self-replication ability) and data used to categorize LLMs into specific RCL-PCL matrices. SOCK offers two primary contributions: (1) Provides the first formalized definitions and benchmark suite for evaluating LLM self-replication, with the goal of establishing a standard for future research; (2) Allows the industry to track the effectiveness of future multi-agent systems and mitigate potential self-replication threat vectors within them. The results compiled from evaluating a variety of open-weight and proprietary frontier models reveal significant obstacles to persistent self-replication and multi-agent systems, including context retention and multi-agent decision-making. We propose future research directions to safely reduce the severity of these obstacles, potentially lowering future risk of more functional multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SOCK, a command line interface (CLI) benchmark designed to measure self-replication capabilities in large language models (LLMs). The authors define self-replication broadly to include not only creating functional copies but also persistence across different computational contexts. The benchmark categorizes models using two metrics: Replication-Capability Levels (RCL) and Persistence-Capability Levels (PCL), evaluated through a five-task suite that progresses from basic file copying to cross-container replication.

### Strengths
The paper addresses a timely and critical research problem—LLM self-replication—which has significant implications for AI safety and multi-agent systems. The focus on persistence across computational contexts extends beyond simple replication, adding depth to the evaluation framework.

### Weaknesses
1. Due to the short page length, key details are omitted. For example:
* The R-score formula (Section 4.1) references constants (e.g., τ, Bᵢ) and task-specific baselines without providing their values or derivation process.
* The "intelligence components" (reasoning, tool use, recovery) are mentioned but not defined operationally, leaving ambiguity in how they are measured.
2. The results (Table 1) are presented without in-depth analysis. For instance:
* Why do models like Gemini-2.5-Flash outperform GPT-5 despite lower general capability? The explanation provided is vague and lacks empirical support.
* Error patterns and failure modes are not analyzed, missing insights into specific bottlenecks (e.g., context retention, tool misuse).

### Questions
Please see the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SOCK, a command-line interface (CLI) benchmark designed to evaluate large language models' (LLMs) ability to self-replicate without human intervention. The benchmark consists of five tasks with increasing complexity, categorized by Replication-Capability Levels (RCL 0-2) and Persistence-Capability Levels (PCL 0-2). The authors test eight frontier models across these tasks and compute an R-score that combines success, speed, stealth, intelligence, and resource efficiency.

### Strengths
1, Practical Implementation: Docker-based CLI benchmark with controlled environments is a solid engineering contribution.

2, Clear Task Categorization: The RCL-PCL taxonomy provides an intuitive framework for categorizing replication capabilities.

3, Multi-Model Evaluation: Testing 8 frontier models provides some comparative insights.

### Weaknesses
1, Severely Limited Scope.
- Only 5 tasks vs. 20 task families in RepliBench.
- Excludes critical capabilities (resource acquisition, weight exfiltration)
- Achieves only RCL 2/PCL 2, missing higher-level threats

2, Flawed Scoring Metric.
- Geometric mean penalizes failures too harshly
- Many hyper-parameters without justification

3, Unclear Threat Model.
The adversarial framing ("user aims to prevent replication; agent aims to maximize replication") is mentioned but not formalized or evaluated

4, Missing Analysis.
No failure mode analysis, no ablation studies, no discussion of why certain models succeed/fail.

### Questions
Could you compare your work with RepliBench and highlight the key differences and contributions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SOCK, a command-line benchmark for evaluating large language models’ (LLMs) ability to self-replicate. The framework defines Replication-Capability Levels (RCL) and Persistence-Capability Levels (PCL), implements five progressively difficult CLI tasks , and computes an overall R-score reflecting success, efficiency, and resource use.

### Strengths
S1: The proposed framework for RCL and PCL  provides a clear and valuable structure for decomposing and categorizing a multi-dimensional problem.

S2: The R-score design is a robust contribution. It looks beyond binary pass/fail rates to include V,  P, and S .

### Weaknesses
W1: The benchmark tasks mainly test basic command execution and do not capture the higher-level reasoning or planning implied by “self-replication.”

W2: The gap between the framing (which includes up to Level 5) and the actual measured capability (which ceilings at RCL 2/PCL 2) weakens the conceptual impact.

W3: The study focuses solely on CLI-based operations, omitting dimensions such as multi-step decision making, communication, or long-term coordination.

W4: Heuristic scoring function: The R-score aggregates multiple factors with fixed weights (e.g., $w_d, w_v, w_s... = 1$ 10), yet the paper provides no analysis of sensitivity, robustness, or interpretability.

W6: The experiments confirm intuitive expectations (more efficient models score higher) but do not yield new understanding of model behavior.

W7: The writing is clear, but much of the paper reads like documentation of a software tool rather than an analysis-driven research study.

### Questions
Q1: What motivated the specific choice of these five tasks and their associated difficulty levels, given the much broader 0-5 level taxonomy?

Q2: Can the RCL–PCL taxonomy be validated or compared to external behavioral definitions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces SOCK, a benchmark designed to systematically evaluate the self-replication capabilities of large language models (LLMs) within controlled environments.  It defines two hierarchical dimensions—Replication-Capability Level (RCL) and Persistence-Capability Level (PCL)—and implements five progressively challenging tasks, from file duplication to cross-container persistence.  The framework uses isolated Docker environments to measure replication success, efficiency, and stealth.  Experiments with eight mainstream LLMs show that while models can reliably reproduce themselves in simple settings, higher-level persistence and cross-environment replication remain challenging.  SOCK provides the first standardized foundation for studying autonomous replication behaviors in LLMs.

### Strengths
Originality:
The work is highly original in defining and operationalizing the concept of LLM self-replication. By introducing RCL and PCL as structured capability levels and implementing them in sandboxed environments, the paper turns a speculative and safety-sensitive topic into a measurable research direction. 

Significance:
The experimental setup is creative and technically complete, offering a reproducible platform for future work on AI autonomy and containment evaluation.

The paper does not have a significant advantage in terms of clarity and soundness.

### Weaknesses
1.  Limited contribution and insufficient depth: 
Although the paper demonstrates strong originality in framing the concept of LLM self-replication, its overall contribution remains narrow. The implemented benchmark covers only low-level replication scenarios, and the experimental scope is relatively small. As a result, the work lacks the richness and depth expected for a comprehensive study.
2. Insufficient experiments
The experimental setup is severely limited and lacks statistical depth.   Results are based on only five trials without reporting variance or significance tests, so the reliability of performance differences is unclear.   The scoring formula uses equal weights for all components but offers no justification or sensitivity analysis.   Tasks are independent and memoryless, preventing evaluation of long-term or cumulative replication.   Overall, the experiments validate feasibility rather than provide rigorous quantitative evidence.
3. Reproducibility and scalability issues
The provides no scalability analysis. No runtime or memory statistics are reported.  Key metrics such as stealth and intelligence are not accompanied by logs or examples, making it difficult to verify scoring.  Without detailed experimental traces or resource profiling, reproducibility and scalability remain uncertain, weakening the benchmark’s credibility for large-scale evaluation.

### Questions
1.  Experimental reliability
The experiments use only a few trials without variance or significance reporting.  Can the authors clarify how consistent the results are across multiple runs or random seeds?

2.  Scalability
How does the framework perform when the number of agents or replication tasks increases?  Is there any data on runtime, resource usage, or LLM-call efficiency?

3.  Higher-level replication
The benchmark currently stops at RCL2/PCL2.  Do the authors plan to extend it to more complex or long-term replication tasks, and how will safety be ensured in such experiments?

### Soundness
2

### Presentation
2

### Contribution
2
