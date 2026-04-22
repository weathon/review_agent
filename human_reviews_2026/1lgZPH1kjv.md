# HiddenBench: Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but may also replicate collective reasoning failures observed in human groups. Yet the absence of a theory-grounded benchmark makes it difficult to systematically evaluate and improve such reasoning. We introduce HiddenBench, the first benchmark for evaluating collective reasoning in multi-agent LLMs. It builds on the Hidden Profile paradigm from social psychology, where individuals each hold asymmetric pieces of information and must communicate to reach the correct decision. To ground the benchmark, we formalize the paradigm with custom tasks and show that GPT-4.1 groups fail to integrate distributed knowledge, exhibiting human-like collective reasoning failures that persist even with varied prompting strategies. We then construct the full benchmark, spanning 65 tasks drawn from custom designs, prior human studies, and automatic generation. Evaluating 15 LLMs across four model families, HiddenBench exposes persistent limitations while also providing comparative insights: some models (e.g., Gemini-2.5-Flash/Pro) achieve higher performance, yet scale and reasoning are not reliable indicators of stronger collective reasoning. Our work delivers the first reproducible benchmark for collective reasoning in multi-agent LLMs, offering diagnostic insight and a foundation for future research on artificial collective intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces HiddenBench, a benchmark designed to test the collective reasoning abilities of multi-agent systems. It addresses the concern that multiple agents might fail to share information correctly, similar to problems observed in human groups. The benchmark is based on the "Hidden Profile" paradigm from social psychology, where individuals are given different pieces of information and must communicate to find the correct answer. The full benchmark contains 65 tasks and is used to evaluate 15 different LLMs. The study found that these multi-agent systems often struggle to combine their distributed knowledge, showing that a model's size or general reasoning skill doesn't guarantee strong collective reasoning, though some models performed better than others.

### Strengths
1. The paper introduces human-subject experiments, which are reasonable and could be an essential baseline.
2. The paper is easy to understand. The motivation is clear.
3. The experiment is comprehensive, and the findings are interesting.

### Weaknesses
1. My primary concern regards the experimental setting and the practical applicability of the proposed testbed. The benchmark's core assumption, i.e., each agent possesses unique information it must not share with others, seems disconnected from real-world multi-agent applications. It is difficult to envision a practical scenario that aligns with this rigid constraint.
2. The underlying mechanism for the multi-agent discussions appears artificial. Despite testing various prompting strategies (as shown in Table A1), the experiments operate on the premise that agents will inherently withhold their private information. This contradicts the nature of collaborative discussion, where claims and points are typically supported by evidence. The rationale for why an agent would deliberately hide the basis for its reasoning is unclear.
3. The setup suggests that the paper may be overcomplicating the problem. If the primary challenge is an artificial unwillingness to share information, a straightforward prompt (e.g., share all the information you have) could potentially resolve the core issue. Consequently, the complex experimental setting and the subsequent experiments might be addressing an artificial constraint rather than a fundamental challenge in multi-agent collaboration.

### Questions
See weaknesses.

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
This paper introduces HIDDENBENCH, a new benchmark for evaluating collective reasoning in multi-agent LLM systems. The authors ground this benchmark in the Hidden Profile paradigm from social psychology, unshared knowledge to arrive at a correct decision. The paper first provides a formalization and demonstrates in a small-scale study (Study 1) that GPT-4.1 agents replicate human-like collective reasoning failures, such as shared information bias. It then details the construction of the full 65-task benchmark. By evaluating 15 different LLMs, the authors show that these collective reasoning failures are persistent and that model scale does not reliably predict better collective performance.

### Strengths
1. The benchmark's core design, which evaluates agents based on asymmetric information distribution (the Hidden Profile paradigm), provides a novel and theoretically-grounded angle for assessing collective reasoning in LLMs.
2. The proposed HIDDENBENCH benchmark covers multiple scenarios. The 65-task set is drawn from custom designs, tasks adapted from prior human studies, and a scalable automatic generation pipeline.
3. This work is easy to follow and reproducible. The authors provide a dedicated reproducibility statement and include the full benchmark, scripts, and prompts in the supplementary material.

### Weaknesses
1. The claim of being the "first benchmark for evaluating collective reasoning" appears to be inaccurate. Prior work has already explored this area, for example, "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents".
2. The paper fails to provide complete examples of the tasks and agent interactions, including the specific information distributed to each agent, the full log of the resulting discussion, or the final reasoning from the agents. Without these qualitative examples, it is difficult to understand what the final agent discussions look like or to verify the claims of human-like failures.
3. The paper provides insufficient analysis of the impact of the communication protocol's depth. The number of rounds (T=15) is presented as a fixed parameter, but there is no ablation study or discussion on how this number affects the outcomes. It is unclear if the core conclusions would still hold with fewer or more rounds of discussion.
4. The experimental design relies on artificially hiding information. It may be testing the agents' willingness to share rather than their collective reasoning. The authors state agents are not told their information differs, but LLM agents, unlike humans, may be prone to simply stating all information from their context window.
5. The evaluation is limited to homogeneous agent groups. This is a significant simplification. Real-world collective intelligence often emerges from heterogeneous groups with diverse capabilities, biases, and knowledge bases. The benchmark's findings may not generalize to these more realistic mixed-model scenarios.
6. Typo: L202 aasinged → assigned, condtion → condition; L255: archived → achieved

### Questions
See Weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper formalizes the Hidden Profile paradigm from social psychology as a controlled framework for testing collective reasoning in multi-agent LLMs, and builds HiddenBench, a 65-task benchmark (crafted, adapted, and automatically generated). It compares post-discussion accuracy against (i) pre-discussion under Hidden Profile, (ii) pre-discussion under Full Profile (upper bound for individual reasoning), and (iii) human groups, using average/majority aggregation rules. Key result: communication helps, but post-discussion accuracy under Hidden Profile remains far below the Full-Profile individual baseline; some model families (e.g., Gemini) fare better, while “stronger at individual reasoning” does not guarantee stronger collective reasoning

### Strengths
- Clear, theory-grounded formalization. The paper precisely defines information splits (shared vs. unshared), decision rules (average/majority), and reference points, which cleanly separates “gain from communication” from “individual upper bound.

- Controlled probe + scalable benchmark. Study 1 uses GPT-4.1 groups vs. human groups (N=4, T=15 rounds; 30 runs/model-condition; 15-minute human chats) to verify the phenomenon; Study 2 scales to 65 tasks across 15 models.

- Strong empirical signals. Under Hidden Profile, GPT-4.1 improves from 0.008 → 0.233 after discussion (p<0.001), yet still lags the Full-Profile baseline (0.733). Humans show similar gaps. These give a crisp diagnosis of collective-reasoning limits.

### Weaknesses
- Limited scale/structure. Most tests use N=4 and a 3-option, elimination-style decision. It is not shown whether results hold with larger groups, different protocols (e.g., facilitator/blackboard), or production-style collaborative tasks.

- Process-level analysis is shallow. The paper notes premature consensus and shared-information bias, and that prompting styles rarely fix it, but does not quantify disclosure/uptake of unique facts over rounds

### Questions
- Could you run a staged ablation to separate “disclose → merge → decide”? For example: (i) forced round-1 disclosure of each agent’s private facts; (ii) a “secretary” agent merges evidence; (iii) free discussion. This would reveal the main bottleneck.

- Do results scale with N? If N grows, do we see worse information pooling or earlier (wrong) consensus? Any plans to add facilitator/blackboard protocols?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a new benchmark based on Hidden Profile paradigm, HIDDENBENCH,  measuring the collective reasoning abilities when agents keep asymmetrical information in multi-agent LLMs. The evaluation show that GPT-4.1 groups fails in integrating distributed information and perform even worse than single agent given full information.

### Strengths
- This work creatively applies Hidden Profile in evaluating the possible failures of LLM formed multi-agent system.
- The performance comparison between LLM based systems and human groups is interesting.
- The proposed benchmark is reproducible and scalable due the automated pipeline.

### Weaknesses
- Only hidden profile based collective reasoning are measured, it may not be generalizable to other multi-agent reasoning settings, e.g.  negotiation, competition and collaboration so on. Therefore it is not suitable to conclude that hidden profile collective reasoning is worse than single-agent full profile reasoning.

- The work conclude collective reasoning fails but there is no deep probe into why specific LLM architecture and reasoning augmentation fails.

- The automatically generated datasets might inherit the generation bias, which tends to lead failures of multi-agent reasoning.

### Questions
- Did you analyze why some LLMs performs better and how is the quality of the generated benchmark tasks?

### Soundness
2

### Presentation
3

### Contribution
3
