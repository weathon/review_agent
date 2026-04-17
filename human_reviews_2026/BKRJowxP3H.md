# VeriWeb: Verifiable Long-Chain Web Benchmark for Agentic Information-Seeking

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Recent advances have showcased the extraordinary capabilities of Large Language Model (LLM) agents in tackling web-based information-seeking tasks. However, existing efforts mainly focus on single-fact retrieval and rely on outcome-only verification, thereby limiting their scalability in realistic knowledge-intensive scenarios that involve long-horizon web tasks requiring large-scale retrieval and synthesis of information from diverse sources. In this work, we introduce VeriWeb, a novel verifiable long-chain web benchmark designed to facilitate the evaluation and development of web agents within realistic web environments. Our benchmark emphasizes two critical dimensions: (1) long-chain complexity, encompassing both breadth- and depth-oriented search tasks to assess how effectively web agents ensure comprehensive information coverage and consistent context tracking in multi-hop reasoning; and (2) subtask-level verifiability, where tasks are decomposed into a sequence of interdependent verifiable subtasks. This structure enables diverse exploration strategies within each subtask, while ensuring that each subtask-level answer remains unchanged and verifiable. The benchmark consists of 302 tasks across five real-world domains, each with a complete trajectory demonstration, annotated by human experts. Extensive experiments on VeriWeb using various agents powered by different foundation models reveal significant performance gaps in handling long-horizon web tasks, highlighting the need for more powerful agentic information-seeking capabilities

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new benchmark for web agents, primarily focusing on information retrieval / deep research tasks. The authors claim that their benchmark is novel and introduces features not previously seen in other benchmarks. 
1. Long, multi-question information retrieval - Each sample spans multiple questions that test both breadth and depth of reasoning, requiring agents to perform multi-hop information retrieval while maintaining contextual coherence across web pages.
2. Intermediate step verifiability - A large task broken into several, intermediate subtasks, each of which has verifiable, fixed ground truth answers.

The authors provide detailed analysis of benchmark’s data statistics and discuss the evaluation metrics used to prove the uniqueness and complexity of their benchmark. The authors use these metrics to evaluate several agentic frameworks and frontier models on their benchmark, showing generally low scores across metrics like Success Rate (SR) and Completion Rate (CR). They also provide a comparative analysis of how models and agent paradigms perform across different domains and across breadth vs depth-oriented tasks.

### Strengths
1. The proposed benchmark introduces long-horizon, multi-question challenge to deep research agents with verifiable subtasks, where the multi-question consistency feature is an important extension to existing benchmarks that largely focus on single-fact tasks or subjective report generation.

2. Empirical result shows that the benchmark is difficult for  most advanced agentic systems and frontiner models, which provides a a potential new direction the research community can make measurable progress towards.

3. The experiments and analysis are relative extensive, and covers a good representation of agent paradigms and frontier models to make sure that the observations of benchmark difficulty is not overfitting particular agent designs or models.

### Weaknesses
1. The main evaluation metrics this paper use are based on LLM as a judge, but there is no study of how well the LLM judge matches human assessment of answer quality on this dataset. This is especially important since this dataset claims to expose reliable subtask-level assessment of agent quality. If the evaluation itself is not stable enough, it puts into question all of the evaluation results presented in this work. One potential alternative, for instance, taken by the GAIA benchmark, is to specify the output format with task instructions and use stricter evaluator functions like string match.

2. One of the main contributions of this benchmark concerns subtask-level verification, which is important and useful for complex information retrieval tasks that is oft-neglected in many benchmarks. However, the paper fails to discuss this contribution in relation to previous work in the literature. For one, HotpotQA (which is cited by this paper) introduced supporting fact evaluation for reasoning, which is a proxy to checking the necessary reasoning steps are reached to arrive at the final answer. As a more recent example, the Agent Company [1] uses subtask rewards to evaluate agents on long-horizon agent tasks.

3. Some claims in the paper are unsubstantiated. For instance, L360 states "Search engine agents, constrained to passive retrieval, typically achieve the lowest success rates." But the agents that achieved the lowest success rates in Table 2 are Browser-Use Agents and Multi-agent Systems.

[1] TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks. (https://arxiv.org/pdf/2412.14161)

### Questions
1. Why does the paper use Browser actions as the main metric for step / action efficiency? Just because this is how humans do it with a browser doesn't mean it's the most effective (as evidenced by the results) or efficient way for agents to do it.

2. Some examples in Figure 4 have SR that's 0 < SR < 1, does this mean that SR is not a binary 0/1 metric, or is this averaged over several runs? What does it mean if a task is 50% successful?

### Soundness
2

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
4

### Summary
The authors introduce a benchmark that tests web-based language agents on complex, multi-hop information-seeking tasks. Each task comes from a human browsing session, where annotators solved real web problems and recorded the steps needed to find and verify facts. The dataset breaks these trajectories into verifiable subtasks. VeriWeb scores agents using three measures: Success Rate for full-task completion, Completion Rate for subtask accuracy, and Action Efficiency for how effectively they use the browser. The authors test several large language models and find that agents often locate some facts correctly but struggle to plan, search deeply, and keep results consistent across steps.

### Strengths
- Tasks are long-chain and information-dense, combining multi-hop retrieval and synthesis with subtask-level verifiability.
- The benchmark introduces several evaluation metrics, including task success rate, completion rate, and action efficiency.
- Human-annotated trajectories provide empirically grounded task structures.

### Weaknesses
- The benchmark’s subtask-level verifiability requires each sub-answer to be fixed and unambiguous, but real-world web tasks often involve context-dependent or time-sensitive information. This design choice may therefore underrepresent the uncertainty present in realistic settings.
- The absence of a human performance baseline makes it hard to interpret how well current agents perform relative to human proficiency on the same tasks.
- Evaluation only uses gpt-4o as the judge, with no analysis on potential LLM judge bias or comparison against human-annotated labels.

### Questions
- Would it be possible to run a quantitative breakdown of failure modes across agents?

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
The paper introduces VeriWeb, a new benchmark for evaluating long-horizon, web-based information-seeking agents. VeriWeb focuses on long-chain complexity (requiring multi-hop reasoning and synthesis across diverse sources) and subtask-level verifiability (allowing fine-grained evaluation of intermediate steps). The dataset comprises 302 human-annotated tasks across five domains, each decomposed into verifiable subtasks. Experiments with multiple agents powered by different foundation models show low success rates, highlighting the difficulty of realistic web reasoning.

### Strengths
- Proposes a novel benchmark emphasizing both long-chain reasoning and verifiable subtasks.

- The dataset is diverse and human-annotated, covering five realistic domains.

- The experimental evaluation is comprehensive, testing multiple agent paradigms and models.

- The paper provides insightful analyses of action efficiency and task difficulty, helping identify weaknesses in current web agents.

### Weaknesses
- Unclear which LLM generated task instructions and subtasks.

- The reasonableness of subtask decomposition is not independently validated.

- Details of the human demonstration process (e.g., annotator number, quality checks, or fairness) are limited.

- Many tasks involve hundreds of steps, but efficiency guarantees or annotation consistency are not analyzed.

- The LLM-as-a-Judge metric may not align with human evaluation; human verification would strengthen credibility.

- Single-run experiments due to API costs limit statistical reliability.

- No human baseline is reported to contextualize task difficulty.

- Error analysis could be broader and more quantitative.

### Questions
- Which model was used to generate and decompose the tasks?

- How is the quality or coherence of subtask decomposition verified?

- What measures ensure fairness and accuracy in human demonstrations?

- How do the authors justify using LLM-as-a-Judge without human correlation studies?

- Could results be re-evaluated with multiple runs perhaps with open-source models to report variance?

- Is there a plan to report human performance per difficulty level?

- Can the authors share the API cost estimates and efficiency trade-offs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces VeriWeb, a novel verifiable long-chain benchmark, intended to address critical limitations in existing web agent evaluation—specifically the overreliance on single-fact retrieval and outcome-only validation. VeriWeb comprises 302 human-annotated tasks across five real-world domains. The benchmark mandates long-chain complexity (requiring large-scale retrieval, multi-hop reasoning, and information synthesis) and incorporates subtask-level verifiability. Experiments confirm the benchmark’s difficulty, showing that state-of-the-art LLM-powered agents achieve consistently poor performance on these complex, long-horizon tasks.

### Strengths
The primary contribution is the development of a benchmark that rigorously enforces two previously neglected dimensions: long-chain complexity (integrating breadth- and depth-oriented search) and subtask-level verifiability. This fine-grained decomposition is essential, providing an informative supervision signal and allowing for error localization, which outcome-only evaluation protocols fail to capture. The dataset, curated through a costly human-annotation process across diverse real-world domains, effectively serves its purpose by revealing significant performance gaps and underscoring current agent limitations in synthesis and complex retrieval.

### Weaknesses
This is a paper proposing a new web agent benchmark. However, a new benchmark must clearly state the problem it solves and rigorously demonstrate why this problem and the evaluation method are important, with the analysis of failure cases being able to guide the direction of field development.
The problem this paper addresses is relatively clear, and the proposal of the dataset and its construction method also have value. However, there is no particularly detailed justification for why evaluation should be conducted through subtasks (and other agent evaluation papers have proposed similar evaluation methods and metrics). Furthermore, the analysis of failure cases lacks sufficient depth and does not offer unique insights.
If this paper merely defines a new benchmark data generation process (with missing details on the data synthesis process) and conducts a certain evaluation of existing model capabilities, then it has a contribution but is not an ICLR-level paper.

### Questions
1. Details of Data Definition:
- Automated Filtering: Batches of generated instructions first undergo automated filtering.
- Multi-Round Model Validation: This is followed by a second, more rigorous validation stage involving multiple model evaluations.
- Final Retention: Only tasks that pass all validation steps are retained as the final instructions.
What are the specific details? What insights are gained?
2. What are your next steps or ideas for addressing the failure cases?

### Soundness
2

### Presentation
3

### Contribution
2
