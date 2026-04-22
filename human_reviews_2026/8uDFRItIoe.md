# SecureAgentBench: Benchmarking Secure Code Generation under Realistic Vulnerability Scenarios

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 4

## Abstract
Large language model (LLM) powered code agents are rapidly transforming software engineering by automating tasks such as testing, debugging, and repairing, yet the security risks of their generated code have become a critical concern. Existing benchmarks have offered valuable insights but remain insufficient: they often overlook the genuine context in which vulnerabilities were introduced or adopt narrow evaluation protocols that fail to capture either functional correctness or newly introduced vulnerabilities. 
We therefore introduce SecureAgentBench, a benchmark of 105 coding tasks designed to rigorously evaluate code agents’ capabilities in secure code generation. Each task includes (i) realistic task settings that require multi-file edits in large repositories, (ii) aligned contexts based on real-world open-source vulnerabilities with precisely identified introduction points, and (iii) comprehensive evaluation that combines functionality testing, vulnerability checking through proof-of-concept exploits, and detection of newly introduced vulnerabilities using static analysis. 
We evaluate three representative agents (SWE-agent, OpenHands, and Aider) with three state-of-the-art LLMs (Claude 3.7 Sonnet, GPT-4.1, and DeepSeek-V3.1). Results show that (i) current agents struggle to produce secure code, as even the best-performing one, SWE-agent supported by DeepSeek-V3.1, achieves merely 15.2% correct-and-secure solutions, (ii) some agents produce functionally correct code but still introduce vulnerabilities, including new ones not previously recorded, and (iii) adding explicit security instructions for agents does not significantly improve secure coding, underscoring the need for further research. 
These findings establish SecureAgentBench as a rigorous benchmark for secure code generation and a step toward more reliable software development with LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a repository-level coding agent benchmark aimed at measuring both functionality and memory safety of the agent-generated code. The benchmark is constructed from repositories where OSS-Fuzz has discovered vulnerabilities before, but have been fixed since. After several rounds of heuristic and partially manual post-processing and filtering, 105 coding tasks are selected for the benchmark. Each task consists of a description of the intended modification on the codebase, a functional test suite sourced from the original repository, and PoC programs for reproducing the vulnerability found by OSS-Fuzz. Then agents are then instructed to execute the coding instruction and are measured on the implementation's functionality and security. The evaluation across three agent scaffolds and three LLMs concludes that even in the best examined setup, barely over $15$% of the generated code is secure and correct.

### Strengths
- Measuring the security of coding agent-generated code on repository-level tasks is important, and has been overlooked by most previous code security benchmarks that focus mostly on function-level or standalone, small, application code.
- Focusing on the code repository state before the vulnerability has been *introduced* in the repository and formulating the same task as the one that introduced the vulnerability is a neat approach.
- Combining dynamic and static analysis follows good practice.

### Weaknesses
- The scope is somewhat limited: i) only memory-safety bugs; ii) only C/C++; and iii) only 41 historic repositories. While this in itself is mostly fine, I would prefer if the paper was explicit about this. Currently, the framing sounds like more general code security evaluation. This should also be highlighted in comparison with prior benchmarks that often measure secure coding performance across different programming languages and types of vulnerabilities.
- To point iii) above; there is a certain potential concern of data contamination, which should be discussed/treated with care.
- From the statistics in Table 2, it is clear that the benchmark focuses on small amount of generated code in large, complex contexts. While this is akin to other agentic coding benchmarks (e.g., SWE-Bench) this should be explicitly mentioned in the comparisons with prior secure coding benchmarks (e.g., BaxBench requires the models to generate up to hundreds of lines of code across several files from scratch).
- I find the evaluation section, given that this is a benchmark paper, rather limited, mostly due to the lack of actual state-of-the-art systems being evaluated. Giving weight to my concerns are some counterintuitive findings, e.g., figure 5, showing that cheaper models are better. This suggests to me that the benchmark might be noisy for these lower-performant models,  and results on stronger agents+models would be more insightful (e.g., Codex + GPT5, Claude Code + Claude Sonnet 4.5). Additionally, I miss two critical further examinations of the SAST results: i) the ratio of correct but suspicious programs is rather high, but is not examined further; ii) it seems to me that the SAST result post-patch are not baselined by SAST results pre-patch and SAST results post-goldpatch. Both of these experiments would give us a clearer view of the actual ratio of new vulnerabilities introduced by the agents. Finally, I am missing a more detailed discussion on the rather surprising stark differences in the overall performance and failure patterns of the agent scaffoldings. It would be nice to link the observed performances to design choices in the agents.
- **Minor**: Presentation. I find some of the tables and figures hard/strenuous to read. Table5: uncomfortable to match the middle column to the left column---I suggest a clearer horizontal divide between the CWEs. Tables 2 and 3 are next to eachother but misaligned, creating a poor visual effect, same for Table 5 and Figure 6. Figure 4 is overloaded---I suggest focusing in the main experiment on less but higher-level metrics (e.g., C&S, CS, and a larger category of incorrect), and then have a subsequent analysis of the errors. It makes for a clearer presentation. Table 4 is dense and hard to follow the column alignments. Figure 5 violates the page formatting, I believe this is an honest error, but I advise the authors to fix it at the next possible chance. Finally, the paper has minor typos and grammatical errors that could be quickly fixed by running a spell-checker or other writing aids.

### Questions
Additionally to my points in "Weaknesses" above, I have the following questions:
- Why is BaxBench excluded from the comparison in Table 1? How does it compare?
- It seems that SecRepoBench already contains most of the main assets needed to construct SecureAgentBench. Why was a new data collection pipeline necessary? Could you not have just extended SecRepoBench (and potentially other prior works)?
- While the at many stages of the benchmark construction pipeline manual intervention is needed, the task descriptions are LLM-generated. Why so? It seems that this would have been a smaller manual effort then for instance verifying and setting up the functional testing suites.
- The data collection pipeline is rather lossy. Starting from almost 5k potential cases, at the end only 105 remain. What do you think, where could you improve your pipeline to decrease this loss? Could you automate more steps to have potentially a constantly evolving benchmark, reducing, e.g., contamination risks?
- For the results in Figure 7, this seems rather counterintuitive. What happens if you increase the budget constraints? What happens if you try better models/scaffolds? Are the 16 C&S cases the same or different with and without the prompt?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SecureAgentBench, a benchmark that evaluates whether an Agent can implement functionalities with secure code. They prepare 105 coding tasks from the data source ARVO and OSS-Fuzz to make sure each task includes realistic multi-file edit requirements and contains potential vulnerabilities from human implementation. For the evaluation metric, they consider both to evaluate the functionality and the security (poc and static analysis). They test open-source agents (SWE-agent, OpenHands, and Aider) with SOTA models on this benchmark, including GPT, Claude, and Deepseek.

### Strengths
1. Promising direction. Establishing the benchmark to highlight the security risks associated with Code GenAI is a direction worth studying.
2. Consider real-world security problems in OSS repo and environment deployments for agentic tasks.
3. Compared with existing baselines from multiple perspectives, the results show the effectiveness of the proposed method.
4. Consider multiple metrics for evaluating code patches generated from the agent. For example, using SAST to illustrate the consideration of the robustness of security.

### Weaknesses
1. Currently, the benchmark only contains C/C++ language data, which might be considered insufficient for comprehensive evaluation. And the data collection method should be considered, whether it can be automatic or semi-automatic for continuous collection. 
2. There are some concerns about the data collection and evaluation that need to be clarified. Generally, there would be more than 1 commit from PVIC/VIC to VFC. For the task description, what would be involved: only the VIC commit context, or from PVIC to VFC? What version of unit test are you using for the functionality test? Does it consist with the task description? For example, if you are asking an agent to implement the functionality of VIC but using the unit test from VFC, some cases might never be considered correct. How can we guarantee the quality of the task description if it is generated by an LLM?

### Questions
1. As we already use SAST for evaluation, it would be great to show the important rules you are running in the tool. Otherwise reader still does not know what this test covers.

2. Have you considered data contamination for the data collection? How can we guarantee the LLMs are not that familiar with the repo and code?

3. I am curious about why Aider has such amount of "empty patch" in the results. It would be good to have a case study on why these happen to exclude the possibilities of misoperation.

4. Unit test: what we mentioned above.

5. It might be good to also provide the time and cost of the experiments for a better understanding of the difficulty of the task. 

6. It might be good to cite some rule-based patching agents, even though they are not evaluated in the experiments.

[1] Agentless: Demystifying llm-based software engineering agents

[2] PatchPilot: A Cost-Efficient Software Engineering Agent with Early Attempts on Formal Verification

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
4

### Summary
This paper introduces a new benchmark, SecureAgentBench, a benchmark of 105 coding tasks designed to evaluate code agents’ capabilities in secure code generation. The coding tasks are sourced from real-world software repositories, undergoing rigorous data cleaning and filtering process. The code patches generated by the code agents are evaluated from various aspects including the correctness of the functionality, whether it contains already known vulnerabilities, and whether it introduces new vulnerabilities. The evaluation results demonstrate that existing code agents struggle to produce correct and secure code.

### Strengths
- The benchmark requires code agents to operate directly on entire software repositories rather than on simplified, function-level completion tasks. This design better reflects how agents perform in realistic software engineering workflows.
- The data collection, cleaning, and filtering pipeline is rigorous, ensuring the benchmark’s reliability and high-quality task instances.
- The paper presents a comprehensive evaluation across multiple agent frameworks, backbone models, and experimental settings (including the effect of explicit security reminders), offering a thorough analysis of current capabilities and limitations.

### Weaknesses
- Many real-world, repository-level feature implementations require multiple commits, whereas this benchmark considers only a single commit per task. The gold patches average around 40 lines of code across two files, which limits the overall task complexity.
- The process for collecting and validating test cases is not clearly described. If the tests are sourced from PVIC, they may not adequately verify the newly implemented feature. Conversely, if they are derived from later commits, it remains unclear how these test cases are identified and extracted.
- The benchmark size is relatively small, with only 105 tasks, which may restrict the statistical robustness of the evaluation.
- The benchmark exhibits potential bias and limited coverage for code generation tasks, sourcing from commits introducing C/C++ memory-safety vulnerabilities historically found by fuzzing.
- The analysis of SAST results lacks sufficient discussion of false positives, which could affect the reliability of the reported “suspicious” vulnerability cases.

### Questions
Check weaknesses.

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
3

### Summary
The paper introduces SECUREAGENTBENCH, a benchmark designed to evaluate how well LLM-powered code agents can generate secure code. It consists of 105 tasks derived from real-world, open-source vulnerabilities, setting the tasks within the repository context at the exact point the vulnerability was originally introduced. The benchmark evaluates agent-generated patches for both functional correctness (using differential testing) and security (using proof-of-concept exploits and static analysis). Experiments on agents like SWE-agent and OpenHands show they struggle significantly, with the best-performing combination achieving only a 15.2% "correct-and-secure" rate. The results also reveal that agents often introduce new, previously unrecorded vulnerabilities, and that simple security reminders in prompts do not improve performance.

### Strengths
- The benchmark addresses the critical and timely issue of secure code generation by agents. SECUREAGENTBENCH requires agents to perform realistic, repository-level edits that often involve multiple files, better simulating real-world software engineering tasks.
- The construction process employs a two-stage pipeline (static SZZ followed by dynamic PoC validation) to identify Vulnerability-Inducing Commits (VIC). This allows tasks to be grounded in the exact historical context where vulnerabilities were originally introduced.
- The work employs a multi-faceted evaluation protocol: 1) Functional Correctness via differential testing against developer reference implementations. 2) Historical Security Verification using specific proof-of-concept (PoC) exploits. 3) New Vulnerability Detection using static analysis (SAST) to flag potential new risks introduced by the agents.

### Weaknesses
- Limitations of Security Oracles: The security evaluation relies on PoCs that only detect the single historical vulnerability targeted by each task, combined with a single SAST tool (Semgrep) known for potential false positives and limited by its specific rule set. Consequently, code classified as "secure" by this benchmark may still contain undetected vulnerabilities, preventing the guarantee of security.

- High Generation Failure Rates affecting Insights: The extremely high rate of "No Output" results, such as Aider+GPT failing to produce any output 77.1% of the time, raises questions about the definitive nature of the findings. While the authors attribute this to inherent task difficulty, it might also stem from suboptimal prompt instructions or simply that the evaluated models are currently incapable of handling these benign coding tasks, obscuring their actual security capabilities.
- Unclear Language Coverage and Extensibility: The specific coverage of programming languages is not explicitly defined in the main text, though the prompt templates in Appendix H explicitly specify a "C/C++ code repository". If the dataset is exclusively C/C++, this should be stated more clearly as a scope limitation. Furthermore, it would be beneficial to discuss whether the proposed construction method can be easily extended to other languages beyond C/C++.

### Questions
- Ambiguity in Vulnerability Verification: While Section 2.2 explicitly mentions manual quality assurance for test cases and requirements, it is unclear if the specific Vulnerability/CWE classifications underwent human verification to ensure they accurately reflect the root cause of the issues derived from OSS-Fuzz.

### Soundness
2

### Presentation
2

### Contribution
2
