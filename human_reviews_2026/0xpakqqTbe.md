# RExBench: Can coding agents autonomously implement AI research extensions?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 6

## Abstract
Agents based on Large Language Models (LLMs) have shown promise for performing sophisticated software engineering tasks autonomously. In addition, there has been progress towards developing agents that can perform parts of the research pipeline in machine learning and the natural sciences. We argue that research extension and its implementation is a critical capability for such systems, and introduce RExBench to support the evaluation of this capability. RExBench is a benchmark consisting of realistic extensions of 12 research papers that aim to investigate novel research hypotheses. Each task is set up as an extension to an existing research paper and codebase, accompanied by domain expert-written instructions. RExBench is robust to data contamination, and supports an automatic evaluation infrastructure that executes agent outputs to determine whether the success criteria are met. We use this benchmark to evaluate 13 LLM agents implemented using three different frameworks: aider, Claude Code, and OpenHands. We find that all agents fail to autonomously implement the majority of the extensions, with the best agent at around 31% success rate. Although the success rate improves with additional human-written hints, the best performance under this setting remains below 48%. This indicates that current agents are still short of being able to handle realistic research extension tasks without substantial human guidance. Based on analyses of prominent failure modes, we put forward actionable short- and long-horizon recommendations for future research coding agent development.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to benchmark the capabilities of LLM agents for modification-based coding implementation with specific focus on NLP and ML. Specifically, this paper proposes REXBench, which takes the original paper, the corresponding codebase, an instruction as the inputs, and then asks the agents to modify the codebase to implement the instruction. Automated evaluation metrics are proposed, i.e., by checking whether the results fall within a reasonable range.

### Strengths
- The writing is of good quality.
- Several LLMs and LLM agents are discussed in empirical results.

### Weaknesses
- From my perspective, this would not be an impactful benchmark for the community, due to several shortcomings: 1) limited number of tasks; 2) computational requirements, considering 8/12 tasks require A100; 3) the evaluation metric is unreliable: the agent would be rewarded as long as the execution results fall within a preset range of values, which is a quite loose evaluation metric.
- As this paper only proposes 12 tasks. It would be better the authors can provide all the task descriptions of REXBench.

### Questions
From my perspective, the proposed benchmark is of limited significance, and cannot benefit the future research of the community.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces RExBench, a benchmark designed to evaluate the capability of LLM-based coding agents to autonomously implement research extensions in AI/NLP/ML. The benchmark consists of 12 tasks based on recently published papers, where agents must implement specified extensions starting from an existing codebase. The authors evaluate 13 agent configurations across three frameworks (aider, Claude Code, OpenHands) with various LLM backbones. Results show that even the best agents (OpenHands + Claude 4 Sonnet/GPT-5) achieve only ~31% success rate without hints, reaching ~47% with detailed hints. The paper provides extensive error analysis and recommendations for future agent development.

### Strengths
1. Research extension is a critical capability for autonomous research agents, distinct from replication
2. Novel implementations stored privately is a major strength over benchmarks like PaperBench
3. 13 agent configurations, 3 hint levels, detailed error taxonomy
4. VM-based evaluation with controlled execution environments ensures reproducibility
5. Distinction between explicit/implicit errors, over-editing observations, overthinking issues
6. Both short-term (scratchpads, repair mechanisms) and long-term (verification, context handling)

### Weaknesses
1. The work lacks a human baseline. It would be insightful to see how PhD students or domain experts perform on the same tasks. Similarly, there are no simpler non-agentic baselines to show what traditional systems could achieve. There’s also no direct comparison between agent-generated and human-written code, especially in terms of readability, maintainability, or long-term usability — all of which matter for research automation.

2. The experimental design has several weaknesses. There’s no statistical significance testing, even though the decoding process involves randomness. Inter-annotator agreement on the gold solutions isn’t reported, so we don’t know how consistent the labeling process was. The hint design also feels somewhat ad hoc — it’s unclear how the three hint levels were calibrated. Finally, the temperature setting differs between models, with some using 0.7 and others using defaults, which introduces inconsistency in evaluation.

3. Methodologically, the benchmark framework follows a fairly standard design. The main novelty lies in applying it to “research extension” tasks, rather than in introducing new benchmarking techniques. While the infrastructure is robust and well-documented, it doesn’t really bring conceptual innovation. The work is strong in execution but not particularly creative in method.

4. The authors themselves note that their instructions are much clearer and more informative than what a real researcher would encounter. This makes the setup somewhat artificial. Real research usually involves a lot of ambiguity, trial and error, and iterative exploration — all of which are missing here. The tasks also seem too narrowly defined and well-scoped, failing to capture the open-ended, exploratory nature of genuine research projects.

5. Some findings are left unexplained. For example, why do hints sometimes *reduce* performance, as seen in Othello and Tree-of-Thoughts tasks? There’s little analysis connecting specific task features to failure modes, and minimal discussion on when these agents should or shouldn’t be applied. The regression analysis in Figure 5 also seems underpowered, given that it’s based on only twelve data points. These gaps limit the depth of insight.

### Questions
1. How do PhD students perform on these tasks? What's the time/success rate comparison?
2. Did author measure inter-annotator agreement on gold implementations? Could multiple valid solutions exist?
3. How were the two hint levels designed? Was there any user study or pilot testing?
4. What percentage of failed attempts were close to success (within 1-2 bugs)? Could authors characterize the "distance to success"?
5.  Were there cases where agents found valid alternative implementations that didn't match authorsr gold numerical output but were scientifically sound?
6. Can authors provide objective difficulty metrics beyond lines of code (e.g., cyclomatic complexity, semantic changes required)?
7. The observation that reasoning models "overthink" is interesting—did authors try adjusting their reasoning effort parameters?
8. The authors mention agents make "unrequested modifications"—could instruction tuning on "minimal edits" help?
9. What's the cost-effectiveness compared to hiring a research assistant?

### Soundness
3

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
The paper proposes a benchmark REXBench which is designed to evaluate whether LLM agents can implement research extensions—i.e., modifications or follow-up experiments extending existing ML/NLP research papers. The benchmark includes 12 real research papers with corresponding codebases and expert-written instructions describing realistic extensions (e.g., changing models, datasets, or algorithms).  The experimental results show that current agents are still short of being able to handle realistic research extension tasks.

### Strengths
The paper proposes a novel problem formulation that tests agents’ ability to extend scientific research, an important while underexplored problem. The benchmark is well designed, including containerized evaluation, de-contamination, etc. The experiments are thorough, using various evaluation metrics and also including cost/time study and an error analysis.

### Weaknesses
Although the problem is novel, it is somewhat too niche. In addition, the scale and scope of the benchmark are very limited — it contains only 12 papers and covers only the NLP/ML domain. Therefore, it does not sufficiently evaluate the model’s capability in research extension.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper propsoes a new benchmark RExBench which can evaluate the code agents for incremental research ideas.The benchmark includes 12 research papers and their corresponding extensions. Compared to previous tasks, the aim is to automatically assess how well an agent can autonomously implement realistic research extensions.

### Strengths
1. The aim of benchmarking is to automatically assess how well an agent can autonomously implement realistic research extensions. This goal is kind of realistic and interesting. The extension proposal is annotated with the gold edits for the target extension, which is a good resource.
2. The paper evaluates 13 LLM agents based on this benchmark. The additional hints setting serves as an ablation study for the bottleneck of the pipeline.  The paper also includes both quantitative and qualitative analysis.
3. The paper includes code. The paper has good visualizations. The paper includes most of the infrastructure pipelines in the main context.

### Weaknesses
1. Some details of the benchmark are missing. What is the average extension for each of those 12 papers? The title is kind of misleading. Instead of research extension, the paper is more on the adaptation of the existing code base. 
2. Although errors are discussed in section 5.1, most of them are pretty high-level. The paper also fails to explain the reason behind those errors. The reason behind those errors can help researchers understand the drawbacks of current methods. Sec 5.2 is kind of high-level. If they can be linked to the specific errors, it would be better.
3. The conclusion seems a bit too long. The paper might need to include a small subset for human evaluation to quantify some of the observations. If section 5 can be supported by more numbers or evidence, it will become stronger.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3
