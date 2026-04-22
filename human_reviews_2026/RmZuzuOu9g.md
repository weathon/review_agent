# WebDART: Dynamic Decomposition and Re-planning for Complex Web Tasks

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2, 2

## Abstract
Large-language-model (LLM) agents are becoming competent at straightforward web tasks, such as opening an item page or submitting a form, but still struggle with objectives that require long-horizon navigation, large-scale information extraction, and reasoning under constraints. We present WebDART, a general framework that enables a single LLM to handle such complex chores. WebDART (i) dynamically decomposes each objective into three focused subtasks—navigation, information extraction, and execution—so the model concentrates on one skill at a time, and (ii) continuously re-plans the decomposition as new webpages are revealed, taking advantage of newly discovered filters or shortcuts and avoiding redundant exploration. Evaluated on WebChoreArena, WebDART lifts end-to-end success rates by up to 13.7 percentage points over previous state-of-the-art agents, while matching their performance on the easier WebArena suite and completing tasks with up to 14.7 fewer navigation steps. Code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
A novel LLM agent framework (named, WebDART) is proposed to automate long-horizon complex web tasks, with hierarchical planning and adaptation. Sub-task decomposition allows the agents to dedicate their power to the allocated focus scope, which is designed with an in-depth analysis of the target problem and insights into human cognitive behaviors. At the same time, WebDART prompts the agent to dynamically re-plan based on their experiences by leveraging discovered shortcuts. The efficacy of WebDART is verified in two benchmarks: WebChoreArena and WebArena. The proposed method significantly outperforms the baselines across various LLM backbones in the first benchmark, and demonstrates remaining competency in the second testbed indicating a descent balance.

### Strengths
In this section, I demonstrate the strengths of this paper.

1. Solid motivation: The design behind the framework stems from grounded observations that complex tasks overload the common agent frameworks. The rationale behind the structural heuristics is also reasonably backed up by mentioning that the quality of sub-tasks differs from each other. The authors also point out the possible limitations, revealing the necessity of the second component (i.e., adaptation).

2. Empirical supports: While navigating all the possible pages to get the task-relevant information is demanding, it can often be exhaustive. Table 2 demonstrates that such limitations can be significantly overcome with their proposed method. These results strongly support the effectiveness of the design of WebDART.
3. Case study analysis: The case study analysis allows readers to understand how WebDART performs well in practice. The readability of the study is high, as it is organized as a concise table.

### Weaknesses
Here, I present the weaknesses/questions/suggestions of this work.

1. Missing discussions/references: In the related work section, as this work focuses on prompting-based agent frameworks, discussions on possible strengths/limitations compared to fine-tuning methods would make the paper more solid. Currently, the authors compare with AgentSymbiotic, as a representative of finetuning-based methods, but with a lack of depth. I provide several finetuning-based methods, which I hope will be discussed [1,2,3]. To clarify, comparisons with these agents in experiments don’t seem very demanding.
2. Cost: How much did it cost to run all the experiments, including the baselines? I believe that cost information allows comparing the compute resources used between the baselines, as well as easy estimation of requirements when experimenting with WebDART as a baseline for other research.
3. Analysis on multi-agent baselines: I think more comparisons with multi-agent frameworks can be included. Mainly, it’d be interesting to see where the main differentiation towards success arises in the pipeline, compared to other multi-agent frameworks. 
4. Marginal improvements on WebArena: While the authors stated that many tasks do not demand complex sub-task planning in WebArena, it is still questionable why the WebDART agents do not gain much in this benchmark, as this phenomenon signals a possibility of biased design in the WebChoreArena benchmark. To be fair, there are stronger baselines in WebArena [4]. I think that the authors should discuss more to clarify this, as (at least) discussing what improvements in WebDART can make it outperform the baselines. Also, the “bypassing” mechanism should be elaborated.


References:

[1] Qi et al., “WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning” (ICLR 2025).

[2] Lee et al., “Learning to contextualize web pages for enhanced decision making by LLM agents” (ICLR 2025).

[3] Qin et al., “UI-TARS: Pioneering Automated GUI Interaction with Native Agents” (preprint 2025).

[4] https://webarena.dev/.

### Questions
Questions and suggestions for the authors are listed in the above weaknesses section for brevity.

### Soundness
3

### Presentation
3

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
This paper introduces WEBDART, a framework that enables large language model (LLM) agents to handle complex web tasks that require long-horizon reasoning and structured exploration. It dynamically decomposes each task into three subtasks: (1) navigation, (2) information extraction, and (3) execution, allowing the model to focus on one ability at a time. During navigation, the agent adaptively re-plans its strategy when new filters or interface shortcuts appear, reducing redundant actions and improving efficiency. This modular and adaptive design enhances task completion and robustness in complex web environments while maintaining strong performance on simpler tasks. Overall, WEBDART demonstrates that dynamic decomposition and real-time re-planning can significantly improve the reasoning and adaptability of LLM-based web agents.

### Strengths
- **Well-Justified Motivation**: The paper effectively addresses the importance of long-horizon web tasks as a fundamental challenge in current web-agent research.

- **Clear Writing and Organization**: The paper is well-written and easy to follow, with a well-organized structure and clear presentation of the proposed approach.

- **Simple Yet Effective Design**:This paper employs an intuitive three-stage decomposition that mirrors how humans naturally approach complex web tasks, resulting in a method that is both easy to understand and practically effective.

### Weaknesses
- **Lack of empirical justification for the conservative decomposition scheme**: The paper adopts the conservative scheme (deferring constraint handling to later stages) as the default strategy. However, this design choice is not supported by any preliminary analysis, empirical comparison, or prior evidence—for example, there is no ablation or user study contrasting conservative versus tightly coupled decompositions. Given that the efficiency of each scheme “hinges on site features” (line 204), a fixed conservative default appears heuristic rather than data-driven, and its general validity across domains remains unclear. A short pilot experiment or reference to earlier literature on adaptive task partitioning would strengthen this methodological decision and clarify why the conservative bias is justified beyond intuition.

- **Heuristic Nature of Information Extraction**: The information extraction pipeline is heuristic, relying on LLM prompts to select relevant pages and extract fields without any quantitative validation or ablation. The paper explains that the model “returns an index set that marks the pages most likely to contain the required information,” yet provides no concrete mechanism or evidence to show how reliable this selection is. Furthermore, the dismissal of the LLM-generated parser baseline is entirely qualitative, lacking any comparative results or failure statistics. Overall, the decision to rely solely on prompt-based extraction appears intuitive rather than experimentally justified, leaving uncertainty about its robustness and reproducibility across diverse web structures.

- **Lack of In-depth Performance Analysis**: While the paper reports overall success rates on the WebChoreArena benchmark [1], it does not provide finer-grained analyses that could strengthen its empirical claims. In the original benchmark, performance is typically broken down by cross-site domains as well as by task types such as Calculate, Long-Term Memory, Massive Memory, and Other. However, WEBDART’s results are aggregated, making it unclear which categories drive the observed improvements. The absence of such detailed breakdowns limits the interpretability of the reported gains and prevents deeper insights into where the proposed method truly excels or struggles.

[1] Miyai, Atsuyuki, et al. "WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks." arXiv preprint arXiv:2506.01952 (2025).

### Questions
- In Section 4.2, the authors claim that the observed improvements “highlight the advantage of shifting constraint handling to the data analysis stage.” However, it is unclear how the empirical results in Table 1 specifically support this interpretation. Could the authors clarify what evidence connects the performance gains to this design choice? 

- Table 3 reports the Results on the WebArena benchmark and includes additional baselines such as HybridAgent [1] and WebPilot [2], which show competitive performance. How do these baselines perform on WebChoreArena, and were they excluded due to reproducibility constraints or unavailability of results? 

- As an ablation, how does performance change when the routing module is disabled, particularly on the WebArena benchmark? It would be helpful to know how much accuracy drops and what types of routing errors occur (e.g., skipping extraction when it is actually required). Additionally, could the authors provide a brief analysis of the common failure cases in WEBDART?

[1] Song, et al. "Beyond browsing: Api-based web agents." arXiv preprint arXiv:2410.16464 (2024).

[2] Zhang, et al. "Webpilot: A versatile and autonomous multi-agent system for web task execution with strategic exploration." AAAI 2025

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
4

### Summary
This paper introduces a training-free framework that improves LLM-based web agents by dividing complex objectives into navigation, information extraction, and execution subtasks while dynamically revising plans as new web elements appear. This modular and adaptive design allows the agent to focus on one skill at a time and adjust strategies in real time, leading to up to 13.7% higher accuracy and 14.7 fewer navigation steps on complex benchmarks, without sacrificing performance on simpler tasks.

### Strengths
S1) Clarity and Organization
- The paper is clearly written and well structured. It effectively identifies the limitations of existing web tasks and presents a complex yet well-justified task definition along with a corresponding solution. The modular framework design and clear categorization of sub-tasks make the methodology easy to interpret. Figures and tables are informative and greatly aid understanding of the workflow.

S2) Technical Soundness and Contribution
- The overall structure and writing are coherent, and the paper presents a convincing motivation for introducing adaptive re-planning, emphasizing its necessity within dynamic web navigation. The proposed WebDART methodology is applied successfully, demonstrating meaningful improvements over existing baselines.

S3) Competitive Performance
- The approach achieves strong quantitative results, showing robustness across different web-based benchmarks and supporting the validity of the proposed framework.

### Weaknesses
W1) Lack of Novelty
- The proposed approach primarily integrates existing techniques rather than introducing a fundamentally new concept. While the composition of prior methods is well executed, the paper lacks clear methodological or theoretical innovation that distinguishes it from prior work.
- Moreover, the claimed design motivation—being inspired by human web search behavior—lacks supporting evidence from prior studies or pilot experiments. Including such references or empirical validation would strengthen this claim.

W2) High Computational and Monetary Cost
- The framework’s multi-stage navigation process, including dynamic re-planning decisions, likely leads to frequent LLM calls and thus high computational and monetary costs. To substantiate the framework’s practical usability, the paper should include quantitative evidence such as the number of LLM invocations per episode or the overall inference cost, along with a discussion of trade-offs between performance and efficiency.
- An efficiency analysis comparing total inference cost, time, or action steps with other methods would be valuable, especially since different baselines may not define navigation steps equivalently.

W3) Limited Generalization and Evaluation Scope
- The framework heuristically decomposes web tasks into three sub-tasks, but it remains unclear whether this decomposition generalizes across diverse web task categories.
- Experiments are conducted only on two benchmarks—WebArena and WebChoreArena—which, despite differing in complexity, share similar task structures. Consequently, the evaluation does not sufficiently demonstrate robustness to broader and more heterogeneous web task types.
- The authors should evaluate the framework on at least one additional web-based agent task (e.g., GAIA [2] or SimpleQA [3]), as the current setting appears tailored to the WebArena family.
- In addition, the prompt design for navigation planning (“navigating through menus and links,” “interacting with buttons and controls”) imposes a strong prior, raising further concerns about generalization capability.

W4) Outdated Baseline Comparison
- In Tables 1 and 2, most baseline performances are cited from prior studies, resulting in comparisons primarily against older methods.
- To more convincingly demonstrate the proposed approach’s effectiveness, the paper should include comparisons with more recent web-agent methodologies (for example, WebWalker [1]).

W5) Lack of Analytical Depth
- The current analysis is shallow. Reporting only accuracy and average steps provides limited insight.
- Additional ablation studies—such as the number of re-planning events, subtasks per instruction, their distribution, and the effect of fast-path routing—would offer deeper analytical value and better explain the framework’s behavior under different task conditions.
References

[1] Wu et al. WebWalker: Benchmarking LLMs in Web Traversal. Arxiv preprint. 2025
[2] Mialon et al. GAIA: A Benchmark for General AI Assistants. ICLR 2023
[3] Jason et al. Measuring short-form factuality in large language models. Arxiv preprint. 2024

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

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
This paper introduces WebDART, a training-free framework for LLM-based web agents that improves performance on long-horizon, multi-step web tasks. The method dynamically decomposes complex objectives into three subtasks: navigation, information extraction, and execution. It also allows continuously re-planning for these subtasks as new webpage elements appear.
WebDART aims to reduce the overload by letting a single frozen LLM focus on one sub-capability at a time and to improve sample efficiency by adapting plans on the fly. Results on WebArena and WebChoreArena show empirical effectiveness.

### Strengths
1. Paper is well-written and easy to understand.
2. Strong quantitative results: The WebDART framework seems to be quite effective and achieves consistent improvements across three model backbones and different web domains.

### Weaknesses
Major:

1. Limited novelty: The three core modules used in WebDART, i.e., navigation, extraction, and execution, have been widely used as standard prompting paradigms and is commonly seen in recent web agent works. The design is also quite heuristic and is only supported by intuition rather than systematic error analysis of previous work.  
2. Lack of learning or adaptation: All components in WebDART are purely prompt-engineered and rule-based. There is no learning or self-improvement involved in the method to adapt the policy itself  and learn a truly intelligent agent. This limits the framework’s scalability and robustness when deployed beyond the benchmark environments.
3. Baseline selection: The baseline comparison focuses on earlier methods, e.g., Table 3 misses many recent baselines on the WebArena  leaderboard with much stronger results, so the credibility of many claims on empirical effectiveness needs to be questioned.
4. Missing related work on multi-agent web navigation systems.

Minor:
1. Missing citation on line 111.

### Questions
NA

### Soundness
2

### Presentation
2

### Contribution
1
