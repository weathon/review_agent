# Benchmarking LLM Tool-Use in the Wild

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Fulfilling user needs through Large Language Model multi-turn, multi-step tool-use is rarely a straightforward process. Real user interactions are inherently $\textbf{wild}$, being intricate, messy, and flexible. We identify three key challenges from user behaviour: $\textit{compositional tasks}$ that demand efficient orchestration of tool-call topologies, $\textit{implicit intent}$ spread across dialogue turns that require contextual inference, and $\textit{instruction transition}$, which mixes task queries, clarifications, and casual conversation, forcing LLMs to adjust their policies on the fly. Existing benchmarks overlook these behaviors, making the apparent progress of LLMs on tool-use spurious. To address this, we introduce $\textbf{\textit{WildToolBench}}$, an LLM tool-use benchmark grounded in real-world user behavior patterns. Comprehensive evaluations of 57 LLMs reveal that no model achieves an accuracy of more than 15\%, indicating a substantial gap in the robustness of LLMs' agentic ability. Controlled experiments and in-depth analyses further indicate that the real challenge for LLM tool-use lies not in artificially complex tasks, but in the wild nature of user behavior, emphasizing the need to reconsider the interactions among $\textit{LLMs}$, $\textit{users}$, and $\textit{tools}$.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes WildToolBench -- a tool-calling evaluation dataset that simulates real-user behavior when interacting with tool-calling models. This work identifies three user characteristics that LLMs find challenging, namely single tool-use, inferring information and intent from multi-turn dialog, and switching intents on the fly. This work creates 256 scenarios, with each scenario containing 4 tasks for a total of 1024 tasks. Evaluation of 57 models finds that the best performance on a session is 15%, indicating that the benchmark is hard to solve for current models.

### Strengths
- The underlying motivation to build a dataset mimicking realistic user conversations is a meaningful contribution. Results showing that models struggle greatly to fulfill user requirements in a session demonstrate the utility of this dataset as a good benchmark.
- Extensive experimentation covering 57 models, ranging from proprietary models to open-source general and function-calling specific models.

### Weaknesses
- I found the process of data curation hard to understand from the main text. The paper would benefit from reordering of the content by moving sections describing data curation from the Appendix into the main text
- Similarly, adding a section comparing the WildToolBench and other function-calling benchmarks to show how the other benchmarks fail to capture the cases that WildToolBench does will help set the motivation.
- Section 3.2 mentions that the first step to data curation is analyzing large-scale real user logs, but no information is provided about the source or the characteristics of these user logs. Similarly, the 1600 APIs collected are also not described.
- Section 4.2 title should most likely read "LLMs Perform Poorly on Tool Orchestration"?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces WildToolBench, a benchmark designed to evaluate Large Language Models' (LLMs) ability to effectively use tools in real-world, multi-turn, multi-step dialogue settings. Unlike existing benchmarks that focus on artificially complex tasks, WildToolBench addresses the complexity of user behavior in real-world interactions. The authors identify three key challenges that current LLM tool-use capabilities fail to handle effectively: compositional tasks that demand complex tool orchestration, implicit intention spread across dialogue turns, and frequent transitions between task queries, clarifications, and casual conversation. The paper benchmarks 57 LLMs, revealing that none of the models achieve higher than 15% accuracy in session completions, underscoring a significant gap in LLMs' agentic capabilities. The experiments and analysis demonstrate that LLMs struggle not with artificially difficult tasks, but with the "wild" nature of real user interactions.

### Strengths
By focusing on multi-turn, multi-step tool-use that mimics actual user interactions, the benchmark is highly relevant to practical applications of LLMs in domains requiring tool integration. The insights provided from this evaluation have the potential to inform the development of more robust, user-aware AI systems. Meanwhile, this paper effectively highlights the specific areas where LLMs currently struggle, such as handling compositional tasks, inferring implicit user intentions, and adapting to frequent task transitions. These findings are essential for guiding future model development and addressing fundamental weaknesses in tool-use.

### Weaknesses
Although the paper introduces a useful benchmark, there is a lack of transparency regarding the usage api/document. The author didn't provide any scratch version of the benchmark, which limits the reproducibility and potential impact of the paper on the community.
Additionally, a significant portion of the benchmark creation was done manually, which limits its scalability and increases costs. The current manual curation process is resource-intensive, making it difficult to expand the dataset further without significant additional effort.

### Questions
referred to weakness paragraph.

### Soundness
3

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
The paper presents WildToolBench, a benchmark designed to evaluate LLMs’ real-world tool-use abilities. Built from 256 human-verified scenarios reflecting authentic user behaviors—compositional tasks, implicit intents, and instruction transitions—it exposes key weaknesses in current LLMs’ reasoning and planning. Benchmarking 57 models shows none exceed 15% session accuracy, revealing significant gaps in robustness and adaptability.

### Strengths
1. The paper is highly original in redefining LLM tool-use benchmarking through real-world user behavior modeling, capturing compositionality, implicit intent, and instruction transitions that prior datasets overlook.

2. The study demonstrates strong methodological rigor and clear exposition, combining large-scale human-verified data with systematic evaluation of 57 models to yield reproducible and insightful findings on LLM agent robustness.

### Weaknesses
1. Heavy reliance on human-verified curation (256 scenarios / 1,024 tasks) constrains scale and may bias domain coverage; moreover, this limitation implies that such a data-construction pipeline is difficult to extend for model training or optimization purposes.

2. The evaluation emphasizes function-call correctness and orchestration (OP/AP) but underplays live constraints—API latency, rate limits, outages, and cost; adding live (or high-fidelity simulated) tool execution with cost/latency/robustness metrics and failure-injection studies would better reflect real deployment trade-offs and stress agent reliability.

### Questions
1. In Appendix C, the paper mentions that behavior patterns were summarized by analyzing real user logs — does this imply a potential information bottleneck, where instead of fully leveraging authentic user dialogues and scenarios, the authors mainly relied on manual summarization combined with LLM synthesis?

2. The same appendix states that human experts conducted four iterative rounds to raise data quality from 62% to 100%. How exactly was this data quality measured, and what were the main issues identified during the intermediate stages?

### Soundness
3

### Presentation
4

### Contribution
2
