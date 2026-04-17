# UniVA: Universal Video Agents towards Next-Generation Video Intelligence

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Recent breakthroughs in visual AI have largely treated video tasks in isolation, with specialized models excelling at generation, editing, segmentation, or understanding individually. We introduce \textbf{UniVA}, a multi-agent framework for universal video intelligence that unifies video understanding, segmentation, editing, and generation in complex workflows. UniVA employs a Plan-and-Act dual-agent architecture: a planner agent decomposes high-level user requests into a sequence of video-processing steps, and executor agents carry out these steps using specialized modular tool servers (for video analysis, generation, editing, object tracking, \textit{etc.}). Through a multi-level memory design (global knowledge, task context, and user-specific memory), UniVA supports long-horizon reasoning and inter-agent communication while maintaining full traceability of each action. 
This design enables iterative and composite video workflows (\textit{e.g.}, image $\rightarrow$ video generation $\rightarrow$ video editing $\rightarrow$ object segmentation $\rightarrow$ content composition) that were previously cumbersome to achieve with single-purpose models or monolithic video-language models. We also introduce UniVA-Bench, a benchmark suite of multi-step video tasks spanning understanding, editing, segmentation, and generation, to rigorously evaluate such agentic video systems. Both UniVA and UniVA-Bench are open-sourced to the community, with the aim of catalyzing next-generation video intelligence research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a system called UniVA to help with complex video tasks. UniVA is designed to combine many different video tasks, such as understanding, cutting, editing, and creating videos, into one workflow. UniVA uses two types of AI agents to work. A planner agent receives a high-level request from a user and breaks it down into smaller, manageable steps. Then, executor agents take over to complete each of these individual tasks using various specialized tools. UniVA is built with a three-level memory design to handle long and complex tasks. The authors also created UniVA-Bench, a new tool for evaluating how well AI systems can handle complex, multi-step video assignments.

### Strengths
1. The paper is easy to understand. The motivation is clear.
2. The proposed system is complete. This needs substantial engineering efforts.

### Weaknesses
I am not an expert in designing video agent benchmarks, and I do not check the appendix. My criticism will focus on the technical parts.
1. This paper contains many over-claimed points, for example, next-generation video intelligence. I do not fully agree with the definition of next-generation video intelligence proposed in this paper.
2. Roughly speaking, the proposed agent works in a ReAct pattern. No significant planning contribution is presented. As for the memory part, the designed working memory is trivial and brings limited insights to the community.
3. As for the competitors, I am only familiar with the understanding part. Important baselines, such as VideoAgent and VideoAgent2, are missing. Only simple MLLMs are used for comparison, which is very unfair.
4. So, my point is that I cannot find truly new things brought to the video agent community with substantial experiments to prove them.

To some extent, HuggingGPT with a ReAct pattern could achieve the same effect as the proposed framework with some engineering efforts. I do appreciate the engineering efforts the authors have made, but the new insights are not enough for ICLR.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces UniVA, a multi-agent framework for unified video intelligence that integrates understanding, segmentation, editing, and generation. UniVA employs a Plan–Act dual-agent architecture, where a planner decomposes user goals into subtasks, and an executor carry out these steps using specialized modular tool servers. Besides, UniVA supports long-horizon reasoning and inter-agent communication while maintaining traceability of each action through a three-level memory system (global, task, user). To evaluate the framework, the authors introduce UniVA-Bench, a benchmark designed for multi-step video tasks spanning understanding, editing, segmentation, and generation. Experiments show that UniVA demonstrates competitive performance across a wide array of video tasks.

### Strengths
1.	The paper is well organized with clear diagrams and easy to follow.

2.	UniVA-Bench provides a systematic evaluation suite with new “agentic metrics” (wPED, DepCov, ReplanQ), filling a gap for multi-agent video systems.

3.	The experiments and visualizations are reasonable and well done.

### Weaknesses
1.	While the integration is strong, most modules (e.g., MCP protocol, planning agents, tool servers) are adaptations of existing frameworks rather than new technical inventions.

2.	While the ablation in Figures 6 and 7 shows improvements when incorporating user and task memory, the gains are relatively small. 

3.	More ablations on user and task memory and comparisons with simpler orchestration baselines could be expanded to strengthen claims.

4.	The paper does not analyze latency, scalability, or hardware requirements, which are important for practical deployment.

### Questions
1.	How does the Planner dynamically adjust its plan when tool outputs deviate from expectations? Is there an explicit feedback or self-correction mechanism?

2.	How scalable is UniVA to hundreds of concurrent video tools or longer-than-minute sequences？Does the MCP protocol become a bottleneck?

3.	Could the authors provide more details on the computational efficiency of the end-to-end UniVA pipeline in practice?

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
In this paper, the authors focus on the goal of creating a universal agent with broad-based video understanding and generation capabilities, sufficient for video creation workflow. To this end, the authors propose UniVA, a unified video agent system that incorporates separate planning and executing agents that interact with tool servers. The system also incorporates various forms of memory to ensure proper context. The authors evaluate their approach on various video understanding tasks, and they also present a novel benchmark named UniVA-Bench for future research.

### Strengths
1. The paper's goals are ambitious: A universal system for general video understanding and synthesis. They demonstrate a plausible system with promising results on some tasks.

2. The paper is well written and easy to understand. 

3. The benchmark suite (UniVA-Bench) can be used for further research in the community.

### Weaknesses
1. The approach is very complex, and while some results on tasks are promising, there is still room for better performance given the high complexity of the approach.

2. It is unclear if there are any novel methods or models presented in the paper. This reads almost like a systems architecture paper that combines a number of prior models into one system. 

3. More qualitative examples would be helpful in understanding the performance of the approach.

### Questions
1. Could the authors please elaborate on the contributions of the paper? Is this more of a systems architecture paper?

2. Would it be possible to include more qualitative examples in the paper? 

3. Are there any ways the systems architecture could be simplified without sacrificing performance?

### Soundness
2

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
4

### Summary
This paper presents UniVA, a unified multi-agent framework designed to handle complex video tasks—including generation, understanding, editing, and segmentation—through a Plan-Act architecture enhanced with multi-level memory and modular tools integrated via the Model Context Protocol (MCP). The authors also introduce UniVA-Bench, a dedicated benchmark for evaluating multi-step video workflows.

### Strengths
- Built on the Model Context Protocol (MCP), UniVA supports a wide range of video, non-video, and non-AI tools in a modular and plug-and-play manner, enabling flexible and extensible task execution.  
- The framework demonstrates comprehensive capabilities across diverse video-related tasks, integrating multiple functionalities into a unified pipeline.

### Weaknesses
- The paper's core technical components—the Plan-Act architecture and multi-level memory mechanism—are based on well-established paradigms in the agent literature, while the Planner itself is implemented using an existing LLM framework, thus providing limited novel technical insight specifically tailored to video intelligence.  
- Evaluations in the benchmark primarily compare against non-agentic models, failing to adequately highlight UniVA’s advantages over recent video-specific agent systems.  
- The small scale of UniVA-Bench—using only 10 videos per task—may undermine the generalizability and robustness of the experimental conclusions.  
- There is a lack of in-depth analysis on failure modes, such as planning errors, tool invocation conflicts, or memory retrieval issues.

### Questions
- How does UniVA perform under real-time or low-resource conditions, and what are the computational requirements for each module?  
- How does the Planner handle ambiguous or underspecified user instructions, and what mechanisms are in place for recovery?  
- Can UniVA support interactive video tasks, and if so, how is dynamic user input incorporated during execution?

### Soundness
2

### Presentation
2

### Contribution
2
