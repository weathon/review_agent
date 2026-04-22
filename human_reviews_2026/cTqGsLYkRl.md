# VideoAgent: All-in-One Agentic Framework for Video Understanding and Editing

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Video editing has become essential in digital media creation, yet existing automated systems are restricted to short segment processing and domain-specific tasks. They face two critical limitations: i) inability to handle diverse video comprehension and editing operations, and ii) lack of long-video understanding for coherent narrative creation. We propose VideoAgent, an all-in-one agentic framework addressing these challenges through two key innovations. First, we develop automated video shot creation with shot planning agents for coherent narratives and cross-modal retrieval for aligned visual content. Second, we design a multi-agent orchestration framework integrating over thirty specialized editing agents. Intent parsing filters relevant tools while self-reflective graph orchestration assembles complex editing pipelines. Extensive experiments on our newly-proposed VideoEdit benchmark and public datasets demonstrate VideoAgent's superiority over existing multimodal LLMs and agentic systems. VideoAgent achieves 87-98% orchestration success rates while reducing API costs by 60%. Human evaluation across six video categories shows VideoAgent produces professional-quality content approaching human-level performance, with ratings only 4% below human-created videos.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies the two main problems in automated video editing (lack of coherence in long-form narratives and inability to handle diverse tasks), and proposes VideoAgent, an all-in-one agentic framework with these main contributions:

* a method for video shot creation with cross-model and context on the full narrative of a target output video
* an orchestration framework that can coordinate a large set of agents to generate a final video edit out of the created video shots
* a new video edit benchmark (VideoEdit)

Complete evaluation follows using both VideoEdit and Shot2Story, which display high performance against the proposed baselines (87-98% success rate), reduced API costs (60% lower costs), and comparable results to those produced by human editors (just 4% below human edits).

The paper also provides detailed prompts and pseudocode for each of the agents used by VideoAgent, and includes source code for this task.

### Strengths
The main strengths of this work are:

* Sound engineering, combining a large amount of agents with a solid orchestration method
* Great quantitative results, generally superior to those of the baselines and, especially, very close to human-created videos (with the caveats discussed in the weaknesses section)
* Novel approach that combines multiple-agents with a long-narrative aware video shot process, and an orchestration framework with self-aware elements
* Clear descriptions of the methods used
* Exhaustive details on prompts and pseudocode used in the agents, and open source code, both of which allow for high reproducibility of the work

### Weaknesses
The main weaknesses of this work are:

* Lack of details about the human baselines. Just 4% below human performance is a very impressive result, however, which is greatly diminished by lack on details about how this human performance is measured. For example, who are the humans, what is their expertise, which tools have they used, for how long, etc.
  * this point is really critical because the non-human baselines are based on systems that aren't designed to handle multi-modal video editing. So it's difficult to understand what the quality of this system is without a valid human comparison.
  * (minor suggestion): Besides an ad-hoc human baseline of videos edited just for the purpose of this evaluation, one can also wonder how the system would compare against video edits seen on the wild, for different categories. What is the success rate against, for example, against fan edits of existing works.

* Lack of actual examples in video format. The paper displays many examples as frame sequences, but given the nature of this work, the addition of examples in video format would be beneficial to this work. Being able to actually watch and listen to the videos produced by the system (and to compare them to the raw input materials) would provide a better understanding of the system quality.

* While this work details the creation of a significant engineering system, with many agents and a solid orchestration method, the research contributions appear more incremental. To make the research contributions clearer, the paper could describe in more detail how novel aspects in the introduced orchestration system differ from those in other multi agent systems based on LLMs that also deal with graphs. The Related Work section at this time merely describes the application to multimodal video editing workflows which I don't think is enough novelty. 

* (minor) Lack of details regarding system latency (though API costs are provided), especially when compared against other baselines. Ideally the paper would include a plot with an axis for latency and another for each key metric, such that the tradeoffs between quality and performance can be better understood. (A similar plot for API costs would be interesting too, and given the reduced API costs of this system, good evidence in favor of this work)

* (minor): Lack of mentions of key downsides for this approach, and potential future work. What do the failure modes look like? What is insightful about them?

* (minor): The appendix may be excessive. The paper could improve by just listing a short summary of each agent behavior, and pointing to the supplementary materials for details.

### Questions
* Have you considered using video generation models too, as an agent, for adding shots not included in the input materials?

* Could a much simpler version of this work approach the same quality? For example, could the graph be non-dynamic but fixed, with each node gated on a selector for whether the node agent needs to be applied or not? (this has been explored to some extent in Table 3 which removes Intent Parsing and Agent Graph elements, but I wonder specifically about an existing but fixed graph)

* Have you considered ablating the specific list of agents, to measure how they rank compared to each other? this could inform which other agents are possible future additions for the system

### Soundness
2

### Presentation
3

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
This paper presents VideoAgent, an multi-agent framework for automated video understanding and editing, aiming to enable general-purpose video creation with coherent narratives and long-video reasoning. The framework consists of two major components including automated video shot creation and multi-agent orchestration. Besides, a new VideoEdit benchmark is introduced for evaluation. Experiments on video understanding, video retrieval, and workflow orchestration show that VideoAgent outperforms existing multimodal LLMs and agentic systems.

### Strengths
1.	The presentation is clear and easy to follow.

2.	The experiments and visualizations are reasonable and well done.

3.	Over 30 tool agents support a wide range of operations (audio, visual, translation, meme creation, etc.), suggesting high practical applicability.

### Weaknesses
1.	While the system integration is impressive, most modules (retrieval, trimming, intent parsing) adapt existing methods rather than proposing new algorithms.

2.	The reliance on proprietary or external APIs (e.g., GPT-4o, Claude-Sonnet, Gemini-2.5) may limit true reproducibility and comparability.

3.	The new VideoEdit benchmark seems self-curated and may not fully represent real-world creative diversity.

4.	The paper lacks discussion on computational efficiency or latency of the full multi-agent pipeline — an important factor for large-scale or real-time production.

### Questions
1.	The paper mentions multi-agent orchestration – integrating more than 30 specialized editing agents for diverse operations (e.g., rhythm detection, voice cloning, translation, and trimming). Can the entire system operate end-to-end automatically, or does it require manual intervention between stages? If so, how efficient is the end-to-end pipeline in real use cases? What is the average generation time and computational cost for producing a multi-scene video?

2.	Given so many external or API-based agents, how reproducible are the results if other researchers attempt to re-run the same pipeline?

3.	How well does the system handle long-form or multi-hour videos? Are there memory or latency constraints when orchestrating dozens of agents?

4.	How does the self-reflective orchestration prevent error propagation between dependent agents, and can failed subgraphs be re-executed automatically?

5.	What is the maximum number of characters/scenes that VideoAgent can process simultaneously while maintaining quality and coherence?

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
This paper proposes VideoAgent, an agent-based framework for automated video editing. By introducing a global-aware video shot creation mechanism and a self-reflective agent graph orchestration strategy, VideoAgent demonstrates promising results. Nevertheless, the paper still has several aspects that could be further improved.

### Strengths
1. The paper focuses on the task of video editing and content creation, which holds significant practical value in real-world applications.

2. The paper is well-written and easy to follow, with a comprehensive appendix that provides detailed explanations of the technical aspects of the proposed work.

### Weaknesses
1. The definition and research scope of the task are not clearly articulated. Video editing is a highly broad concept, and the authors should explicitly specify which sub-tasks are covered by this work.

2. In Section 2.3.1, the authors mention functionalities such as face swapping and lip synchronization, yet there appears to be no corresponding agent described in Appendix A.5.

3. The paper lacks methodological novelty and sufficient contribution; the proposed system is largely built upon existing techniques and relies heavily on prompt engineering rather than introducing new algorithmic insights.

4. The overall framework appears redundant and overly complicated. Constructing a dedicated dataset to train a more compact and unified model would likely be more effective.

5. The paper makes extensive use of LLMs, but does not include a dedicated section “Usage of LLMs”.

### Questions
Please refer to Weaknesses.

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
The paper presents VideoAgent, an all-in-one agentic framework that integrates video understanding, editing, and workflow orchestration within a unified system. It introduces a shot planning agent for coherent long-form video generation and a self-reflective agent graph orchestration module that dynamically assembles workflows using specialized agents. Evaluations on the new VideoEdit benchmark show significant improvements over baselines such as VideoRAG and VideoMind. Human evaluations rate its outputs close to professional-level quality, demonstrating strong potential for scalable, automated video creation.

### Strengths
1. Comprehensive End-to-End Pipeline:
The manuscript describes a full pipeline for generating video content from multi-modal inputs. By integrating narrative planning with an execution engine, it effectively links high-level creative intent with concrete video editing and synthesis tasks, offering a coherent workflow from intention to output.

2. Flexible Multi-Agent Orchestration:
The multi-agent orchestration framework constitutes a strong systems contribution. Its graph-based, self-reflective architecture that dynamically assembles workflows from over thirty specialized agents demonstrates scalability and adaptability. This design is well suited to the non-linear and modular nature of video editing tasks.

3. Benchmark and Evaluation Framework:
The introduction of the VideoEdit benchmark is a valuable contribution to the community, offering a standardized resource for comparison in future work. The empirical validation includes ablation studies and performance analyses that provide credible evidence of the system’s efficiency and effectiveness.

### Weaknesses
1. Novelty and Positioning:
The shot-planning module and graph-based orchestration are conceptually related to existing systems such as TeaserGen (Xu et al., 2025) and GPTSwarm (Zhuge et al., 2024). The paper would benefit from clearer articulation of domain-specific innovations tailored to video editing.

2. Evaluation Methodology:
The human evaluation lacks detail on criteria, sample selection, and scoring consistency. The use of a single quality metric limits interpretability, and missing information about excluded baselines and cost calculations weakens transparency.

3. Benchmarking Scope:
The evaluation compares only against general-purpose frameworks. Including domain-specific systems such as ReelDeal or VideoRepurpose would better contextualize performance claims.

### Questions
1. How does the proposed system fundamentally differ from existing narration-driven or graph-based orchestration frameworks like TeaserGen or GPTSwarm?

2. Could the authors provide more detail on the human evaluation protocol, including rating criteria and inter-rater agreement?

3. Does the reported cost-efficiency include the entire pipeline or only selected phases?

4. Are there plans to extend evaluation to additional video categories or domain-specific baselines for a fairer comparison?

### Soundness
2

### Presentation
2

### Contribution
3
