# Agent Data Protocol: Unifying Datasets for Diverse, Effective Fine-tuning of LLM Agents

- Decision: Accept (Oral)
- Scores: 6, 4, 8, 8

## Abstract
Public research results on large-scale supervised finetuning of AI agents remain relatively rare, since the collection of agent training data presents unique challenges. In this work, we argue that the bottleneck is not a lack of underlying data sources, but that a large variety of data is fragmented across heterogeneous formats, tools, and interfaces. To this end, we introduce the Agent Data Protocol (ADP), a light-weight representation language that serves as an "interlingua" between agent datasets in diverse formats and unified agent training pipelines downstream. The design of ADP is expressive enough to capture a large variety of tasks, including API/tool use, browsing, coding, software engineering, and general agentic workflows, while remaining simple to parse and train on without engineering at a per-dataset level. In experiments, we unified a broad collection of 13 existing agent training datasets into ADP format, and converted the standardized ADP data into training-ready formats for multiple agent frameworks. We performed supervised finetuning on the unified data, and demonstrated an average performance gain of $\sim$20\% over
corresponding base models, and delivers state-of-the-art or near-SOTA performance on standard coding, browsing, tool use, and research benchmarks, without domain-specific tuning. All code and data are released publicly, in the hope that ADP could help lower the barrier to standardized, scalable, and reproducible agent training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes the Agent Data Protocol (ADP), a lightweight actions–observations schema that standardizes heterogeneous agent datasets into a common “interlingua.” Thirteen public datasets spanning coding, software engineering, tool use, and web browsing are converted into ADP, then compiled into multiple agent harnesses via a single ADP→SFT step. The authors report consistent SFT gains for 7–8B models on SWE-Bench Verified, WebArena, AgentBench OS, and GAIA, and argue that ADP reduces conversion effort from quadratic in datasets × harnesses to linear, supported by lines-of-code accounting. The work targets community-scale reuse by releasing schemas, converters, and a balanced training mixture.

### Strengths
Concrete unification with measurable engineering payoff. A minimal, well-scoped schema (APIAction, CodeAction, MessageAction; Text/WebObservation) plus validators collapses integration complexity from O(D×A) to O(D+A), with clear LOC evidence for both Raw→ADP and ADP→SFT paths.

Broad empirical utility across agents and tasks. ADP-trained models improve over base models on four benchmarks and exhibit positive cross-task transfer, suggesting the protocol enables effective multi-domain SFT rather than task siloing.

Adoption-oriented tooling. Bidirectional converters, automated quality checks, and a balanced sampling strategy make the proposal practical for other groups to plug in new datasets or harnesses.

Reusability and standardization value. A shared, open schema can reduce duplicated one-off converters and facilitate reproducible studies on agent data at scale.

### Weaknesses
- The unified data is not yet available. 

- Limited contamination and licensing analysis. Decontamination, deduplication across sources, license compatibility, and provenance controls are not documented sufficiently, which weakens confidence in reported gains.

- Ablations do not isolate the protocol’s causal effect. There is no comparison against naïve harmonization baselines such as prompt-normalized concatenation, nor granular ablations on validators, sampling multipliers, or per-dataset contributions.

- Incomplete statistical reporting. Variance across seeds, confidence intervals, and sensitivity to mixture weights are missing, making robustness unclear.

- Schema coverage gaps. Current ADP emphasizes text, tools, code, and web HTML or AX trees; richer multimodality, GUI desktop state, error and rollback semantics, and environment replay metadata are not fully specified.

### Questions
See weakness

### Soundness
3

### Presentation
4

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
The paper observes that the current bottleneck in the development of LLM-based agents lies not in the shortage of data, but in the lack of a unified data standard. To address this, the authors propose ADP as an interlingua for diverse downstream tasks, unifying the formats of agent actions and environment observations. They convert 13 existing agent datasets into the ADP format and train models on this unified dataset. The results show that agents trained with ADP achieve significant performance improvements over the base models, and that models trained on mixed ADP data outperform those trained solely on single-task data in certain tasks.

### Strengths
1. The paper presents an excellent motivation, keenly identifying data fragmentation as a key engineering bottleneck in current agent research and providing a clear direction for future data standardization efforts.

2. It provides a valuable contribution to the open-source community by integrating 13 diverse datasets into a unified ADP format and demonstrating the value of mixed data, thereby establishing a solid data foundation for building general agent capabilities.

3. The experiments are conducted across multiple agent frameworks and consistently show performance improvements, highlighting the effectiveness and generality of the proposed ADP approach.

### Weaknesses
1. The experiments involve an unfair comparison. While the paper emphasizes the importance of unifying agent fine-tuning data formats through ADP, the comparative experiments do not use an equal amount of ADP and non-unified data. Due to the inconsistency in data scale, it is difficult to convincingly demonstrate the advantages of ADP over other data formats, or to support the claim that mixed data is superior to single-task data.

2. The experiments are incomplete. Although the paper selects Qwen3-8B and Qwen2.5-7B-Instruct as baseline models, several tasks lack results for Qwen3-8B, which may raise concerns about the generality of the proposed approach.

3. As a data-centric work, the paper does not provide a complete example of the ADP data format, making it difficult for readers to intuitively understand how ADP differs from existing data representations.

4. The paper has several issues in presentation and formatting. For instance, some acronyms are not capitalized (e.g., "agent data protocol" in Line 015 should be capitalized), inconsistent capitalization appears throughout (e.g., Line 145), some abbreviations are introduced before their full forms (e.g., Line 245), and table captions are inconsistently formatted (Tables 3 and 4 have captions placed below, unlike others).

### Questions
1. I am curious about the extent to which ADP can outperform non-unified data when the amount of training data is controlled to be equal.

2. The ADP framework currently supports only TextObservation and WebObservation, which limits its applicability to certain interaction scenarios. In real-world scenarios, environment observations may be more complex (e.g., structured JSON data or game environment states). How does ADP handle or generalize to such out-of-scope observation types?

3. Table 2 shows that most datasets have high coverage of Function Thought. However, the paper does not further analyze these "thought" processes. I wonder whether training with mixed ADP data not only improves the final action accuracy but also enhances the agent's reasoning quality.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an agent data protocol to standardize various agent tuning datasets into a single format. The goal is to help the community better utilize available agent datasets and reduce the engineering effort required to use them for agent tuning tasks. To this end, authors have defined the ADP standard and converted 13 different agent datasets into this format to show the coverage of the proposed standard. Using this converted data, the authors fine-tune agent models and show that they perform better on various benchmarks, from coding/software engineering/tool calling, etc., demonstrating that this is indeed helpful.

### Strengths
The paper is well written and easy to follow. 
Having a data standard that can help various agent datasets into a single format would help the research community in this area to a great extent and can help with the reusability of the assets with ease.
Adopting existing 13 benchmark datasets to the format and open-sourcing them for the community
Analysis of the various datasets after the conversion and fine-tuning results to show the power of having a standardized data format and the kind of generalization it can bring to the table.

### Weaknesses
Can authors comment on the SOTA numbers for various tasks with similarly sized models? I can see improvements for a selected model from its base performance. Are there any fune-tuned models in a similar parameter range that get better numbers than what is reported here with ADP data?

### Questions
Please check the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents the Agent Data Protocol (ADP), a unified and extensible framework for representing and exchanging multi-agent data in LLM-based systems. ADP standardizes key concepts—such as tasks, agents, trajectories, and scores—into a cohesive schema, enabling interoperability across platforms and supporting tasks like agent analysis, fine-tuning, and benchmarking. Through real-world use cases, the authors show that ADP simplifies data handling and fosters reproducible, collaborative research in the agent ecosystem.

### Strengths
1. **Standardized and Extensible Data Schema**  
   The paper introduces a well-structured, unified schema that captures essential components of agent-based interactions (tasks, agents, trajectories, and scores), addressing long-standing fragmentation in agent data representation.

2. **Practical Utility Across Diverse Agent Systems**  
   ADP demonstrates strong real-world applicability by enabling seamless data sharing and transformation across different platforms, toolchains, and evaluation pipelines, which is crucial for scalable multi-agent benchmarks and model comparison.

3. **Empirical Validation Through Real Use Cases**  
   The usefulness of ADP is effectively showcased through compelling use cases such as cross-agent analysis, supervised fine-tuning, and multi-agent evaluation—providing concrete evidence that the protocol supports reproducible and collaborative agent research.

### Weaknesses
See questions

### Questions
**Relation to Prior Work on Unified Agent Data Frameworks**  
   The paper introduces ADP as a unified schema for agent data, but does not sufficiently discuss its relationship to prior work on similar frameworks, such as *AgentOhana* (Zhang et al., 2024), which also proposes a unified data and training pipeline for agent learning. Could the authors clarify the conceptual and technical differences between ADP and AgentOhana? Specifically, how does ADP improve upon or diverge from previous unification efforts in terms of schema design, data transformation capability, or supported use cases?

[1] Zhang, Jianguo, Tian Lan, Rithesh Murthy, Zhiwei Liu, Weiran Yao, Ming Zhu, Juntao Tan et al. "Agentohana: Design unified data and training pipeline for effective agent learning." arXiv preprint arXiv:2402.15506 (2024).

### Soundness
3

### Presentation
3

### Contribution
3
