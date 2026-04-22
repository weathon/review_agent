# MA-EgoQA: Question Answering over Egocentric Videos  from Multiple Embodied Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 8, 4

## Abstract
As embodied models become powerful, humans will collaborate with multiple embodied AI agents at their workplace or home in the future. To ensure better communication between human users and the multi-agent system, it is crucial to interpret incoming information from agents in parallel and refer to the appropriate context for each query. Existing challenges are to effectively compress and communicate high volumes of individual sensory inputs in the form of video and to correctly aggregate multiple egocentric videos to construct system-level memory. In this work, we first formally define a novel problem of understanding multiple long-horizon egocentric videos simultaneously collected from embodied agents. To facilitate research in this direction, we introduce MultiAgent-EgoQA (MA-EgoQA), a benchmark designed to systemically evaluate existing models in our scenario. MA-EgoQA provides 1.7k questions unique to multiple egocentric streams, spanning five categories: social interaction, task coordination, theory-of-mind, temporal reasoning, and environmental interaction. We further propose a simple baseline model for MA-EgoQA named EgoMAS, which leverages shared memory across embodied agents and agent-wise dynamic retrieval. Through comprehensive evaluation across diverse baselines and EgoMAS on MA-EgoQA, we find that current approaches are unable to effectively handle multiple egocentric streams, highlighting the need for future advances in this direction.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MA-EgoQA, a benchmark dataset for question answering over long-duration, multi-agent egocentric videos. The benchmark is built on the EgoLife dataset, which features 266 hours of video from 6 agents interacting in a shared house over 7 days. The authors' core contribution is a set of 1.7k question-answer pairs specifically designed to be answerable only by aggregating information from multiple agents' video streams. The benchmark spans five challenging, multi-agent categories: Social Interaction, Task Coordination, Theory-of-Mind (ToM), Temporal Reasoning, and Environmental Interaction.

The paper uses a "single-agent filtering" step to remove any questions solvable by one agent's perspective alone. The authors also propose EgoMAS, a training-free baseline model that uses an event-based shared memory and agent-wise dynamic retrieval.

Experimental results, testing over 16 baselines (including SOTA LLMs and Video LLMs), show that MA-EgoQA is challenging. The top model, Gemini-2.5-flash, achieves only 36.93% accuracy (vs. 20% random chance), and all models struggle with the Theory-of-Mind category.

### Strengths
- The paper addresses a new and interesting topic of multi-agent egocentric videos, where video streams are captured continuously during operation.

- The data is verified and selected by human annotators.

### Weaknesses
- Not enough qualitative analysis. Can the author show and analyze more benchmark samples? One sample is not enough to help audience understand the scope and quality of the benchmark data.

- The performance gain from adding video frames in the EgoMAS (Text+Video) variant over the EgoMAS (Text) variant is very modest (35.96% vs. 35.55%). This suggests either that the text is sufficient for most questions or that the model's method of incorporating video (sampling 8 frames) is not sophisticated enough to be impactful.

### Questions
- More qualitative analysis 

- How do the VLM models handle long videos in the evaluation of Table 2? 

- What is the distribution of video lengths in the benchmark? How does this influence the performance?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces a novel MA-EgoQA benchmark designed for multi-agent egocentric video question answering. It aims to evaluate collaborative understanding and reasoning among multiple embodied agents. Each question in the dataset is designed to depend on multiple agents’visual observations, temporal relations, and mental states. The authors further propose a training-free multi-agent framework with shared memory and dynamic retrieval to perform system-level reasoning over multi-agent video data.

### Strengths
1. MA-EgoQA is the first benchmark to target multi-agent collaborative reasoning in egocentric settings, which extends beyond single-agent VQA and VideoQA tasks.
2. The dataset integrates multiple synchronized first-person videos and corresponding QA pairs that require cross-agent temporal and social reasoning.
3. It provides a unified evaluation platform for multi-agent perception, memory sharing, and cross-view reasoning, a meaningful direction for embodied AI research.
4. The proposed EgoMAS baseline introduces a reasonable training-free design based on shared memory and agent-wise retrieval, showing competitive performance with larger commercial models.

### Weaknesses
1. Current experiments mainly compare different models’overall performance but do not disentangle the sources of task difficulty (e.g., long-horizon temporal reasoning, multi-agent information fusion, or multi-modal dependency).
2. Although the paper claims that ToM questions rely on dialogue and semantic context, it provides no quantitative evidence showing the impact of transcribed speech versus visual-only inputs.
3. There is no sensitivity analysis on the number of agents used. It remains unclear how performance changes when reasoning over fewer or more viewpoints, leaving the true impact of multi-agent data unverified.
4. The shared-memory and dynamic-retrieval modules are not compared with simpler baselines (e.g., uniform sampling, or event-based summarization), which weakens the argument for their necessity.
5. The design of ToM questions likely depends heavily on ASR transcripts, given the absence of raw audio or gaze data. The extent of this dependency is not analyzed, raising uncertainty about whether these tasks truly reflect visuo-cognitive reasoning or are predominantly text-driven.
6. The following datasets are missing from the background or comparisons:

[R1] Mm-ego: Towards building egocentric multimodal LLMs, ICLR, 2025.

[R2] Egotextvqa: Towards egocentric scene-text aware video question answering, CVPR, 2025.

[R3] Assistq: Affordance-centric question-driven task completion for egocentric assistant, ECCV, 2022.

### Questions
Please refer to the weaknesses.

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
4

### Summary
This paper introduces **MA-EgoQA**, a multi-agent egocentric VideoQA benchmark designed to advance research on multi-agent collaboration and human–robot communication. The dataset is built upon **EgoLife**, featuring six actors (agents) and seven days of continuous life-log recordings. The question–answer pairs are automatically generated to cover six aspects: social interaction, task coordination, theory of mind, temporal reasoning, and environmental interaction. To address this task, the authors propose the **EgoMAS** framework, which incorporates a shared memory module and a system-to-individual retrieval mechanism. Experimental results demonstrate that EgoMAS consistently outperforms all baseline methods.

### Strengths
1.	MA-EgoQA: The first multi-agent egocentric VideoQA benchmark covering multiple multi-agent relevant QA tasks.
2.	EgoMAS: a simple but reasonable solution framework that achieves superior performance than strong baselines. 
3.	Comprehensive baseline analyses, with well-structured presentation.

### Weaknesses
1.	The biggest concern in my side is that the questions seem to favor text-based information for answering, as indicated in Table 2. For example, EgoMAS, which utilizes both text and video inputs, only achieves a 0.4% improvement over its text-only counterpart. I also observe that the QA pairs are generated from captions and transcripts. Such text sources inherently lack fine-grained visual details and 3D spatial cues that are critical for embodied agents to understand and navigate indoor environments. I recommend the authors discuss this limitation in the paper.

2.	Except for its egocentric aspect, both multi-agent theory-of-mind [1], social interaction and temporal reasoning [2], environmental interaction [3] are featured in previous datasets. There is a lack of related discussion and comparison in the paper—what are the differences (new challenges) of such questions in MA-EgoQA and in previous datasets?

[1] Shi H, Ye S, Fang X, et al. Muma-tom: Multi-modal multi-agent theory of mind[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(2): 1510-1519.

[2] Xiao J, Shang X, Yao A, et al. Next-qa: Next phase of question-answering to explaining temporal actions[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 9777-9786.

[3] Patraucean V, Smaira L, Gupta A, et al. Perception test: A diagnostic benchmark for multimodal video models[J]. Advances in Neural Information Processing Systems, 2023, 36: 42748-42761.

### Questions
Will the dataset be released?

### Soundness
4

### Presentation
4

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
This paper proposed MA-EgoQA, a benchmark for multi-agent egocentric video QA of embodied agent scenarios. It constructs QA datasets grounded in long-horizon video, and categorizes them in 5 categories. It evaluates multiple LLMs and agents, and proposed a new method, EgoMAS.

### Strengths
1. the benchmark fills a recognized gap in existing datasets
2. the dataset construction pipeline and quality control is thorough and clearly described
3. the evaluation is comprehensive, containing most of the open-source LLMs, close-source LLMs, and other agents

### Weaknesses
1. the dataset is generated based on EgoLife, it would be better if the author could include more other scenarios rather than only EgoLife
2. the proposed EgoMAS is simple, and does not exhibit a significant improvement, the presence of this method is not so meaningful
3. the windowing strategies and retrieval granularities may not be optimized equivalently across these methods
4. the ablation study should verify the contribution of submodules 4W1H and BM25
5. the efficiency of shared memory, which might be the most important part when this method is applied in real-world, is not discussed in this paper

### Questions
1. Why ToM is the hardest task, did you conduct any thorough analysis on the possible reasons?

### Soundness
2

### Presentation
2

### Contribution
2
