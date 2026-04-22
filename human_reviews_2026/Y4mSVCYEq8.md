# ERNav: A Unified, Realistic Benchmark for Embodied AI with Exploration, Representation, and Navigation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Current embodied AI benchmarks typically focus only on the final stage of the embodied process, such as following instructions or answering scene-related questions. These evaluations often unrealistically assume access to perfect perception data of the environment and overlook the earlier stages of exploration and representation construction, which are indispensable for real-world deployment. In addition, these benchmarks are often restricted to smaller-scale, room-level environments and short, object-centric instructions, falling to capture the complexity of larger buildings where agents must operate across multiple rooms and floors while reasoning over long instructions tied to global layouts. To address these gaps, we introduce ERNav, the first unified benchmark for embodied AI that integrates Exploration, Representation, and Navigation into an end-to-end task pipeline. In ERNav, agents must actively explore the environment, construct global representations from noisy RGB-D observations, and then localize targets directly from natural language instructions that often require reasoning over entire buildings. This unified formulation differs from existing benchmarks by aligning all stages of the embodied pipeline and scaling evaluation to realistic building-level settings, creating a challenging and practical testbed for embodied AI. We also propose 3D-LangNav as a strong baseline. As a divide-and-conquer framework, it employs a dual-sighted exploration strategy to collect diverse observations and construct high-quality 3D representations, followed by language grounding and spatial reasoning via a fine-tuned large language model (LLM). Extensive experiments show that ERNav poses significant new challenges for existing methods, while 3D-LangNav achieves strong performance, reaching more than twice the success rate (SR) of state-of-the-art 3D-MLLMs. Moreover, by structuring the task into three progressively harder, sequentially dependent subtasks as a whole pipeline, ERNav enables systematic analysis of how each stage contributes to overall performance, providing clear directions for future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces ERNav, a unified benchmark designed to integrate Exploration, Representation, and Navigation into a single embodied AI evaluation framework. The benchmark defines three sub-tasks—EnvExp, EnvRep, and EnvNav—to separately assess exploration ability, representation quality, and navigation performance, respectively. The authors also propose a baseline model, 3D-LangNav, which jointly leverages language and 3D perception for embodied navigation. Extensive experiments across multiple embodied AI models and 3D-LLMs are reported, aiming to establish a standardized evaluation protocol for realistic embodied reasoning and navigation.

### Strengths
1. Timely and relevant topic.
The work tackles an important and emerging direction in embodied AI—unifying exploration, reconstruction, and navigation—which are often studied in isolation.

2. Clear task formulation.
The benchmark design into three sub-tasks (EnvExp, EnvRep, EnvNav) is logical and well-motivated. The inclusion of EnvRep, in particular, provides a valuable bridge between perception and action evaluation.

3. Informative baseline.
The proposed 3D-LangNav baseline achieves reasonable results across multiple tasks and offers insights into the interplay between language grounding and navigation performance.

### Weaknesses
1. Limited related work discussion.
The paper omits discussion of closely related efforts such as MSR3D [1], which already provides a comprehensive evaluation protocol connecting scene understanding and navigation. The relationship and differences between ERNav and such benchmarks need clarification.

2. Lack of dataset statistics and quality assessment.
As a benchmark paper, it is important to include dataset composition, diversity metrics, and quality control analysis. Without these, it is difficult to judge the representativeness or reliability of the proposed benchmark.

3. Unclear experimental settings.
The description of how representation-based methods and 3D-LLMs are adapted for navigation tasks is vague. The implementation details, training configurations, and evaluation criteria should be explained more explicitly.

4. Ambiguous comparisons in Table 3.
The results in Table 3 mix heterogeneous model categories (representation-based, language-based, 3D-LLMs) without clarifying the evaluation setup. This makes it hard to interpret relative performance. Improving table organization and adding clearer experimental notes would enhance readability and fairness.

[1] Multi-modal Situated Reasoning in 3D Scenes

### Questions
1. How are the representation-based methods in Table 3 integrated into the navigation framework? Are they directly used as frozen feature extractors, or fine-tuned for embodied tasks?

2. Were the 3D-LLMs in Table 3 fine-tuned on the same navigation trajectories or pretrained independently?

3. How large is the dataset used for each sub-task (EnvExp, EnvRep, EnvNav), and how is data quality validated?

4. Could the authors elaborate on the differences between ERNav and MSR3D, particularly in evaluation design and practical difficulty?

### Soundness
3

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
3

### Summary
This paper introduces ERNav, a new benchmark that incorporates exploration, map construction and navigation at building level. The proposed benchmark introduces a higher level of challenge as it better mimics the complex real-world scenario. The paper also introduces a strong baseline 3D-LangNav that can address the above mentioned 3 points simultaneously. Extensive experiments and comparison with baseline models suggested that ERNav poses challenge to existing methods.

### Strengths
1. The paper addresses the long-standing problem where embodied AI benchmarks and 3D vision benchmarks are not fully aligned with the real-world setting.

2. Extensive experiments show that there still exist gaps between the existing methods and the more realistic setting.

3. Building level benchmarks could inspire research towards more realistic settings.

### Weaknesses
1. The third subtask provided by the benchmark, i.e. EnvNav, seems to be too easy if standalone. The model is only asked to provide a coordinate, without actual execution in the scene, making the input coordinates useless since without it the model can also predict a target coordinate.

2. Therefore, it is questionable whether the provided benchmark is more challenging since all the actual navigation done in VLN works are simplified when given actual instructions.

3. The benchmark doesn't take into consideration the dynamic scenes, where objects locations can vary largely, making the proposed task setting less efficient.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes a new benchmark for vision-language navigation (VLN) called ERNav. The benchmark includes three subtasks: scene exploration, global scene representation construction after exploration, and target location identification. Separate metrics are designed for each subtask. In addition to the ERNav benchmark, this work also introduces a baseline model named LangNav. Experimental results show that ERNav presents new challenges for existing methods, while 3DLangNav achieves strong performance.

### Strengths
1. Existing embodied AI benchmarks indeed have many limitations and remain far from real-world applications. This work clearly identifies these limitations and proposes a new benchmark to address them. The motivation is well-grounded and appreciated, and the problem studied in this paper is both important and timely.
2. The writing is clear and easy to follow.
3. For the new benchmark, the authors conducted comprehensive experiments to evaluate existing methods.
4. The paper also introduces a baseline that outperforms many previous approaches.

### Weaknesses
1. Although the proposed benchmark includes three stages—exploration, representation, and navigation—the ultimate goal in real-world applications remains navigation. Beyond providing separate metrics for each subtask, it is more meaningful to investigate how the results of the first two subtasks influence the final navigation performance. This aspect is missing from the paper. The authors also claim that the benchmark is end-to-end, but in my view, it is actually modular, since each subtask is evaluated independently without a unified metric that supports end-to-end performance analysis.
2. It would be useful to support methods that can perform representation and navigation but not exploration (e.g. ConceptGraphs), by incorporating a rule-based exploration module. This would allow those methods to participate in all three subtasks and enable a fairer comparison in Table 3.
3. For a new benchmark in embodied AI, it would be valuable to measure human performance on the proposed tasks, providing the community with an estimate of the potential upper bound.
4. It would also be beneficial to include demonstrations and a discussion of current method limitations, along with potential future research directions.

### Questions
1. Lines 67–71 state that ERNav requires the model to explore the scene based on noisy RGB-D inputs, but I could not find any explanation in the paper describing how the benchmark introduces noise to the RGB-D inputs or ensures that the noise realistically reflects real-world conditions.
2. Line 212: It is unclear why the authors use L1-CD instead of L2-CD for evaluation.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a new end-to-end benchmark where an embodied agent must first explore a building-scale environment, construct a global spatial representation from noisy RGB-D input, and then interpret natural-language instructions to locate targets in a unified pipeline. Prior benchmarks often assume perfect perception or focus only on navigation or question-answering, neglecting the full embodied process of exploration → representation → language-grounded navigation. They present a baseline method ‹3D‑LangNav› that uses a dual-sighted exploration strategy plus an LLM-grounded reasoning component, and show that existing methods struggle on this benchmark, highlighting significant gaps in current embodied-AI capabilities.

### Strengths
The authors propose a single pipeline of exploration-representation-navigation stages, which have been explored in part in previous work.
The authors conduct experiments with various baselines for respective stages (e.g., exploration, representation and navigation),

### Weaknesses
- It is unclear why we need separate, explicit exploration and representation stages. For example, prior work such as Chaplot et al., 2020a conducts the exploration, representation and navigation simultaneously. What are the benefits of building a map through pre-exploration before navigation, which may require more computational cost compared to prior work?
  - In addition, most of the building blocks come from existing work (Sec. 3.3), raising a novelty concern about the proposed framework. The main contribution may come from the motivation of it, but this is not well described and supported by numbers.
- The paper mentions that the proposed method takes noisy RGBD observations, but it is unclear which part is noisy (e.g., color, depth, camera poses if applicable, ...). It lacks a description.
  - In addition, how much sensitive is the proposed method to the level of noisy? No quantitative analyses of this is conducted.
- In Table 1, the comparison is not fair as the proposed approach uses exploration steps far more than the baselines. The result with the fixed exploration cost should be provided.
- In Table 2, the proposed model underperforms the baselines in several metrics (e.g., R@1%, MR and MRR in Object-level), but the relevant discussion is little to no provided.
- In Table 3, GridMM shows better (NE) or competitive (SR) performances compared to the proposed model without exploration and representation stages, raising a concern of the necessity of the stages.
- (Minor) The paper structure could be improved by avoiding single, long paragraphs and inappropriate capitalization in the title.

### Questions
- Does the proposed agent generalize to representations obtained from different camera configurations, such as different FOV, heights, etc.?

### Soundness
3

### Presentation
3

### Contribution
2
