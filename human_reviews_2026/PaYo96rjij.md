# Lifelong Embodied Navigation Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Embodied navigation agents powered by large language models have shown strong performance on individual tasks but struggle to continually acquire new navigation skills, which suffer from catastrophic forgetting. We formalize this challenge as lifelong embodied navigation learning (LENL), where an agent is required to adapt to a sequence of navigation tasks spanning multiple scenes and diverse user instruction styles, while retaining previously learned knowledge. To tackle this problem, we propose Uni-Walker, a lifelong embodied navigation framework that decouples navigation knowledge into task-shared and task-specific components with Decoder Extension LoRA (DE-LoRA). To learn the shared knowledge, we design a knowledge inheritance strategy and an experts co-activation strategy to facilitate shared knowledge transfer and refinement across multiple navigation tasks. To learn the specific knowledge, we propose an expert subspace orthogonality constraint together and a navigation-specific chain-of-thought reasoning mechanism to capture specific knowledge and enhance instruction-style understanding. Extensive experiments demonstrate the superiority of Uni-Walker for building universal embodied navigation agents with lifelong learning. We also provide the code of this work in the Supplementary Materials.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of lifelong embodied navigation (LENL), where an agent must continually adapt to new navigation tasks without catastrophic forgetting. The authors propose Uni-Walker, a novel framework that combines LoRA-based expert decomposition with task-aware knowledge aggregation (TAKA) to effectively manage task-specific and shared knowledge across sequential navigation tasks. Uni-Walker is evaluated on a newly introduced LENL benchmark, comprising 18 task sequences with varying instruction styles and environments. Results show significant improvements over state-of-the-art continual learning methods, achieving an average SPL of 61% compared to 38% for the strongest baseline (SD-LoRA+TAKA). The work highlights the importance of efficient knowledge reuse and dynamic adaptation in lifelong learning scenarios.

### Strengths
- **Originality**:  The paper tackles a timely and important problem—how to enable agents to continuously learn new navigation tasks while avoiding catastrophic forgetting. The proposed Uni-Walker framework introduces a novel combination of LoRA experts and CLIP-based retrieval for task-aware knowledge aggregation, offering a fresh perspective on how to balance shared and task-specific knowledge in lifelong learning settings.

- **Quality**: The experimental design is rigorous and comprehensive. The authors introduce a new LENL benchmark consisting of 18 task sequences, covering various instruction styles and environments. They report detailed results and demonstrate consistent performance improvements across early and late tasks. Extensive ablation studies validate the effectiveness of each component (DE-LoRA, TAKA) and provide insights into their contributions.

- **Clarity**:  The paper is well-written and clearly structured. Key concepts are explained with clear diagrams and pseudocode. The overall framework is well illustrated and the evaluation setup helps readers understand the technical details. The comparison with prior work is fair and balanced, acknowledging both the strengths and limitations of existing approaches.

### Weaknesses
- **Limited evaluation**: While the experiments show strong performance on the LENL benchmark, it would be beneficial to evaluate Uni-Walker on additional datasets or more diverse environments and benchmarks to assess its generalizability. The current focus on indoor navigation tasks may limit the broader applicability of the findings.

- **Insufficient comparison with non-LoRA methods**: The paper primarily compares Uni-Walker with other LoRA-based continual learning methods (e.g., MoLA, BranchLoRA). Including comparisons with other continual learning methods would provide a more comprehensive understanding of Uni-Walker’s relative advantages and limitations.

- **Real World Applications**: The success of TAKA depends heavily on the quality of CLIP embeddings for aligning instructions and observations. If there is significant semantic drift or if instructions vary widely in style, the retrieval mechanism might degrade. Furthermore, the paper emphasizes “training-free” inference, but it still requires training a new expert for each incoming task. Clarifying the assumptions about task boundaries and the feasibility of applying Uni-Walker in truly online settings would strengthen the claims.

### Questions
See Weaknesses

### Soundness
3

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
3

### Summary
In this work, the authors introduce a lifelong embodied navigation learning task, where an agent is required to continually acquire new navigation skills, without catastrophic forgetting on the previously learned knowledge. To achieve this goal, they design a Uni-Walker to decouple navigation knowledge into task-shared and task-specific components. Then they design different strategies to learn shared and specific knowledge for life-long learning.

### Strengths
* Novelty

In this work, the author propose a kind of new task in navigation, namely, lifelong embodied navigation learning. This task itself is interesting and practical for embodied navigation in the real world. 

* Clarity

The paper is with good structure. Hence, the clarity is basically good.

* Significance

This paper focuses on a new task of lifelong embodied navigation learning. It would possibly bring new research works for navigation. Hence, the significance is basically OK for this community.

### Weaknesses
* Method

Even though the task is kind of new, the techniques to address this task are relatively straightforward. It basically leverages LORA adaption in the decoder for decoupling navigation knowledge into shared and specific parts. Then, they integrate the existing strategies to learn these two types of knowledge for life long learning. Please futher clarify the key novel design in the proposed framework, instead of adapting and integrating the straightforward strategies together on a newly-defined task.

* Figure Issue

 The figures (e.g., Fig 1-2) should be further re-arranged. The words are too small to see. The whole figure looks very crowded.

### Questions
Please see the weakness section.

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
This paper presents a comprehensive approach to Lifelong Embodied Navigation Learning (LENL), a novel problem setting where embodied agents must continually adapt to new navigation tasks across diverse scenes and instruction styles while mitigating catastrophic forgetting of previously acquired knowledge. The authors propose Uni-Walker, a framework built upon a Decoder Extension LoRA (DE-LoRA) architecture that explicitly decouples navigation knowledge into task-shared and task-specific components.

Uni-Walker introduces several key strategies:
- Knowledge Inheritance Strategy (KIS) and Experts Co-Activation Strategy (ECAS) to facilitate the transfer and refinement of task-shared knowledge.
- Expert Subspace Orthogonality Constraint (ESOC) and Navigation-Specific Chain-of-Thought (NSCoT) to capture and enhance task-specific knowledge and instruction understanding.
- Task-Aware Knowledge Aggregation (TAKA) for dynamically activating relevant experts during inference.

The paper also introduces a new lifelong embodied navigation benchmark consisting of 18 tasks with varying scenes and instruction styles (Vision-Language Navigation - VLN, Object Localization Navigation - OLN, Dialogue Understanding Navigation - DUN). Extensive experiments demonstrate that Uni-Walker significantly outperforms state-of-the-art LoRA-based continual learning methods and other universal navigation agents, achieving superior success rates and substantially lower forgetting rates across multiple metrics.

### Strengths
1. The paper introduces a novel problem (LENL) and a corresponding benchmark, which is a vital contribution. The Uni-Walker framework, with its DE-LoRA and specialized strategies (KIS, ECAS, ESOC, NSCoT, TAKA), demonstrates high originality in combining and extending existing ideas for the specific challenges of embodied lifelong learning.
2. The empirical results are robust and convincing. Uni-Walker achieves state-of-the-art performance, significantly reducing catastrophic forgetting compared to numerous strong baselines. The comprehensive ablation studies rigorously validate the effectiveness of each component, reinforcing the quality of the proposed solution.

### Weaknesses
1. While each component of Uni-Walker (DE-LoRA, KIS, ECAS, SSC, ESOC, NSCoT, TAKA) is well-justified, the overall framework is quite intricate with multiple interacting mechanisms. It might be challenging to disentangle the precise individual contributions or to simplify the architecture without performance degradation. A discussion on the trade-offs between complexity and performance, or potential avenues for simplification, could be beneficial.
2. While the new benchmark is excellent, 18 tasks, even with diverse scenes and instruction styles, might still be considered limited for truly "lifelong" learning in the vast and unpredictable real world. All scenes are from the Matterport3D simulator. A discussion on how the framework might scale to hundreds or thousands of tasks, or transfer to real-world robots (as briefly mentioned in future work), would strengthen the paper.
3. The paper lists several hyperparameters. While values are provided, a more detailed sensitivity analysis of the most critical hyperparameters could provide further insights into the robustness and general applicability of Uni-Walker.
4. The method involves managing multiple experts, computing Fisher Information Matrices, and performing retrieval for Task-Aware Knowledge Aggregation. While LoRA is parameter-efficient, the runtime and memory overhead for a very large number of tasks (e.g., hundreds or thousands) could become substantial. A more explicit discussion of the computational scalability with increasing task numbers would be valuable.
5. The paper mentions 3 tasks (S16, S17, S18) for unseen generalization. While these are included in the average results, a dedicated discussion or a separate table comparing Uni-Walker's performance specifically on these unseen tasks against baselines would highlight its true generalization capabilities more clearly.

### Questions
1. Could the authors elaborate on the computational and memory overhead of managing the Fisher Information Matrix (FA,t) and the retrieval index (Re) as the number of tasks (T) grows very large (e.g., hundreds or thousands)? Are there strategies to mitigate this growth, or is it a known limitation for extremely long lifelong learning sequences?
2. For the Task-Aware Knowledge Aggregation (TAKA), what is the sensitivity of Uni-Walker's performance to the choice of K (number of activated experts) and µ (similarity threshold for masking)? Have you experimented with adaptive ways to determine K or µ?
3. The paper states that "task-id t is agnostic during the testing phase." How is the instruction style (VLN, OLN, DUN) determined at inference time to apply the correct Navigation-Specific Chain-of-Thought (NSCoT)? Is it inferred from the user instruction, or is the instruction format implicitly indicating the style?
4. For the Knowledge Inheritance Strategy (KIS), PCA is applied to learned experts having the "same instruction knowledge." How is this grouping of "same instruction knowledge" determined? Is it hardcoded based on the benchmark's predefined instruction styles, or is there an automatic mechanism to identify similar instruction knowledge for new, previously unseen instruction styles?
5. Could the authors provide a more focused analysis or a separate table comparing the performance of Uni-Walker and the top-performing baselines specifically on the 3 "unseen generalization" tasks (S16, S17, S18)? This would provide clearer evidence of the framework's ability to generalize to truly novel environments/instruction combinations.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces Lifelong Embodied Navigation Learning, a new problem setting where an agent must continually learn a sequence of navigation tasks across diverse scenes and instruction styles without suffering from catastrophic forgetting. To address this, the authors propose Uni-Walker, a novel framework built upon large language models. The core of Uni-Walker is a Decoder Extension LoRA architecture that decouples knowledge into a task-shared subspace and multiple task-specific "expert" subspaces. The framework includes specific strategies to manage this knowledge: a KIS and anECAS to learn shared knowledge, and an ESOC and NSCoT to learn distinct, task-specific skills. At inference time, a TAKA mechanism selects the most relevant experts for the current task. The authors create a new benchmark for LENL and demonstrate through extensive experiments that Uni-Walker significantly outperforms existing continual learning methods, achieving a higher success rate while drastically reducing forgetting.

### Strengths
- Novel Problem: The paper formally introduces and tackles the problem of lifelong learning for embodied navigation, a crucial challenge for developing general-purpose agents.
- Comprehensiv Method: Uni-Walker is a well-thought-out framework that addresses the core challenges of knowledge transfer and catastrophic forgetting by explicitly decoupling and managing shared and task-specific knowledge.
- Rigorou Evaluation: The experimental validation is a major strength. The creation of a new benchmark, comparison against a wide array of strong baselines, and thorough ablation studies for each component of the model provide convincing evidence for the method's effectiveness.
- Significant Performance Gains: The results are not incremental. Uni-Walker shows large improvements in both task success and resistance to forgetting.

### Weaknesses
System Complexity: The final Uni-Walker model is a composite of many components (KIS, ECAS, SSC, ESOC, NSCoT, TAKA). While the ablation studies justify each part's inclusion, the overall system is quite complex. A discussion on the relative importance of these components or potential simplifications would be beneficial.

### Questions
Could you comment on the scalability of the Uni-Walker framework as the number of sequential tasks grows into the hundreds? Specifically, how does the performance of the Task-Aware Knowledge Aggregation (TAKA) mechanism degrade, if at all, as the pool of retrievable experts becomes very large?

### Soundness
3

### Presentation
3

### Contribution
3
