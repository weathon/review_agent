# CogVLN: Cognitive Map-Guided Vision-and-Language Navigation in Large-Scale Environments

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Vision-and-Language Navigation (VLN) requires agents to interpret language instructions and navigate to target locations via visual observations. While progress has been made indoors, large-scale outdoor VLN remains underexplored. The large-scale environment representation as a primary challenge. Though dense maps (e.g., point clouds) enable flexible environment modeling, their high memory usage fails to meet navigation’s real-time needs. Additionally, aligning long language instructions with complex environments remains a notable issue. In this paper, we introduce CogVLN, a novel method for VLN in large-scale environments. First, inspired by how humans encode environments, we propose a method for constructing a cognitive map to represent large-scale environments. This method prioritizes encoding key scenes that embody distinct environmental features, while allocating fewer coding resources to scenes with higher consistency. Subsequently, leveraging the constructed cognitive map, we design three core functional modules: a localization module responsible for identifying start and goal vertex, a path planning module tasked with planning navigation routes, and a navigation module dedicated to carrying out the navigation task. During the navigation process, driven by the multimodal large language model (MLLM), CogVLN expresses scene information and receive user feedback in an interactive manner, and further dynamically adjusts the route accordingly. Experimental results in the CARLA Town01 and Town07 environments demonstrate the remarkable performance of our CogVLN.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes CogVLN, a cognitive map-guided framework for large-scale outdoor Vision-and-Language Navigation (VLN). The core contribution is a "cognitive map" that removes redundant waypoints via visual (DINOv2) and semantic (Grounded-SAM + MLLM + BERT) similarity filtering, retaining key scenes (e.g., intersections). CogVLN also includes three modules (localization, path planning, navigation) that enable interactive navigation via user feedback. Experiments on CARLA Town01/Town07 show CogVLN achieves 34.17%/28.33% success rate (SR) without training, outperforming random policies and equidistant topological maps.

### Strengths
1. Cognitive map design: Integrating visual and semantic filtering aligns with human spatial cognition, addressing the trade-off between map compactness and information retention (better than dense point clouds or equidistant maps).

2. Interactive navigation: The user-feedback mechanism effectively solves the long-instruction alignment problem, which is more practical for real-world scenarios (e.g., road blockages) than one-time instruction input.

3. Zero-training utility: Using pre-trained MLLMs and feature encoders avoids tedious environment-specific fine-tuning, lowering the barrier for practical application.

### Weaknesses
1. Inadequate baseline comparisons: The paper only compares with random policies and equidistant topological maps, ignoring recent SOTA outdoor VLN methods (e.g., VLN-VIDEO (Li et al., 2024) which uses driving videos for data augmentation, PReP (Zeng et al., 2024) which optimizes MLLM-driven navigation). Without these comparisons, it is impossible to confirm CogVLN’s competitiveness in the field.

2. Limited experimental scope: All experiments are conducted in CARLA simulation, with no real-world testing. Simulated environments lack dynamic noise (e.g., variable lighting, pedestrian occlusion, traffic jams) that exists in real outdoor scenes, making the generalization of CogVLN highly questionable.

3. Unanalyzed key parameters: The cognitive map’s radius $r$ and similarity threshold $\tau\$ directly affect navigation performance, but the paper only shows how $r$ impacts waypoint count (Appendix Figure 6) without explaining why the chosen $r$ is optimal. Also, no parameter sensitivity analysis (e.g., how SR changes with $\tau\$  is provided, reducing the method’s reproducibility.

4. Fragile interaction mechanism: The framework assumes user feedback is always correct, but in practice, users may provide wrong information (e.g., misjudging road blockages). The paper does not propose a mechanism to detect or correct such errors, leading to potential navigation failures in real use.

5. Unassessed real-time performance: Large-scale outdoor navigation requires low latency, but the paper does not report critical metrics like cognitive map construction time or navigation decision latency. It is unclear whether CogVLN can meet real-time requirements.

6. Some related works are missing citations: 
[1] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation 
[2] Test-time Adaptive Vision-and-Language Navigation 
[3] Learning Vision-and-Language Navigation from YouTube Videos
 [4] Magic: Meta-ability guided interactive chain-of-distillation for effective-and-efficient vision-and-language navigation

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces cognitive map-guided vision-and-language navigation, a method inspired by NeuroBayesSLAM, which aims to improve Vision-and-Language Navigation (VLN) in large-scale environments. The approach proposes a novel cognitive-map-based representation that determines the start and goal locations through visual and linguistic cues and allows dynamic human-agent interaction during navigation.

### Strengths
1. Efficient mapping and querying in large-scale environments remain key challenges in VLN. Building upon NeuroBayesSLAM, this work innovatively integrates cognitive mapping principles with visual-language reasoning, partially addressing this long-standing issue.

2. The paper is clearly written, with a well-defined symbol system and few grammatical or formatting errors. Figures are well-designed, minimalistic, and easy to interpret, which enhances readability.

3. The evaluation protocol and metrics are well structured, providing a fair and interpretable framework for measuring performance.

4. The ablation experiments examine several meaningful factors that may influence effectiveness, such as different MLLM backbones and edge pruning within the cognitive map, offering useful insights into the method's robustness.

### Weaknesses
1. Runtime and Efficiency Analysis: The paper lacks a quantitative analysis of computational efficiency. It would be beneficial to report runtime comparisons, such as the time cost for start/goal determination and human-agent interaction during navigation.

2. Lines 20–21 and Line 178 claim that humans build spatial representations through cognitive maps.  Please add appropriate theoretical grounding.

3. Line 45: The  paper claims that “building an efficient spatial representation is a prerequisite for navigation”. Please add appropriate theoretical grounding.

4. The experiments would be more convincing if tested on more complex outdoor datasets, such as Touchdown or Map2Seq, which are standard benchmarks for large-scale urban VLN.

5. Real-World Deployment: Please discuss whether the proposed system has been (or can be) deployed in real-world scenarios, and what challenges or limitations are expected in transferring from simulation to reality.

6. Clarify whether the implementation will be open-sourced, which is essential for reproducibility.

7. Figure 1: The framework illustration is visually appealing, but it lacks detailed explanations of how the inputs and outputs of each module interact with subsequent components. 

8. Line 86–87: Please add appropriate citations when referring to existing methods.

9. Line 243: The choice of BERT as the encoder needs justification. Why was it selected over other multimodal or VLN-specific language encoders?

10. Line 218: Please explain why DINOv2 was chosen as the visual encoder. How would the performance change if ViT were used?

11. Fix the typo in Line 89. Correct the incorrect Figure reference in Line 213. Resolve reference formatting issues in Line 345. Line 364: The word 'Star' likely should be 'Start'.

### Questions
Please see weaknesses.

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
This paper introduces CogVLN, a framework for vision-and-language navigation in large-scale outdoor environments. The authors propose a new method to construct a cognitive map by first generating a dense topological graph and then pruning redundant waypoints using a two-stage process based on visual (DINOv2) and semantic (MLLM+BERT) similarity. This map is then utilized by a modular, zero-shot system composed of localization, planning, and navigation modules, all orchestrated by a MLLM. 

A key aspect of the proposed framework is its interactive nature; instead of following a single long instruction, the agent provides descriptions of its current location and adjusts its path based on simple user feedback, allowing for dynamic re-planning. 

The system is evaluated in two CARLA environments and claims to achieve excellent performance without any task-specific training.

### Strengths
(1) The work addresses the challenging and highly relevant problem of scaling VLN agents to large, complex outdoor environments, which is a critical step toward real-world applications.

(2) I think the core idea of constructing a cognitive map via a two-stage pruning process is a definite strength. The method, which filters redundancy based on both visual appearance and high-level semantics, is well-motivated by principles of human cognition and offers a promising way to create efficient and meaningful environmental representations. So the shift from a traditional one-off instruction-following task to an interactive, human-in-the-loop navigation paradigm is a significant conceptual contribution.

### Weaknesses
My main concerns with this submission revolve around the experimental evaluation, which I believe is insufficient to support the paper's claims.

(1).The only baseline presented in Table 1 is a random policy. For a task as complex as navigation, a random agent is expected to have a performance near zero, making it a trivial point of comparison. Without comparisons to any other established or even simple heuristic-based methods, it is impossible to assess whether the reported success rates are actually good. 

(2).The paper proposes a novel interactive navigation module. However, its benefit is not isolated. I'd like to know how a non-interactive agent, using the same cognitive map and MLLM-based planner, would perform. Such an ablation would be crucial to demonstrate the true value of the human-in-the-loop component.

### Questions
Could you please kindly justify the decision to only compare CogVLN against a random policy? Why were no other baselines, even simple ones, included in the main results table(Table 1) ?

The interactive navigation is a core part of your framework. To understand its contribution, what is the performance of your system if the interaction is removed? For instance, what happens if the agent follows the initially planned path to completion without any user feedback?

And a confusion regarding the task itself: The input seems to be visual and textual descriptions of a start and end point. How does this setup relate to existing outdoor VLN datasets like Touchdown, where the agent is given a series of directions to follow? I'd like to know the rationale behind this particular problem formulation.

### Soundness
2

### Presentation
2

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
This paper proposes CogVLN, a novel vision-and-language navigation (VLN) framework for large-scale outdoor environments. Inspired by human cognitive mapping, it constructs a compact topological representation of the environment by prioritizing distinctive scenes and filtering redundant ones based on visual and semantic similarity. Leveraging a Multimodal Large Language Model (MLLM), the system features three core modules for localization, path planning, and interactive navigation, where the agent dynamically adjusts its route based on user feedback. Evaluated in CARLA's Town01 and Town07, CogVLN achieves notable success rates (34.17% and 28.33%) without any training, demonstrating robust performance in complex, large-scale settings.

### Strengths
1. The cognitive map construction is a key strength, moving beyond dense or equidistant topological maps. By selectively encoding visually and semantically distinct waypoints, it creates a highly compact and efficient representation that aligns with human spatial memory, reducing redundancy and computational load while preserving critical navigational features for large-scale environments.
2. The framework effectively addresses the challenge of aligning long instructions with complex environments by introducing an interactive navigation module. Instead of relying solely on initial instructions, the agent outputs scene descriptions and re-plans routes in real-time based on user feedback, enabling dynamic adaptation and significantly improving robustness in unpredictable scenarios.
3. This paper is good writing.

### Weaknesses
1. The paper fails to provide a clear analysis and elaboration of the problem definition. Compare the traditional VLN task (inputs all instructions at once), what is the different and challenge of providing dynamic interactions? Assuming that adding human interaction makes the task easier, a success rate of only 30% is not a very good performance.
2. The paper's empirical validation is significantly weakened by the lack of comparison with established sota methods in indoor or outdoor VLNs. The only benchmark provided is a random policy, which is not a meaningful benchmark. Furthermore, the method lacks deployment in real-world environment to verify its robustness and operational efficiency.
3. The cognitive map construction, while inspired by neuroscience, lacks empirical validation of its superiority over a well-designed equidistant topological map. The ablation shows better performance, but the baseline "Equ" map's construction details are not specified. It is unclear if the performance gain comes from the human interaction or simply from having a more intelligently pruned graph. Additionally, the end-to-end system's computational cost is not discussed. The process involves multiple calls to large models (DINOv2, Grounded-SAM, MLLM) for map construction, localization, and planning, which likely leads to error accumulation and latency, making its applicability to real-time navigation questionable.

### Questions
1. The modeling of vision-language navigation (VLN) for drones can be understood as a partially observable Markov decision process that does not rely on known maps for navigation. However, the global map constructed using NeuroBayesSLAM seems to allow the task to navigate on a pre-explored map. Is this inconsistent with the traditional VLN task?
2. The article lacks some details:
    - How is the Equ map in Table 2 constructed?
    - What is the significance of the localization module in locating the starting and target locations?
    - The Navigation Module lacks a detailed description of its specific details. What is the image matching method and the exploration mechanism of the new node?

### Soundness
2

### Presentation
3

### Contribution
2
