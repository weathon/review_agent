# Topo-AeroVLN:  Cognitive Topological Mapping for Brain-Inspired Aerial Vision-Language Navigation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Navigating large-scale environments remains a major challenge for autonomous agents.  Traditional methods often rely on detailed metric maps, whereas biological systems efficiently navigate using sparse, cognitive topological maps that support high-level reasoning.  We present Topo-AeroVLN, a brain-inspired framework enabling unmanned aerial vehicles (UAVs) to perform vision-and-language navigation from a top-down perspective.  Our method incrementally constructs a multi-level topological map by abstracting aerial observations into road-bounded regions and internal semantic objects.  A dynamic graph update mechanism, combining multimodal embedding similarity with spatial containment, ensures efficient and scalable map construction.  Multimodal Large Language Models (MLLMs) align natural language instructions with map vertices, supporting robust language-driven topological planning.  Experiments demonstrate strong spatial coverage and navigation performance in complex urban environments.  Topo-AeroVLN provides a generalizable, interpretable framework for UAV navigation that adapts to unseen environments without prior maps or extensive retraining.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a brain-inspired cognitive topological mapping framework for UAV vision-and-language navigation from aerial views. Topo-AeroVLN incrementally constructs a multi-level topological map by abstracting aerial observations into road-bounded regions and semantic objects within those regions. The proposed algorithm updates the map graph dynamically via multimodal embedding similarities and spatial containment, and aligns natural language queries with map vertices using multimodal large language models (MLLMs) to enable language-driven topological path planning. Experiments conducted on the CARLA Town07 simulator demonstrate high spatial coverage with less than 1 GB memory consumption and validate language-driven topological path planning against classical shortest-path algorithms like A*.

### Strengths
Combining brain-inspired cognitive mapping with aerial vision-and-language navigation is conceptually novel.

### Weaknesses
(1) The practical motivation is unclear. Although the brain-inspired concept is interesting, the aerial-view scenario typically allows access to GPS and classical global path-planning algorithms, raising doubts about the necessity of complex language-driven navigation. A more practical and clearly motivated problem setting is required.

(2) Experiments are limited to a small-scale simulated environment. Given numerous previous studies on aerial vision-and-language navigation like OpenUAV (ICLR25), AerialVLN (ICCV23) and CityNav (arXiv2406.14240), it remains unclear if the proposed approach generalizes or performs competitively on existing benchmarks. While the problem formulation is somewhat novel, experiments of similar scope and scale to previous studies are necessary for clearer validation.

(3) Despite the heavily heuristic design choices in the proposed algorithm (e.g., similarity-distance scoring, containment-based merging, density clustering), the paper lacks a detailed ablation study. Table 3 is described as an ablation but merely compares various LLMs rather than systematically evaluating the contributions of each proposed mapping component.

### Questions
(1) What is the practical motivation for using complex language-driven navigation in aerial-view scenarios, given the availability of GPS and traditional global path-planning algorithms?

(2) Can the proposed approach generalize or perform competitively on established benchmarks such as OpenUAV, AerialVLN, or CityNav?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a framework that enables UAVs to navigate large-scale environments using a cognitive topological map. The system builds a multi-level map from UAV observations, organizing the environment into road-bound regions and semantic objects, with a dynamic update mechanism for scalable construction. Multimodal language models align natural language instructions with the map, enabling language-guided navigation. Experiments in the CARLA Town07 simulator showed high spatial coverage (81.48%) and effective navigation.

### Strengths
1.Cognitive Topological Map Construction: The framework constructs a multi-level topological map using UAV observation data, organizing the environment into road-bounded regions and semantic objects.
2.Dynamic Map Update: A mechanism based on embedding similarity and spatial containment is used to ensure the efficient and scalable construction of the map.
3.Multimodal Large Language Models (MLLMs): These models help align natural language instructions with map vertices, enabling language-driven navigation planning.

### Weaknesses
1. The model involves multiple modules, including large language models. Will this result in insufficient UAV inference speed in real-world scenarios?
2. As the UAV gradually builds the cognitive map, could small errors in localization or map updates accumulate over time, leading to reduced accuracy of navigation paths in large-scale environments?
3. why was only CARLA Town07 used for the experiments?

### Questions
same as weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Topo-AeroVLN, a novel framework for aerial vision-language navigation inspired by cognitive mapping in biological systems. The authors propose a method to incrementally construct hierarchical cognitive topological maps from aerial observations, integrating geometric structure with semantic abstraction. The framework leverages Multimodal Large Language Models (MLLMs) to align natural language instructions with map vertices, enabling robust language-driven navigation. Experiments conducted in the CARLA simulator demonstrate strong spatial coverage and navigation performance, highlighting the potential of this approach for scalable and interpretable UAV navigation without reliance on detailed metric maps.

### Strengths
1.	The paper presents a novel method for constructing cognitive topological maps from aerial observations, effectively integrating geometric and semantic information. This approach offers a scalable and interpretable solution for UAV navigation in large-scale environments.
2.	The paper is well-organized with a clear logical flow. The authors conducted thorough ablation studies comparing different MLLMs, which helped identify the most effective model for their framework. This approach demonstrates rigorous experimental design and enhances the credibility of their findings.

### Weaknesses
1.Mismatch Between Title and Content: The title of the paper focuses on Vision-Language Navigation (VLN), but the proposed framework appears to be more aligned with a semantic-enriched topological mapping approach rather than the conventional VLN task setting. The paper consider navigation in high-altitude aerial environments, where data collection and topological map generation suggest that the drone operates at a height where obstacles are not encountered. Given the typical VLN task design, where the prompt only requires navigating from a start point to an end point without additional obstacles, the drone’s navigation path should ideally be a direct route from start to finish. The reliance on a topological map to guide the path seems unnecessary in this context, deviating from the standard VLN task assumptions.

2.Writing and Presentation: The authors are encouraged to revise the manuscript to enhance clarity and readability, particularly in the presentation of formulas and figures. For example:
- In Formula (4), the symbols for weights (sim and dis) and the function name (sim) appear at the same level, which may cause confusion for readers. The authors should consider clarifying these symbols or reformatting them, such as using subscripts for weights (e.g. ω_sim).
- In Figure 3(a-c), the two curves are not accompanied by a legend, and the text does not directly explain their meanings. Readers are left to infer the significance of these curves, which can increase the difficulty of understanding the paper. The authors should add a clear legend to the figure and provide a more explicit explanation in the text.

### Questions
See the weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents Topo-AeroVLN, a brain-inspired framework for aerial vision-and-language navigation (VLN). It builds a two-level cognitive topological map—road-bounded regions (high-level nodes) and clustered semantic objects (sub-level nodes)—and updates it via embedding similarity + polygon containment. Each region is described by an MLLM-generated caption and aligned to language queries for topological navigation. Experiments in CARLA Town07 show >80% coverage and competitive retrieval and path-planning accuracy.

### Strengths
- Addresses an underexplored high-altitude aerial VLN scenario with sparse semantics.
- Proposes a region-based topological representation that supports scalable and interpretable navigation.
- The set-theoretic merging rule and language grounding via MLLMs are conceptually neat and well-integrated.
- Experiments include several MLLMs for grounding and planning, offering practical insights.

### Weaknesses
- No direct comparison with existing aerial VLN baselines (AerialVLN, CityNavAgent, See-Point-Fly).
- Evaluation assumes perfect segmentation and stitching, lacking robustness analysis for real UAV data.
- Ablations only test different LLMs, not the mapping components (e.g., merging, sub-level hierarchy).

### Questions
- Can the authors benchmark against existing aerial VLN methods with SR/SPL metrics?
- How robust is the mapping to segmentation errors or partial occlusion?
- Are there plans to validate on real UAV data or release the dataset?

### Soundness
3

### Presentation
3

### Contribution
3
