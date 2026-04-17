# HGWM: Hierarchical Graph-guided World Model for Zero-shot Object Navigation via Scene-Goal Graph Matching

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Object Goal Navigation, which requires an agent to locate specific objects in unknown indoor environments, remains a fundamental challenge in embodied AI that demands sophisticated spatial-semantic understanding. Although recent Vision-Language Model (VLM) based approaches have shown promise through effective perception and reasoning capabilities, current methods lack systematic world model architectures that can predict environmental states and reduce exploration inefficiency. We introduce HGWM (Hierarchical Graph-guided World Model), a novel navigation framework that integrates dual-graph matching with a unified Spatial-Semantic World Model to enable robust object localization. HGWM constructs complementary graph representations: a goal subgraph encoding LLM-derived spatial knowledge about target objects and room hierarchies, and dynamically maintained scene graphs derived from our persistent Spatial-Semantic World Model. These graphs interact through a dual-matching mechanism that combines implicit VLM-guided semantic alignment with explicit structural correspondence. Our multi-stage exploration strategy adapts dynamically based on the degree of graph matching, transitioning from systematic exploration to focused search and finally target verification. Experiments on HM3D v0.1, v0.2, and MP3D benchmarks demonstrate HGWM's effectiveness, achieving state-of-the-art performance with 59.6% success rate and 31.5% SPL on HM3D v0.1, and 45.8% success rate and 17.3% SPL on MP3D, outperforming previous methods by up to +1.5% SR and +0.3% SPL. Our code will be made public soon.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposed HGWM, a method for object goal navigation. HGWM simultaneously constructs a goal subgraph and dynamic scene graph. HGWM introduces a novel dual-graph matching mechanism. HGWM is evaluated on standard benchmark, HM3D and MP3D, and shows SOTA results.

### Strengths
- The paper improves on existing methods, notably WMNav, and introduces a new dual-graph matching mechanism. 
- The proposed method is evaluated comprehensively on appropriate benchmarks and baselines.

### Weaknesses
- The method innovation is incremental and the performance gain is pretty small.

### Questions
Suggestion: Add experiments with newer model to show method innovation is not subsumed by stronger base model

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the Hierarchical Graph-guided World Model (HGWM) for zero-shot object-goal navigation. The method builds on a unified spatial-semantic world model: an LLM constructs a goal subgraph at the start, and RGB-D plus pose are used to maintain a dynamic scene graph during navigation. For decision-making, the paper introduces dual-graph matching—a VLM provides goal-aware panoramic semantic scores as the implicit cue, while explicit structural matching is performed between direction-specific local scene graphs and the goal subgraph. An LLM then fuses these signals to drive a three-stage exploration strategy and a hierarchical navigation policy. The approach reports modest improvements on HM3D v0.1 and MP3D.

### Strengths
- The figures are well-designed and clearly illustrate the overall framework and key components.
- The paper is generally well-written, with clear organization and straightforward explanations that make the technical details accessible even to readers who are not deeply familiar with the specific subfield.
- The spatial–semantic world model accumulates scene knowledge by anchoring objects in 3D and maintaining relational edges, so subsequent decisions are based on shared memory rather than single-step observations.

### Weaknesses
- The main weakness of this paper lies in its limited novelty. The use of graph-based representations for modeling spatial and semantic structures in ObjectNav has already been extensively explored in prior works such as Hierarchical Object-to-Zone Graph for Object Navigation (HOZ, ICCV 2021), SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation (NeurIPS 2024), and UniGoal: Towards Universal Zero-shot Goal-oriented Navigation (CVPR 2025). Furthermore, the proposed dual-graph matching mechanism is conceptually similar to UniGoal’s graph alignment strategy, which also matches scene and goal structures at the graph level. While HGWM integrates these elements within a world-modeling framework, the methodological differences appear incremental, and the overall contribution does not clearly establish a fundamentally new insight beyond existing graph-based navigation approaches.
- The method stacks multiple modules—implicit semantics, explicit matching, and LLM fusion—leading to a long inference pipeline and nontrivial overhead, while the SR and SPL gains over strong baselines are relatively small.
- The goal subgraph is a static prior and relies on LLM generation; in atypical layouts it may bias the search, and the scene-graph “correction” step is guided by the same prior, which may treat rare but valid structures as errors and thus amplify the bias.
- The scalability and maintenance cost of the spatial–semantic world model are unclear; as exploration increases, the storage and computation required for updates are not quantified or controlled, making it hard to assess long-horizon feasibility.

### Questions
- How do you avoid double-counting or mutual interference when fusing the implicit semantic cues and the explicit matching scores?
- How is the determinism of goal-subgraph generation ensured? For the same input, does it produce an isomorphic graph, and how is randomness controlled?

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
2

### Summary
This work proposes HGWM—a Hierarchical Graph-Guided World Model for zero-shot ObjectNav. The agent maintains two complementary graphs: (i) an LLM-derived goal subgraph (room hierarchy, anchors, connectivity) and (ii) a dynamically updated scene graph grounded in a unified spatial-semantic world model; decisions come from dual graph matching (implicit VLM alignment + explicit structural matching) and an adaptive three-stage exploration policy. Experiments show small but consistent gains over WMNav.

### Strengths
- **Unified representation:** Both goal and observations are cast as graphs tied to a persistent spatial-semantic memory (objects projected to world coords and keyed; scene graphs built per direction). This reduces information loss vs. pure embeddings.
- **Dual matching:** Combines implicit VLM scoring with explicit multi-dimensional structural matching (with position-aware enhancement).

### Weaknesses
- The method extends WMNav with a more structured, graph-centric integration; the conceptual distance may feel narrow without sharper head-to-head analyses isolating where dual matching buys the gains.
- Reported improvements over WMNav are relatively small; ablations show most lift comes from adding any memory (voxel map), with hierarchical graph + dual matching adding only a small amount of improvement. I think a stronger justification of cost/benefit is needed.
- The explicit-matching formula introduces weights and direction/relevance terms w_i, D(α), R(α), without reporting how they’re chosen or tuned, or their sensitivity.
- Even though the dynamic scene graph with a semantic world model is the main component that the authors highlight, there is no controlled ablation or analysis that isolates the “dynamic” part of the scene graph.

### Questions
1. It would be better to include 1–2 rollout pages with synchronized frames and graph visualizations (nodes/edges/room labels) so readers can see how the graph changes and why the policy turned a certain way.
2. Could you quantify loop reduction/coverage efficiency (e.g., re-visit ratio, coverage %, path overhead) beyond the qualitative Figure 3? 
3. The ablation introduces a “Voxel Map” baseline and reports improvement over “Basic VLM Nav”, but the paper does not specify its design. Could you explain some details and how this voxel memory differs from your world model?

### Soundness
2

### Presentation
2

### Contribution
2
