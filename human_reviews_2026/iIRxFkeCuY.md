# PAT3D: Physics-Augmented Text-to-3D Scene Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
We introduce PAT3D, the first physics-augmented text-to-3D scene generation framework that integrates vision–language models with physics-based simulation to produce physically plausible, simulation-ready, and intersection-free 3D scenes. Given a text prompt, PAT3D generates 3D objects, infers their spatial relations, and organizes them into a hierarchical scene tree, which is then converted into initial simulation conditions. A rigid-body simulator ensures realistic object interactions under gravity, driving the scene toward static equilibrium without interpenetrations. To further enhance scene quality, we introduce a simulation-in-the-loop optimization procedure that guarantees physical stability and non-intersection, while improving semantic consistency with the input prompt. Experiments demonstrate that PAT3D substantially outperforms prior approaches in physical plausibility, semantic accuracy, and visual quality. Beyond high-quality generation, PAT3D uniquely enables simulation-ready 3D scenes for downstream tasks such as scene editing and robotic manipulation. Our code and data are available at https://github.com/Simulation-Intelligence/PAT3D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PAT3D, a physics-augmented framework for text-to-3D scene generation that addresses the critical gap of physical plausibility in multi-object scene synthesis. The technical approach integrates vision-language models with differentiable rigid body simulation to produce scenes that are both semantically consistent and physically stable. The method decomposes the generation pipeline into three stages: object and spatial relation extraction via VLMs, physics-aware layout initialization using hierarchical scene trees, and simulation-in-the-loop optimization. Experiments demonstrate improvements over existing methods in physical stability metrics and semantic consistency.

### Strengths
- This paper focuses on a meaningful and underexplored problem. The lack of physical plausibility in text-to-3D scene generation is important for downstream applications in robotics and simulation.
- Demonstrates practical applications in scene editing and robotic manipulation tasks.
- Clear experimental validation showing zero interpenetration and stable equilibrium states.

### Weaknesses
- Limited video demonstrations make it difficult to fully assess the dynamic behavior and stability of generated scenes under various conditions. The supplementary materials would benefit from more extensive video results. Currently there are only 2 examples.
- Although the paper claims the first work on physics-plausible scene generation, I'm not sure how different the problem setting is from CAST. From the demo examples of CAST, they also have physics simulation to make sure objects are placed in a stable and non-penetrating way. How is the proposed work different from it?

### Questions
Generally I think this is an important problem and the results look interesting. But even in the two examples (which are very limited), the simulation looks a bit unrealistic. So I'm not sure if the technical approach is really addressing the challenges stated in the paper.

Also, I'm not fully convinced why the work is different from CAST.

### Soundness
2

### Presentation
3

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
The paper introduces PAT3D, a framework for generating 3D scenes from textual prompts, with emphasis on physical plausibility and simulation-readiness. 
This work uses a 3D Objects and Spatial Relation Extraction module by generating reference images, a Layout Initialization module with the help of point clouds, scenetree, and a Layout Optimization module with self-defined physical constraints to generate simulatable and physically plausible 3D scenes.
In short: PAT3D bridges text→3D scene generation with physics simulation, aiming not just for plausible appearance but also for physically stable, actionable 3D scenes.

### Strengths
- The visualization results are obviously better than previous works.
- The pipeline is intuitive and direct. With the help of depth and point cloud, this work is able to obtain precise 3D scene layouts and spatial relationships.
- Applying simulations in the process of 3D scene generation is an interesting method, which alleviates a lot of trouble in the nuisance 3D layout optimization process.

### Weaknesses
- The novelty of this work is relatively weak. The '3D OBJECT AND SPATIAL RELATION EXTRACTION' part also appears in [1], and the novel parts are the simulation instead of 3D scene optimization, which is a little bit weak.
- The use of rigid-body simulation and “simulation in the loop” likely requires non-trivial compute and careful tuning. The reproducibility, speed, and resource requirements might limit broader adoption.
- Too few quantitative results, since the paper features the simulation ability of this work, I guess some more metrics that can reflect the physical plausibility and the advantage for manipulations are needed.

[1]: Layout-your-3D: Controllable and Precise 3D Generation with 2D Blueprint

### Questions
- What is the success rate of this work? Will the simulation finally converge to a plausible point with multiple iterations? Or it is largely dependent on the initialization?
- Are there more failure cases? This failure case is not that typical, and I guess it can not reflect the major limitation of this work.
- How much time does it take to generate a single 3D scene with the simulation-in-the-loop? What is your edge over [1]?

[1]: Embodiedgen: Towards a generative 3d world engine for embodied intelligence

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
1

### Summary
PAT3D is a physics-augmented text-to-3D scene generation framework that overcomes common issues in geometry-only layouts (floating, unstable stacks, wrong supports) by integrating differentiable rigid-body contact simulation into the pipeline. From a text prompt, it synthesizes a reference image, generates segmented object meshes, and uses a VLM to infer inter-object physical dependencies, forming a hierarchical scene tree. It then constructs an intersection-free, physics-aware initialization by inserting small gravity-aligned gaps for support pairs, and lets objects settle via simulation before optimizing the layout with differentiable, artificially time-stepped dynamics for semantic and physical consistency. Experiments on contact-rich scenes show state-of-the-art visual quality, semantic alignment, and physical plausibility, with scenes remaining editable and directly usable for simulation-based robotics evaluation.

### Strengths
- More realistic physics: Differentiable rigid-body contact simulation reduces floating, interpenetration, and unstable stacks.
- Better semantic–layout alignment: A VLM infers object dependencies to build a scene tree that preserves text-described spatial relations.
- Stable, collision-free initialization: Physics-aware initialization with small gravity-aligned gaps improves convergence and avoids intersections.
- Optimizable and editable scenes: Layout is refined via differentiable simulation; outputs remain easy to edit and interact with.

### Weaknesses
- Scalability to very large scenes: Many-object, dense-contact scenes increase solver complexity (contact graph size), potentially impacting stability and runtime.
- Lack of quantitative study in ablation:   Measurable gains introduced by each component are necessary, making audience further understand the contributions.

### Questions
Please check Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3
