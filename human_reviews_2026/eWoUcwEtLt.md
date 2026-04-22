# VisionLaw: Inferring Interpretable Intrinsic Dynamics from Visual Observations via Bilevel Optimization

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 8

## Abstract
The intrinsic dynamics of an object governs its physical behavior in the real world, playing a critical role in enabling physically plausible interactive simulation with 3D assets. Existing methods have attempted to infer the intrinsic dynamics of objects from visual observations, but generally face two major challenges: one line of work relies on manually defined constitutive priors, making it difficult to align with actual intrinsic dynamics; the other models intrinsic dynamics using neural networks, resulting in limited interpretability and poor generalization. To address these challenges, we propose VisionLaw, a bilevel optimization framework that infers interpretable expressions of intrinsic dynamics from visual observations. At the upper level, we introduce an LLMs-driven decoupled constitutive evolution strategy, where LLMs are prompted to act as physics experts to generate and revise constitutive laws, with a built-in decoupling mechanism that substantially reduces the search complexity of LLMs. At the lower level, we introduce a vision-guided constitutive evaluation mechanism, which utilizes visual simulation to evaluate the consistency between the generated constitutive law and the underlying intrinsic dynamics, thereby guiding the upper-level evolution. Experiments on both synthetic and real-world datasets demonstrate that VisionLaw can effectively infer interpretable intrinsic dynamics from visual observations. It significantly outperforms existing state-of-the-art methods and exhibits strong generalization for interactive simulation in novel scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes VisionLaw, a bilevel framework that uses LLM-driven search to generate/edit symbolic elastic and plastic constitutive laws (upper level) and a differentiable MPM+renderer loop to fit continuous material parameters from video supervision (lower level). The experiments report the best average Chamfer distance on synthetic data, and improved visual fidelity/generalization vs baselines.

### Strengths
1. The idea of combining LLM-based hypothesis search with differentiable, vision-guided evaluation to discover interpretable constitutive laws directly from videos is very compelling.
2. Implementation details are transparent and the source code is provided, largely promoting reproducibility.
3. The paper is clearly written and easy to understand.
4. The comprehensive experimental results are solid, proving the effectiveness very well.

### Weaknesses
1. While the decoupled strategy is motivated, the paper could quantify search efficiency (e.g., wall-clock, simulator calls, accepted offspring) and compare against naive joint search under equal budgets.
2. The pipeline depends on accurate camera parameters and a 3DGS reconstruction from the first frame. Its sensitivity to calibration errors, occlusions, and reconstruction artifacts is not analyzed.

### Questions
Do the learned laws transfer across different objects with similar physical properties in real scenes without re-search?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper propose a framework for evolving symbolic constitutive models and optimizing material parameters from multi-view videos. The task setting follows PAC-NeRF and use PhysGaussian-based simulator for better reconstruction quality. The main contribution compared to PAC-NeRF is its LLM-based constitutive law evolution framework: the non-differentiable discrete constitutive class optimization is done by LLM code generation based on feedbacks from the lower-level continuous physical parameter optimization. The lower-level optimization uses differentiable simulation and differentiable rendering, similar to PAC-NeRF.

### Strengths
- LLM-based code generation is a promising way to handle discrete constitutive models, as it can reduce the need for mechanics experts to manually set up these models.

### Weaknesses
- Since the upper-level optimization is the main contribution, a more thorough examination is needed:
    - How many upper-level evolutions are need to converge to the correct one?
    - How are different elasticity models distinguished by LLM? Since many models behave similarly at small deformation magnitudes. I don't think LLM can differentiate StVK, neo-Hookean, fixed-corotated when the object is pure elastic.
    - What if the LLM outputs python code with bugs that can crush the program?
    - Why does direct joint evolution not perform well? It appears that the difference lies primarily in prompting and the available information and feedbacks are the same. Ablation studies are needed to motivate decoupled evolution.


- Although the high-level ideas are clear, some details are lacking for reproducibility:
    - What (modalities) are included in the feedback from the lower-level optimization?
    - How is the first batch of constitutive model candidates are initialized? Are they generated by LLM based solely on multiview videos?
    - How are the initial physical parameters set when the predicted constitutive models change?

### Questions
- The plastic corrections are derived from plastic flows applied on a specific elastic model. That is, even with the same plastic flow, different elastic model can result in different discrete deformation corrections (called plastic return mappings). For example, the von-Mises plasticity and Drucker-Prager plasticity used in the paper assume stVK elasticity model. The combination of fixed-corotated elasticity and von-Mises plasticity in H.1 BOUNCYBALL is fundamentally wrong.

- How many lower optimizations are expected to run? It seems that each upper-level optimization are followed with multiple independent lower optimizations. Considering the above combination constraints, there are not many available constraints, for example, fewer than 10. I think it is doable to run brute-force lower-level optimizations to find the most fit, i.e., PAC-NeRF + brute-force sweep. This could be a strong baseline to compare running time.

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
4

### Summary
This work, titled VisionLaw, proposes a bilevel framework for the visual grounding of physical laws. At the upper level, the Vision-Language Model (VLM) generates a hypothesis about the physical materials. At the lower level, a differentiable simulation-and-rendering pipeline verifies this hypothesis. The main contribution claimed is the bilevel framework for symbolic estimation. The authors evaluate VisionLaw on real and synthetic datasets.

### Strengths
- The presentation is straightforward and easy to understand. 

- The discussed topic—inferring interpretable intrinsic physics—is fascinating.

- The experiments conducted are thorough and comprehensive.

### Weaknesses
Please take the time to read the following two papers:  
[1] "LLM and Simulation as Bilevel Optimizers: A New Paradigm to Advance Physical Scientific Discovery."  
[2] "Neuma: Neural Material Adaptor for Visual Grounding of Intrinsic Dynamics."  

In comparison to these papers, one concern I have is that the contribution of the current work seems weak. The bilevel framework closely resembles that of Paper [1] (which also uses a bilevel framework to search optimal material parameters), and the vision-guided constitutive evaluation mechanism appears to be based on concepts from Paper [2].

### Questions
I believe the author should clarify the difference between [1,2] and their work. If this concern is addressed, I will consider changing my score.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a bilevel optimization framework for estimating the physics of a scene. This is done with an LLm at the high level iteratively refining the physical formulas governing the particles in the scene (through an evolutionary algorith) and a low-level differentiable particle-based simulator refining the individual particle properties.

### Strengths
- **S.1:** Great idea. I think this is a really beautiful and elegant idea. Having the physics defined in the outer loop and the parameters tuned in the inner loop is great.
- **S.2:** Clear Results. As far as I can tell, the results look pretty impressive and the paper compares its method to a variety of contemporary baselines, which is great.
- **S.3:** Reproducibility. I appreciate that the authors released their source code, additional videos, and all LLM prompts.

### Weaknesses
- **W.1:** The writing is incredibly dense. I've worked in parameter identification for simulator tuning and I've written my own physics engines and I was barely able to follow all of this. A bit simpler writing would greatly benefit this paper. For example, it took me a while to understand why you're not just using the gradient that you get from the differentiable simulator rollout to also tune the high-level physics.

### Questions
- **Q.1:** More of a comment: Your contributions are weirdly written. I think contibution 1 is actually made up of of contibutions 2 and 3. So that should be one bullet point. And running experiments that validate your method is not a contribution, so contribution 4 should be removed.
- **Q.2:** Another suggestion: since MPM simulation is so important to your method, maybe spend a short paragraph explaining it to everyone in the main body of the paper.

### Soundness
4

### Presentation
2

### Contribution
4
