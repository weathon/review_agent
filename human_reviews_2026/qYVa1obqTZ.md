# Neural Modular Physics for Elastic Simulation

- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
Learning-based methods have made significant progress in physics simulation, typically approximating dynamics with a monolithic end-to-end optimized neural network. Although these models offer an effective way to simulation, they may lose essential features compared to traditional numerical simulators, such as physical interpretability and reliability. Drawing inspiration from classical simulators that operate in a modular fashion, this paper presents Neural Modular Physics (NMP) for elastic simulation, which combines the approximation capacity of neural networks with the physical reliability of traditional simulators. Beyond the previous monolithic learning paradigm, NMP enables direct supervision of intermediate quantities and physical constraints by decomposing elastic dynamics into physically meaningful neural modules connected through intermediate physical quantities. With a specialized architecture and training strategy, our method transforms the numerical computation flow into a modular neural simulator, achieving improved physical consistency and generalizability. Experimentally, NMP demonstrates superior generalization to unseen initial conditions and resolutions, stable long-horizon simulation, better preservation of physical properties compared to other neural simulators, and greater feasibility in scenarios with unknown underlying dynamics than traditional simulators.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper Neural Modular Physics for Elastic Simulation , introduces a fully modular neural simulator for elastic dynamics. This represents a new direction relative to the monolithic, end-to-end neural simulation paradigms (NO, GNN) and hybrid simulators that replace partial components demonstrated through a complete modular neural architecture that mirrors traditional finite element method (FEM) computation flows. The approach enables direct supervision of intermediate physical quantities and interchangeable traditional-numerical and neural components, which is a novel result in the neural simulation literature. The contribution of this paper lies more in systematic integration and training methodology (the two-stage modular physics training) than in the individual components.
Given the surge of interest in physics-informed ML, differentiable simulation, and scientific machine learning. The paper addresses a key pain point of neural simulators—lack of interpretability and physical soundness—making it relevant to computational physics and graphics communities as well. The papers focus on elastic simulation is narrower than universal PDE solvers, but serves as an excellent proof-of-concept domain.

### Strengths
1. Concept: 
The paper provides a fairly new approach to learning-based, physically interpretable simulation approach that modularizes elastic dynamics into neural subcomponents aligned with traditional numerical solvers. The idea of decomposing physics simulation into learnable yet interpretable modules, supervised via intermediate physical quantities, represents a meaningful conceptual and architectural advance.
While the idea of combining neural networks with modular solvers has appeared before the innovation lies more in systematic integration and training methodology (the two-stage modular physics training) than in the individual components.
2. Practicality/Application:  
When combining neural networks with physics-based modular solvers, one of the hardest problems is ensuring boundary condition consistency between modules that are partly data-driven and partly physically constrained. In hybrid or modular networks, each subnetwork (e.g., constitutive law, integrator) may implicitly assume slightly different boundary behavior, leading to inconsistencies at their interfaces — this is known as a boundary-condition coupling error.  The NMP framework proposes to address this through architectural and Training level strategies.  The Neural Integration Module preserves the same interface as an implicit FEM integrator but replaces the internal update terms with a neural network. After the neural update, boundary condition enforcement and collision handling are explicitly re-applied.  
Hence, NMP preserves:
Dirichlet BCs 
Neumann BCs
Contact constraints
This design ensures neural predictions remain compatible with traditional BC enforcement — solving the “boundary leakage” problem found in prior hybrids
3. Results: Published results show improved performance over top baselines as well as improved stability (reduced collapse) 
4. Generalizability: is demonstrated against unseen initial conditions and higher mesh resolutions. 
5. Evidence: The SPOT and BOB experiments are specifically designed to test BC handling
SPOT: multiple fixed boundary points (head and tail)
BOB: fixed vertices (head) and free elastic regions

### Weaknesses
1. Practicality/Application:  solution process and validation
Computational cost and scalability are not discussed (e.g., time vs. FEM or other neural simulators). This is a critical metric to be evaluated.
Lack of error analysis: No discussion on failure modes or interpretability visualization for intermediate variables.  This is also critical for real world applications on elastic body problems. 
2. Results: are limited to non real world elastici body  FEM problems  
3. Generalizability: is limited to elastic solids only. Extensions to fluids or multi-physics are not shown and this gap is acknowledged by the authors.  
4. Evidence and support: the following weaknesses exist:
No ablations on module architecture: How sensitive is performance to neural constitutive/integration model complexity?
Limited physics validation metrics: Energy conservation, stress-strain correlation, or physical invariants could further substantiate realism.

Writing clarity:
line 121 -"dynamics on top of analytical dynamics to account for unmodeled" ... is an incomplete sentence
line 135: "In this paper, we notice the internal modularity". ... please fix
line 141: "As aforementioned," --> as mentioned earlier ?
line 160 : "derives two disentangled modules" --> two decoupled ?
line 182: "with vertice position" --> vertex position ?
line 329: Transolver (?). --> what is the "? " provide citation, Wu ?

### Questions
Please address the following practical issues 
1. Discuss time to converge relative to traditional solvers on a real world engineering problem if possible
2. discuss error analysis and boundary condition matching
 3. provide more evidence on physics validation metrics: Energy conservation, stress-strain correlation, or physical invariants could further substantiate realism.

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
The paper proposes a hybrid-physics architecture and training method that achieves better physical consistency and generalizability than traditional simulators and neural models for elastic physics rollout. The modular design for elastic physics is inspired by the FEM method.

### Strengths
- The experiments provide good empirical evidence that their NMF model has better long term rollout performance for both seen and unseen scenarios in comparison to the baselines.
- Their NMF model allows the addition of soft physics constraints in a fairly straightforward way.
- The paper provides a thorough explanation of their modular model design and reasoning.

### Weaknesses
- The architecture is specialized for elastic physics, which will not generalize to other equations and setttings. However, I acknowledge the idea of replacing components of a simulator with neural modules is an interesting idea.
- The paper proposes a way to replace parts of an FEM method with neural networks, but this is not straightforward to apply to other types of solver schemes such as finite difference methods and spectral methods.
- The strain-displacement matrix is required a priori to use the NMF model.

### Questions
- Do you plan to release the code for your model training and simulations?
- What are the costs in terms of flops and time of the NMF model inference vs the simulator used to generate ground truth trajectories?
- Since the neural constitutive model requires the $B_e$ matrix and $V_e$ values to work, do you provide these in your test scenarios to the model? Is the $B_e$ term easy to derive in some way or should it also be learned?
- How do you determine when to stop the separate training phase for each module? Do you train for a fixed number of steps or use some other condition for stopping?
- You use a regularization constant of 0.1 for the volume loss term you add during joint training. How did you select this value? What happens if you increase it to make the physical constraint stronger? Does the overall trajectory become more physically plausible or does your model fail to learn due to the difficulty of satisfying this constraint?
- There are few typos I noticed in the paper that can be corrected:
  - Line 121: "Examples include learned data-driven discretization stencils (Bar-Sinai et al., 2019) and learned residual
dynamics on top of analytical dynamics to account for unmodeled (Yin et al., 2021)." Maybe you want to write unmodeled dynamics?
  - Line 211: "Still start from classical FEM." Rewrite this to be a more clear introductory sentence.
  - Line 424: "With out the elaborative physical-aligned architecture and specialized training strategy in NMP, we cannot release the unique benefits of modularization." I think you mean to write "we cannot realize" instead of "we cannot release".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper propose an neural simulator framework for elastic simulation. The framework follows closely the traditional FEM simulation pipeline, with two modules replaced by neural networks: per-element Piola–Kirchhoff stress tensors computation and per-vertex velocity increment computation.

### Strengths
- The overall presentation is clear.

### Weaknesses
- The framework seems just a replication of some groundtruth FEM simulation method. In separate trainings for each module, it seems that the groundtruth constitutive model law and the groundtruth time integration are both available. The two groundtruth pieces are basically all you need to implement the whole simulation in the traditional way. I am concerned about the motivation of the method: if you need to implement the whole traditional pipeline to get groundtruth, why bother replacing modules with neural networks? Are there any applications beyond replication?

- The framework seems not applicable on real data. The assumption of known per-vertex internal forces is not practical in reality.

- The framework seems to only work for interactions between single objects and a fixed environment. 
    - The contact information, especially the offset of the ground plane, is stored in time integration network. Updated ground plane offset may not work. 
    - And I don't think the neural time integration can work for multiple objects. The neural modeling of time integration is not aware of other objects.

### Questions
- How to make sure when F is a rigid transformation, the stress is zero? A unit test may need to make sure the simulation can maintain its rest shape forever if there is no external force.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to use data-driven methods to replace the traditional solvers in the elastic simulation. The experiments show the effectiveness of the proposed method, leading to low errors and stable long-horizon stability. The experiments also show the proposed method is better for unseen conditions or inputs. Another contribution is that since the proposed method supervises the intermediate variables, it means the proposed method can use physics knowledge as contraints.

### Strengths
1. This paper is easy to follow and well organized.
2. This paper evaluates their method from multiple aspects: 1. Comparison with neural operator based baselines. 2.  Joint training versus separate training. 3. Comparison with physics simulator. 4. Inference by combining with traditional simulators. 5. Comparison with no direct physical constraint included.

### Weaknesses
1. The biggest weakness is that the proposed method is just substituting the immediate two steps that are done by traditional methods with data-driven methods. Thus, novelty is the biggest issue to me. Besides, the neural constitutive step is following the existing paper.
2. Another contribution claimed by the authors is the separate training plus joint training. From what I understand, this training method was also proposed in the existing work to solve the “collapse” issue, which is also not new.
3. For the experiments, the authors compare it with purely data-driven methods. However, their method uses physics. I don’t think the experiments are good enough for showing the advantages of their methods. For example, they should compare with the traditional methods and we can see the gap when physics is used for both methods.
4. It will be good to see the advantages/disadvantages of the methods including baselines in terms of inference time.
5. The method is only tested on the specific problem, which means the scope is limited.

### Questions
1. For strengthening the experiments, do the authors think the baselines can augment with physics knowledge?
2. What is the error for predicting just one step forward for each method?
3. For baselines, have you tried non-autoregressive prediction, e.g., directly predicting all the following 100 steps.

### Soundness
3

### Presentation
3

### Contribution
2
