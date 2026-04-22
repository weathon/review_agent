# Rapid Training of Hamiltonian Graph Networks Using Random Features

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 2

## Abstract
Learning dynamical systems that respect physical symmetries and constraints remains a fundamental challenge in data-driven modeling. Integrating physical laws with graph neural networks facilitates principled modeling of complex N-body dynamics and yields accurate and permutation-invariant models. However, training graph neural networks with iterative, gradient-descent-based optimization algorithms (e.g., Adam, RMSProp, LBFGS) often leads to slow training, especially for large, complex systems. In comparison to 15 different optimizers, we demonstrate that Hamiltonian Graph Networks (HGN) can be trained 150-600× faster - but with comparable accuracy - by replacing iterative optimization with random feature-based parameter construction. We show robust performance in diverse simulations, including N-body mass-spring and molecular dynamics systems in up to $3$ dimensions and 10,000 particles with different geometries, while retaining essential physical invariances with respect to permutation, rotation, and translation. Our proposed approach is benchmarked using a NeurIPS 2022 Datasets and Benchmarks Track publication to further demonstrate its versatility. We reveal that even when trained on minimal 8-node systems, the model can generalize in a zero-shot manner to systems as large as 4096 nodes without retraining. Our work challenges the dominance of iterative gradient-descent-based optimization algorithms for training neural network models for physical systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a novel iterative method to train Hamiltonian Graph Neural Networks (GNNs) orders of magnitude faster than standard optimizers. The approach leverages random feature sampling over the small shared node, edge, and message MLPs of the GNN, followed by a least-squares fit to determine the final linear layer that outputs the Hamiltonian. To ensure translation and rotation invariance, the particle positions are processed by aligning the simulation frame of reference to the center of mass and rotating the coordinates to a fixed orthonormal basis. The proposed training algorithm is evaluated on four different graph configurations and compared against multiple standard optimizers, achieving at least two orders of magnitude speedup (even outperforming the second-order L-BFGS optimizer). Furthermore, the method is benchmarked against seven state-of-the-art structure-preserving architectures using publicly available datasets.

### Strengths
* The method is tested over Neurips 2022 open source benchmarks.
* The paper proposes an interesting way to enforce rotation-invariances in the system.
* The experiments are varied, exploring N-body systems (chains, regular grids) and molecular interactions in Lennard-Jones systems
* The generalization and rollout results are good. These are the precise experiments needed to show the power of structure-preservation in learning conservative dynamics.

### Weaknesses
* Even though the method is tested over a wide variety of systems, they are still small scale toy problems.

### Questions
* Section 4.3: Is there a particular reason why coosing Hamiltonian GNNs instead of, say, Lagrangian GNNs? Have the authors tried to train the tested architectures (GNODE, LGNN, FGNN, etc) with the presented method to see which architecture performs better?
* Have the authors tried to use simpler invariance tricks, like using relative distances or relative angles?
* Line 1079: $ 5^{-3}$ might refer to $5\cdot 10^{-3}$?
* Line 1079: Here is specified that the molecular dynamics experiments are only rolled-out to 50 timesteps. However, Table 2 shows T=99999 timesteps. Which is the correct one? 50 timesteps is very few time horizon for a molecular system, given that the timestep is very small for stability.
* Equation 6: Is the least squares minimization well-posed? Have the authors found any problems when training? I'm thinking about a bad conditioning number for the normal equations $Z^TZ$, or some degeneracies induced by the rotation-translation invariance.

### Soundness
3

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
3

### Summary
The paper introduces Random-Feature Hamiltonian Graph Networks, whose hidden layers are fixed random features, with only the final linear readout solved in one least-squares step instead of iterative gradient descent. The model encodes translation/rotation/permutation invariances, reports 100-600x training time speedups over many optimizers on mass-spring and Lennard-Jones systems, and shows zero-shot scaling from tiny to very large graphs without retraining.

### Strengths
1. Replacing long iterative GD with a single convex least-squares solve is simple and yields substantial speedups without complicated tuning.
2. The model encodes translation, rotation, and permutation invariances appropriate for N-body dynamics, which improves data efficiency and generalization within a family.
3. Demonstrates training on small systems and inference on much larger ones without retraining.

### Weaknesses
1. The core idea is somehow similar to reservoir computing: both avoid training the feature generation module and optimize only the final readout layer.
2. With a single message passing stage and a linear head, it’s unclear how the method scales to long range or multi scale interactions. There is no ablation on stacking multiple RF blocks.
3. The training formulation assumes no external forces and exact energy conservation. Robustness to mild non-conservation (damping, stochasticity) is only lightly tested.
4. Transfer across distinct topologies is under-explored.

### Questions
1. What are the key differences between your approach and reservoir computing? 
2. What happens with 2-3 random feature message passing blocks versus a single block? How does performance scale with feature width, and do you observe conditioning issues in the least-squares solve as capacity grows?

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
This paper introduces RF-HGN, a method for training Hamiltonian Graph Networks using random feature sampling instead of iterative gradient-descent optimization. The authors demonstrate significant training speedups compared to other optimizers (Adam, LBFGS, etc.) while maintaining competitive accuracy on mass-spring and molecular dynamics systems.

### Strengths
The proposed method offers a significant speedup without sacrificing accuracy in multiple systems. 

- comprehensive benchmarking
- elegant solution by merging  Hamiltonian Graph Networks (for physics-informed modeling), Random Features (for fast, non-iterative training), and careful construction of physical invariances (translation, rotation, permutation). 
- generalizability from small to larger systems

### Weaknesses
- the paper could give more background on why random features work
- acknowledge the limitation more prominently and discuss it
- the acceleration is significant, but the claimed 600× speed-up is overstated; compared to the second-best model, it is actually 150×.

### Questions
- The method is presented in the context of Hamiltonian systems. How readily can it be applied to other physics-informed graph networks, such as Lagrangian or Port-Hamiltonian networks, or even non-conservative systems? Does the "random features + linear solve" recipe generalize? 

-  The method demonstrates impressive zero-shot generalization to larger systems, but does this generalization hold for structurally heterogeneous systems? For instance, if a model is trained on a regular lattice (where all nodes have the same degree), can it accurately predict the dynamics for a system with a mix of node degrees, or for a node with an unexpectedly high degree not seen during training?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes RF-HGN, a method to accelerate training of Hamiltonian Graph Networks by replacing gradient-based optimization with random feature sampling and least-squares solvers. The approach samples dense layer parameters randomly (using ELM or SWIM) and optimizes only the final linear layer, achieving 100-600× speedups while maintaining comparable accuracy. The method demonstrates zero-shot generalization from small training systems (8 nodes) to large test systems (4096 nodes) across mass-spring, lattice, and molecular dynamics systems.

### Strengths
1. Dramatic Practical Speedups: The 100-600× training acceleration is genuinely impressive and could enable new applications in physics
  simulation.
2. Strong Zero-Shot Generalization: Training on 8-node systems and testing on 4096-node systems demonstrates remarkable scalability - this is perhaps the paper's most valuable contribution.
3. Comprehensive Experimental Validation:
    - Comparison against 15 different optimizers provides robust baselines
    - Multiple physical systems (springs, lattices, molecular dynamics)
    - Use of established NeurIPS 2022 benchmark dataset
4. Physical Consistency: The method maintains energy conservation and incorporates essential symmetries (translation, rotation, permutation invariance).
5. Clear Algorithmic Contribution: The two-stage training procedure (random sampling + least squares) is well-defined and reproducible.

### Weaknesses
1. Weak Theoretical Foundation:
    - No convergence guarantees or approximation bounds
    - Limited analysis of when/why the method works
    - Missing connection to random feature theory for this specific setting
2. Poorly Motivated Architectural Choices:
    - No compelling justification for choosing HNNs over alternatives (Lagrangian, Port-Hamiltonian, etc.)
    - Graph networks not well-motivated for many test systems (regular lattices better suited for CNNs)
    - Missing literature review of Hamilton graph neural networks
3. Limited System Complexity:
    - Mostly simple spring-mass systems and basic molecular dynamics
    - ~10% relative error on Lennard-Jones systems suggests limitations for complex potentials
    - No testing on truly challenging physics (e.g., turbulence, phase transitions)
4. Accuracy Trade-offs Not Well Characterized:
    - Sometimes less accurate than second-order methods (LBFGS)
    - Large variance in results (Table 1) raises questions about reliability
    - No principled way to predict accuracy vs. speed trade-offs
5. Scalability Questions:
    - Memory complexity O(MNe) may become prohibitive for very large systems
    - Linear solver bottleneck O(Kd²L) not thoroughly analyzed
    - Integration constant handling appears ad-hoc
6. Missing Key Comparisons:
    - No comparison with other physics-informed ML acceleration techniques
    - No baseline comparisons with structure-specific alternatives (CNNs for lattices)
    - Limited comparison with other random feature applications to physics

### Questions
1. Can you provide convergence guarantees or approximation bounds for the random feature approach in the
  physics-informed setting?
2. Why specifically Hamiltonian neural networks? How does performance compare when applying random features to
  Lagrangian or other physics-informed architectures?
3.Can you provide principled guidelines for when accuracy degradation becomes significant? What physical properties are most affected?
4. What are the practical limits of your approach? At what system size does the linear solver become prohibitive?
5. How does the method perform on more challenging physical systems beyond simple spring-mass dynamics?

### Soundness
2

### Presentation
3

### Contribution
2
