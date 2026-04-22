# Disentangled representation learning through unsupervised symmetry group discovery

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
Symmetry-based disentangled representation learning leverages the group structure of environment transformations to uncover the latent factors of variation. Prior approaches to symmetry-based disentanglement have required strong prior knowledge of the symmetry group's structure, or restrictive assumptions about the subgroup properties. In this work, we remove these constraints by proposing a method whereby an embodied agent autonomously discovers the group structure of its action space through unsupervised interaction with the environment. We prove the identifiability of the true action group decomposition under minimal assumptions, and derive two algorithms: one for discovering the group decomposition from interaction data, and another for learning Linear Symmetry-Based Disentangled (LSBD) representations without assuming specific subgroup properties. Our method is validated on three environments exhibiting different group decompositions, where it outperforms existing LSBD approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a method to learn disentangled representations using symmetry groups without requiring prior knowledge of the group structure. The approach consists of three stages: learning an entangled representation (A-VAE), discovering the group decomposition via clustering, and learning a disentangled representation (GMA-VAE). This addresses a key limitation of existing LSBD methods that assume known group structure, and the work constitutes a valuable contribution to unsupervised representation learning.

### Strengths
1. This paper proposes a complete framework for automatically discovering group structure and utilizing it for disentanglement, addressing the limitation of prior LSBD methods that rely on known group structure. This contribution demonstrates considerable novelty and significance.
2. This paper provides solid theoretical foundations for the proposed method while conducting systematic evaluation on multiple datasets, comparing not only standard disentanglement metrics but also evaluating long-term prediction and generalization capabilities.

### Weaknesses
1. The treatment of non-symmetry-based disentanglement methods is insufficient. While the paper situates itself within the LSBD literature, it provides minimal discussion of how symmetry-based approaches compare to traditional disentanglement methods.
2. The experimental comparison is confined to symmetry-based methods, leaving the performance of the proposed approach against mainstream baselines such as QLAE and FactorQVAE unclear.
3. The extensive mathematical notation throughout the manuscript hinders readability. A consolidated symbol table would significantly improve accessibility for readers less familiar with group theory.

### Questions
1. Could the authors elaborate on the tangible advantages of symmetry-based methods, particularly when augmented by the improvements proposed in this paper, over mainstream approaches like QLAE and FactorQVAE?
2. How realistic are the underlying assumptions in practical settings? All experiments employ relatively simple synthetic datasets. Would the assumptions hold on more complex benchmarks such as MPI3D or Isaac3D?

### Soundness
3

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
This paper tackles unsupervised symmetry group discovery for Linear Symmetry-Based Disentanglement (LSBD). It proposes a two-stage pipeline: (i) learn an “entangled” action representation and encoder via an Action-VAE trained on transitions, then (ii) cluster actions into subgroup factors using a group-theoretic pseudo-distance, and finally (iii) learn a block-masked GMA-VAE that enforces LSBD without assuming subgroup structure. The paper proves identifiability under injectivity and “disentangled action set” assumptions and evaluates on Flatland-style grids with color cycles/permutations, COIL rotations with permutations, and 3DShapes. Reported wins are on LSBD metrics and long-horizon prediction versus Forward-VAE/SOBDRL/LSBD-VAE variants.

### Strengths
1. Clear LSBD pipeline that separates (i) equivariant pretraining, (ii) group discovery, (iii) block-structured LSBD learning, with explicit assumptions and a clustering rule grounded in group algebra. 

2. Good baseline experiments within the LSBD family (Forward-VAE/SOBDRL/LSBD-VAE variants) and consistent reporting on multiple disentanglement metrics and multi-step prediction. 

3. Reproducibility: code, dataset generation, and hyperparameters are described and (per the authors) released.

### Weaknesses
1. **Lack of realistic interactive experiments.**
The environments are synthetic (Flatland, COIL with permutations, 3DShapes). There are no tests on widely used embodied/control suites (e.g., DeepMind Control/ProcGen/Habitat/ManiSkill) where continuous groups (SO(2), SE(2), SE(3)) and sensor noise dominate, and where symmetries are only approximate. By contrast, prior interactive symmetry/LSBD works motivate interaction explicitly and evaluate on non-trivial dynamics (e.g., SOBDRL/Forward-VAE), and the broader equivariant RL literature demonstrates robustness/sample-efficiency gains in control domains—precisely the use-case this paper claims to address.

2. **Strong assumptions.**
Identifiability and clustering rely on (i) fully injective observations, (ii) a disentangled action set with each action belonging to exactly one subgroup, and (iii) the existence of short compositions mediating within-subgroup relations. These are rarely true in practical agents (redundant actuators, coupled controls, unmodelled dynamics). No robustness analyses are provided for violations (e.g., aliased actions, missing transitions, approximate commutativity). The empirical advantage may be specific to the LSBD formulation and the chosen metrics.

3. **Limited group diversity and no continuous-group evaluation.**
All experiments use finite groups (cyclic, symmetric). There is no assessment on continuous groups or approximate symmetries (e.g., SE(2)/SE(3) rotations and translations, scaling), where many RL/control tasks live and where identifiability and clustering are harder.

4. **Limited discussion of related work.** This paper is missing a lot of citations and related works from the group structured representation learning in interactive environments literature [1, 2]. Many of these work uses actions of the environment to disentangle or induce structure in the latent space and shows it's benefits in reinforcement learning etc. I would suggest the authors to do a more thorough literature review. 

[1] Learning Symmetric Embeddings for Equivariant World Models. JY Park et. al.
[2] Structuring Representations Using Group Invariants. M Shakerinava et. al.

### Questions
1. How does action clustering perform when the action set is not strictly disentangled (shared actuation across factors), or when a subset of transitions is missing/noisy? Please report clustering accuracy and downstream LSBD under controlled violations.

2. Can the pseudo-distance and masking be extended to continuous Lie groups (SO(2), SE(2)/SE(3))? If not, what fails? Compare against Lie-symmetry discovery approaches. 

3. Can you demonstrate that discovered groups help in standard control tasks by instantiating ρ inside an equivariant policy/value network (as in group-equivariant RL)? Measure sample efficiency and generalization versus non-equivariant baselines.

4. Partial observability: Your proofs assume injective observations. What happens in POMDPs (occlusions, distractors)? Any way to incorporate history (RNN/state-space models) while preserving identifiability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
They propose an algorithm to learn and disentangle the group structure underlying an environment, through interactions with this environment. Unlike previous methods, their method does not assume specific subgoup properties.

### Strengths
My understanding of the framework is relatively superficial and I did not check the maths carefully. 
Strengths:
- Careful explanations.
- Thorough comparison with other methods and with different metrics of disentanglement, which show an advantage over other unsupervised methods.

### Weaknesses
- How is the method less supervised than LSBD-VAE? It seems pretty supervised to me, with access to actions and consequences of these actions? What does it mean that "Both of our methods rely on a strong assumption which requires the available actions to be disentangled"?

### Questions
- The "Geomancer" should probably be cited as they somehow address a related question: https://arxiv.org/abs/2006.12982
- How to understand the problem of separating subgroups in different dimensions, when it is not always possible with a continuous encoder, as shown in https://arxiv.org/abs/2102.05623?
- How did Caselles-Dupre ́et al. (2019) demonstrate that symmetry-based disentanglement is only possible when access is granted to transitions? Consider explaining this in the paper. 
- "Such an approach to disentanglement," => these alternative frameworks seem only tangentially relevant, and could be moved to appendix.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studied symmetry-based disentanglement learning. To remove assumption about the symmetry group’s structure,  this work introduces a novel, multi-step approach to learning disentangled representations by having an agent autonomously discover the underlying symmetry group structure of its environment. The proposed method consists of three main steps: 1) Learn an Entangled Representation (A-VAE). 2) Discover the Group Structure. 3) Learn a Disentangled Representation (GMA-VAE). The authors validate their method on three environments with different group structures (Flatland, COIL, and 3DShapes). The results show that their action clustering algorithm perfectly recovers the ground-truth group decomposition , and the final GMA-VAE model achieves disentanglement performance comparable to supervised methods that are given the group structure in advance, outperforming other self-supervised approaches.

### Strengths
1. **Novelty:** The core contribution—learning an LSBD representation via three steps—is instructive for disentanglement learning. This shifts the paradigm from requiring prior knowledge to autonomously learning it from interaction.
2. **Theoretical Grounding:** The paper is built on a solid theoretical foundation, providing formal proofs for its key claims. This guarantees the existence of actions belonging to the same subgroup (Theorem 2) and the  of learning a Linear Symmetry-Based Disentangled (LSBD) representation (Theorem 3).  This level of rigor adds significant weight to the proposed methods.
3. **Strong Empirical Validation:** The method is thoroughly tested across multiple benchmarks. The final disentanglement performance is shown to be on par with a supervised method (LSBD-VAE), which is a very strong result.
4. **Other properties:** The paper goes beyond standard disentanglement metrics to show the other properties of the learned representations. The experiments on long-term prediction clearly demonstrate that the entangled methods prefer short-term predictions, while self-supervised ones prefer long-term predicitons.

### Weaknesses
1. Multi-Stage Pipeline: The method is not end-to-end. It requires training two separate models sequentially: first the A-VAE to learn action matrices, and then the GMA-VAE to learn the final representation. The authors acknowledge this limitation, noting that a future direction would be to unify these steps into a single optimization process.  
2. Limited Scope of Environments: The experiments are conducted on synthetic, visually simple datasets.  While these are well-suited for proving the group-theoretic concepts, it is unclear how the approach would scale to more complex, realistic, or stochastic environments where the underlying symmetries might be less perfect or harder to learn from raw pixels.
3. No visualisation evaluation for  reconstruction quality and latent traversal to explicitly demonstrate the tradeoff. 
4. The number of MIG on 3Dshapes is low.

### Questions
Why the MIG score is low on 3Dshapes? The discussion about failing cases is important to understand the drawbacks of proposed method.

### Soundness
3

### Presentation
2

### Contribution
3
