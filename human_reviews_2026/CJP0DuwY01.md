# Entangled Schrödinger Bridge Matching

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Simulating trajectories of multi-particle systems on complex energy landscapes is a central task in molecular dynamics (MD) and drug discovery, but remains challenging at scale due to computationally expensive and long simulations. Previous approaches leverage techniques such as flow or Schrödinger bridge matching to implicitly learn joint trajectories through data snapshots. However, many systems, including biomolecular systems and heterogeneous cell populations, undergo *dynamic* interactions that evolve over their trajectory and cannot be captured through static snapshots. To close this gap, we introduce **Entangled Schrödinger Bridge Matching (EntangledSBM)**, a framework that learns the first- and second-order stochastic dynamics of interacting, multi-particle systems where the direction and magnitude of each particle's path depend dynamically on the paths of the other particles. We define the Entangled Schrödinger Bridge (EntangledSB) problem as solving a coupled system of bias forces that *entangle* particle velocities. We show that our framework accurately simulates heterogeneous cell populations under perturbations and rare transitions in high-dimensional biomolecular systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work extends Schrödinger Bridge Matching with a bias force that depends on the velocity of all particles (instead of just one) and is therefore “entangled”. The bias force is designed to always point “towards the data distribution” (i.e. have positive dot product with the score function) and trained via a proposed cross-entropy objective in path space. Experiments on cell perturbation experiments and transition path sampling for fast-folding proteins are presented showing the validity of the method on real-world data.

### Strengths
- The paper is overall easy to follow and well-presented/well-written
- Experiments are on two different real-world data sets and show that the methods can solve real-world tasks.
- The parameterizations of the bias force are sound and novel.
- All statements are supported by clear derivations and proofs.

### Weaknesses
- The experimental results are limited and benchmarks against prior methods are lacking. If this was not possible, it would be good to highlight why such comparisons were not done to make it understandable to the reader.
- The motivation of the work could be presented more clearly. While it is clear that the entangled bias force is a mathematical possibility, it would be great to have a clearer description of why we would want that. In short, what is the motivation behind introducing this? SBM does not learn any physical dynamics of interacting particles (this is purely data-driven), so it is not really clear to me what the motivation is. It would be great if this could be carved out more clearly.

### Questions
- Cell perturbation experiments: How do the interaction of cells represented by the model represent actual biological of cells in that data set? A random subset of n=10 cells is chosen. But how would you know that these cells have interacted in the physical/biological system? To my knowledge, you cannot. Therefore, it is not clear to me whether you can really speak here of an interacting particle system. I would appreciate a clarification, as I acknowledge that this could be limited by my understanding of the dataset.

- Line 63: It is defined how R_t and r_t^i are related (it is implicitly clear but this should be explicitly stated)
- In Proposition 4.1. The initial distribution is suddenly assumed to be a delta function. That should be stated as an assumption.
- Line 226: Kabasch -> Kabsch algorithm

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
2

### Summary
The authors extended Schrodinger bridge problem when the particles are not independent but interacting and following interacted Langevin dynamics. The authors then test their algorithm in single cell sequencing and molecular dynamics examples.

### Strengths
The paper targeted on an important problem of considering SB problems when particles are not independent samples anymore but are interacting. The method is designed to more aligned with momentum SB type of methods rather than handling over-damped dynamics.

### Weaknesses
- The presentation on setups and notations is hard to follow when samples are from interacting particles.
    - The current score is mostly driven by this. I am open to change if this is clarified. See questions. 
- The theoretical results are less careful in handling position part which has no diffusion on it. 
- In experiments the authors seem did not compare to other baselines, e.g., vanilla SB.

Note: I tried to reread the paper several times with fresh eyes but I am really having a hard time understanding the setup hence my comments beyond setups and proof questions are less reliable. I am quite familiar with SB literature but I think the authors are more from physics side that contributed to my struggle. I am more than happy to be convinced that my current score is due to lack of understanding rather than issues of the work.

### Questions
## Setup
Overall I don't think I fully understood the setup, mostly on what should be considered a sample and if there is any independent samples. This can be well clarified if the author can provide an example in the setup. 

- I am a bit confused by the notation $\mathbf{X}\_{t}$. Do we have multiple observation of  $\mathbf{X}\_{t}$ so that information about $\pi\_{\mathcal{A}}$ is available or we only have one observation $\mathbf{X}\_{t}$? 

- When we are thinking two marginals, what are we having? Do we have 1) an ensemble of these multiple particle systems, but they are different ensembles at two ends, but we observe all particles within each ensemble? or 2) we only have one ensemble and at each end we see different part of it? 

- Do we need to assume at the beginning and the end samples have same number of particles and they are all interacting? 

- Do we know $\pi_{\mathcal{A}}$ and $\pi_{\mathcal{B}}$ or we only have sample to some extend. 

- How is the setup different from an augmented vanilla SB whose state space is the concatenation of all particles?  

- Use MD as an example, is the author saying all the molecules are interacting or the measured angle are interacting? Are there multiple molecules in the sample? 

## Notation
- In section 2, the authors used some convention from physics like Boltzmann constant $k_B$ temperature $\tau$ and mass $m_i$. These are obvious for people with some level of physics background but less obvious for people in ML and it will be nice to briefly explain, or just absorb things into constants like $\gamma$. 

## Proofs
- Using Girsanov: there are some conditions needed that are not spelled out. More importantly, the diffusion matrix $\Sigma$ on the position part are 0 since this is a Langevin rather than over-damped one, this makes $\Sigma$ not invertible. The conclusions *are* correct since the drift on locations is known but one need to be careful about these dynamics.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Entangled Schrödinger Bridge Matching (EntangledSBM), a framework designed to model the second-order Langevin dynamics of multi-particle systems with entangled bias forces. The EntangledSBM framework solves the Entangled Schrödinger Bridge (EntangledSB) problem using stochastic optimal control theory and further enables conditional sampling during inference. Experiments on cell cluster dynamics and molecular systems demonstrate the effectiveness of the proposed framework

### Strengths
- The paper is well structured and easy to follow.
- The paper provides clear, step-by-step theoretical derivations.

### Weaknesses
- **Limited experimental scale.**  
  The experiments are restricted to relatively small-scale systems. Specifically, they are limited to small, fast-folding proteins (fewer than 20 amino acids), which are significantly smaller than those encountered in real-world applications. In addition, quantitative comparisons with relevant baselines such as [1, 2, 3] are necessary for the transition path sampling experiments.

- **Lack of efficiency and scalability analysis.**  
  The practical training objective in Eqn. (19) depends on the number of simulation steps, which should have a noticeable impact on performance. Additional analyses on training and inference efficiency, as well as scalability, are needed to demonstrate that the proposed framework is computationally efficient.

**Minor issues:**

- There are several writing issues, such as the misspelling of “initial” in Line 93, the redundant “where” in Lemma 1 (Lines 821–826), and the same issue in Eqn. (12), Line 198.  
- Eqn. (6) is somewhat confusing, as it appears to be a discrete-time expression while still containing the term $dt$.  
- The notation \(X_{0:K}\) in Eqn. (18) has not been introduced previously. It likely represents the discretized trajectory, and additional explanations regarding this discretization should be provided.


[1]. Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, and Max Welling. Stochastic optimal control for collective variable free sampling of molecular transition paths. Advances in Neural Information Processing Systems, 36:79540–79556, 2023

[2]. Kiyoung Seong, Seonghyun Park, Seonghwan Kim, Woo Youn Kim, and Sungsoo Ahn. Transition path sampling with improved off-policy training of diffusion path samplers. In The Thirteenth International Conference on Learning Representations, 2025.

[3]. Yuanqi Du, Michael Plainer, Rob Brekelmans, Chenru Duan, Frank Noe, Carla P Gomes, Alan Aspuru-Guzik, and Kirill Neklyudov. Doob’s lagrangian: A sample-efficient variational approach to transition path sampling. Advances in Neural Information Processing Systems, 37:65791–65822, 2024.

### Questions
- Could you provide a more detailed comparison with the prior work [2]? It appears that the main differences lie in the use of a cross-entropy (CE) training objective and the replacement of the MLP architecture with a transformer.

### Soundness
3

### Presentation
2

### Contribution
3
