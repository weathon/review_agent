# Improving Long-Range Interactions in Graph Neural Simulators via Hamiltonian Dynamics

- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Learning to simulate complex physical systems from data has emerged as a promising way to overcome the limitations of traditional numerical solvers, which often require prohibitive computational costs for high-fidelity solutions. Recent Graph Neural Simulators (GNSs) accelerate simulations by learning dynamics on graph-structured data, yet often struggle to capture long-range interactions and suffer from error accumulation under autoregressive rollouts. To address these challenges, we propose Information-preserving Graph Neural Simulators (IGNS), a graph-based neural simulator built on the principles of Hamiltonian dynamics. This structure guarantees preservation of information across the graph, while extending to port-Hamiltonian systems allows the model to capture a broader class of dynamics, including non-conservative effects. IGNS further incorporates a warmup phase to initialize global context, geometric encoding to handle irregular meshes, and a multi-step training objective that facilitates PDE matching, where the trajectory produced by integrating the port-Hamiltonian core aligns with the ground-truth trajectory, thereby reducing rollout error. To evaluate these properties systematically, we introduce new benchmarks that target long-range dependencies and challenging external forcing scenarios. Across all tasks, IGNS consistently outperforms state-of-the-art GNSs, achieving higher accuracy and stability under challenging and complex dynamical systems. Our project page: https://thobotics.github.io/neural_pde_matching.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Information-preserving Graph Neural Simulators (IGNS), a graph-based neural simulator that improves modeling of complex physical systems. IGNS enforces Hamiltonian dynamics to preserve long-range interactions and extends to non-conservative systems. It includes warmup initialization, geometric encoding, and multi-step training to enhance stability. Evaluated on new benchmarks with long-range dependencies and external forces, IGNS outperforms state-of-the-art methods in accuracy and robustness for dynamic systems.

### Strengths
- The proposed Information-preserving Graph Neural Simulator introduces a principled integration of port-Hamiltonian dynamics into graph-based simulators, marking a significant step beyond existing message-passing and oscillatory GNN frameworks.
- Theoretical analyses are thorough and provide a clear justification for the model’s ability to capture complex and long-range physical interactions. 
- Experimental evaluation is comprehensive, spanning six datasets and consistently demonstrating the superior accuracy and stability of IGNS compared to strong baselines.
- The paper is clearly written and well structured: the motivation for each component (port-Hamiltonian core, warmup phase, geometric encoding, and multi-step loss) is clearly articulated, and the accompanying figures effectively convey both the methodology and the empirical findings.

### Weaknesses
- Although the theoretical analysis establishes information preservation and universality, it remains largely qualitative in linking these properties to the observed empirical improvements. A more quantitative or ablation-based verification (e.g., measuring gradient norms or energy conservation over rollouts) would provide stronger evidence for the theoretical claims.
- The training/testing computational overhead of the port-Hamiltonian formulation and the warmup phase is not explicitly analyzed; reporting runtime or memory costs relative to standard GNSs would clarify the practical trade-offs. 
- Although the benchmarks are diverse, most tasks are synthetic or controlled simulations. It would strengthen the paper’s significance to include or discuss applications in more realistic or large-scale physical systems.
- The geometric encoding used to map edges to features follows the same formulation as previous works (e.g., MGN) and therefore cannot be considered a novel contribution.
- Several related approaches are not cited or compared [1–4], which limits the contextual positioning of this work within recent advances in graph-based physical simulation. 
- Introducing "warmup phase" in GNNs is not novel. Eagle [2] employs a warmup-like phase in its encoder, using multiple message-passing blocks to aggregate local and global context before rollout, which parallels the proposed initialization strategy.
- The separation of state variables into coordinates and momenta, as well as the coordinate–momentum supervision in Eq. (10), are established techniques already used in [2–3].

[1] EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions. ICML 2025

[2] Eagle: Large-Scale Learning of Turbulent Fluid Dynamics with Mesh Transformers. ICLR 2023

[3] Efficient Learning of Mesh-Based Physical Simulation with BSMS-GNN. ICML 2023

[4] Physics meets Topology: Physics-informed topological neural networks for learning rigid body dynamics

### Questions
1. In L201-203 and in Appendix D, the paper states that $\gamma_\theta(t)$ and $\tau_\theta(t)$ are time-varying coefficient vectors produced by MLPs with parameters θ. Could the authors clarify what exactly is used as the input t to these MLPs? Is t normalized to a fixed range (e.g., [0, 1]) or directly represented as the raw timestep index? Additionally, if the model is trained on trajectories with 400 steps, can the learned time-dependent MLPs generalize to longer rollouts (e.g., 1000 steps) without retraining, or does the model rely on an absolute temporal scale?
2. L231-233 states: "Thanks to the energy conserving core of IGNS, this globally informed latent state is preserved throughout the rollout, rather than being dissipated." Could the authors clarify why this happens? Can you provide qualitative or quantitative analysis showing how the latent state is preserved over time？
3. Regarding the multi-step loss: how many time steps were included in the loss computation? Are the reported results based on single-step MSE or on the rollout of the entire sequence? Please include these experiment details.
4. Why does WaveBall not require a warmup phase?
5.  In the supplementary code (`igns.py`), L651: `x = self.one_step(x, edge_index, edge_weight, batch, t=i)` passes `i` as `t`. Here, `i` corresponds to the layer index rather than the time step. Can the authors explain why this is done?

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
4

### Summary
The paper introduces a graph neural simulator built on a port-Hamiltonian system. Unlike the existing relevant literature, the core idea of the proposed method is to introduce the symplectic integrator while proposing a port Hamiltonian involving non-conservative energy terms in order to closely align the proposed emulator with the ODE dynamics. The warmup iteration is also proposed in an attempt to enhance the long-range message propagation. The paper also conducts theoretical analysis about the universality and the sensitivity of the model. The proposed framework is evaluated on a range experiment including a couple of new scenarios designed to assess the long-range propagation capability under external forcing.


Overall, I think the paper is not ready for the publication in the current form, because 1) the main arguments about the theoretical analysis and proposed architecture are over-claimed and loose, and 2) the experiments miss some relevant baselines and essential ablation study. In particular, Theorem 2 is almost completely identical to a theoretical result in an existing literature. A detailed clarification is necessary to highlight the difference. The detail of the warmup iteration is also missing, which is another factor that causes the difficulty to assess the significance and soundness of the contribution of the paper. The details are given in "Weaknesses" and "Questions" columns.

### Strengths
- The universality result (although I do not fully understand the proof yet) is novel. 
- Proposal of new tasks, specifically designed to test long-range propagation and oscillatory dynamics under external forcing.

### Weaknesses
**Over-claimed assertions:**
- A sentence starting at line 290 regarding Theorem 1 is over-claiming. Being with/without compact supports makes a huge difference in the significance.
- The multi-step objective is a pretty common loss function for training auto-regressive models. The authors need to cite relevant papers or argue that this is a pretty common approach. The use of symplectic integrator in this context is also not novel. 
- Theorem 2 is almost identical to a sensitivity result in [1], and it is unclear if this theorem is different enough from the result of [1] to claim it as novel and/or original.

**Misdirected experiments:**
- The core idea of the paper is closely related to the idea in [1], but apparently the baselines in the main experiments do not include the model, which makes the experiments setting look unfair and the result unfairly unconvincing.
- The ablation version of damping and forcing/residual terms is missing. This ablated model is essentially equivalent to the idea in [1] adopted to symplectic integrator with well-adopted multi-step loss, which I believe is valuable to be compared.
- The other fundamental difference is use of symplectic integrator and warmup iteration, but the ablation experiment also misses these aspects.
- The paper addresses the long-range propagation problem, but the experiment misses the impact of increasing the resolution of the space, which controls the difficulty of the long-range message propagation. 

**Inaccurate description on the assertion and proof of Theorem 1:**
- Line 287: $\dot{x}_{0}$ should not be included.
- Theorem 1 should need a compact support on which $F$ is approximated by $\Psi_{\theta}$.
- Line 855: The second Hamiltonian equation. This should be introduced formally.
- Line 1028 misses the definition of $\bar{p}$ and the relation between $p$ and $q$ is unclear, so the derivation of (24) is non-trivial.

**Minor:**
- Typo: the map $\Phi$ is duplicated in line 987
- Diameter is important metric in this context, but it is missing from Table 3


[1] Heilig, et.al., "Port Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks.", ICLR 2025.

### Questions
**Regarding the warmup iteration:**
- How exactly does the warmup iteration work? I cannot find its detail including the update formula for each iteration.
- Is the port-Hamiltonian used in the warmup phase shared with the time-evolving forward model? 

**Proof of Theorem 1:**
- Why is it fine to set $D$ to be $0$ without the loss of generality? It is not obvious at first glance.
- What does B represent at line 989. Is it a parameter involved in $r(t)$?

### Soundness
2

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
3

### Summary
This paper presents Information-preserving graph neural simulators (IGNS). It is a novel approach aimed to improve long-range propagation of information and to reduce error accumulation in task related with physical modelling. By using port-Hamiltonian dynamics, IGNS preserves information across graphs and captures conservative and non-conservative dynamics. Authors use some key innovations, for example: a warmup phase for initializing global context, geometric encoding for irregular meshes, and a multi-step training objective for stable long-horizon predictions.

Additionally, the authors provide strong theoretical guarantees for IGNS's universality and gradient preservation. Authors provide comprehensive experiments on six benchmarks, including new tasks. IGNS consistently outperforms many baselines.

### Strengths
1. The idea is novel and interesting. 
2. The paper has solid theoretical foundations (universality, non-vanishing gradients) and robust experimental design against many strong baselines across different physical systems.
3. The paper is generally well-written, logically structured, with clear problem statements and architecture descriptions. Good use of figures, tables, and appendices.
4. It offers a substantial advancement in GNSs by solving critical long-standing limitations.

### Weaknesses
1. A direct ablation for authors' static geometric encoding feels like missed. The authors make an argument that model's specific geometric encoding helps avoid "overfitting," which is important for generalizing well. However, authors don't really show us an experiment where they directly compare their own model with and without developed static encoding. Instead, they rely an indirect comparison to other architectures.
2. No inference time analysis. It it is hard to understand, beyond just training time, how fast do the models predict one or multiple steps during inference? This is crucial for "accelerate simulations". 
3. No scale analysis. How does the computational cost scale with increasing graph size (nodes, edges) or rollout horizon?

### Questions
1. See W2 and W3. It is interesting to see results of inference time and scale analysis.

2. In the message-passing update you use only q_{i} and q_{j} (see eq. 9). Does this exclude the momentum variables p_{i} and p_{j} from exchange between neighbors, or is q there meant to denote the full node state [q,p]? Please clarify.

3. Did you conduct an ablation study to directly compare static geometric encoding of IGNS against a dynamic geometric encoding strategy (e.g., updating edge features per time step) within the IGNS framework (not changing the model)? It is needed to empirically validate its claimed benefit in preventing overfitting. If so, where are these results presented? If not, please conduct this ablation study.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Information-preserving Graph Neural Simulators (IGNS) for learning physical dynamics of  both conservative and non-conservative port-Hamiltonian systems. Key features of the work are: 1) Hamiltonian formalism to eliminate part of the model systematic drifts; 2) a warmup phase for global context initialization; 3) a multi-step training loss for long-horizon stability.
The authors also present new benchmarks (Plate Deformation, Sphere Cloth, Wave Balls) to test long-range dependencies.
IGNS achieves good results across all datasets, showing higher or comparable accuracy and stability than MeshGraphNets and GraphCONs together with some other methods.

### Strengths
* novelty in introducing a port-Hamiltonian formalism to graph simulators
* theoretical proofs of universality and non-vanishing gradients
* data efficiency, importance of warm-up steps and length of confident prediction horizon were investigated
* code attached

### Weaknesses
* the advantages of Hamiltonian dynamics simulation were not properly studied (like the energy conservation for the conservative systems)
* the generalizability is under question
* the file dataset.zip in supplimentary link is corrupt
* the paper lacks qualitative discussion of the distinction between two versions of the algorithm is not clear (IGNS, IGNS_ti (time-independent))

### Questions
* What are the computational expensies of your model training and inference compared to competitors and numerical simulator?
* What are the limits of your model generalization? (out of distribution, new geometries, etc.)
* Why for Kuramoto-Sivashinsky equation the results for IGNS_ti are so much better than for IGNS?

### Soundness
3

### Presentation
1

### Contribution
2
