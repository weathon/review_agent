# Physical Dynamics as Next Geometric Graph Prediction

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Physical dynamics simulation serves as a foundational component in scientific computing and AI applications. This paper presents a novel approach that redefines the problem as autoregressive prediction of spatiotemporal graph sequences. Built upon the expressivity of Transformer, we propose an Equivariant Spatiotemporal Transformer (EST), extending conventional Transformers with specialized equivariant spatiotemporal blocks. These blocks systematically alternate between spatial and temporal modules, rigorously maintaining E(3) symmetries throughout the process. Moreover, the design incorporates a novel Temporal Difference Graph (TDG) module derived from frame-wise variations, effectively modeling global dynamics and addressing cumulative errors in autoregressive predictions. Unlike traditional graph neural networks, our EST can process variable-length historical sequences and mitigate the persistent challenge of error accumulation in autoregressive processes. Comprehensive evaluations across multiscale physical systems (molecular-, protein-, and macroscopic-scale) demonstrate that our method achieves state-of-the-art performance, thereby showcasing its robust and versatile dynamics simulation capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the problem of simulating physical dynamics by framing it as an autoregressive prediction of spatiotemporal graph sequences. The authors propose the Equivariant Spatiotemporal Transformer (EST), a novel encoder-decoder architecture that maintains E(3) symmetries. The model incorporates specialized spatiotemporal blocks that alternate between spatial message passing and temporal self-attention. A key contribution is the Temporal Difference Graph (TDG), a module initialized from frame-wise differences that interacts with the historical trajectory to model global dynamics and mitigate cumulative error, a persistent challenge in autoregressive simulation. The authors evaluate EST on molecular, protein, and human motion dynamics datasets, demonstrating state-of-the-art performance, particularly in long-horizon predictions.

### Strengths
1. The introduction of the Temporal Difference Graph (TDG) is a creative and promising approach to the critical problem of error accumulation in autoregressive models (see Sec. 1, Sec. 3.2). This moves beyond simple frame-to-frame prediction and introduces a mechanism to explicitly model system dynamics over time.
2. The experiments are extensive, spanning three distinct and relevant physical scales—molecular (MD17), human motion (Motion Capture), and protein dynamics (Adk trajectory)—which convincingly demonstrates the model's versatility and generalizability (see Sec. 4).
3. The ablation study presented in Table 4 systematically deconstructs the model to validate the contribution of its key components. These experiments confirm the importance of maintaining equivariance, the efficacy of the TDG, the benefit of the encoder-decoder structure, and the rationale for using causal attention in the encoder (see Sec. 4.4).

### Weaknesses
1. The paper claims the TDG helps circumvent cumulative errors, but the theoretical justification provided (Taylor expansion in Sec. B.6.1) is a general argument for predicting velocity rather than position. It does not fully explain why the specific implementation of the TDG—as an independent graph token that interacts with all historical frames via attention—is superior to simpler or more direct velocity prediction schemes used in prior work.
2. The paper downplays the computational cost, stating that for short sequences the overhead is "negligible" (see Sec. E.4 ). However, the runtime analysis in Table 17 shows that EST-V is approximately 4 times slower than the previous SOTA (ESTAG) and 10 times slower than the fastest baseline (ST_EGNN). This is a significant trade-off that deserves a more prominent and nuanced discussion in the main text.
3. Why is predicting the difference from the previous frame simpler than predicting the next frame directly? How can this be demonstrated?
4. In the description of Eq. (4), please clarify whether the mean reduction term $\overline{x}$ is computed only once based on the initial input trajectory or is dynamically recomputed at each step of the autoregressive rollout process.

### Questions
Reference weaknesses.

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
2

### Summary
This paper proposes a novel approach to modeling physical dynamics by casting it as a next-step geometric graph prediction problem. Instead of representing physical systems as sequences of state vectors, the authors model each timestep as a graph and use geometric GNNs to predict the evolution of these graphs over time. The approach aims to integrate structural inductive biases with learned dynamics, leveraging the flexibility of GNNs in representing complex interactions and spatial relationships. The method is evaluated on several physical simulation tasks and is compared against existing neural and physics-based baselines.

### Strengths
1. The idea of formulating physical dynamics as a geometric graph prediction problem is conceptually interesting, and could potentially offer a unified framework for structured dynamical modeling.

2. The paper leverages geometric deep learning techniques, such as equivariant GNNs, which are well-suited to modeling the symmetries inherent in physical systems.

### Weaknesses
1. While the overall framework is promising, the technical details are somewhat underdeveloped. Key components (e.g., how graphs are constructed at each timestep, how node/edge features evolve) are described at a high level and lack sufficient mathematical clarity or justification.

2. The experimental results look weak. Only a small number of benchmarks are used, and comparisons with strong recent baselines (e.g., learned simulators like GNS or differentiable physics engines) are missing.

3. The model’s ability to generalize to systems with different numbers of components or interaction types is not convincingly demonstrated, which is a critical capability for physical simulation models.

### Questions
Pls refer to weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper reframes physical dynamics simulation—traditionally treated as predicting continuous trajectories—as a next-graph prediction problem using a Transformer-based architecture.
It introduces the Equivariant Spatiotemporal Transformer (EST), a model that predicts future graph states autoregressively while preserving E(3) symmetries (rotations, reflections, translations).
This allows EST to simulate molecular, protein, and human motion dynamics with geometric and temporal consistency.

### Strengths
This reframing connects sequence modeling (from NLP) with geometric dynamics, allowing the use of autoregressive Transformers for temporal evolution in physical systems.

It provides a unifying perspective that bridges graph-based physics learning, trajectory modeling, and generative dynamics under a single formalism.

### Weaknesses
Limited Physical Grounding Beyond Equivariance
Autoregressive Rollout Still Accumulates Error

### Questions
Why “next-graph prediction”? and this method is not new.
How does EST compare to Hamiltonian or Lagrangian neural networks that explicitly encode energy and momentum conservation?
Can the model generalize to continuous-time prediction (e.g., irregular timesteps)?
Can model predict rapid change unseen?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an attention-based encoder-decoder for molecule dynamic problems. I think the key part is Equivariant Spatiotemporal Attention. I think the overall design is not complicated, but useful for the future work.

### Strengths
The design is simple and easy to follow. And the experiments are promising. The overall design is close to modern NLP (gpt-like), but also considers the property of the molecules.

### Weaknesses
1. It looks like the model can not handle large molecules due to the token number (each node represents one token, not efficient.)
2. The model seems not to consider the connection for the 3d molecules.
3. The title is not proper. The proposed method can only handle dynamic molecule problem, not general physical dynamics.

### Questions
1. What is the average node number for the dataset used?
2.  10 rollout steps looks very small, is it possible to expand it?

### Soundness
3

### Presentation
3

### Contribution
3
