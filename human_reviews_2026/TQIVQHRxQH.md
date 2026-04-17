# Ghost in Topological Neural Flux Prediction: Boundary Conditions Matter

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Topological flux prediction (TFP) aiming to model spatiotemporal fluid transport over networked systems, has inspired and lent itself to various predictive methods. Whereas Graph Neural Networks (GNNs) demonstrate successes in  related prediction tasks, recent studies suggest that they can underperform even simple baselines in TFP, concluding that GNNs may be ill-suited for such problems. In this paper, we re-examine this claim by dissecting the learning behavior of GNNs on fluid networks, decoupling the roles of boundary nodes,
which regulate total influx, from interior nodes. We find that the dominant prediction errors arise at boundary nodes, which do not necessarily imply a fundamental limitation in the expressive power of GNNs. We interpret this phenomenon from a dynamical-systems perspective, arguing that GNNs incur substantial boundary losses mainly due to the lack of explicit modeling of boundary conditions. To compensate this information deficit, we propose a novel ghost-TFP framework, which learns ghost node proxies with an implicit solver to capture boundary-aware representations. Experimental results on two real datasets show that our method ghost-TFP improves standard GNNs by reducing the average MSE by 8.35\% and 5.0\%, and the boundary node MSE by 11.2\% and 7.1\%, respectively. For efficiency, we further devise an explicit solver that learns inverse operators which, depending on the underlying GNN backbone, can accelerate inference by $2\times$ on both datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper identifies that the poor performance of Graph Neural Networks (GNNs) in topological flux prediction (TFP) tasks is primarily due to high prediction errors at boundary nodes. To address this, the authors propose the "ghost-TFP" framework, which augments the graph with learnable ghost nodes to compensate for the lack of explicit boundary condition modeling. The framework includes both an implicit solver (using fixed-point iteration) and an explicit, more efficient variant (learning an inverse operator). Experimental results on two fluid network datasets (a river system and a blood flow simulation) show that the proposed method reduces errors, particularly at boundary nodes, and improves the topology-awareness of GNNs.

### Strengths
- Problem Identification: The paper stated a common and clear-eyed diagnosis of a specific failure mode (boundary node errors) in GNNs applied to fluid networks, aiming to challenge a oversimplified conclusion that GNNs are inherently unsuitable for TFP.

- Interdisciplinary Inspiration: The idea of borrowing the "ghost node" concept from computational physics (FDM) and integrating it into a GNN framework is intuitive.

- Extensive Ablation: The paper includes a thorough ablation study (RQ1-RQ5) that systematically builds up the proposed method and analyzes the impact of its components.

### Weaknesses
- Limited Conceptual Novelty and Technical Depth: The core idea, adding auxiliary nodes to a graph to improve information flow, is a well-established technique in GNN literature (e.g., virtual nodes, master nodes). The adaptation to boundary conditions, while contextually appropriate, does not represent a significant conceptual leap. The proposed framework heavily relies on and repurposes existing building blocks: the ghost node concept from FDMs, Implicit GNNs (Gu et al., 2020) for the solver, and learned inverse operators. The work is more of a skillful assembly of these components than the introduction of a fundamentally new principle or architecture.

- Narrow and Simplistic Empirical Validation: The evaluation is conducted on only two datasets, one of which is extremely small (the Blood flow graph has only 14 nodes). This severely limits the generalizability of the claims. The performance on such a tiny graph is not convincing evidence of a method's robustness or scalability. The baselines are outdated and limited. The comparison is made against standard GNNs like GCN and GAT, but there is no comparison with more recent, advanced GNN architectures designed for complex physical systems or those that inherently handle boundary conditions (e.g., various GNN4PDE hybrids, transformer-based architectures). This makes it difficult to assess the method's standing in the current research landscape.

- Superficial "Physics-Inspired" Integration: The connection to physics, while a good starting point, remains superficial. The Robin boundary condition is used as a loose motivation for the ghost node update rule (Eq. 5), but the MLP used to learn the ghost node embedding is a black box. The method does not strictly enforce physical constraints but rather uses physics as a soft inductive bias. There is no analysis or guarantee that the learned solutions better respect underlying conservation laws or dynamics. The derivation in the appendix, while detailed, serves more as a post-hoc justification for the matrix structure rather than as a foundational element that deeply guides the neural network design. The link between the continuous PDE derivation and the discrete GNN operations is analogical rather than rigorous.

- Absence of Critical Analysis and Failure Modes: The paper lacks a discussion of the limitations or failure modes of the proposed ghost-node approach. For instance, how does the method perform if the "downstream neighbor" assumption is incorrect or ambiguous? What is the effect of noisy or missing data at boundary nodes? There is no theoretical analysis regarding the expressive power gained by adding ghost nodes. The argument is purely empirical.

### Questions
- The Blood flow dataset has only 14 nodes. Can you provide evidence that the observed improvements are statistically significant and not an artifact of the very small graph size? Have you tested on larger or more complex physical networks?

- Beyond the standard GCN/GAT backbones, how does ghost-TFP compare against more recent state-of-the-art GNNs developed for physical systems (e.g., MP-PDE, MeshGraphNets, other neural operators adapted to graphs, or transformer-based methods)?

- The ghost node embedding is learned via an MLP. How crucial is the specific architecture of this MLP? Is there a risk that the network is simply learning to overfit the boundary error pattern on these specific datasets rather than learning a generalizable representation of boundary conditions?

- The paper claims the method improves "physics-awareness." Beyond lower MSE, what quantitative evidence is there that the predictions are more physically consistent (e.g., better adherence to mass conservation, more plausible flow directions)?

- Given that boundary nodes are known in advance, why was a standard heterogeneous GNN approach not adopted or compared against? A heterogeneous graph with "boundary" and "interior" node types seems a more natural and principled way to model this structural prior. Can you justify the necessity of the ghost-node framework over such a baseline?

Overall, the novelty is limited and technical depth are insufficient. In its current state, this work does not meet the bar for this venue.

### Soundness
2

### Presentation
2

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
This paper argues that Graph Neural Networks (GNNs) underperform in Topological Flux Prediction (TFP) for fluid networks not due to inherent limitations, but because they fail to model crucial boundary conditions, leading to high errors at boundary nodes. To address this, the authors propose ghost-TFP, a framework using learnable "ghost nodes" (inspired by numerical methods) as proxies for missing boundary information. This creates a coupled system solved either via an implicit GNN (accurate but slow) or a more efficient explicit inverse operator learner. Experiments show ghost-TFP significantly reduces boundary and overall MSE compared to standard GNNs, restoring topology-awareness, with the explicit solver variant offering up to 2x speedup

### Strengths
1. Creative Integration of Numerical Methods Concept: The adaptation of ghost nodes from finite difference methods into a GNN framework is highly innovative. It provides a principled way to inject boundary-awareness into the message-passing mechanism.

2. Dual Solver Approach (Implicit & Explicit): Proposing both an implicit solver (theoretically grounded in fixed-point dynamics for handling the coupled ghost-boundary system) and an explicit inverse operator learner (computationally efficient) is a strong contribution. This addresses both correctness and practical usability.

3. Strong Empirical Validation: The experiments clearly demonstrate the effectiveness of ghost-TFP. It consistently improves accuracy over standard GNN backbones, significantly reduces the problematic boundary errors, and restores the expected topology awareness (i.e., performing better with the correct forward flow graph $A$ than the reversed $A^T$).

4. mproved Topology Awareness: The results in Table 3 strongly support the hypothesis. Adding ghost nodes significantly widens the performance gap between using the correct forward graph ($A$) versus the incorrect reverse graph ($A^T$), indicating that the model becomes much more sensitive to the physical flow direction once boundary conditions are better handled

### Weaknesses
Computational Cost of Implicit Solver: The implicit GNN solver, while effective, incurs a substantial runtime overhead (reported as $\approx 13\times$ slower than standard GNNs), limiting its practical applicability for large graphs or real-time scenarios


Scalability Concerns for Explicit Solver: The explicit solver relies on learning an inverse operator, which approximates the inverse of the potentially dense parametric adjacency matrix $A_{\Theta}^{\prime}$. Learning and applying dense operators can scale poorly ($O(N^2)$ or worse) for very large graphs, which is not fully addressed. The ablation uses a fully connected adjacency for the inverse operator learner, which definitely does not scale

Limited Exploration of Boundary Condition Types: The ghost node construction and derivations are based on discretizing a Robin-type boundary condition. While flexible, the paper doesn't explicitly discuss or evaluate how the framework adapts to other common types like Dirichlet (fixed value) or Neumann (fixed gradient).

### Questions
Scalability of the Explicit Inverse Operator: The explicit solver approximates $(A_{\Theta}^{\prime})^\dagger$ using a GNN ($\Psi$) or potentially a dense matrix. How does the computational complexity (training and inference) of learning and applying this inverse operator $\Psi$ scale with the number of nodes $|V'|$? Is it feasible for graphs much larger than the River dataset (358 nodes)? Have the authors considered structured approximations (e.g., low-rank, sparse) for the inverse to improve scalability?

Generalization Across Boundary Conditions: The ghost node MLP is derived from a Robin-type condition discretization. How would the ghost node construction (Eq. 5) and the resulting augmented adjacency $A'$ need to be modified to handle pure Dirichlet or Neumann boundary conditions? Does the framework maintain its effectiveness across different BC types without significant re-engineering?

Sensitivity to Ghost Node MLP: The ghost node embedding $h_g$ is learned via an MLP (Eq. 5) taking $h_b$ and $h_{nbr}$ as input. How sensitive is the overall performance to the architecture and capacity of this MLP $\theta_{gh}$? Does a simple linear layer suffice, or is a more complex MLP required to capture the boundary dynamics accurately?

Trade-offs in Explicit Solver Regularization: The explicit solver uses a regularization term $\lambda || (A_{\Theta}^{\prime})^{\dagger} A_{\Theta}^{\prime} - I ||_F^2$ (Eq. 10) to enforce the inverse constraint. How was the hyperparameter $\lambda$ chosen, and how does the trade-off between the node prediction loss ($\mathcal{L}_{node}$) and the regularization loss ($\mathcal{L}_{reg}$) affect the final accuracy and the learned inverse operator's properties?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The work presents a model, ghost-TFP, dedicated to machine learning for the topological flux prediction problem, e.g., river and blood flows. The authors focus on the fact that the previous study [Kirschstein & Sun, ICML 2024] has larger errors on the boundary than in the interior. To address this issue, they propose learning boundary conditions using ghost nodes, virtual computational points lying outside the boundary. Ghost nodes provide the same mechanism for computing spatial differentiation at boundary points as at interior points. To enable learning with unknown boundary conditions, the authors proposed implicit and explicit computational schemes. The experimental results demonstrate that the proposed method outperforms the previous study.

### Strengths
1. The motivation to address boundary conditions is clear, as the authors find that the previous study has greater error at the boundaries than in the interior.
2. Learning an inverse operator for the explicit method is an interesting approach, and in fact, it is more efficient than the implicit one.

### Weaknesses
1. The problem setting addressed in the paper is somewhat too specific and yields fewer takeaways for the community. If the problem setting is new, we can learn about the application of machine learning. However, the present paper focuses on the problem addressed elsewhere, proposing a specific model for it. The authors should emphasize the importance of the problem and/or the broader impact of the work.
2. The reason why the authors focus on ghost nodes is not clear since there are a lot of ways to handle boundary conditions, such as standard schemes in the finite element method and the finite volume method. The authors should clarify the benefits of ghost nodes, both qualitatively and quantitatively.
3. The presentation of the problem analysis section is not clear. The physical dimension of F is inconsistent between Equation 2 and line 129, which complicates the rest of the paper. In addition, it is not clear why Equation 3 has $\approx$ rather than equal.
4. The benefit of proposing the implicit method is unclear if the explicit method has almost the same performance with shorter computation time. The authors should clarify the intrinsic benefit of the implicit one.
5. The descriptions of the equations are incomplete. There is no clear explanation of what $\theta_\mathrm{gh}$ and $\Theta$ are. $\mathcal{I}$ is explained as the implicit solver, but the argument of $\mathcal{I}$ is not an equation, and the output of $\mathcal{I}$ is not clear, either.
6. Presentation in the experiments section is unclear and hard to follow. The authors present scores in the main text, but since they are presented in the tables, the authors should elaborate further on the analysis of the results. In addition, Tables 3 and 4 are difficult to understand because the captions are not self-contained. Also, the reviewer recommends including the runtime in a table.
7. The analysis of the experimental results is hard to follow. In particular, the reviewer cannot understand the critical difference between RQ1 and RQ2 and why the authors conclude “yes” in RQ2.
8. The baselines are not strong enough. Most baselines are GCN-variant, which correspond to the Laplacian operator and are not suitable for convective phenomena. The authors should compare against more sophisticated GNN methods, e.g., MP-PDE[Brandstetter+ ICLR 2022].
9. Captions of figures and tables are difficult to read. The font size is too small, and there is almost no space between the main text. The author should keep enough space for the captions.

Minor points:

* Equation 2 has $F$, whereas line 128 says $\mathcal{F}$, maybe a typo?

### Questions
1. In Table 3, why is the reverse setting for ghost-TFP not tried?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes ghost-TFP, a novel GNN framework that introduces learned ghost node proxies and an efficient explicit inverse-operator solver to overcome the dominant prediction errors caused by the lack of explicit boundary condition modeling in existing GNNs for topological flux prediction.

### Strengths
1. The paper provides an insight by decomposing prediction errors into boundary vs. interior nodes. The authors reveal that the perceived failure of GNNs in flux prediction is primarily a boundary modeling issue rather than a fundamental architectural limitation. 

2. The connection between GNN message-passing and numerical PDE solvers is well-articulated, and the ghost node formulation is properly grounded in classical computational fluid dynamics techniques.

3. The paper evaluates multiple GNN backbones and provides both implicit and explicit solver variants.

### Weaknesses
1. This paper motivates the problem by claiming that GNNs violate physical laws. Still, the proposed ghost-TFP only demonstrates MSE reduction without any physics-informed constraints or verification that predictions actually satisfy these physical principles, and uses a purely data-driven MLP.
2. The explanation of the architecture and generalization capability of the Explicit Inverse Operator $\Psi$ introduced for efficiency is insufficient, which may make it difficult for readers to understand the model.
3. The evaluation uses only two real datasets with small scales (River: 358 nodes, Blood: 14 nodes). This makes it difficult to demonstrate extrapolation to networks with different scales, topologies, and boundary ratios 
4. The conclusion that "the main cause of GNN failure is boundary information deficit" may be strongly specialized to the structure of the two presented datasets (where boundaries constitute more than half of the nodes). 
5. While the paper presents boundary condition motivation, the implementation seems to diverge from this motivation. Although the claim itself is interesting, the evidence is insufficient to generalize from two cases without validation on diverse data domains and scales.

### Questions
1. Can you provide a sensitivity analysis for the regularization hyperparameter $\lambda$ in the inverse constraint of the Explicit Solver?
2. Can you justify the 58.4% boundary node ratio in the River dataset, and explain why the Blood Flow dataset is evaluated on a 14-node subgraph rather than the full 30-node network?
3. How does the MLP-based ghost node (Eq 5) relate to the Robin boundary condition (Eq 4)? And what happens when a boundary node has zero or multiple downstream neighbors?
4. How sensitive are the results to choosing a boundary condition type (Robin vs. Dirichlet vs. Neumann)?
5. Why do you use a single shared $\Theta$ across multiple layers in the implicit solver? Have you tried layer-specific weights?

### Soundness
2

### Presentation
2

### Contribution
2
