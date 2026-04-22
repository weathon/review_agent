# MeGA-MP: Metric Graph Advection Message Passing - Solving Dynamical Processes on Metric Graphs with Graph Neural Networks

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Many real-world systems are organized as networks, where spatio-temporal dynamics unfold not only at nodes, but also along the connections between them. 
Such networks are known as $\textit{metric graphs}$.
Examples include utility networks and the propagation of signals in physical or biological media.
The methods that approach such problems are mostly PINN-based with limited generalizability to PDE parametrization and boundary conditions. 
A recent work addresses the limitations of PINNs by proposing a neural operator tailored to drift-diffusion dynamics.
However, in many real-world settings, hyperbolic dynamics like advection dominate the spatial evolution of a system, which has not been addressed so far.
In this work, we propose a novel graph operator that solves linear advection on metric graphs via message-passing. 
We provide an error bound on the approximation of ground truth obtained through multiple MP-iterations without the necessity of training.
Empirically, we show that it solves advection competitive to numerical and neural solvers. 
Combined with trainable components like MLPs, we demonstrate how it can be applied to realistic advection-reaction dynamics in water distribution systems, where we achieve superior performance compared to baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces MeGA-MP, a novel message-passing learning-free operator designed to model dynamical processes governed by linear advection on metric graphs. MeGA-MP leverages semi-Lagrangian backtracing, a classic method for solving PDEs, to formulate message passing updates. The authors also provide theoretical analyses, including convergence guarantees and interpolation error bounds. Experimental results demonstrate MeGA-MP’s effectiveness both as a standalone numerical solver for advection-type systems and as a component within learnable models.

### Strengths
1. The paper is well-written and easy to follow. 
2. The problem in concern is novel and interesting.
3. Using the method of characteristics to define message functions is an elegant way to incorporate advection priors into message passing.
4. Experimental results are generally convincing and include both synthetic PDE settings and real-world water distribution datasets.

### Weaknesses
1. The authors position their work at the “intersection of physics-informed ML, numerical PDE solvers, and neural PDE solvers.” However, this framing feels somewhat misleading, as it implies a general applicability that the proposed method does not have — not all PDEs can be reformulated into the specific graph advection form considered here. To improve clarity, the authors should qualify this statement and explicitly state that the method targets linear advection-type equations on metric graphs. Moreover, the paper’s main focus isn't about PDE solving — which fundamentally relies on the method of characteristics — and is more about its novel adaptation of this framework to metric graph structures.
2. For Table 1, runtime and computational cost comparisons with neural baselines are missing. Such results are important to evaluate the practical efficiency of MeGA-MP relative to standard MPNNs. Furthermore, the authors use EPANET-MSX to generate all the ground truth data for this experiment. However, they do not compare their method to EPANET-MSX as a baseline.
3. For Table 2, MeGA-MP’s performance appears comparable or weaker than classical numerical solvers (e.g., semi-Lagrangian, RK4). The paper should clarify under what conditions MeGA-MP offers benefits. Furthermore, computation time should be brought forward to fully evaluate the effectiveness of MeGA-MP. I would be more than happy to raise my score if computation-related statistics are provided.
4. The paper does not explore the performance of MeGA-MP under noisy, incomplete, or perturbed flow fields, which are common in real systems such as water networks. An ablation or discussion on robustness to noise, measurement error, and missing temporal samples would strengthen the empirical evaluation.

### Questions
1. How MeGA-MP works in noisy settings, which are common in real systems such as water networks? 
2. The paper argues that general-purpose PDE solvers cannot be directly applied to advection on graphs. However, could one instead apply a neural PDE solver such as FNO to each edge, or integrate a traditional ODE integrator like RK4 for temporal updates? How would such hybrid baselines compare? 
3. I am not an expert with metric graphs. I notice that only NNConv and PDE-GNN are used as baselines in the WDS task, but PDE-GNN performs poorly. Are these two methods considered state-of-the-art for dynamical learning on metric graphs?

### Soundness
4

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
This paper deals with solving differential equations on metric graphs, which is a very interesting an timely topic

### Strengths
Interesting problem with a clear goal. Numerically challenging examples are chosen.

### Weaknesses
The comparison to classical numerical methods is somewhat missing and the authors only introduce a machine learning model in the final steps without much discussion. I feel this to be more of a numerical analysis contribution than an ML paper.

### Questions
I remain a bit confused about this paper whether it really is a learning paper. The authors refer to previous work, e.g. current ICML, but there main contribution seems to be using the method of characteristics and a nodal condition that is labeled as a message passing equation.

- Why do the authors claim in the introduction that numerical approaches are having a hard time as they require large amounts of compute and memory. The paper cited by the authors of Blechschmidt et al. in ICML provides a finite volume implementation that is difficult to beat in the simulation case. In my opinion learning approaches shine for inverse or parameter identification approaches.
- What are the weights in equation (3)? The usual formulation on networks requires nodal conditions such as Kirchoff-Neumann conditions (only mentioned in the appendix). I guess the information aggregation plays this role for their formulation? 
- What is the learning component of this paper? The message passing formulation seems to be a numerical approach for implementing nodal conditions. How would this compare to a Kirchoff-Neumann formulation? The MLP is only added to the mix in Section 5 without much discussion of the details.
- How does this scale for large graphs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors claim that they introduce a new advection operator and that this have not been addressed so far.
They bring convincing results to show that this is indeed a needed operation.

### Strengths
Advection is indeed needed for graphs. The numerical experiments demonstrate that.

### Weaknesses
Advection was already introduced for graphs. There are a number of recent references. The author mention a few but fail to say why what they do it different. Furthermore, new theoretical results on graph advection were analyzed in recent work Mathematical Models and Methods in Applied Sciences Vol. 35, No. 5 (2025) 1237–1265, Modeling advection on distance-weighted directed networks



I find the novelty of the paper very minimal

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes MeGA-MP, a message-passing framework that models linear advection on metric graphs, iterating an MPNN-style update to propagate information between nodes. The authors provide theoretical results and two experimental evaluations: (i) advection-reaction forecasting on simulated water distribution systerms, where MeGA-MP outperforms baselines; and (ii) solving advection on a 1D domain comparing MeGA-MP to standard numerical solvers, where it achieves comparable perfomance.

### Strengths
- On 1D advection problems, the method achieves accuracy comparable to semi-Lagrangian and RK4 schemes across different discretizations.
- The appendices include detailed proofs and descriptions of the experimental setup, contributing to the theoretical rigor and partial reproducibility of the work.

### Weaknesses
Limited novelty and scope:
- Advection-based dynamics have already been explored in GNNs (i.e., Advection–Diffusion–Reaction GNN [1]). A comparison with ADR-GNN is therefore necessary to clarify MeGA-MP’s novelty and relative performance.
- The paper restricts its scope to linear advection, which limits applicability to more realistic systems that may also involve diffusion or nonlinearities.

Insufficient empirical comparison and limited experimental validation:
- The experimental evaluation is limited only to 2 GNN baselines and 3 tasks, weakening the empirical benefit of MeGA-MP. Authors should consider including [1], transformer based GNNs (e.g. [2,3]), dynamical system based GNNs (e.g., [4,5]), or other spatiotemporal GNNs (e.g., [6,7]).
- A complexity and runtime analysis (both theoretical and empirical) with respect to other baselines is missing; this would strengthen the claims of efficiency and scalability, and could further highlight the potential runtime advantages of MeGA-MP compared to baseline methods.
- The authors are encouraged to test MeGA-MP on spatiotemporal benchmarks such as Metr-LA and Pems-Bay [5] to demonstrate broader applicability to real-world problems, where coupling advection and reaction proved beneficial in [1]. 
- The paper lacks an ablation study comparing the learned and unlearned versions of MeGA-MP, which would help isolate and quantify the contribution of the learnable components to the overall performance.

Technical aspects lack clarity or are insufficiently detailed:
- The presentation is mathematically dense, with some overly complex notation (e.g., for defining forecasting problems). Including more intuitive explanations or illustrative examples in the main text would improve accessibility.
- To improve readability and ease of understanding, the paper should include pseudocode or a concise algorithmic summary of the proposed MeGA-MP method.
- It is unclear whether model selection (e.g., hyperparameter tuning, early stopping criteria) was performed properly and consistently across baselines.
- It is not clearly stated whether MeGA-MP evolves dynamics solely from initial conditions or whether it can incorporate external or time-varying inputs during simulation. Clarifying this point would help assess its flexibility for real-world forecasting tasks.

-----

[1] Eliasof et al. Advection Diffusion Reaction Graph Neural Networks for Spatio-Temporal Data. In LoG 2023

[2] Rampášek et al. Recipe for a General, Powerful, Scalable Graph Transformer. In NeurIPS 2022

[3] Shi et al. Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification. In IJCAI 2021

[4] Gravina et al. Anti-Symmetric DGN: a stable architecture for Deep Graph Networks. In ICLR 2023

[5] Heilig et al. Port-Hamiltonian Architectural Bias for Long-Range Propagation in Deep Graph Networks. In ICLR 2025

[6] Li et al. Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. In ICLR 2018

[7] Wu et al. Graph wavenet for deep spatial-temporal graph modeling. In IJCAI 2019

### Questions
see weaknesses

### Soundness
2

### Presentation
1

### Contribution
1
