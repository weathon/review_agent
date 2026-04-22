# Graph Rewiring based on Flow Alignment for Improving Fluid Simulation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
To overcome computation burden of traditional computational fluid dynamics (CFD) simulations, researchers have explored different architectures to develop physics-informed simulation methods. Among them, graph neural networks (GNN) are most suitable for adopting CFD meshes, which are extensively used in engineering and industrial applications. However, classical GNNs propagate information among neighbour nodes, which highly restrict information exchange within the network. To address this issue, graph rewiring methods have been developed for generic graph problems, but not particular for fluid simulation. PIORF, introducing edges connecting distant nodes, is the first graph rewiring method to do so, and previous experiments have demonstrated its effectiveness against state-of-the-art generic rewiring methods. Nevertheless, in this work, we found that simply connecting all 2-hop nodes can provide competitive performance with PIORF. This result raises three questions: 1) Is physics-informed rewiring really useful for improving flow predictions? 2) Should we consider just local connection, instead of connecting distant nodes? 3) Do we need to change the connections based on input flow for rollout simulations? By thoroughly adopting physical fluid principles, we propose a simple yet very efficient method, Flow Alignment Rewiring (FLARE) technique, which connects 2-hop nodes only when the node direction aligns with input flow direction. Hence, FLARE is a physics-informed local rewiring method, different from PIORF and well-aligned with fluid physics. Extensive numerical experiments on flows over a cylinder and single and tandem airfoil under different flow conditions and deep network architectures demonstrate that FLARE outperforms PIORF and various 2-hop rewiring approaches by a significant margin.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FLARE, a simple graph rewiring strategy that is 2-hop, flow-aligned, directional, and updated each time step. At every step, it adds new along-flow edges within the local 2-hop neighborhood, so message passing follows physical advection and reduces over-squashing. FLARE is easy to plug into different backbones and shows lower long-horizon error, outperforming static or long-range baselines such as 2-HOP-ALL and PIORF on cylinder and airfoil benchmarks.

### Strengths
A lightweight, plug-and-play rule that rewires 2-hop, flow-aligned, directional edges dynamically well-matched to advective transport and slows long-horizon error growth.

### Weaknesses
Relies on strict unidirectionality and a fixed 2-hop scope (no CFL-based adaptivity, vortex/backflow handling, or conservation metrics), and lacks fair comparisons against directional/dynamic PIORF under matched edge budgets.

### Questions
1. In practical CFD, during one time step, roughly how many graph hops does advection cover? If the mesh/flow is anisotropic (Δx differs by direction), should you keep one hop count everywhere?
2. In regions with backflow or vortices, does the strict unidirectional downwind assumption break down?
3. From a physical standpoint, can PIORF’s rewiring be interpreted as an approximation to a non-local operator?
4. Can PIORF be implemented in directional and/or dynamic variants, and are there comparative results against related methods?
5. Do local unidirectional edge additions risk violating conservation constraints (e.g., mass, divergence-free, energy)?

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
The authors explore graph rewiring for GNN-based surrogates for CFD modeling. They note that previous rewiring schemes and PIORF' long-range connections ignore fluid dynamics and can violate physical principles of fluid dynamics. Authors propose their FLARE method (Flow alighment rewiring) that adds selected 2-hop neighbours and uses dot product between sender's velocity and the displacement vector to decide if a directed edge should be added. This methodology aligns flow with the velocity and graph is rewired at each time step. Experiments on three datasets show that simply adding all 2-hop neighbors already rivals PIORF, and FLARE offers gains. Ablations show that adding edges opposite to the flow decreases performance and 3/4-hop rewiring also degrades results.

### Strengths
The idea is novel, the paper is well-written, good presentation, also:
1. The physics-informed rewiring heuristic uses simple fluid-dynamics principles and often gives a gain in performance.
2. The method is easy to implement.
3. The authors made ablation study.

### Weaknesses
1. Gains of FLARE are dataset-dependent sometimes. But the paper draws broad conclusions. For example, "By comparing FLARE with 2-HOP-ALL and PIORF,we can conclude the effectiveness of physics guided design. The directional and local connections determined by flow directions are essential to achieve these performance gains." 
2. Authors have not analyzed threshold parameter sensitivity. It is simply fixed to T=0 in main experiments as I understand. Therefore, ablation study is limited.
3. No runtime/memory cost analysis was performed.

### Questions
1. Make more solid explanation of superiority of FLARE. See W1. Try not to use vague proves. Authors explain dataset dependency using this sentence "It is worth noting that FLARE gains more significant improvements on CylinderFlow than Airfoil likely because the former has more dynamics and challenging  flow conditions, giving more room for FLARE to improve."  Could the authors provide a more detailed analysis of why FLARE benefits more from dynamic flows, and how the method might be adapted to perform better on simpler cases such as the Airfoil density prediction?
2. Analyze flow-alighment threshold parameter sensitivity.
3. Conduct runtime/memory cost analysis.
4. Can you show rollout error vs. time for Airfoil and Tandem-Airfoil, not only CylinderFlow?

### Soundness
2

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
The paper proposes FLARE, a physics-informed GNN rewiring method for fluid simulation that selectively adds directional 2-hop edges based on instantaneous flow alignment. The authors show good performance over prior methods like PIORF and generic 2-hop rewiring.

### Strengths
1. The paper demonstrates consistent and substantial performance improvements over PIORF and structural baselines across three diverse datasets (unsteady, steady, compressible/incompressible) and 3 architectures.
2. The authors present a more straightforward solution than existing physics-based rewiring methods.
3. The introduction clearly articulates what problems of existing methods the paper aims to solve.

### Weaknesses
1. The proposed FLARE and its evaluation approach are insufficient to support the authors' claimed motivation of "rigorous adherence to physical principles."

2. The use of a zero threshold ($T=0$) raises questions about whether structural directionality (directional 2-hop) is the primary driver of performance rather than the alignment principle itself. Thus, the paper weakens the rigor of the "physics-informed" claim.

3. There is insufficient fluid dynamics or GNN-theoretical (e.g., curvature, over-squashing) justification for determining 2-hop as the optimal locality. The paper presents only experimental results.

4. The computational cost of dynamic graph reconstruction at every time step during rollout (inference time overhead) is not analyzed.

### Questions
Q1. What is the rationale for using $T=0$ in the base FLARE experiments, and do you believe this sufficiently tests the physical alignment principle? What happens in the case of positive thresholds where $T>0$?

Q2. More evidence is needed to support the claim that PIORF "does not align with physical principles" (line 237, footnote 2). The differences between FLARE and PIORF need to be discussed more clearly, and the velocity gradient (strain rate) aspect of PIORF should be addressed. Can you explain more precisely which aspects of PIORF are less physically straightforward?

Q3. Why does FLARE underperform PIORF on Airfoil density with BSMS-GNN?

Q4. Please provide a detailed computational cost comparison between baseline, PIORF, 2-HOP-ALL, and FLARE for each dataset. How much overhead does dynamic rewiring add?

Q5. Your comparison confounds multiple factors (locality, directionality, selection criterion). Can you provide ablations that isolate each factor? Specifically, factors such as whether all connections other than inverse connections are directional, or the proportion of bidirectional connections?

Q6. What are the detailed settings for 3-hop and 4-hop in the ablation study?

### Soundness
1

### Presentation
3

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
This paper introduces FLARE (Flow Alignment Rewiring) for improving GNN-based CFD simulations.
Classical GNNs operating on CFD meshes suffer from limited information propagation (over-squashing) and physics-misaligned connectivity, since message passing is confined to mesh adjacency that is unrelated to flow direction.
The proposed method rewires the mesh dynamically based on local flow alignment:
* only 2-hop local connections are considered (for locality);
* edges are directional (for unidirectional transport);
* new edges are added only when the velocity aligns with the added edge.

This yields a direction-aware, physics-consistent connectivity pattern that adapts during rollout as the predicted velocity field changes.
Experiments on three datasets—CylinderFlow, Airfoil, and Tandem-Airfoil-Cruise and across three backbone architectures (MeshGraphNet, BSMS-GNN, Transolver+) show that FLARE outperforms both the prior physics-informed rewiring method PIORF and 2-hop variants.

### Strengths
* the proposed approach is based on the first principles of fluid mechanics—locality, directionality, and flow alignment
* model-agnostic rewiring approach that can plug into any message-passing GNN or hybrid GNN-Transformer architecture without altering its core equations
* during rollout, rewiring is based on predicted velocities, so it's dynamically adjusted
* ablations provided: direction reversal, hop distance
* clear answers to guiding questions: The experiments directly address the three motivating questions—confirming that (1) physics-informed rewiring is beneficial, (2) local directional links suffice, and (3) dynamic flow-based updates matter.

### Weaknesses
* the proposed rewiring way doesn't depend on fluid velocity magnitude
* the paper is mainly empirical and proposes mostly an engineering solution
* dynamic rewiring at each step may add runtime cost, but no timing or complexity study is reported
* limited physical validation metrics: The study reports RMSE/MSE only, some physics-informed metrics would improve the results

### Questions
* What are computational expenses for your model?
* Can your approach be extended to multiphase flows or multi-physics simulations?
* How large is the runtime overhead of recomputing the rewired graph at each timestep?
* Your alignment score is dimensional, so how will you handle the cases of extremely slow flows in some points or extremely strong flows?
* Could the same idea be applied beyond fluids (e.g., heat diffusion, elasticity)?

### Soundness
3

### Presentation
3

### Contribution
2
