# Towards Constraint-aware Learning for Resource Allocation in NFV-enabled Networks

- Decision: Reject
- Scores: 5, 6, 6, 3

## Abstract
Virtual Network Embedding (VNE) is a challenging combinatorial optimization problem that refers to resource allocation associated with hard and multifaceted constraints in network function virtualization (NFV). Existing works for VNE struggle to handle such complex constraints, leading to compromised system performance and stability. In this paper, we propose a \textbf{CON}straint-\textbf{A}ware \textbf{L}earning framework for VNE, named \textbf{CONAL}, to achieve efficient constraint management. Concretely, we formulate the VNE problem as a constrained Markov decision process with violation tolerance. This modeling approach aims to improve both resource utilization and solution feasibility by precisely evaluating solution quality and the degree of constraint violation. We also propose a reachability-guided optimization with an adaptive reachability budget method that dynamically assigns budget values. This method achieves persistent zero violation to guarantee the feasibility of VNE solutions and more stable policy optimization by handling instances without any feasible solution. Furthermore, we propose a constraint-aware graph representation method to efficiently learn cross-graph relations and constrained path connectivity in VNE. Finally, extensive experimental results demonstrate the superiority of our proposed method over state-of-the-art baselines. Our code is available at \href{https://anonymous.4open.science/r/iclr25-conal}{https://anonymous.4open.science/r/iclr25-conal}.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a Constraint-aware Network Abstraction Layer (CONAL) tailored for Virtual Network Embedding (VNE) to advance constraint management and improve training robustness, key factors for optimizing network system performance and reliability. By framing VNE as a violation-tolerant Constrained Markov Decision Process (CMDP), the authors aim to enhance solution quality and feasibility, ensuring complete solutions that accurately assess solution quality. The paper introduces a reachability-guided objective, paired with an adaptive feasibility budget method, to guarantee ongoing constraint satisfaction while reducing policy conservativeness and stabilizing policy optimization even with unsolvable instances. To address the complexity of VNE constraints, a constraint-aware graph representation is proposed, featuring a heterogeneous modeling module to capture cross-graph relationships and a path-bandwidth contrast module for heightened sensitivity to bandwidth constraints.

### Strengths
The authors propose a violation-tolerant Constrained Markov Decision Process (CMDP) modeling approach, which effectively evaluates solution quality and constraint violation levels, thereby enhancing solution feasibility and resource utilization efficiency.

### Weaknesses
- There has been some work examining about RL and NFV [1,2] and various approaches have been devised for solving the constraint violations therein, and it is not clear where this paper excels in relation to them.
  [1] Gu L, Zeng D, Li W, et al. Intelligent VNF orchestration and flow scheduling via model-assisted deep reinforcement learning[J]. IEEE Journal on Selected Areas in Communications, 2019, 38(2): 279-291.
  [2] Zeng Y, Qu Z, Guo S, et al. SafeDRL: Dynamic Microservice Provisioning With Reliability and Latency Guarantees in Edge Environments[J]. IEEE Transactions on Computers, 2023.
- Why applying DRL to cope with VNE is unclear, where existing heuristics do not always face high time overhead and it is not clear for what specific problems face what specific limited performance and why?
- The fact that real-world system validation is not based on real-world system implementations but is still based on simulation should not be blown out of proportion.
- The author claims that reinforcement learning learns effective strategies from unlabeled datasets, however, reinforcement learning actually learns strategies through interaction with the environment.
- Is constraint violation really acceptable for VNE ? Is it reasonable that constraint violations are allowed in the designed solution?

### Questions
Compare their work with more existing studies on solving constraint violations in applying RL to NFV （not just the papers mentioned above）, and compare it with them in experiments and related works.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a solution based on a Constrained Markov Decision Process for resource allocation in NFV-enabled networks. The problem of resource allocation in those networks (called Virtual Network Embedding in the related literature) is well-known in the research community and several solutions for it have already been proposed. Anyway, the proposed solution is sufficiently original and shows to achieve good performance results if compared with the primary baselines already existing in the literature.
However, the paper is weak in terms of in-depth technical insights about how to efficiently implement the proposed solution, of insufficient experimental evaluation and validation, and of potential impact in the field (see the following parts of this review form).

### Strengths
- The addressed topic is interesting and relevant, even if already well investigated in the related literature
- The proposed problem formulation and the deriving algorithmic solution are technically sound and do not exhibit big technical flaws
- The reported performance results are interesting and show that the proposed solution can outperform several related baselines in the existing literature
- The paper is generally well organized and well written

### Weaknesses
- The VNE problem has been investigated several times in the related literature. To be impactful, there is the need that novel solutions in the field do not propose only an algorithmic solution but also the design and implementation of a prototype integrated into real cloud/edge deployment environments. Otherwise, the level of technical originality and relevance could only be limited, given the status of maturity of the research field
- The paper does not include in-depth technical insights about how to exactly achieve an effective and efficient design/implementation of the proposed solution into a real prototype. No lessons learned from the experience of real deployment and evaluation in in-the-field deployment scenarios
- No systems engineering considerations and lessons learned about how to optimally configure and deploy the proposed solution
- The reported performance results are obtained by adopting simulation assumptions that are not realistic for many real deployment environments. I can understand that other papers in the literature have adopted a similar approach, but this is too simplistic. At least the validity of the used assumptions should be better justified and motivated in the paper. In addition, why not using real traces from real deployment environments, in particular for request demands?
- Even if the paper is generally well organized and well written, a few writing inaccuracies are still present in the manuscript and call for some minor revision work in order to improve the paper presentation style. Only to mention one example: "Addtional" in page 24.

### Questions
Please see the previous parts of this review form, in particular the weaknesses part above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tackles the VNE problem within NFV networks. Recognizing the limitations of existing solutions in handling intricate constraints and unsolvable instances, the authors propose a framework called Constraint-Aware Learning, formulates the VNE problem as a violation-tolerant constrained Markov Decision Process and introduces a reachability-guided optimization with adaptive reachability budgets. Additionally, the framework incorporates a constraint-aware graph representation method to capture cross-graph interactions and bandwidth-constrained path connectivity.

### Strengths
The paper is well-written and tackles a significant problem, offering practical implications for real-world network systems and potential applicability to other optimization challenges. The performance is benchmarked against several state-of-the-art baselines. Experiments are conducted across a wide range of network scenarios

### Weaknesses
The framework seems assumes a static PN setting. This assumption may not hold in highly dynamic network environments, such as mobile edge computing. 

The focus is mainly on computing and bandwidth constraints. Other important factors, such as latency, reliability, and energy efficiency, are not addressed.

### Questions
1. How does the proposed method perform in dynamic network environments where the physical network topology and resource availabilities could change over time? 

2. Can you provide a more detailed analysis of the computational complexity of the proposed method, especially in comparison to baseline methods? 

3. Could you elaborate on the rationale behind using contrastive learning in the constraint-aware graph representation module?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a new framework called constraint-Aware Learning (CONAL) to address the Virtual Network Embedding (VNE) problem in network virtualization. Specifically, the paper models the VNE problem as a violation-tolerant CMDP and introduces an adaptive reachability budget (ARB) to handle unsolvable instances.

### Strengths
The paper models the VNE problem as a violation-tolerant CMDP and introduces an adaptive reachability budget (ARB) to handle unsolvable instances.

### Weaknesses
1. The paper models the VNE problem as a violation-tolerant CMDP and introduces an adaptive reachability budget (ARB) to handle unsolvable instances. However, when dealing with unsolvable instances, no policy can satisfy the constraints, the Lagrange multiplier λ may tend to infinity, leading to numerical instability during training. Instability may affect the policy's performance on solvable instances. Provide empirical evidence of the behavior of the Lagrange multiplier λ during training. Specifically, plot the variation of λ over training iterations or time to illustrate how it evolves, especially in the presence of unsolvable instances.
2. The augmentation methods used in the path-bandwidth contrast module (physical link addition ϕA  and virtual link addition ϕB) lack sufficient theoretical and empirical justification. The choice of augmentation ratio ϵ significantly affects model performance, but the paper does not provide detailed analysis or guidelines for selecting these parameters. Provide theoretical explanations for how the augmentation methods contribute to improved bandwidth awareness. 
3. The integration of virtual and physical networks into a heterogeneous graph with numerous cross-graph links can lead to information redundancy and noise. Noise from irrelevant links can hinder the model's ability to learn meaningful representations.
4. The experiments are mainly conducted on simulated environments and limited network topologies (e.g., GEANT and BRAIN). This may not adequately demonstrate the model's performance.
5. Given the rapid development of NFV, some relevant literatures are missing to be discussed, e.g., NFVdeep: Adaptive Online Service Function Chain Deployment with Deep Reinforcement Learning, iwqos’19; Adaptive VNF Scaling and Flow Routing with Proactive Demand Prediction, infocom’18; FlexNFV: Flexible Network Service Chaining with Dynamic Scaling, network’19; Joint Optimization of Chain Placement and Request Scheduling for Network Function Virtualization, icdcs’17, etc.

### Questions
Overall，when infeasible instances exist in the Virtual Network Embedding (VNE) problem (i.e., there are no embedding solutions that satisfy all constraints), the optimization method employed in the paper causes the Lagrange multipliers (λ) to grow unbounded during training. The unbounded growth of λ leads to overflows or underflows in numerical calculation. The paper does not provide a robust method for detecting infeasible instances, nor does it implement any controls or mitigations for the growth of λ. This oversight means that, when faced with infeasible instances, the model may fail to operate correctly.

### Soundness
2

### Presentation
2

### Contribution
2
