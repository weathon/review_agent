# Unified Analyses for Hierarchical Federated Learning: Topology Selection under Data Heterogeneity

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 2, 4, 6

## Abstract
Hierarchical Federated Learning (HFL) addresses critical scalability limitations in conventional federated learning by incorporating intermediate aggregation layers, yet optimal topology selection across varying data heterogeneity conditions and network conditions remains an open challenge. This paper establishes the first unified convergence framework for all four HFL topologies (Star-Star, Star-Ring, Ring-Star, and Ring-Ring) with full/partial client participation under non-convex objectives and different intra/inter-group data heterogeneity. Our theoretical analysis reveals three fundamental principles for topology selection: (1) The top-tier aggregation topology exerts greater influence on convergence than the intra-group topology, with ring-based top-tier configurations generally outperforming star-based alternatives; (2) Optimal topology strongly depends on client grouping characteristics, where Ring-Star excels with numerous small groups while Star-Ring is superior for large, client-dense clusters; and (3) Inter-group heterogeneity dominates convergence dynamics across all topologies, necessitating clustering strategies that minimize inter-group divergence. Extensive experiments on CIFAR-10/CINIC-10/Fashion-MNIST/SST-2 with ResNet-18/VGG-9/ResNet-10/MLP validate these insights, and provide practitioners with theoretically grounded guidance for HFL system design in real-world deployments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper establishes a unified theoretical framework for the topology selection problem in Hierarchical Federated Learning (HFL). In traditional HFL practice, topology selection often relies on intuition and experience, which the authors vividly describe as a “topology lottery.” To address this problem, the paper conducts rigorous convergence analysis under a non-convex setting for four HFL topologies composed of star-shaped (parallel) and ring-shaped (serial) aggregations. Theoretical analysis and extensive experimental results jointly reveal three key design principles: 
(1) the top-layer aggregation topology has a greater impact on convergence than the bottom-layer topology, and a ring-shaped top layer performs better under data heterogeneity; 
(2) inter-group data heterogeneity is the main performance bottleneck; and 
(3) the optimal topology depends on the client grouping structure (e.g., Ring-Star is suitable for many small groups, while Star-Ring is suitable for a few large groups).

### Strengths
1. Important and Practical Problem: This paper focuses on a problem that is crucial for the design of large-scale federated learning systems. As HFL is increasingly applied in fields such as IoT and edge computing, providing theoretical guidance for its architectural design has become an urgent need.

2. Novel Unified Theoretical Framework: This is the core contribution of the paper. To the best of my knowledge, no prior work has systematically compared all four mainstream HFL topologies within a single unified mathematical framework. This work fills that theoretical gap and integrates previously fragmented studies into a coherent whole.

3. Insightful and Actionable Conclusions: The three principles distilled in this paper are clear and highly insightful. These findings go beyond being mere mathematical results—they serve as practical guidelines that can directly guide engineers and researchers in system design, significantly reducing design uncertainty.

### Weaknesses
1. The mathematical derivations in this paper are element-wise, and the four topologies are discussed separately. This differs from matrix-level derivations: for example, if the intra-group and inter-group communication processes could be modeled by a communication matrix $\mathbf{W}$, then the convergence rate would be characterized by the eigenvalues of $\mathbf{W}$. That would yield more general conclusions than those in this paper — for instance, one could compute the eigenvalues of $\mathbf{W}$ corresponding to Ring–Star. Therefore, can the authors re-derive the convergence theory from a matrix perspective to obtain more general results?

2. Could you please provide the learning rate settings for each part of the experimental section?
The specific magnitudes and relative values of the learning rates are crucial, as they directly affect the conclusions about the *effective learning rate* discussed in **Corollary 1**.

### Questions
see the Weaknesses

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
This work studied the convergenve of an existing algorithm, hierarchical federated learning algroithm. They analyzed the convergence of this algorihtm over different topology, including ies (Star-Star, Star-Ring, Ring-Star, and Ring-Ring, under non-convex objectives.

### Strengths
The paper studied an interesting problem in hierarchical federated learning, which is relevant to scaling federated systems in practical deployments. It provides an extensive theoretical analysis covering multiple possible topologies (Star-Star, Star-Ring, Ring-Star, and Ring-Ring) under non-convex objectives, which is relatively comprehensive compared to typical HFL studies.

### Weaknesses
The discussion of related works is rather brief. Since the main focus of this paper lies in the analysis of hierarchical federated optimization algorithms, it would be beneficial to include a more comprehensive review of prior studies on hierarchical federated learning to better contextualize the contributions.

The comparison of convergence speed is also not convincing. The authors claim that the dominating term is $1/\sqrt{R}$ (Line 304), but according to the derivations in Lines 270–290, it should instead be $1/\sqrt{PMKR}$ for the star topology and $1/\sqrt{GPMKR}$ for the ring topology. 

The results derived fro star-star topology is not tight. Please comare your results with the existing results. For illustration, when the number of clients in each group reduces to 1, the hierarchical fedavg reduces to fedavg. The rate should be $1/\sqrt{GKR}$. But your current results cannot recover this. I believe your derivation for star topology is not tight. 


Furthermore, the analysis of FedAvg under the star-star, ring-star, and star-ring topologies has already been studied in prior works. The assumptions adopted in this paper are largely the same as those made in the existing literature. Therefore, the contribution of this work appears incremental, and the novelty over established results is limited.

### Questions
It is hard for me to understand why star topology lead to bad results. In my opinion, star topology enjoys the highest communication efficiency.

In Line 247, where do you define $\bar{x}^R$? 

In single tier topology, existing works have already proposed some algroithms and demonstrated convergence without data heterogeneity assumption. Is it possible to eliminate this assumption in hierarchical federated learning. 

The ring topology scheme asks clients to update model sequentially, which is ineffective. You cannot enojy the benefit of parallel computation. The conclusion that ring topolgy should be carefully checked. Based on your logic, can I argue that fedavg is not good, we should let clients update sequentially?

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
4

### Summary
This paper aims to provide a unified convergence framework for analyzing HFL topologies under non-convex objectives and heterogeneous data distributions. It addresses the lack of theoretical guidance on how to select optimal HFL topologies for real-world deployments, given different combinations of intra-group and inter-group heterogeneity.

### Strengths
++The motivation behind the analysis of HFL topologies under heterogeneity is strong
++The paper provides convergence bounds for all four topologies within a single analytical framework
++Experiments were conducted on multiple datasets, architectures, and heterogeneity conditions

### Weaknesses
++Key proof steps (especially those establishing cross-tier coupling and topology-specific bias) are only summarized without detailed proof
++The paper extends prior FedAvg-type analyses to hierarchical topologies rather than introducing fundamentally new analytical techniques, which exhibits limited novelty
++Only small to medium-scale vision datasets are used. Large-scale, high-variance tasks (e.g., NLP, speech) are missing
++There is no systematic study of how convergence changes with varying group size, local steps, or communication rounds
++The claim that HFL prioritizes scalability over acceleration is made but not empirically validated
++No variance/error bars or significance testing are presented in the figures

### Questions
Please address all the weaknesses above.

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
5

### Summary
The paper analyzes four kinds of two-tier HFL topologies (Star-Star, Star-Ring, Ring-Star, Ring-Ring) with respect to inter- and intra-group heterogeneities and the impact from top and lower layers. Then, the paper derives topology-dependent convergence bounds and claims that: 1. the top tier dominates the convergence over intra-group topology, while ring-based structure at the top helps more; 2. the best topology depends on overall grouping structure; 3. inter-group heterogeneity dominates the overall convergence. Experiments on CIFAR-10/CINIC-10/Fashion-MNIST datasets are carried out to support the claimed patterns.

### Strengths
1. The paper provides a clear, unified scope across all four two-tier HFL topologies. It offers a single convergence framework for Star-Star, Star-Ring, Ring-Star, and Ring-Ring under non-convex objectives and both intra- and inter-group heterogeneity.
2. The paper offers an actionable topology guidance by summarizing 3 critical and practical principles on top and lower tiers importance, topology selection and intra-/inter-group heterogeneities.
3. The definition of effective learning rate is introduced to show the impact of hierarchical structure on convergence. It also provides the intuition for the trade-offs between optimization and noise/heterogeneity.

### Weaknesses
1. The paper claims a "first" unified comparison of all four topologies, but related work already analyzes subsets and topology variants. The author is suggested to clarify more clearly what is new in the math or convergence bounds beyond those existing researches.
2. The analysis completely ignores communication costs and latency. The convergence rate is presented in terms of global rounds. This is misleading as the wall-clock time of a round differs vastly between topologies. A star topology is parallel (O(1) latency), while a ring topology is sequential (O(G) or O(M) latency), making it extremely sensitive to stragglers. The author is suggested to provide a formal analysis of the communication complexity for each topology. The final convergence guarantee should be presented in total wall-clock time or communication cost, not just rounds.
3. The analysis and algorithms assumes full participation from all clients in all groups in every global round, while partial participation is mentioned in the existing researches. Full participation is a highly unrealistic assumption in federated learning. The author is suggested to redesign the scenario where a subset of groups, and a subset of clients within those groups, are sampled for updates.
4. All experiments are image classification tasks, while no speech/NLP/time-series or cross-silo medical/IoT workloads where HFL is popular. The author is suggested to include at least one non-vision dataset.
5. Different ~𝜂 definitions per topology in the convergence analysis make it tricky for fair lr tuning. The experiments tune learning rates but lack in details. The author is suggested to report a transparent learning rate tuning strategy to ensure the results match the analysis.
6. The paper claims HFL is for scalability, not convergence acceleration and that "carefully selected single-tier FL configurations may actually converge faster." This is a strong claim, but the paper provides no theoretical or experimental comparison to single-tier FL. The author should add comparisons vs single-tier FL with tuned participation/aggregation under identical budgets.

### Questions
1. The derivation of the client drift bounds in Lemma 2 (Appendix C.1) is difficult to follow. For instance, the star-star derivation starts with "<=5" without justification for the constant '5'. Where is this number derived from?
2. Do the conclusions still hold with standard BatchNorm+momentum training? Or what will be the difference?
3. The analysis and experiments are presented in global rounds, which ignores the fact that a ring topology's round is slower than a star topology's. How would your primary conclusion that "ring-based top-tiers are superior" change if convergence was measured in wall-clock time?
4. Can you provide experimental results comparing your best HFL topology to a well-tuned single-tier FedAvg with N=100 clients?
5. How did you tune learning rate for the main results in Figure 2? Was it held constant across all four topologies to an unfair comparison, or tuned individually for each for a fair comparison?
6. How will the performance of the topologies change under a more realistic partial participation scheme, where only a subset of clients are active?

### Soundness
2

### Presentation
3

### Contribution
3
