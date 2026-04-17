# Inter-Agent Relative Representations for Multi-Agent Option Discovery

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Temporally extended actions improve the ability to explore and plan in single-agent settings. In multi-agent settings, the exponential growth of the joint state space with the number of agents makes coordinated behaviours even more valuable. Yet, this same exponential growth renders the design of multi-agent options particularly challenging. Existing multi-agent option discovery methods often sacrifice coordination by producing loosely coupled or fully independent behaviours. Toward addressing these limitations, we describe a novel approach for multi-agent option discovery. Specifically, we propose a joint-state abstraction that compresses the state space while preserving the information necessary to discover strongly coordinated behaviours. Our approach builds on the inductive bias that synchronisation over agent states provides a natural foundation for coordination in the absence of explicit objectives. We first approximate a fictitious state of maximal alignment with the team, the Fermat state, and use it to define a measure of spreadness, capturing team-level misalignment on each individual state dimension. Building on this representation, we then employ a neural graph Laplacian estimator to derive options that capture state synchronisation patterns between agents. We evaluate the resulting options across multiple scenarios in two simulated multi-agent domains, showing that they yield stronger downstream coordination capabilities compared to alternative option discovery methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper targets at multi-agent option discovery. Specifically, they propose a joint-state abstraction that compresses the state space while preserving the information necessary to discover strongly coordinated behaviors.

### Strengths
- Multi-agent option is an important and challenging research problem.
- This paper is tecnically solid and theretically grouned.
- The comparison with related work is comprehensive.

### Weaknesses
- MAPPO results are not provided for overcooked tasks.
- Please provide a discussion on the computation cost compared to other methods.
- Please provide a discussion on the compatibility with mainstream MARL methods, such as. VDN and MAPPO. Can they be integrated together?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work tackles the problem of option discovery in multi-agent reinforcement learning (MARL), where existing methods (like Eigenoptions) are impractical because they discover too many options, scaling with the massive joint state space. The authors introduce a novel inter-agent relative state abstraction that compresses this joint state space into a compact latent representation. This abstraction is centered around the "Fermat state," which represents the point of maximal alignment among agents, thereby focusing the discovery process on inter-agent relational dynamics rather than the full state space. By discovering options within this compressed, relation-focused space, the method drastically reduces the number of options and encourages the emergence of highly coordinated joint behaviors.

### Strengths
1. Novel State Abstraction: The core contribution is a new inter-agent relative state abstraction centered around the "Fermat state." This is a new technique for compressing the massive state space. A direct, practical benefit of this abstraction is that it drastically reduces the number of options that need to be discovered, making the problem computationally tractable. The abstraction is not just a random compression; it's designed to focus the discovery process on inter-agent relational dynamics and synchronization. This is highly relevant for cooperative tasks.

2. The method is empirically shown to discover options that represent highly coordinated joint behaviors, which is the goal of many MARL tasks. The paper demonstrates that its method discovers a more diverse and generalizable set of coordination skills that can better support agents in various downstream tasks compared to other methods.

### Weaknesses
1. The authors define the distances between agents' states using the Fermat inter-agent state distance and further approximate this distance by learning a parameterized function. The training process for this function seems to map the joint state space to a single-agent state space by minimizing a temporal distance. The authors claim that this distance metric is capable of comparing the similarity between the states of two agents. However, it is unclear whether this temporal distance can distinguish between the states of two agents, since agents may have different states at the same time step. Moreover, what are the differences between an agent's observations and its 'single state'?

2. The proposed method requires that the MI between each feature distance and the remaining state-feature pairs does not exceed the MI between the feature pairs themselves. Why not directly maximize the MI objective between the state-feature pair and the random vector $Z$? This would also seem to distinguish different features.

3. More evaluations on some commonly used benchmarks, such as SMAC and GRF, should be included.

### Questions
Please see the Weaknesses above.

### Soundness
3

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
This paper tackles multi-agent option discovery by proposing an inter-agent relative state abstraction that centers the joint state around a “Fermat state” of maximal alignment, measures team “spreadness” per feature dimension via learned temporal distances, then discovers options by estimating Laplacian eigenvectors on this relative representation rather than on the raw joint state. The key idea is that synchronisation patterns between agents are a natural substrate for coordination, so compressing the joint state into a feature-wise, alignment-centric embedding both reduces option count and biases discovery toward genuinely joint behaviours. The method trains a Fermat encoder together with a temporal distance model, adds a conditional mutual information regularizer to keep each feature’s distance prediction focused on its own signal, and adapts MacDec-POMDPs to execute joint options under simple team-level synchronisation assumptions. Experiments on Level-Based Foraging and Overcooked show that options learned on the relative representation improve IQL downstream performance and often beat options produced by raw joint state or Kronecker-product constructions.

### Strengths
The paper isolates a real bottleneck in multi-agent option discovery, the exponential blow-up of joint state spaces and the tendency to produce loosely coupled skills, then proposes an elegant fix, compress the joint state via a Fermat-state-centred, feature-wise temporal distance representation that foregrounds coordination structure. The learning objective for the Fermat encoder and temporal distances is straightforward, the use of successor-style temporal distances fits the coordination intuition and is representation agnostic, the conditional mutual information penalty is a thoughtful touch to keep features disentangled and to avoid degenerate solutions, the adaptation of MacDec-POMDPs to joint options with simple voting and synchronisation assumptions is clearly described and easy to implement. On the empirical side, the method improves IQL on LBF and Overcooked, it outperforms Kronecker-product options and raw joint-state options, the multi-dimensional variant is especially helpful in the richer Overcooked feature space, and the ablations align with the story the paper tells.

### Weaknesses
The empirical scope is still moderate for ICLR standards, two domains and a few scenarios, the evaluation would benefit from additional environments with heterogeneous agent state spaces where the homogeneity assumption breaks, and from tasks with explicit communication limits to stress the information sharing assumption used during joint option selection. Sensitivity to key choices is only partially explored, for example the number of eigenvectors retained, the dataset size for pretraining the distance and Laplacian estimators, the effect of temporal distance quality on option quality, and the robustness of the CMI regularizer. The assumptions for joint option initiation and information sharing are reasonable for a first pass, yet the paper could better characterise failure modes when the “enough agents agree” condition is not met, or when shared observations are stale. Finally, while Figure 2 is very helpful, a more systematic probe into what kinds of synchronisation patterns each discovered option encodes would make the contribution crisper, for example, controlled grid tests that manipulate only one relational factor at a time.

### Questions
Can you report sensitivity to the number of eigenvectors and to the size and policy used to collect the 500k transition dataset for distance and Laplacian estimation, in particular, how do option quality and downstream performance vary when the dataset is smaller or when trajectories come from a partially coordinated policy rather than random, can you quantify how the accuracy of the temporal distance model affects discovered options, for example by adding noise or using a weaker successor-distance estimator, can you provide a diagnostic of the synchronisation motifs your eigenoptions encode, for instance by plotting controlled rollouts that show which feature subsets a given option aligns, can you stress test the joint option initiation and information sharing assumptions, for example by injecting delays or dropouts into the shared signals and reporting how often options fail to trigger, can you add a small heterogeneous-agents study, even synthetic, to illustrate how the approach might extend beyond identical S* spaces.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper presents a framework for discovering temporally extended actions (options) in MARL settings. The authors propose a method that compresses the joint state space into a latent representation centered around a "Fermat state". This abstraction enables the discovery of coordinated behaviors by focusing on inter-agent relational dynamics rather than raw joint states. Empirical evaluations in Level-Based Foraging and Overcooked domains demonstrate that the proposed method outperforms existing option discovery techniques and baseline MARL algorithms, particularly in tasks requiring strong coordination.

### Strengths
By introducing the Fermat state and leveraging n-distance metrics, the authors effectively re-center the representation space to emphasize coordination. The use of multi-dimensional disentangled representations allows for richer behavioral diversity, and the mutual information constraint ensures that each feature contributes meaningfully to the learned options. 

The empirical results are thorough, covering multiple domains and scenarios, and the visualizations of eigenvectors provide intuitive insights into the discovered coordination patterns. The integration of the framework into the MacDec-POMDP model and the use of decentralized training further enhance its practical relevance.

### Weaknesses
I am not familiar with this area, so hard to understand the motivation. It is expected to show more background and examples about the problem. 

The reliance on homogeneity among agent state spaces limits its applicability to more diverse multi-agent systems. The assumption of full observability and shared information among agents, while practical for evaluation, may not hold in real-world scenarios. 

The computational overhead introduced by multi-dimensional distance estimation and mutual information regularization might be significant.

### Questions
Is the proposed option learning generalizable to different environments? 

What is the difficult part of extending beyond 2 agents?

### Soundness
3

### Presentation
3

### Contribution
3
