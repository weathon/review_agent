# CSE-ET: Joint Optimization for 3D Multi-UAV Explore-and-Track under Connectivity, Separation, and Energy Constraints

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
We study cooperative target tracking by UAV swarms in open airspace under realistic resource and safety constraints (limited communication, limited power, and inter-UAV safety separation), and introduce active exploration based on spatial information entropy. Most existing methods focus on one or two subtasks (e.g., only tracking accuracy, or treating exploration/communication/power as independent modules), and their metrics and scenario coverage are limited, making it difficult to reflect practical performance under the above constraints. Targeting an evading UAV with an optimal avoidance policy, this paper jointly considers communication, spatial exploration, power consumption, and inter-UAV collision avoidance for the tracking team, and designs a unified joint-optimization framework based on multi-agent reinforcement learning. The framework models low-level action control with a base model, high-level behavior control with a multi-agent decision model, and is updated by a joint optimization algorithm. In simulation, we construct evaluation metrics covering tracking, power, and exploration, and compare with the mainstream multi-agent baselines. Experiments show that our method outperforms the baselines on average by $\textbf{+23\\%}$ in tracking success rate, $\textbf{+25\\%}$ in power saving, $\textbf{+24\\%}$ in spatial exploration, and $\textbf{+18\\%}$ in aggregate reward. To our knowledge, this paper presents the 3D multi-UAV cooperative tracking framework that is most closely aligned with practical constraints.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles the problem of targeting an evading UAV, with an optimal avoidance policy, by jointly considering communication,
spatial exploration, power consumption, and inter-UAV collision avoidance for the tracking team. It proposes to design a unified joint-optimization framework based on multi-agent reinforcement learning is ambitious.

### Strengths
The problem of targeting an evading UAV has been tacked using a holistic approach considering communication,
spatial exploration, power consumption, and inter-UAV collision avoidance.

### Weaknesses
The authors claim in the abstract that ‘To our knowledge, this paper presents the 3D multi-UAV cooperative tracking framework that is most closely aligned with practical constraints.’, But, the system model considered by the authors present serious problems which make their problem unrealistic.
 
For instance, the model used (1a-1c) supposedly represents the UAV. The problem is that according to Fig. 1 they are considering a quadrotor which has  4 different variables to control it; this allow to control the position at x,y and z independently. But the model (1a-1c) has only two angles to control the 3 velocities (x,y,z). This implies that they cannot control all 3 velocities independently which means that this model does not seem to accurately describe the motion of the UAV considered in the paper.
Furthermore, the choice for the angles in model   (1a-1c) seems unortodox. Usually the orientation is described with the Euler angles (roll, pitch and yaw) or with quaternions.
 
 In (2), the authors present an arbitrary model for the Evader, without any support of the literature. This model seems arbitrary and heuristic, but it also seems to disregard the rich body of literature on the problem of Hunter-Prey which present already sophisticated equations for the behaviour of the Evader agent.
 
In (4), the authors are comparing an SNR with a distance, which is inconsistent in terms of the physical units.
 
The power consumption model (6) presented by the authors cannot be used for the scenario considered by them. The reason for this is that in (Zeng et al 2019), the original authors mention that propose such power consumption model, they state that V is the forward velocity, impliying a 2D motion (i.e. no velocity on the z axis). But the authors of the paper under evaluation assume a 3D motion which implies a z-axis velocity which is not supported by the power consumption model used by the authors.

The reward expressions have been introduced without justification or citing references.

### Questions
See points raised in the Weaknesses section

### Soundness
1

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
This paper presents a comprehensive multi-agent reinforcement learning (MARL) framework for solving practical and complex multi-target tracking problems using UAV swarms. Despite making very strong contributions, several potential weaknesses exist with respect to the proposed "sequential policy update" mechanism, which requires further analysis.

### Strengths
The paper's biggest contribution is that it clearly diagnosed the instability experienced by standard MARL algorithms (MAPPOs) in complex and conflicting multi-compensation (tracking, exploration, power) environments. The authors show that this instability is attributed to a "policy conflict" and present a unique and effective solution called "sequential policy update (CSE-ET) to address it.

The paper is very clearly and systematically described in three stages: 'Base Model', 'Decision Model' (MARL Definition) and 'Optimization Algorithm'. In particular, the overall framework, Figures 2 and 3, etc. greatly help the reader intuitively understand the core ideas and their effects of the proposed technology.

By setting strong and appropriate SOTA MARL algorithms such as MAPPO, HAPPO, and HATRPO as comparators, we quantitatively demonstrate the superiority (tracking success rate, power saving, etc.) of the proposed CSE-ET. This robust experimental design gives high confidence in the performance improvement results of the proposed framework.

### Weaknesses
The paper's biggest contribution is that it clearly diagnosed the instability experienced by standard MARL algorithms (MAPPOs) in complex and conflicting multi-compensation (tracking, exploration, power) environments. The authors show that this instability is attributed to a "policy conflict" and present a unique and effective solution called "sequential policy update (CSE-ET) to address it.

The paper is very clearly and systematically described in three stages: 'Base Model', 'Decision Model' (MARL Definition) and 'Optimization Algorithm'. In particular, the overall framework, Figures 2 and 3, etc. greatly help the reader intuitively understand the core ideas and their effects of the proposed technology.

By setting strong and appropriate SOTA MARL algorithms such as MAPPO, HAPPO, and HATRPO as comparators, we quantitatively demonstrate the superiority (tracking success rate, power saving, etc.) of the proposed CSE-ET. This robust experimental design gives high confidence in the performance improvement results of the proposed framework.

### Questions
The paper's biggest contribution is that it clearly diagnosed the instability experienced by standard MARL algorithms (MAPPOs) in complex and conflicting multi-compensation (tracking, exploration, power) environments. The authors show that this instability is attributed to a "policy conflict" and present a unique and effective solution called "sequential policy update (CSE-ET) to address it.

The paper is very clearly and systematically described in three stages: 'Base Model', 'Decision Model' (MARL Definition) and 'Optimization Algorithm'. In particular, the overall framework, Figures 2 and 3, etc. greatly help the reader intuitively understand the core ideas and their effects of the proposed technology.

By setting strong and appropriate SOTA MARL algorithms such as MAPPO, HAPPO, and HATRPO as comparators, we quantitatively demonstrate the superiority (tracking success rate, power saving, etc.) of the proposed CSE-ET. This robust experimental design gives high confidence in the performance improvement results of the proposed framework.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents CSE-ET, a multi-agent reinforcement learning framework for cooperative 3D multi-UAV target tracking under constraints such as communication range, inter-UAV safety separation, and limited energy. CSE-ET integrates four key modules: a tracking model, an escape (evasion) model, a U2U (UAV-to-UAV) communication model, and a power consumption model. The main contribution is a three-layer system: a Base model, a Decision model that includes a derived 3D spatial information entropy, and finally, joint optimization performed using each agent having an actor network and a common shared critic, sequential policy updates, advantage decomposition, GAE, and PopArt. To explore the tradeoff between exploration and power consumption, the method integrates a tracking–exploration–power multi-objective reward and a curriculum-style fusion strategy. The simulation experiments conducted on NVIDIA Isaac Gym compare CSE-ET to MAPPO, HAPPO, and HATRPO under several reward weightings.

### Strengths
- The key strength of this work lies in the combination of UAV dynamics, evasion modeling, communication graph Laplacians, and detailed propulsion energy models.
- The 3-D extension and closed-form derivation of spatial information entropy are valuable.
- The two-stage curriculum learning and sequential policy update scheme describes how to handle multiple objectives within a single algorithm, enabling reward accumulation and faster convergence.
- The paper is clearly written and well-organized.

### Weaknesses
- Algorithmically, CSE-ET is a slight variant of MAPPO/HAPPO that incorporates sequential updates and normalization. Its novelty lies more in system integration (realistic two-phase reward design, energy and communication modeling, curriculum training) than in core learning methodology. 
- The paper does not isolate the effect of sequential updates, reward shaping, or normalization. It remains unclear which component drives improvement. The paper would benefit from a controlled ablation study comparing pure MAPPO, MAPPO with sequential updates, MAPPO with PopArt, and full CSE-ET to isolate the effect of each modification.
- Other relevant baselines, such as graph neural networks, independent critics, or communication-aware MARL methods, have not been compared to.
- The experimental scope is narrow as it is limited to a single UAV tracking setup. Experiments are conducted in simulated open airspace, free from wind, sensor noise, communication delays, and obstacles.

### Questions
1. In what specific way does the CSE-ET algorithm diverge from MAPPO or HAPPO beyond sequential updates and normalization? Can an ablation study be performed? This would help clarify which component contributes to performance gains.
2. Are the values presented in Table 2 the mean or median of the experimental data? It could include the standard deviations to be complete. 
3. How would the performance change when evaluated in noisy environments?
4. Has the algorithm been tested on a larger number of UAVs?

### Soundness
3

### Presentation
3

### Contribution
2
