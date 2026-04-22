# EECE: Ensemble-based Epistemic and Cooperative Exploration for Multi-Agent Reinforcement Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Efficient exploration in multi-agent reinforcement learning (MARL) remains a fundamental challenge, particularly in complex cooperative tasks with sparse rewards. 
In MARL, agents must discover both novel and strongly cooperative state–action pairs in high-dimensional state–action space to effectively facilitate policy learning. 
In this paper, we propose Ensemble-based Epistemic and Cooperative Exploration (EECE), a unified framework that leverages an ensemble dynamics model to simultaneously capture epistemic uncertainty for directed exploration and the level of cooperation required for coordinated behavior discovery. 
To achieve this, EECE introduces two information-theoretic intrinsic rewards: (i) an epistemic information gain signal that directs agents toward transitions with high uncertainty, and (ii) a cooperative signal that maximizes the aggregated marginal influence of individual agents on global state variation, quantified via mutual information. It then employs a dynamic weighting strategy to leverage the complementary effects of intrinsic rewards during training. Moreover, it incorporates a dual-policy mechanism that stabilizes exploration and avoids introducing additional non-stationarity and credit assignment issues. 
We demonstrate the advantages of our method through cooperative benchmarks with sparse rewards, including the StarCraft Multi-Agent Challenge (SMAC) and Google Research Football (GRF), showing that EECE achieves substantial improvements in both exploration efficiency and final performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces a new framework called Ensemble-based Epistemic and Cooperative Exploration (EECE), designed to solve a key challenge in multi-agent reinforcement learning (MARL): efficient exploration in complex, cooperative tasks with sparse rewards. EECE uses an ensemble dynamics model to simultaneously encourage both. It creates two distinct information-theoretic intrinsic rewards, including an epistemic signal that rewards agents for exploring transitions with high uncertainty and a cooperative signal that rewards agents for working together to create changes in the global state, quantified using mutual information. EECE achieves some improvements in both exploration efficiency and final performance.

### Strengths
1. The authors propose EECE, a single, unified framework that addresses two critical and often separate challenges in MARL: encouraging effective exploration and promoting inter-agent cooperation.

2. EECE uses information gain to quantify uncertainty, guiding agents toward truly novel states (directional exploration) rather than just random exploration. EECE Uses mutual information to measure each agent's influence on the global state. This reward is aggregated to encourage collaborative, proactive behaviors. Both the epistemic and cooperative rewards are derived directly from the same ensemble of dynamics models. 

3. This work develops a novel dual-policy mechanism where exploration and exploitation policies are learned independently. This helps stabilize the learning process by mitigating non-stationarity and credit assignment problems that intrinsic rewards can often cause.

### Weaknesses
1. The authors claim that previous methods encouraging either diversity or cooperation often fail to provide reliable exploration signals in high-dimensional state-action spaces and lack a unified treatment of both aspects, which limits their overall effectiveness. However, there is no theoretical or empirical evidence provided to substantiate the authors' claims. I suggest that the authors add more discussion on the limitations they claim to strengthen the contributions of this work.

2. EECE needs a reliable model of environment dynamics to support epistemic and cooperative exploration. However, the dynamics of high-dimensional multi-agent systems, especially with a large number of agents, are highly complicated. The convergence of the learning of the dynamic model should be addressed. Moreover, an ensemble prediction may require training $K$ deep neural networks. The computational complexity can be high. The discussion of the effects of different values of $K$ on the performance of agents is also missing.

3. The definition of cooperation as "the coordinated effort of all agents to collectively induce variations in the global state, with the actions of individual agents contributing to these changes," is confusing. These variations cannot be referred to as cooperation because non-cooperative behaviors among agents can also lead to variations. I suggest the authors reconsider the concept of cooperative exploration proposed in this work.

4. The dual-policy strategy seems to significantly improve the performance. However, if EECE uses the previous combining methods, the ablation results show limited performance improvement, which demonstrates the high dependency between dual-policy learning and the two proposed exploration methods. Why does the necessary dual-policy strategy improve the proposed exploration methods?

### Questions
Please see the weaknesses above.

### Soundness
3

### Presentation
3

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
The paper proposes Ensemble-Based Epistemic and Cooperative Exploration (EECE), a framework that promotes
both exploration and cooperation in sparse-reward, partially observable environments. It combines information
gain and conditional mutual information based intrinsic rewards through a dynamic weighting scheme, and trains
two policies (exploration and exploitation) per agent. The former learns from intrinsic rewards, the latter extrinsic
rewards. Agents actions are sampled from the two policies through a dual-policy mechanism. 

The paper makes two main contributions. Firstly, the authors propose a solution to address the lack of a method
that accomplishes both diverse exploration and cooperation amongst agents. Secondly, they point out the
detriments of naively combining intrinsic and extrinsic rewards and introduce a novel dual-policy scheme that
effectively utilises these rewards separately. They also highlight the effectiveness of using information theoretics,
through their construction of the to promote coordination.

### Strengths
The application of the ensemble-based exploration method to the MARL domain and introduction of the 
cooperation intrinsic reward constructed using conditional mutual information is a nice idea. 

In contrast to existing methods that combine intrinsic and extrinsic rewards using weights, the authors used these rewards separately to train two policies per agent. The motivations for this framework, basis for each EECE component, and its implementation are
well communicated through clear writing and effective diagrams. Enough experimental details are provided for
reproducibility and the workings of the algorithm are clearly documented. 

The claim that EECE promotes both diverse exploration and cooperation is mostly well supported. This is
demonstrated through the superior performance of EECE compared to state-of-the-art baselines. The
contributions of each key component of EECE is also quantified through rigorous ablation studies, demonstrating
soundness in the adopted principles for component construction. Considering these aspects along with clarity
and significance, this paper is of good quality.

### Weaknesses
The authors justify their choice of using the ensemble to predict state variance instead of $s_{t+1}$ by stating that
epistemic uncertainty is preserved as the expected variance is unaffected. Either than this note, no further
justifications or proofs were provided. Furthermore, although the provided figures visualising exploration policy
does demonstrate emergent cooperative behavior, no other results were included. Given the claims that EECE
promotes cooperation, perhaps this is insufficient. 

The authors outlined that the dynamic weighting strategy adopted shifts focus from epistemic exploration to coordination over time but don't provide justification as to why this is effective. 

* It is not clear why the authors chose to compare EECE with QMIX in section 5.3 to demonstrate that EECE promotes diverse exploration. 
 
* Some highly relevant works aren't discussed or compared against e.g. [1].

* It would be beneficial to include motivations for the design and purpose of the adopted dynamic weighting
strategy. 

* Although the effects of varying the parameter has been investigated, no information was provided
regarding how this hyper parameter should be tuned. Provision of some guidance regarding this or its
inclusion for future work would improve the paper. 

* Relying solely on visualisations of agents’ exploration policies might be insufficient to prove that EECE
promotes coordination. Inclusion of qualitative metrics to complement these visualisations, such as the
measured mutual information between agents’ actions, would be more convincing. 

[1] Ensemble Value Functions for Efficient Exploration in Multi-Agent Reinforcement Learning. Schäfer et al. AAMAS 2025.

### Questions
1. Can you justify why predicting state variance has no effect on the expected variance observed from predicting the next state itself? 

2. Can you explain why you chose to compare EECE with QMIX to demonstrate that EECE promotes
exploratory behaviours?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces EECE, a framework for exploration in MARL under sparse rewards. The method leverages an ensemble of dynamics models to generate two information-theoretic intrinsic rewards: an epistemic reward for novelty-seeking and a cooperative reward for promoting coordinated actions. These rewards are integrated via a dynamic weighting strategy and a dual-policy mechanism, leading to performance gains over existing methods on challenging SMAC and GRF benchmarks.

### Strengths
1. The proposed method is well-designed and logically constructed. Each component is thoughtfully designed to contribute to the goal of encouraging both diverse and cooperative exploration.
2. The empirical evaluation is quite good. The authors have conducted thorough experiments on challenging and appropriate benchmarks (SMAC and GRF), and the comprehensive ablation studies effectively demonstrate the contribution of each part of the EECE framework. The qualitative analyses and visualizations are also very helpful in providing insight into the learned behaviors.

### Weaknesses
1. The exploration in MARL is a research topic that has been investigated by many previous research work. The novelty of this paper seems marginal. Specifically, 
    - The paper would benefit from a more precise definition of the research gap it addresses. The stated goal of "enhancing diverse exploration and inter-agent cooperation" is a central theme in many prior works, including those cited. To better distinguish this work, it would be helpful to clarify paper's unique contribution.
    - Regarding the technical contributions, the authors have integrated several powerful techniques. However, these individual components—ensemble-based uncertainty estimation [1-4], information-theoretic rewards for cooperation [5], and dual-policy training architectures [6-7]—are established concepts in the literature. Consequently, the paper's primary contribution appears to be the specific synthesis of these ideas.
2. The literature review could be made more comprehensive by including and discussing several closely related works, such as [8-9].

Improvement Suggestions (less important)
1. The authors acknowledged the increased computational time in the main text and reported the increased training time in appendix H. From my opinion, the increased computational time to trade off the enhanced performance is acceptable. For the sake of completeness, it would be better to also include the test-time latency. 
   

[1] "Constrained Ensemble Exploration for Unsupervised Skill Discovery", ICML 2024 \
[2] "SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning", ICML 2021 \
[3] "On the Importance of Exploration for Generalization in Reinforcement Learning", NeurIPS 2023 \
[4] "Ensemble Value Functions for Efficient Exploration in Multi-Agent Reinforcement Learning", AAMAS 2025 \
[5] "Influence-based multi-agent exploration", ICLR 2020 \
[6] "Individual Contributions as Intrinsic Exploration Scaffolds for Multi-agent Reinforcement Learning", ICML 2024 \
[7] "Cooperative exploration for multi-agent deep reinforcement learning", ICML 2021 \
[8] "Lazy agents: a new perspective on solving sparse reward problem in multi-agent reinforcement learning", ICML 2023  \
[9] "An adaptive entropy-regularization framework for multi-agent reinforcement learning", ICML 2019

### Questions
1. Could the authors please clarify the specific research gap this paper aims to address, beyond the general goal of combining diverse and cooperative exploration? What specific shortcomings in prior methods does EECE resolve?
2. In light of the prior work mentioned ([1-7]), could the authors elaborate on the primary technical novelty of EECE? Why is this a meaningful step forward compared to simply combining existing techniques?
3. (less important) Could the authors expand the related work section to more explicitly differentiate EECE from methods like [5-7] and discuss its relationship to other relevant works like [8-9]?
4. (less important) The appendix provides training time, which is helpful. For completeness, could the authors also report the test-time latency/computational overhead compared to baselines? This is particularly relevant given the use of an ensemble model.
5. (less important) The results on the selected SMAC maps are quite good, with EECE achieving near-100% win rates in many cases. Have the authors considered evaluating EECE on more challenging, modern benchmarks like SMACv2 to further demonstrate its robustness and scalability?

### Soundness
4

### Presentation
3

### Contribution
1
