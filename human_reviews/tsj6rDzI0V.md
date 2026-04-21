# STAGE Net: Spatio-Temporal Attention-based Graph Encoding for Learning Multi-Agent Interactions in the presence of Hidden Agents

- Avg Score: 4.75
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 5

## Abstract
Accurate prediction of trajectories for multiple interacting agents following unknown dynamics is crucial in many real-world critical physical and social systems where a group of agents interact with each other, leading to intricate behavior patterns at both the individual and system levels. In many scenarios, trajectory predictions must be performed under partial observations i.e., only a subset of agents are known and observable. Consequently, we can only observe the trajectories of a subset of agents with a sampled interaction graph from a larger topological system while the behaviors of the unobserved agents and their interactions with the observed agents are not known. In this work, we propose STAGE Net, a sequential spatiotemporal attention-based generative model to learn system dynamics with multiple interacting agents where some agents are completely unobserved (hidden) all the time. Our network utilizes the spatiotemporal attention mechanism with neural inter-node messaging to capture high-level behavioral semantics of the multi-agent system. Our analytical results motivate STAGE Net design using spatiotemporal graph with time anchors to effectively model complex multi-agent interactions with unobserved agents and no prior information about interaction graph topology. We evaluate our method on multiagent simulations with spring and charged dynamics and a motion trajectory dataset. Empirical results illustrate that our method outperforms existing multiagent interaction modeling networks in predicting trajectories of complex multiagent interactions even when we have a large number of unobserved agents.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a Spatio-Temporal Graph Attention Network (called STAGE Net) to learn multi-agent dynamics where some agents are completely unobserved (hidden) all the time. The network used the spatiotemporal attention mechanism with neural inter-node messaging to capture high-level behavioral semantics of the multi-agent system. They showed analytical results motivating STAGE Net using spatiotemporal graphs with time anchors to effectively model complex multi-agent interactions with unobserved agents and no prior information about interaction graph topology. They also show the evaluation results on multi-agent simulations with spring and charged dynamics and a motion trajectory dataset. STAGE Net outperformed existing multiagent interaction modeling networks in predicting trajectories of complex multiagent interactions even when having a large number of unobserved agents.

### Strengths
- The paper developed a framework to address the problem about complex multi-agent systems with unobserved agents. 
- The STAGE Net used a dynamic spatiotemporal graph to model structural information across time using observations from visible nodes to recover knowledge representations missing due to unobserved agents.
- They performed theoretical analyses provided on why the spatio-temporal graph obtained superior representations compared to just using the visible agents' interaction graph.
- The experimental results showed that the method outperformed several baselines on multiple datasets with spring, charged, and motion trajectory dynamics.

### Weaknesses
- There have been many spatiotemporal graph attention networks in previous work (in Google scholar, 56 items), but the proposed method’s name is based on this. Can the authors reconsider the name and clarify the differences from these papers? In other words, the novelty of the methodology in STAGE Net was unclear and in the experiments, some similar networks can be compared (the baselines were old; dNRI was proposed in 2020). 
- In the experiments, the model performances were evaluated extensively on simulated physics datasets and single-agent (and multi-joint with physical constraints) CMU dataset, but real-world multi-agent trajectory datasets can be used to demonstrate applicability.
- As written in conclusion, there is no analysis provided on how the performance changes for heterogeneous agents with diverse dynamics, but this may not be a fatal problem in this paper (considered as the limitation).

### Questions
- Again, there have been many spatiotemporal graph attention networks in previous work. Can the authors reconsider the name and clarify the differences from these papers? In other words, the novelty of the methodology in STAGE Net was unclear.
- P3: may be a typo:  “the is”
- P4: subscripts of \mathcal{X} (time interval) in the third and fourth lines of the first paragraph in 2.2.2. Did they correspond with the definition of 2.2.1 and are they correct? Can the t be arbitrary and is the T_h necessary for the former? 
- P4: The definitions of the nonlinear activation function and the concatenation operation after Eq. (2) can be moved to Eq. (1). 
- Experiments (methods): again, some similar (spatiotemporal graph attention) networks can be compared (the baselines were old; dNRI was proposed in 2020).  
- Experiments (datasets): again, the model performances were evaluated extensively on simulated physics datasets and single-agent (and multi-joint with physical constraints) CMU dataset, but real-world multi-agent trajectory datasets can be used to demonstrate applicability. For example, dNRI paper used an NBA basketball dataset.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the task of multi-agent trajectory prediction in a system with a fixed number of total agents, where a consistent ratio of agents remains hidden throughout the entire prediction process. 

The paper proposes a sequential attention-based generative model that learns latent representations of observable agents with a learned temporal graph. It provides an analytical analysis of the advantages of learning representations through constructing a temporal sub-graph over a spatial sub-graph. 

This work conducts experiments on three datasets where the hidden agents are simulated by randomly hiding M out of the total N agents. It compares with two types of prior methods. One deals with the multi-agent trajectory prediction with full observability on the agents' topological graph. Another one is a latent RNN model. The proposed method outperforms the others on the three datasets with simulated hidden agents.

### Strengths
This paper presents an interesting and challenging task, and it provides both analytical analysis and comprehensive empirical experiments.

### Weaknesses
1. The paper introduces a scenario where a fixed number of agents are constantly hidden, presenting a challenging and intriguing task. However, my concern lies in its constrained nature; it seems to be a specific case within a broader context where agents may not be observable throughout specific horizons (rather than constantly unobservable as in this paper). This may limit the method's real-world applicability, potentially diminishing its overall impact.
2. The paper lacks any discussion about its connection to prior research on multi-agent trajectory prediction under partial observation, e.g., Stochastic Prediction of Multi-Agent Interactions from Partial Observations ICLR 2019. I think this paper has strong relevance to prior works on multi-agent trajectory prediction under partial observation.
3. The method of constructing a spatiotemporal graph for multi-agent trajectory prediction seems not novel.

### Questions
1. The hidden agents in the three datasets are all simulated; can the authors provide experiments on datasets where the hidden agents are not simulated or provide real-world examples where systems of a fixed number of hidden agents? The paper has already provided motivating examples where agents are partially observable on page 1 and page 23, but those are not the scenarios under the problem definition in sec 2.1. 
2. confusing notations in sec. 2.2.4 on page 4, "M is the total number of observed agents." which conflicts with the definition in sec 2.1 -where it says, "N agents could be observed." 
3. confusing colorization in Figure 1: the color for latent states z_{1}^{0}, z_{2}^{0}, z_{3}^{0} does not match with the observable nodes, and the color of z_{2}^{0}  is the same as one of the unobservable nodes o_{3}.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces StageNet, a machine learning model designed to predict the trajectory of multi-agent systems with unknown (hidden) dynamics and in the presence of unknown (hidden) agents. StageNet leverages a spatiotemporal attention mechanism with neural inter-node messaging to capture high-level behavioral semantics of the multi-agent system. The proposed framework utilizes a dynamic spatiotemporal graph attention mechanism, specifically tailored for systems where only a subset of agents is observable at any given time. The paper demonstrates the effectiveness of StageNet in learning meaningful representations for multi-agent systems, using three datasets with different dynamics and a real-world dataset of motion trajectories experiencing sensor failures.

### Strengths
1. The paper provides analytical motivation for constructing a spatiotemporal graph from visible nodes in a multi-agent system, yielding a superior representation of the entire system.
2. StageNet presents a novel approach to predicting the trajectory of multi-agent systems with unknown dynamics and hidden agents, which is a complement to the research field of multi-agent trajectory prediction, providing new insights and methodologies for future studies.
3. The paper is clearly written and easy to understand, making it accessible to a wide audience.

### Weaknesses
1. Could you provide more insight into the scalability and computational efficiency of StageNet in more complex and large-scale tasks?
2. Discussing potential issues related to the robustness of the model in the presence of noisy or incomplete data could further strengthen the paper.
3. One potential improvement to the paper could be to visualize the dynamic spatial-relational patterns in the simulated datasets. This could give a more intuitive understanding of the underlying dynamics and interactions.

### Questions
See weakness

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper deals with a trajectory prediction task with unobservable hidden objects in the system, which focuses on interaction modeling between hidden and visible agents. The authors propose STAGE Net, a sequential spatiotemporal attention-based generative model to learn system dynamics with multiple interacting agents where some agents are completely unobserved all the time. This framework utilizes a dynamic spatiotemporal graph attention mechanism, specifically tailored for systems where only a subset of agents is observable at any given time. The proposed network utilizes the spatiotemporal attention mechanism with neural inter-node messaging to capture high-level behavioral semantics of the multi-agent system. They employ a graph neural network applied to a spatiotemporal graph to approximate the initial latent posterior distribution. The proposed method was evaluated on multiagent simulations with spring and charged dynamics and a motion trajectory dataset.

### Strengths
1. The paper is generally well-written and easy to follow.

2. The problem of modeling the influence of unobservable hidden objects is interesting.

3. The experimental results seem to support the authors' claims.

### Weaknesses
1. This paper deals with unobservable hidden agents in trajectory prediction. However, it is not clear which parts of the proposed model have specific advantages in addressing this issue. Meanwhile, it would be better to elaborate on more theoretical rationale about why the proposed mechanism or model design could improve the prediction performance in addition to empirical results.

2. Given a certain set of trajectories of observable agents, there may be multiple different settings of hidden agents (e.g., different numbers, different states) that lead to the same observations of the observable agents, so the future could be multi-modal due to different potential situations. It is not clear how the proposed model handles this issue. 

3. With unknown numbers of hidden agents, the future trajectories should naturally have uncertainty or multi-modality. However, there seems no discussion regarding this.

4. Regarding the experiments with different percentages of hidden agents, it is not clear how hidden agents are determined. Was it based on random sampling? Different choices of hidden agents may significantly influence the predictability of the system. Therefore, it would be better to clearly explain the experimental setting regarding this aspect to ensure a fair comparison with baseline methods.

5. It would be better to also provide qualitative results on the motion prediction dataset as well as for the ablation study, which will be more straightforward to understand how the proposed model handles unobservable objects better.

### Questions
1. In Figure 3, should "velocity" be changed to "trajectory"?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
