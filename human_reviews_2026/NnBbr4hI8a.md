# Look-ahead Reasoning with a Learned Model in Imperfect Information Games

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Test-time reasoning significantly enhances pre-trained AI agents’ performance. However, it requires an explicit environment model, often unavailable or overly complex in real-world scenarios. While MuZero enables effective model learning for search in perfect information games, extending this paradigm to imperfect information games presents substantial challenges due to more nuanced look-ahead reasoning techniques and large number of states relevant for individual decisions. This paper introduces an algorithm LAMIR that learns an abstracted model of an imperfect information game directly from the agent-environment interaction. During test time, this trained model is used to perform look-ahead reasoning. The learned abstraction limits the size of each subgame to a manageable size, making theoretically principled look-ahead reasoning tractable even in games where previous methods could not scale. We empirically demonstrate that with sufficient
capacity, LAMIR learns the exact underlying game structure, and with limited capacity, it still learns a valuable abstraction, which improves game playing performance of the pre-trained agents even in large games.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors proposed a novel algorithm that enables subgame search in imperfect-information games with a large number of public states. Specifically, they trained a clustering network to group similar information sets and solved the resulting abstracted subgames using CFR+. In addition, to constrain the size of the subgame, they trained a value network to predict state values, enabling depth-limited subgame search.

### Strengths
- The performance of the algorithm looks good. It outperforms RNaD consistently.
- The writing of the paper is clear.
- The paper enables subgame search in games with a large public states.

### Weaknesses
- The techniques presented in this paper are not particularly novel. It appears to combine several existing approaches -- subgame search, game modeling, and abstraction. That said, integrating all of these components simultaneously is a non-trivial task.
- The games evaluated in this paper do not seem to require model learning, as used in MuZero. I would prefer to see the method tested on games where developing an accurate simulator is intractable.

### Questions
See Weaknesses.

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
4

### Summary
This paper introduces LAMIR (Learned Abstract Model for Imperfect-information Reasoning), an algorithm designed to enable look-ahead reasoning in large-scale imperfect information games by an abstract model, which can directly learn from experience without any domain-specific knowledge. LAMIR achieves up to an 80% win rate in large imperfect information games compared to RNaD, and also shows robustness in some smaller games.

### Strengths
1. This paper provides a novel approach that combines model learning, abstraction techniques, and value function learning to address scalability challenges in large-scale imperfect-information games.

2. The comparison to the baseline, RNaD, is persuasive, as it was previously used in DeepNash in Stratego.

3. The experimental figures and methodology descriptions are clear and well-structured, providing a comprehensive understanding.

4. The future work proposed by the authors is sound. For instance, integrating abstract action spaces could further enhance the applicability of LAMIR to various types of games, particularly those with large or continuous action spaces.

### Weaknesses
1. The description in Section 5 is insufficient. In particular, the explanations of the training and testing processes should be clarified, as they are somewhat difficult to follow in the current version. This is especially important because Section 5 presents the overall algorithm that integrates all components discussed in the previous sections. Including a flowchart or pseudocode would help readers better understand the steps. 

2. Section 4 uses a significant amount of space to describe the clustering process and explains how to train the clustering of the public state. However, it seems that this approach is applied only to the root node of the test-time search tree. The tree search process requires a more detailed description, including how each component introduced in previous sections is used.

3. Since LAMIR requires clustering during training, the overall training time is expected to increase. The paper should clearly report the additional training cost of LAMIR and compare it with previous approaches, such as RNaD. It would also be helpful to add an additional experiment discussing how the computational cost increases as the number of clusters $L$ increases, as well as how much data is required to train both the clustering and abstraction processes.

4. As mentioned in Section 7, the proposed approach seems to be constrained by the limitations of the CFR algorithm, where it is not applicable to many other games like Dark Chess and Stratego. However, the abstraction method itself should not be affected by this. Have there been any attempts to apply the core ideas of abstraction to these games?

5. The experiments are conducted on relatively simple imperfect information games. Are there any experiments on more complex games, such as poker? Including such results would make the work more convincing by better demonstrating the generality and effectiveness of LAMIR.

### Questions
Please address each concern raised in the weaknesses.

### Regarding the training process:
1. How is $\Lambda_{i,\theta}$ (the green component in Figure 1) specifically trained? Is it implemented using a clustering algorithm, or is it also modeled as a neural network? If it is a neural network, what are its input and output representations? Does it use the clustering result as the training target? It seems that the training loss in equations (2) and (3) is related only to $\Lambda^{I}_{i,\theta}$.

2. In equation (2), during training, does it require enumerating all $\overline{s^t_i}$? Or is it simply using sampling? If it is sampling, how many samples are required? Is this sampling process required to be performed for every public state? What is the computational cost of this sampling process? If the number of samples is too small, is the clustering into $L$ groups still effective? How is the value of $L$ determined in practice? Does the number of samples increase as $L$ increases? If so, how does the computational cost scale with $L$? Overall, what is the additional computational cost of training LAMIR compared to RNaD?

3. What kind of training data is used? Is it based on a pre-defined dataset or trajectories obtained from its self-play? 

### Regarding test time:
1. Section 5 presents the overall algorithm that integrates all components discussed in the previous sections. However, the description in Section 5 is unclear, particularly regarding how the components are used in the tree search. It would be helpful to include a figure illustrating the test-time architecture, along with a more detailed explanation of each component.

2. During the search tree, is it still required to enumerate all $\overline{s^t_i}$ or sample $\overline{s^t_i}$ to obtain the latent abstract information sets?

3. It seems that $\Lambda_{i,\theta}^{I}$ is used only at the root node of the game tree, where a specific abstraction set is chosen. However, this abstraction set only represents a single category. After performing a sequence of actions during the search tree, can we guarantee that this category still adequately covers all possible scenarios at the leaf nodes? Or is $\Lambda_{i,\theta}^{I}$ repeatedly used during the search at each internal node?

4. The inference time is not reported. Currently, the depth is set to 1. Why not use a larger depth? It would be helpful to include an ablation study analyzing different depth settings, along with a discussion of the additional computational cost and potential performance gains as the depth increases.

### Regarding the experiments:

1. How are the different bases for clustering ($\kappa$) implemented?
2. Regarding training and test time, are LAMIR with different $L$ values and RNaD all trained under the same training time limits?
3 . Based on the results in Table 1, why does using Legal Actions for clustering lead to a higher win rate than the RNaD strategy in the Goofspiel 13 and Goofspiel 15 environments? Additionally, would it be fair to use the RNaD strategy for clustering within LAMIR and then compare it directly with RNaD itself, given that LAMIR would already know the RNaD strategy?

### Other questions and suggestions
1. The reference formatting in the main text is incorrect. Most citations are missing brackets.
2. How are the public state and the player's information set generated during training, especially in large-scale games? Additionally, what is the approximate number of public states in the "large games" used in the experiments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose abstract look-ahead search at test time for imperfect-information games. While training RNaD policy, they learn a world model over latent information sets. They then learn a per-public-state abstraction that clusters many real infosets into latent prototypes, yielding a depth-limited latent subgame that's smaller than the original game or world model. At the depth limit they attach a policy-transformation leaf gadget (from Kubíček et al., 2024) and solve the subgame with CFR. Test-time look-ahead achieves lower exploitability than vanilla RNaD in small games and beats RNaD in larger games.

### Strengths
- This paper makes important and large contributions towards using latent-representation world models effectively in a 2p0s imperfect information self-play setting.
- The proposed method is relatively straightforward.
- The experimental results are sufficient and convincing.
- The limitation to games without chance and lack of convergence grantees for games without A-loss recall are well explained.

### Weaknesses
- The proposed feature encoding for latent infostate clustering is a heuristic.
- Code is not provided.
- Minimal ablations on the world model architecture design.
- Missing related works: TD-MPC and TD-MPC2

### Questions
a) This method seems like it would be extendable to other self-play methods like magnetic mirror descent. Would you expect clear integration issues or challenges in using this lookahead method with population-based methods like PSRO?

b) Is any regularization or activation applied to shape or constrain the space of latent infostates $\bar{S}_i$?

c) Without any loss of novelty, the base world model dynamics and encoder model seem mechanically very similar to the one used in TD-MPC [1] and TD-MPC2 [2] (A key distinction is that LAMIR operates over information sets under partial observability.)
 Would the world model stability improvements introduced in TD-MPC2 potentially be applicable to this work?
e.g. 
1. Only training the initial state encoding $\Lambda_\theta^I(s_i^{t})$ using dynamics loss and stopping the gradients through dynamics targets $\Lambda_\theta^I(s_i^{t+k})$
2. Bounding and regularizing the latent infostate space with SimNorm

[1] Hansen, N., Wang, X., & Su, H. (2022). Temporal Difference Learning for Model Predictive Control.ICML 2022 (PMLR).

[2] Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control.ICLR 2024.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents a new method for building an agent that plays two-player zero-sum imperfect-information games, using a lookahead search technique similar to MuZero.

The main Dynamics function maps a joint latent infostate and a joint action to a new joint latent infostate. It is learned through sampled trajectories via a simulator. It can then be used at test-time with depth-limited solving to play well.

Additionally, the method uses an abstract model to deal with large public states.

The paper includes experiments computing exact exploitability in small Goofspiel and Oshi-Zumo games, and head-to-heads results against RNaD in large Goofspiel.

### Strengths
The research direction and idea are sensible. The method is explained well. The experiments are solid.

### Weaknesses
I don't quite understand the Abstract Model (see questions).

I think more could be said about the motivation/application of this work. It's practically useful only if we (A) don't already have a perfect simulator but (B) have access to trajectories that include both players' infostates and actions. When could this be useful?

### Questions
1. Apologies if I missed it in the paper, but is it true that this method only works on games with no chance?
2. While playing the game, is it possible that the root public state of the game tree contains more than $L$ information states? 
3. I'm a bit confused on Section 4, especially what it is used for. I think Section 4 describes how to train $\Lambda_{i,\theta}$, $\kappa_{\theta}$, and $\Lambda^I_{i, \theta}$.

However, the description of the second loss says that it updates $\Lambda^I_\theta$. Is this a typo? Should it be $\Lambda^I_{i, \theta}$? But the equation describing the second loss only includes $\Lambda_{i, \theta}. 

In the description of LAMIR at the end of Section 5, these are only used to map the real current infoset $s_i$ to the abstracted one $\bar{s_i}$. Is this all it is used for? And why can't we just use $\Lambda^I_\theta$ to do this? 
4. In Section 4, why do we need to map latent infostates and infostates to a K-dimensional space? Why can't we just do everything in the latent space?

### Soundness
3

### Presentation
3

### Contribution
3
