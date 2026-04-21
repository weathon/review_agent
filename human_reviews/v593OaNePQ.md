# Learning to Search from Demonstration Sequences

- Avg Score: 8.00
- Decision: Accept (Oral)
- Scores: 6, 8, 8, 10

## Abstract
Search and planning are essential for solving many real-world problems. However, in numerous learning scenarios, only action-observation sequences, such as demonstrations or instruction sequences, are available for learning. Relying solely on supervised learning with these sequences can lead to sub-optimal performance due to the vast, unseen search space encountered during training. In this paper, we introduce Differentiable Tree Search Network (D-TSN), a novel neural network architecture that learns to construct search trees from just sequences of demonstrations by performing gradient descent on a best-first search tree construction algorithm. D-TSN enables the joint learning of submodules, including an encoder, value function, and world model, which are essential for planning. To construct the search tree, we employ a stochastic tree expansion policy and formulate it as another decision-making task. Then, we optimize the tree expansion policy via REINFORCE with an effective variance reduction technique for the gradient computation. D-TSN can be applied to problems with a known world model or to scenarios where it needs to jointly learn a world model with a latent state space. We study problems from these two scenarios, including Game of 24, 2D grid navigation, and Procgen games, to understand when D-TSN is more helpful. Through our experiments, we show that D-TSN is effective, especially when the world model with a latent state space is jointly learned. The code is available at https://github.com/dixantmittal/differentiable-tree-search-network.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This study presents a new differentiable tree search network comprising multiple submodules to estimate Q values. The network is optimized by minimizing the distance between the estimated Q values and the ground-truth Q values within a provided dataset. To address the substantial changes in the tree structure resulting from updates to the Q value function, a stochastic expansion policy is introduced to guide expansions in the search process. This policy ensures the continuity of the parameter space irrespective of changes in the tree structure.

### Strengths
i) This work introduces the integration of the algorithmic inductive bias from the best-first search algorithm into a differentiable network, enabling automatic guidance of the tree search process through gradient backpropagation.

ii) This work underscores the significance of maintaining continuity of both the parameter space and the loss function which are dependent on the tree structure. To address this, the authors advocate for the adoption of a stochastic expansion policy to fulfill these prerequisites. 

iii) The experiment results are compelling. And it is particularly noteworthy to see the success achieved in tasks involving LLM.

### Weaknesses
i) Previous methods have introduced diverse differential network architectures to integrate different search algorithms into networks, as mentioned in the related works. It is unsurprising that integrating the best-first algorithm into the network has also yielded success. Thus, it would be beneficial to compare the architectural variances between this method and previous methodologies.


ii) This work trains the overall network using an offline dataset. However, as extensively deliberated in preceding offline RL studies, this paradigm may get stuck when facing out-of-distribution states or actions. Thus, a comparative analysis between online training and offline training for the newly proposed network architecture could provide valuable insights.

### Questions
Do differences in the convergence rates of submodules exist? If present, could these differences impact the overall network's performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a differentiable tree search and planning network to alleviate the suboptimal search results that arise when using only action and observation sequences to learn to plan in many modern data-driven planning problems. The tree search basically synergizes submodules in the form of an encoder, value, reward and transition functions from a network by inducing the bias of a best first search into the neural architecture.

Arguing that for a slight perturbation in network parameters, the implementation of the loss function in equation (2) could generate a tree structure that causes the loss function to become noisy, the authors equated this to a lack of continuity in the loss function space. I think they mistook stochasticity in gradient propagation with discontinuity. The whole premise of the contribution of the paper is based on this assumption that is barely proven to be true or false before the authors dived into a host of mathematical analysis that resulted in the loss function on Line 272 (please number all your equations to allow for easy referencing in the future). As a result of this key oversight, it is not clear to this reviewer that the whole contribution of the paper is warranted. 

As a result, I am assigning a score of fair to this contribution until I see a well-laid out argument for why this new invention is necessary.

### Strengths
+ This paper introduces a differentiable treee seach and planning network to alleviate the suboptimal search results that arise when using only action and observation sequences to learn to plan in many modern data-driven plannning problems. The tree search basically synergizes submodules in the form of an encoder, value, reward and transition functions from a network by inducing the bias of a best first search into the neural architecture.

+ I love the motivation stated for constructing an search tree expansion policy that is stochastic in nature. But the justification for why it ensures the continuity of the loss function when the search tree is not fixed is missing. I am referring to lines 55-56.

+ I love the conceptualization and the synergy of REINFORCE, mathematical mechanisms to reduce variance in the REINFORCE Loss owing to possibly biased estimates, the continuity proof (though I have questions hanging over the proof of Lemma B.3 to be fully satisfied with this poposition) .

+ I love that the conclusion section meticulously summarizes the problem, contributions, and shortcomings. Kudos to the authors.

### Weaknesses
While I do love the mathematical contributions of the paper, I think there are essentials that are missing in the logic, organization, and flow of arguments that require a thorough review before this paper makes it into an acceptance. A principal one is the following (mentioned in the summary box but repeated here. The authors would do well to alleviate my concerns): Arguing that for a slight perturbation in network parameters, the implementation of the loss function in equation (2) could generate a tree structure that causes the loss function to become noisy, the authors equated this to a lack of continuity in the loss function space. I think they mistook stochasticity in gradient propagation with discontinuity. The whole premise of the contribution of the paper is based on this assumption that is barely proven to be true or false before the authors dived into a host of mathematical analysis that resulted in the loss function on Line 272 (please number all your equations to allow for easy referencing in the future). 

+ The claim in the last paragraph of Theorem 3.1 that slight changes in the network parameters could cause discontinuity in the tree structure seems anecdotal and not backed up by a solid proof. I would love to see a concrete reasoning (analytical proof or abundant empirical  proofs) behind this claim that warrants section 3.5 and Appendix C.

+ Grammatical errors fly out of the page hither and yon throughout the paper; the uthors would do well to carefully organize their arguments, present their logic convincingly throughout the paper, and punctuate and label every equation appropriately!

+ The logic in the paper could use a more thoughful presentation. Here is an example critique:
     - In the "introduction", it is stated in the first paragraph that constructing a search tree from expert data is infeasible due to lack of practicality or scarcity. The authors make an assumption that a search tree is a principal prerequisite for information retrieval (IR) without any justification as to why it may be better than alternative IR methods. Then in the second paragraph, they mentioned how search and planning could be better executed in the presence of a simulator or world mode. While I find this premise alluring, I find it disingenious that the authors claimed that the search could be incomplete because the search process may visit regions unexplored during training. I think the reasoning here is incomplete and should be revisited by the authors.

### Questions
+ How is the full expansion mechanism in this work not a shallow tree (in comparison to the TreeQN) that you mentioned in the RELATED WORKs section since you mentioned that you have a fixed depth in your search protocol?

+ When the encoder, value, reward, and transition functions are unavailable, how do they get jointly optimized with the search algorithms?

+ In paragraph III, the justification for why end-to-end learning improves the reliability of world-models used in preconditioning the search process is not given. Please address this.

+ Lines 124-126: "Dividing the network into these submodules reduces
total learnable parameters and injects a strong search inductive bias into the network architecture, preventing overfitting to an arbitrary function that may align with the limited available training data." Where is the proof for this?

+ Is there no way the illustration in Figure 2 can be moved to the main text from the Appendix? Probably as a subfigure on page one or so would be really nice! Also, I think the algorithm should justifiably be compressed (e.g. as a pseudocode or flow chart) somewhere in the main text. An elaborate version can be embedded in the Appendix if space is an issue.


+ What metric are you using to report the measured quantities in Table 1? Can you include it in the heading/sub-headings?


+ Appendix B, Lemma B.3: I'm sorry, is there supposed to be a running sum over all of x in the equation you wrote? 

+ Line 915, I think you meant Lemma B.3, not Theorem B.3, no?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a novel differentiable neural tree search architecture that can be learned directly from trajectories of data in a supervised manner. It essentially introduces a search-like inductive bias within the weights of the neural network. The proposed algorithm builds on TreeQN (Farquhar et al., 2018), and is crucially different since the proposed method allows building a tree structure that can stochastically sample from the action space, and not just be a fixed tree structure as in TreeQN. The authors empirically evaluate the strength of their approach by comparing against a Model-free Q network, a Model-based search method that trains the individual modules separately, and TreeQN, and report performance gains in various RL environments.

### Strengths
- Builds on TreeQN and improves a significant limitation of the prior work, i.e., having a fixed tree structure. In reality, search algorithms should attempt to filter large action spaces and focus computation on promising variations in the tree. The proposed work gets around this limitation by sampling from the action space, and using REINFORCE to differentiate through the discontinuity of sampling.
- Strong empirical evidence that the proposed method improves on TreeQN, and having the modules trained separately.
- Strong ablation results showing the effectiveness of the proposed method.

### Weaknesses
- Seeing how the approach handles large action spaces remains an empirical question, since currently, there is no policy that directly outputs a distribution over actions, instead the method requires the application of the transition network and the reward network for every action, which is not scalable to settings like, say, Go.
- The proposed method is mostly applicable to discrete action spaces with deterministic environments. Improving on this remains a future empirical question.

### Questions
- Why did the authors choose to use REINFORCE? Did the authors experiment with using methods like Gumbel-Softmax for sampling?
- How are the compute budgets between model-based search, model-free Q network and D-TSN equalized for a fair comparison?
- In Line 22, the authors say they “introduce an effective variance reduction technique”. Isn’t this just Guez et al. (2018)? The way this is written in some places suggests that this is novel.
- Line 45,46 seems contradictory with lines 48,49. If learning from demonstration sequences fails due to compounding errors due to the agent getting into states not during training, then the training distribution of the proposed method is also not sufficiently covered by the training distribution. I understand the CQL term attempts to address this issue.
- What are the scores of the PPG sub-optimal policy?
- Table 3 reports Mean Scores and Mean Z-Scores, but no standard deviations or error bars?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces the Differentiable Tree Search Network (D-TSN), a novel neural network architecture designed to learn search strategies from sequences of demonstrations without access to explicit search trees. D-TSN integrates the inductive bias of a best-first search algorithm into its structure, enabling the joint learning of essential planning submodules, including an encoder, value function, and world model. To construct the search tree, the authors employ a stochastic tree expansion policy, formulating it as a decision-making task optimized via the REINFORCE algorithm. They introduce a variance reduction technique using a telescoping sum to address high variance in gradient estimates. D-TSN is applicable in scenarios where the world model is known or needs to be jointly learned with a latent state space. The authors evaluate D-TSN on tasks from both scenarios, including the Game of 24, a 2D grid navigation task, and Procgen games. Experimental results demonstrate that D-TSN is effective, particularly when the world model with a latent state space is jointly learned, outperforming baselines in terms of success rate and generalization capabilities.

### Strengths
1. Novel Architecture: The paper proposes a novel neural network architecture, D-TSN, which embeds the inductive bias of a best-first search algorithm, allowing for the end-to-end learning of planning components from demonstration sequences. 

2. Joint Learning of Planning Components: D-TSN jointly learns the encoder, value function, and world model. This is advantageous when the world model is not given but is needed to be learned from data. 

3. Variance Reduction Technique: Authors use an effective variance reduction technique using a telescoping sum in the REINFORCE algorithm to addresses the high variance associated with policy gradient methods. 

4. Comprehensive Experiments: The method is applied to a wide variety of tasks, such as reasoning problems, navigation, and game environments, supporting the claim that it is versatile and effective across domains. 

5. Improved Performance: The authors show that D-TSN outperforms baselines, showing its problem solving performance in challenging tasks with limited supervision, especially in jointly learned world model settings.

### Weaknesses
1. Limited to Deterministic Environments: The current implementation is restricted to deterministic decision-making problems with discrete action spaces.

2. Computational Complexity: The computational complexity for the approach  might be high, because it consists of constructing search trees and performing REINFORCE updates. This can be a problem especially when applying for deeper trees or larger action spaces.

3. Scalability: scalability is not thoroughly analyzed for longe-horizon tasks or higher dimensional state space.

### Questions
1. How would you extend D-TSNs to use for solving stochastic decision making problems? What modification would be required in the current work to accommodate stochastic transitions and rewards?

2. Could you provide more details on how computationally expensive D-TSN is and how does the method scale w.r.t. the action space size and the search tree depth?

### Soundness
4

### Presentation
4

### Contribution
3
