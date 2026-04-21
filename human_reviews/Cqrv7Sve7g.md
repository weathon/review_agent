# Offline Reward Inference on Graph: A New Thinking

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 6, 6

## Abstract
In offline reinforcement learning, reward inference is the key to learning effective policies in practical scenarios. Due to the expensive or unethical nature of environmental interactions in domains such as healthcare and robotics, reward functions are rarely accessible, and the task of inferring rewards becomes challenging. To address this issue, our research focuses on developing a reward inference method that capitalizes on a constrained number of human reward annotations to infer rewards for unlabelled data. Initially, we leverage both the available data and limited reward annotations to construct a reward propagation graph, wherein the edge weights incorporate various influential factors pertaining to the rewards. Subsequently, we employ the constructed graph for transductive reward inference, thereby estimating rewards for unlabelled data. Furthermore, we establish the existence of a fixed point during several iterations of the transductive inference process and demonstrate its at least convergence to a local optimum. Empirical evaluations on locomotion and robotic manipulation tasks substantiate the efficacy of our approach, wherein the utilization of our inferred rewards yields substantial performance enhancements within the offline reinforcement learning framework, particularly when confronted with limited reward annotations.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposed a graph based method that infers the rewards of unlabelled state-action pairs for offline RL tasks. The method is tested on several robotic environments and the empirical results show that the proposed algorithm outperforms the existing methods.

### Strengths
The paper has clear presentation and the overall idea is easy to understand. The paper performs sufficient empirical experiments and the results are convincing.

### Weaknesses
1. The paper doesn't present complete details of the experiment setup. While $f_\Theta$ is an important component for constructing the graph weights, throughout the paper no details on the setup of $f_\Theta$. The paper also doesn't reveal any details about the policy formulation or any parameters related with the training process. 

2. While the paper does sufficient comparison over several existing methods, one additional thing that might be worth presenting is comparison with different reward inference methods. For example, one might want compare the paper's method with reward inference simply by KNN.

### Questions
1. One potential limit of the method is the size of the weight matrix grows with the number of data points. In this case, it could be very costly to compute the matrix inverse or other related things. Can the author give some comments on the computation cost?

2. An alternative way to infer the reward is to simply use the known part of the reward, i.e., $R_U = W_{UL}R_L$. What's the advantage of the proposed method, compared with this simpler formulation?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers the offline RL setting where reward labels are scarce. In possible application of offline RL, such labels are difficulty/unethical to obtain (such as in healthcare), therefore we need to find ways to maximize the knowledge contained in the available reward labels. The authors propose to construct a graph out of the available data and perform reward inference on nodes for which no reward is available. The method is evaluated on continuous control problems, such as DeepMind Control and Meta World.

### Strengths
* The method proposes an intuitive solution to the problem of reward scarcity in the offline RL setting
* The presentation of the paper is good and understandable
* The method is evaluated across a wide range of environments

Learning efficiently in offline RL necessarily depends on having reward information for each state action pair. When that is not the case practitioners have to find ways to infer the missing rewards labels. The paper proposes a way to infer such labels by considering the underlying geometry of the problem together with considerations with respect to different factors that influence the reward.

### Weaknesses
* The baselines used are not well presented. Closely related work is not compared to
* The empirical evaluation relies on only 5 seeds. Moreover, the presentation of the results does not consider statistical significance
* No investigation is done to understand the impact of different choices in the algorithm

The TGR and UDS are presented too succinctly. If the reader is not familiar with these methods it is impossible to understand how they relate. For example "UDS relabels unlabeled data with zero rewards"  seems like a very strange way to deal with unlabeled data - it essentially doesn't deal with it. More details would be needed.

The paper proposes to "learn a reward propagation graph and infer rewards", yet a quick google search on "reward propagation" reveals a paper from 2020 that proposes a similar method to infer rewards on nodes with missing reward labels [1]. Although their method is for the online RL setting, it seems straightforward to use it in the offline RL setting. This would be an important baseline.

The empirical evaluation is not adequate from a scientific rigour point-of-view. Only 5 seeds are used and Table 1 highlights the score of methods for which the standard deviation intersects with other methods. This is misleading, only non-intersecting standard deviations should be highlighted. Perhaps additional seeds could help in this. 

Very little work is done to understand the impact of different components in the method. For example, what is the impact of using a reward propagation graph? Why not simply infer the reward through a classification/regression loss that takes as input only the current state. This is another important baseline. Finally what is the impact the different reward factors?

### Questions
Section 4.4 "we" typo -> it should be capitalized

"we regard each part of the state and action as a factor that influences the reward" Doesn't this add more prior knowledge into the problem? What happens if we don't know the factors?

Figure 2 "prpagation" typo

"Cabi et al. (2020) is hard to sketch tasks" perhaps this needs some rephrasing

"Konyushkova et al. (2020) assumes rewards are binary, not adaptation to any value reward learning question." This could perhaps also use some rephrasing.


================================================

[1] Reward Propagation Using Graph Convolutional Networks. Klissarov and Precup. 2020

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This study introduces a reward inference technique designed for offline Reinforcement Learning (RL) scenarios in which only a subset of the available data are annotated with reward information. This approach leverages a graph-based framework to extend reward inference to unannotated data points. In this approach, a neural network is trained to predict the weights between nodes in the graph, facilitating the propagation of reward information. Empirical evaluations demonstrate that this approach yields superior reward quality compared to existing baseline methods.

### Strengths
* This paper offers a theoretical foundation by providing guarantees that the proposed reward inference method can converge to at least a local optimum, which enhances its credibility and applicability.

* The manuscript is well-written and presents the research in a clear and easily understandable manner.

* The problem addressed in this work is importance, as it tackles a critical challenge in offline reinforcement learning when applying to real world situation, enabling its practical application in situations where traditional methods might fall short.

### Weaknesses
See Questions

### Questions
* In the paper, the reward shaping function $f_{\theta}$ is trained using annotated data to convert state-action pairs into scalar values, and this information is used to calculate the weights between nodes in the graph. Since $f_{\theta}$ must learn the relative importance of each dimension in the node data, it raises the question of the necessary diversity within the annotated data. For instance, if all the annotated data consists of expert transitions with the highest rewards, it might be challenging for the function to discern which dimensions of the node data contribute most significantly to changes in reward. Hence, an important question to address is: "What level of diversity within the annotated data is required for the proposed method to learn an effective function $f_{\theta$?"
* Is the reward calculated by the graph method bounded by the range of rewards in the annotated data, i.e., $\max{(R_U)}\leq \max{(R_L)}$? This question seeks to understand the extent to which the graph-based reward inference can maintain the upper bound of reward values, as derived from the annotated data.
* To make the proposed method practically feasible, it's essential to investigate the computational cost requirements. Specifically, showing the computational time and memory consumption required for reward inference helps provide insights into the practical applicability and efficiency of the method in real-world scenarios.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good
