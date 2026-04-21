# Stochastic Unrolled Federated Learning

- Avg Score: 5.67
- Decision: Reject
- Scores: 6, 6, 5

## Abstract
Algorithm unrolling has emerged as a learning-based optimization paradigm that unfolds truncated iterative algorithms in trainable neural-network optimizers. We introduce Stochastic UnRolled Federated learning (SURF), a method that expands algorithm unrolling to a federated learning scenario. Our proposed method tackles two challenges of this expansion, namely the need to feed whole datasets to the unrolled optimizers to find a descent direction and the decentralized nature of federated learning. We circumvent the former challenge by feeding stochastic mini-batches to each unrolled layer and imposing descent constraints to mitigate the randomness induced by using mini-batches. We address the latter challenge by unfolding the distributed gradient descent (DGD) algorithm in a graph neural network (GNN)-based unrolled architecture, which preserves the decentralized nature of training in federated learning. We theoretically prove that our proposed unrolled optimizer converges to a near-optimal region infinitely often. Through extensive numerical experiments, we also demonstrate the effectiveness of the proposed framework in collaborative training of image classifiers.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces Stochastic UnRolled Federated learning (SURF), a novel approach that applies algorithm unrolling, a learning-based optimization paradigm, to the server-free federated learning scenario. The authors aim to leverage these benefits to address the challenges faced by low-end devices in collaborative deep model training. The paper identifies two main challenges in applying algorithm unrolling to federated learning: the necessity of feeding whole datasets to unrolled optimizers and the decentralized nature of federated learning. The authors propose solutions to these challenges by introducing stochastic mini-batches and a graph neural network (GNN)-based unrolled architecture, respectively. The stochastic mini-batches address the data feeding issue, while the GNN-based architecture preserves the decentralized nature of federated learning. The authors also provide theoretical proof of the convergence of their proposed unrolled optimizer and demonstrate its efficacy through numerical experiments.

### Strengths
1. Originality: The paper introduces a novel approach. Algorithm unrolling is a learning-based optimization paradigm where iterative algorithms are unfolded into trainable neural networks, leading to faster convergence. Federated learning, on the other hand, is a distributed learning paradigm where multiple devices collaboratively train a global model. The originality of the paper lies in its integration of these two concepts, addressing specific challenges in server-free federated learning such as the need for whole datasets in unrolled optimizers and the decentralized nature of the learning process.

2. Clarity: The paper is well-structured and presents its ideas in a clear and concise manner. 

3. Algorithm Simplicity and Neatness: Despite addressing complex challenges in federated learning, the algorithm proposed in the paper is simple and neat. The use of stochastic mini-batches and a GNN-based architecture provides a straightforward yet effective solution. The simplicity of the algorithm makes it accessible and easy to implement.

### Weaknesses
1. Vulnerability of Assumption 1: The paper assumes convexity in its problem formulation, which might not align with the real-world scenarios where deep learning models, predominantly used in Federated Learning (FL), are non-convex. This assumption is quite vulnerable as it oversimplifies the complexity of the learning models, potentially leading to over-optimistic results and conclusions. In practice, dealing with non-convex optimization problems is more challenging, and the algorithms need to be robust enough to handle such complexities.

2. Practicality of Assumption 2: The assumption that  g=f and g=∣∣∇f∣∣ (f=∣∣∇f∣∣) is very rare to satisfy in real-world applications. These conditions impose strict requirements on the relationship.

3. Local Minima and Convergence: In non-convex optimization problems, the paper should consider replacing the goal of reaching local minima with finding stationary points, which are points where the gradient is close to zero. This adjustment would provide a more accurate representation of the convergence behavior in non-convex settings, since two neural nets are involved.

4. Heterogeneity of Local Models and Fair Comparison: The paper adopts the heterogeneity of local models and data distribution in federated learning settings. However, the comparison of SURF with FedAvg-type methods might not be entirely fair due to this heterogeneity. To address this issue, the paper should conduct more extensive experiments, comparing SURF with a broader range of personalized federated learning methods that are designed to handle heterogeneity more effectively. Some of the methods that could be considered for comparison:

pFedMe: Personalized Federated Learning with Moreau Envelopes Dinh et al., 2020
PerFedAvg: Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach Fallah et al., 2020
APFL: Adaptive Personalized Federated Learning Deng et al., 2020
Ditto: Fair and Robust Federated Learning Through Personalization Li et al., 2022
Mobilizing Personalized Federated Learning in Infrastructure-Less and Heterogeneous Environments via Random Walk Stochastic ADMM
, Parsons et al., 2023

### Questions
1. Given that the assumption of convexity might not hold in many real-world deep learning scenarios, how does this affect the applicability of SURF, and are there plans to extend SURF to non-convex settings?
2. How can we ensure that the conditions g=f and g=∣∣∇f∣∣ are met?
3. How much does the heterogeneity of local models and data distribution in federated learning environments affect the performance of SURF?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a framework named SURF, focusing on stochastic algorithm unrolling in federated learning contexts. The authors specifically employ descending constraints on the outputs of unrolled layers to make sure convergence. They also leverage the Lagrangian dual problem for optimization, with empirical validation on Graph Neural Networks (GNN).

### Strengths
1. The SURF framework stands out for its innovative method of implementing stochastic algorithm unrolling in federated learning. This novel approach, particularly the use of duality and gradient descent ascent in solving the Lagrangian dual problem, is a significant departure from traditional federated learning methodologies.
2. The paper provides a mathematical analysis of the convergence bound of SURF, indicating thorough theoretical underpinning. Also, the key technique of imposing descending constraints on the outputs of the unrolled layers to ensure convergence appears novel to me.

### Weaknesses
1. **Strong Assumptions**: The assumption of convexity in Assumption 1 is a significant limitation, given that many real-world scenarios involve non-convex functions. This assumption could restrict the applicability of the SURF framework in broader federated learning contexts.

2. **Lack of Comparative Analysis**: The paper does not provide an upper bound for the number of communication rounds needed to converge to a certain precision $\varepsilon$. This omission makes it difficult to compare SURF with other federated learning works, raising questions about the significance and practicality of the contribution.

### Questions
1. In the (SURF), there is no explicit representation of $\mathbf{W}_L = \boldsymbol{\Phi}(\boldsymbol{\vartheta}; \boldsymbol{\theta})$. Is this an intentional choice?
2. What is the complete formulation of the function $f$ in Assumption 2? Since the parameter of $f$ seems to depends not only on $\theta$ but also on other factors like $l, w_0$ etc., a clear definition is necessary.
3. Given that the fomula in (5) is based on expectations without explicit randomness, why does Theorem 2 require that (5) holds with a certain probability?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel approach to accelerate FL convergence in a server-less setting. This is achieved via incorporating descending constraints on unrolled architectures. The proposed approach SURF is theoretically (Theorem 2) and empirically (Figure 2) substantiated. These findings demonstrate that an unrolled optimizer trained with SURF converges to a region close to optimality, ensuring its ability to generalize effectively to datasets within the distribution.

### Strengths
The paper is well written and the problem is well motivated. I find the descent constraints to arrive at a convergence guarantee very clever. As far as I'm aware, this method is novel, although I am not quite familiar with the L2O/unrolled algorithm literature so I can't say for sure.
Experiments are quite basic, but show some promising results.

### Weaknesses
There have been some existing works on serverless FL. For example, I find the following paper "FedLess: Secure and Scalable Federated Learning Using Serverless Computing" (Grafberger et al., 2021). I would suggest the authors to compare to some of these methods rather than standard FL approaches.

I also do not understand how the SURF method is limited to serverless FL. Can it be applied to standard FL instead? 

It feels quite strange seeing that the accuracy curves of all other methods are very similar. Could it be due to this setting?

 I think Fig. 2 does not say anything about your convergence. What do accuracy and loss value at one point have to do with convergence guarantee? If anything, it would be Fig. 1, but it feels quite amazing to achieve perfect accuracy on CIFAR10 with only 20 training epochs. Can you please elaborate on what is happening in one communication round here?

Some of the experiment setup descriptions are quite vague. Could you elaborate on the following points:
- How were the other FL baselines modified to account for the serverless setting? 
- What does it mean by "randomly-chosen agents are asynchronous with the rest of the agents". What is being asynchronous here, and how do you simulate it?
- What exactly is happening in one communication round?

### Questions
I have put my concerns in question form above.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good
