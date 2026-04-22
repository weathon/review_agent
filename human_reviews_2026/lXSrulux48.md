# Byzantine-Robust Federated Learning with Learnable Aggregation Weights

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 8

## Abstract
Federated Learning (FL) enables clients to collaboratively train a global model without sharing their private data. However, the presence of malicious (Byzantine) clients poses significant challenges to the robustness of FL, particularly when data distributions across clients are heterogeneous. In this paper, we propose a novel Byzantine-robust FL optimization problem that incorporates adaptive weighting into the aggregation process. Unlike conventional approaches, our formulation treats aggregation weights as learnable parameters, jointly optimizing them alongside the global model parameters. To solve this optimization problem, we develop an alternating minimization algorithm with strong convergence guarantees under adversarial attack. 
We analyze the Byzantine resilience of the proposed objective.
We evaluate the performance of our algorithm against state-of-the-art Byzantine-robust FL approaches across various datasets and attack scenarios. Experimental results demonstrate that our method consistently outperforms existing approaches, particularly in settings with highly heterogeneous data and a large proportion of malicious clients.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper uses an alternating minimization algorithm with strong convergence guarantees to address Byzantine attacks in FL. This method treats aggregate weights as learnable parameters and optimizes them jointly with global model parameters. Furthermore, the paper conducts experiments on various datasets and attack scenarios, and provides theoretical and convergence analysis of the method.

### Strengths
1. The paper is highly clear. The methods section provides sufficient proof and comprehensive theoretical analysis.
2. The appendix provides detailed information and experimental results supplementing the main text. The paper is highly comprehensive and logically organized. The experimental setting includes five different attack methods and various heterogeneous conditions.
3. The paper proposes learnable aggregation weights, providing a new approach to addressing the convergence problem of equal-weighted average damage.

### Weaknesses
1. Whether learnable aggregation weights increase the weight of byzantine clients and thus increase attack risk is lacking theoretical analysis and experimental proof.
2. The experimental datasets used are MNIST and CIFAR10. Both datasets contain only 10 classes, which cannot demonstrate the effectiveness of the method on more complex dataset.
3. In the main text, the method and theoretical analysis section of the paper lacks a figure explaining the pipeline, which increases the reading difficulty.

### Questions
1. I'd like to know the specific experimental results of the algorithm on more complex datasets, such as CIFAR100 and TinyImageNet, which contain more classes.
2. Would learnable aggregation weights exacerbate attacks by byzantine clients tricking the server into learning higher weights? I recommend that the paper provide theoretical analysis of this scenario.

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
3

### Summary
This paper proposes a Byzantine robust federated learning algorithm where aggregation weights are learnt jointly with the model parameters via a nested optimization formulation. A Byzantine resilience analysis is provided, along with a convergence analysis. Experiments on MNIST and CIFAR-10 datasets show that the method outperforms baselines robust aggregations under varying heterogeneity scenarios, and for different Byzantine attacks.

### Strengths
- The method seems novel and provides a new avenue for research on Byzantine robustness. 
- The algorithm intuition is clearly described, and many details are given on how to implement the algorithm in practice. 
- The added computation and communication complexities are discussed in details, which is appreciated. 
- The theoretical analysis is provided for the both the cases when only the sent gradients are corrupted and when also the sent loss evaluations are corrupted (even though only the latter truly matters).

### Weaknesses
1. It seems to me there is something conceptually wrong with the proof, precisely when decomposing the error in part E3. The bound between $F$ and $\tilde{F}$ assumes that the same byzantine clients will be selected by the aggregation for either the mini-batch or full gradients. This is wrong.
2. It also seems that the proof uses the exact minimum of equation (6), whereas the algorithm provides an approximation through the first order decomposition. 
3. The theoretical results are not compared with baseline methods. Previous works such as [1] show robust methods that achieve the lower bounds, it is not clear how the presented method improves on those.
4. I do not agree with Remark 1 of the authors. Comparing the performance based on the total number of communication rounds is not fair.
5. Assumption (C) on Stochastic Gradient Model is not standard. Why isn’t the standard upper bound on the variance of the SGD noise (which seems to be used as inter-client variance in D1) enough for the guarantees ?
6. The Byzantine resilience guarantees are given probabilistically, which is highly non-standard in the literature. Compared to state-of-the-art Byzantine robust methods, no fundamental randomization is added by the FedLAW algorithm that would justify probabilistic bounds.
7. 321  “in practice $\epsilon_k$ is typically very small even under a high heterogeneity” the authors do not provide any justification for this statement.
8. 320 “Similarly, assuming bounded heterogeneity is a standard prerequisite for any non-IID analysis.” Can the authors please provide references for this ? The Magnitude Heterogeneity part does not look standard.
9. In Table 7, the accuracy scores on CIFAR-10 seem abnormally high (reaching even 100% with backdoor attacks). 

**Minor issues**

- Theorem 2 does not specify the choice of s and t (which are specified only in the proof in the appendix)
- Shouldn’t the sparse unit capped operator be noted $\Delta_{t,s}$ instead ?

### Questions
1. Is there any reason why the probabilistic framework is necessary for the resilience results ?
2. It is not clear to me what the $g_i$s represent. The expectation on $v_{i,t}$ is taken with respect to what ?
3. In 170, the authors claim that the method conceptually favors clients whose gradients align with the descent direction of $f_i(\theta_k - \alpha  G_k w)$. I believe this intuition needs to be explained further, as it is not clear why this should be the case ? Why does the descent direction of one single client matter ?
4. In the experiments, why is the performance in the case of no defense almost the same as the other robust aggregation rules for many attacks and heterogeneity levels (and sometimes it is even better than some defenses)?
5. As multiple local steps are shown to improve the performance in FL (including Byzantine robustness), can the method be extended to support multiple local steps ?

**Minor questions**
- Isn’t it possible to link $L_w$, the smoothness coefficient of $\Phi_k$ to $L_{max}$ ?
- What is the point of Theorem 7 ?
- 270 $G_k^i$ is not defined. $v_{k,i}$ is defined in 253 as the full batch gradient

### Soundness
2

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
This paper proposes FedLAW, a Byzantine-robust federated learning method that treats client aggregation weights as learnable parameters, jointly optimized with the global model. The key contributions are:

A novel optimization problem formulation incorporating adaptive aggregation weights with sparsity constraints to exclude malicious clients.

An alternating minimization algorithm with theoretical convergence guarantees under adversarial settings.

Comprehensive theoretical analyses demonstrating Byzantine resilience and convergence properties.

Extensive experiments on MNIST and CIFAR-10 under various attack scenarios and non-IID data settings, showing FedLAW outperforms state-of-the-art methods, especially under high heterogeneity and malicious client ratios.

### Strengths
The paper presents a genuinely novel approach to Byzantine-robust federated learning by treating aggregation weights as learnable parameters. This isn't just a minor tweak to existing methods—it represents a meaningful shift in how we approach the aggregation problem. The theoretical foundation is particularly impressive, providing not just convergence guarantees but also a detailed Byzantine resilience analysis that clearly explains why the method works. What makes the contribution stand out is how well the empirical results support the theory; the method maintains strong performance even under challenging conditions like 40% malicious clients and high data heterogeneity, which is exactly where many existing methods struggle. The writing is clear and the figures effectively illustrate the method's behavior, especially the weight evolution plots that show how it dynamically identifies and suppresses malicious clients.

### Weaknesses
While the method is compelling, it does come with some practical trade-offs. The two-round communication per update is a noticeable overhead, and while the authors argue that faster convergence might compensate for this, the paper doesn't provide a conclusive analysis of the total communication cost compared to alternatives. Some of the theoretical assumptions, like the bounded gradient deviation and heterogeneity bounds, feel somewhat idealistic—in real-world non-IID settings, these assumptions might not hold as neatly. The experimental validation, while thorough on standard datasets, leaves me wondering how the method would scale to more complex problems or different data domains. The hyperparameter selection also seems non-trivial, particularly the sparsity level, which appears to require some knowledge of the malicious client ratio.

### Questions
Communication Efficiency: Could the two-round communication be optimized? Can you provide a standardized comparison of total communication rounds versus accuracy?

Practicality of Assumptions: How realistic are assumptions C1 and D1 in real non-IID settings? Are there methods to verify or relax them?

Scalability: How does FedLAW perform with a very large number of clients (e.g., >1000)? Are there distributed optimization strategies to improve efficiency?

Hyperparameter Tuning: Does the selection of $s$ and $t$ rely on prior knowledge of the malicious client ratio? Can these parameters be adapted dynamically?

Integration with Privacy Techniques: Can FedLAW be combined with differential privacy or cryptographic methods to enhance privacy protection?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies proposes an algorithm that can learn the aggregation weights for benign clients in Byzantine attack environment. The idea is novel and is effective in enhancing accuracy after data in Byzantine attackers are ignored. The contributions include both new algorithm in robust federated learning and theoretical analyses for Byzantine resilience and algorithm convergence.

### Strengths
1. The paper studies Byzantine adversarial tolerance, which is an important problem in federated learning. 
2. The paper is well written and easy to follow.
3. The theoretical analysis is rigorous and the experiments are extensive to support the effectiveness of the proposed algorithm.

### Weaknesses
1. Figure 1 is not clear, with confusing color to denote different algorithms. 
2. How to enforce sparsity is not discussed, i.e., how to determine how many clients are malicious?

### Questions
1. Did you consider removing the malicious client identify step and automatically learning/assigning low weights to Byzantine clients?

### Soundness
4

### Presentation
3

### Contribution
3
