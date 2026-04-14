# On the Power of Federated Learning for Online Sparse Linear Regression with Decentralized Data

- Decision: Reject
- Scores: 6, 5, 3, 8

## Abstract
In this paper, we study the necessity of federated learning (FL) for online linear regression with decentralized data. Previous work proved that FL is unnecessary for minimizing regret in full information setting, while we prove that it can be necessary if only limited attributes of each instance are observed. We call this problem online sparse linear regression with decentralized data (OSLR-DecD). We propose a federated algorithm for OSLR-DecD, and prove a lower bound on the regret of any noncooperative algorithm. In the case of $d=o(M)$, the upper bound on the regret of our algorithm is smaller than the lower bound, demonstrating the necessity of FL, in which $M$ is the number of clients and $d$ is the dimension of data. When $M=1$, we give the first lower bound on the regret and improve previous upper bounds. We invent three new techniques including an any-time federated online mirror descent with negative entropy regularization, a paradigm for client-server collaboration with privacy protection, and a reduction from online sparse linear regression to prediction with limited advice for establishing the lower bound on the regret, some of which might be of independent interest.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper delves into the importance of federated learning (FL) in the context of online sparse linear regression (OSLR) with decentralized data (DecD). The authors challenge the previous consensus that FL is unnecessary for minimizing regret in full information settings by demonstrating that FL becomes essential when only limited attributes of each instance are observable.

The paper introduces the problem of online sparse linear regression with decentralized data (OSLR-DecD), where data is scattered across multiple clients, and each client can only observe a limited number of attributes for each instance. The authors propose a federated algorithm called FedAMRO for OSLR-DecD and establish a lower bound on the regret of any noncooperative algorithm. They prove that in scenarios where the dimensionality of data (d) is much smaller than the number of clients (M), the upper bound on the regret of FedAMRO is smaller than the lower bound for noncooperative algorithms, thus highlighting the necessity of FL.

### Strengths
1. An upper bound on the regret for FedAMRO, showing its effectiveness in OSLR-DecD.

2. A lower bound on the regret for any noncooperative algorithm, reinforcing the advantage of federated learning in OSLR-DecD when d is of smaller order than M.

3. The first lower and upper bounds on the regret for OSLR when M equals 1.

4. Three new techniques: any-time federated online mirror descent with negative entropy regularization, a client-server collaboration paradigm with privacy protection, and a reduction from online sparse linear regression to prediction with limited advice for establishing regret bounds.

### Weaknesses
1. The paper assumes that the dimensionality of data (d) is much smaller than the number of clients (M) to establish the necessity of federated learning. This assumption might not hold in real-world scenarios where the data can be high-dimensional or the number of clients limited. Whether this result can generalize to  high-dimensional sparse regression method ? such as lasso?  

2. The experimental section of the paper validates the theoretical findings on a limited number of datasets. The generalization of the results to other types of data or different problem domains is not extensively tested. A more rigorous experimental validation involving a broader and more diverse set of datasets would strengthen the paper's conclusions.

### Questions
1. How does the proposed Federated Algorithm for Online Sparse Linear Regression with Decentralized Data (FedAMRO) scale with the increase in the number of clients (M) and the dimensionality of data (d), especially when both M and d are large? What are the potential bottlenecks, and how might they be addressed?

2. How robust is the proposed FedAMRO algorithm to data heterogeneity across clients? 

Because I am not a professional researcher in the field of federated optimization, I find the conclusions of this article quite interesting. In the sparse case, it actually draws a contrary conclusion to the general setting, so I give it a score of 6.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper addresses the challenge of online linear regression with partial feedback in a decentralized data environment (OSLR-DecD). The authors explore the role of federated learning in this context, proposing and proving the upper bound of the regret for a federated algorithm named FedAMRO for OSLR-DecD. They establish a regret lower bound for any noncooperative algorithm, showing that, in cases where $d=o(M)$, the regret upper bound of FedAMRO is smaller than this lower bound. This result underscores the effectiveness and importance of the federated learning approach within the decentralized, partial-feedback setting.

### Strengths
1. This paper addresses the problem of online linear regression with partial information feedback, which is a topic with broad applicability across real-world scenarios where data is decentralized and feedback is limited.

2. By proving both the upper bound of the regret under the federated setting and the lower bound of the regret under the noncooperative setting for the problem, the authors provide a thorough and well-rounded analysis, and answer the question of federated learning in this problem.

### Weaknesses
A primary weakness of this paper is the lack of clarity in conveying the intuition behind the core analytical ideas. The authors introduce numerous complex notations early on, often before providing a clear explanation of the algorithms’ key ideas, leaving many of the notations insufficiently defined. Additionally, the absence of a full algorithm description in the main text makes it challenging to grasp the algorithm’s foundational approach. The paper also lacks a proof outline or sketch, making it difficult to follow the reasoning behind the results. Sections 4 and 5 introduce extensive variable definitions for the pseudocode, yet both the algorithm and analysis remain inadequately explained. Furthermore, certain parts of the problem definition in the introduction and Sections 3.1 and 3.2 appear repetitive, occupying space that could be more effectively used to clarify the algorithm and its analysis.

### Questions
1. Could the authors give some intuition or high-level explanation of the algorithm design in Algorithms 2 and 3?
2. Could the authors explain how the core part of the analysis of Algorithm 2 is different from previous work?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies the problem of federated learning for online sparse linear regression (OSLR) with decentralized data. The authors first propose a new algorithm AMRO for OSLR, which improves the previous best regret, then extend AMRO to federated learning setting, and demonstrates the necessity of federated learning for OSLR by comparing the upper regret bound of FedAMRO and the lower regret bound of noncooperative algorithm.

### Strengths
1) This paper studies the problem of online linear regression with decentralized data, and proposes a federated algorithm with theoretical guarantees on its regret bound.
2) The lower bound on the regret of any noncooperative algorithm is also analyzed, which can demonstrate that federated learning is indeed necessary for the studied problem.
3) Experimental results are provided to verify the performance of the proposed algorithm.

### Weaknesses
1) In this paper, the definition of regret for federated learning seems a bit strange. I understand that it follows Patel et al., 2023. However, in decentralized learning, the regret is commonly defined as $Reg^{(i)}(\\mathbf{w})=\sum\_{j=1}^M \sum\_{t=1}^T [l(f_t^i(\\mathbf{x}_t^{j}),y\_t^{(j)}) - l(\\mathbf{w}^\top \\mathbf{x}\_t^{(j)}, y_t^{(j)})] $ [1][2][3]. Note that this definition considers the global loss of the learned hypothesis $f_t^i(\cdot)$ of the single client $i$. By contrast, the definition in this paper considers the sum of each local loss, i.e., $\sum\_{j=1}^M l(\\hat{y}\_t^{(j)},y\_t^{(j)})=\sum\_{j=1}^M l(f_t^j(\\mathbf{x}_t^{j}),y\_t^{(j)})$. 

2) Moreover, I believe this strange definition of regret is the main reason for the pessimistic result of Patel et al., 2023 (i.e., the collaboration among clients is actually unnecessary in the full information setting). If we adopt the commonly used regret, the pessimistic result is no longer valid. In this case, the motivation of this paper may be questionable, since collaboration among clients is indeed necessary.

3) The literature review is poor. There do exist many related but missed studies, especially the mentioned works on decentralized online convex optimization [1][2][3] and the previous works on online sparse linear regression.

4) It is also unclear why the problem of online sparse linear regression with decentralized learning should be investigated.

[1] Yan et al. Distributed Autonomous Online Learning: Regrets and Intrinsic Privacy-Preserving Properties. IEEE Transactions on Knowledge and Data Engineering, 2013.

[2] Hosseini et al. Online distributed optimization via dual averaging. In CDC, 2013.

[3] Wan et al. "Nearly Optimal Regret for Decentralized Online Convex Optimization". In COLT, 2024.

### Questions
1) In this paper, the derived regret bound is related to $b$. Should different values of $b$ be tested on a dataset to verify its impact on loss?
2) When extending AMRO to FedAMRO, you modified $\\hat{\\mathbf{x}}$ and $\\widetilde{\\mathbf{x}}$ due to privacy constraints. Are there any theoretical results on privacy analysis?
3) Why does FedAMRO update $\\mathbf{w}\_{t+1,i}$ on the server? It seems that the communication costs can be reduced by updating it on the clients, calculating the cost, and then transmitting it to the server.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the federated online sparse linear regression problem, where each client can only observe part of the feature vector at each round. The authors propose the Aggregated OMD algorithm and establish the regret bound. They also establish the lower bound for non-cooperative algorithms. Their results reveal an interesting fact: unlike the full-information setting, where non-cooperative algorithms are optimal, federated learning can achieve better performance than non-cooperative algorithms.

### Strengths
1. The problem is well-motivated, and the results provide an interesting contrast to federated full-information online linear regression.

2. Technically solid: improvements on regrets of OSLR and a new reduction from sparse linear regression to online prediction with limited advice.

3. Numerical experiments are conducted to illustrate the superior performance of the proposed algorithm.

### Weaknesses
1. There is still an $O(\sqrt{d})$ gap between the proposed upper bound and the lower bound.

2. Several explanations in the algorithm design are unclear, as mentioned in the questions section.

### Questions
1. In line 265, the authors state that subtracting $y_t^2$ from the loss can enhance privacy protection. However, it seems that the reported information still involves $y_t$. Could the authors explain this point further?

2. In line 282, the authors mention that there are many approaches to define $\delta_{t,m}$. Could the authors clarify which general properties $\delta_{t,m}$ must satisfy to achieve the same regret guarantee in the main theorem?

3. The authors state that their operations in equations (3) and (4) differ from previous algorithm designs (subtracting $y_t^2$ or using non-zero $\delta_{t,m}$). Are these differences intended for privacy protection or to improve regret performance?

### Soundness
3

### Presentation
3

### Contribution
3
