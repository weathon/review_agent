# Distributed Estimation of Sparse Covariance Matrix under Heavy-Tailed Data

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
In this paper, we study high-dimensional covariance matrix estimation over a network of interconnected agents, where the data are distributed and may exhibit heavy-tailed behavior. To address this challenge, we propose a new estimator that integrates the Huber loss to mitigate outliers with a non-convex regularizer to promote sparsity. To the best of our knowledge, this is the first framework that simultaneously accounts for high dimensionality, heavy tails, and distributed data in covariance estimation. We begin by analyzing a proximal gradient descent algorithm to solve this non-convex and non-globally Lipschitz smooth problem in the centralized setting to set the stage for the distributed case. In the distributed setting, where bandwidth, storage, and privacy constraints preclude agents from directly sharing raw data, we design a decentralized algorithm aligned with the centralized one, building on the principle of gradient tracking. We prove that, under mild conditions, both algorithms converge linearly to the same solution. Moreover, we establish that the resulting covariance estimates attain the oracle statistical rate in Frobenius norm, representing the state of the art for high-dimensional covariance estimation under heavy-tailed distributions. Numerical experiments corroborate our theoretical findings and demonstrate that the proposed estimator outperforms existing baselines in both estimation accuracy and robustness.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a novel covariance matrix estimation problem as the minimization of a Huber loss combined with a log-determinant barrier and a non-convex penalty. Both centralized version and distributed version of an algorithm solving the problem are introduced, convergence analysis are provided. The theoretical findings are validated through numerical experiments.

### Strengths
(1) The paper reformulate the problem of estimation of sparse covariance matrix as a minimization problem of Huber loss plus log-det barrier term plus penalty term, which is novel. The motivation and intuition are explained.

(2) A centralized version based on the proximal gradient algorithm is proposed, along with its extension in the distributed setting. Convergence guarantees are provided with rigorous theoretical proofs and analysis.

(3) The authors provide justification of the reformulation by providing an statistical guarantee on the estimation error based on Frobenius norm of the estimation $\hat{\Sigma}$ and $\Sigma^\star$, justifying the approach they took.

(4) Numerical results are provided, which further validates the effectiveness of the approach.

### Weaknesses
(1) The major contribution seems to lies at the reformulation of finding the covariance matrix to the form of minimizing the Huber loss plus the additional barrier and penalty term. If I am not mistaken, the major algorithm is simply the proximal gradient algorithm and its distributed version relies on standard distributed optimization frameworks and techniques such as gradient tracking. This seems to be rather straightforward since they follow directly from existing literature, which makes the contribution incremental.

(2) There are many parameters there in order to use the algorithm, for example, in the convergence guarantee for the decentralized version (Theorem 2), the assumption that $\rho < F(\kappa)$ is assumed, but how do we make sure it is true? $\kappa$ itself is the condition number and we do not know it typically, and $\rho$ itself depends on the graph/network we are working on, does it mean that we need to constrain on certain types of network? In addition, we need to have $\gamma$ bounded from below by a complicated expression, $\theta$ also bounded by a complicated expression, and those expressions involve many quantities we do not know. In practice it seems that we still need to manually tune those hyperparameters to ensure convergence. This makes the theoretical results less significant. 

(3) The theorems only state complexity in terms of iteration, but what about the computational complexity and the communicational complexity in the decentralized setting? For a high dimensional problem, computing the Huberized gradient and perform the proximal mapping seems to be quite expansive, since the paper is focused on high-dimensional covariance-matrix estimation, it is better to make those things clear. Right now it seems a bit hard to judge if the methods are really scalable or not.

### Questions
(1) What do the authors mean by heavy-tailed exactly? There is no formal definition of this in the paper, perhaps it is better to mention this explicitly.

(2) In Theorem 1, we assume $\hat{r} + \frac{\underline{r}}{2} < \sqrt{\frac{\tau}{L_q}}$, which ensures that the iterates stays in a positive definite region, which is critical to the proof. But in reality we cannot check whether this holds or not, and it is not easily tuned by the user. The authors mention that $\gamma$ and $\theta$ can be selected empirically, but what about $\tau$? 

(3) There are many notations in the paper. Could the authors provide a list in the Appendix or somewhere else summarizing all the notations appeared in the paper, this would make the reading of the paper much smoother.

### Soundness
2

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
3

### Summary
The paper proposes a new estimator for high-dimensional covariance matrix over a network of interconnected agents where the data are distributed and might be heavy-tailed. This the first framework that incorporates high dimensionality, heavy tails and distributed data simultaneously in covariance estimation. The paper first analyzes a proximal gradient descent algorithm to solve this problem in the centralized setting. Next, built on gradient tracking, the paper proposes a decentralized algorithm. Linear convergence is obtained for both algorithms. Numerical experiments are provided to support the theory.

### Strengths
(1) Overall, the paper is well-written. The theoretical analysis seems to be rigorous, and the problem setup is well motivated.

(2) There is extensive literature review.

(3) Numerical experiments are extensive, and comparisons to the existing methods in the literature are provided.

### Weaknesses
(1) In the algorithm, the Huber loss is used. I think in terms of theory, it would be more interesting to have a more general setup that includes the Huber loss as a special case. If it is challenging, then you should add some discussions why it is hard to extend beyond the Huber loss setting.

(2) I am wondering how difficult it is to incorporate stochastic gradients in the algorithms. Otherwise, when the number of data points is huge, it may not scale well.

### Questions
(1) In the last sentence in Appendix C.1.3., you wrote that we can obtain the linear convergence result in Theorem equation 1, and in the first sentence in Appendix C.1.3., you also wrote Theorem equation 1. I think you should write equation 1 in the theorem instead or equation 1 in Theorem x, where x is made explicit. 

(2) In equation (23), . should be ,

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the problem of distributed sparse covariance matrix estimation under heavy-tailed data, an important and technically challenging topic in modern high-dimensional statistics and distributed learning. The authors formulate the estimation task as a non-convex optimization problem involving a Huber loss to ensure robustness, a log-determinant barrier to ensure positive definiteness, and a non-convex sparsity-inducing regularizer. They propose both centralized and decentralized proximal gradient algorithms (Algorithms 1 and 2), establish linear convergence guarantees, and prove that their estimators achieve the oracle statistical rate under the Frobenius norm.

### Strengths
The paper provides linear convergence and statistical guarantees for both centralized and distributed settings, supported by detailed proofs in the appendix.

### Weaknesses
The paper is written poorly. In particular, the problem formulation is not clear. There is a matrix of wights W in the definition, which said that encode the interaction between agents. However there is no example of what type of interactions the authors means. For example, in the abstract it is said that the proposed algorithm takes into account the bandwidth and local storage. However there is no discussion about them in the problem formulation. It is perhaps encoded in the matrix W. Also, the authors seek to find a robust algorithm, however, there is no trace of robustness in the problem formulation except the use of Huber loss in the optimization problem. It should be made clear that what type of robustness the authors investigated. In summary, the problem formulation only indicates that the paper tries to solve a suitable optimization problem involving a distruted fashion without accounting for the merits like bandwidth, storage, robustness, etc claimed in the abstract of the paper.

### Questions
**Problem Formulation**
The paper suffers from significant clarity issues, particularly in the problem formulation. The definition introduces a weight matrix W, said to encode interactions among agents, yet the nature of these interactions is never illustrated with examples. For instance, the abstract claims that the proposed algorithm accounts for bandwidth and local storage constraints, but these aspects are not discussed or formalized anywhere in the problem setup. It is possible that such considerations are implicitly embedded in W, but this is never explained.

Furthermore, the authors emphasize robustness as a key objective, yet apart from the inclusion of a Huber loss in the optimization problem, there is no clear indication of what type of robustness is being studied or against what sources of uncertainty or perturbation it is intended to protect.

In summary, the problem formulation as written appears to describe a generic distributed optimization framework without incorporating or clearly articulating the claimed contributions related to bandwidth, storage, or robustness mentioned in the abstract.

**Robustness**
Both the centralized and decentralized algorithms are referred to as “robust,” yet this claim is neither theoretically justified nor empirically demonstrated. Beyond the earlier mention of the Huber loss, there is no formal analysis, theorem, or experimental evidence supporting robustness in any meaningful sense. While the Huber loss can mitigate the effects of heavy-tailed noise, it is insufficient to guarantee robustness against adversarial perturbations or structured outliers. Numerous alternative approaches—such as median-based aggregation or robust consensus schemes—are more suitable for such settings. Without theoretical backing or numerical evaluation of robustness properties, the use of the term “robust” appears unsupported.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies sparse covariance estimation problem under heavy-tailed noise data. A centralized proximal gradient descent algorithm and a decentralized gradient tracking based algorithm are analyzed, both achieving linear convergence.

### Strengths
1. This paper proposes a new single optimization objective that avoids a two stage process to ensure robustness and sparsity. 
2. Both algorithms achieve optimal statistical rates. 
3. Handles high dimension, heavy-tailed observations, and distributed data the same time.

### Weaknesses
1. The statistical rates depend on prior of many unknown problem parameters such as $\Sigma^*$.
2. Hyper-parameter setups are complicated, and the cost of hyper-parameter tuning in practice is not clear. 
3. The algorithms are designed for high-dimensional case but the per-iteration cost is in $d^3$, which is high.

### Questions
1. The optimal solution presented in this paper is local, can you discuss more on this point, please also contrast with other works. 
2. The heavy-tailed assumption is strong, can you add more discussions and compare with other works using conditions such as sub-Gaussian.

### Soundness
3

### Presentation
3

### Contribution
3
