# LoCoDL: Communication-Efficient Distributed Learning with Local Training and Compression

- Decision: Accept (Spotlight)
- Scores: 6, 8, 8, 8

## Abstract
In $D$istributed optimization and $L$earning, and even more in the modern framework of federated learning, communication, which is slow and costly, is critical. We introduce LoCoDL, a communication-efficient algorithm that leverages the two popular and effective techniques of $Lo$cal training, which reduces the communication frequency, and $Co$mpression, in which short bitstreams are sent instead of full-dimensional vectors of floats. LoCoDL works with a large class of unbiased compressors that includes widely-used sparsification and quantization methods. LoCoDL provably benefits from local training and compression and enjoys a doubly-accelerated communication complexity, with respect to the condition number of the functions and the model dimension, in the general heterogeneous regime with strongly convex functions. This is confirmed in practice, with LoCoDL outperforming existing algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper combines local update with compression to improve communication efficiency in distributed estimation.

### Strengths
Proves the proposed algorithm the accelerate communication complexity due to both components used.

### Weaknesses
Only demonstrate strongly convex case. Assuming g is smooth (exclude non-smooth penalty such as lasso, for example)

### Questions
1. What is the intuition that make the proposed algorithm work compared to existing works that combines LT and CC?
2. Is the rate obtained optimal?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes LoCoDL, an algorithm for distributed optimization that incorporates both local training and communication compression. The design of LoCoDL uses a primal-dual approach and randomization, provably achieving double acceleration in strongly convex landscape with unbiased compression. Empirical advantages of LoCoDL are demonstrated from experiments.

### Strengths
* The design of LoCoDL shows novelty such as the introduction of primal-dual updates.
* The convergence of LoCoDL with unbiased compression under strongly convex landscape is supported by rigorous theoretical analysis, provably showing a reduced communication overhead.
* LoCoDL achieves empirical advantages in experiments on regularized logistic regression.
* The presentation of the paper is well-organized and clear to follow.

### Weaknesses
* The theory of this paper has some limitation. For example, it assumes the usage of unbiased compression, which does not encompass some popular biased schemes such as Top-k compression. The proof also exploits strong convexity and nonconvex landscape poses a challenge.
* Empirical evaluations are conducted only on regularized logistic regression with homogeneous distribution of local dataset. The experiments are not able to demonstrate the performance in heterogeneous settings (which is claimed by the theory) and nonconvex problems; also see Questions.

### Questions
* Please consider a more comprehensive experimental evaluation.
  * Since nonconvex landscape is not covered by theory, experiments on deep neural networks can be done to show its performance in handling nonconvex problems.
  * The theory claims heterogeneous setting but experiments are done on a homogeneous distribution of dataset. Experiments with data heterogeneity can be conducted, as in previous related works like [1,2].
* I suggest the authors provide more discussions on the primal-dual design. How does this help with your proof? What challenges will you encounter without this primal-dual approach?

[1] Huang, X., Chen, Y., Yin, W., and Yuan, K. (2022). Lower bounds and nearly optimal algorithms in distributed learning with communication compression. Advances in Neural Information Processing Systems, 35, 18955-18969.

[2] Chen, S., Li, Z., and Chi, Y. (2024). Escaping Saddle Points in Heterogeneous Federated Learning via Distributed SGD with Communication Compression. In International Conference on Artificial Intelligence and Statistics (pp. 2701-2709). PMLR.

### Soundness
3

### Presentation
3

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
This paper studies how to develop communication-efficient algorithms for centralized optimization problem by combining the idea of local training and communication compression, and proposes an algorithm called LoCoDL. LoCoDL is both validated theoretically through a nearly optimal convergence rate and via empirically by comparing with previous SOTA algorithms.

### Strengths
1. Though the area of local training and communication compression has been explored by previous works, it may still be challenging to develop algorithms combining these ideas to achieve the theoretically optimal convergence rate. LoCoDL is shown to be optimal in cases where $n$ is smaller than the problem dimension $d$, which is often the case in real settings. Considering the assumptions used are standard, this result is believed to be strong enough.
2. It is impressive that LoCoDL performs much better than ADIANA, which has been the SOTA algorithm in distributed deterministic convex optimization with communication compression, and enjoys an optimal convergence rate matching the convergence lower bound.

### Weaknesses
1. The theoretical results are not perfect. Since LoCoDL has a sub-optimal rate in the case where $n$ is larger than $d$, its unclear whether it is the algorithm or the analysis is sub-optimal. As ADIANA has already achieves the optimal rate, it should be better if the author discuss the challenges to achieve a same rate.
2. It's not clear how LoCoDL is derived by combining the two ideas "local steps with local models" and "compressing the difference of two estimates". It is recommended that the authors discuss the motivation behind each algorithm line.

### Questions
1. It is a little bit strange that ADIANA performs worse than other baseline algorithms. Are the hyperparameters not sufficiently tuned?
2. Since with communication compression only we can achieve the optimal theoretical convergence rate, how to illustrate the theoretical benefit of using "local steps with local models"?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposed the LoCoDL algorithm, which applies both communication compression and local updates in distributed training, and analyzed its complexity, showing that with suitable compression scheme the total communication complexity matches that of the accelerated compressed methods

### Strengths
The proposed method, LoCoDL manages to combine both local updates and communication compression, and show that in the strongly convex regime, the method enjoys the communication-saving benefits of both techniques. LoCoDL combines ideas from ProxSkip and EF21 (and the likes) with primal-dual updates to achieve this impressive result.

### Weaknesses
see questions.

### Questions
1. Can the authors please explain in more details and provide some intuitions on why does LoCoDL "require" a regularisation $g$ to work? I put "require" in quotes because, as explained in section 3.1, one can reduce the case with $g=0$ to some strongly convex $\tilde g$. But such reduction highlights the fact that the algorithm, on its own, does need a strongly convex $g$, which seems quite mysterious to me. I think it would significantly improve my understanding of the technique if the authors can point out why/if LoCoDL, as is and without the reduction, would fail when $g=0$.

2. Based on the choice of the Lyapunov function, it seems to me that the convergence of LoCoDL is characterised in terms of $\frac{1}{n}\sum\lVert x_i^t-x^\star\rVert^2$. While ProxSkip's analysis also uses this convergence criteria, there the situation is slightly different since if the composite part can attain infinity then there is just no way to prove convergence in terms of the optimality gap when the prox step is skipped. But here the situation seems different, I wonder if it's possible to prove the convergence of LoCoDL in terms of the optimality gap, say, $f(\bar x) + g(\bar x)-f(x^\star)-g(x^\star)$ where $\bar x$ is the average of the primal variable (or perhaps the average optimality gap of each clients)? If not, what's the difficulty here? I think some of the algorithms that the authors compare LoCoDL against do use optimality gap as the convergence criteria, so there might be some mismatch.

3. The paper focuses on the strongly convex case. While the authors mentioned that non convex case might require significantly different techniques, I wonder what about the general convex case? It seems that LoCoDL would fail in that case (see also my question regarding $g$), but can the authors provide some insights in what would fail in the general convex case?

4. I wonder if LoCoDL can handle stochastic gradients? It seems to me that the idea of compressing the difference between two estimates works well with full gradient computations, but might not work so well when there is noise (e.g. EF21 fail to converge beyond the variance of the noise and ProxSkip's original analysis cannot achieve linear speedup with the noise). Is it also the case here? Have the authors tried the stochastic case?

### Soundness
3

### Presentation
3

### Contribution
3
