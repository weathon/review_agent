# Delayed Local-SGD for Distributed Learning with Linear Speedup

- Avg Score: 4.00
- Decision: Reject
- Scores: 5, 3, 5, 3

## Abstract
Local-SGD-based algorithms have gained much popularity in distributed learning to reduce the communication overhead, where each client conducts multiple localized iterations before communicating with the central server. However, since all participating clients are required to initiate iterations from the latest global model in each round of Local-SGD, the overall training process can be slowed down due to the straggler effect. To address this issue, we propose a Delayed Local-SGD (DLSGD) framework for distributed and federated learning with partial client participation. In DLSGD, each client performs local training starting from outdated models, regardless of whether it participates in the global aggregation. We investigate two types of DLSGD methods applied to scenarios where clients have identical or different local objective functions. Theoretical analyses demonstrate that DLSGD achieves asymptotic convergence rates that are on par with the classic Local-SGD methods for solving nonconvex problems, and guarantees linear speedup with respect to the number of participating clients. Additionally, we carry out numerical experiments using real datasets to validate the efficiency and scalability of our approach when training neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This study examines the FedAvg (local-sgd with partial participation) in the context of communication delays (asynchronous communication). It establishes the method's convergence, and the provided bounds demonstrate an asymptotic linear speedup, where the error decreases linearly with the number of clients.

### Strengths
Establishing convergence rate bounds for the delayed local-SGD that match the performance of the non-delayed version asymptotically and imply linear speedup.

### Weaknesses
The current form of the algorithm appears inefficient for the heterogeneous case. The server sends its estimate to all nodes, regardless of whether they are sampled or not, potentially wasting communication. Moreover, all clients continuously execute local updates. If a client isn't selected for several rounds, it simply overwrites its old estimate with a newer one derived from a more recent client estimate. This approach seems wasteful and is a significant weakness unless I've misunderstood.

Assuming a bounded gradient is a stringent condition.

### Questions
The rate for local-SGD in the heterogeneous case appears to differ from the loca-SGD rate established in the Scaffold paper. Is this rate in terms of communication or computation?

In standard local-SGD implementations, the server sends updates only to clients participating in local updates, leaving others idle. However, in this method, it seems all clients undergo local updates, whether or not they're selected by the server. Additionally, the server dispatches its updated model to every node. One of FedAvg's defining features is its selective communication with clients, which, in practice, cuts communication costs. It might be more practical for the server to send the updated model only to a subset of clients that weren't chosen in the preceding iterations. This way, nodes won't receive a new model until they've completed their local updates based on the last received version. Otherwise, it seems like processing power is expended needlessly.


Moreover, in delayed SGD, once an active client concludes its update, it sends its estimate immediately without waiting for other active clients to finish. Given this, a comprehensive comparison between these models is necessary.

Why is there a distinction between homogeneous and heterogeneous cases? Shouldn't the results from the heterogeneous case inherently encompass those from the homogeneous case?

In theorem 1, it's unclear why the noise variance is in the denominator. For a deterministic scenario where $\sigma=0$ what would the result be? This finding seems counterintuitive.

### Soundness
3 good

### Presentation
2 fair

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
This paper proposes a delayed local-SGD framework for decentralized and federated learning with partial client participation to address the straggler issue. Theoretical results guarantee the linear speedup of the proposed DLSGD under non-convex settings and numerical results validate the efficiency of the proposed algorithm.

### Strengths
This paper proposes a delayed local-SGD framework for decentralized and federated learning with partial client participation to address the straggler issue.

### Weaknesses
1. The first concern is about the feasibility of the proposed algorithms. To tackle the straggler issue, in each round, the server only needs to wait for the fast n workers to upload the model updates. The question is how the server determines the number n in real implementations. If there is no clear way to determine the number of $n$, the proposed algorithm will lack practical value.

2. How do the receive buffers update in each round $t$, especially for the stragglers? What is the size of it? There seems to be no discussion on it. Please explain it clearly.

3. What is the purpose of send buffer? Can you clearly illustrate the difference between DLSGD-homo and DLSGD-hetero?

4. The reviewer is confused about the linear speedup claimed by this paper. There are total $N$ workers and the server samples $n$ fastest worker per round. The current linear speedup in Theorem 1 and Theorem 2 is for sampler workers $n$, not total number of workers $N$. It is not the linear speedup in previous work cited in this paper. Please explain this. 

5. To show the convergence, this paper assumes the bound gradient in Assumption 5. This is actually a weakness compared with most previous work which does not require such an assumption. Please explain why this assumption is necessary. 

6. It is weird to this reviewer that the learning rates are set according to $F(w^0)-F^*$ and $\sigma^2$. This is not a common approach.

### Questions
see the weakness above.

### Soundness
2 fair

### Presentation
2 fair

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
This paper proposes a Delayed Local SGD (DLSGD) framework for Local SGD and Federated Learning settings, in the presence of stragglers among clients. The main motivation is that traditional synchronous Local SGD suffers from the straggler effect, where the overall training time is bottlenecked by the slowest clients in each round. This paper aims to address this issue. They propose DLSGD for both homogeneous and heterogeneous cases, where in this algorithm the global aggregation only uses a subset of client updates. This reduces waiting time and improves efficiency. The main difference is that DLSGD-homo uses first-arriving client updates while DLSGD-hetero randomly samples clients. They claim that both DLSGD variants achieve convergence rates comparable to synchronous Local SGD, with linear speedup w.r.t. number of clients. Allows delayed updates without hurting asymptotics. They show the effectiveness of the proposed algorithm on several image classification tasks, compared to local SGD and AsySGD.

### Strengths
This paper proposes the novel DLSGD framework for asynchronous distributed and federated learning that reduces waiting times and improves efficiency compared to synchronous Local SGD  and FL methods. A major strength is the rigorous theoretical analysis that proves DLSGD achieves asymptotic convergence rates comparable to Local SGD, despite allowing delayed model updates. Additionally, the paper considers both homogeneous and heterogeneous data settings relevant to parallel optimization and federated learning. The algorithms are intuitive and complemented by proofs that are easy to follow. Comprehensive experiments on image classification tasks validate the practical efficiency gains of DLSGD in terms of a faster decrease in training loss and an increase in test accuracy over wall-clock time compared to basic local SGD and FedAvg baselines. Overall, the combination of an asynchronous framework design, provable guarantees, and empirical gains demonstrates DLSGD's strengths in improving large-scale distributed training both theoretically and practically.

### Weaknesses
There are several concerns regarding the proposed algorithm:
1. My main concern is that how the stale gradients can contribute to updating the global model. The gradients from different clients have different starting points of $\boldsymbol{w}^{\tau_i(t),0}$, and hence, averaging them together might have adverse effects and counter-intuitive. This criteria has not been explained in both proposed algorithms.
2. Psudo algorithms seems to be confusing to me. In the heterogenous case, the client only starts updating if it gets a new global model (in the recieve buffer). However, global model only broadcasts after getting all N clients update. Hence, if the client has been selected several times,without getting a new model, how should they update the models and send new gradients to the sever.
3. The heterogenous algorithm is a little bit confusing. When you randomly select the clients, if a slow clients selected multiple times, would that make the whole process slower? If we are willing to wait for those slow clients, why not getting updates (with same computing budget) from all clients that might be faster?
4. DLSGD-homo in section 2.1 is intoduced in a way that each client has access to the "same" local dataset rather than IID access to a shared dataset. Assuming that each client has different seed (otherwise they will perform the same update, which is meaningless), this case is similar to sampling with replacement in IID optimization, which is different from sampling without replacement that normally happens in Local SGD methods. Could you explain how they can be compared?
5. Assumption 3 of bounded variance seems to be too relaxed for this problem. Mostly it is bounded by the norm of the gradients as well.
6. The main claim of the paper is the linear convergence rate with respect to the number of clients $\mathcal{O}\left(\frac{1}{\sqrt{kT}}\right)$, and compare it to local SGD methods' convergence rate. However, there are improved convergence rate for these algorithms in order of $\mathcal{O}\left(\frac{1}{{kT}}\right)$, which has not been discussed nor compared with in this paper. For instance see [A].
7. The experimental results are limited. The datasets selected are good and diverse. However, the methods they compared their proposed algorithms with is not sufficient to make the decision. They should compare with more SOTA federated learning and local SGD methods. Especially FL methods with variance reduction could be useful in this scenario to avoid the fast clients dominate the global model updates.


[A] Haddadpour, Farzin, and Mehrdad Mahdavi. "On the convergence of local descent methods in federated learning." arXiv preprint arXiv:1910.14425 (2019).

### Questions
Questions are covered in the last section.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper combines asynchronous sgd and local sgd, and then provide convergence analysis. The novelty is incremental and the convergence analysis is straightforward.

### Strengths
The problem studied is important.

This paper gave a good literature review.

### Weaknesses
1. The convergence rate of (Stich, 2019) provided in this paper is NOT correct. 

2. The novelty is incremental. The convergence analysis is relatively standard. The proof is a straightforward combination of federated learning and asynchronous sgd.  

3. The number of iterations has a quadratic dependence on $\lambda$. It is suboptimal. [1] can achieve a sharper bound. 

[1] Sharper Convergence Guarantees for Asynchronous SGD for Distributed and Federated Learning

### Questions
1. The convergence rate of (Stich, 2019) provided in this paper is NOT correct. 

2. The novelty is incremental. The convergence analysis is relatively standard. The proof is a straightforward combination of federated learning and asynchronous sgd.  

3. The number of iterations has a quadratic dependence on $\lambda$. It is suboptimal. [1] can achieve a sharper bound. 

[1] Sharper Convergence Guarantees for Asynchronous SGD for Distributed and Federated Learning

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
