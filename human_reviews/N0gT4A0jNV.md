# Low Rank Matrix Completion via Robust Alternating Minimization in Nearly Linear Time

- Avg Score: 6.00
- Decision: Accept (poster)
- Scores: 6, 6, 6

## Abstract
Given a matrix $M\in \mathbb{R}^{m\times n}$, the low rank matrix completion problem asks us to find a rank-$k$ approximation of $M$ as $UV^\top$ for $U\in \mathbb{R}^{m\times k}$ and $V\in \mathbb{R}^{n\times k}$ by only observing a few entries specified by a set of entries $\Omega\subseteq [m]\times [n]$. In particular, we examine an approach that is widely used in practice --- the alternating minimization framework. Jain, Netrapalli and Sanghavi showed that if $M$ has incoherent rows and columns, then alternating minimization provably recovers the matrix $M$ by observing a nearly linear in $n$ number of entries. While the sample complexity has been subsequently improved, alternating minimization steps are required to be computed exactly. This hinders the development of more efficient algorithms and fails to depict the practical implementation of alternating minimization, where the updates are usually performed approximately in favor of efficiency.

In this paper, we take a major step towards a more efficient and error-robust alternating minimization framework. To this end, we develop an analytical framework for alternating minimization that can tolerate a moderate amount of errors caused by approximate updates. Moreover, our algorithm runs in time $\widetilde O(|\Omega| k)$, which is nearly linear in the time to verify the solution while preserving the sample complexity. This improves upon all prior known alternating minimization approaches which require $\widetilde O(|\Omega| k^2)$ time.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this work, the authors proposed the analysis of a robust and fast matrix completion algorithm based on the alternating minimization method. The total running time is linear in terms of the complexity of verifying the correctness of a completion.

### Strengths
The work provides novel theoretical results on the alternating minimization approach for matrix completion. The results should be interesting to researchers in the optimization and machine learning fields.

### Weaknesses
In my opinion, the presentation of the paper can be improved. The current paper spends too much space on providing the intuition of the proposed results. This makes the paper too dry to understand. In addition, I think the authors need to include more technical details in the main body of the paper. Due to the time limit, I do not have time to check the appendix. So I cannot be sure about the correctness of the theoretical results given the limited information in the main manuscript.

### Questions
(1) Page 2, line 1: \epsilon is not defined.

(2) Page 2: "as the perturbation to incoherence itself is not enough the algorithm approaches the true optimum". It seems that the sentence is not complete.

(3) Section 2: it might be better to compare the running time and sample complexity of alternating minimization and (stochastic) gradient descent.

(4) Page 3: "weighted low rank approximation" -> "weighted low-rank approximation"

(5) Page 5: the (total) instance size for the multiple response regression is not defined.

(6) Section 4: I wonder if the partition of samples across iterations can be avoided? Namely, can we use all samples in different iterations? It would be better to clarify if the reuse of samples will fail the proposed algorithms, or only makes it technically difficult to prove the theoretical results.

(7) Algorithm 1, line 5: it seems that U_0 is not used later in the algorithm.

(8) Section 4.1: "but to conclude the desired guarantee on the output, we also need to show that..." It would be helpful to be more specific on the connection between the desired guarantee and the incoherence condition of \hat{U}_t, \hat{V}_t.

(9) Theorem 4.2: I think it might be better to provide the main results (Theorem 4.2) a little earlier in the paper. The preparation for the statement of the main results is too long. The contents of Section 4 is basically a more detailed version of Section 1. Given the page limit, I feel that it is more efficient to simplify the discussion of techniques in Section 1, but include more details in Sections 4.1-4.3.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the low rank matrix completion problem using alternating minimization. Existing algorithms based on alternating minimization for this problem (Jain, Netrapalli and Sanghavi) takes time $\tilde{O}(|\Omega|k^2)$ time where $\Omega$ is the set of entries samples (nearly linear in $n$ for an $m \times n$ incoherent matrix $M$) and $k$ is the target rank $(k<<m,n)$. The main computational bottleneck in alternating minimization comes from solving two multiple response regressions per iteration (once for U and once for V). The algorithm presented in this paper proposes solving the regression problems approximately using off-the shelf sketching based solvers which take nearly linear time in input size per iteration and thus, the time for solving each regression problem reduces to $\tilde{O}(|\Omega|k)$ (with extra $\log(1/\epsilon)$ steps for convergence). However, this complicates the analysis of the algorithm as the solution at every iteration cannot be written exactly in factorized form (so the previous analysis doesn't carry through). This runtime is equal to verification time for the solution upto polylog factors. To analyze this, an inductive argument is presented which shows that at every step, the approximate solution for $U$ or $V$ is close to the optimal solution $U^*$ and $V^*$. Moreover, it is shown that the incoherence of the exact solutions to the regression problem is preserved. Finally, to show that the incoherence of the approximate solution is also preserved, some matrix perturbation bounds are developed which show that as long as any two matrices are very close in spectral norm and one matrix is incoherent, the other matrix will also be incoherent. The sample complexity for the proposed algorithm is the same as that of the old algorithm.

### Strengths
1) The algorithm presented improves upon the runtime to make it nearly linear in verification time of the solution (up to log factors) i.e. $\tilde{O}(|\Omega|k)$. Previous alternating minimization based algorithms try to solve the regressions exactly and hence incur $\tilde{O}(|\Omega|k^2)$ time. Moreover, the proposed algorithm is practical and easy to implement as different off-the-shelf sketching based solvers can be used for this regression step. Also, the sample complexity remains the same as previous algorithms.

2) Some interesting technical result are developed for the theoretical analysis of the algorithm. Specifically, some matrix perturbation bounds are proven which show that if a matrix with incoherent rows and columns is close in spectral norm to another matrix, that matrix will also be incoherent with the incoherence factor depending on the condition number of the first matrix. This seems to be an interesting result which could be of independent interest (though I have some questions related to the proof, please see the questions section).

Remark: I haven't checked all the proofs in the appendix closely (especially the proofs related to induction in Section E and F of the appendix).

### Weaknesses
1) Though the runtime of the proposed algorithm is nearly linear in verification time and improves on the runtime compared to previous algorithms, without any discussion on computational limitations or lower bounds, it is hard to judge if this is indeed a significant theoretical result for this problem. Some discussion on runtime or sample complexity lower bound could be useful to understand what is the runtime one should be aiming for this problem.

2) I'm unsure of certain key steps in the proofs for the forward error bound and the matrix perturbation results (please see the questions).

3) The proofs in the appendix seems some confusing notations and sometimes uses certain notations without defining first them which cause problems while reading the proofs:

  i) For example, in Lemma B.6, in some places $D_{W_i}$ seems to be indicate a matrix with $W_i$ on diagonal and on other places, a constant? When defining $\| z\|_w^2$ is should be just ||z ||_w^2=\sum_{i=1}^n  w_i^2   I think for a vector $w$?

  ii) Also, in definition 3.5, M is factorized as $U \Sigma V^T$ while in the appendix, it seems $U^* \Sigma^* V^*$ is used?


Though the paper has interesting results, I'm recommending a borderline reject with a major concern being some of the key steps in the proofs (please see the questions).

### Questions
I could be misunderstanding the following steps in the proofs:

1)  In forward error bound of Lemma B.5, I'm confused why the step ||Ax'-b-(Ax_{OPT}-b)||_2^2=||Ax'-b||_2^2-||Ax_{OPT}-b||_2^2 should be true. Why should the Pythagorean theorem hold in this case? Is Ax_{OPT}-b orthogonal to Ax'-b-(Ax_{OPT}-b) due to some condition? 
Also, it seems A is assumed to have full column rank for the proof. Is A guaranteed to have full column rank whenever this result is applied i.e. in all iterations wheneer B.7 is invoked?

2) In Lemma G.3, I'm not able to understand how $\sigma_{min}(B) \geq 0.5\sigma_{min}(A)$ follows from $||A-B|| \leq \epsilon_o \leq \sigma_{min}(A)$.

### Soundness
2 fair

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors give a nearly linear time algorithm (in the number of samples) for low-rank matrix completion. Specifically, they give a $O(|\Omega| k)$ time alternating-minimization based algorithm that converges to the original underlying rank-$k$, $\mu$-incoherent matrix $M \in \mathbb{R}^{m \times n}$ when $|\Omega| = \tilde{O}(\kappa^2 \mu^2 n k^{4.5})$ samples are drawn from it. 

The running time guarantee improves on a line of works, starting with that of Jain et. al (2013), on the efficiency of each step in AM framework -- going from $O(|\Omega| k^2)$ running time to $O(|\Omega| k)$ time. This compares however to a recent paper (that uses a different approach altogether) by Kelner et. al (2023) also achieving a $O(|\Omega| k)$ running time with significantly fewer samples $|\Omega| = O(n k^{2 + o(1)})$.

The improvement comes from solving each multiple response regression problem (to obtain the low-rank factorization $UV$) approximately instead of exactly. 

The authors main technical contribution is in analyzing the how the error introduced in solving for $U$ and $V$ approximately, propagates in the iterative process. Specifically they show, using a careful double induction argument and an incoherence bound on the perturbation of row norms of incoherent matrices that the incoherence of the approximate factors in the $t$-th iteration $\hat{U}_t, \hat{V}_t$ as well as the exact solutions they are approximating $U_t, V_t$ are incoherent as well as approach the true subspaces $U^*, V^*$. 

\textbf{References}

Prateek Jain, Praneeth Netrapalli, and Sujay Sanghavi. Low-rank matrix completion using alternating
minimization. In Proceedings of the forty-fifth annual ACM symposium on Theory of computing,
pp. 665–674, 2013

Jonathan Kelner, Jerry Li, Allen Liu, Aaron Sidford, and Kevin Tian. Matrix completion in almostverification time. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science,
FOCS’23, 2023

### Strengths
Originality and Significance 
The main contribution is the error analysis in the AM iterations, showing how the subspaces of the approximate solutions to the multiple response regressions $\hat{U}_t, \hat{V}_t$ converge to the true factors. The novelty comes from a double induction argument tying the incoherence and the closeness of the approximate solution to that of the exact solution in each iteration.

The technique sheds light on how AM algorithms for low-rank matrix completion can be sped-up (using approximate solvers). Since AM algorithms are popular in practice for this problem, this theoretical result can help substantiate the design of new more efficient algorithms.

Quality and Clarity 
Overall the paper is well organized and written. The paper compares to relevant works sufficiently well and highlights the difference facets in which this result compares.

### Weaknesses
The main weakness might be in the significance of the final running time result in the context of the recent result by Kelner et. al (2023). Given that Kelner et. al achieve a significantly lower sample complexity of $\tilde{O}(n k^2)$ (with no dependence on $\kappa$), the novelty of this result could be questioned. Especially since the result is theoretical and no experiments have been provided to justify the efficiency of this approach.

### Questions
- Can you speak more to the significance of your result as compared to that of Kelner et. al (2023)? Specifically to the significance of the running time result given they achieve an asymptotically smaller running time result (please correct me if that is incorrect).

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
