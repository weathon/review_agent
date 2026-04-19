# Efficient Alternating Minimization with Applications to Weighted Low Rank Approximation

- Decision: Accept (Poster)
- Scores: 6, 5, 6, 6

## Abstract
Weighted low rank approximation is a fundamental problem in numerical linear algebra, and it has many applications in machine learning. Given a matrix $M \in \mathbb{R}^{n \times n}$, a non-negative weight matrix $W \in \mathbb{R}_{\geq 0}^{n \times n}$, a parameter $k$, the goal is to output two matrices $X,Y\in \mathbb{R}^{n \times k}$ such that $\\| W \circ (M - X Y^\top) \\|_F$ is minimized, where $\circ$ denotes the Hadamard product. It naturally generalizes the well-studied low rank matrix completion problem. Such a problem is known to be NP-hard and even hard to approximate assuming the Exponential Time Hypothesis. Meanwhile, alternating minimization is a good heuristic solution for weighted low rank approximation. In particular, [Li, Liang and Risteski, ICML'16] shows that, under mild assumptions, alternating minimization does provide provable guarantees. In this work, we develop an efficient and robust framework for alternating minimization that allows the alternating updates to be computed approximately. For weighted low rank approximation, this improves the runtime of [Li, Liang and Risteski, ICML'16] from $\\|W\\|_0k^2$ to $\\|W\\|_0 k$ where $\\|W\\|_0$ denotes the number of nonzero entries of the weight matrix. At the heart of our framework is a high-accuracy multiple response regression solver together with a robust analysis of alternating minimization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors proposed a new alternating minimization method for the weighted low-rank matrix approximation problem. They utilized an efficient multiple-response regression solver to reduce the computational time from ${O}((\|W\|_0k^2 + nk^3)\log(1/\epsilon))$ to $\tilde{O}((\|W\|_0k + nk^3)\log(1/\epsilon))$.

### Strengths
The paper provides a novel algorithm for the low-rank matrix approximation problem. The results in this work may also be applied to other optimization and machine learning problems. After checking the paper, I think the theoretical results in the paper should be sound, although I did not check all proofs due to the time limit.

### Weaknesses
Although this paper is mostly a theoretical paper, I think it is still beneficial to include numerical illustrations to verify the theoretical findings and show the usefulness of the proposed method. For example, the authors may generate artificial datasets that satisfy the assumptions in this work and compare their method with the methods in (Li et al., 2016). It would be even better if the authors can verify that Assumption 2.3 holds in some practical examples and exhibit the empirical performance on these real-world examples.

Moreover, in Line 223, the authors claimed that Assumption 3 is a generalization to the RIP condition. Although I agree with the authors that the assumption is more general in the sense that it contains the weight matrix $W$, I think it is more restrictive than the RIP condition. Suppose that $W$ is a boolean matrix and each column of $W$ contains at most $k$ 1's. In this case, the assumption considers at most $n$ combinations of rows of $U$ and $V$. In contrast, the RIP condition requires the error bound to hold for all $k$-dimensional subspace. In addition, I feel that Assumption 3 is not directly comparable with the classical RIP condition. The definition of the classic $(r, \delta)$-RIP condition would imply that $(1-\delta) \lVert K \rVert_F^{2} \leq \lVert W \circ K \rVert_F^{2} \leq (1+\delta) \lVert K \rVert_{F}^{2}$ holds for any matrix $K$ of rank at most $k$, which is independent of $M^*$. I would suggest the authors include more detailed explanations to this claim.

### Questions
I have a few other minor comments for the authors to consider:

- Theorem 4.6: I wonder if the time complexity also depends on the scale of $M^*$ and $N$. For example, let $c>0$ be a constant. If we multiply $M^*$ and $N$ by $c$, then the solution $\tilde{M}$ will also be multiplied by $c$. However, the value $\epsilon$ is not changed in the error bound. In addition, Assumption 2.3 does not bound the scale of $M^*$ and $N$ either. If I did not miss anything, I feel that the number of iterations or the error $\epsilon$ should depend on the scale of $M^*$ and $N$.


- Line 499: it would be helpful if the authors could provide supporting evidence for the claim that the spectral norm is more robust than the Frobenius norm.

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
4

### Summary
This paper considers the weighted low-rank approximation problem, which can be solved using alternating minimization. In previous methods, the alternating minimization algorithm requires solving n linear regression problems each iteration. In this work, the authors develop a framework based sketching, and they prove that alternating minimization is robust enough such that the approximate solution return each iteration also converges towards the optimal solution. As a result, they obtain an improved running time compared to Li et al. 2016.

### Strengths
I think the theoretical results in this paper is solid: the proof that the alternating minimization framework is robust enough to tolerate large errors induced in the approximate solution is interesting, and can possibly be generalized to other similar problems in matrix optimization.

### Weaknesses
I see two main weaknesses. First, I think the writing of this paper needs to be improved, especially in section 3, where the algorithm is first introduced. Algorithm 1 is outlined in a dense way that makes it hard for the reader to parse exactly what is going on. Also, lines 270-280 contain numerous sentences that are a bit awkward or hard to understand. I suggest that the authors use a separate section to slowly present algorithm 1 step by step, and describe which parts are new. 

Another weakness i see is the lack of any numerical results. In the introduction (lines 80) the authors say that the results of Li et al. 2016 are hard to deploy in practice because of the high time complexity. While this paper reduces the time complexity in theory, there are no numerical section to demonstrate that the reduction in time complexity is useful in practice. I suggest that the authors at least run some synthetic experiments to demonstrate a reduction in real runtime.

### Questions
Can the authors comment on the performance of their algorithm on some real or even synthetic problems?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors propose an efficient framework for alternating minimization, which allows the subproblems to be solved approximately. They apply the framework to the weighted low rank approximation problem, theoretically improving the runtime.

### Strengths
1. The authors propose a high-precision regression solver and extend it to the weighted case. They also demonstrate the robustness of the alternating minimization framework. The theoretical analysis is a significant contribution.
2. The motivation behind their method is detailed and well-presented in the paper.

### Weaknesses
The authors did not conduct experiments to verify the performance of their algorithm. I am interested in whether the algorithm is feasible in practice.

### Questions
1. Could the authors explain the connection between the problem they considered (Problem 2.4) and the original problem? Are they equivalent under some conditions?
2. Could the authors explain the meaning of $\epsilon_0$ in Line 351?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies the problem of weighted low-rank matrix approximation. In this problem one is given a weight matrix W
 and observes a matrix M', and the goal is to recover a matrix M of rank at most k that minimizes  $\Vert W \circ (M'- M)\Vert$..

This paper provides new theoretical guarantees within the alternating minimization framework for weighted low-rank matrix approximation. The authors propose a new alternating minimization algorithm that uses high-accuracy regression solver based on sketching instead of exact regression solvers in the optimization step of the algorithm compared to the work of (Li et al, 2016). This allows authors to improve runtime by a factor of $k$. 
.

### Strengths
The problem of weighted low-rank matrix approximation is an important practical problem and an important algorithmic primitive that arises in numerical linear algebra, and various machine learning applications. The problem naturally generalizes classical low-rank matrix approximation problems and a matrix completion problem. Alternating minimization is an extremely successful practical approach to matrix completion problem, so establishing theoretical guarantees for this approach to the weighted low-rank matrix approximation problem is of high interest.

This paper builds on top algorithm analyzed in (Li et al., 2016) and the key difference between the algorithm in (Li et al., 2016) and this paper is that exact solvers for linear regression is replaced with an approximate solver that can be traced back to (Rokhlin & Tygert, 2008). This allows authors to improve runtime by a factor of $k$ and to get a nearly optimal runtime.

### Weaknesses
The assumption 2.4.2 asks that $||W - J||<\gamma n$, where J is an all-one matrix with $\gamma = O(n^{-c})$ (compared to O(1) in  (Li et al., 2016)). This assumption sounds quite strong to me. This means that in the matrix completion scenario, this allows one to have only o(1)-fraction of entries not observed in every row. Considering that one of the key regimes for low-rank problems assumes that the rank satisfies k = O(1), I am not sure if improvement in runtime completely justifies slightly stronger assumption on W.

Another slight weakness of the paper is that most of the key techniques used in this paper were introduced in prior work, so the paper lacks a bit of novelty in this sense.

### Questions
Are you aware of any practical application where the improvement in the running time by a factor of k obtained in this paper may be significant to applications, and where one might expect assumptions to hold?

### Soundness
3

### Presentation
3

### Contribution
3
