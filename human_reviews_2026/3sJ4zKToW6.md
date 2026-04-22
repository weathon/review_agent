# Consistent Low-Rank Approximation

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 8, 4

## Abstract
We introduce and study the problem of consistent low-rank approximation, in which rows of an input matrix $\mathbf{A}\in\mathbb{R}^{n\times d}$ arrive sequentially and the goal is to provide a sequence of subspaces that well-approximate the optimal rank-$k$ approximation to the submatrix $\mathbf{A}^{(t)}$ that has arrived at each time $t$, while minimizing the recourse, i.e., the overall change in the sequence of solutions. We first show that when the goal is to achieve a low-rank cost within an additive $\varepsilon\cdot||\mathbf{A}^{(t)}||_F^2$ factor of the optimal cost, roughly $\mathcal{O}\left(\frac{k}{\varepsilon}\log(nd)\right)$ recourse is feasible. For the more challenging goal of achieving a relative $(1+\varepsilon)$-multiplicative approximation of the optimal rank-$k$ cost, we show that a simple upper bound in this setting is $\frac{k^2}{\varepsilon^2}\cdot\text{poly}\log(nd)$ recourse, which we further improve to $\frac{k^{3/2}}{\varepsilon^2}\cdot\text{poly}\log(nd)$ for integer-bounded matrices and $\frac{k}{\varepsilon^2}\cdot\text{poly}\log(nd)$ for data streams with polynomial online condition number. We also show that $\Omega\left(\frac{k}{\varepsilon}\log\frac{n}{k}\right)$ recourse is necessary for any algorithm that maintains a multiplicative $(1+\varepsilon)$-approximation to the optimal low-rank cost, even if the full input is known in advance. Finally, we perform a number of empirical evaluations to complement our theoretical guarantees, demonstrating the efficacy of our algorithms in practice.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies low rank approximation in a streaming model, where in addition to standard goals of small space and update time, they also do not want the provided solution to change too much across the lifetime of the stream.  This is modeled as "recourse" which is the sum of squared distances between subspaces at each step.

### Strengths
Standard streaming subspace approximation algorithms like FrequentDirections and Ridge Leverage Score Sampling can have very large recourse as shown theoretically, and empirically on real data.  That means they can bounce between solutions.  

The algorithm is subtle yet simple.  It is careful about when to update the estimate with extra care to not to change the subspace too much if it does not have to.  It reminds me of distributed streaming algorithms (e.g., https://arxiv.org/abs/1404.7571) that try to minimize total communication of updates, but with focus on ensuring a stable answer.  

I think the recourse setting is natural and useful.  It is a nice way to quantify stability of the sketch.  

A strength is that feels like a complete paper on this topic.  It has a variety of upper bounds for additive and relative error, and shows lower bounds on recourse that asymptotically match the upper bounds.  There are basic experiments that show that the algorithm is not just theoretical, but works in practice -- whereas baselines like FrequentDirections does not.

### Weaknesses
Nothing to note.

### Questions
None.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper studies the online low rank approximation problem. In this problem one is given a matrix $A\in \mathbb{R}^{n\times d}$ with integer entries bounded by $M$ and whose rows $a_1,\ldots, a_n$ arrive one by one. Let $A^{t}$ denote the matrix of the first $t$ rows at time $t\in [n]$, the goal is to output a matrix $V^{t}\in \mathbb{R}^{k\times n}$ such that $A^{t}(V^{t})^T V^{t}$ is a $1+\epsilon$ approximation rank $k$ approximation to $A^{t}$ at every time $t\in [n]$. In particular the paper studies \emph{consistent} algorithms for online low rank approximation. More precisely the goal is to minimize the recourse of the algorithm measured as 

$$\sum_{t=1}^n \|P_A-P_B\|_F^2$$

for $A = V_t$ and $B = V_{t-1}$ where $P_{V}$ is the orthogonal projection matrix of the subspace spanned by $V$. Thus a low recourse algorithm ensures that the subspace of low rank approximation does not change drastically on average over the stream. Note that recourse of $nk$ can be achieved trivially by computing the best rank $k$ approximation at each step from scratch.

The first result shown in the paper is an algorithm that achieves a recourse of $O((k/\epsilon)\log(ndM))$ but incurs an additional additive error $\epsilon \|A^{t}\|_F^2$ at each step $t\in [n]$. Furthermore they show that a recourse of $O((k/\epsilon^2)\log^2 n)$ assuming an online condition number of poly$(n)$ and no additive error. Finally for matrices with integer entries they also obtain improved bounds. On the negative side they prove a lower bound on the recourse of $\Omega(n/\epsilon \log(n/k))$ for obtaining a $1+\epsilon$ approximation at every time step by constructing a hard sequence of rows.

### Strengths
The paper introduces a novel model for studying low rank approximation of consistency. Consistent and low recourse algorithms have been studied for other problems in data science thus making low rank approximation a natural problem to study from a theoretical perspective. Moreover the authors show good upper and lower bounds for low rank approximation in this model.

### Weaknesses
The paper does not have many weaknesses but one is that although the problem has a natural theoretical motivation, it would be interesting for the authors to discuss more concrete practical motivations for studying low recourse algorithms for low rank approximation.

### Questions
Although the authors very briefly discuss potential applications in feature engineering, it would be interesting to see if there are any concrete applications of low recourse algorithms for low rank approximation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the problem of low-rank approximation (LRA). Specifically given a matrix $A$, this work studies the problem of approximating $A$ with a matrix $AV^TV$, such that $||A-AV^TV||_F^2 \leq (1+\epsilon)||A-A_k||_F^2$ where $A_k$ is the best rank-$k$ approximation of $A$, and rows of $A$ arrive sequentially in time. This is a very widely studied problem. A primary contribution of the paper is the following problem: given rows of $A$ arrive sequentially over time, define measure called recourse computed as $||P_t - P_t-1||_F^2$ where $P_t$ is the orthogonal projection matrix corresponding to $V^TV$ at time $t$. This work studies LRA through the lens of recourse and demonstrates that -- 1) when the goal is to approximate $A$ with $\epsilon$ additive error, an $O(k\log(nd)/\epsilon)$ recourse is feasible, 2) when the goal is to approximate $A$ with $1+\epsilon$ multiplicative error, an $O(k^2\text{poly}\log(nd)/\epsilon^2)$ recourse is feasible. This is further improved to $k^{3/2}\text{poly}\log(nd)/\epsilon^2$ for matrices with integer entries that are bounded, and $k^{2}\text{poly}\log(nd)/\epsilon^2$ when condition number is bounded. A lower bound of $\Omega(k\log(n/k)/\epsilon)$ is also shown for $1+\epsilon$ multiplicative approximation algorithms.

### Strengths
- The problem setting is interesting, i.e., studying of the subspace corresponding to streaming updates and understand how subspace can differ for different algorithms is an interesting idea. Mostly because one can imagine having to reconstruct the approximation matrix again and again if the subspace is changing significantly (e.g., as stated for the Frequent directions method). 

- I have only glossed over the proofs, which are pretty simple, and believe they are correct. Given the authors present a lower bound to the problem, it helps us ground the theoretical upper bounds presented in this work. 

- I really appreciate the simple algorithms which helps maintain the approximation at time $t$. The algorithm basically checks importance of an incoming row by first identifying the bottom $\sqrt{k}$ singular values among the top $k$ singular vectors. If these vectors have very low spectral contribution, they are "disposable" and so can be replaced by any incoming vector. 

- Good empirical evaluations help us understand how the algorithms presented here work in practice.

### Weaknesses
- There is a significant body of work on rank-$k$ approximation algorithms. However, only frequent directions has been empirically compared against. I am surprised as why this is the case.

- Most of the theoretical contributions are really derivative of prior work. While I really appreciate the problem setting, the contributions are really understanding how the subspace are drifting with time given the subspace approximation algorithm. 

- Algorithm 2 requires computing SVD at each round in the worst case. So while one may be easily able to reduce recourse, the run time of the algorithms grows with $tk^3$, which seems expensive!

- For distributional shifts, just checking the bottom $\sqrt{k}$ may not be enough, e.g., for windowed algorithms due to Braverman et. al. 2020, or the works of Musco-Musco, or Cohen et. al. on online leverage score sampling, one might need to re-evaluate samples which was heavy at some point and might become of low importance in future. What do we do then?

### Questions
Please check the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2
