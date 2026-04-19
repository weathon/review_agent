# Implicit bias of SGD in $L_2$-regularized linear DNNs: One-way jumps from high to low rank

- Decision: Accept (spotlight)
- Scores: 8, 8, 5, 10

## Abstract
The $L_{2}$-regularized loss of Deep Linear Networks (DLNs) with
more than one hidden layers has multiple local minima, corresponding
to matrices with different ranks. In tasks such as matrix completion,
the goal is to converge to the local minimum with the smallest rank
that still fits the training data. While rank-underestimating minima
can be avoided since they do not fit the data, GD might get
stuck at rank-overestimating minima. We show that with SGD, there is always a probability to jump
from a higher rank minimum to a lower rank one, but the probability
of jumping back is zero. More precisely, we define a sequence of sets
$B_{1}\subset B_{2}\subset\cdots\subset B_{R}$ so that $B_{r}$
contains all minima of rank $r$ or less (and not more) that are absorbing
for small enough ridge parameters $\lambda$ and learning rates $\eta$:
SGD has prob. 0 of leaving $B_{r}$, and from any starting point there
is a non-zero prob. for SGD to go in $B_{r}$.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This papers considers the problem of minimizing parameters of a deep linear network with the matrix completion loss. Specifically the authors considered an $\ell_2$ regularized version of this problem and show that with certain conditions on the learning rate and the regularization parameter, SGD can jump from a high-rank local min to a low-rank local min, while it cannot jump from a low-rank local min to a high-rank local min.

### Strengths
I think the theoretical results of this paper are interesting because it gives a nice characterization of the implicit bias of GD/SGD for deep linear networks. The proof that SGD can "jump" from high-rank to low-rank local minima is new to me and I think its a good step towards understand the training dynamics for deep neural networks.

### Weaknesses
I think the main weakness of this paper is that both the theoretical results and the experiments require specific conditions on the $\ell_2$ regularization parameter $\lambda$ and the learning rate $\eta$, which seem to be a bit artificial. For example, the requirement that $\lambda$ is large in Theorem 3.2 seem to artificially cause $\|\theta\|$ to decay more quickly, thus biasing it towards low numerical rank. In the numerical experiments, a similar annealing technique is used run SGD with a large $\lambda$ and $\eta$, before switching to smaller parameters. 

My main point is that the implicit bias observed in this paper could be a result of a deliberate choice of parameters, instead of a natural property of SGD and GD. I hope the authors can clarify this point.

### Questions
Please see previous section. Also, I wonder if the proof in this paper for Theorem 3.2 also works for GD? In other words, is $B_r$ also absorbing for GD?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the optimization landscape of regularized stochastic gradient descent applied to matrix completion with linear networks problems (which is equivalent to matrix completion with a $2/L$ Shatten norm regularizer). Several properties of the optimization landscape are proved, including the fact that the only critical points of the optimization problem over the factor matrices (minimizing $\mathcal{L}_{\lambda}(\theta)$ must be local minima of the optimization problem over the full matrix $A$, unless they are strict saddle points in the original problem.

 In addition, it is shown that if gradient flow converges to a global minimum, then a version of gradient flow with a sufficiently small regularization parameter will converge to a minimum with a larger rank than the ground truth. Arguably the most significant final result is that stochastic gradient descent jumps from high rank local minima to lower rank local minima, with the jumps being one directional: one cannot return to a higher rank region after entering a lower rank region. Here, the lower rank regions should be understood as defined on page 5, in an approximate sense. Throughout the proofs, the fact that the local minima of the optimization problem over $\mathcal{L}_\lambda(\theta)$ must be balanced (cf. Proposition A.1). An approximate version of this condition is also present in the definition of the low absorbing low rank spaces in the main results of the paper.

### Strengths
The main paper is well-written and the results appear generally sound. The results are of great importance to the field and interest to the community. **This is highly non trivial and important work**.

### Weaknesses
Although the main paper is well written, the **proofs are not reader-friendly** at all. 

The writing of the proofs is very terse and laconic, omitting many details. Although this is reminiscent of some great pure mathematics papers that were ahead of their time and I enjoyed the challenge some of the time, I strongly believe this style should only be considered acceptable if there is absolutely zero tolerance for any errors or inaccuracies whatsoever. I don't think the proofs actually stand up to this amount of scrutiny: there are **at least a few typos, minor errors and imprecisions** in the subset of the proofs I was able to look at, and since a lot of information is left out for the reader to figure out, the additional presence of even a small number of actual errors dramatically expands the "search space" from the point of view of the reader.  I would really like to see a substantial revision of the paper with more detailed and careful proofs (and maintaining my score is conditional on that). 

For instance, in page 12, point "(0)", the definition of the $U_i, V_i$ is not really consistent: the index under the $U,V,S$ is used both to mean the iteration step in the sequence and the position in the product $W^L...W^1$. 

In addition, in page 13, consider the following statement the authors make " as $\lambda\rightarrow 0$, the critical points of the loss move continuously. Consider a continuous path of critical points, as $\lambda\rightarrow 0$, it converges to..."
Although the argument makes sense intuitively, filling in the gaps with rigorous proofs is definitely beyond the scope of what can be expected of the reader to do. At least some citations are a minimum. I doubt that simple continuity is enough to guarantee convergence (even if a subsequence converges, the path could oscillate widely), probably the only way to rigorously prove the statement is to use a quantitative version of the statement relying on calculus of variations. 

This is not the only example. In point (1) in page 14, the authors say "the singular value ..... must converge to a non zero eigenvalue".  It is not clear **why this is the case**, or why the the *singular value* turns into an *eigenvalue* after convergence. Far more details are required. 

In the middle of page 14, it is hard to imagine that the equation $U_{\ell,i}(\lambda)U_{\ell-1,i}(\lambda)$ can be correct without **at least a transpose missing**. Of course, the lack of a rigorous and consistent definition of $U_{\ell,i}$ does not help here. 

At the bottom of page 14: the line starting with "other directions" ends with " $L-1,)$" and a few lines below we have the equation $U_\ell^\top dU_{\ell}+ = -dU_\ell^\top U_\ell$. What does "+=" mean here? The same issue is present in many other parts of the paper, including in the third line of text on page 15.

Towards the end of Appendix A in page 17, the term "saddle to saddle" is mentioned with absolutely no explanation or citation. 

In the middle of page 13, the authors use the fact that "a matrix cannot be approached with matrices of strictly lower rank", which is true but should probably warrant a citation since the equivalent statement is not true for tensors. 


The proof of Proposition A4 is very hard to make sense of without further information: the first sentence is ""let A(\lambda) be path of global minima restricted to the set of matrices of rank $r^*$ or less." how do you construct the path? Even for $L=2$, there can be a continuous set of global minima of local intrinsic dimension higher than 2, how do you use the axiom of choice to construct a "path"? 
Sentences such as "going along directions that increase the rank of $A(\lambda)$, the regularization term increases at a rate of $d^{2/L}$ for $d$ the distance" definitely need more mathematical details. 

Similarly, the statement about $\phi$ being differentiable in the directions which do not change the rank should be made more precise (although I agree with it, probably at least a citation to [1] is a minimum) 

For proposition A.5, the proof starts with the following sentence "We know that L2 regularized GF $\theta_\lambda(t)$ converges to unregularized GF $\theta(t)$ as $\lambda\rightarrow 0$". There are two parameters here, $\lambda$ and $t$, is the convergence uniform over all $t$?













========more minor points:=====

Many apologies if I am being picky but as a relative outsider to optimization literature, even the statement that the point 0 is a critical point was not immediately  obvious to me (perhaps either a calculation of the gradient or a mention of the fact that $L>1$ would help). 

In the bottom of page 13, the equation before equation (1) is presumably the end of a sentence, thus the next line should be rewritten. Below, that "no such thing happen" should be "no such thing happens"

Some citation for Fact C.4 (Ky Fan?) would be nice. 


In page 19, just before the beginning of Section D.1. Do the authors mean $G_{\theta,ij}$ instead of $G_{\theta,j}$?


Just above equation (6), $\|W_\ell|^2$ should be $\|W_\ell\|^2$ and the sentence is missing a period. 








[1] Characterization of the subdifferential of some matrix norms, G.A. Watson. 1992, linear algebra and its applications.

### Questions
1. In the third line of page 13, ou mention that the quantity in the limit is strictly positive but possibly infinite. Apologies if I  lack some background knowledge but could you explain your reasoning there? It is not at all obvious to me. 

2 At the beginning of the proof of proposition A5 in the first equation, should the infimum run over  $Rank A>r$ instead of $Rank A<r $ as written?

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper analyzes the matrix completion task with deep linear neural networks. It shows that the critical points that are not local minima can be avoided with a small ridge. And it shows that GD cannot avoid overestimating minima but  SGD can jump from any minimum to a lower rank minimum.

### Strengths
I think analyzing the training dynamics of the deep linear neural networks on the matrix completion task is a very interesting problem. This paper provides insights on the advantages of using SGD to get a low-rank minima and provides experimental results.

### Weaknesses
1. I feel the statement of theorems is not very clear. It uses a lot of ''small enough'', "large enough".  I think the statement should be more rigorous. 

2. This paper claims that it shows GD can avoid rank-underestimating minima by taking a small enough ridge $\lambda$. But Proposition 3.2 is for Gradient Flow with a very strong assumption. I believe there is a gap. 

3. For the function $f_\alpha$, it takes a very specific form. The authors claim that changing $f_\alpha$ with similar properties should not affect the results. I don't see the reason why the condition cannot be extended to more general functions. I believe it could improve the results.

4. A small suggestion is that the proof in the appendix is not easy to read. I think it can be more organized and add more explanation.

### Questions
1. In the remark before section 3.2, it's said that it's possible that GD can recover the minimal rank solution easily. Can you say something about this case? 

2. I have a concern about the constant in Theorem 3.2. It is said $\lambda$ and $C$ are large enough, but an example of acceptable rates in terms of $\lambda$ is $C \sim \lambda^{-1}$. Is it contradictory?

3. In the proof of Theorem 3.2, r columns with the most observed entries are taken.  What if all the columns have the same number of observed entries? Will the $d_{out}-r$ other columns of $W_L$ decay exponentially? I don't see the relation between rank $r$ and the number of observed entries here.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work shows that when applied to matrix completion with deep linear networks, SGD can transition from local minima with higher ranks to solutions with lower ranks, while transitions in the opposite direction are zero. Crucially, this results depends on the gradient distribution of SGD which leads to drastically different outcomes than what common SDE-based models for SGD exhibit. The authors further provide numerical experiments that exhibit the predicted transitions in practice.

### Strengths
This work provides an interesting theoretical insight on the distinction between stochastic and deterministic gradient descent. I find it especially exciting that it provides a concrete example how the gradient distribution of SGD enables phenomena that are not apparent from SDE-based models.

### Weaknesses
I can not think of a weakness. It's a very nice paper in my opinion.

### Questions
As you point out, the common Langevin-based models predict a transition among each pair of points and thus fall short to capture the phenomenon shown by your result. Do you know if they would nevertheless exhibit a systematic low-rank bias (i.e. the transition toward lower-rank solutions being much more likely than toward higher-rank solutions)?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent
