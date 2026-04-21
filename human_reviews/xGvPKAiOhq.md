# How Over-Parameterization Slows Down Gradient Descent in Matrix Sensing: The Curses of Symmetry and Initialization

- Avg Score: 8.00
- Decision: Accept (spotlight)
- Scores: 8, 8, 8, 8

## Abstract
This paper rigorously shows how over-parameterization dramatically changes the convergence behaviors of gradient descent (GD) for the matrix sensing problem, where the goal is to recover an unknown low-rank ground-truth matrix from near-isotropic linear measurements.
First, we consider the symmetric setting with the symmetric parameterization where $M^* \in \mathbb{R}^{n \times n}$ is a positive semi-definite unknown matrix of rank $r \ll n$, and one uses a symmetric parameterization $XX^\top$ to learn $M^*$. Here $X \in \mathbb{R}^{n \times k}$ with $k > r$ is the factor matrix. We give a novel $\Omega\left(1/T^2\right)$ lower bound of randomly initialized GD for the over-parameterized case ($k >r$) where $T$ is the number of iterations. This is in stark contrast to the exact-parameterization scenario ($k=r$) where the convergence rate is $\exp\left(-\Omega\left(T\right)\right)$. Next, we study asymmetric setting where $M^* \in \mathbb{R}^{n_1 \times n_2}$ is the unknown matrix of rank $r \ll \min\{n_1,n_2\}$, and one uses an asymmetric parameterization $FG^\top$ to learn $M^*$ where $F \in \mathbb{R}^{n_1 \times k}$ and $G \in \mathbb{R}^{n_2 \times k}$. We give the first global exact convergence result of randomly initialized GD for the exact-parameterization case ($k=r$) with an $\exp\left(-\Omega\left(T\right)\right)$ rate. Furthermore, we give the first global exact convergence result for the over-parameterization case ($k>r$) with an $\exp\left(-\Omega\left(\alpha^2 T\right)\right)$ rate where $\alpha$ is the initialization scale. This linear convergence result in the over-parameterization case is especially significant because one can apply the asymmetric parameterization to the symmetric setting to speed up from $\Omega\left(1/T^2\right)$ to linear convergence. Therefore, we identify a surprising phenomenon: asymmetric parameterization can exponentially speed up convergence. Equally surprising is our analysis that highlights the importance of imbalance between $F$ and $G$. This is in sharp contrast to prior works which emphasize balance.  We further give an example showing the dependency on $\alpha$ in the convergence rate is unavoidable in the worst case. On the other hand, we propose a novel method that only modifies one step of GD and obtains a convergence rate independent of $\alpha$, recovering the rate in the exact-parameterization case. We provide empirical studies to verify our theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the matrix sensing problem where one observes y_i = <A_i, M^*> and A_i, and aims to estimate M^*. 
This problem makes sense for either symmetric or asymmetric matrices. 
The most significant contribution is that this paper unveils a surprising phenomenon that even for the symmetric version of the problem, introducing asymmetry in the initialization and the parametrization produces qualitatively faster convergence rate.

### Strengths
The matrix sensing problem requires no more motivation (at least to me) and the results in this paper bring further insights to this classical problem. 
The paper is reasonably well-written and the results are definitely sufficiently interesting for ICLR. 
The punchlines (the fact that asymmetry helps and why/how this is the case) are clearly addressed within the first 4 pages. 
Sec 4.1 is pedagogically helpful. 
Several versions of the problems are treated to a reasonably systematic extent.

### Weaknesses
I don't see major weaknesses. 
Please see technical comments below.

### Questions
1. All results only assume RIP for A_i. If A_i are i.i.d. Gaussian matrices, is it possible to derive sharper or even asymptotically exact (in the sense of e.g. https://arxiv.org/abs/2207.09660) results?

2. Could the authors comment on how crucially the results rely on the "linearity" of the problem? Does it make sense to consider a "generalized" matrix sensing problem in which y_i = phi(<A_i, M^*>) for some non-linearity phi? This is somewhat motivated by other models with similar structures such as generalized linear models or single-index models. I guess the information exponent of phi or something like that will play a role in the convergence rate. 

3. In Sec 5, an accelerated method is proposed. In particular, step (5.1) should be executed once the iterates are sufficiently close to the optimum. But in practice, how can one verify this neighborhood condition? Note that Sigma is unknown. Please let me know if I missed something simple here. 

4. It seems that both the model and the algorithms are deterministic. What happens if the observations are noisy?

5. It's claimed on top of page 6 that the results easily extend to the rectangular case. Could the authors state such results formally (even without formal proofs)? I'm curious to see how the results depend on the aspect ratio n_2 / n_1. In fact, if the matrices are extremely rectangular (e.g. n_2 / n_1 is growing or decaying), I actually doubt if such extensions are so straightforward. Thanks in advance for the clarification. 

6. Lemma G.1 assumes x, y are "random vectors". Are they actually independent and uniform over the sphere? For generic joint distribution, not much can be said about their angle. Please make the statement more precise. 

Minor notational/grammatical issues. 
1. The ground truth is interchangeably denoted by M^* or M^\star. I suggest stick to M^\star to avoid conflict with adjoint operator. 

2. In the title of Sec 1.2, where is the word "symmetric" repeated twice?

3. Statement of Theorem 1.3: t-the iteration --> t-th iteration. 

3. Page 4: which we require it to be small --> which we require to be small.

4. Description of Table 1: by "row" I think the authors meant "column". 

5. Right after equation (2.2): definition of A should be A^*.

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this work, the authors provided the analysis on the different convergence rates when exact-parameterization or over-parameterization are used. They also proposed a new algorithm to avoid the dependence of the convergence rate on the initialization rate for the asymmetric and over-parameterized case.

### Strengths
The results of this paper are novel and should be interesting to audiences in optimization and machine learning fields. The theory provides an explanation for the slow-down of GD in the over-parameterized case, and the paper offered a partial solution to this problem. However, due to the time limit, I cannot check the appendix. So I am not sure about the correctness of the results in this work.

### Weaknesses
I can only see a few minor problems with the presentation. For example, the requirement on the sample complexity can be briefly discussed when the informal results are introduced.

### Questions
(1) Theorem 1.1: it would be better to say that each entry of X is independently initialized with Gaussian random variable with variance \alpha^2. Similar comment applies to other theorems.

(2) In Section 1, I think the authors did not mention any requirements on the sample size m. It might be better to briefly mention the requirement on the sample complexity or the RIP constant in Section 1.

(3) For the asymmetric case, I think most convergence results require a regularization term \|F^TF - G^TG\|_F^2 to penalize the imbalance between F and G. It would be better to mention the intuition why the regularization term is not required in this work.

(4) After Theorem 1.3: I think it should be "Comparing Theorem 1.3 and Theorem 1.1".

(5) Section 1.3: It might be better to also mention the current state-of-the-art results on landscape analysis:

Zhang, H., Bi, Y., & Lavaei, J. (2021). General low-rank matrix optimization: Geometric analysis and sharper bounds. Advances in Neural Information Processing Systems, 34, 27369-27380.

Bi, Y., Zhang, H., & Lavaei, J. (2022, June). Local and global linear convergence of general low-rank matrix recovery problems. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 36, No. 9, pp. 10129-10137).


(6) Section 2: "Asymmetric Matrix Sensing"

(7) Theorem 3.1: it seems that the "ultimate error" does not appear in Section 3.1.

(8) Also, it might be better to mention that the over-parameterization size k depends on \alpha and briefly explain what happens if the size k is smaller than this threshold.

(9) In (3.3a), I think T should be T^{(0)}?

(10) Below Theorem 3.1: For the inequality \|X_tX_t^T - \Sigma\|_F^2 \geq A_t / n, I wonder if it can be improved to \|X_tX_t^T - \Sigma\|_F^2 \geq A_t?

(11) I wonder if there is a reason that initialization scales are chosen as \alpha and \alpha/3? Would it be possible to use, for example, \alpha and \alpha / 10 to achieve a better convergence rate?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The reviewed paper is a theoretical investigation of the convergence properties of gradient-descent, and other first-order based methods, for over-parameterized matrix factorization/sensing for symmetric matrices. The specific focus is on the role of using **symmetric** versus **general** Burer-Monteiro factorization as parameterization and how it effects the convergence properties. The unexpected result is that the *symmetricity* versus *imbalance* plays a significant role.

The main "positive" result states that the over-parameterized gradient descent on $FG^T$ factorization is able to achieve linear convergence when the two components are imbalanced in the sense of the spectrum of $\Delta = F^\top F - G^\top G$, and the specific convergence rate depends on this imbalance. The main "negative" result shows that there will always exist a positive measure of cases when symmetric parametrization $FF^\top$ cannot have faster than sublinear convergence.

The work provides simple, but well explained numerical examples of small matrix sizes ($50 \times 50, \mathrm{rank} =3$) that clearly demonstrate this phenomenon.

The proofs take more than 30 pages in the appendices, they are technically involved and not easy to check in their entirety, but at first sight the result seems correct.

### Strengths
I believe this paper has several very strong points:
* It presents a novel and surprising result
* It gives rigorous proofs for the two main statements which together describe a very interesting behaviour
* The numerical examples corroborate the proven theory
* The paper is very clearly written, the structure and main message is clear (although the theorems themselves can be a bit complicated to interpret)
* It gives a very good comparison with existing literature

### Weaknesses
There is not much that I would consider a weakness to this paper. That said, I would like to know, how much the results of the numerical experiments in terms of the neat convergence rate depend on a specific initialisation of the methods and whether these result would also occur for larger ranks and problem sizes.

### Questions
1) In Fig 2 we see that for larger $alpha$ the convergence rate is faster. What is the limit of how large $\alpha$ can be?
2) Do the numerical results hold also for larger ranks of the true matrix and over-parameterized ranks? Also larger imbalance of ranks, lets say k = 20 and r = 5?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides several new results for over-parameterized matrix sensing. First, the authors rigorously prove that with a symmetric parameterization, over-parameterization slows down GD.  In particular, they give a lower bound rate of $\Omega(1/T^2)$. Second, the authors also show that with an asymmetric parameterization, GD converges at an linear rate depending on the initialization scale. This is in contrast with GD with symmetric parameterization, which has a sublinear rate. Finally, the authors extend their algorithm so that the linear convergence rate is independent of the initialization scale.

### Strengths
Overall I think this is a good paper. The fact that over-parameterization slows down GD for matrix sensing has been observed by quite a few previous papers. However, this is the first paper that I'm aware of to rigorously establish a lower bound. The authors also show that with asymmetric parameterization, GD converges at an exponential rate that depends on the initialization scale. This is somewhat surprising, given that the asymmetric case has traditionally been considered harder due to potential imbalance of the factors.

### Weaknesses
My main concern is with the experiments in this paper. I think the paper could benefit from a more thorough experimental section, perhaps in the appendix. 

In the symmetric case, if we use GD with small initialization, then it is often the case that GD goes through an initialization phase where the loss is relatively flat, and then converges rapidly to a small error. However, in the experiments in Figure 2, I do not see this initialization phase in Figure 2b. Instead, linear convergence is observed right from the start, even when a small initialization is used. I wonder why is this the case? For the asymmetric case, is the initialization phase much faster?

Additional experiments which i think should be nice: on the same plot, compare the convergence of asymmetric versus symmetric parameterization, using the same initialization. Also perform the experiment for different initialization scales. I think the authors should also plot convergence for ill-conditioned versus well-conditioned matrices, as GD with small initialization performs differently based on the eigenvalues. 

In any case, i would like to see a more detailed comparison of symmetric versus asymmetric parameterization, even just using synthetic experiments.

### Questions
In Theorem 1.3, the convergence rate depends on the initialization scale $\alpha$. This is also observed empirically in figure 2b. In practice, does this mean that small initialization has no advantage? One could just set $\alpha$ to be large to ensure rapid convergence?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
