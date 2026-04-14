# Convergence of Distributed Adaptive Optimization with Local Updates

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
We study distributed adaptive algorithms with local updates (intermittent communication). Despite the great empirical success of adaptive methods in distributed training of modern machine learning models, the theoretical benefits of local updates within adaptive methods, particularly in terms of reducing communication complexity, have not been fully understood yet. In this paper, for the first time, we prove that \em Local SGD \em with momentum (\em Local \em  SGDM) and \em Local \em  Adam can outperform their minibatch counterparts in convex and weakly convex settings in certain regimes, respectively. Our analysis relies on a novel technique to prove contraction during local iterations, which is a crucial yet challenging step to show the advantages of local updates, under generalized smoothness assumption and gradient clipping strategy.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors consider problems from distributed optimization and federated learning, where communication can be a bottleneck. In this setup, they analyze the celebrated Local SGD method with momentum, clipping, and Adam steps. In this context, they provide new theoretical convergence rates under heavy-tailed distribution assumptions that hold with high probability.

### Strengths
The considered problem is important in modern optimization. Using local steps is one of the possible strategies to mitigate the communication burden. The assumptions and goals of the paper are very challenging. I have not checked the proofs in detail (that is why I choose confidence 2), but the final theorems seem to be reasonable and consistent with those in previous papers. Overall, this paper appears to be a solid and technical mathematical work that extends the known results of Local SGD.

### Weaknesses
At the same time, I want to point to some weaknesses: 
1. The choice of Assumption 3 is unusual. In most previous works, this assumption holds with $1 < \alpha \leq 2,$ while the authors choose $\alpha \geq 4,$ which is a much stronger assumption. What is the reason for that? The reason should be discussed in the main part.
2. In many places of the paper (abstract, contributions, ...), there is a claim that Local SGDM outperforms the minibatch counterparts. But this is not true in general. [1] point to some regimes when Local SGD is worse. Moreover, (4.5) can be much better than (4.4) if $\sigma$ and $M$ are large. In this regime, the third term in (4.4) can dominate.
3. I agree that when $K$ is large, Local SGDM can outperform Minibatch SGDM. However, looking at (4.4), due to the third term, it is clear that taking $K \to \infty$ is a suboptimal choice (we can't just take $K \to \infty$ because one global iteration would take an infinitely long time), so there should be an optimal choice of $K^*,$ and I'm not convinced that for this choice of $K^*$ Local SGDM will be better than Minibatch SGDM.
4. The rate (4.9) of Local ADAM, in the way it is presented now, does not provide meaningful convergence guarantees due to the norm $||\cdot||_{H_r^{-1}}.$ What happens if the maximum eigenvalues of $H_r$ are huge? I think the authors should bound $H_r$ somehow to get a a more meaningful rate.

Conclusion: Despite all these weaknesses, I think that the authors did a very nontrivial and technical mathematical work, which I appreciate. However, due to time limits and technicalities, I cannot check the proofs in detail, so I will choose a low confidence.

[1]: Woodworth, Blake, et al. "Is local SGD better than minibatch SGD?." International Conference on Machine Learning. PMLR, 2020.

### Questions
(see weaknesses)

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the convergence rates of Adam/SGDM with local steps, showing that local steps can improve the convergence rates in the homogeneous setting.

### Strengths
* This paper analyzes the convergence rates of Local SGDM and Local Adam, showing that their convergence rates are better than Minibatch SGDM and Adam.
* Overall, the reviewer feels that the result shown in this paper is not very surprising, but it is solid.

### Weaknesses
* In Assumption 3, the authors assume $\alpha \geq 4$. The reviewer feels that this assumption is a bit different from the assumption commonly used in the existing literature. For instance, [1] used a similar assumption (see Assumption 1), while they called that the stochastic noise is "heavy-tailed" when $\alpha<2$. Although the authors claimed that it is easy to extend their analysis to the case where $\alpha>4$ in Remark 1, the reviewer feels that the authors should show the analysis with arbitrary $\alpha$ if this extension is really easy. 
* In Theorem 1 and 2, the authors omit the dependence of $\beta_1$ by using $1 - \beta_1 = \Omega(1)$. Thus, we can not discuss the effect of $\beta_1$ on the convergence rate. The reviewer feels that it would be better to show the precise convergence rate, at least in the Appendix.
* Theorem 1 and 2 used the assumption that $\| x - x^\star \| \leq O( \| x_0 - x^\star\| )$. Thus, these theorems only hold when the set of feasible solutions is bounded. Similar to these assumptions, Theorem 3 used the assumption that $f(x) - f^\star$ is bounded.


## Reference
[1] Zhang et. al., Why are Adaptive Methods Good for Attention Models?, in NeurIPS 2020

### Questions
See the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies distributed adaptive algorithms with local updates in the homogeneous data regime, aiming to establish theoretical guarantees proving the benefits of local iterations in reducing communication complexity. Specifically, the authors analyze a distributed variant of Adam - Local Adam with gradient clipping, which also reduces to Local SGD with momentum (Local SGDM). They establish that both Local SGDM and Local Adam can outperform their minibatch SGDM / minibatch Adam in convex / weakly convex settings, respectively.

### Strengths
- The work is the first to offer high-probability bounds for distributed optimization algorithms with local steps.

- Some assumptions are relatively weak; for example, smoothness and (strong) convexity are required on a subset rather than the entire space.

- The first theoretical convergence guarantees showing that Local SGDM and Local Adam can outperform their minibatch versions in some regimes (large $M$ and $K$ regime, where $M$ is the number of clients and $K$ is the number of local steps).

- The paper is well-written and easy to follow. The authors clearly introduce the problem and the paper's contributions.

- Despite a few limitations, this work represents an important theoretical contribution to understanding the empirical success of Adam with local steps as observed in prior research.

### Weaknesses
- The paper addresses only the homogeneous data case, where all clients have access to the same data. Client drift from data heterogeneity - one of the main challenges for local training methods - is not explored.

- The noise assumptions are somewhat restrictive (see, e.g., [1]). Specifically, the authors assume a bounded $\alpha$-moment of the noise with $\alpha \geq 4$. In contrast, most works only assume this condition for $\alpha \in (1,2]$, and recent high-probability and in-expectation convergence rates have been obtained under the strictly weaker condition $\alpha \in (1,2)$ [2].

[1] Khaled, Ahmed, and Peter Richtárik. "Better theory for SGD in the nonconvex world." arXiv preprint arXiv:2002.03329 (2020).

[2] Puchkin, Nikita, Eduard Gorbunov, Nickolay Kutuzov, and Alexander Gasnikov. "Breaking the heavy-tailed noise barrier in stochastic optimization problems." In International Conference on Artificial Intelligence and Statistics, pp. 856-864. PMLR, 2024.

### Questions
- What are the advantages of the proposed algorithms over non-adaptive methods in terms of theoretical guarantees?

- The paper claims that "there is no end-to-end convergence guarantee for distributed adaptive algorithms with local iterations." However, this seems inaccurate. For instance, [3] introduces the Extrapolated SPPM algorithm, which replaces local SGD-type training with the computation of proximal terms, and [4] provides tighter convergence theory for the same algorithm. Both approaches use adaptive strategies based on gradient diversity and Polyak step sizes, and their results hold in heterogeneous data settings.

[3] Li, Hanmin, Kirill Acharya, and Peter Richtarik. "The Power of Extrapolation in Federated Learning." arXiv preprint arXiv:2405.13766 (2024).

[4] Anyszka, Wojciech, Kaja Gruntkowska, Alexander Tyurin, and Peter Richtárik. "Tighter Performance Theory of FedExProx." arXiv preprint arXiv:2410.15368 (2024).

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
2

### Summary
The paper analysed the Local Adam method, which performs local Adam-like updates and intermittent communication. The paper provides a high probability convergence analysis for local Adam with heavy-tail noise, and demonstrated its advantage against mini-batch sgd in some regimes. The analysis also covers the local SGDM method.

### Strengths
The paper is a technical tour de force and the first one to analyse local methods with adaptive updates without some artificial assumption. The algorithm also incorporates the clipping mechanism to deal with the heavy-tailed noise. I think overall this is a pretty impressive display of technical achievement.

### Weaknesses
see questions.

### Questions
I did not go into the technical details in the appendices so I cannot attest to the correctness of the statements, but I have some questions/concerns with the followings in the main text:

1. In page 2 line 77 the author claimed that "The conventional in-expectation rate seems fail to capture some important properties like heavy/light tailed noise distribution." Can the author please elaborate on this? It seems to me that the in-expectation convergence is stronger than the high probability convergence presented in this paper since one should be able to convert the in-expectation convergence to high probability convergence using Markov inequality.

2. The paper claim to cover the $(L_0,L_1)$ smoothness case, but I fail to see how is this the case? It seems to me that all the theorems presented in the main paper are based on Assumption 2, which is just the usual smoothness assumption. 

3. Regarding Assumption 3: 
    - I was under the impression that most existing works make their heavy-tail noise assumption in the form: $\mathbb{E}\lVert \nabla F(x,\xi)-\nabla f(x)\rVert^\alpha\leq \sigma^\alpha$, i.e. a upper bound on the norm of the deviation. Here it seems to be an entry-wise upper bound. Can the author please elaborate more on how does the result based on the entry-wise upper bound translate to the usual result based on the usual norm upper bound? 
    - Remark 1: it seems a bit curious that the authors does not deal with the case $1<\alpha \leq 2$ which is what most existing works deal with, but instead consider the case with $\alpha \geq 4$? The authors claim that it's easy to extend the analysis to the case where $\alpha < 4$, then why not just include it here which will certainly make the results more easily comparable to the existing results?
    - Regarding the $\lVert \mathbf{\sigma}\rVert_{2\alpha}d^{\frac{1}{2}-\frac{1}{2\alpha}}= O(\sigma)$ assumption in the theorems. This seems to relate to the entry-wise noise bound. Can the authors elaborate on this and its consequences in the convergence rate (e.g. whether or not it's introducing additional dimension dependence compared to other works)?

4. For Theorem 3, can the author please elaborate more on the convergence criteria $\lVert \cdot\rVert^2_{H_r^{-1}}$. I understand that some existing works used convergence criteria similar to this one, but it nonetheless seems a bit unnatural to me. In particular, since $H_r$ is defined with respect to $v_r$ and $\lambda$, algorithm iterates and hyperparameter, it seems that the convergence is algorithm/parameter-dependent. Wouldn't we be able to claim any convergence rate if we set $\lambda$ large enough? Could the authors please explain the choice of $\lambda$ as well?

I would be happy to raise my score if the authors can provide more clarities regarding these questions/concerns.

### Soundness
3

### Presentation
3

### Contribution
4
