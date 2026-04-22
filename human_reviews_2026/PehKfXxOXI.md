# Revisiting the Effect of Topologies on Decentralized SGD

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 2, 4, 6

## Abstract
Decentralized SGD is a fundamental algorithm in decentralized learning, although the influence of an underlying network topology on its convergence behavior is not yet fully understood. Existing convergence analyses have shown that topologies with a small spectral gap significantly deteriorate the convergence rate of Decentralized SGD in both homogeneous and heterogeneous cases. However, many prior papers have reported that indeed the choice of the topology has a significant experimental impact in the heterogeneous case, but has little experimental impact on training behavior in the homogeneous case. In this paper, we present a tighter convergence analysis of Decentralized SGD for both convex and non-convex cases, offering a more precise understanding of how topologies affect the convergence rate than the prior analysis. Specifically, unlike existing convergence analyses that used only the spectral gap as a property of the topology, our novel analysis shows that all eigenvalues of the mixing matrix affect the convergence rate. This leads to the key insight that in homogeneous settings, the effect of topology on the convergence rate is notably smaller than expected from the existing analyses, especially for commonly used topologies, such as the ring and torus. Throughout the experiments, we carefully evaluated the convergence behavior of Decentralized SGD and demonstrated that our novel convergence analysis can more accurately describe the effect of topology on the convergence rate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed improved analysis on how network topologies affect convergence rates of decentralized SGD. Previous analysis mainly focus on spectral gap of communication networks, while this paper's analysis is based on all eigenvalues of gossip matrix, which leads to improved bounds including many classical topologies such as ring, torus and hypercube. Results for both convex and nonconvex objective functions are provided.

### Strengths
(1) The motivation is clear and strong. The new analysis based on eigenvalues of gossip matrix could be extended to more general settings.

(2) The proofs are technically sound and easy to read. The assumptions are standard.

### Weaknesses
(1) Only results for decentralized SGD are provided. It would be better to include results for more modern approaches such as gradient tracking.

(2) Analysis based on eigenvalues requires undirected graphs.

(3) It would be better to expand experiments on various tasks.

### Questions
(1) Is it possible to extend the analysis to directed graphs?

(2) Is it possible to derive any tightness results of the bounds in this paper?

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a convergence analysis of Decentralized SGD, aiming to provide a more precise understanding of how communication topologies affect convergence. In my opinion, the contribution is incremental in both theory and experiment, and the work is not yet suitable for publication at ICLR.

### Strengths
This paper presents a convergence analysis of Decentralized SGD, aiming to provide a more precise understanding of how communication topologies affect convergence.

### Weaknesses
My main concerns are as follows:

1.	The paper lacks experimental results demonstrating the performance of Decentralized SGD under different topologies in both IID and non-IID settings, which are crucial to support the theoretical findings. Simply citing previous works or describing results in words is not convincing.

2.	In Proposition 1, the term $\frac{\xi^2}{p^2}$indicates that the impact of topology in the IID case is smaller than in the non-IID case, since $\xi = 0$ under IID data. The authors should more clearly explain their motivation to improve the analysis.

3.	The authors claim that, even in the non-IID case, the coefficient still depends on p. Can the authors provide a corresponding lower bound to justify this claim? Otherwise, one might question whether the analysis is tight and whether the coefficient should indeed depend on p in the non-IID setting.

4.	The presentation needs improvement. Instead of only giving proof sketches, the authors should provide more insight into why the coefficient can be improved in the IID case, what analytical challenges arise, and how they are addressed. At present, it is difficult to identify the main contributions, especially since the convergence of Decentralized SGD has been extensively studied.

5.	Is the analysis in the IID case tight? Can the coefficient be further improved?

6.	Is the proposed coefficient a proper measure for all topologies? The spectral gap is a widely used and well-understood measure that captures the influence of topology. The authors should justify whether their proposed coefficient is theoretically sound and more precise than the spectral gap. Additional theoretical and experimental results are needed to support this claim.

7.	Why does the hypercube topology in Figure 1 exhibit a different trend compared to the others? 

8.	The experiments are insufficient. The authors should compare different topologies with the same number of nodes but varying communication links, and plot both the spectral gap and their proposed coefficient to more clearly illustrate the impact of topology.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a tighter convergence analysis of the optimization error in the Decentralized SGD algorithm. The analysis shows that not only the spectral gap of the gossip matrix matters, but the entire spectrum also plays a role. While this insight does not directly improve the worst-case upper bounds in general heterogeneous settings, it does lead to better rates in homogeneous scenarios. This is clearly demonstrated in Table 1, where the authors compare the transient time (i.e., the number of iterations required to achieve linear speedup) of the state-of-the-art result by Koloskova et al. with their own findings across different graph topologies. The authors further argue that their results help explain why, in homogeneous settings, the choice of graph appears to have less impact. Finally, they support their theoretical findings with a set of experiments.

### Strengths
The paper is well written and enjoyable to read. The question of how the communication graph impacts decentralized learning, although extensively explored, remains highly relevant, and demonstrating that the entire spectrum of $W$ matters is a natural yet insightful contribution, which could be useful for the future design of communication topologies. I also appreciated the results presented in Table 1, which clearly illustrate the improvement of the proposed upper bounds over those of Koloskova et al.

### Weaknesses
As currently written, the paper feels somewhat premature for ICLR. The question of the communication graph’s impact is interesting and relevant, but the authors’ improvement over Koloskova et al. mainly applies to homogeneous settings, which are less central in decentralized learning than heterogeneous ones. The manuscript would be substantially strengthened by addressing the following points:

1) **Strongly convex case.** Add results for strongly convex objectives so the paper provides a complete improvement over Koloskova et al. across common problem classes.

2) **More intuition on the spectrum.** Explain more precisely what the full spectrum of $W$ captures and why it primarily affects the stochastic-variance term of the decentralization error (the middle term). Intuition, toy examples, or a short illustrative derivation would help readers understand the phenomenon.

3) **Richer experiments.** As noted, Fig. 2 omits error bars because they were “too small”: this suggests that the experiments may be too simple (MNIST). I recommend running experiments on more challenging benchmarks (e.g., CIFAR-10 and other realistic datasets) and adding cases that show a graph with a worse spectral gap can outperform one with a better gap thanks to a more favorable spectrum. 

4) **Formalize Table 2.** Provide a formal derivation or rigorous bounds for $\frac{1}{n}\sum_i \frac{\lambda_i^2}{1-\lambda_i^2}$ reported in Table 2; currently these quantities are only estimated.

5) **Graph learning discussion.** Include or discuss in more depth a method for learning the communication graph (currently deferred to future work). Given the paper’s claims about spectrum-driven design, even a preliminary approach would increase the paper’s impact.

### Questions
Other comments/questions.

1) What exactly changes in the graph structure when $\frac{1}{n}\sum_i \frac{\lambda_i^2}{1-\lambda_i^2}$ varies, while keeping the spectral gap fixed (Fig.2)? Clarifying this point would help readers build intuition about how this spectral quantity reflects graph topology (see also point 2 in the main review).


2) The conclusion is sometimes misleading. The authors claim that their results explain why the graph does not have a large impact in homogeneous setting. But in fact it does (the obtained bounds are still vacuous for $p$ close to 0), it is simply provably less impactful. 

3) In convex settings, $x^*$ may not be unique. How do you handle that in Assumptions 6-7 and in your proof?

4) In line 459, you mention “the same training dataset.” Could you clarify whether the datasets are truly identical across nodes, or if they share the same distribution but different realizations?

Typos:

- line 020: "[...] rate than the priori analysis" <-- "[...] rate **less** than the prior analysis"
- line 462: "See Section 6.2" <-- "See Appendix E"?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a refined convergence analysis of Decentralized Stochastic Gradient Descent (DSGD) under a given gossip communication scheme. While previous work primarily characterizes the convergence rate in terms of the spectral gap (i.e., the second-largest eigenvalue of the gossip matrix), this paper goes further by analyzing the influence of the entire spectrum of eigenvalues. The resulting convergence bound is tighter and reveals that, in the case of data-homogeneous decentralized learning, all eigenvalues—not just the second largest—significantly impact the convergence behavior. Theoretical insights are supported by experimental results that validate the findings.

### Strengths
- **Theoretical Contribution**: This paper provides a detailed theoretical analysis of the convergence behavior of Decentralized SGD by examining the full spectrum of eigenvalues $\lambda_2, \ldots, \lambda_n$ of the gossip matrix. The refined convergence bound, presented in Theorem 1, captures the influence of all eigenvalues, offering a more precise characterization compared to earlier results, which primarily rely on the spectral gap. For comparison, the classic convergence result is summarized in Proposition 1. The mathematical analysis is rigorous and well-supported.

- **Empirical Validation**: The numerical experiments compare the key convergence terms from previous work and this paper. The results clearly verify that the newly derived term is tighter, confirming the theoretical advantage.

### Weaknesses
This paper is mainly a theoretical paper. The major theorems are Theorem 1 and 2. However, there are limitations in both theorems.

- **Marginal Theoretical Improvement in Theorem 1**: In the main Theorem 1, Convex Case, the new rate improves the term
$$ \left(\frac{1-p }{p}\cdot\frac{\sigma^2Lr_0^2}{R^2}+\frac{(1-p)\zeta^2}{p^2}\cdot\frac{Lr_0^2}{R^2}\right)^{\frac{1}{3}}$$ to
$$\left(\frac{1}{n}\sum_{i=2}^n\frac{\lambda_i^2}{1-\lambda_i^2}\cdot\frac{\sigma^2Lr_0^2}{R^2} + \frac{(1-p)\zeta^2}{p^2}\cdot\frac{Lr_0^2}{R^2}\right)^{\frac{1}{3}}.$$
However, the new rate is only significantly tighter when $\sigma^2 \gg \frac{\zeta^2}{p}$, meaning that many of the theoretical advantages heavily rely on the data homogeneity assumption. The similar problem also lies in the Non-convex Case part of the theorem.

- **Overly Strong Assumptions in Theorem 2**: The assumptions in Theorem 2 are quite restrictive, requiring convexity and that all local functions are identical $(f_i = f_j$ for all $i, j$). Under such conditions, even running local SGD without communication would eventually lead each agent $x_i$ to converge to the same minimizer. Moreover, the convergence bound in Theorem 2 is given as the minimum of two terms:
The first term is essentially derived from Theorem 1 when $\zeta = 0$.
The second term reflects the performance of independent local SGD, which is intuitive and not surprising.
As a result, the contribution of Theorem 2 is somewhat limited.

### Questions
As I mentioned in the weakness on Theorem 1, Convex Case, the new rate improves the term $$ \left(\frac{1-p }{p}\cdot\frac{\sigma^2Lr_0^2}{R^2}+\frac{(1-p)\zeta^2}{p^2}\cdot\frac{Lr_0^2}{R^2}\right)^{\frac{1}{3}}$$ to $$\left(\frac{1}{n}\sum_{i=2}^n\frac{\lambda_i^2}{1-\lambda_i^2}\cdot\frac{\sigma^2Lr_0^2}{R^2} + \frac{(1-p)\zeta^2}{p^2}\cdot\frac{Lr_0^2}{R^2}\right)^{\frac{1}{3}}.$$

You've already presented the comparison between 
$
\frac{1}{n}\sum_{i=2}^n \frac{\lambda_i^2}{1 - \lambda_i^2}
$
and $
\frac{1 - p}{p}.
$
Could you also include a comparison with 
$
\frac{1 - p}{p^2}
$
in front of the $\zeta^2$ term? I would like to gain a clearer understanding of how the ratio $\frac{\sigma^2}{\zeta^2}$ affects the bound in Theorem 1 and under what conditions this bound becomes significantly tighter than the previous bound.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present a new convergence analysis for Decentralized SGD, finding that all eigenvalues of the mixing matrix affect the convergence rate. This analysis leads to the key insight that in homogeneous settings, the effect of topology on the convergence rate is notably smaller than expected from existing analyses, which primarily rely on the spectral gap.

### Strengths
* The motivation is strong. The authors address a well-known and long-standing discrepancy in the decentralized learning community: the empirical observation that the spectral gap has little experimental impact on training behavior, despite theoretical analyses suggesting otherwise.
* The core finding that all eigenvalues of the mixing matrix affect the convergence rate is both interesting and novel. It offers a more precise characterization of the algorithm's dynamics.
* The paper is very well-written. The motivation and background are clearly articulated, the new theoretical results are clearly contrasted with prior work (e.g., in Table 1), and the inclusion of a clear proof sketch (Section 5) is helpful for understanding the main technical ideas.

### Weaknesses
No major weaknesses identified. Some potential limitations are listed below.

1. The paper's core motivation and main contribution are focused on the homogeneous case. This may limit the contribution to "near-homogeneous" settings. The analysis explains why sparse graphs (like rings) perform well on homogeneous data (as the $\sigma^2$ term is improved), but it also confirms that the spectral gap $p$ remains the critical bottleneck for heterogeneous data ($\zeta > 0$). This remains a limitation, as data heterogeneity is the important in modern decentralized and federated learning. 

2. There is a mismatch between the scope of the theory and the experiments. The theory (e.g., Theorem 1) covers both convex and non-convex cases. However, the experiments in Section 6.2 are only run on two convex models (Logistic Regression and Ridge Regression). The paper is missing an empirical validation of its non-convex analysis. Additionally, the experiments are only on MNIST, which is a relatively simple dataset.

### Questions
1.  Assumptions 6 and 7 (bounding noise and heterogeneity at the optimum $x^*$) are not standard in the decentralized learning literature. Can the authors provide more detail on their motivation, reasonableness, and specifically where they are critically used in the proof for the convex case?
2.  The paper discusses Neglia et al. (2020) and Vogels et al. (2022). Could the authors provide a more direct and detailed comparison of the final bounds/rates and the underlying assumptions between this work and those two papers?

3.  Regarding the experimental methodology described in Section 6.2 and detailed in Appendix E: The process appears to involve factoring a base mixing matrix $W$ into $W = V \Lambda V^{-1}$, and then reconstructing a new matrix $W' = V \tilde{\Lambda} V^{-1}$ by altering the eigenvalues $\lambda_3, ..., \lambda_n$ while keeping $\lambda_2$ (and thus $p$) constant. What was the main motivation for this synthetic approach? An alternative might be to directly compare the performance on different standard sparse graphs (e.g., Ring vs. Torus), which would naturally have different values for both $p$ and the new metric $\frac{1}{n}\sum_{i=2}^{n}(\lambda_{i}^{2}/1-\lambda_{i}^{2})$.

### Soundness
4

### Presentation
4

### Contribution
3
