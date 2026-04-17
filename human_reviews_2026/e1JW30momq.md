# Mixtures Closest To A Given Measure: A Semidefinite Programming Approach

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Mixture models, such as Gaussian mixture models (GMMs), are widely used in machine learning to represent complex data distributions. A key challenge, especially in high-dimensional settings, is to determine the mixture order and estimate the mixture parameters. We study the problem of approximating a target measure, available only through finitely many of its moments, by a mixture of distributions from a parametric family (e.g., Gaussian, exponential, Poisson), with approximation quality measured by the 2-Wasserstein ($\operatorname{W2}$) or the total variation ($\operatorname{TV}$) distance. Unlike many existing approaches, the parameter set is not assumed to be finite; it is modeled as a compact basic semi-algebraic set. We introduce a hierarchy of semidefinite relaxations with asymptotic convergence to the desired optimal value. In addition, when a certain rank condition is satisfied, the convergence is even finite and recovery of an optimal mixing measure is obtained.
We also present an application to clustering, where our framework serves either as a stand-alone method or as a preprocessing step that yields both the number of clusters and strong initial parameter estimates, thereby  accelerating convergence of standard (local) clustering algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper applies a recently published approach to the approximation of W1 and TV distances between measures (Laserre 2024) to the specific case of approximating a given density by a finite mixture. The basic idea is to solve a semidefinite program based on a subset of  the moments of the finite mixture: by increasing the number of moments considered one can prove that eventually the recovered mixture will be optimal in terms of W1 or TV but of course this comes with a computational price since solving the SDP will be more expensive. 

Some experimental results are shown on scalar and 2D problems. They show that initializing EM with the suggested method converges faster than EM with random initialization.

### Strengths
The paper addresses a difficult and foundational problem: even evaluating the TV between two given distributions is challenging so finding the optimal mixture using that criterion is a major challenge. The paper also brings the important advances of (Laserre 2024) to the attention of the ML community.

### Weaknesses
I am mostly concerned with the relevance to ICLR and the significance.

I believe the paper will be difficult to follow for most ICLR attendees. The derivation builds heavily on prior work (Laserre 2024a,b) and without reading those papers, it is hard to understand the contribution in the present paper. 

The significance to a ML audience is limited by several factors. First, the computational cost is such that the method can only be applied to one dimensional or two-dimensional data. This is acknowledged by the authors and they suggest ways to allow scaling up in the appendix but in its current formulation the method is highly limited. Second, the benefit of using the method in an ML application is questionable. The experiments in the paper (using the method to initialize EM) show a decrease in the number of iterations required to achieve convergence but given the high computational cost of computing the initialization it is not clear that this is an attractive method for practitioners: I assume almost any practitioner would prefer to run EM for more iterations instead of solving an SDP.  

Another issue of relevance is the reliance on TV and W1 metrics. For better or worse, most clustering algorithms use the likelihood objective which is equivalent to the KL metric.

### Questions
What is the motivation for using TV or W1 as initialization for clustering using EM and K-means (which are based on a KL divergence) ?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the problem of approximating a target measure, known through finitely many of its moments, by a mixture from a parametric family (e.g., Gaussian, exponential, Poisson). The authors investigate optimality under two regularized moments
The authors propose a hierarchy of semidefinite relaxations for the corresponding optimization problem, with guarantees of asymptotic convergence and finite recovery under a rank condition. The work is mathematically rigorous and builds on the Lasserre hierarchy and related moment-SDP techniques.

Overall Assessment:
While the paper presents a mathematically elegant and theoretically sound framework, it is not presented in a manner accessible to the ICLR community, and the proposed approach does not scale beyond low-dimensional problems. The contribution may be more suitable for a venue focusing on mathematical optimization or computational statistics, rather than a machine learning conference that emphasizes scalable algorithms and empirical evaluation.

### Strengths
- The paper has a very clearly defined objective and proposes a mathematically precise solution via optimality conditions.
- Gaussian mixture continues to be highly relevant for several tasks.
- Research into mixtures is a very relevant topic
- I think that the interplay between measure optimality with respect to different divergences is very interesting
- Overall, the paper has a mathematically very interesting foundation,

In conclusion, this is a very interesting paper from the point of view of mathematical optimization and computational statistics.

### Weaknesses
*Main Text*:

- The paper is very closely related to (Lasserre, 2009, 2015, 2024 a, 2024b) and not very intuitive for the uninitiated, e.g., lines 199-202. While I think that this is certainly a very interesting line of research, understanding this paper in detail requires an in-depth understanding of previous works.

- Structure: Choosing Wasserstein and TV seems somewhat arbitrary at first glance. While it makes sense given the existing literature, it would make sense to discuss potential extensions to MMD, sliced Wasserstein and other divergences.

- Mathematical statements in the main text are not formulated very clearly, e.g., Thm 1. While the supplementary provides an exact formulation, the formulations in the main text seem rather vague.

- Line 187: “ The optimization problem in (7) is formulated over
probability measures, making it infinite-dimensional, and thus intractable in its original form.” Note that this might be somewhat misleading since parametric mixtures and empirical measures are not infinite-dimensional. Maybe the conditions should be specified.

- lines 255-257: “However, the sample size $N$ is often large enough, and therefore, by the law of large numbers, the difference between empirical and true population moments is negligible.” I think that the joint estimation of all moments depends on the ambient dimension $d$. Therefore, the phrase in the paper might be misleading, i.e., $N$ would have to be *very* large in high dimensions. Please comment on the potential impact of the curse of dimensionality.

*Numerical Evaluation*:

- The paper ‘Sliced Wasserstein Distance for Learning Gaussian Mixture Models’ (Kolouri et a., 2017) proposes another method to fit Gaussian mixtures with a (sliced) Wasserstein distance. While there are key differences, I think that this is a relevant reference and that there should be a numerical comparison between this method and the proposed one. Similarly, the cited paper ‘Learning Gaussian Mixtures using the Wasserstein-Fisher-Rao Gradient Flow’ (Yan et al., 2024) learns Gaussian mixtures using the Wasserstein-Fisher-Rao geometry, and I would have appreciated a numerical comparison.

- Figure 1: This is an interesting visualization, but then again, many methods have been proposed to curb the number of mixture components. Moreover, the regularization term works as a kind of prior on the component number. Thus, a comparison with Bayesian mixture techniques (Richardson et al., 1997; Nobile, 2005; Miller et al., 2018) would have made more sense.

- Figure 2: Please note that k-means is a highly efficient and fast algorithm and that there exist many fast and effective initialization heuristics beyond ‘random’ ones. Moreover, there is no clear advantage of ‘TV’-initialization in general and no clear advantage of ‘Wasserstein’-initialization for large ‘mixture indices’. Moreover, the runtime of initialization+K-means should be reported since this is the value that is actually interesting in practice. 
- Figure 3: This experiment gives no additional insight beyond the ones presented in Figure 2. Projecting MNIST onto 2 dimensions is not really a high-dimensional experiment.
- Gaussian mixtures allow for very nice visualizations in 2D. Therefore, I would have appreciated a density heatmap or a similar plot that showcases the behaviour of the Wasserstein-optimal vs. the TV-optimal vs the KL-optimal (obtained via EM) mixtures for a fixed number of components. Given that the paper focuses on such settings, readers would have gained a more intuitive understanding. 


*Conclusion*:
Despite the interesting theoretical background, the paper suffers from a lack of realistic high-dimensional experiments, comparative/ablation studies, and visual intuition. Moreover, the contents of this paper are not presented in a very accessible manner and are hard to follow for readers who are not familiar with (Lasserre, 2009, 2015, 2024a, 2024b). I would recommend that the authors add more visual studies and investigate the adaptation of their optimality formulations to higher dimensions. Personally, I would find a thorough investigation of the interplay between KL, Wasserstein, and TV optimality for 2d mixtures highly interesting.

### Questions
- I would have been interested in the impact of optimizing a given divergence on another divergence. What is the interplay of TV and Wasserstein? What is the interplay of KL (as used by EM) and Wasserstein/KL beyond simple 1d uniform distributions?
If you minimize the sliced Wasserstein using the algorithm of (Kolouri et al., 2017), what is the impact of the Wasserstein (and vice versa)?

- Can you quantify the number of constraints with respect to the dimension?

- There exists a paper on reducing mixture components using Wasserstein distance (Assa et al., 2018). Given the goal of reducing the components of a target mixture, how could one compare their reduction outcome with your method? 

- Beyond CV and BIC, how does your component number estimation compare to estimation via Bayesian methods, see (Richardson et al., 1997; Nobile, 2005; Miller et al., 2018)? I feel as if this is closer to the presented work since a regularization term/component number prior is used. 

- What is the impact of varying the regularization? I assume that the smoothness of the estimated mixtures in Figure 1 is at least partly the result of the regularization and not of the Wasserstein/TV optimality.

- A whole line of research, e.g., (Delon et al, 2019; Dusson et al., 2023) and more recently (Nguyen et al.; Bonet et al.; Piening et al., all 2025), is devoted to developing Wasserstein-type metrics between mixtures. While they are not meant to fit mixtures, they are meant to compare between a target and a reference mixture. Did you consider metrics of this kind for your evaluation? Do you see any relevance of the studied `Mixture Wasserstein'/`Wasserstein-over-Wasserstein'-type metrics to your work?

-  There are papers (Kolouri et al, 2017; Geppert et al., 2021) investigating mixture estimation via gradient descent. Did you consider such gradient-based solvers to `approximately’ solve the optimization problem to enable approximate solutions to the proposed optimization problem in higher dimensions?

- The proposed discrepancy-regularization reminds me of research on Wasserstein gradient flow for regularized f-divergences, see (Stein et al, 2025). While this work is not directly relevant, do you think that you could extend some of your results to general f-divergences?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies the problem of approximating a target probability distribution (known through its moments) using a mixture of distributions. The goal of the proposed method to identify the mixture that is closest to the target under either the 2-Wasserstein or Total Variation (TV) distance. The paper assumes that the parameters of the mixture components lie within a compact semi-algebraic set. Combined with the assumption of polynomial moments, this formulation allows the problem to be converted into a Generalized Moment Problem (GMP). Furthermore, when the distance between the target and the learned distribution is measured using the 2-Wasserstein or TV metric, the relaxation of this problem can be expressed as a Semidefinite Programming (SDP) problem. The paper also presents experiments on synthetic data and the MNIST dataset to evaluate the practical effectiveness of the proposed approach.

### Strengths
The paper addresses an important problem of learning with mixtures of distributions, where each component is constrained to lie within a compact semi-algebraic set. The formulation is theoretically elegant. It is nice to see that the authors also evaluate the practical effectiveness of their proposed algorithm.

### Weaknesses
- I am not entirely sure about the technical novelty of this work compared to prior studies, particularly Lasserre (2024a) and Lasserre (2024b). The SDP relaxation of the Wasserstein distance appears closely related to Lasserre (2024a), while the relaxation of the TV distance seems inspired by Lasserre (2024b). One of the main contributions seems to lie in formulating the problem itself—such as defining each mixture component within a compact semi-algebraic set. As a result, the novelty of the proposed technique and/or results appears somewhat limited. Given that the experimental performance of the algorithm is also not particularly strong, the overall technical contribution remains unclear. 

- The paper is also difficult to follow in several sections, especially for a venue like ICLR. Readers may find Sections 2 and 3 overly technical. It may be beneficial to present these sections in a slightly informal manner, while moving the formal statements and results to the appendix.

Minor: 

- The following paper [1] also considers the problem of estimating the number of components in Gaussian mixtures, so it is somewhat related to this work considering this work also considers the problem when the mixing distribution is unknown. 

[1] Robust Model Selection and Nearly-Proper Learning for GMMs. Liu et al. 2022

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors develop semidefinite programming (SDP) relaxations for computing the W2 and TV distances between two probability measures within a Moment-SOS framework (Lasserre, 2009), which has suitable regularization to mitigate numerical instabilities and promote sparse support as well. They tested the proposed methods on a range of unsupervised clustering tasks on the underlying
parametric mixture family, which produces mixture estimates that can significantly improve the performance of standard clustering algorithms.

### Strengths
The paper addresses approximating (via either W2 or TV distance minimization) an arbitrary given measure by mixtures from specific parametric families by introduceing two regularized Moment-SOS-type hierarchies of SDP relaxations.

As for the proposed method, the authors establish comprehensive theoretical results of the convergence guarantees.

Empirical results confirm the method’s performance gains among clustering tasks under GMMs.

### Weaknesses
1. There is ambiguity or confusion in the statement of theorems. In Theorem 3.2 and Theorem 3.3, The phrase “μ can be best approximated by …” is ambiguous and needs clarification. Please state explicitly the admissible set against which the optimum is taken. 

\
2. In practical applications, Algorithm 2 involves Cholesky factorization and Schur decomposition, which can be computationally expensive and may limit its scalability. Therefore, it might not be as efficient as lightweight alternatives such as k-means++. Moreover, several recent studies have proposed preprocessing techniques for clustering that provide strong initial parameter estimates, such as semidefinite programming (SDP)–based methods by Chen and Yang (2021) and Zhuang et al. (2024). Have you compared your algorithm with these approaches to evaluate computational efficiency and initialization performance?

\
3. The current applications are limited to clustering tasks under Gaussian Mixture Models (GMMs). Could the authors discuss how their proposed methods performed in experiments when applied to other problems beyond GMM clustering?



\
[1] Xiaohui Chen and Yun Yang. Cutoff for exact recovery of gaussian mixture models. IEEE Transactions on Information Theory, 67(6):4223–4238, 2021. 
[2] Yubo Zhuang, Xiaohui Chen, Yun Yang, and Richard Y. Zhang. Statistically optimal k-means clustering via nonnegative low-rank semidefinite programming, 2024. URL https://arxiv.org/abs/2305.18436.

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2
