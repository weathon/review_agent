# Gradient Descent Dynamics of Rank-One Matrix Denoising

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Matrix denoising is a crucial component in machine learning, offering valuable insights into the behavior of learning algorithms  (Bishop and Nasrabadi, 2006). This paper focuses on the rectangular matrix denoising problem, which involves estimating the left and right singular vectors of a rank-one matrix that is corrupted by additive noise. Traditional algorithms for this problem often exhibit high computational complexity, leading to the widespread use of gradient descent (GD)-based estimation methods with a quadratic cost function. However, the learning dynamics of these GD-based methods, particularly the analytical solutions that describe their exact trajectories, have been largely overlooked in existing literature. To fill this gap, we investigate the learning dynamics in detail, providing convergence proofs and asymptotic analysis. By leveraging tools from large random matrix theory, we derive a closed-form solution for the learning dynamics, characterized by the inner products of the estimates and the ground truth vectors. We rigorously prove the almost sure convergence of these dynamics as the signal dimensions tend to infinity. Additionally, we analyze the asymptotic behavior of the learning dynamics in the large-time limit, which aligns with the well-known Baik-Ben Arous-Péchée phase transition phenomenon n (Baik et al., 2005). Experimental results support our theoretical findings, demonstrating that when the signal-to-noise ratio (SNR) surpasses a critical threshold, learning converges rapidly from an initial value close to the stationary point. In contrast, estimation becomes infeasible when the ratio of the inner products between the initial left and right vectors and their corresponding ground truth vectors reaches a specific value, which depends on both the SNR and the data dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the rank-one Jonstone's spiked model in matrix denoising problem within random matrix theorem (RMT) scenerio. The main contributions are two theorems. In theorem 1, they derive a closed-form deterministic approximation for the inner products between the learned vectors and the ground truth when ratio $\frac{p}{n}\to c>$. In theorem 2, they show that $c^{1/4}$ is the critical threshold for the gradient flow estimation.

### Strengths
- Use random matrix theory (RMT) derive a closed-form solution for the learning dynamics of matrix denoising problem. By RMT, naturally extend the matrix denoising problem to high dimensional scenario.
- The theorems and derivations are solid, and for example, the complex but precise expression in Theorem 1 is derived.
- The problem and assumptions are stated clearly.
- Provide a more comprehensive understanding of the dynamics of gradient-based learning in high-dimensional matrix problems

### Weaknesses
- Lack of explanation of the application of matrix denoising in the random matrix scenario, i.e. $\frac{p}{n}\to c$ with $p, n \to \infty$.
- The statement about computational complexity is too vague, like "the  complexity is affordable" on line 220 and "We note that $\hat{t}(\alpha_u,\alpha_v)$ can be efficiently computed by standard numerical methods." on line 244.
- On line 244, need more details of the so called "standard numerical methods".
- On line 42, "Extensive research has shown that" lacks reference.
- The theoretical work on which the article is based is very classic, and there is a lack of reference from recent new theoretical work.
- Lacks explanation of Riemannian gradient operator on line 144.
- In experiments, only one set of $(p,n)$ is tested, need more testing sets.
- In experients, the critical threshold is about 0.93, but the SNR is set to 0.3 and 1.5, which are far away from the threshold. More attempt on SNR need to be tested, especially the SNR near the threshold.- The ground truth is too trival.

### Questions
In Remark 1, the assumption of knowing the SNR $\lambda$ is practical in reality?In experiment, what will the results change if $p$ and $n$ are not that large, like $p=20$, i.e. beyond the RMT scenario.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors address the problem of rank-one matrix denoising, focusing on the gradient descent dynamics underlying this process. The main contribution of the paper lies in two theoretical results. The first theorem establishes a deterministic approximation for q_u and q_v, two widely used metrics that measure the alignment between the ground truth and the estimated components. The second theorem characterizes the asymptotic behavior of these deterministic approximations.

### Strengths
The paper is well written, well structured, and clearly explained. The theoretical analysis is rigorous, and the overall presentation is easy to follow. I find the work interesting and relevant to the study of optimization dynamics in low-rank estimation problems.

### Weaknesses
My only minor concern relates to the clarity of the experimental results. In Figure 2, which illustrates the effect of the critical SNR threshold, it is visually difficult to distinguish the different curves, especially in the middle subfigure. I understand that it is challenging to convey a large amount of information within a single figure, but since this result serves as an important validation of the theoretical findings, it should be presented more clearly to enhance its impact.

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the gradient-flow dynamics of the rectangular rank-one spiked matrix denoising problem in the high-dimensional limit.
Using tools from random matrix theory and Laplace transforms over the Marčenko–Pastur law, the authors derive explicit deterministic equations describing the evolution of the overlaps between the estimated and true directions.
They prove convergence to a deterministic limit, identify a BBP-type phase transition at the expected signal-to-noise threshold, and show that the limiting alignment carries the same sign as the initialization (“signed BBP”).
A kernel-based argument establishes existence and uniqueness of the solution.
Experiments nicely confirm the theoretical predictions.
Overall, the work extends the analysis of Bodin and Macris (2021), which treated the symmetric Wigner case, to the rectangular Wishart setting.

### Strengths
The results are technically sound and contribute to the understanding of optimization dynamics in high dimension.Extending the analysis from the symmetric to the rectangular setting is nontrivial and closes a natural gap in the literature. Conceretly:
- Provides explicit, analytic trajectories for gradient flow in the rectangular spiked model.
- The results are rigorous and match empirical observations very well.
- The “signed BBP” effect offers a clear dynamical interpretation of initialization dependence.
- Writing and figures are clear; the paper is enjoyable to read.
- Strengthens the theoretical link between random matrix theory and learning dynamics

### Weaknesses
- The work feels somewhat incremental relative to Bodin & Macris (2021); the main novelty lies in adapting the approach to the rectangular case.
- The “signed BBP” result, while nicely explained, is largely an expected property of continuous gradient flow.
- Experiments are limited to confirming the theory in the simplest setting; discrete-time or noisy gradient dynamics are not discussed.
- The discussion of related literature could be more complete and precise.

### Questions
I) Clarify the exact novelty relative to previous analyses.
The paper’s results are very close in spirit to Bodin & Macris (2021, arXiv:2105.12257), which already provided deterministic gradient-flow equations and asymptotic limits in the symmetric spiked Wigner setting.
It would help to spell out precisely which technical steps differ in the rectangular case and which parts of the proof had to be redone — for instance, changes in the resolvent structure, contour integration, or Laplace-transform kernel.
Are there specific mathematical obstacles that make the rectangular case substantially more difficult, or is it mainly a matter of replacing the semicircular law with the Marčenko–Pastur one?
A short paragraph clearly highlighting these differences would make the contribution much clearer.

II) Connection to the broader literature surveyed by Macris.
The “Related Work” section of Bodin & Macris (2021) already offers a remarkably complete overview of the theoretical ecosystem surrounding gradient descent, AMP, and high-dimensional inference. Many of those references are directly relevant here.
The Bayesian analyses of the spiked Wigner and tensor models (Korada & Macris 2009; Barbier et al. 2016; Lelarge & Miolane 2018; Lesieur et al. 2017; Perry et al. 2020) provide precise information-theoretic benchmarks in the form of mutual information and MMSE.
The dynamical behavior of AMP and the existence of computational-to-statistical gaps (Barbier et al. 2016; Lesieur et al. 2017) are also well understood.
Given this context, could the authors explain what new qualitative insight is gained from their explicit time-evolution formulas?
For instance, does the analytic expression for the transient trajectories reveal any phenomenon that is not already implicit in the AMP state evolution or in the energy-landscape picture?

III) Relation to the matrix–tensor and Langevin dynamics literature.
Macris notes that recent works (Sarao Mannelli et al., 2019; 2020) analyzed the optimization of mixed matrix–tensor inference problems using integro-differential Cugliandolo–Kurchan (CSHCK) equations — a fully dynamic, spin-glass-inspired formalism.
It would be interesting for the authors to comment on how their much simpler kernel/ODE formulation compares conceptually to those dynamical equations.
Does the present approach capture similar information about the saddle structure or convergence rates, but in a more tractable regime?
Or is it strictly a deterministic “mean-field” limit without the stochastic thermal components appearing in the CSHCK-type equations?

IV) Discrete gradient descent versus continuous gradient flow.
Since the work focuses entirely on continuous-time dynamics, a natural question is whether these results extend (even approximately) to discrete gradient descent with a finite learning rate.
Previous works, such as Lee et al. (2016) and Ge et al. (2017), established convergence for discrete GD under the “strict saddle” property, while Saxe et al. (2013) and Mei & Montanari (2019) studied learning-rate effects in linear and nonlinear models.
Could the authors discuss whether the signed-BBP phenomenon and transient behavior persist under discrete updates?
Even a conjectural statement or some preliminary numerical evidence would be welcome.

V) Connections to the energy landscape and spin-glass literature.
Macris also situates the work within the broader context of non-convex optimization in random energy landscapes (Subag & Zeitouni 2017; Ros et al. 2019; Auffinger et al. 2013).
It would be valuable to comment on whether the present gradient flow can be interpreted as traversing a spin-glass-like energy surface with a small number of global minima and exponentially many saddles.
Does the deterministic flow derived here correspond to the typical trajectory that avoids these saddles in the large-n limit?
Such a discussion would help bridge the current mathematical analysis with the well-developed physical intuition from statistical mechanics.

Overall, the paper is technically clean and the results are credible, but the authors could significantly increase its impact by situating it more deeply within the broad theoretical lineage summarized by Macris — including AMP and Bayesian limits, non-convex low-rank recovery, spin-glass Langevin dynamics, and deterministic gradient-flow analyses.
Clarifying what the present framework adds to that landscape, and where it could go next, would make the work more compelling for the ICLR audience.

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
The paper analyzes the statistical properties of gradient flow (as a proxy for gradient descent) for rank-one matrix denoising under a deformed Wishart model (rectangular matrices) with noise that has i.i.d. entries and . It derives a deterministic term for the limit of the inner products between limiting singular vector estimates and ground truth singular vectors in the asymptotic limit of matrix dimensions becoming infinite with fixed relation fraction. As an implication of these results, the authors are able to relatively accurately predict the behavior of gradient flow depending on the problem's signal-to-noise ratio (SNR) threshold akin to the BBP [Baik, Ben Arous, Péché 2005] phase transition. The results also can be used to quantify the dependence of the dynamics on initialization (value of $\alpha_u$/$\alpha_v$) and reasonable stopping times can be theoretically derived (see Remark 1 and Remark 2). From a technical perspective, the results lean on the analysis of [Bodin & Macris 2021], who have showed similar results for the symmetric case. Simulations are presented that substantiate the qualitative accuracy of the asymptotic analysis in the finite sample / matrix dimension case of $p$ and $n$ fixed.

### Strengths
The analysis presented in the paper seems to be new and studies a foundational problem in high-dimensional statistics / linear algebra, the behavior of singular value decomposition under the influence of noise in the case of rectangular matrices. The noise model is rather general, which is positive. It is of interest that the gradient flow dynamics more or less matches information theoretical phase transitions that are intrinsic to the problem. 
While carefully checking many proofs in the appendix was beyond my abilities in the allocated time-frame as a reviewer, the results are plausible from a perspective of a reviewer who is familiar with tools for analyzing non-asymptotic high-dimensional problems.
Beyond covering the asymmetric case, some assumptions are weaker than in the related paper [Bodin & Macris 2021], such as the finite fourth moment assumption (as a opposed to assuming existence of all moments).

### Weaknesses
A fundamental weakness of the work is that it applies only in the high-dimensional limit of $\lim_{p, n \to \infty} p/n = c$, which is in contrast to many analyses of iterative algorithms in machine learning. Related to this issue, it can be pointed out that the title containing "Gradient Descent" is to a certain extent a misnomer as gradient flow, which is less relevant than gradient descent in practice in machine learning, is being analyzed. Thus, a lack of treatment of the discrete-time gradient descent method is a weakness of the paper given the framing of the paper.
A more unified discussion pointing out the differences and similarities between a power method algorithm for computing the leading singular vector pair and the presented algorithm would also have been insightful - I somewhat disagree with the framing that "SVD is intractable" as it is clear that a reasonable algorithm for the problem would involve a partial SVD implemented via randomized techniques [see, e.g., Martinsson, Tropp 2020].
Finally, it can be be pointed out that, while the asymmetric case being more challenging, the analyses / simulations presented are relatively close aligned to the ones of [Bodin, Macris 2021].

### Questions
1. In lines 141-145, it is mentioned that $\operatorname{grad](\cdot)$ is a "Riemannian gradient operator, which enforces the unit norm constraint". However, I do not see that the update equation of (4) enforces such a constraint. In some sense, this is statement is incompatible with the framework of Riemannian optimization as the Riemannian gradient lives in the tangent space onto Riemannian manifold and requires a retraction back onto the manifold (here, enforcing unit-norm vectors) to enforce the constraints.
Can you clarify or correct this discussion? In particular, how does your studied gradient flow algorithm enforce the unit norm constraints throughout its flow?

2. What are the limitations of the presented analysis for higher-rank ground-truths? Where does your current analysis fail to go through?

### Soundness
3

### Presentation
3

### Contribution
2
