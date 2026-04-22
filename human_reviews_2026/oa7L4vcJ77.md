# Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
We address the problem of efficiently computing Wasserstein distances for multiple pairs of distributions drawn from a meta-distribution. To this end, we propose a fast estimation method based on regressing Wasserstein distance on sliced Wasserstein (SW) distances. Specifically, we leverage both standard SW distances, which provide lower bounds, and lifted SW distances, which provide upper bounds, as predictors of the true Wasserstein distance. To ensure parsimony, we introduce two linear models: an unconstrained model with a closed-form least-squares solution, and a constrained model that uses only half as many parameters. We show that accurate models can be learned from a small number of distribution pairs. Once estimated, the model can predict the Wasserstein distance for any pair of distributions via a linear combination of SW distances, making it highly efficient.  Empirically, we validate our approach on diverse tasks, including Gaussian mixtures, point-cloud classification, and Wasserstein-space visualizations for 3D point clouds. Across various datasets such as MNIST point clouds, ShapeNetV2, MERFISH Cell Niches, and scRNA-seq, our method consistently provides a better approximation of Wasserstein than the state-of-the-art method, Wasserstein Wormhole, and classical methods, particularly in low-data regimes. To illustrate its robustness, we also experiment the method with intra- and inter-class settings. Finally, we demonstrate that \emph{RG} can accelerate Wasserstein Wormhole training, yielding \emph{RG-Wormhole}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposed a method to estimate the Wasserstein distance efficiently. Specifically, the proposed method utilizes the fact that the sliced Wasserstein distance is the lower bound of the Wasserstein distance, and the lifted sliced Wasserstein distance is the upper bound of the Wasserstein distance. Then, the proposed method is the weighted average of the sliced Wasserstein distance and the lifted Wasserstein distance and provides a better estimation of the Wasserstein distance by learning these weights.

### Strengths
* This paper is well-written and easy to follow.
* The proposed method is simple, and it is intuitively correct that the proposed method is a better estimation of the Wasserstein distance than the sliced Wasserstein distance and the lifted Wasserstein distance.

### Weaknesses
* The proposed method is incremental and trivial. It is just a linear combination of the sliced Wasserstein distance and the lifted sliced Wasserstein distance.
* The experimental results are not comprehensive. The main motivation of the proposed method is to efficiently estimate the Wasserstein distance. However, this paper did not evaluate and discuss this efficiency, e.g., the time required to compute, the time required for training.
* There are a lot of papers that proposed the efficient approximation of the Wasserstein distance, but this paper did not compare the proposed method with them in the experiments. Specifically, it is at least necessary to compare the proposed method with the sliced Wasserstein distance, the lifted sliced Wasserstein distance, the max-sliced Wasserstein distance, and other types of approximation of the Wasserstein distance.

### Questions
See the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In many modern machine learning problems, the datasets are collections of distributions, i.e. a meta-distribution over distribution. In those cases, the Wasserstein distance (WD) is an alluring metric for analysis, but constitutes and significant bottleneck in applications that require many pairwise comparisons. This paper proposes a method to estimate the WD by learning a mapping from computationally cheap Sliced Wasserstein (SW) variants to the true WD. The learning process relies on a small number of "ground-truth" WD calculations to fit the estimator, which can then be applied to new pairs.

The proposed method regresses the true WD onto a set of its sliced variants. The predictors include SW-based distances that are known lower bounds (e.g., standard SW, Max-SW) and "lifted" SW distances that serve as upper bounds (e.g., Projected Wasserstein, Min-SWGG). The paper details two linear models for this regression: an unconstrained model with a closed-form least-squares solution, and a constrained model. The constrained model pairs lower and upper bounds, reducing the number of parameters and adding an inductive bias, which is intended for "few-shot" learning scenarios with limited ground-truth samples.

This regression estimator (dubbed "RG") is evaluated in two applications. First, as a standalone estimator, it is applied to a k-NN classification task, where its accuracy is reported to be close to that of the exact WD. The paper also shows it achieves higher accuracy than the Wasserstein Wormhole model in low-data regimes. Second, the authors propose "RG-Wormhole," a hybrid that replaces the expensive WD calculations in the Wormhole training loop with the fast RG estimate. This substitution is shown to reduce training time substantially while reportedly maintaining comparable performance on tasks like point cloud reconstruction and interpolation

### Strengths
The method is elegant, principled, and provides a practical way to "bootstrap" a collection of cheap estimates to learn a high-quality one. The approach is clearly effective, as demonstrated by extensive comparisons and benchmarks across multiple datasets and tasks (k-NN, embedding, reconstruction). Furthermore, the data efficiency is a major plus as the ability to learn a good regression model from very few "ground-truth" Wasserstein distance calculations makes this highly applicable. Finally, the application within the "RG-Wormhole" model is a strong contribution, as it directly addresses the primary computational bottleneck of a current method.

### Weaknesses
The presentation of "Propositions" 1 and 2 (Section 3.2, Page 6) is a notable weakness. These appear to be standard, well-known closed-form solutions for unconstrained and constrained linear regression, respectively. Presenting these basic results as "propositions" with proofs in the appendix feels unnecessary and could be interpreted as an attempt to add theoretical weight where none is needed. An algorithmic and empirical paper like this does not require such justifications for using standard linear regression.

### Questions
The paper highlights the "explainable" nature of linear regression as a reason for its use. Given this, have the authors analyzed the learned coefficients ($\omega$)? Is there a discernible pattern in the learned weights, such as a consistent difference in those assigned to lower bounds versus upper bounds? Similarly, in the constrained model, is there a consistent pattern in the weights for paired "lifted" (upper) and "unlifted" (lower) SW distances across different datasets or dimensions (e.g., does the model learn to trust one more than the other)? The Gaussian simulation in the appendix (Fig 4) hints at this, but a broader analysis on the real datasets would be valuable.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper's central hypothesis is that for a given meta-distribution $\mathbb{P}(\mu, \nu)$, the true $W_p$ can be accurately and efficiently modeled as a simple linear function of various sliced Wasserstein distances. The core technical insight is to use as predictors a vector of SW based metrics that includes both established lower bounds (e.g., SW, Max-SW, EBSW) and upper bounds (e.g., PW, Min-SWGG, EST).

The authors claim the weights $\omega$ for this regression $W_p \approx \omega^\top S_p(\mu, \nu)$ are a stable characteristic of $\mathbb{P}$ and can be learned in a few-shot setting by computing a small number of ground-truth $W_p$ pairs. This yields the RG surrogate which is claimed to be far more accurate than any single SW baseline and outperform deep learning estimators (like Wasserstein Wormhole) in low-data regimes.

### Strengths
- This is a nicely presented paper. The authors' central (and successful) bet is that for a given problem domain $\mathbb{P}$, $W_p$ doesn't just lie somewhere in this interval, but at a consistent relative position that a simple linear model can capture. Thus moves beyond using SW as a fast proxy and instead models the approximation error by bracketing the true value.

- The paper's primary claim of data-efficiency is validated with solid empirical evidence. Table 2 for example demonstrates that with only 100 training pairs, the SOTA deep baseline (Wormhole) is completely unusable (e.g., $R^2$ of 0.65 on ShapeNetV2, -3.6 on MERFISH, 0.04 on scRNA-seq). In stark contrast, the proposed unconstrained RG variants achieve $R^2$ scores of 0.93-0.99 across all datasets, even in the 2,500-dimensional scRNA-seq domain. This high correlation translates directly to downstream utility. For example in the k-NN classification task (Table 1), the RG-seo surrogate (83.5\% accuracy @ k=5) performs almost identically to the true, expensive $W_p$ (84.2\% accuracy @ k=5), while the best single SW baseline is not competitive.

- By replacing $W_p$ inside the Wormhole training loop the authors achieve training speedup. Figure 14 shows almost flat scaling of compute with batch size for RG-Wormhole compared to Wormhole’s quadratic growth, showing that the surrogate can be dropped into OT-based architectures without much performance loss.

### Weaknesses
- The entire method relies on a bootstrap phase where $M$ ground-truth $W_p$ pairs are computed to fit the regression weights $\omega$. The paper frames this as a minor setup cost because $M_0=10$ (yielding $M = \frac{10 \times 9}{2} = 45$ pairs) works for their experiments. However, the paper seems to provide no analysis on how to determine the sufficient $M$ for a new problem.
- OLS is only optimal if the error term $\epsilon$ has constant variance. However figures 6-13 suggests otherwise. Would a simple weighted least-squares (WLS) model yield more accurate surrogate?
- Def 3 adds inductive bias and has half of the parameters, which is often helpful when having limited observed samples. In Table 2 (the $M_0=100$ low-data setting), the unconstrained model is superior in nearly every case (e.g., RG-seo on ShapeNetV2: 0.95 $R^2$ unconstrained vs. 0.92 constrained). This holds even for data size ($M_0=10$) in the appendix (e.g., Fig 8, ShapeNetV2: RG-seo 0.93 unconstrained vs. 0.90 constrained). The empirical justification for the constrained model seems week here.

### Questions
- The RG-seo model requires computing 6 SW-based predictors, some of which (Max-SW, Min-SWGG) require their own iterative optimization. The paper's time complexity analysis and RG-Wormhole training plot (Fig 14) do not provide a direct inference-time comparison. What is the wall-clock time (for a single pair) comparing RG-seo (at inference) to a standard, converged entropic-regularized $W_p$ (Sinkhorn) approximation?
- How sensitive is the learned regressor to the choice and number of slicing directions (L) and to the specific mix of lower vs upper SW variants?
- What if we train an RG model only on intra-class pairs (e.g., (chair, chair)) and test its $R^2$ on inter-class pairs (e.g., (chair, airplane))? vice versa?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a novel regression-based framework for fast estimation of Wasserstein distances between probability distributions. Instead of directly solving the optimal transport (OT) problem or training deep neural networks (e.g., Wasserstein Wormhole, DWE), the authors propose to regress the true Wasserstein distance on a collection of Sliced Wasserstein (SW) and lifted SW distances, which act as lower and upper bounds respectively. 

Two linear regression models are introduced:
Unconstrained model, admitting a closed-form least-squares solution;
Constrained model, which enforces the upper/lower bound structure with half as many parameters.

Once trained on a small set of distribution pairs, the model can predict Wasserstein distances for new pairs at the cost of computing SW distances, significantly reducing computational overhead.

The authors validate the approach on Gaussian mixtures, ShapeNetV2 point clouds, MNIST point clouds, MERFISH cell niches, and scRNA-seq datasets, showing that:

The regression estimator achieves high accuracy even with a few training samples.

It outperforms Wasserstein Wormhole in low-data regimes. Substituting it into Wormhole yields RG-Wormhole, which preserves accuracy while substantially reducing training time.

### Strengths
The paper presents a conceptually simple yet impactful framework for approximating Wasserstein distances through regression on sliced Wasserstein (SW) and lifted SW distances.
This formulation is the first to explicitly treat Wasserstein estimation as a supervised learning problem over a meta-distribution of random distribution pairs, bridging the gap between classical OT theory and data-driven surrogate modeling.

### Weaknesses
### **Weaknesses**

**1. Limited contribution and unclear computational advantage**  
The main technical novelty of this paper lies in Equations (8) and (11), which propose to approximate the Wasserstein distance using a linear regression model on various sliced Wasserstein (SW) and lifted SW distances. While this formulation is conceptually interesting, I have several reservations about its contribution and computational benefits.

**1.1 Computational complexity**  
The claimed time complexity is $O(MKL n(\log n + d))$, where $n$ is the average number of support points per distribution, $M$ is the number of sampled distribution pairs, $K$ is the number of SW variants used, and $L$ is the number of slicing directions.  
However, this is not necessarily more efficient than established OT approximations. For instance, Sinkhorn OT has a complexity of $O(n^2 / \varepsilon)$ and can exploit GPU parallelism. It is therefore unclear in which practical regime the proposed regression estimator is computationally faster or more scalable. A more quantitative runtime comparison would be needed.

**1.2 Lack of transport plan**  
The proposed approach only estimates the **transportation cost**, not the **transport plan**. Hence, it cannot support downstream applications that depend on the coupling (e.g., barycenters, alignment, gradient flows). This limitation makes it less general than classical OT accelerations such as Sinkhorn or low-rank OT methods. If the goal is to compute the Wasserstein distance for a single pair of distributions, this regression-based method would likely not outperform existing solvers.

**1.3 Missing comparison to linear OT**  
If the intended use case is computing pairwise Wasserstein distances among many distributions, it would be natural to compare against **Linear Optimal Transport** (see [Bunne et al., 2020, “Linear Optimal Transport”](https://arxiv.org/abs/2008.09165)), which precomputes a reference map and achieves $\mathcal{O}(C(K,2) n + K n^3)$ complexity for $K$ distributions, where O(n^3) can be improved to o(n^2/\epsilon) if we use Sinkhorn to construct the embedding. Such a baseline would help clarify the advantages (if any) of the proposed regression-based approach for large-scale pairwise computation.

1.4 
I suggest including **direct runtime and accuracy comparisons** against classical OT accelerations such as **Sinkhorn** and **Linear OT** in two regimes:  
(1) a single pair computation, and  
(2) pairwise computations among many distributions.

---

**2. Minor technical and presentation issues**

**2.1 Typographical error**  
Equation (13) in Appendix A.1 appears to have a notational error: the term $\omega^{\top} S_p^{(k)\top}$ is not correctly written, as both $\omega$ and $S_p^{(k)}$ are vectors.

**2.2 Unclear expression in Figure 1**  
Figure 1 may be misleading. $W_p(\mu,\nu)$ is a scalar value, not a vector; similarly, each $S_p^{(k)}(\mu,\nu)$ is also a scalar. Thus, describing $W_p$ as an “L2 projection” onto the “span” of the $S_p^{(k)}$ values is conceptually vague, since the term *span* is not well-defined in this scalar regression context. The authors may consider rephrasing this illustration or providing an intuitive geometric explanation.

**2.3 Limited theoretical analysis**  
While the paper provides clean closed-form regression formulas, it lacks theoretical characterization of the estimator’s bias, variance, or generalization error. Since the model relies on a finite sample of distribution pairs, it would strengthen the paper to analyze how the regression error propagates when predicting Wasserstein distances for unseen distributions.

---

**Overall comment:**  
The idea of regressing Wasserstein distances on sliced variants is creative and potentially useful for few-shot estimation, but the **computational advantage, theoretical justification, and empirical comparisons** to existing OT approximations remain insufficient. The paper would benefit significantly from a clearer positioning against established methods (Sinkhorn, linear OT) and from a more rigorous analysis of its approximation properties.

### Questions
### **Questions**

**Q1. Notation confusion around $P(\mu,\nu)$ in Eq. (7)**  
The notation of $P(\mu,\nu)$ is confusing. From Eq. (7), it appears that $P(\mu,\nu)$ represents the *meta-distribution* used to define the expectation in Eq. (8). In this case, $\mu,\nu$ should be dummy random variables drawn from that meta-distribution. However, as currently written, $P(\mu,\nu)$ seems to be defined as a function of $\mu,\nu$ themselves, which are undefined at this point. Could the authors please clarify this notation and its relation to the sampling process in practice?

**Q2. Sampling and training details**  
How are the training pairs $(\mu_i, \nu_i)$ sampled for regression fitting?  
- Are they generated synthetically (e.g., Gaussian mixtures) or drawn from actual datasets (e.g., ShapeNet, MNIST)?  
- What is the typical number of samples $M_0$ used for estimating regression coefficients, and how sensitive are results to $M_0$?  
Providing more details would help assess reproducibility and stability.

**Q3. Choice of sliced directions ($L$) and stability**  
The paper fixes $L$ as the number of slicing directions for each SW computation. How sensitive are the regression results to the choice of $L$?  
In high dimensions, random projections can yield high-variance estimates of SW distances—does the regression mitigate or amplify this variance?

### Soundness
3

### Presentation
3

### Contribution
2
