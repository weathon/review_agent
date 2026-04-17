# Adaptive kernel selection for Stein Variational Gradient Descent

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
A central challenge in Bayesian inference is efficiently approximating posterior distributions. Stein Variational Gradient Descent (SVGD) is a popular variational inference method which transports a set of particles to approximate a target distribution. The SVGD dynamics are governed by a reproducing kernel Hilbert space (RKHS) and are highly sensitive to the choice of the kernel function, which directly influences both convergence and approximation quality. The commonly used median heuristic offers a simple approach for setting kernel bandwidths but lacks flexibility and often performs poorly, particularly in high-dimensional settings. In this work, we propose an alternative strategy for adaptively choosing kernel parameters over an abstract family of kernels. Recent convergence analyses based on the kernelized Stein discrepancy (KSD) suggest that optimizing the kernel parameters by maximizing the KSD can improve performance. Building on this insight, we introduce Adaptive SVGD (Ad-SVGD), a method that alternates between updating the particles via SVGD and adaptively tuning kernel bandwidths through gradient ascent on the KSD. We provide a simplified theoretical analysis that extends existing results on minimizing the KSD for fixed kernels to our adaptive setting, showing convergence properties for the maximal KSD over our kernel class. Our empirical results further support this intuition: Ad-SVGD consistently outperforms standard heuristics in a variety of tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an adaptive kernel selection approach, Ad-SVGD, for Stein Variational Gradient Descent (SVGD). The main idea is to dynamically adjust kernel parameters by maximizing the kernelized Stein discrepancy (KSD) during the SVGD process, such that the descent direction can better fit the data. The authors also provide a theoretical convergence analysis for the proposed method. The experimental results show the effectiveness of Ad-SVGD, which outperforms the standard SVGD with the median heuristic.

### Strengths
1. The idea of dynamically adjusting the kernel parameters using KSD is reasonable.

2. The theoretical convergence guarantee has been established.

### Weaknesses
1. From the theoretical perspective, it seems that some assumptions are too strong and not easy to check, e.g., Assumption 4, which depends on the behavior of the parameter to be optimized.

2. The experiments are not comprehensive enough. First, only a one-dimension toy example is considered in Section 5.2, and examples with higher-dimension should also be implemented. Second, only the original SVGD is used for comparison, while there are indeed lots of variants and improvements have been developed in recent years. Third, as a machine learning paper, tasks like classification and regression should be employed for evaluation, such as that used in the original SVGD paper and its followers.

3. The idea of adjusting or learning descent direction, with respect to certain criteria, for SVGD is not new, and related studies, e.g. [1][2], should be discussed or better compared with.

   [1] L. di Langosco et al. Neural Variational Gradient Descent. 4th Symposium on Advances in Approximate Bayesian Inference, 2022.

   [2] Q. Zhao et al. Stein variational gradient descent with learned direction. Information Sciences, 2023.

### Questions
The authors should address the issues mentioned in the "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to improve Stein Variational Gradient Descent (SVGD) by addressing one of its known limitations — the strong dependence on kernel choice. Standard SVGD typically uses a fixed kernel (such as RBF) with a manually chosen or heuristic bandwidth (e.g., the median heuristic). In practice, this choice heavily influences both stability and convergence, especially in high-dimensional or multimodal settings.

To mitigate this issue, the paper proposes Adaptive SVGD (Ad-SVGD), which dynamically updates kernel parameters (such as the bandwidth) during training. The kernel is optimized by maximizing the kernelized Stein discrepancy (KSD) at each iteration, based on the intuition that a larger KSD corresponds to faster reduction in the KL divergence between the particle distribution and the target posterior.

Experimental results indicate that Ad-SVGD can better maintain posterior variance and alleviate particle collapse compared to standard SVGD with a fixed bandwidth.

### Strengths
1. The dependence of SVGD on kernel choice is well-known. Adapting the kernel parameter through KSD maximization is an intuitive and well-motivated idea that could improve practical robustness.
2. The modification to SVGD is simple and lightweight, involving only an additional optimization step for the kernel parameter. It can be easily integrated into existing SVGD frameworks without altering the main update rule.

### Weaknesses
1. Most experiments are performed on toy or synthetic datasets with relatively low dimensionality. While results show qualitative improvements, more challenging or high-dimensional tests—such as Bayesian neural networks or large-scale posterior inference—would make the empirical claims more convincing.
2. The experiments are conducted mainly on small synthetic problems, so the practical effectiveness of the approach on larger or more complex tasks remains uncertain.
3. The paper does not discuss computational cost or scalability, even though the adaptive kernel adds extra optimization steps that could affect efficiency.
4. The comparison with other adaptive or multi-kernel SVGD variants is limited, making it difficult to clearly position this work among existing approaches.

### Questions
1. How sensitive is the method to the kernel learning rate and the number of inner optimization steps? Would overly aggressive kernel updates cause instability in particle dynamics?
2. If the kernel bandwidth is initialized poorly, can the adaptive process still recover a reasonable value, or does it tend to get stuck in suboptimal regions?
3. How does the adaptive kernel perform in higher-dimensional inference problems (e.g., 100–1000 dimensions)? Are there potential scalability issues related to the additional KSD computation?
4. It would strengthen the paper to include comparisons with related adaptive or multi-kernel approaches such as Multiple Kernel SVGD (Ai et al., 2023) and Matrix Kernel SVGD (Wang et al., 2019), to better position this work within the literature.
5. Since all current experiments are based on small or synthetic datasets (e.g., 1D Gaussian mixtures, linear ODE inverse problems, and small Gaussian process regression), it remains unclear whether the approach can handle complex or real-world posteriors, such as Bayesian neural networks.
6. The adaptive step introduces extra computation for matrix traces and kernel gradients, yet there is no discussion of runtime or complexity. Providing even a brief analysis would clarify the practical overhead.
7. The theoretical analysis assumes that the algorithm can approximately maximize KSD at each step, which may be difficult to guarantee in practice. Could the authors discuss how sensitive the theoretical results are to this assumption, and whether approximate optimization affects convergence?

### Soundness
2

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
The paper presents an adaptive kernel learning approach for Stein Variational Gradient Descent (SVGD) through the use of a parameterized family of kernels. The proposed method incorporates an additional optimization step to learn the optimal kernel parameters at each SVGD particle update. A convergence analysis for this adaptive setting is also provided, extending existing results for SVGD with fixed kernels. Experimental evaluations on synthetic datasets demonstrate that the proposed learning algorithm enhances the reliability and robustness of SVGD compared to a baseline SVGD with heuristic-based adaptive kernels.

### Strengths
The paper presents a meaningful contribution by introducing a parameterized kernel learning mechanism for Stein Variational Gradient Descent (SVGD), supported by theoretical analysis. The proposed approach adds an adaptive optimization step that learns kernel parameters during each SVGD update, thereby reducing dependence on manually selected kernels and heuristic tuning. The paper is clearly written and logically structured, with mathematical derivations and algorithmic steps explained in a way that is easy to follow. The experimental results, though limited to synthetic data, convincingly demonstrate that the method improves the reliability and adaptability of SVGD compared to heuristic-based adaptive kernel baselines.

### Weaknesses
While the proposed approach is promising, the experimental validation remains limited. The experiments are conducted only on synthetic datasets, without evaluation on real-world problems such as posterior estimation, one of the most prominent and successful applications of SVGD. This limitation makes it difficult to assess the practical usefulness and scalability of the proposed adaptive kernel learning method. Additionally, the chosen parameterized family of kernels appears relatively narrow in scope. A broader discussion or exploration of alternative parameterized kernel classes would strengthen the work and better demonstrate the generality and adaptability of the proposed framework.

From a theoretical perspective, the convergence analysis relies on several strong assumptions. It is unclear how these assumptions are satisfied or approximated in practical applications. In particular, since the authors acknowledge that Assumption 4 cannot be guaranteed, it would be valuable to include a more detailed discussion of how this limitation affects the theoretical results and whether any relaxation or empirical verification of this assumption is possible.

### Questions
Since the authors acknowledge that Assumption 4 cannot be guaranteed, how does this limitation affect the validity and interpretation of the convergence analysis results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a sampling algorithm, namely Adaptive Stein Variational Gradient Descent (**Ad-SVGD**), which is a modification of the standard SVGD by adaptively tuning the kernel parameters during sampling. The key idea is to maximize the kernelized Stein discrepancy (KSD) with respect to the kernel parameters at each iteration, thereby improving the rate of KL decrease and enhancing sample diversity. 

The authors:

1. develop an adaptive kernel selection mechanism via gradient ascent on the KSD (**Algorithm 1**).

2. provide a convergence analysis extending prior KSD-based results (e.g., Salim et al., 2022) to adaptive kernels under Talagrand’s inequality (**Theorem 3**).

3. present empirical results on Gaussian mixtures, ODE-based inverse problems, showing that Ad-SVGD mitigates variance collapse and performs better than the median heuristic.

### Strengths
1. **Good motivation**: the choice of kernel is essential in improving the performance of SVGD. The Ad-SVGD provides a solution from the perspectively of maximizing the KL-decay from a set of parametrized kernels, which can provide insights to further study of the choice of kernels in SVGD.

2. **Clear Math Formulation**: the mathematical formulations of the Ad-SVGD and its convergence are sound and clear.

### Weaknesses
**The advantages of the proposed method Ad-SVGD is not studied completely**. In the introduction, the authors mentioned multiple advantages of using adapting kernels, including 

(1) various kernels induce geometries can make SVGD converge in stronger metric 

(2) larger KL decay to make the convergence faster. 

However, none of these advantages are studied either theoretically or numerically in the paper: the current theory only show asymptotic convergence in maximum KSD within the parametrized kernel class, and it is not clear how the maximum KSD induces convergence in stronger metric and how the Ad-SVGD is more efficient than SVGD. In the numerical examples, the running times for Ad-SVGD and Med-SVGD are also not reported.

### Questions
1. What is the difference between using finite many kernels as in [1] and using parametrized kernels as in Ad-SVGD? What are the advantages of using parametrized kernels?

2. Based on Theorem 3 and results in [2], what is the iteration complexity comparison between Ad-SVGD and SVGD in the mean-field sense?

3. Typos: line 115, equation for $\mathcal{A}_\pi^k(x)$; line 155: proportional to KSD not squared KSD; in Assumption 4, the subindex $n+1$ should be $n$

### Soundness
3

### Presentation
3

### Contribution
2
