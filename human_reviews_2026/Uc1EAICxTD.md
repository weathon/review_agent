# Independence Test for Linear Non-Gaussian Data and Applications in Causal Discovery

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Independence testing involves determining whether two variables are independent based on observed samples, which is a fundamental problem in statistics and machine learning. Existing testing methods, such as HSIC, can theoretically detect broad forms of dependence, but may sacrifice statistical power when applied to limited samples with background knowledge of the distribution. In this paper, we focus on the linear non-Gaussian data, a widely supported model in scientific data analysis and causal discovery, where variables are linked linearly with noise terms that are non-Gaussian distributed. We provide a new theoretical characterization of independence in this case, showing that constancy of the conditional mean and variance is sufficient to guarantee independence under linear non-Gaussian models. 
Building on this result, we develop a kernel-based testing framework with provable asymptotic guarantees. Extensive experiments on synthetic and real-world datasets demonstrate that our method achieves higher power than existing approaches and significantly improves downstream causal discovery performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the problem of independence testing under linear non-Gaussian data, which is an important problem in both statistics and machine learning. Under the framework where variables exhibit linear relationships and the noise terms follow non-Gaussian distributions, the authors provide a new theoretical characterization of independence, proving that the constancy of conditional mean and conditional variance is sufficient to guarantee independence in this specific model. Based on this insight, they develop a test statistic and derive its asymptotic distribution. Finally, extensive experiments on both synthetic and real-world datasets are conducted to validate the effectiveness of the proposed method.

### Strengths
Originality: This paper provides a new theoretical characterization of the independence testing problem under the framework of linear non-Gaussian models. It introduces an innovative perspective by proposing that the constancy of conditional mean and conditional variance serves as a sufficient condition for independence, which represents a clear conceptual advancement.

Quality: The theoretical derivations in the paper are rigorous and logically coherent. The authors not only define the test statistic and analyze its consistency but also derive its asymptotic distribution, thereby providing a solid theoretical foundation for significance evaluation.

Clarity: The paper is clearly written and well-structured, presenting a coherent flow from problem motivation and theoretical formulation to methodological development and experimental validation. The notation is well-defined, and the main theorems and lemmas are easy to follow.

Significance: The proposed work offers a new theoretical tool for independence inference under non-Gaussian conditions, providing a strong complement to existing independence testing methods that rely on Gaussian assumptions.

### Weaknesses
1. The paper lacks an empirical comparison with RDC [1] and MI [2], two methods capable of detecting a wide range of dependency types.
2. Some relevant literature on independence testing and dependence measurement is missing, such as references [3] and [4].
3. The paper claims to have conducted experiments on the SACHS real-world dataset, but the corresponding results are not reported. In addition, three figures are provided in the Appendix E.3, yet it is unclear which one is used as the ground truth. If only the first figure in Figure 7 is used, the other two seem unrelated to the paper.
4. There are no empirical results for cases where $X$ and $Y$ are high-dimensional. To my knowledge, the main challenges in independence testing arise from nonlinearity and high dimensionality, but the paper presents no results concerning the high-dimensional setting.
5. The definitions of the two evaluation metrics in Table 1 are not formally provided.

Refs: [1] The randomized dependence coefficient. Advances in neural information processing systems, 2013.
[2] Nonparametric independence testing via mutual information. Biometrika, 2019.
[3] Distribution-free tests of independence in high dimensions. 2017, Biometrika.
[4] A new coefficient of correlation. 2021, Journal of the American Statistical Association.

### Questions
1. As stated in line 336 of the paper, I am very interested in seeing how the proposed method performs when using a permutation approach. Specifically, how do the Type I error rate, testing power, and computational efficiency compare with the current approach that uses a Gamma distribution to approximate the null distribution?
2. When evaluating the test power in independence testing, could you fix the parameters $a_i$ and $b_i$, and examine how the testing power of the proposed method changes as they vary from small values (e.g., 0.1) to large values (e.g., 1)?
3. Could you provide more simulation experiments when applying the proposed method to causal discovery, similar to the experimental setups presented in the “Simulations” section of reference [1]? Using only the causal graph corresponding to real-world data does not demonstrate the generalizability of the proposed method to different graph structures.
4. Line 278 of the paper states that both Gaussian and polynomial kernels are used, but the simulations emphasize that only the Gaussian kernel is adopted, which is quite confusing.

Ref: [1] A linear non-Gaussian acyclic model for causal discovery. Journal of Machine Learning Research, 2006.

### Soundness
2

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
3

### Summary
The paper is related to causal discovery and proposes a novel independence test for data with linear dependencies and non-Gaussian noise. The kernel-based testing framework comes with asymptotic guarantees. Several numerical experiments for synthetic as well as real-world data are performed and results are compared to different baseline approaches.

### Strengths
S1 The considered problem is interesting and relevant.
S2 The paper is technically sound and overall well-written.
S3 The methods used are suitable.
S4 The evaluation and comparison against baselines is convincing.

### Weaknesses
W1 The required computational costs and scalability remain somewhat unclear.
W2 Limitations of the paper’s methods could be better discussed.

### Questions
What are the required computational costs?

Does the approach scale in the number of components d and the sample size n?

Which model quantity could become a bottleneck?

Could the assumption of linear mixtures be relaxed to more general ones?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper targets independence testing tailored to linear non-Gaussian settings. Its main theoretical observation is a characterization: for linear mixtures of independent non-Gaussian components, $X \mathrel{\unicode{x2AEB}} Y$ holds iff both the conditional mean and the conditional variance are constant. Building on this, the authors design a test statistic (LiNGIC) that couples a universal kernel on $X$ with a quadratic feature/polynomial kernel on $Y$, making it sensitive precisely to departures in those two conditional moments. They derive an asymptotic null and use a Gamma approximation for fast calibration. Empirically, the paper (i) reports Type-I control and power under several non-Gaussian families and increasing mixture complexity, and (ii) demonstrates downstream causal discovery by swapping this test into Direct-LiNGAM. Across synthetic settings, the method shows higher power than standard unconditional tests in the target regime, while keeping computation low thanks to the closed-form calibration.

### Strengths
1. The paper targets the dependence between linear non-Gaussian mixtures and proves a specific and clean equivalence between independence and constancy of the first two conditional moments. 
2. The theoretical work of the test statistic seems to be solid and complete, although standard results have been given in other papers.
3. The empirical results on pairwise dependence clearly show the advantage of the proposed method, as it generally has higher powers across all regimes.

### Weaknesses
1. The proposed method can only be applied to pairwise independence testing. A joint independence test (e.g. dHSIC) between a number of variables would have more potential applications, especially in causal discovery. I wonder whether such an extension could be possible. 
2. The computational advantage of the proposed method would be strengthened if the author could provide runtime comparisons of the method on pairwise independence testing and application to causal discovery. 
3. In the experiments, the proposed method and HSIC both used the Gamma approximation. Although it alleviates the computational burden, its performance might be reduced. A usual alternative is to use the permutation test, and the author did not discuss this part. Therefore,  results for HSIC and LiNGIC with permutation calibration should be included to see whether LiNGIC + Gamma approximation remains superior.
4. In general, the power issue is more severe in high-dimensional causal discovery problems, i.e., many (joint) independence tests may simply reject dependence if multiple variables are involved. Therefore, I would like to see more experiments on LiNGAM with $d = 20, 50, 100$ with SHD, F1 score comparison to evaluate whether the proposed method brings a significant advantage to this scenario.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes a new independence test, LiNGIC, specifically for linear non-Gaussian (LiNGAM) data. It argues that general tests like HSIC lack statistical power in this structured setting. The core contribution is Theorem 4.2, which proves that for LiNGAM data, independence is equivalent to the conditional mean and conditional variance both being constant. The LiNGIC statistic is a kernel-based test designed to check these two conditions simultaneously. Experiments demonstrate that LiNGIC achieves higher statistical power than baselines and improves downstream causal discovery performance.

### Strengths
- The paper provide strong theoretical results. 
- In general, the paper is well-written.

### Weaknesses
- Although the authors have made efforts to justify developing conditional independence tests under the strong linear non-Gaussian assumption, I remain unconvinced. In practice, it is difficult to verify that real-world data strictly satisfies this assumption. Moreover, it is not entirely fair to compare their method with approaches that assume general nonparametric settings, as the learning problem is substantially simpler under the linear non-Gaussian constraint.

- The evaluation of causal discovery performance is limited, with insufficient analysis across different graph structures, noise types, and data settings. Key dataset details (e.g., observational vs. interventional) are also unclear.

- The paper does not include comparisons with general nonparametric conditional independence tests or established causal discovery methods, limiting the empirical context of the results.

### Questions
1. The authors suggest that in certain applications, real-world data follows a linear non-Gaussian setting. Can they provide a concrete example? In particular, the proposed method is evaluated on the Sachs dataset, yet there is no evidence supporting a linear non-Gaussian assumption. In fact, methods designed for nonlinear settings often achieve better performance, indicating that the Sachs dataset may not be an ideal benchmark for this paper.

2. While the paper emphasizes applications in causal discovery, the evaluation in this context is quite limited. I recommend providing detailed causal discovery results under various graph structures and data settings. Additionally, the description of the Sachs dataset in the appendix is insufficient. Which type of data was evaluated: observational, interventional, or both?

3. To make the evaluation more comprehensive, the authors should consider including results from general nonparametric constraint-based conditional independence tests or causal discovery methods, such as:

> [1] Strobl, Eric V., Kun Zhang, and Shyam Visweswaran. "Approximate kernel-based conditional independence tests for fast non-parametric causal discovery." Journal of Causal Inference 7.1 (2019): 20180017.

> [2] Zarebavani, Behrooz, et al. "cuPC: CUDA-based parallel PC algorithm for causal structure learning on GPU." IEEE Transactions on Parallel and Distributed Systems 31.3 (2019): 530-542.

> [3] Zhang, Hao, et al. "Residual similarity based conditional independence test and its application in causal discovery." Proceedings of the AAAI conference on artificial intelligence. Vol. 36. No. 5. 2022.

4. The "Application in Causal Discovery" (Section 6.2) shows mixed results. Although the proposed method performs best on data with "Uniform" noise, it fails on "Laplace," "Student t," and "TruncNorm" noise, achieving F1 scores near 0.0—comparable to other methods. This is a significant weakness, as it contradicts Figures 3 and 4, which show that LiNGIC has superior statistical power for these distributions. The paper does not address this discrepancy.

### Soundness
3

### Presentation
2

### Contribution
2
