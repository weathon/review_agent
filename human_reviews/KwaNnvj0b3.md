# Permutation-based Rank Test in the Presence of Discretization and Application in Causal Discovery with Mixed Data

- Decision: Reject
- Scores: 6, 6, 6, 6

## Abstract
Recent advances have shown that statistical tests for the rank of cross-covariance matrices play an important role in causal discovery. These rank tests include partial correlation tests as special cases and provide further graphical information about latent variables.
   Existing rank tests typically assume that all the continuous variables can be perfectly measured,
   and yet, in practice many variables can only be measured after discretization.
   For example, in psychometric studies,
    the continuous level of certain personality dimensions of a person can only be measured after being discretized into order-preserving options such as disagree, neutral, and agree.
   Motivated by this, we
propose Mixed data Permutation-based Rank Test (MPRT), which properly controls the statistical errors even when some or all variables are discretized.
Theoretically, we establish the 
exchangeability and 
estimate the asymptotic null distribution by  permutations;
as a consequence,
MPRT can effectively control the Type I error in the presence of discretization while previous methods cannot. 
Empirically, our method is validated by extensive experiments on synthetic data 
and real-world data to demonstrate its effectiveness as well as applicability in causal discovery.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the Mixed data Permutation-based Rank Test (MPRT) to address the challenge of testing the rank of cross-covariance matrices in the presence of discretized variables. MPRT controls statistical errors even when some or all variables are discretized by establishing exchangeability and estimating the asymptotic null distribution through permutations, thereby effectively controlling Type I error. The method is validated through extensive experiments on both synthetic and real-world data, proving its effectiveness and applicability in causal discovery.

### Strengths
This paper effectively handles the challenge of discretized variables in causal discovery, a significant advancement over traditional methods.

### Weaknesses
The main limitation of this paper is its reliance on the multivariate Gaussian distribution assumption. This restriction significantly limits the method’s applicability, and it remains unclear how the proposed method would perform with non-normal variables.

### Questions
1. The authors consider discretizing continuous observations into only three categories. What happens when the number of categories increases? Additionally, what is the computational cost associated with the proposed method?

2. The authors examine three scenarios for correlation estimation: between two continuous variables, between a continuous and a discrete variable, and between two discrete variables. How can one distinguish whether the data is discrete or continuous in practice, particularly when the number of categories is large?

3. Does the estimation of correlations in different scenarios impact the asymptotic distribution of the test statistic? Is there any theoretical justification for this?

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
3

### Summary
This paper tackles the challenge of conducting rank tests when continuous variables are discretized, a common scenario in practice that often leads to information loss and failure of existing rank tests. To address this, the authors propose the Mixed-data Permutation-based Rank Test (MPRT), the first approach of its kind in the literature. They provide an asymptotic analysis of MPRT, demonstrating effective control of Type I error. The test accommodates scenarios with fully continuous, partially discretized, or fully discretized variables. Numerical experiments further confirm MPRT’s control of Type I error and its effective performance on Type II error.

### Strengths
To the best of my knowledge, this paper introduces the first statistical rank test designed to address the issue of discretized variables. Great presentation with clear motivation, notation and theorem statements.

### Weaknesses
More theory on the asymptotic power and Type I error would be interesting to think about. Please refer to Questions section for more details.

### Questions
1. Line 208: should be "...sample covariance $\frac{\mathbf{D}^{\mathbf{X}^T}\mathbf{D}^{\mathbf{Y}}}{N-1}$" not $\mathbf{D}^{\mathbf{X}T}$ on the numerator.
2. The authors have empirically verified the Type I and Type II errors. I’m curious if the asymptotic behavior of the Type I error and power of the test could also be theoretically verified. This can better help understand how the power can be further improved. Some convergence rate results of the power or Type I error can also help. Would like to hear more thoughts from the authors on this direction.

### Soundness
3

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
3

### Summary
This paper presents a novel statistical method that addresses limitations in rank tests for cross-covariance matrices with discrete data. The traditional rank tests assume continuous data, leading to challenges in controlling Type I errors in the presence of discretized variables. The proposed method allows effective control of Type I errors with the existence of discretized variables. The method is empirically validated through experiments to demonstrate it is a general rank test method when the data are all continuous and also able to handle the discretization of the data.

### Strengths
- The problem is relevant and well-motivated. 
- This paper introduces a novel approach for the permutation-based rank test, which extends the use of t-separation to the discrete data.
- The paper provides solid theoretical contributions, establishing the exchangeability of canonical variables in permutation tests with discretized data. 
- The proposed method is demonstrated with both synthetic and real-world datasets, showing it's broad applicability.

### Weaknesses
- The results could benefit from deeper intuition and context to make the theoretical contributions more accessible to readers who are unfamiliar with the context. For example, the use of exchangeability for the permutation test, and the definition for the t-separation.

- Although the MPRT is compared with CCA rank tests, a comparison with some CI tests may provide a more comprehensive view of MPRT's positioning.

### Questions
Typo:

l241. result of -> rest of

- How would the proposed method affected by the level of discretization? A discussion of whether different discretization levels affect the method’s robustness would be beneficial.

- Does the result of Theorem 5 affected if the variables are not unit variance?

- How does the proposed method work for the general discrete variables instead of discretized joint Gaussian variables?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper proposes the rank test framework applicable for mixed data. The proposed framework is a novel statistical test for determining the rank of cross-covariance matrices in the presence of discretized data. It is shown and demonstrated in causal discovery across datasets with mixed continuous and discrete variables, where traditional rank tests fail due to discretization challenges.

### Strengths
1. The paper addresses a significant limitation in traditional rank tests, which struggle with discretized data. The proposed method effectively extends rank testing to datasets with mixed continuous and discrete variables, a common scenario in practices. 

2. Extensive empirical results on synthetic and real-world datasets demonstrate the ability of the proposed framework to maintain accurate Type I and Type II error rates, outperforming traditional methods. Additionally, the application to causal discovery with real-world personality data illustrates its practical value in inferring causal structures in realistic, mixed-type data scenarios.

3. By using a permutation-based approach, MPRT avoids complex adjustments for non-continuous data distributions and offers a computationally feasible way to approximate the null distribution, making it suitable for large datasets with mixed data types.

### Weaknesses
1. Many real-world relationships are inherently non-linear. However, the proposed method is restricted to linear relationships. This limits its potential applicability.

### Questions
-

### Soundness
4

### Presentation
3

### Contribution
3
