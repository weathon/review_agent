## Human Reviewer 1

### Summary
This paper proposes an improved Maximum Mean Discrepancy (MMD) method, namely norm-adaptive maximum mean discrepancy (NAMMD), for testing distribution similarity. By incorporating hypothesis testing, this method introduces a practical approach for distribution closeness testing in real-world scenarios.

### Strengths
Rigorous theoretical analysis provides reliability for the paper.
A comprehensive background description makes it easy for readers to understand the problem the paper aims to address.

### Weaknesses
The author does not explain why the definition of NAMMD is formatted as Equation (1). Although the author proves the effectiveness of NAMMD from the perspectives of complexity, hypotheses testing power, and closeness testing power in sections 4.1, 4.2, and 4.3, the fundamental motivation for designing NAMMD is not clearly described.

### Questions
As the author stated, MMD has an inherent issue in closeness testing: "the same MMD value may reflect different levels of closeness between distributions, which makes it less informative." Why doesn't the author redesign a kernel-based closeness testing method? Such approach seems to be able to circumvent the inherent problems of MMD and further address the closeness testing of high-dimensional data.

Why the definition of NAMMD is formatted as Equation (1) and sufficient to achieve the desired goal? This point lacks explanation. After defining NAMMD, the authors do not provide a corresponding justification.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper proposed the norm-adaptive MMD as a testing statistic for distribution closeness testing. Compared with the MMD statistic, it scales MMD with the norm of distributions in RKHS. Theoretical analysis and numerical study demonstrate the superior performance of this framework.

### Strengths
1. The paper is well-written, with explicit technical assumptions and technical proofs included. 
2. The numerical study is solid in justifying the good performance of this framework.  
3. Based on the theoretical analysis and numerical study, I am convinced that NAMMD is a reasonable framework.

### Weaknesses
1. The threshold plays a key role in hypothesis testing. Under the null hypothesis test, the authors utilize the permutation test to estimate the threshold, which is commonly used in literature. Under the case where $NAMMD(P,Q)=\epsilon$, the authors estimate it based on the variance of asymptotic distribution, which further approximates it using empirical variance estimator $\sigma_{X,Y}$. Can the authors provide more theoretical analysis regarding this estiamtor (such as bias, variance, etc?) to justify the soundness of this approximation?
2. Can the authors elaborate more on how to modify NAMMD in the fusing statistics approach?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper develops a new distribution closeness testing method, called norm adaptive MMD.

MMD can be used to measure the discrepancy between two distributions. The statistical properties of MMD-based estimators are well-known, and these can be used to create closeness testing methods. 

The naive MMD-based method, however, is not very informative, because, the closeness measured by MMD varies with the norms of the distributions. This implies that pairs of distributions with different variances can have the same MMD value, even though these pairs of distributions visually can be very different. To overcome this issue, the authors propose to scale the MMD values with the norms of the distributions. This new method is called Norm-Adaptive MMD (NAMMD).

### Strengths
* The paper is well-written and easy to follow.
* The new method is simple and the paper provides new theoretical results for the proposed NAMMD estimator.
* The paper provides experiments on five datasets, and demonstrates that it can work better than MMD and Canonne's tests.

### Weaknesses
The paper claims that standard closeness measures don't work well on complex, high-dimensional datasets, e.g. images. 
The provided numerical experiments show that this distribution closeness works better than other hypothesis tests on benchmark datasets, but it doesn't demonstrate how this improved test can make a difference in some important real-world applications, e.g. on high-dimensional images.

### Questions
Would it be possible to show how the proposed distribution closeness test can make a difference in some important real-world applications?

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 4

### Summary
The authors study distribution closeness testing (DCT) using the maximum mean discrepancy (MMD). 
They find that the MMD can be the same for distributions with different norms in a reproducing kernel Hilbert space (RKHS). 
To address this issue, they propose a novel kernel DCT with a norm-adaptive MMD (NAMMD), which scales the MMD along with the norms of the distributions in the RKHS. 
Theoretical results for the NAMMD test are presented, which demonstrate that the proposed DCT achieves higher test power compared to the standard MMD test. Furthermore, they derive upper bounds on the sample complexity of the NAMMD test. Through empirical analysis, they demonstrate that their kernel DCT can effectively test the closeness of two distributions using both synthetic and real-world data.

#### A brief disclaime:
Please note that my review reflects a limited level of expertise in the specific area for this study. I would appreciate it if this could be taken into account when considering my assessment.

### Strengths
* An important issue of existing DCT using the MMD has been identified, specifically the MMD can be the same for distributions with different norms in a reproducing kernel Hilbert space.
* Rigorous theoretical results are provided.
* Sufficient experiment results are included.

### Weaknesses
#### Major Weaknesses:
* The issue identified by this study may reflect a different perspective on a known issue of the existing MMD DCT related to kernel selection (see quesions below).

#### Minor Weaknesses:
Lines 475-476:
Does “$\|z_0, z_1\| \le \|z_0, z_2\| \le \cdots \le \|z_0, z_9\|$” mean “$\|z_0 - z_1\| \le \|z_0 - z_2\| \le \cdots \le \|z_0 - z_9\|$”?

### Questions
The selection of kernels and the configuration of kernel hyperparameters may be related to the issue identified in this study. 
Some research has addressed issues related to kernel selection and its hyperparameter configuration ([1], [2] and [3]).
I have questions regarding both determination of kernel hyperparameters and kernel selection. 
These considerations could be beneficial for clarifying the challenges and advancing your method.

#### Questions Regarding Kernel Parameter determination.
In lines 106-107, it is possible to adjust the magnitude of the norm by modifying the length-scale hyperparameter $\gamma$ of the Gaussian kernels. Specifically, using a smaller value of $\gamma$ in Figure 1b than in Figure 1a will yield norms of the same magnitude. 

1. In your numerical experiments, did you observe any significant changes in test results by manually adjusting kernel hyperparameters, such as those of the Gaussian kernels?
2. For existing MMD methods, can the selection of kernel parameters help mitigate the issues identified in this study? If so, what are the strengths and weaknesses of your approach compared to other existing MMD DCT methods that adjust kernel parameters ([1], [2] and [3])?


#### Questions Regarding Kernel Selection.
3. Does the issue observed in this study, i.e., the phenomenon where the MMD can be the same for distributions with different norms in a reproducing kernel Hilbert space (RKHS), also occur for the MMD DCT when using the following two types of kernels?

   - Unbounded kernels: For instance, kernels defined by polynomials or matrix products are unbounded but (could be) available within your theoretical framework when using observational data from bounded variables.

    - Kernels with a positive limit at infinity: kernels satisfying $\lim_{\\| \mathbf{x}-\mathbf{x}' \\|_{\infty} \rightarrow \infty} \kappa(\mathbf{x}, \mathbf{x}') = c > 0$.  An example might be a kernel defined as $\kappa (\mathbf{x}, \mathbf{x}') = \exp \left( - \frac{ \\|\mathbf{x} - \mathbf{x}'\\|^2 }{2\gamma} \right)$
    when $\\| \mathbf{x} - \mathbf{x}' \\|\_{\infty} < K$, and otherwise $\kappa(\mathbf{x}, \mathbf{x}') =c$ with positive constats $K$ and $c$.


###### Background for Questions 3.
In lines 084–102 and Figure 1, a kernel has been used such that 
$\lim_{\\|\mathbf{x}-\mathbf{x}'\\|_{\infty} \rightarrow \infty} \kappa(\mathbf{x}, \mathbf{x}') = 0$. 
This phenomenon could arise when a kernel fails to effectively measure similarity between distant data points. I am concerned that this issue might result from choosing a kernel that cannot capture similarity for data points beyond a certain distance.

---

[1] Biggs, F., Schrab, A., & Gretton, A. (2024). MMD-FUSE: Learning and combining kernels for two-sample testing without data splitting. Advances in Neural Information Processing Systems, 36.

[2] Schrab, A., Kim, I., Albert, M., Laurent, B., Guedj, B., & Gretton, A. (2023). MMD aggregated two-sample test. Journal of Machine Learning Research, 24(194), 1-81.

[3] Schrab, A., Kim, I., Guedj, B., & Gretton, A. (2022). Efficient Aggregated Kernel Tests using Incomplete $ U $-statistics. Advances in Neural Information Processing Systems, 35, 18793-18807.

---
### *The following comment was added to my initial review on December 17, 2024*


Dear Authors,

Thank you for your submission and the effort you have invested in this work. I appreciate the depth of your research and the importance of the problem you are addressing.

I would like to sincerely apologize for the delay in identifying and communicating the following issue. After the review period, I reported a potentially critical error in the manuscript to the Area Chair. Under the guidance of the Area Chair, I have now updated my official review to reflect this concern. I deeply regret the delay in providing this feedback and appreciate your understanding.



---

Main Comment:  

The concern pertains to the proofs regarding the test power of the proposed method. Specifically, it appears that the test statistic under the null hypothesis was used to compute the test power, whereas the test statistic under the alternative hypothesis should have been employed. This discrepancy may impact the conclusions regarding the advantage of the proposed method, NAMMD, over MMD.

In particular, the following test statistic seems to have been used in the calculation of the test power:

$$
r_N = \frac{r_M }{4K - \\| (\mu_P + \mu_Q)/\sqrt{2} \\|^2\_{\mathcal{H}\_{\kappa}}}. \tag{E1}
$$

According to lines 1172–1174, this test statistic (Equation (E1)) assumes that $ \mu_P = \mu_Q = (\mu_P + \mu_Q)/2 $, as discussed in lines 848–849. Thus, it appears to be derived under the null hypothesis.

However, since the test power is defined as the probability of rejecting the null hypothesis under the alternative hypothesis, the appropriate test statistic should instead be:

$$
r_N = \frac{r_M}{4K - \\| \mu_P \\|^2\_{\mathcal{H}\_{\kappa}} - \\| \mu_Q \\|^2\_{\mathcal{H}\_{\kappa}}}. \tag{E2}
$$

The proof of Theorem 10 appears to rely on Equation (E1). Specifically, the calculations in lines 1221–1225 employ Equation (E1) rather than Equation (E2). Furthermore, when Equation (E2) is used, the test powers $ p_M $ and $ p_N $ for MMD and NAMMD, respectively, appear identical under the alternative hypothesis (lines 1211–1218).

This raises a concern that the test powers of both MMD and NAMMD may indeed be identical, which could imply that the proposed method does not offer a significant theoretical advantage over MMD.

---

Additional Comment:  

While this issue may be critical to the theoretical justification of your method, I acknowledge that the numerical experiments in your paper demonstrate the effectiveness of the proposed approach. Therefore, it remains unclear why a discrepancy arises between the theoretical analysis and the numerical results.

If your method does indeed exhibit superior performance in the numerical experiments, I encourage you to investigate this phenomenon more deeply. For example, it may be beneficial to analyze whether a specific distribution, such as the t-distribution, emerges in the proposed approach. Additionally, since both the MMD and the NAMMD exhibit different convergence orders under the null and alternative hypotheses, it is essential to evaluate the tail behavior of their asymptotic distributions under the alternative hypothesis. More specifically, carefully revisiting the two equations presented in lines 1211–1218 could help clarify the observed differences.

Identifying the root cause of this discrepancy might be challenging, but I believe such an analysis will not only strengthen the validity of your results but also contribute to further advancing your proposed method.

---
**Update on Scores**:  *(Added to my additional feedback on January 6, 2025)* 

Based on the identified issue, I have updated the following scores to better reflect the paper's current state:  

- **Soundness**: Changed from "4: excellent" to "1: poor" due to the critical error in the proof.  
- **Rating**: Changed from "8: accept, good paper" to "3: reject, not good enough" as the identified issue significantly impacts the validity of the paper's main contributions.  

---

Once again, I apologize for the delay in raising this concern, and I greatly appreciate your understanding. Please feel free to clarify or address this issue in your response.

Best regards,  
Reviewer TXYq

### Soundness
1

### Presentation
2

### Contribution
3

### Rating
3

### Confidence
2