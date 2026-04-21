# Improved Analysis of Sparse Linear Regression in Local Differential Privacy Model

- Avg Score: 6.33
- Decision: Accept (poster)
- Scores: 8, 5, 6

## Abstract
In this paper, we revisit 
the problem of sparse linear regression in the local differential privacy (LDP) model. Existing research in the non-interactive and sequentially local models has focused on obtaining the lower bounds for the case where the underlying parameter is $1$-sparse, and extending such bounds to the more general $k$-sparse case has proven to be challenging. Moreover, it is unclear whether efficient non-interactive LDP (NLDP) algorithms exist. To address these issues, 
we  first consider the problem in the $\epsilon$ non-interactive LDP model and provide a lower bound of $\Omega(\frac{\sqrt{dk\log d}}{\sqrt{n}\epsilon})$ on the $\ell_2$-norm estimation error for sub-Gaussian data, where $n$ is the sample size and $d$ is the dimension of the space. 
We propose an innovative NLDP algorithm, the very first of its kind for the problem. As a remarkable outcome, this algorithm also yields a novel and highly efficient estimator as a valuable by-product. Our algorithm achieves an upper bound of $\tilde{O}({\frac{d\sqrt{k}}{\sqrt{n}\epsilon}})$ for the estimation error when the data is sub-Gaussian, which can be further improved by a factor of  $O(\sqrt{d})$ if the server has additional public but unlabeled data. 
For the sequentially interactive LDP model, we show a similar lower bound of $\Omega({\frac{\sqrt{dk}}{\sqrt{n}\epsilon}})$. As for the upper bound, we rectify a previous method and show that it is possible to achieve a bound of $\tilde{O}(\frac{k\sqrt{d}}{\sqrt{n}\epsilon})$. Our findings reveal fundamental differences between the non-private case, central DP model, and local DP model in the sparse linear regression problem.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies sparse linear regression in the different local differential privacy models (LDP). 

For non-interactive LDP they propose an algorithm with estimation error $\tilde{O}(\frac{d\sqrt{k}}{\epsilon\sqrt{n}})$, and show a lower bound $\Omega(\frac{\sqrt{dk\log d}}{\epsilon \sqrt{n}})$ (for sub-Gaussian covariates). In addition, they show that it is possible to improve the upper bound by a factor $\sqrt{d}$ given public unlabeled covariates.

For interactive LDP they propose an algorithm with estimation error $\tilde{O}(\frac{k\sqrt{d}}{\epsilon\sqrt{n}})$, and show a lower bound $\Omega(\frac{{\sqrt{dk}}}{\epsilon \sqrt{n}})$ (for sub-Gaussian covariates)

### Strengths
There are a few new non-trivial results that improve over the state of the art. The paper is well-written, I really enjoyed reading it. The contribution and comparison with prior works is clear. The idea behind Algorithm 1 is very nice and seems to be new (though I'm not an expert in the field, so I'm not 100% sure).

In addition, they found and fixed a bug in one of the results of prior work [1] on the iterative LDP settings that implied an incorrect upper bound. I briefly checked it, and indeed Hoelder's inequality in the proof of Theorem 9 is used incorrectly there, so it is good that this mistake was found and fixed.

### Weaknesses
I didn't find major weaknesses. However there is one thing (which I formulate in the Questions below) that is confusing to me.

### Questions
Your proof of Theorem 7 looks very similar to the proof of Theorem 9 from [1] (and you mention that). Could you please explain what are important differences, assuming linear regression settings? (I didn't check the details, so maybe I missed something). From the first glance it looks like the proof from [1] works not only for the uniform distribution, but also for 1-sub-Guaussian distributions (modulo their wrong bound in the very beginning), and if it is the case, it should also work for you settings, or did I miss anything important?

And one minor thing: I suggest to move Table 1 to the introduction.

[1] Di Wang and Jinhui Xu. On Sparse Linear Regression in the Local Differential Privacy Model.
IEEE Transactions on Information Theory 2021

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the problem of sparse linear regression under the local differential privacy setting. The authors provide new lower bound results for this problem with a $k$-sparse underlying model parameter. In addition, the authors develop efficient upper bound algorithms for the same problem.

### Strengths
The strengths of the current paper are summarized as follow:
1. The authors provide new lower bound results for sparse linear regression under local differential private model.
2. The authors develop new efficient algorithm for solving the same problem.

### Weaknesses
The weaknesses of the current paper:
1. It is unclear the dimension dependence in the lower bound is due to the hardness of the LDP setting or the norm of the data.
2. It is unclear why the authors need the $\ell_1$ norm bound in their results.
3. It is unclear why Assumptions 1 and 2 are both required in the upper bound results.
4. Why it is reasonable to consider the sparse model in the classical setting?
5. The sample complexity requirement seems to be very bad in terms of $d$.

### Questions
Here are some additional questions I have for the current paper:
1. For the Remark 2, why the authors claim that the sparse linear models in the non-interactive LDP setting are ill-suited? It seems to me that the dimension dependence in Theorem 1 comes from the norm of the data, what will the result look like if you assume the data vector to be $\ell_2$ norm bounded? In addition, the results in Raskutti et al. 2011 and Cai et al. 2021 seem to assume the data vector to be $\ell_2$ norm bounded.
2. For Theorem 3, why do you assume Assumptions 1 and 2 holds simultaneously? In Assumption 1, you assume $x$ with covariance $\Sigma$, and the transformed data to be Sub Gaussian. In Assumption 2, you further assume $x_i$ has variance $\sigma^2$. In addition, what is the assumption on $\zeta$?
3. If the lower bound has nothing to do with $\ell_1$ norm bound, you should give the results in terms of the $\ell_2$ norm bound. 
4. Whether the upper bound results can be extended to the $\ell_2$ norm bound case?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies sparse linear regression under local differential privacy. Firstly, it establishes a lower bound under a non-interactive LDP protocol for sub-gaussian data. Secondly, it proposes the first upper bound that has a $\sqrt{d}$ gap compared to the aforementioned lower bound. It also demonstrates that this gap can be closed if public unlabeled data is available. Lastly, in the case of sequentially interactive protocol, this paper presents a lower bound and corrects the results of the iterative hard thresholding algorithm from prior work.

### Strengths
1. This paper is thorough and clearly written. 
2. The problem is well-defined and important.

### Weaknesses
The upperbound and lowerbound do not match. It is unclear which bound is tight. Also $n$ has to be greater than $O(d^4)$ to achieve a rate of $O(d)$ in Theorem 3.

### Questions
1. Is the l2 norm the right metric for linear regression? For example, Cai at el (2021) consider $\|\theta^{priv}-\theta^*\|_\Sigma$, which corresponds to minimal emprical risk. Do the results also hold under this normalized metric?
2. Is k used in Algorithm 1?
3. Regarding Remark 3, is it necessary to release the covariance matrix privately in LDP model? Can you privatize the two terms in OLS solution together?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
