# UKAT: Uncertainty-aware Kernel Association Test

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Modern data collection methods routinely provide uncertainty estimates alongside point measurements, yet standard statistical tests typically ignore this valuable information. We introduce \texttt{UKAT} (Uncertainty-aware Kernel Association Test), a general framework for testing associations between variables while explicitly incorporating measurement uncertainties. \texttt{UKAT} treats each observation as a distribution characterized by its mean and uncertainty, then applies the Hilbert-Schmidt Independence Criterion (HSIC) to compare these distributional representations in kernel Hilbert spaces. Through extensive simulations, we demonstrate that \texttt{UKAT} achieves substantially higher statistical power than traditional association tests while maintaining proper Type I error control. We also validate \texttt{UKAT}'s versatility across diverse scientific domains in proof-of-principle applications, including detecting prompt-induced effects in large language model responses on self-reported confidence and identifying associations in physical measurements with error estimates.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the **Uncertainty-aware Kernel Association Test (UKAT)**, a framework for statistical testing that explicitly incorporates per-observation measurement uncertainty. The authors argue that standard tests ignore this valuable information, leading to reduced statistical power.

### Method

UKAT's core innovation is to treat each observation not as a single value, but as a probability distribution characterized by its mean ($X$) and uncertainty ($U$). It represents each observation as an augmented data point $\Theta = [X, U]$, assumed to be Gaussian. The framework uses a distance metric on these points, equivalent to the Wasserstein distance for Gaussians, to construct a kernel for the Hilbert-Schmidt Independence Criterion (HSIC) test.

### Applications

Real-world applications demonstrate UKAT's ability to uncover novel insights. It detected significant behavioral changes in LLM responses based on self-reported confidence that accuracy-only tests missed. In astronomy, it identified potential systematic biases in exoplanet data by finding associations within measurement errors.

### Strengths
The idea is quite simple to understand and the exposition of the idea was straightforward.

### Weaknesses
These are the main weakness of the paper: 

1.  **Limited methodological novelty.** The core technical proposal is to concatenate an observation’s mean and uncertainty into a two-dimensional vector and apply the standard HSIC test. This represents a straightforward application of an existing statistical tool to augmented inputs. The connection to the Wasserstein distance for Gaussians appears to serve mainly as an interpretation rather than a design principle, and the work does not introduce a new kernel, test statistic, or theoretical framework. Overall, the methodological advance is limited.

2.  **Narrow Scope and Restrictive Assumptions.** The reliance on a Gaussian assumption to make the link to Wasserstein distance undermines the primary advantage of HSIC as a non-parametric test, making the approach less flexible than claimed.

3.  **Confusing experiments.** The experimental results only go up to n=50, and does not provide a sample size power plot, nor does it consider size plots for larger sample sizes. The paper also doesn't elaborate what the AUC metric does. Further it's potentially misleading to label HSIC applied to (mean, std) data pairs as UKAT-C, it should be considered a baseline method.

In its current form, the manuscript presents a simple idea without the necessary methodological depth, novelty, or rigorous comparison to established alternatives to be considered a significant contribution.

### Questions
1. The AUC metric is unclear, can you elaborate exactly how it is calculated?
2. Why do you only consider n=50? What happens at larger n? 
3. Why is HSIC with  (mean, std) called UKAT-C? It appears to be a direct application of HSIC to this type of data and should be considered a baseline
4. The standard T-test seems to be doing perfectly fine in terms of power and size? Can you construct a more convincing case when the standard T-test breaks and your proposed method works much better? 
5. Can you provide a density plot of the data you're testing? It would help understanding what exactly you're testing for.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UKAT, a framework that incorporates uncertainty into independence testing, which is often ignored by traditional statistical tests. UKAT represents each observation not as a point $X$, but as a distribution $N(X, U^2)$, where U is measurement
uncertainty. The core idea is to use the Wasserstein distance between these distributions, which simplifies to a Euclidean distance on $\Theta=[X, U]$, to construct an energy kernel. This kernel is then used within the standard HSIC framework to test for associations. Simulations and applications demonstrate that UKAT achieves higher statistical power than traditional tests while maintaining proper Type I error control.

### Strengths
1. The idea of embedding uncertainty directly into hypothesis tests/kernel methods represents an interesting direction. 
2. The writing is generally clear and well-structured. I appreciate the figures. 
3. The proposed UKAT framework is conceptually simple yet intuitive.

### Weaknesses
1. There is no new theorem or substantive analytical insight beyond restating existing kernel-independence theory under a Gaussian-uncertainty parameterization. As such, the paper’s theoretical contribution appears limited. 
2. The proof of Proposition 3.1 seems incorrect. The author stated that the universality of the proposed kernel follows from the fact that k is characteristic and translation-invariant. However, k is not translation-invariant, therefore fails to establish universality as claimed.
3. The paper is framed as a general association/independence test, leveraging the HSIC framework to detect arbitrary dependencies. However, the entire simulation study fails to test it. Instead, two special cases: detecting differences in group means and group variances. Therefore, the paper provides no evidence that the proposed energy kernel outperforms simpler tests for general association testing. 
4. There is no formal theoretical grounding for the robust variant UKAT-R. 
5. If I understand correctly, UKAT essentially implicitly reweights samples using the uncertainty estimates contained within the dataset as prior information. Therefore, its practicality heavily depends on the quality of these uncertainty estimates, which are often untestable or unverifiable. In real-world scenarios, this leaves us uncertain about when this approach can be reliably applied. Furthermore, the baseline lacks adaptive kernel-learning or data-driven reweighting methods [1-4]. If such adaptive strategies can be learned without explicit uncertainty priors, the practical significance of UKAT would be substantially diminished.

References:  
[1] Liu et al, Learning Deep Kernels for Non-parametric Two-Sample Test. ICML 2020.  
[2] Ren et al, Learning Adaptive Kernels for Statistical Independence Tests. AISTATS 2024.   
[3] Xu et al, Learning Deep Kernels for Non-Parametric Independence Testing. Arxiv.   
[4] Li et al, Extracting Rare Dependence Patterns via Adaptive Sample Reweighting. ICML 2025.

### Questions
1. Is there any discussion/experimental evidence why RBF and Laplacian kernels yield suboptimal power?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new dependence test: this is a statistical test aimed to determine whether two random variables $X$ and $Y$ (observed through their joint realizations) are statistically dependent. Unlike typical existing dependence tests, the new test is “uncertainty-aware” in the sense that each realization of $X$ is allowed to be accompanied by an uncertainty measure. For instance, this can be a standard deviation (associated with one realization, not the full distribution of $X$). The new test is built on the well-known kernel-based HSIC test.

### Strengths
The direction this paper aims to tackle (i.e., accounting for uncertainty on each realization when doing a statistical test) is technically interesting. The approach essentially views each realization as a distribution, which is a rather unusual view (in a positive way); though, it is not the first work to do so. I think tackling this problem can lead to significant development down the line in the future. At a high level, the paper is easy to understand (though several technical details are missing).

### Weaknesses
While the goal of treating each observation as a distribution in a dependence test is technically interesting, the paper falls short of what is expected of a statistical test in a number of ways. To briefly provide a few examples, the paper does not mathematically precisely describe how $u$ (the uncertainty measure) and $x$ (a realization of $X$) are related. There is an implicit assumption but the assumption is not sufficiently articulated.Secondly, it is unclear for what kind of joint distribution (that jointly generates $(x, y, u)$) would the proposed test provide a consistent result. This is a natural theoretical result expected from a new statistical test since it will clearly define the class of distributions that the test can work. Without this result,  given a problem, it is unclear whether the proposed test is applicable.  No such result is provided. 

Further the new test builds on top of the well-known HSIC (Hilbert-Schmidt Independence Criterion) dependence test of Gretton et al. The present work proposes to use two positive definite kernels of a particular form with HSIC, limiting the novelty.

More discussion points and specific questions are given in Questions.

### Questions
**Q1**: Standard HSIC operates on two random variables $(X,Y)$ following some (unknown) joint distribution $P$. The proposed test in this work operates on three random variables $(X, U, Y)$, where $X$ and $U$ are univariate variables, and $U$ represent some kind of uncertainty measure on $X$. **Question:** What is the assumption on the relationship between $X$ and $U$. This is an important point that is never elaborated precisely. In Sec 3, at L166,
 
> We therefore characterize each observation as $N(x_i , u_i^2 )$.

Is this just an example, or an assumption? If it is an assumption, it must be stated more clearly. Does this mean $X$ follows a Gaussian distribution? Or does a realization $x_i$ act as the mean of another random variable? If so, what is that random variable?


**Q2**: Following Q1, with the normality assumption, what happens in practice if $u$ is a standard deviation for $x$, but $x$ does not follow a normal distribution? This is an important point that is not discussed sufficiently. In practice, it is highly unlikely that the normality assumption would hold in general. 


**Q3**: Theoretically, for what kind of joint distribution $P$ (that generates $(X, U, Y)$) would the test provide a consistent result? To be more concrete, for instance, what is the assumed factorization form of $P(X, U, Y)$? If the three variables are independent (i.e., $P(X,U,Y) = P(X)P(U)P(Y)$), would the test be able to control the type-I error, for instance? What about other less trivial forms of factorization? I would like to see this kind of consistency statement:

> For $P \in $ (Some Distribution Class), under the alternative hypothesis $H_1$, the new test gives a test power of 1 as the sample size goes to infinity.

What is “(Some Distribution Class)”? This point is related to Q2. It is important to understand the scope that the new test can apply to.


**Q4**: On a related note, at line 160, 

> Under the null hypothesis, the distribution of distributions is independent of other covariates.

Could you please write down mathematically what the null hypothesis $H_0$ is? This is never stated mathematically. And what is the alternative hypothesis $H_1$? 

**Q4.1**: If $X,Y$ are independent, is it possible that the presence of $U$ can result in a false positive (i.e., reject $H_0$ when it should not be rejected)?

**Q4.2**: The other way. If $X, Y$ are dependent, is it possible that the presence of $U$ can result in a false negative?


**Q5**: Sec 3.2, L201,

> RBF and Laplacian kernels are also universal but yield uncalibrated p-values…

Could you please precisely describe what this means? Do you mean, with a wrong kernel choice, the new test can give an uncontrolled type-I (false positive) error? If so, then this is a big problem. The existing HSIC test at least has a well-controlled type-I error for any kernels under $H_0$; though, it may have very low test power under $H_1$ if inappropriate kernels are used.


Owing to the above concerns, I cannot give a strong recommendation.

### Soundness
1

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
A new test of dependence, incorporating uncertainty.

### Strengths
Seems like it works good.

### Weaknesses
I am really not that sure about some critical things.

### Questions
My name is Joshua Vogelstein, I’ve written many papers on two-sample testing, including my favorite one on the topic, which is relevant (because it also leveraged ranks), https://elifesciences.org/articles/41690. Of note, we implemented this in SciPy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multiscale_graphcorr.html


I like the idea of this paper, but I am confused about a few things.

1. How is it that we are observing or measuring uncertainty?  Is the idea that somehow we directly have an estimate of uncertainty, without multiple samples.  In our work, we are often faced with multiple samples per subject, eg, we have 50 subjects, each sampled 2 times.  So, we can get an estimate of the variance from those 2 observations.  Is that what you have in mind? If so, why not just use all the observations, rather than use them to estimate uncertainty? This is a fundamental misunderstanding that I have, which makes evaluating the paper quite difficult for me. 
2. There are lots of ways to model uncertainty, only estimating the variance is one option.  The simulations seem to assume this is a good option.  I wonder what happens when this is not a good option, eg, the uncertainty is bimodal.
3. When you write “AUC”, you mean area under the which curve, power? Assuming what null and alternative? And assuming alpha = 0.05?
4. In the figures, you don’t compare to just ignoring the uncertainty, eg, just running HSIC, or MGC, etc.  That makes me wonder.  
5. If we have a distribution, sure, we can use a 2 parameter estimate of the distribution, but we could do other things, eg, a 2-bin histogram, or a k-bin histogram.  I wonder about such options. 
6. I don’t understand the LLM experiment. Did the LLM give you a numerical estimate of its standard deviation? If not, how did you estimate it?
7. I don’t really understand Fig 1, and Fig 2 did not do much for me. I’d rather have more text/pseudocode on the alg.

### Soundness
3

### Presentation
2

### Contribution
3
