# Variance-Reducing Couplings for Random Features

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6, 8

## Abstract
Random features (RFs) are a popular technique to scale up kernel methods in machine learning, replacing exact kernel evaluations with stochastic Monte Carlo estimates. They underpin models as diverse as efficient transformers (by approximating attention) to sparse spectrum Gaussian processes (by approximating the covariance function). Efficiency can be further improved by speeding up the convergence of these estimates: a variance reduction problem. We tackle this through the unifying lens of optimal transport, finding couplings to improve RFs defined on both Euclidean and discrete input spaces. They enjoy theoretical guarantees and sometimes provide strong downstream gains, including for scalable inference on graphs. We reach surprising conclusions about the benefits and limitations of variance reduction as a paradigm, showing that other properties of the coupling should be optimised for attention estimation in efficient transformers.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes an efficient construction of random features for kernel methods. Motivated by the optimal transport, it finds couplings to reduce the variance of random features defined on both Euclidean and discrete input spaces. Numerical experiments are conducted and theoretical properties are analyzed.

### Strengths
1. The connection between variance reduction of RFs and OT is clearly presented.
2. RFs for different settings including RFFs, RLFs and GRFs are deeply studied. 
3. A through simulation study has been conducted and some interesting phenomenons show up in the numerical experiments.

### Weaknesses
Please see following questions.

### Questions
1. In Lemma 3.1, the forms of c_{RFF} and c_{RLF} are very complicated and the solution \mu^* seem to depend on x and y. In Theorem 3.2 with m=2, the solution seems to be independent of x and y. Is it the case for m=2? Or the solution will always be irrelevant to x and y no matter how large m is? In addition, it will be better to give more explanation about the statement ``provided z is sufficiently small". For example, how small z is w.r.t. the order? How to check if z is small enough in practice? 

2. In Section 3.3, what is the definition of RMSE? Is it the point-wise RMSE or the Frobenius norm error for the gram matrix approximation?  What is the choice of d and how RMSEs change with d empirically? 

3. Though the paper provides that Theorem that the pairwise norm-coupled RFs is guaranteed to be lower than orthogonal RFs with independent norms, it will be better to provide an explicit error upper bound for the variance, which will be easier to see the dependence of the error on different parameters. 

4. It is interesting to see that the variance reduction alone cannot guarantee an improvement in the downstream tasks. I am wondering is it the case for most datasets considered in Table 1? If so, the application of this RFs may be somehow limited. 

5. In page 8, the paper proposes to ``move Edirs inside the square" to make the computation feasible. It will be better to discuss the effect of such adjustment.  By moving the expectation, the objective function indeed changes and solving the adjusted objective function does not necessarily reduce the variance? 

6. In the numerical experiments w.r.t. RGFs, the proposed method performs the best also in the downstream tasks for different datasets, which shows a different performance from RFFs and RLFs.  Is there any intuition behind this?

### Soundness
3

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
The paper considers the problem of finding a suitable joint measure for random features in a kernel evaluation by applying a variance reduction principle. The application of such a principle in general results in a coupling of the features, expressed, for example, by a constraint on their modulus in the case of features in the Euclidean case. The principle is given in terms of an optimal transport formulation and its application is exemplified by a number of experiments on both Euclidean kernels and graph kernels. The authors show that if on the one hand, a smaller kernel variance is obtained as expected, other quantities might benefit from different strategies, encouraging a rethinking of the relevance of kernel variance reduction.

### Strengths
The paper aims to solve the problem of minimizing the variance of kernel estimators by random features by mapping the problem in an optimal transportation (OT) problem. By using OT concepts, (sub-optimal but interesting) coupling strategies are identified in both the Euclidean case and in a discrete case, namely graph kernels. The paper links therefore two ideas (kernel random features and optimal transport) that are very popular in the machine learning community, with an interesting digression on graph random features and graph kernels. Moreover, I appreciate the rich numerical exploration of the results, that highlighted the benefits and limitations of the method.

### Weaknesses
My main concern is that the proposed OT formulation does not obviously lead to a simple, applicable rule, as shown by the authors themselves. Even in the random Fourier features case, under the simplifying pairwise coupling assumption, guarantees can be proven only for close enough arguments of the kernel function to estimate. The examples discussed and the appendices show that the application of the OT ideas might be cumbersome, not straightforward, and require a certain ingenuity to perform a number of simplifications and approximations (whose effectiveness is only shown *a posteriori*): this comes with no surprise as solving an OT problem is generically hard. More importantly, if the quantities of interest are nonlinear functions of the kernel, counterintuitive results might be found, e.g., *maximizing* the variance might be beneficial. It appears then unclear how to decide *a priori* what the best coupling strategy is. Even if, in principle, one might think to generalize the minimization problem in Eq 2 (so that the functional $\mathcal I(\mu)$ depends on the variance of other target quantities), as the authors themselves explain, the possible complication of such generalization, and its feasibility, seem hard to anticipate.

### Questions
Here below some comments and observations:

- After Eq 1, it is not clear if *each* $\phi$ has to be intended depending on $m$ random frequencies (also, $\\{\boldsymbol\omega\\}$ should be replaced with $\\{\boldsymbol\omega_i\\}$: it would be sufficient, for example, to write down $\phi(\boldsymbol x;\boldsymbol\omega_{1:m})$ in place of $\phi(\boldsymbol x)$. At line 048 I am not sure if Eq 1 is a "MonteCarlo estimate" of $k$, as in Eq 1 there is no empirical sampling. 
- I might being missing something trivial, but why $\mathcal I(\mu)=\mathbb E_\mu[(\phi^\top\phi)^2]$ and not $\mathcal I(\mu)=\mathbb E_\mu[(\phi^\top\phi)^2]-\mathbb E_\mu[\phi^\top\phi]^2$? I suppose that, given the constraint on the marginals, the second term is irrelevant if $\phi(\boldsymbol x;\boldsymbol\omega_{1:m})^\top\phi(\boldsymbol y;\boldsymbol\omega_{1:m})=\sum_{i=1}^m\psi_i(\boldsymbol x;\boldsymbol\omega_i)^\top\psi_i(\boldsymbol y;\boldsymbol\omega_i)$ for some $\psi_i$, but this might be not true in general. Unless I am wrong, would the more general case lead to a much more complicated treatment? The treatment of RFF and RLF seems to be strongly case-specific, although relevant.
- The discussion of the application to GRFs leads to encouraging results. Do the authors have an intuition on why this is the case? Are such positive performances retained independently from the level of the sparsity of the graph? In general, I was wondering how much the "sparsity" of the underlying space where the kernel lives affects the effectiveness of the method.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper explores variance reduction techniques for random features (RFs) in kernel methods to improve efficiency in large-scale machine learning tasks. By framing variance reduction as an optimal transport (OT) problem, the authors propose couplings that enhance the convergence of RFs on both Euclidean and discrete input spaces. They introduce novel coupling techniques for various RF classes, including Random Fourier Features (RFFs), Random Laplace Features (RLFs), and Graph Random Features (GRFs). The paper highlights the effectiveness of the proposed methods in reducing variance and improving downstream performance in certain cases, although it also discusses limitations.

### Strengths
1. **Innovative Use of Optimal Transport**: The paper employs OT as a novel framework to address the variance reduction problem in RFs, which is a creative approach that ties together theoretical insights with practical application.
2. **Comprehensive Coverage**: It addresses both Euclidean and discrete input spaces, providing a unified strategy applicable across different domains.
3. **Theoretical Guarantees and Empirical Validation**: This paper offers theoretical guarantees and empirical validations, enhancing the understanding of variance reduction in RF performance.

### Weaknesses
1. The definition and relationship between variance reduction in kernel estimation and optimal transport should be more detailed and introduced for non-specialists to understand directly.
2. The computation and theorem (Theorem 3.2) is only for $m=2$, and the authors apply the copula tool as numerical OT solvers for multi-marginal OT.
3. Despite variance reduction, the downstream impact on model performance can be inconsistent, especially for transformers.
4. More comparisons between this paper and [1] and [2] should be clarified, since some of the idea and technical proof skill is from these paper.


[1]. *General Graph Random Features*. Nips 2023.
[2]. *Simplex Random Features*. ICML 2023.

### Questions
1. What are the computational implications of implementing these optimal transport techniques at scale?
2. Can the theoretical findings on variance reduction be generalized to other kernel methods or models?

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
4

### Summary
The paper presents a novel approach for improving kernel estimation methods via random features. By utilising optimal transport, they tackle the specific cases of random fourier features, random laplace features, and graph random features. They demonstrate this technique on a range of real world datasets. And while some practical advantages are found, it is also noted that reducing the kernel variance does not necessarily improve the quality of the forecasts.

### Strengths
The manuscript is well presented, structured, and clearly written.

There is a high degree of novelty in the application of optimal transport in determining the feature couplings. 

The proposed method is tested on a good selection of real world datasets.

### Weaknesses
I have some concerns regarding some of the experimental results presented in the manuscript:

Regarding Table 1 - Running only d features for RFF strikes me as a rather low number to choose, especially for these modestly sized datasets. It would be helpful to clarify how the results impacted when using a larger numbers of features. It would be great to see something analogous to the Figure 1 of the Yu et al 'Orthogonal Random Features' paper which depicts 1 < d/D < 10. 

While I can appreciate that normalisation of RMSE's to 1.0 makes for a clean reference point, this also obfuscates the absolute performance from the reader (which is significant for the purposes of reproducibility, and comparisons with other works).

Regarding Table 3 - In addition to, or instead of, providing the KL results, I think the predictive performance (NLL or NLPD) would be preferable. I believe this would provide a better proxy for the model evidence. My concern with the KL divergence is that it does not heavily penalise models which are overconfident.

### Questions
Regarding the choice of metric, what are the benefits of using KL over predictive NLL (or NLPD)? I noticed that on page 22, it is mentioned that the predictive NLL is computed, so perhaps I missed where these results are presented.

Are there applications where the kernel reconstruction is the primary focus, rather than the derived confidence intervals? If so it might be beneficial to emphasise these to the reader.

One further small suggestion - at present the authors describe as "surprising" the result that kernel accuracy improvement often does not lead to downstream test improvements. On the contrary, I think there have been several reports of this phenomenon in the literature. For example, from Liu et al 2004.11154 "We find that ORF and SSF yield the best approximation quality. Despite that most algorithms achieve different approximation errors, there is no significant difference in the test accuracy."

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a well-known Monte Carlo method for estimating the value of kernel functions using random feature embeddings. In particular, it proposes a method to reduce the variance of the estimate by coupling individual random features. This task is framed as a general optimal transport problem and is further discussed in two different scenarios: first random Fourier and Laplace features, and second features obtained by a random walk on graphs, corresponding to graph kernels. In the first case, exact results for two features are obtained and some extensions to more features is discussed. Moreover a search algorithm over a parameterized family of couplings is proposed. In the second case, a simplified coupling of the length of the random walks is discussed.

Numerical experiments are presented for each case that show the effectiveness of these algorithms.

### Strengths
I find the general idea of the paper fascinating, namely formulating the variance reduction task as optimal transport. This approach can also be applied to similar Monte Carlo methods in future studies. It also allows for the application of various techniques in OT for studying and improving kernel estimation methods.

### Weaknesses
Although the paper is well-written and easy to understand, the presentation of the paper can be improved. Much of the technical details that are summarized in the main body cannot be easily understood without referring to the appendix. There are confusing notations and definitions, especially for the case of graph kernels. I will mention some of them in the next part. 

I also have a more specific questions/concerns about the justification of this framework which is explained in the next part.

I am willing to raise my score if my concerns are addressed.

### Questions
My main concern is about the justification and also computational aspects of the proposed framework. As the authors explain, the main limitation of this work is that variance is reduced at a specific pair of points, which also leads to some limitations in the performance of downstream tasks. The condition of fixed marginals in the OT framework actually guarantees an unbiased kernel as a process, i.e. for every pair of points. Is not it more reasonable to minimize the variation of the entire process rather than a specific point?

The pointwise view also leads to a potential computational limitation. If I understood correctly, in practice one needs to generate an independent set of coupled features for each pair of points, which requires to run the variance reduction method once for each pair. This leads to a huge computational burden in a large-scale problem.

More specific questions regarding the presentation:

1- I am not familiar to the notion of negative monotone coupling and for me (7) does not make sense. Can you provide more explanation?
2- If I am correct, theorem 3.2. only works for sufficiently small z, this should be explained in the introduction before this equation to avoid confusion.
3-In line 290, what is the definition of \hat{a}_i?
4-In line 387, does "walker's direction" mean the random walk conditioned on its length?
5-In Theorem 4.1, I do not understand the definition of q^{i,j}. Also q^{i} is not exactly defined.

### Soundness
3

### Presentation
3

### Contribution
3
