# Predicting Kernel Regression Learning Curves from Only Raw Data Statistics

- Decision: Accept (Poster)
- Scores: 8, 0, 8, 8, 8

## Abstract
We study kernel regression with common rotation-invariant kernels on real datasets including CIFAR-5m, SVHN, and ImageNet.
We give a theoretical framework that predicts learning curves (test risk vs. sample size) from only two measurements: the empirical data covariance matrix and an empirical polynomial decomposition of the target function $f_*$.
The key new idea is an analytical approximation of a kernel’s eigenvalues and eigenfunctions with respect to an anisotropic data distribution.
The eigenfunctions resemble Hermite polynomials of the data, so we call this approximation the \textit{Hermite eigenstructure ansatz} (HEA).
We prove the HEA for Gaussian data, but we find that real image data is often ``Gaussian enough’’ for the HEA to hold well in practice, enabling us to predict learning curves by applying prior results relating kernel eigenstructure to test risk.
Extending beyond kernel regression, we empirically find that MLPs in the feature-learning regime learn Hermite polynomials in the order predicted by the HEA.
Our HEA framework is a proof of concept that an end-to-end theory of learning which maps dataset structure all the way to model performance is possible for nontrivial learning algorithms on real datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a theoretical framework for predicting the learning behavior of kernel ridge regression (KRR) on high-dimensional possible complex datasets using rotation-invariant kernels.
It proposes the Hermite Eigenstructure Ansatz (HEA): the kernel's eigensystem (for rotation-invariant kernels) closely matches a "Hermite eigensystem" derived from multivariate Hermite polynomials in the data space, where the latter depends only on low-order statistics of the data distribution and has an explicit analytical form.
For two data models, wide Gaussian kernel on a Gaussian measure and fast-decaying dot-product kernel on a Gaussian measure, the ansatz is proven to hold asymptotically.
Numerical simulations suggest that the ansatz also holds approximately for real datasets like MNIST and CIFAR-10.
Moreover, MLPs in the feature learning regime are shown to learn Hermite polynomials in the order predicted by HEA (lower-degree first), suggesting broader applicability.

### Strengths
* The paper is well-written and clearly structured, making it easy to follow the main ideas.
* It is novel to propose the Hermite Eigenstructure Ansatz (HEA) to predict the learning curves of KRR using only low-order statistics of the data distribution. It would help to better understand the generalization behavior of kernel methods and intrinsic structure of data.
* The theoretical analysis seem to be solid for the two specific data models.
* Extensive numerical experiments are provided to validate the proposed theory, including synthetic data and real datasets (MNIST and CIFAR-10). The approximation seems to hold well, justifying the practical relevance of the theory.

### Weaknesses
* Most of the theory in this paper is heuristic, relying on the proposed Hermite Eigenstructure Ansatz (HEA) without rigorous proof except for two specific data models.
In addition, it would be helpful to bound the difference between the prediction errors (the original model and the Hermite one) in terms of the approximation error of HEA, giving non-asymptotic guarantees.

* The theoretical results are limited to rotation-invariant kernels and Gaussian data distributions, which may not generalize to more complex real-world scenarios.

### Questions
1. What are the connections between the HEA and the Gaussian equivalence assumption (GEA) used in prior works on learning curves of kernel methods? 
2. What are the datasets that HEA fails to approximate well? Can you provide some more insights into the limitations of HEA?
3. How will HEA be useful for practical applications, such as kernel selection or hyperparameter tuning?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper introduces the "hermite eigensystem" as an approximate formula for the eigenvalues and eigenfunctions of dot-product kernels, expressed in terms of the data covariance matrix. The authors present two limiting cases where their approximation holds exactly, then show that the learning curves obtained by applying the standard theory of kernel ridge regression under their approximation captureswell the empirical learning curves of kernel ridge regression on real data.

### Strengths
- The paper is well written, and the results are clearly and logically organised.  
- Developing an end-to-end theory of learning based directly on data statistics is a highly valuable goal.  
- The experimental validation appears convincing and supports the proposed theoretical framework.

### Weaknesses
1. Theoretical learning curves for kernel regression have been established in prior work (e.g., *Optimal Rates for the Regularized Least-Squares Algorithm* by Caponnetto and De Vito). It is unclear what new insights an approximate, simplified formula based on Gaussian assumptions contributes to this existing body of knowledge.  
2. Gaussian Equivalence Principles (e.g., *The Gaussian Equivalence of Generative Models for Learning with Shallow Neural Networks* by Goldt et al.) already characterise when Gaussian approximations like the one made by the authors are valid or not. The paper does not sufficiently discuss its relation to these frameworks.  
3. The proposed approximation appears *uncontrolled*, as the limits of its validity are neither explored nor commented upon, leaving uncertainty about its general applicability.

### Questions
Presumably, if the data are actually assumed to come from the Gaussian distribution with covariance matrix $\Sigma$, then the eigenfunctions should be multivariate Hermite polynomials. Have the authors tried to relate the kernel eigenvalues to the Taylor coefficients of the Kernel expansion into powers of $\mathbf{x}\cdot\mathbf{y}$ exactly? Is capturing the right eigenvalues even important for the prediction of learning curves, or is it really the order of the eigenfunctions that matters?

### Soundness
4

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Authors study kernel ridge regression on anisotropic data for rotation-invariant kernels.
They introduce a Hermite eigenstructure ansatz (HEA), that allows to approximately compute the eigendecomposition of the kernel function.
The main finding is that the covariance of the data and the coefficients of the target function in the basis of this eigendecomposition are enough to predict the test error of the kernel ridge regression.
The result holds when the data is 'Gaussian enough', which is empirically verified for certain real datasets. 
In particular, the width of the kernel needs to be sufficiently large, as well as the effective dimension of the data. 
The cases when the predicted test errors are further from the real values, are thoroughly studied.
Theoretical results prove that the predictions hold when data is Gaussian under two limiting regimes.

One of the technical issue that the authors had to deal with is how to obtain the coefficients of the expansion of the target function in the HEA of the kernel. They suggest to iteratively project the target function on the top eigenfunctions and use Gram-Schmidt orthogonalization to remove correlations between the Hermite polynomials that empirically works well even when the data is non-Gaussian, but the target is sufficiently smooth.

### Strengths
The paper is very well-written and constitutes a timely contribution as the role of the data in learning is extremely important.

Hermite eigenstructure ansatz is a clean and theoretically motivated tool to understand KRR.
The authors present a thorough analysis of when the proposed HEA method holds, both on synthetic and real datasets. 
They empirically study the effects of data dimension, target function, and various kernels.
Interestingly, the authors also study the regime 'beyond kernel ridge regression' and find that their predictions hold in that scenario too, to some extent.

For the case of Gaussian data, two theorems regarding different scaling regimes (very wide kernel and very fast coefficient decay) are present. 
This work is likely to motivate further research to understand rigorously where the limits of the use of the Gaussianity assumption are, possibly beyond kernel regimes.

### Weaknesses
I don't find serious weaknesses in the methodology, experiments or theoretical results of the paper, although I did not check the proofs in detail.

Extended related work discussion would help the reader to position current work:
1. In (Refinetti et al., 2023), the authors claim that the neural networks, trained on 'Gaussian versions' of CIFAR-10 (with the same mean and covariance), perform worse than on the real dataset. This can be interpreted that CIFAR-10 is not 'Gaussian enough'. Perhaps whether the data is 'Gaussian enough' depends on the class of learning algorithms, with KRR not being able 'to see further' than the covariance of the data. Is such interpretation correct?
2. There is a line of work that studies 'staircase property' of learning (e.g., Abbe et al., 2021). Is there a relation between this property and the results in Figure 4? Why both Figure 3 and Figure 4 only use monomials as a target and what would happen when polynomials are used instead?

Also, the notation of $h_i$ (eg in line 1135) can be confused with $h_{\mathbf{\alpha}}$.

Refinetti, Maria, Alessandro Ingrosso, and Sebastian Goldt. "Neural networks trained with sgd learn distributions of increasing complexity." International Conference on Machine Learning. PMLR, 2023.

Abbe, Emmanuel, et al. "The staircase property: How hierarchical structure can guide deep learning." Advances in Neural Information Processing Systems 34 (2021): 26989-27002.

### Questions
See the section above. Furthermore:
1. Throughout the text, both 'population data covariance, $\Sigma$' (eg line 47, Figure 1) and 'empirical data covariance, $\hat \Sigma$' (eg line 13, Figure 3) are used. From the description in Section 4, it seems that $\Sigma$ is the main quantity of study. Could the authors clarify to what extent they expect these results to hold when having access only to $\hat \Sigma$?  
2. Line 1643 and Figure 18: what is $\gamma$? Should it be $\zeta$?
3. Figure 5: in the legend, shouldn't the labels for predicted and empirical be swapped?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces the hermite eigenstructure ansatz (HEA), which gives a closed form expression for the eigensystem of rotation invariant kernels. Using the eigenframework, this gives the learning curvevs for KRR on real image datasets, which are empirically shown to approximate the learning curves well using only the covariance matrix and the hermite decomposition of the target function. They also show that HEA holds for gaussian data and two limiting cases of the kernel function. Finally, the papers empirically shows that MLPs learn hermite polynomials on real datasets in the order predicted by HEA.

### Strengths
The hermite eigenstructure ansatz solves one of the main difficulties of using the eigenframework predictions in predicting the learning curves of KRR on real datasets: estimating the eigenvalues and eigenfunctions of the kernel under the given data distribution. This a very important contribution and expands the applicability of this line of work.


Further, the paper provides extensive empirical evidence that the approximate closed form expression for eigenvalues and eigenfunctions can be used to very accurately predict the learning curves of KRR on a number of real image dataset and various synthetic setups, which is needed given the harndess of verfying the for real datasets HEA. 


Finally, the paper clearly demonstrates the usefulness of insights from HEA, which they use to predict how MLPs learn functions.


Overall, I think that this is a very strong contribution that is presented clearly and concisely and should get accepted.

### Weaknesses
1. The paper is unnecessarily and sometimes confusingly written in an underdefined and informal way. For example, the use of undefined approximate sign $\approx$, which is crucial to the Hermite eigenstructure ansatz lines 251-260, is at time confusing. Since we are interested in predicting the learning curves, it is quite easy to define this approximate equivalence between two eigenstructures as having learning curves differing by at most something small. It is also unclear whatt does the final prediction mismatch depend on after using a number of these approximations. I think the whole framework would greatly benefit if some care was taken to quantify or at least characterize the dependence of the prediction error of using the hermite eigenstructure ansatz.
2. As explained in lines 968-969, and if I understand correctly, the hermite ansatz framework depends on eigenlearning framework to map predictions of eigenstructure to predictions of learning curves. Since the hermite ansatz is not formal (and not quantitative, see #1), the applicability of hermite eigenstructure ansatz also depends on the applicability of the eigenframework to the case considered (so it also depends on Gaussian Universality Ansatz). This way, the error of the final prediction aggregates the error of the hermite eigenstructure prediction and the eigenframework prediction. So it’s a bit unclear whether the conditions for success (Section 4.2) are sufficient for both of these errors to be small. I feel like this “hidden” dependence on the eigenframework should be more transparently discussed in the main body. This is especially given some recent work questioning whether the eigenframework actually applies in the case of NTK.
3. The paper only shows that HEA holds for the Gaussian data with a Gaussian kernel or other fast-decaying dot product kernels in a certain limit of the parameters of those kernels. It’s unclear what is the interpretation of this limit (e.g. taking the width of the Gaussian kernel to infinity). This section would also be much more clear if the limit of is more quantitatively defined.

### Questions
1. Can the approximate relationship in the definition of HEA and elsewhere be formalized in terms of the error of the prediction of the learning curves? 
2. Is the HEA expected to hold whenever the Gaussian Universality Ansatz holds? How the two interact?
3. What is the interpretation of the infinite width limit of Gaussian kernel?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a principled framework for predicting learning curves of kernel ridge regression (KRR) using only empirical data statistics, specifically the data covariance matrix and the Hermite polynomial decomposition of the target function.  
The central theoretical idea is the Hermite Eigenstructure Ansatz (HEA), which posits that for rotation-invariant kernels on approximately Gaussian datasets, the kernel’s eigenfunctions coincide with multivariate Hermite polynomials of the data, and the corresponding eigenvalues factorize as simple monomials in the data covariance eigenvalues.



This work offers a conceptually novel and practically valuable contribution: predicting learning curves from minimal dataset statistics without explicit kernel computation.  
It bridges data geometry, kernel methods, and neural tangent kernel theory, showing a concrete mapping from data covariance to performance prediction.  
The observed similarity between MLP learning order and HEA predictions further strengthens its relevance.

### Strengths
- Introduces the Hermite Eigenstructure Ansatz, a compact analytical surrogate for kernel eigenfunctions and eigenvalues grounded in probabilistic data geometry.  

- Combines proofs, ablations, and large-scale experiments (CIFAR-5m, SVHN, ImageNet-32) with excellent agreement between theory and observation.  

- The conceptual pipeline links data covariance to feature-space statistics and test error.  

- Provides a data-to-performance theory that is both interpretable and predictive.  

- Clearly delineates success and failure regimes (e.g., Figures 12–16).

### Weaknesses
- No analytical error bounds are provided for the eigenvalue approximation.  

- The Gram–Schmidt coefficient estimation step can be computationally heavy for large datasets.

- Minor comment: Dependence on Gaussianity: the HEA requires data to be "Gaussian enough"; it fails for structured or discrete datasets (MNIST, tabular). Extending it beyond this regime would strengthen the framework, but would be very difficult though (but does not undermine the results though).

### Questions
- Could the Gaussian assumption be relaxed to a mixture-of-Gaussians or empirical-moment approach to handle non-Gaussian data?  


- Can the authors provide quantitative error bounds between predicted and empirical eigenvalues as a function of effective dimension or kernel width?

### Soundness
3

### Presentation
3

### Contribution
3
