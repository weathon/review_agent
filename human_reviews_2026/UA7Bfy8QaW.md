# ML Estimation from Bits

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Estimating statistical parameters from quantized signals has received significant attention in recent years, as recovering information from quantized measurements has numerous applications across signal processing, communications, and data analysis. In this work, we focus on maximum likelihood (ML) estimation of statistical parameters from quantized samples. Directly solving the ML problem is challenging, as the likelihood function involves multiple integrals that are difficult to evaluate.
To address this challenge, we propose an expectation-conditional-maximization (ECM) algorithm under a general distributional framework. Our approach generalizes the quantization model to multi-bit settings and allows the underlying signal to follow any distribution within the normal mean-variance mixture family. By designing suitable surrogate functions, the ECM algorithm ensures that all model parameters can be updated in closed form at each iteration. Leveraging the ECM framework, we provide convergence guarantees, and under specific distributional assumptions, we further derive bounds on the convergence rate and the statistical error. Extensive experiments demonstrate the effectiveness of our method in recovering statistical parameters from quantized data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper, the authors consider maximum likelihood estimation of the parameters $(\mu,\xi,\Sigma)$ of a random vector $x \in \mathbb{R}^d$ following a normal variance-mean mixture model from quantized observations $y_i = Q(x_i)$ where $Q$ is a scalar quantizer and $x_1,…,x_n$ are iid samples of $x$. According to this model

$$x = \mu + z\xi + (z \Sigma)^{1	/2} \epsilon,$$

where $\mu$ is the location parameter, $\xi$ is the skewness parameter, $\Sigma$ is the scattering parameter, $\varepsilon$ is a standard normal random vector, and $z$ is a non-negative random variable with density $p(z)$.

Since evaluating the likelihood function $L$ involves double integration and is not possible analytically, the authors use Jensen’s inequality to construct a surrogate functional that lower bounds $L$ and can be maximized more efficiently by alternating steps of taking expectations and maximizing the current surrogate.

In Theorem 2 they show global convergence of the alternating method to stationary points of L at a linear rate.

### Strengths
+ Paper is easy to read and tackles a relevant problem

### Weaknesses
- Convergence result is not clearly stated
- Numerical evaluation is hard to interpret
- Efficiency of the method for covariance estimation is not analyzed (neither theoretically nor experimentally)

### Questions
Since the paper does not convince me that the ML approach outperforms existing methods for quantized covariance/correlation estimation, I do not recommend it to be accepted yet. In my opinion, the numerical comparison with existing approaches must be clarified to clearly prove gains in computational or sample efficiency. Moreover, the convergence result needs to be revised.

Here a list of questions/issues that should be addressed:

- l. 61: Q is quite loosely defined at the moment. In particular, the statement that Q becomes sign for e=2 is not correct since the present definition would also allow Q taking values -a and b for values in (-\infty,c) and [c,\infty), respectively, where $a,b > 0$ and $c$ can be chosen arbitrarily.

- Some relevant more recent literature has not been mentioned:

[1] Chen, Junren, and Michael K. Ng. "A parameter-free two-bit covariance estimator with improved operator norm error rate." Applied and Computational Harmonic Analysis (2025): 101774.

[2] Dirksen, Sjoerd, and Johannes Maly. "Tuning-free one-bit covariance estimation using data-driven dithering." IEEE Transactions on Information Theory 70.7 (2024): 5228-5247

- l. 109: The works of Chen et al. and of Dirksen et al. are not restricted to Gaussian distributions

- ll. 169+177: irrelevant -> independent ?

- l. 188: independent with -> independent of

- Proposition 1: For which stationary point does the result hold? It cannot hold for all stationary points simultaneously. From the proof, I cannot see that a particular stationary point is picked, so I doubt the correctness of the argument. I did not have time to check the proof in detail though.

- Section 5 is suddenly discussing matrix completion and compressed sensing as applications of the work without specifying how the results can be applied. For matrix completion this is then done in Section 6, for compressed sensing it’s completely missing. I find this structure strongly confusing.

- Section 6: I find the labelling of the plots highly unclear and I cannot really connect the lines to the methods discussed in the text. To make this comparison rigorous and interpretable, please clearly name the methods you compare and use the same labelling in text and plots. The present presentation does not convince me of the value of the method. Furthermore, efficiency in approximation is something that should definitely be evaluated and compared in the covariance estimation setting since existing approaches are quite cheap to compute. 

- Finally, is there any hope to analyze the estimation error of the ML approach in dependence of the number of samples?  Chen et al. and of Dirksen et al. provide rigorous and non-asymptotic error guarantees for their respective estimators, which are essential for reliability of the method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper considers parameter estimation with quantized samples. This builds theoretically using on an ECM algorithm approach developed in a line of other works, now generalizing to the normal mean-variance mixture case.  An approximate ML function is used to approach ML estimation. An ECM algorithm is developed and convergence proofs are derived.  Numerical examples compare to baselines.

### Strengths
The theory is clearly developed and the ECM is shown to linearly converge to a stationary point, similar to prior analysis for the less general cases.  The method adds some generalization over past Gaussian, now with normal mean-variance mixture, adding some additional distribution parameters. The work is shown to reduce to previously developed Gaussian methods.  The authors present examples with matrix completion for a recommendation system which is an interesting real-life application. 

The method might be useful for robustness against Gaussian mismatch, and the paper makes statements about heavy tailed distributions.

### Weaknesses
Overall the paper is strong on theory, and less so on the examples and testing. The examples are relatively benign and limited.  It isn’t clear if a local minimum is possible in the optimization, and what is lost between max-likelihood and using the bounded surrogate function approach. The overall complexity is not well characterized.  The quantization loss is also not clearly considered in the examples. 

The robustness question is interesting, e.g., heavy tailed cases.  It would be useful to explore this.

### Questions
What happens with more than one bit quantization?  Complexity, for example.

How to characterize complexity and compare against other algorithms?

Show also a "full" quantized case to see what is lost?

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
3

### Summary
The paper proposes an expectation/conditonal maximization (ECM) algorithm for estimation of signals following a mean variance mixture model from quantized measurements. Following the common premise for EM algorithms, the E step estimates latent variables based on observations and existing model parameter estimates and the CM step updates the parameters to best match the latent variables and observations. The paper includes convergence results for the model parameters and shows examples in covariance estimation, matrix completion, and compressed sensing.

### Strengths
The paper generalizes existing frameworks by letting the quantization scheme be arbitrary rather than just binary (e.g., sign preservation) and using a broader model for the signals (vs. the usual multivariate Gaussian) and shows agreement between new and existing results in these cases.

### Weaknesses
Since the paper is focused on broadening the model applied to solve existing problems currently addressed with narrow models, it would be good to get a discussion of the benefits afforded by this contribution. The experiments do not seem to need to leverage the broader model, and a comparison with existing approaches is lacking as detailed below.

The presentation is not always fully clear; there are multiple instances of notation used without introduction (detailed at the end of this response). 

Noise does not appear to be considered in the acquisition process.

The numerical comparisons are for the most part focusing on the role of mismatch in the modeling on the performance of estimation, but there is a lack of comparison with previously proposed approaches. Although some competitors are mentioned in page 8, it is not clear if the comparison is against an ECM approach that uses a particular narrower data model or against the approaches of the cited references, which are not EM-based. There is no consideration of the computational cost of the EM approach to contrast against accuracy metrics (e.g., Table 2).

It would be good to have a discussion of applications for the quantized covariance estimation problem.

Line 115: Parameters mu, xi, Sigma have not been defined
Line 131: the Sigma^{-1} norm has not been defined (but one can guess)
Line 168: "theta underlined" is not previously defined
Line 169: “irrelevant of” does not seem to be appropriate wording. Perhaps independent or "does not depend"?
Title: the PDF title does not match the title in the database record.

### Questions
Is there a source for the statement about Netflix movie recommendations in line 275? It may hark back to "The Netflix Prize" from the decade of the 2000's, but it's not clear this is still true; a more generic statement could be made about recommendation systems though. See for example https://dl.acm.org/doi/10.1145/2891406

Can you elaborate on "the constraints in certain prior works" mentioned in line 312? It would better inform the choices made in Section 6.1, particularly because it's not clear that a comparison to existing approaches is being provided.

Given the closeness of the lines in Figure 1, would a wider zoom or a table be more appropriate to convey the results? Or alternatively, is it important to show which algorithm is better when their performances are very similar to one another?

Can you provide motivation or a citation for the use of the surrogate (16)? It is not clear how it connects to the mean variance mixture model being assumed. Perhaps it would be clearer to explicitly write an instantiation of the ECM algorithm of page 4 for each application?

### Soundness
3

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
3

### Summary
This paper proposes the ECM (expectation conditional maximization) algorithm for quantized maximum-likelihood estimation. The authors prove convergence guarantees for the algorithm and test the algorithm on covariance estimation and matrix completion tasks.

### Strengths
1. This paper extends previous works on quantized statistical estimation to multiple bit and a more general normal mean-varaince mixture model.
2. The proposed method demonstrates generality to some extent, which can be applied to covariance estimation and matrix completion, the latter task is very important in recommendation systems.

### Weaknesses
1. It is not clear whether to what extent the normal mean-variance mixture model is useful in machine learning applications. The experiment is only performed on MovieLens 100k data, and the performance is not compared with other machine learning approaches to demonstrate the effectiveness of the maximmum-likelihood estimator.
2. The scalability of the approach is not clear. The MovieLens 1M and 20M benchmark may be more relevant for modern machine learning applications.

### Questions
What are other potential applications of the normal mean-variance mixture model in other machine learning tasks?

### Soundness
3

### Presentation
2

### Contribution
2
