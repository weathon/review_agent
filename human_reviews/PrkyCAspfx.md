# Bayesian Uncertainty Quantification Meets Topology

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Computational topology recently started to emerge as an overarching paradigm for characterising the ‘shape’ of high-dimensional data, leading to powerful algorithms in (un)supervised representation learning. While capable of capturing prominent features at multiple scales, topological methods cannot readily quantify the uncertainty of their respective descriptors. We develop a novel approach that bridges this gap, making it possible to employ topology-based loss functions to perform parameter estimation with Bayesian uncertainty quantification. Our method affords easy integration into topological machine learning algorithms. We demonstrate its efficacy for parameter estimation in different simulation settings.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes to do posterior inference when using topological features (e.g. representing point cloud data with their persistence diagrams). The problem in doing so is how to obtain a valid notion of posterior, given that the likelihood function is typically unknown and only available through simulation.

The authors propose a notion of a posterior in this setting and prove that it arises as a solution to a generalized variational inference problem. In particular, they connect the simulation data to the observed (real) data through a "loss" function which computes the 2-Wasserstein distance between their respective persistence diagrams. 

Experiments show that the proposed method is accurate for various scientific applications, such as simulation-based inverse problems.

### Strengths
I believe imbuing uncertainty in computational topology/geometry is a great thing to strive for. This paper provides a way to do so.

### Weaknesses
My main issue with this paper is several-fold.

First, the experiments are lacking baselines. The authors provide two kinds of data: point clouds and greyscale images. By themselves, they are independent of computational topology/geometry and Bayesian inference, and thus, there must be some recent non-topological, non-Bayesian baselines out there that the authors can compare against. (I have to admit that I'm not well-versed in this particular area.) Moreover, the authors should discuss and compare the cost of their method to those baselines. In its current form, it is very hard to judge the soundness of the results.

Second, one of the main selling points of Bayesian inference is the uncertainty associated with the estimate. However, the authors don't seem to explore this---they only show accuracy. Since the form of the posterior considered in this paper is rather non-standard, I think it is important to empirically study its uncertainty calibration. Moreover, I do think uncertainty can be very useful in many simulation tasks, so I urge the authors to try to find applications that cannot be done without uncertainty, in the settings considered in this paper. (E.g., uncertainty in Bayesian neural networks opens up avenues in tasks such as weight pruning, sequential decision making, etc.) This will boost the strength of the paper significantly.

Third, I think the paper's presentation needs to be improved. Sec. 2, for example, is quite dense and might put non-topology people off. The figures can also be improved, e.g. Fig. 1 is very unclear, Fig. 2 is hard to see (in print, at least), and Fig. 3's line width is too thin.


## Minor stuffs

After reading the paper, I'm not sure how to implement the proposed method---an algorithm about the whole method is much preferable to Alg. 1 which only shows a standard importance sampling algorithm.

### Questions
I have doubts about Prop. 3.1. In your proof, your strategy is to show that (5) arises in the KL-divergence with $q$, and hence you can conclude that (5) is the solution to the variational problem (8).

However, one of the standard ways of deriving the ELBO (i.e. the objective of the form of (8)) is via the KL-divergence itself. So, I can define any other arbitrary distribution in the form of (5) and can prove that it is a "valid posterior" by following your proof idea. So, to me Prop. 3.1 seems to suffer from a chicken-and-egg problem, and thus I'm not sure about its validity. 

Can the authors please clarify this?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes to use the recently introduced generalized Bayesian posteriors for estimating the parameters corresponding to the topological features of a dataset (e.g. radii of a torus) and computing uncertainties associated with them. The framework of generalized Bayesian posterior uses a distribution over parameters together with a generic loss function and provides theoretical guarantees for the existence of appropriate posterior beliefs about the parameter. In this paper, since building likelihood models that connect the parameters directly to observations seems to be a challenging task the authors propose to use an auxiliary generative model that connects the parameters to a variable $x$ and define the loss based on the similarity of the sampled (or simulated) $x$ to the observed data $y$. In this model, the (generalized) posterior over the parameter can be obtained by marginalizing out the variable $x$. The authors use multiple loss functions including Wasserstein and Hausdorff losses and provide importance sampling and MCMC algorithms for the inference of the parameters. Results are shown on synthetic datasets in the context of swarm behavior, fluid dynamics, percolation, and inference of radii of a sphere or torus.

### Strengths
The paper takes a unique perspective in connecting topological data analysis (TDA) to uncertainty estimation. Generative modeling has been rarely applied to TDA both for computational reasons and traceability of the likelihood and prior terms. However, often times we tend to infer topological parameters in TDA based on a limited sample size, highlighting the importance of proper uncertainty quantification over the parameters of persistence diagrams.

The results are shown on multiple simulations, showing the versatility of the proposed approach in various settings.

The recently introduced generalized Bayesian posteriors seem to be powerful tools for data analysis in applied settings but I haven't seen many examples of them being used in real-world scenarios. Proper application of these techniques can bring value to the applied research communities and uncover scientific phenomena. This work provides an instance of such applications.

### Weaknesses
I'm a bit lost about the motivation of the paper. Although the authors motivate the method by quantifying uncertainty in applications where building a generative model is intractable, almost every application shown in the paper follows a clear generative process.

The presented applications are very unclear. Can the authors elaborate on what the $x$ is in each setting and what prevents us from running a gradient-based algorithm (such as a neural network) to solve the inverse problem and estimate the associated $\theta$? The main argument of the paper is that for these applications we can't simply solve the inverse problem using gradient-based methods but it's not clear to me what's the main obstacle. This has caused some confusion for me about applying generalized Bayesian posteriors in this context. To get error bounds on the parameters a variety of methods are introduced. For neural networks, one can use conformal prediction strategies or simply apply other frequentist techniques such as Bootstrap.

The experimental part of the paper is weak in its current form. No comparisons are presented against alternatives. Perhaps the presented method is not the only way of performing these types of analyses. A discussion of the alternative approaches should be included in the introduction and used for comparisons in the results section.

Generalized Bayesian posteriors existed before, the extension presented in the paper is rather elementary. In addition, the losses used in the paper all existed before. Furthermore, the inference algorithms presented in the paper are standard algorithms. The contributions seem to lie in bringing the generalized Bayesian posteriors to the TDA world and showing its application in the presented settings. Given that the results and comparisons are relatively weak and the motivation is unclear I'm leading towards a negative rating for this paper.

### Questions
How does the method scale with the number of dimensions (both for $\theta$ and $y$)?

What are alternative strategies to estimate the parameter $\theta$? Can you include comparison tables to give the reader a sense of what circumstances are better suited for the presented method?

What's the run time of the algorithm (in terms of wall clock time) and how does it compare to the alternative methods? MCMC and importance sampling both take a long time to converge (compared to optimization-based inference strategies). What are the conditions in which one should use this method as opposed to alternatives?

Can you include a clearer description of the generative model, the prior model for $\theta$, the likelihood model for $x$, and the loss for each application presented in the results section? Some of the technical descriptions of the datasets can be moved to the supplementary to open up more space for this information as well as comparisons.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The contribution of this paper is to use topological losses in the comparison based posterior approach

### Strengths
The paper is largely well-written except from a couple of things i mention in the weakness.

### Weaknesses
The contribution of this paper is to use topological losses in the comparison based posterior approach of Schmon et al 2021. This is plug-n-play, and I am not sure if this is novel enough. There are a few reasons to reject this paper:

1)	Novelty --- Like I said the paper is simply proposing to plug in topological losses into a sampling loss function. The paper discusses some experiments to show its importance e.g. Fig 2 is interesting. 
2)	Limited empirical results – The empirical results are all based around the fact that plugging in topological loss is better than plugging in the geometrical loss. This is useful to know in general. These results are not suprising, better losses would give better estimations. There are only a few empirical settings explored. 

Minor: There a re a couple of places where the writing could be improved. I would suggest the authors make their contributions explicit in the intro section. The tangentially relevant material about sampling from the loss function could be expanded but moved from to the appendix. I would suggest the authors expand on their contributions to make the paper stronger. 

The authors mention that pseudo-marginal MCMC algorithms can sample from (6) even when p(x|\theta) may not support automatic differentiation. Can they please expand on this? It is not clear how this could happen? I understand this could be available in Andrieu & Roberts 2009 but it would be nice to expand on it in this paper itself for completeness. 

There are a few typos that require a bit more thorough reading. E.g. Eq (8) it should be p(\theta) instead of \pi(\theta). Similarly Eq (12) there seems to be a typo. Typos in English make the paper hard to read but typos in equations just break the paper.

### Questions
please see the weakness section. My only concern is the paper does too little.

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor
