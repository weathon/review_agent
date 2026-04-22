# Principal component analysis for very heavy-tailed data

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Principal component analysis (PCA) is a ubiquitous tool for dimensionality reduction and exploratory data analysis. However, most theoretical and empirical studies implicitly assume that noise is light-tailed. When data are corrupted by heavy-tailed noise, as is increasingly common (e.g. in omics or brain connectivity data), standard PCA techniques can fail dramatically. While recent work in robust statistics has addressed this problem in certain contexts, many existing methods remain sensitive to extreme outliers, performing poorly under truly heavy-tailed distributions. Furthermore, many of the methods which have been designed for heavy-tailed distributions do not scale well to large data sizes. In this work, we propose a novel algorithm for PCA that is designed for extremely heavy-tailed noise and which is computable for even very large data matrices. Our approach is designed to reduce sensitivity to such deviations while recovering informative low-rank structure. In the case of very heavy-tailed data with a large number of observations, we demonstrate significant improvements over classical PCA and existing robust PCA variants.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies the problem of performing principal component analysis (PCA) under extremely heavy-tailed noise, where classical PCA and many existing robust PCA approaches fail. The authors observe that for heavy-tailed data, the true principal component may lie in the span of several leading sample eigenvectors rather than aligning with the top one. Based on this observation, they propose a heuristic algorithm that repeatedly subsamples columns of the data matrix, computes leading eigenvectors of each subsample, and aggregates these subspaces to recover the principal direction. The method incorporates a weighted sampling strategy to avoid repeatedly selecting outlier-dominated columns. Empirically, the approach demonstrates improved robustness over classical PCA and several robust PCA variants on synthetic data and two biological datasets. The paper claims scalability and practical advantages, especially in high-dimensional scenarios with heavy-tailed noise.

### Strengths
- The paper addresses an important and challenging problem: PCA for extremely heavy-tailed data, where standard methods are known to break down.
- The proposed method is simple, scalable, and easy to implement using basic linear algebra routines, making it potentially useful for large-scale applications.
- The authors provide extensive experimental evaluation on synthetic data and two real-world datasets, demonstrating that the approach can achieve improved empirical robustness compared to many baselines.
- The observation that the signal may lie in the span of multiple sample principal components under heavy-tailed noise is interesting and worth further theoretical investigation.

### Weaknesses
The primary concern is that the algorithm is entirely heuristic and lacks any theoretical guarantees. While heuristics are valuable in practice, the paper does not provide sufficient justification for the proposed approach beyond empirical observation. In particular:

- The method does not come with formal guarantees regarding recovery accuracy, convergence, or robustness, unlike prior work in robust PCA and heavy-tailed estimation.
- The motivation relies heavily on a qualitative empirical observation, but no theoretical explanation or analysis is offered to support the key claim.
- The experiments, although extensive, are not sufficiently diverse to fully establish the reliability of the heuristic. For such a method, more varied real-world benchmarks and stronger empirical gains are necessary to justify its contribution.
- In the absence of theory, the paper risks lacking generality; it remains unclear under which regimes or distributional assumptions the algorithm can be expected to perform well.

### Questions
- Can the authors provide theoretical insights, even partial, into why aggregating PCA subspaces from subsampled data approximates the true principal direction under heavy-tailed noise?
- How sensitive is the algorithm to the choice of hyperparameters in practice, particularly for datasets with different scales and tail behaviors?
- Would the method still hold up if evaluated on a broader set of real-world datasets beyond transcriptomics and neural connectivity data?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a simple and scalable method for performing principal component analysis when data exhibit heavy-tailed noise, a setting where classical PCA and many robust variants often fail because the sample covariance is dominated by a few extreme observations. The proposed approach repeatedly draws random subsamples of the data, computes the top-$P$ principal components for each subsample, and aggregates the resulting subspaces by averaging their projection matrices. The final estimate of the leading principal direction is obtained as the top eigenvector of this aggregated projection matrix, which stabilizes the estimate by diluting the influence of heavy-tailed outliers across many subsamples. The method uses only standard linear algebra operations and empirically outperforms classical PCA, geometric-median PCA, and convex robust PCA on synthetic and biological datasets, particularly in regimes with infinite-variance noise.

### Strengths
1.  Introduces a simple yet effective geometric aggregation approach for PCA under heavy-tailed noise, based on repeated subsampling and subspace averaging. The method is conceptually simple yet addresses a challenging regime (including infinite-variance data) rarely handled by existing algorithms.

2. Demonstrates substantial gains over classical PCA, Minsker’s geometric-median PCA, and convex robust PCA across synthetic and real-world datasets, particularly under extreme heavy-tailed noise.

3. Achieves scalability by relying only on standard linear-algebraic primitives (SVD, eigen-decomposition), avoiding convex optimization and achieving near-linear time in data size.

4. The algorithm is transparent and easy to implement, offering intuitive insight into why subspace aggregation stabilizes principal directions.

5. Includes empirical sensitivity analyses over hyperparameters $(P, R, N)$ and tests on diverse biological datasets (transcriptomic and synaptic connectivity), highlighting robustness and generality.

6. Effectively connects the BBP phase transition and random-matrix theory results to the degradation of PCA under heavy-tailed noise, grounding the algorithm’s rationale.

### Weaknesses
The paper lacks formal theoretical guarantees, deeper analysis of hyperparameter sensitivity, and a comprehensive comparison to established robust-scatter PCA frameworks (e.g., Tyler's (1987), ROBPCA (Hubert et al. 2012) etc.). These omissions limit the perceived depth of contribution. Here are some of the key weaknesses want to outline:

1. The method is supported primarily by geometric intuition and empirical evidence, but lacks formal theoretical guarantees. In particular, the paper does not provide asymptotic analysis, finite-sample error bounds, perturbation-theoretic results, or convergence and robustness guarantees.

2. The algorithm’s dependence on $(N, P, R)$ is discussed qualitatively, but tuning strategies are heuristic and not systematically analyzed, particularly for real-world data.

3. The relationship to established robust-scatter and shape-based PCA approaches (e.g., Tyler's M-estimator, ROBPCA, spatial-sign PCA) is underdeveloped, limiting clarity on conceptual novelty.

4. Baselines focus on convex robust PCA and geometric-median PCA, omitting newer high-dimensional or probabilistic heavy-tail estimators (e.g., Catoni-type covariance, truncation-based PCA, Lerman \& Maunu 2018). 

5. While the method performs well across the presented experiments, the paper offers little guidance on when the proposed approach may fail or be inappropriate. In particular, there is no analysis of how performance depends on the data regime (e.g., high-dimensional settings with $p\gg n$, low sample size scenarios, or weak signal-to-noise conditions), nor any diagnostic tools for practitioners to assess whether HT-PCA is likely to provide a reliable estimate on a given dataset. Given that robustness methods can degrade sharply outside their intended regimes, a clearer discussion of failure cases, limitations, and practical checks (e.g., subsample stability tests) would strengthen the utility and transparency of the proposed approach.

### Questions
In relation to the weaknesses outlined  above, here are the questions for the authors:

1. Instead of taking the mean of projection matrices, have you considered alternative aggregation measures (e.g., geometric or median subspace averaging)?

2. Can you provide any theoretical intuition or formal result about convergence of the leading eigenvector of $W$?

3. How should practitioners choose the hyperparameters $(N,P,R)$ in different regimes?

4. Could your approach be extended to recover multiple principal components simultaneously?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a different idea for what a principal component for data should be when dealing with am emphasis on heavy tailed data.

They suggest the following procedure for defining the principal component of data:
- Given a set of vectors $\vec x_1, \ldots, \vec x_n \in \mathbb R^p$
- Take a random subsample of those vectors, say $\vec{x}_{s_1}, \ldots, \vec{x}_{s_N}$ for $N \ll n$
- Let $\mathbf V \in \mathbb R^{p \times P}$ contain the top $P$ left singular vectors of $\vec{x}_{s_1}  \ldots  \vec{x}_{s_N}$
- The principal component is then the top eigenvector of $\mathbb E[VV^\top]$

This principal component is then estimate by simple monte carlo: Generate many such $V$ matrices, form an empirical estimation of $\tilde W = \mathbb E[VV^\top]$, and return the top eigenvector of $W$.

$V$ is not sampled uniformly at random. It uses importance sampling which is inversely proportional to the norms of the vectors.


The paper gives evidence suggesting that for heavy-tailed data, this recovers a more natural notion of a principal component when compared to classical PCA (i.e. returning the top eigenvector of the sample covariance matrix).

Evidence is empirical throughout the paper; theory is not provided.

### Strengths
I think this is a really interesting text. Overall, I'm inclined to accept, pending some adjustments to the text.

The intuition underlying the proposed estimator makes sense, and is a simple linear algebraic notion. I've never seen it used for this purpose, but I've certainly see people analyze the expected projection $\mathbb E[VV^\top]$ in theoretical problems (see eg Thm 3.1 of [this paper](https://arxiv.org/pdf/2208.09585); no need to cite this or anything just being thorough about the connection).

The paper provides convincing evidence that this notion of a principal component is meaningful, and that it is empirically useful.

The paper is well written and was even kinda fun to read!

I'm no expert on the statistical side of PCA, so I can't really comment effectively on the originality of this work relative to prior work. Taking their long prior work section at face value, this seems like a valuable contribution!

### Weaknesses
The paper suffers four core weaknesses, not all equal:
1. The experiments lack confidence intervals (plus other smaller issues on the figures)
2. The experiments fail to consistently and effectively compare the proposed PCA method to alternative PCA methods on
3. There is not a very crisp formalization of what makes a PCA method "good" for heavy tailed data
4. There is not a clear notion of how to produce more than one principal component


Let's start with experiments.

The figures in this paper are slightly disastrous.
- Despite the proposed method being a randomized algorithm, and the variance of randomized methods for PCA often having non-trivial confidence intervals, none of the experiments seem to have any confidence intervals. In my view, EVERY plot should contain confidence intervals for work like this. I personally prefer seeing the median error with 10/90 quantiles or 25/75 quantile; though mean +- standard deviation is okay. Line 242 says the code was run 20 times, so this should be an easy fix.
- Figure 2 has no real caption (page 5)
- Figures throughout the paper have unexplained parameters. The notion of "ERROR" and "alpha" isn't defined until after figure 2.
- Printed on paper, it's very hard to read the axes of many figures. The text should be larger on the axes (and in some legends)

Next, the paper lacks some baseline comparisons and confuses me at some points.
- "Sample Cov w/ Del." is absent from Fig 3 for some reason
- Section 4.1 (page 8) studies the "self-consistency" metric on real data, but only reports the error achieved by the proposed PCA method, and does not show the error achieved by the other methods considered on synthetic data. No confidence interval on the error is given.
- Section 4.2 (page 9) studies the same error metric on a different real data source, but only reports the error achieved by two PCA methods. The metric here used is confusing as the authors refer to both "self-consistency" and "mean cosine similarity" but only define the former, and perhaps only report the latter? Either way, the metric used here is confusing, and not enough estimators are compared.
- Section 4.2 has a high standard deviation of their error metric, nearly as large as the average value of the metric. Some further discussion about runtime to lower that standard deviation would be good (I know it's discussed elsewhere in the paper; but it needs acknowledgement here as well)


Next, let's get more conceptual. There's not a clear notion of what a good PCA method is.
The paper proposes two tests that a good PCA method should achieve on heavy-tailed data
1. Have good "self-consistency": if you split a dataset in half and run your PCA method on each half, it should return nearly identical vectors
2. Work on data from a specific generative model that has noise distributed as a heavy-tailed Student distribution

These are both... good things we want from PCA, but neither one really is a fundamental notion of what good PCA should be defined as. I'd like to see a more fundamental model for what the authors consider good PCA to be. The authors have this interesting ansatz that with heavy-tailed noise, the fundamental principal components should be distributed amongst the first few eigenvectors of the sample covariance matrix. I'd love to see this pushed a step further, into a potential guess for what a good formalization of PCA would be.

I'll acknowledge that my question here is somewhat underspecified; I'm not sure the authors have a good super formal notion of what PCA should be defined as, and I don't want to give them an undue burden to do such a thing. But if they have a more formal idea, I'd love to see that written out more. (to clarify, not an algorithm, but a more statistical notion of what a principal component should be)



This bleeds into my final topic -- the fact that this paper only considers producing a single principal component.
It's a very obvious question to ask: How should I generate a second principal component?
And how about the $k^{th}$?
Explicit iterative deflation may be needed as in LazySVD; or maybe just returning the top $k$ eigenvectors of the monte carlo projection suffices?
I think this should be acknowledged within this paper, at least to some minimal extent.

### Questions
## List of typos & recommended edits

_ Feel free to ignore anything in here you disagree with, without any need for further discussion _

1. [60] Specify what community these real-world datasets come from. Neuroscience?
2. [68] Last sentence here is phrased too strong. Maybe "Unless many other PCA methods for heavy-tailed data..."
3. [79] "reasonably small matrices"? at ICLR, a 100 by 500 matrix is small but already shows what you want it to show
4. [144] Usually, to me, Pi is a projection not a subspace
5. [throughout] P and p both being symbols in this paper is kinda annoying... maybe swap p for d, or swap P for k?
6. [154] Usually, to me, V is a tall matrix so that V'V is the identity and VV' is a projection. Transpose the definition?
7. [Fig 1] Specify the data used to generate table 1
8. [Throughout] Actually formalize the method used to importance sample. IID sampling with/without replacement? Wrt squared col L2 norms, or non-squared norms? Any smoothness/regularization used?
9. [189] I think this is just Courant-Fisher. Would be good to name-drop that here.
10. [Sec 2.3] This is very long. Shorten this a bunch. Will help you with the page limit.
11. [314] You're using kappa for both kurtosis and signal strength. Split these two different things up.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a method to compute the leading singular vector(s) of a matrix, motivated by scenarios where the matrix data is heavy-tailed (which is known to degrade the performance of other methods)

The method involves iterative taking random columns (ie datapoints) and finding the principal components of this, then adding up these found principal components and finally fining the principal component of the agglomerate. 

There is no rigorous guarantee of when or if the method works, but it is evaluated on synthetic and real datasets.

### Strengths
Method has been clearly described.

### Weaknesses
The main weakness of this paper is that there is no rigorous guarantee that this method will work (or even, a characterization of specific scenarios of when it will work). Most existing methods for Robust PCA have such guarantees, and it is important to establish at least a basic rigorous guarantee of when such an algorithm will work (and, will work better than simple PCA)

Also, the paper is missing many works on robust PCA as comparison baselines, e.g.

https://arxiv.org/abs/1010.4237

https://arxiv.org/abs/2305.02544 (and references therein)

Etc.

### Questions
Not currently

### Soundness
2

### Presentation
3

### Contribution
2
