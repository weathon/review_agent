# SVD Provably Denoises Nearest Neighbor Data

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We study the Nearest Neighbor Search (NNS) problem in a high-dimensional setting where data originates from a low-dimensional subspace and is corrupted by Gaussian noise. Specifically, we consider a semi-random model where $n$ points from an unknown $k$-dimensional subspace of $\mathbb{R}^d$ ($k \ll d$) are perturbed by zero-mean $d$-dimensional Gaussian noise with variance $\sigma^2$ on each coordinate. Without loss of generality, we may assume the nearest neighbor is at distance $1$ from the query, and that all other points are at distance at least $1+\varepsilon$. We assume we are given only the noisy data and are required to find NN of the uncorrupted data. We prove the following results:

1. For $\sigma \in O(1/k^{1/4})$, we show that simply performing SVD denoises the data; namely, we provably recover accurate NN of uncorrupted data (Theorem 1.1).
2. For $\sigma \gg 1/k^{1/4}$, NN in uncorrupted data is not even {\bf identifiable} from the noisy data in general. This is a matching lower bound on $\sigma$ with the above result, demonstrating the necessity of this threshold for NNS (Lemma 3.1).
3. For $\sigma \gg 1/\sqrt k$, the noise magnitude ($\sigma \sqrt{d}$) is significantly exceeds the inter-point distances in the unperturbed data. Moreover, NN in noisy data is different from NN in the uncorrupted data in general.
\end{enumerate}

Note that (1) and (3) together imply SVD identifies correct NN in uncorrupted data even in a regime
where it is different from NN in noisy data. This was not the case in existing literature (see e.g. (Abdullah et al., 2014)). Another comparison with (Abdullah et al., 2014) is that it requires $\sigma$ to be at least an inverse polynomial in the ambient dimension $d$. The proof of (1) above uses upper bounds on perturbations of singular spaces of matrices as well as concentration and spherical symmetry of Gaussians. We thus give theoretical justification for the performance of spectral methods in practice. We also provide empirical results on real datasets to corroborate our findings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the Nearest Neighbor Search (NNS) problem where data points, originating from an unknown $k$-dimensional subspace within a $d$-dimensional space ($k \ll d$), are corrupted by Gaussian noise . The objective is to recover the nearest neighbor of the *uncorrupted* data, given only noisy observations and queries. The authors propose a simple SVD-based algorithm that involves splitting the data matrix, computing the top-$k$ subspace for each half, and then projecting the data from one half onto the subspace derived from the other to find the nearest neighbor . The primary contribution is a proof that this method successfully recovers the true nearest neighbor even when the noise variance $\sigma$ is as large as $O(1/k^{1/4})$ . This is a significant finding, as it holds in a noise regime where the nearest neighbor in the noisy data may differ from the true nearest neighbor. The authors establish this as a sharp threshold by providing a matching information-theoretic lower bound, demonstrating that recovery is impossible for $\sigma \gg 1/k^{1/4}$.

### Strengths
The paper addresses a practical and fundamental problem in data analysis. The main strength is the substantial improvement over prior SOTA (e.g., Abdullah et al., 2014), which required the noise level $\sigma$ to be bounded by an inverse polynomial in the *ambient* dimension $d$. This work's bound of $\sigma = O(1/k^{1/4})$ depends only on the intrinsic dimension $k$, which is a major advancement for $k \ll d$ scenarios .

The work extends our understanding of this problem and identifies critical thresholds for $\sigma$ and providing both an algorithmic upper bound and a matching lower bound. This comprehensive analysis is a key strength. Another significant contribution is showing that the algorithm works even when the noise is large enough ($\sigma \gg 1/\sqrt{k}$) to change the identity of the nearest neighbor in the observed data, a regime not handled by previous work.

The algorithm itself is simple and clearly explained. The theoretical claims are supported by experiments on both synthetic and real-world datasets (Glove and MNIST), which confirm the algorithm's practical benefits over a naive approach and validate the theoretical dependency on key parameters .

### Weaknesses
The primary weakness is the lack of an explicit discussion of the paper's technical novelty. The analysis appears to rely on standard matrix perturbation bounds (like Davis-Kahan and Wedin) and concentration inequalities. The authors do not clearly articulate what new analytical techniques or core technical innovation enables them to achieve the $O(1/k^{1/4})$ bound, which is the paper's central improvement. It is unclear if the novelty lies simply in the data-splitting algorithm design, which simplifies independence arguments.

This new bound comes at the cost of a dependency on $s_k(B)$, the $k$-th singular value of the unperturbed data matrix. Prior work did not require this assumption. While the authors argue in Section 2.3 that $s_k(B)/\sqrt{n}$ is likely a non-zero constant for "well-conditioned" data, this is a significant trade-off, especially when data is approximately embedded in a subspace (which is one core motivation of the model considered in this paper); see the question below about overspecification. 

The experimental comparison is made against a naive baseline, not against the (Abdullah et al., 2014) algorithm that serves as the main theoretical comparator. The authors state this was due to the implementation infeasibility of the prior work, but this omission makes it difficult to empirically assess the practical performance gain over the previous state-of-the-art.

Finally, some typos and clarity suggestions
* "weel-known" instead of "well-known".
* In Section 3.1, the query point $\tilde{q}$ is missing from the problem description (Line 062), which makes the notation for $q$ confusing.

### Questions
1.  Could you please clarify the core technical novelty of your analysis? The data-splitting trick  simplifies the probabilistic argument, but is this the key element that allows you to break the dependency on the ambient dimension $d$ and achieve the $O(1/k^{1/4})$ bound? Or is there a new, non-standard bound or analytical step being used?

2.  The discussion in Section 2.3 regarding the requirement to know $k$ is confusing . You state that using a "larger dimensional SVD subspace projection" (i.e., overspecifying $k$) "may be of use if we want to work with weaker assumptions" . This seems counter-intuitive. Your bounds in Theorem 1.1 depend on $s_k(B)$. If the true rank is $k$ and you use a $k' > k$, the $k'$-th singular value $s_{k'}(B)$ would be zero. This would make your noise bound infinitely restrictive, not weaker. Can you clarify how overspecifying $k$ could be helpful?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies nearest neighbor search problem for high-dimensional spaces where the data lies low-dimensional subspace and is coordinate-wise corrupted by Gaussian noise. The authors shows that when the noise have small variance i.e. $\sigma = O(k^{-1/4})$ they can recover the correct nearest neighbor; for large variance, recovery becomes impossible.

### Strengths
The paper establishes tight upper and lower bounds on the noise threshold for nearest neighbor recovery, providing a clear theoretical characterization of when SVD-based denoising succeeds. The proposed algorithm is conceptually simple and well-presented, with a clean and transparent analysis that makes the results easy to follow.

### Weaknesses
I think the main critism here is the setting is too ideal seems a bit far from practical: It assumes the points are exactly in a k-dimensional subspace.  Moreover, the guarantees depend on the singular value of the clean data matrix, which could be very large for ill-conditioned data.

### Questions
1. The model assumes data drawn exactly from a low-dimensional linear subspace corrupted by isotropic Gaussian noise. Could you identify any realistic scenarios or application domains where this setting meaningfully reflects observed data distributions.
2. Do you think the dependence on the s_k is an artifact of the analysis or it is actually tight?

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
This paper studies nearest neighbor problem while data is corrupted with random Gaussian noise. That is, given arbitrary $n$ points in $d$-dimensional space that can be embedded in a $k$-dimensional subspace, with zero-mean $d$-dimensional $\sigma$-variance Gaussian noise, the algorithm is able to distinguish the nearest neighbor while all other points are at least $(1+\epsilon)$ distance away. The paper gives detailed analysis on how large a $\sigma$ may affect the distinguishability of the neighbor points. The algorithm is based on spectral method, specifically only two SVD calls, which outperforms the prior work that builds on a more complicated PCA tree, however, with an assumption that the $k$-th singular value is large enough.

### Strengths
Comparing to the prior work, i.e. Abdullah et al., 2014, this paper achieved an improved noise tolerance with a simple spectral method with two calls of SVD on randomly separated data points. While Abdullah et al., 2014 can tolerate up to Gaussian noise with variance of at most an inverse polynomial of $d$, this paper designs an algorithm that can handle $\sigma=O(1/k^{1/4})$. On the other hand, the paper further extends the theoretical foundation for spectral methods that perform well on nearest neighbor search problems in many occasions, sometimes even better than the worst-case optimal random projection. The theoretical framework follows from Abdullah et al., 2014 by considering a semi-random model.

### Weaknesses
The paper assumes a random Gaussian noise, which follows from Abdullah et al., seems strong. In this case, the algorithm is highly dependent on a high amount of randomness. Is it possible to find the nearest neighbor when corruptions on $d$ coordinates are no longer independent? On the other hand, given the large top-$k$ sigular values assumption, it is not obvious why random projection will necessarily fail in this case. Therefore, is $\sigma=\Theta(k^{-1/4})$ a necessary criteria for spectral methods to outperform random projection? It would be nice to list all noise level thresholds when or when not SVD would be preferred to random projection.

### Questions
See weaknesses.

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
3

### Summary
The paper studies the NNS problem in high dimensions and shows that if the noise is corrupted by Gaussian noise, then simply performing SVD recovers the NN of the uncorrupted data. This, I think, is a really nice result. 
They study this phenomenon for various settings of variance of the Gaussian noise. It also improves the result of previous works that relied on the variance to be inverse polynomial in the ambient dimension.

### Strengths
The strength of the paper lies in the range of the variance for which they can show denoising using simple SVD.

### Weaknesses
It is hard to parse Theorem 1.1. It would be great if the authors had added more discussion on it right after the theorem. 

219: lemma not theorem
Line 237: well-known
Why the growth of singular values has to be distributed?
I am a little confused. Data matrix being well conditioned is a normal assumption in real datasets, but is it also the case for geometric problems like NN?
My personal opinion is that having a worse noise assumption is better than that on the data matrix because we can control the latter. I would love to hear the authors’ perspective on this front. 

One important issue with the submitted version is that it takes an unfair advantage of the page limit by making the margin smaller than the ICLR format (at least it looks like that to me; I might be wrong). I wanted to flag this in case other reviewers also have an objection with that. I understand that margin bound is not explicitly stated in the Call for Papers, but this feels wrong to me, mainly due to this line in the Call for Papers: "Papers with main text beyond the page limit will be desk-rejected."

I leave the last point to the meta reviewers and AC to make a judgement on, especially in regards to fairness to other submissions.

### Questions
i would say that performing SVD is an expensive process, especially for high dimensional data points. Is there some other way one can do to speed up this proces?

Why should one believe that data matrix is well condition when the underlying data is the one used in NN? Is there any empirical evidence that the authors can point to?

### Soundness
3

### Presentation
3

### Contribution
3
