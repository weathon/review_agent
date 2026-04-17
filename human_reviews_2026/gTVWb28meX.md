# Approximate Inference Suffices for Statistical Distance Estimation

- Decision: Reject
- Scores: 8, 4, 4, 6, 6

## Abstract
Statistical distance (also known as total variation distance) and probabilistic inference are fundamental notions, widely used in machine learning, information theory, and high-dimensional statistics.
While there are efficient algorithms that can estimate statistical distance or probabilistic inference in some specific settings, it has remained an open problem to see whether these two notions can be approximately reduced to each other.
In this work, we take the first step in addressing this problem, and show that estimating statistical distance can be reduced to estimating probabilistic inference, via an efficient structure preserving randomized reduction.
This allows us to use approximate inference algorithms to multiplicatively estimate statistical distance in directed graphical models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors consider the problem of estimating statistical distance between two Bayes nets using approximate probabilistic inference queries. Roughly, the problem is defined as follows. 

First, the probability distribution of a Bayes net is as follows: Given a directed graph and a collection of conditional probability tables, the probability of a labeling of the vertices is given by the product of the conditional probabilities of all vertex labels. 
The statistical distance between two distributions is the $\ell_1$ distance between two distributions viewed as vectors.
The goal is to compute statistical distance using approximate probabilistic inference queries: for any arbitrary sets S_1, …, S_n, compute (up to approximate multiplicative error), the probability $X_i \in S_i$ for all $I$.

The general approach of previous works was to define an estimator $f$ and distribution $\pi$ such that the expected value of $f$ over $\pi$ is the statistical distance (possibly normalized by some easy to compute constant $Z$). The authors compute an auxiliary Bayes net where the labels are distributed over a joint distribution $(X, Y)$ with $X \sim P, Y \sim Q$. The two key steps are then to construct the appropriate $f, \pi$ and estimate $Z$. In this work, the authors show how to sample from and estimate parameters of the relevant distributions using approximate probabilistic inference oracles, rather than exact oracles.

Conclusion

Overall, the paper studies an interesting question and finds a nice reduction between two well studied problems. For this reason, I tend towards accept. However, as I am not a domain expert, I found the paper a bit hard to follow (see below comment on usage of prior work) and thus assign a low confidence score. While I did not check the details very closely, the claims seem reasonable and the proofs I did check seem correct.

### Strengths
The paper considers the natural problem of estimating statistical distance in Bayes nets. They establish an interesting result, that statistical distance can be estimated with approximate probabilistic inference queries. This can be viewed both positively and negatively: 1) hardness results in statistical distance estimation can imply that probabilistic inference is hard, or 2) algorithms for probabilistic inference can directly by applied to statistical distance estimation.

### Weaknesses
Perhaps due to the fact that I am not very familiar with the techniques in the paper, the algorithm are a bit hard to follow. Many details seem to be elided over: 1) what is importance sampling? (Yes a standard technique, but good to define and motivate). 2) How do you “carefully” define the CPT to ensure the condition required in 152? 3) Why is $Z = \Pr(X \neq Y)$? And this seems crucial so it is strange there is no justification for this. More generally, it feels that this paper builds off Bhattacharya et al 2024 in a significant way, but does not do so in a self-contained way, assuming the reader has read the prior work.

### Questions
It seems to me that the joint distribution $(X, Y)$ is supposed mimic coupling such that $\Pr(X \neq Y) \sim TVD(P, Q)$ i.e. the optimal coupling. If this is the case, why not just estimate $Z$? 

Minor Comments 

110 - should running time of FPRAS by polynomial in $\log(1/\delta)$?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The authors prove that estimating total variation distance between probability distributions over the same underlying directed acyclic graph (DAG) can be reduced to approximate probabilistic inference while previous work by Bhattacharyya et al. (2024) showed this reduction was possible with exact probabilistic inference.
The main idea is to perform normalization estimation, approximate sampling and Monte Carlo estimation and hence construct the reduction by designing an FPRAS (Fully Polynomial-time Randomized Approximation Scheme).
The authors show that one only needs $O(n^3\ell^2\epsilon^{-2} \log \delta^{-1})$ approximate inference queries where $n$ is the number of nodes in the DAG and $\ell$ is the range of the random variables to approximate the total variation distance up to an $(1\pm \epsilon)$ factor with probability $1-\delta$.

### Strengths
- The result extends the previous limitation that such a reduction can be done with exact probabilistic inference.

### Weaknesses
- The result may be incremental that the main idea is from the previous work (Bhattacharyya et al.)

### Questions
- Line 137: I am not sure what we are normalizing with $Z$.

- Line 144: It may be helpful to mention that $g$ is the surrogate function for $g^\ast$

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper studies the computational relationship between two fundamental problems—statistical distance estimation (specifically total variation distance) and probabilistic inference in Bayesian networks. Building on recent work (Bhattacharyya et al., 2024), which showed a reduction from statistical distance estimation to exact probabilistic inference, this paper proposes a reduction to approximate probabilistic inference instead. The main claim is that there exists an FPRAS for estimating total variation distance using approximate inference queries, potentially broadening the applicability of such methods to practical settings where exact inference is infeasible.

### Strengths
- The paper targets an important and fundamental question in computational learning theory, i.e., linking statistical distance estimation and probabilistic inference

### Weaknesses
- The paper is entirely theoretical with no numerical experiments or case studies to demonstrate how the proposed algorithms might perform in realistic scenarios.

- While technically detailed, the paper is difficult to follow. Algorithmic steps (especially in Algorithms 2 and 3) are densely described without clear intuition or illustrative examples. The exposition would benefit from diagrams or toy examples linking the theory to intuitive settings.

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors study the problem of computing the total variation distance between two probability distributions $P$ and $Q$, each represented as Bayesian networks defined on the same directed acyclic graph (DAG) $G$. The objective is to estimate the total variation distance $d_{TV} (P,Q)$ between these two distributions.

In their ICML 2024 paper, Bhattacharyya et al. established a key connection between total variation distance estimation and probabilistic inference. They showed that if one has access to an exact probabilistic inference oracle, the total variation distance between two Bayes nets can be computed in polynomial time via an exact reduction. However, this reduction critically depends on exact computation of certain normalization constants such as $Z=Pr⁡[X\neq Y]=1-Pr⁡[X=Y]$. When only approximate inference is available, multiplicative errors in estimating $Pr⁡[X=Y]$ can amplify dramatically in its complement $1-Pr⁡[X=Y]$, especially when $Pr⁡[X=Y]$ is close to $1$. As a result, even small relative errors in inference can lead to unbounded relative errors in the estimated total variation distance.

The present paper overcomes this instability by introducing a new, structure-preserving randomized reduction that avoids direct dependence on such complements. It demonstrates that access to a $(1+\varepsilon)$–relative approximate inference oracle suffices to obtain a $(1+\varepsilon)$–approximation of $d_{TV}(P,Q). This establishes the first reduction from total variation distance estimation to approximate probabilistic inference, thereby strengthening the algorithmic connection between the two problems.

To obtain this result, the authors adapt the classical Jerrum–Valiant–Vazirani paradigm that connects approximate counting and sampling, extending it to the setting of Bayesian networks. They develop an importance-sampling framework that estimates normalization constants and draws approximate samples using only approximate inference queries, while carefully controlling the propagation of multiplicative errors.

### Strengths
The paper makes an interesting contribution by strengthening the connection between two classical and well-studied problems in machine learning, namely, statistical distance estimation and probabilistic inference. The technical approach appears non-trivial.

### Weaknesses
My main concern is that the result feels somewhat incremental relative to prior work. Since both exact and relative approximate inference, as well as total variation distance estimation, are known to be #P-hard, the strengthened reduction does not yield new tractable cases. Consequently, the practical algorithmic implications of this improved connection appear limited.

### Questions
Are there specific classes of models or structural assumptions under which approximate inference is known to be tractable?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the relation between learning total variation distance and probabilistic inference in Bayesian nets. The authors show that total variation distance can be approximated in polynomial time using an approximate probabilistic oracle for Bayesian nets.

### Strengths
The authors establish a strong theoretical connection between probabilistic inference and estimating total variation distance in Bayesian nets. The authors prove that it is possible approximate the total variation distance between Bayes nets in fully polynomial time using approximate oracles for probabilistic inference, which estimates the probability of any given event. This improves upon the work of Bhattacharya et.al. 2024 which required exact probabilistic inference oracles.

### Weaknesses
The writing and presentation needs improvement. While many results are developed from the previous work, Bhattacharya et.al. 2024, it is important to make the proof and statements self-contained.

### Questions
1. Are there time complexity lower bounds for the reduction? I guess a $\Omega(n^2)$ lower bound is trivial which is the time to write down the nodes and edges. Could one argue something better than this?

### Soundness
3

### Presentation
2

### Contribution
3
