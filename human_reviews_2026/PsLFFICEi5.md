# Algorithms and Hardness for Estimating Statistical Similarity

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
We introduce and study the computational problem of determining statistical similarity between probability distributions.
For distributions $P$ and $Q$ over a finite sample space, their statistical similarity is defined as $S_{\mathrm{stat}}(P, Q) := \sum_{x} \min(P(x), Q(x))$.
Despite its fundamental nature as a measure of similarity between distributions, capturing essential concepts such as Bayes error in prediction and hypothesis testing, this computational problem has not been previously explored.
Recent work on computing statistical distance has established that, somewhat surprisingly, even for the simple class of product distributions, exactly computing statistical similarity is \#$\mathsf{P}$-hard.
This motivates the question of designing approximation algorithms for statistical similarity.
Our first contribution is a Fully Polynomial-Time deterministic Approximation Scheme (FPTAS) for estimating statistical similarity between two product distributions.
Furthermore, we also establish a complementary hardness result.
In particular, we show that it is $\mathsf{NP}$-hard to estimate statistical similarity when $P$ and $Q$ are Bayes net distributions of in-degree $2$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the problem of estimating the statistical similarity between two discrete probability distributions, defined as the sum (over all points in the support) of the smaller of the two probabilities at each point. This quantity is closely related to the well-known total variation (TV) distance, since the two are exact complements: knowing one determines the other. While additive approximations of TV distance translate directly to additive approximations of statistical similarity, this correspondence does not extend to relative (multiplicative) approximations.

The paper focuses on computing statistical similarity between product distributions over the corners of a unit hypercube, i.e., $[0,1]^d$, where each coordinate has an independent one-dimensional marginal. Since the exact computation of TV distance between such product distributions is known to be #P-hard, the same hardness applies to statistical similarity. The authors present a fully polynomial-time approximation scheme (FPTAS) for estimating statistical similarity in this setting. An FPTAS for the TV distance between product distributions was previously known, and my understanding is that the proposed algorithm extends and adapts those ideas to handle the statistical similarity formulation.

The paper also establishes that obtaining a relative approximation is NP-hard for Bayesian networks of in-degree two, complementing the positive result by showing that multiplicative approximations are generally intractable. Together, these results delineate the boundary between tractable and intractable cases for this class of problems.

My main concern is the level of algorithmic novelty relative to prior work on TV distance approximation—the techniques appear to be reparameterization of the TV-distance approximation rather than containing any fundamentally new algorithmic ideas. Nevertheless, the results are technically sound and provide a clear and useful extension, showing that statistical similarity can indeed be approximated efficiently in polynomial time.

### Strengths
The results are technically sound and provide a clear and useful extension of TV-distance, showing that statistical similarity can indeed be approximated efficiently in polynomial time.

### Weaknesses
The level of algorithmic novelty relative to prior work on TV distance approximation is small—the techniques appear to be reparameterization of the TV-distance approximation rather than containing any fundamentally new algorithmic ideas

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper examines the computational complexity of approximating the statistical similarity of two distributions $P,Q$ on a finite sample space. This quantity can be defined as $S_{\text{stat}}(P,Q) = 1-d_{\text{TV}}(P,Q)$, where $d_{\text{TV}}$ is the total variation distance. Previous work has shown that an FPTAS exists for the TV-distance between product distributions. However, proving this existence for statistical similarity is not straightforward, as the existence of an FPTAS for a function $f$ need not imply the same for $1-f$. The paper resolves this question of existence in the affirmative by constructing an explicit FPTAS for $S_{\text{stat}}(P,Q)$ when $P,Q$ are product distributions. The authors further demonstrate that this property is ``sharp'' in the sense that computing a multiplicative approximation for a slightly more complex family of distributions (Bayes nets of in-degree two) is NP-hard.

### Strengths
The paper defines an FPTAS for computing the statistical similarity between product distributions, and confirms the NP-hardness of obtaining a multiplicative approximation for distributions given by Bayes nets of in-degree two. The work fills an interesting theoretical gap, complementing similar results for the total variation distance. The description of the algorithm, and the proofs of the various results are clear and easy to follow.

### Weaknesses
There are some potential issues with the proof of Theorem 6 (existence of an FPTAS algorithm) that I have highlighted below. I would be willing to provide a more positive score if these can be resolved. Perhaps some comments on the practical aspects of the proposed algorithm relative to the FPTAS for the TV distance (or other estimation techniques for this) could further motivate the work.

### Questions
Concerning the proof of Theorem 6 (existence of an FPTAS):

1. In the proof of Lemma 9, moving from line 327 to 329 requires the-inequality $\sum^{n-2}_{k=0}(1+\delta)^k \leq (1+\delta)^n$ where $\delta = \epsilon/2n$. I believe one can show that for a fixed $\epsilon$, the LHS is unbounded in $n$ while the RHS is bounded.

2. In the same proof, moving from line 329 to 330 requires $(1+\delta)^n \leq (1+n \delta)$. This is contradicted by Bernoulli's inequality (for $n>1,\delta>0$, asserts $>$ holds rather than $\leq$).

3. The same thing occurs in lines 335-336 where it is claimed that $(1+\delta)^n S_{\text{stat}}(P,Q) \leq (1+\epsilon/2) S_{\text{stat}}(P,Q)$.

4. I would appreciate some clarification on how the inequality $S_{\text{stat}}(P,Q)(1+\epsilon) \geq 2n^2 \gamma B (1+\epsilon) + S_{\text{stat}}(P,Q)(1+\delta)^n$ on lines 361-362 is attained. The text claims that this uses similar reasoning to Lemma 9, which may be erroneous given the above.

Minor issues (presentation, typos, etc.):

5. I would appreciate some further detail on how the final expression for the computational complexity of the FPTAS algorithm is obtained (lines 278-282). Perhaps this could be included in the supplementary material.

6. In the proof of Lemma 9, the chain of (in)equalities from lines 289 to 315 could be condensed without affecting readability. For examples, lines 294 and 297 could be combined, as could 306 to 312.

7. In lines 306 and 607, moving the constant through the ``min'' function should lead to a $\leq$ sign rather than an equality.

### Soundness
2

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
This paper studies the problem of computing the statistical similarity between two distributions. This is defined as S(P,Q) = sum_{x in support} min(P(x), Q(x)). This natural notion can be interpreted as the error achieved by an optimal Bayes classified (for the similarity of the label 0 and label 1 distributions) and similarly in hypothesis testing.

The paper studies two new results about the computational complexity.

First, for product distributions, it gives the first FPTAS. (Note that the product distribution case is specifically the one which applies to hypothesis testing.) The problem was previously known to be #P hard to compute exactly in this case.

Second, if P and Q come from Bayes nets, then it shows it is NP-hard to estimate the similarity.

### Strengths
The problems studied here are fundamental. The paper provides strong theoretical results with clean proofs, both for algorithms and for lower bounds. The introduction motivated well that these are important problems to study, particularly these special cawses (product distribution adn Bayes net) which I was skeptical of at first.

### Weaknesses
My main concern is that the proof techniques apper quite standard or follow from prior work. Could you comment on where the technical novelty is? In particular, why does the result of Bhattacharyya et al. (2023) not already essentially imply Theorem 7?

### Questions
Can we extend the results to the case when Q is known and only P is unknown? Such as testing whether a Bayes net has desired behavior?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper solves the following problem: There are two distributions $P$ and $Q$ which are product distributions over $[\ell]^n$. The goal is to "estimate" $\sum_x \min(P(x),Q(x))$, where the sum runs over the entire domain. Given any fixed precision parameter $\epsilon \in (0,1)$, the paper proposes an algorithm to compute a $1+\epsilon$ multiplicative approximation of this quantity. The algorithm assumes query access to the underlying probability values of all the marginals and it is not clear what the exact query complexity dependence is in the main Theorem 6.

### Strengths
Estimation properties of distributions is a fundamental topic in machine learning and at least the "complement" problem of estimating the TV distance is a well studied topic.

### Weaknesses
A major weakness of the paper is that it feels slightly misleading for the following reasons: There is actually nothing 'statistical' about the paper as the title/intro suggests. The algorithm is not sampling from the distributions! Rather, this is a purely query complexity result where the result assumes knowledge of the underlying probability values. There is no notion of sample complexity which is what the title 'statistical similarity' would suggest. 

In fact, this problem does not seem to admit any meaningful approximation using samples. This is perhaps why the authors are assuming access to the probability values? But this should be spelled out in the text as getting samples is usually the standard way of accessing distributions. The reason this problem is difficult with samples is that the quantity $\sum_x \min(P(x),Q(x))$ can be very close to 0 (e.g. imagine distributions that are supported on disjoint subsets) and we would not figure this out unless we took many samples. Indeed, as the authors point out, this relates to estimating the total variation distance which is known to require samples almost equal to the domain size. 

But I don't see why assuming the knowledge of the exact marginal probabilities (in the case that P and Q are product distributions) is a natural assumption. In fact, I also don't see why assuming P and Q are product distributions in the first place is a natural assumption. This makes the problem feel very niche (note that the interesting cases of TV estimation assumes only sample access and it is known that one needs sample complexity scaling with the domain size to obtain additive error).

Thus it is unclear to me if this is an interesting result for the learning theory community since there is no aspect of sample complexity.

The presentation of the main theorem statement also leaves a bit to be desired. Theorem 6 is written extremely informally and does not state what the exact epsilon dependence is on the query complexity (of how many of the underlying probabilities are queried). It also seems to 'hide' the PDF access model by saying "product distributions succinctly represented by their component distributions" , which should not be appropriate for a formal theorem statement.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
1
