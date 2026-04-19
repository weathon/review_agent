# Oracle efficient truncated statistics

- Decision: Accept (Poster)
- Scores: 6, 8, 3, 8

## Abstract
We study the problem of learning from truncated samples: instead of observing
samples from some underlying population $p^\ast$, we observe only the examples that fall in some survival set $S \subset \mathbb{R}^d$ whose probability mass (measured with respect to $p^\ast$) is at least $\alpha$.  Assuming membership oracle access to the truncation set $S$, prior works obtained algorithms for the case where $p^\ast$ is Gaussian or more generally an exponential family with strongly convex likelihood --- albeit with a super-polynomial 
dependency on the (inverse) survival mass $1/\alpha$
both in terms of runtime and in number of oracle calls to the set $S$.  In this work we design a new learning method with runtime and query complexity polynomial in $1/\alpha$.  
Our result significantly improves over the prior works 
by focusing on efficiently solving the underlying optimization problem using a general
purpose optimization algorithm with minimal assumptions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the problem of learning an exponential family with truncated samples. Specifically, for any distribution $p$, the truncated distribution $p^S$ corresponds to the distribution of $p$ restricted to a set $S$. The paper shows that if a distribution $p$ is selected from an exponential family, and we are able to observe samples from $p^S$ and access a membership oracle for $1\\{x \in S\\}$, then one can find an estimator $\hat{p}$ such that $KL(p^S ||\hat{p}^S) \leq \epsilon$. The main technical contribution is to obtain bounds on the sample complexity and oracle complexity that depend poly-logarithmically on $\alpha:=p^S[S]$, with a computationally efficient estimator $\hat{p}$.

### Strengths
The main strength of this paper comes from the polylogarithmic dependency on the parameter $\alpha$ in both the sample and oracle complexities. As far as I'm aware, this seems to be original, at least for bounding the KL-divergence of the truncated distributions themselves. The paper also introduces several new techniques, such as the minimization of the "truncated" negative log-likelihood (NLL) constrained by a bound on the original NLL.

### Weaknesses
The main weakness of the paper is its presentation; it is quite difficult to follow the proofs provided in Section 3. I have the following specific comments:

1. The paper emphasizes that its primary contribution is improving the poly($1/\alpha$) dependency from Lee et al. (2023) to poly-log($1/\alpha$). However, a quick look at that paper reveals that Lee et al. (2023) focuses on estimating the parameter, not the truncated distribution. Therefore, I am unsure how significant this improvement is.

2. Section 3 is quite dense, and the logical flow of the proof is not well explained. Overall, the section feels extremely unstructured.

I suggest that the authors improve the writing in Section 3 by, for example, providing a clear roadmap of the proof at the beginning of this section and clearly explaining how each lemma contributes to the final proofs. Please also refer to the "Questions" section.

### Questions
1. I assume you are using $\theta^*$ to denote the ground truth parameter throughout the paper, but in Corollary 3.9, it seems to be defined differently. Is the $\theta^*$ in Corollary 3.9 the ground truth as well?

2. In the second assertion of Corollary 3.10, why is the consequence stated for $\theta^*$ when your condition is for $\theta^0$?

3. Can you provide a detailed proof of Corollary 3.10? Specifically, it is unclear how the definition $\theta^0 = \mathbb{E}_{p}[X]$ plays a role in the proof.

4. In the proof of Theorem 3.3, how do you derive $p_{\theta}(S) \ge \frac{1}{\alpha^2} e^{-\epsilon}$? I assume you are referring to Corollary 3.10, not Corollary 3.9? Moreover, what is the significance of this result in the overall proof?

5. Can you clarify the terms $w$ and $f$ in the proof of Theorem 3.3? How do they relate to $\theta$ and $\mathcal{L}$?

6. Typos:
    - Line 466: $\min\\{\min...\ \\}$
    - Line 519: It should be "Here" instead of "Where" to start a new sentence.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of learning a high-dimensional probability distribution from "truncated samples" -- that is, the goal is to learn a distribution $p$ given samples from $p$ conditioned on some event $S$, called the "survival set".This is not possible with reasonable (polynomial) sample complexity for an arbitrary distribution $p$, so, like prior work, the paper focuses on exponential family distributions.

More concretely, the model is that one gets:

-- samples from $p$ conditioned on $S$; the set $S$ is not known to the algorithm
-- access to an oracle which takes as input a probability distribution $q$ and returns a samples from $q$ conditioned on $S$

Here $p$ and $q$ should be exponential family distributions so that they have a short description which can be handed to the oracle. There is a further constraint that the probability mass of $S$ under $p$ and $q$ is at least some parameter $\alpha > 0$. As $\alpha$ gets small, this problem gets harder, since we are observing only samples from some low-probability event under $p$.

Prior work showed how to accomplish this task for exponential family distributions, with sample complexity and running time depending polynomially on dimension of the distribution and the inverse of the accuracy to which the distribution is learned, but super-polynomially on $1/\alpha$. (Prior work achieved only quasipolynomial dependence.)

This dependence on survival probability is important for overall running time, because one way to implement the sampling oracle is via rejection sampling, using a membership oracle for $S$ and a sampling algorithm for $q$. But the running time of such an implementation will be about $1/\alpha$ to get a single sample.

The main contribution of this work is to give an algorithm with running time and sample complexity having polynomial dependence on dimension, inverse of the accuracy, and $1/\alpha$. My understanding is that this result is novel even in the special case of Gaussian distributions.

The problem is extremely well motivated -- truncation is a common situation for "in the wild" datasets. And the technical contribution is certainly sufficient for ICLR. I recommend acceptance.

### Strengths
See above

### Weaknesses
See above

### Questions
Minor comments:

-- It would be nice if the analogues of 1.6 and 1.7 from prior work were stated formally rather than alluded to in prose, for easier comparison. For example, I am still a little unsure after reading the prose if prior work gives an algorithm with $poly(1/\alpha)$ dependence for the special case of Gaussians?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors address the problem of estimating a parametric distribution from truncated samples. 
They propose an efficient procedure for parameter estimation within an exponential family under these conditions, improving the dependence on the probability mass of the survival set from a super-polynomial to a polynomial rate.

### Strengths
This work fits well within the existing literature; the efficiency problem is well-motivated and clearly defined. 
The authors present a clear improvement in terms of computational efficiency.

### Weaknesses
The presentation of the paper is poor:
- The structure of the paper is unclear, the main approach is not even presented in the main body of the paper;
- There are significant redundancies; for example, Section 2.1 is nearly identical to Section 1.1;
- There are many typos.
 
The proof of the main result contains several gaps (see Questions), and I do not find it convincing at this stage.

There are no numerical experiments to illustrate their theoretical results or to compare with existing approaches.

### Questions
- l.45: the definition of $A(\theta)$ is wrong given the previous definition of $p_\theta$;

- Definition 1.1 should not be a definition but a proposition;

- Theorem 3.3 and Theorem 3.4 are nearly identical, differing only in their dependencies on $\alpha$ and $\alpha^2$. I find it confusing to present both; what is the rationale behind this choice?;

- In Lemma 3.7, $u$ and $\hat{u}$ are never defined, I guess that they authors meant $\theta^*_S$ and $\hat{\theta}$?


**Proof of Theorem 3.3:**

- Inconsistent notation for $\theta_0$ that appears as $\theta_0$ and $\theta^0$;

- l.519: why is there a $L^3$ term in the lower bound on $N$? It appears as $L$ in Lemma 3.7;

- l.521: the bound on $\lVert \theta_0 - \mathbb{E}[T(x)] \rVert_2$ holds with probability at least $1-\delta$ (according to Lemma 3.7). However, the bound is treated as if it hold almost surely in the rest of the proof. I believe that this will cause an issue when bounding the deviation of $f(\bar{\theta})$ from $f(\theta^*)$ (l.529). Can it be easily fixed?

- I do not understand the inequalities l.525. Since it it never defined, I assumed that $w^{(0)} \coloneqq \theta_0$. In this case I don't understand where the $1/\lambda$ multiplicative term comes from in the first inequality. Moreover, I cannot reproduce the second inequality using the current information in the proof. Can the authors clarify this part?

- There are many typos in the proof: $f$ instead of $\mathcal{L}$, $w$ is not defined, l.525 the bound holds for the norm of the gradient, the index $\theta$ is not defined l.524...

### Soundness
1

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper revisits the problem of learning an exponential family of distributions under truncated samples. The main focus is the dependence of the runtime and sample complexity on the mass of the truncation set. 

In the truncated samples model, the samples are only observed when they fall into a subset S of the domain that has some measure $\alpha$. Such a sampling oracle can be implemented using a membership oracle for the set S. Although for constant values of the mass $\alpha$ the sample (and query) complexity asymptotically matches the bounds from prior work on the problem, there is an exponential improvement in the dependence on $1/\alpha$, when $\alpha$ gets close to $0$. Another difference from prior work is the use of weaker assumptions, which also leads to a weaker type of learner. More specifically, while prior work by Lee et’ al can estimate the parameter vector of the exponential family up to error $\varepsilon$, in this paper, the algorithm is a proper learner that outputs an exponential family that has small KL divergence with the true distribution. 

From a technical standpoint, the novelty lies in the definition of a sequence of constrained optimization problems. Specifically, the authors use projected SGD in a region where the negative log likelihood is not allowed to exceed a certain threshold during each iteration. This prohibits the algorithm from moving to regions in the parameter space that would assign very small mass to the truncation set and seems to be crucial for achieving the exponential improvement in terms of the inverse mass $1/\alpha$ of that set S. 



Minor comments:

-Line 53: “resoursed”->“resourses”

-Lines 109-114: The example here needs some rephrasing. I get the point you are making, but technically, when you fix a set (S=[0,1] in this case), the mass assigned to it and the variance of the distribution, there is a unique distribution (NOT multiple ones) satisfying these constraints. Maybe you could say $O(\alpha)$ mass.

-Lines 131 and 371: In the second argument of the KL distance, it should be $\hat{\theta}$ in the subscript.

-Line 388: Usually the word “requires” is used for lower bounds. Maybe say “a good initialization can be found”

-Line 399: “estimate of”->”bound on”

-Line 448: Proof of Observation 3.8: The statement seems reasonable,  but I think that the proof is incomplete. You show that the exponential is at most 1, but you need something stronger since it could be that $Pr_{\theta^\prime}[S]>Pr_{\theta}[S]$, right?

-Line 490: “$SE(K,\beta)$”->“$SE(K^2,\beta)$”

### Strengths
The paper gives an interesting perspective to an important question in statistics involving data corruption and potentially expands the applicability of state-of-the-art techniques to settings where the corruption is much stronger.

### Weaknesses
I believe the authors should make the assumptions they need to use clearer from the introduction to improve the presentation quality. For example, the strong convexity and smoothness assumption is not mentioned in the statement of Theorem 1.5, which makes it seem too strong.

### Questions
In section 1.1 (line 133), you mention that: “the fact that our algorithm only requires an α-truncated sample oracle is what enables us to achieve a polynomial number of oracle calls to the membership oracle to S”. Can you elaborate on that? Is this not due to the threshold that you introduce in the optimization problems? It seems that assuming the truncated oracle access only allows logarithmic dependence instead of polynomial, which you would get via rejection sampling.

### Soundness
3

### Presentation
2

### Contribution
3
