# Mirror Descent-Ascent for mean-field min-max problems

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
We study two variants of the mirror descent-ascent algorithm for solving min-max problems on the space of measures: simultaneous and sequential. We work under assumptions of convexity-concavity and relative smoothness of the payoff function with respect to a suitable Bregman divergence, defined on the space of measures via flat derivatives. We show that the convergence rates to mixed Nash equilibria, measured in the Nikaidò-Isoda error, are of order $\mathcal{O}\left(N^{-1/2}\right)$ and $\mathcal{O}\left(N^{-2/3}\right)$ for the simultaneous and sequential schemes, respectively, which is in line with the state-of-the-art results for related finite-dimensional algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the convergence properties of simultaneous and sequential mirror descent for solving min-max games. The paper shows that under some technical assumptions, the convergence rates (in terms of NI error) are $O(N^{-1/2})$ in the simultaneous case and $O(N^{-2/3})$ in the sequential case. An example of training GANs is implemented with both variants of MD, and the behavior corroborates the faster convergence rate of sequential MD.

### Strengths
- The main technical results are clearly presented and while the notation is quite heavy, the authors do a good job of making the paper readable.
- The convergence result for sequential MD is the first such result for mean-field min-max games, and the proof technique is quite interesting (if reliant on several technical assumptions).
- The motivating example of GANs is clearly presented and the experimental results nicely corroborate the theoretical statements.

### Weaknesses
- A concern is how extensible the convergence results are -- it is not made clear what other problem settings would satisfy the required technical assumptions to obtain a fast convergence rate. In particular, Assumption 3.4 seems like a very restrictive condition on the second variation, and I believe it would improve the paper to show more examples of divergences and min-max problems that satisfy all technical assumptions.
- Beyond GANs and mean field neural nets, not much is said about the other concrete applications of the framework. Considering that prior work has shown connections and examples ranging from Sinkhorn and EM algorithms to reinforcement learning, a natural question which is not investigated is how sequential MD performs in these settings. I would be more positive on the paper if it can be shown that sequential MD is a superior method for other applications. As it stands, the theoretical results are novel but the significance and applicability of the results are not convincing.
- The notation in the paper is quite dense (and somewhat unavoidable given the topic), but giving some additional clarifications on the assumptions and also providing more detailed proof sketches (in particular emphasizing the way the proofs diverge from standard techniques) would help readers who might be unfamiliar with the material.

### Questions
- The MD variants studied are Euler discretizations of the Fisher-Rao gradient flow in continuous time. I am curious if other discrete-time algorithms which can be obtained by taking different discretizations? Does the continuous-time convergence rate given insights into the behavior of its discretizations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies mirror descent (MD) algorithms on spaces of measures. It provides analysis for both the simultaneous and sequential (a.k.a. alternating) versions of MD for general convex-concave objectives and general objectives, generalizing existing results.

### Strengths
The generalization of existing results to this setting requires, at least in my understanding, a significant amount of technical work, and the paper seems to do this well. The paper is also generally well-written and clear.

### Weaknesses
### Issue with the spaces under consideration
I have trouble understanding on which spaces assumptions and statements are valid.

In particular, it is not clear how Assumption 3.4 can be satisfied for the entropy regularizer with the current wording.
Indeed, Assumption 3.4 requires the existence of a constant $L_{h^* }$ such that the inequality l368 holds uniformly over all bounded functions.
But the constant $L_{h^* }$ given by Prop. G.15 depends on the functions themselves (through their sup norm).
Moreover, in this proof, the authors refer to (Lascu et al., 2025, Lemma A.2) for the Lipschitzness of $\phi$ but I could not find this result in the lemma mentioned (which is several pages long). I also do not see how to obtain such a pointwise Lipschitz bound.
(This issue is the main reason behind my current rating.)

Moreover, in example 1.3, the authors appear to have to restrict the space of probability measures to those with density wthin bounded distance from a reference distribution. This seems to be restrictive compared to (Hsieh et al., 2019) for instance.

### Lack of examples of divergence
In its comparison to previous work, this paper insists on the generality of the divergences considered compared to previous works which were only on KL divergence.
However, most of the assumptions are verified only for the entropy regularizer, see Asm 2.1 and 3.4.
Moreover, for the examples 1.2 and E, the choice of divergence is not discussed, and the reader is refered to Remark 2.2. If I understand correctly, this means that these examples are only valid when $h$ satisfies the inequality l286. If this is the case, this should be made more explicit in the paper and this assumption should be better discussed.


### Minor
- l260-267 make it sound as if the form of the update is novel while, unless I am mistaken, it is standard in mirror methods.
- The appendix is quite long, it could benefit from a table of contents and an introductory section explaining its organization.

### Questions
Could the authors address "Issue with the spaces under consideration" and "Lack of examples of divergence" in the section above?

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
4

### Summary
This paper studies the simultaneous and sequential mirror descent-ascent (MDA) algorithms in solving convex-concave min-max problems with infinite-dimensional action spaces.
For simultaneous MDA, they show that the averaged iterate converges at a rate of $O(1/\sqrt{N})$, where $N$ is the number of iterations.
For sequential MDA, they show that the averaged iterate converges at a rate of $O(1/N^{2/3})$.
Both convergence rates match the known rates for the finite-dimensional bilinear case.
They also implement many numerical experiments to prove the efficiency of the algorithms in its applications, e.g., training GANs.

### Strengths
*   The paper seems to be the first paper to study the sequential MDA for games with infinite action spaces;
*   This paper establishes theoretical foundations for the faster convergence of the sequential MDA than its simultanuous counterpart;
*   The paper is fairly well-written and polished.

### Weaknesses
*   It looks like the new result on the sequential MDA is obtained by the parallel analysis in the discrete-case. It would be better if the authors can point out the main differences between the discrete-time and continuous-time proofs;
*   It seems that in the statement of their main theorems, they missed some key assumptions:
    *   It appears that they need the reference function $h$ to be Legendre, i.e., the norm of the derivative of $h$ goes to infinity as the iterate approaches the boundary. For example, they may need that assumption to get the first-order optimality conditions in Eq. (20). If correct, this is a big limitation that should be called out in many places. Currently, the paper repeatedly emphasizes being for "general Bregman divergences."
    *   The stepsizes need to be set depending on the total number of iterations $N$ for both algorithms, while it seems that they did not include that setup in the statements of Theorem 2.4 and Theorem 3.6.
*.  The GANs motivation does not feel very compelling. Having a probability distribution over network parameters feels highly unrealistic. How would you train this?

### Questions
*   Is my understanding of part 3 correct?
*  How do you imagine realistically training a probability distribution over parameters in a neural network?

Minor points:
*   In Assumption 1.5, it would be clearer if the authors clearify the notations $D_{F(\cdot, \mu)}(\nu', \nu)$ and $D_{F(\nu, \cdot)}(\mu', \mu)$. As $D_h$ is used to denote Bregman divergence in the paper, maybe they can consider use alternative notations for the second-order derivatives if possible;
*   The motivation of MDA formulation in Section 1.6 can be simplified in my opinion, as it aligns with the general gradient-based optimization methods;
*   Remark 2.5 is not entirely right in my opinion, since the usual $1/\sqrt{N}$ rate does not require relative Lipschitzness.
*   In much of the related literature, "sequential" algorithms are called "alternating" rather than "sequential".
*   Typos:
    *   p18 l929: due [to] Assumption 3.5 or [by] Assumption 3.5.
    *   Why are $\mu^\*,\nu^\*$ included in Theorem 2.4? As far as I can tell they play no role?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This manuscript studies mirror descent/ascent algorithms for nonlinear convex–concave problems defined on the space of probability measures. The authors consider two variants of a saddle-point optimization algorithm: simultaneous and sequential. They leverage relative smoothness and Lipschitzness properties of the convex–concave objective to prove a $1/\sqrt{N}$ bound for the simultaneous algorithm. Additionally, they use Hessian Lipschitzness and an $L_{\infty}$ assumption on derivatives to improve the convergence rate to $1/N^{2/3}$ for the sequential algorithm. The results extend earlier results for bilinear objectives and can be extended to training mean-field neural networks.

### Strengths
- The paper is well-written.
- It tackles the technical arguments needed to extend optimization guarantees for bilinear objectives on probability spaces to nonlinear convex–concave objectives.
- It empirically verifies its theoretical results.

### Weaknesses
- The analysis assumes both Lipschitzness and smoothness, yet the convergence of the simultaneous algorithm is $1/\sqrt{N}$, which may be restrictive for nonlinear objectives. The cited analogous result (Bubeck 2015 Theorem 5.1) assumes only Lipschitzness. That said, the authors clearly explain the proof bottleneck, which I appreciate.  
- The implementation requires an internal sampler to trace the algorithm’s trajectory, but the approximation error introduced by this sampler is not characterized.

### Questions
Do you think it is possible to derive a $1/N$ rate in the relative-smoothness case using an algorithm similar to (Bubeck 2015, Section 5.2.3)? What are the main technical challenges? If not, is there an impossibility result—e.g., a matching lower bound—establishing that faster rates are unattainable under these assumptions?

### Soundness
3

### Presentation
3

### Contribution
3
