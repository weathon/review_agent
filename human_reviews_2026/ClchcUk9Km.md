# Generalization of Gibbs and Langevin Monte Carlo Algorithms in the Interpolation Regime

- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
The paper provides data-dependent bounds on the test error of the Gibbs algorithm in the overparameterized interpolation regime, where low training errors are also obtained for impossible data, such as random labels in classification. The bounds are stable under approximation with Langevin Monte Carlo algorithms. Experiments on the MNIST and CIFAR-10 datasets verify that the bounds yield nontrivial predictions on true labeled data and correctly upper bound the test error for random labels. Our method indicates that generalization in the low-temperature, interpolation regime is already signaled by small training errors in the more classical high temperature regime.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a PAC-Bayesian bound based generalization error. To make the bound applicable in practice, it is calibrated on the data with randomized labels. The bound is then tested on benchmark problems.

### Strengths
The paper is very transparent in the assumptions it makes and the conclusions it draws. I appreciate appendix C2 which provides experiments with different architectures and training losses.

### Weaknesses
It is disappointing that the bound needs to be calibrated to hold in practice. To estimate the bound, one needs to repeat the experiment with randomized labels and for different temperatures, which is significantly more expensive than directly estimating the test loss. Combining this with the fact that the calibrated bound does not rigorously hold I am unsure of the proposed use the bound. 

Given that the left-hand-side of Figure 1 is not a valid test of the bound (it is calibrated to hold there) the set of experiments where the bound is tested is somewhat small.

### Questions
1. What is the proposed use of this bound? Could it be used instead of cross validation?
2. The experiments section seems to suggest that the bound is not practically applicable for some losses. Can the authors provide some guidance here?
3. How does the bound accuracy depend on the number of temperature levels?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes new generalization bounds for the Gibbs algorithm, where the algorithm output is sampled from a Gibbs posterior whose potential is proportional to the empirical risk. The main result are based on PAC-Bayesian argument and an integral decomposition of the error across the temperature parameter. This bound suggests that the generalization error at a given temperature is linked to the generalization error at higher temperatures. An important feature of the proposed theory is to be stable when the Gibbs posteriors at various temperatures are approximated (eg, by MCMC algorithms), making the bound computable in practice. Therefore, the theory is supported by experiments on the MNIST and CIFAR10 datasets.

### Strengths
- The proof technique explicitly links the generalization error at different temperatures.
- The stability by approximation makes the bounds computable in practice (with potentially an important computational cost). 
- The new bounds are data-dependent and fully computable in practice

### Weaknesses
*Main weaknesses:*
 - All results are written with placeholder functions in the main text. It would improve the readability to include actual generalization bounds directly in the main text.
 - Due to the representation of the bound as a discrete integral over several higher temperatures, the experiments are very computationally heavy, as Gibbs posteriors at several temperatures have to be approximated. This might diminish the practical reach of the proposed theory.

*Other (more minor) issues:* 
 - Line 73: $\epsilon$ should be $\epsilon_{\beta_k}$
 - Line 105, $P(H) \times P(H)$ should be $P(H \times H)$.
 - Lline 206, as $\exp(F)$ is positive, it seems to me that we do not need much condition to exchange the two expectations
 - $KL$ would be more beautifully written $\mathrm{KL}$.

### Questions
- Is it correct that Corollary 4.2 is a consequence of known results?
- Line 415: why do we need to distinguish between $\beta$ and $2\beta$
- Do you think that it could be possible to make the experiments less computationally heavy by using the estimation of the posterior at higher temperatures as a kind of "warm-start" for lower temperatures, if that can make sense in your setting?

### Soundness
3

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
1

### Summary
This paper is way out of my expertise, I am not in a position to offer a meaningful evaluation.

### Strengths
.

### Weaknesses
.

### Questions
.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper provides a mechanism for bounding test error in overparametrized regimes for Gibbs sampling algorithms (and approximations such as Langevin Monte Carlo), which accurately captures generalization error in the low temperature regime.

### Strengths
The paper shows a scheme for bounding the test error which is rigorously derived and implementable.

In the experiments presented, the predictions appear to align well with the ground truth.

### Weaknesses
I think this paper is not doing anything particularly novel. On the sampling side, the rates of Vempala and Wibisono are already somewhat outdated by the standards of the field and much better analysis is known for the LSI setting. See for instance some of the works on the proximal sampler.

Conversely, although I am less familiar with the learning theory elements, Theorem 3.2 appears to me to be straightforward. Thus, even if this exact result has not appeared in the literature before, the ideas in this paper do not seem particularly novel or surprising.

It is not really possible to assess whether this schema will be useful in practice, as the experiments appear somewhat small in scale. I would imagine that Gibbs sampling on large datasets is both expensive and unlikely to yield informative bounds.

### Questions
The terminology \emph{interpolation regime} is never defined.

219: iid -> i.i.d.\

224 has an extra comma

326:  the Theorem 1 -> Theorem 1

364: sand -> and

### Soundness
3

### Presentation
2

### Contribution
1
