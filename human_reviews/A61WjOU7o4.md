# Identifying Optimal Output Sets for Differential Privacy Auditing

- Decision: Reject
- Scores: 6, 6, 6, 3

## Abstract
Differential privacy limits an algorithm's privacy loss, defined as the maximum influence *any* individual data record can have on the probability of observing *any* possible output. Privacy auditing identifies the worst-case input datasets and output event sets that empirically maximize privacy loss, providing statistical lower bounds to evaluate the tightness of an algorithm's differential privacy guarantees. However, current auditing methods often depend on heuristic or arbitrary selections of output event sets, leading to weak lower bounds. We address this critical gap by introducing a novel framework to compute the *optimal output event set* that maximizes the privacy loss lower bound in auditing. Our algorithm efficiently computes this optimal set when closed-form output distributions are available and approximates it using empirical samples when they are not. Through extensive experiments on both synthetic and real-world datasets, we demonstrate that our method consistently tightens privacy lower bounds for auditing differential privacy mechanisms and black-box DP-SGD training. Our approach outperforms existing auditing techniques, providing a more accurate analysis of differentially-private algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel framework for privacy auditing. The framework leverages the likelihood ratios function thresholding to select the optimal output set, which maximizes the privacy loss lower bound. The optimality of the proposed framework is proved and the advantage over existing approaches is validated by empirical results.

### Strengths
The result is proved to be optimal as stated in Theorem 4.3. This theoretical finding is supported by empirical evidence presented in Sections 5 and 6.

### Weaknesses
(1) The proposed framework builds upon existing techniques such as likelihood ratio thresholding.

(2) Some implementation details are unclear to me. I include them in the questions (1) and (2).

### Questions
(1) Can the author provide more details on how to sample from p and q for the experiments presented in section 6? 

(2) What's the size of levels $\tau$ ?

(3) In proposition 4.1, what's the purpose of setting $\delta=0$? Does it only apply to auditing pure-DP?

(4) In algorithm 1, samples drawing from p and q are assumed to be equal. Is this assumption needed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies privacy auditing for differential privacy. The paper proposes an approach that claims to identify or approximate the optimal output event sets that can achieve maximal privacy loss lower bound in auditing.

### Strengths
The paper studies the problem of identifying optimal output event sets for differential privacy auditing, which is an important problem that can lead to better and more efficient privacy auditing.

### Weaknesses
However, the paper has significant flaws in both the theoretical claims and the methodological support for its core assertions. Please refer to the Questions below for more details. 

I am open to increasing my score if the authors can provide convincing clarifications or solutions to my concerns in the rebuttal.

### Questions
Comment 1: 

Theorem 4.3 cannot prove that the $\tau$-log-likelihood-ratio-set enables maximum auditing lower bound objective (4) among all possible choices of output set $S$.  I will explain this as follows.

First, by fixing p and q, there always exists a $\tau'$ for any feasible output set $\mathcal{O}$ such that this $\mathcal{O}$ is $\tau'$ -log-likelihood-ratio-set; i.e.,  $\mathcal{O}=\mathcal{O}$\_$\tau'$. 

Now, in Section B.2 PROOF FOR THEOREM 4.3, let's replace $\mathcal{O}$_$\tau$ by $\mathcal{O}$\_$\tau'$, and replace $\tau$ by $\tau'$. In addition, let's replace $\mathcal{O}$ by any $\mathcal{O}'$ that satisfies $p(\mathcal{O}')$ + $q(\mathcal{O}')$ = p($\mathcal{O}$\_$\tau'$) + q($\mathcal{O}$\_$\tau'$) .

By following the same steps, we can obtain the similar inequality of Eq (24), where the left-hand side integration is over the set $\mathcal{O}$\_$\tau'$ and the right-hand side integration is over the set $\mathcal{O}'$ for all $\mathcal{O}'$ satisfying $p(\mathcal{O}')$ + $q(\mathcal{O}')$ = p($\mathcal{O}$\_$\tau'$) + q($\mathcal{O}$\_$\tau'$). Let's call this inequality as Virtual-Eq (24).

Since the original setting in the paper is p($\mathcal{O}$\_$\tau'$) + q($\mathcal{O}$\_$\tau'$) ) = p($\mathcal{O}$\_$\tau$) + q($\mathcal{O}$\_$\tau$) (recall that  $\mathcal{O}=\mathcal{O}$\_$\tau'$), it is obvious that $\mathcal{O}$\_$\tau$ is one of $\mathcal{O}'$ that satisfies $p(\mathcal{O}')$ + $q(\mathcal{O}')$ = p($\mathcal{O}$\_$\tau'$) + q($\mathcal{O}$\_$\tau'$). 

Therefore, from the original Eq (24) and the Virtual-Eq (24), we obtain the following: 

Integral-over-$\mathcal{Q}$\_$\tau$ max\{p(x), q(x)\} dx = Integral-over-$\mathcal{Q}$ max\{p(x), q(x)\} dx, for all $\mathcal{Q}$ satisfying  $p(\mathcal{O})$ + $q(\mathcal{O})$ = p($\mathcal{O}$\_$\tau$) + q($\mathcal{O}$\_$\tau$). 

That is, Eq (24) holds only at equality, and it cannot imply $\hat{\epsilon}$ ($\mathcal{O}$\_$\tau$; p, q) $\geq$ $\hat{\epsilon}$ ($S$ p, q) for all possible choices of output set $S$.

Hence, the conclusion given by Section 4.2 IDENTIFYING OPTIMAL OUTPUT SET FOR AUDITING is incorrect. The paper does not provide the theoretical claims or methodological support for the assertion that the proposed approach can identify or approximate the optimal output set. 

Comment 2: 

Even if Theorem 4.3 shows some reasonable inequality-based conclusion, the conclusion only applies for the output set satisfying $p(\mathcal{O})$ + $q(\mathcal{O})$ = p($\mathcal{O}$\_$\tau$) + q($\mathcal{O}$\_$\tau$) for a given tau, and cannot be directly generalized to all possible output set $S$.


Comment 3:

In addition, the choice $\tau$ of the proposed approach seems to be heuristic or arbitrary. That is, the paper does not show how to choose $\tau$. Since the $\tau$-log-likelihood-ratio-set output set $\mathcal{O}$\_$\tau$ depends on the choice of the threshold $\tau$, the optimality of $\mathcal{O}$\_$\tau$ in general depends on $\tau$. Any related threshold-based results, claiming to be optimal without characterizing the optimality of the $\tau$, is problematic and not rigorous.

Other Comments:

Is $p(\mathcal{O})$ + $q(\mathcal{O})$ = $\tau$ above Theorem 4.3. a typo? 

It is unclear how the proof of Theorem 4.3 is related to the Neyman-Pearson lemma.

If the mechanism is the training process of a machine-learning model, then does each empirical sample used in Algorithm 1 require a run of the training process? The authors should discuss the related computational costs and complexity to approximate the densities.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the problem of auditing differential privacy (DP), which involves finding the maximum privacy loss across all possible outputs and possible adjacent datasets. The authors propose a new approach for DP auditing that can be applied with either white-box or black-box access to the mechanism. This approach aims to improve the efficiency and accuracy of DP auditing by optimizing over output sets.

### Strengths
The paper identifies an interesting problem in the DP-auditing community. Since the output space can be arbitrarily large, directly finding an output set where the log-ratio is maximized can reduce the number of samples needed.

### Weaknesses
**Conceptual confusion:** 
- A key weakness of the paper is that it blurs the lines between two distinct concepts: accounting and auditing. For example, Figure 2 wants to provide an improvement over methods that only have black-box access which seems unfair. The authors' method, with access to the distributions, could directly perform accounting and find the exact privacy bound. However, they compare it to black-box access methods that aim to find a lower bound to verify if a mechanism meets a given guarantee. This comparison seems unfair.

- The assumption on access to the densities questions the need for auditing: Having access to the distribution facilitates directly estimating the measure of the set where the density ratio is large, without the need for sampling. 

**Misleading claims**:
- Line 69-70 is false. In [4], the authors propose 3 new lower bound methods using only samples.  that do take into account the approximation error. The bounds can be tight for certain distributions. Since a priori the auditor has only black-box access then one cannot relax this.  DP-sniper also incorporates approximation error. 
- Not all DP auditing techniques require explicitly finding an optimal output set. Some approaches can indirectly compute privacy bounds using the divergence definition of DP and empirical estimates.

**Limited scope:** 
- The paper focuses on DP-SGD, and not more general mechanisms (e.g. exponential mechanisms, histograms, or the sparse vector technique). Their only motivation is computational, but not all mechanisms require training a machine learning model. E.g., reporting the number of COVID cases, counts and aggregates, census statistics, etc. 

**Missing references:** 
- Introduces an auditing technique based on a regularized renyi divergence. 

[1] Domingo-Enrich, C., & Mroueh, Y. (2022). Auditing Differential Privacy in High Dimensions with the Kernel Quantum R\'enyi Divergence. arXiv preprint arXiv:2205.13941.

- Develops upper and lower bounds with white-box access (as this paper assumes). 
 
 [2]Doroshenko, V., Ghazi, B., Kamath, P., Kumar, R., & Manurangsi, P. (2022). Connect the dots: Tighter discrete approximations of privacy loss distributions. arXiv preprint arXiv:2207.04380.

- Develops a statistical test with an approximation error that finds lower bounds on DP parameters:

[3] Property testing for differential privacy
Gilbert, A. C., & McMillan, A. (2018, October). Property testing for differential privacy. In 2018 56th Annual Allerton Conference on Communication, Control, and Computing (Allerton) .

- Introduces three novel tests based on approximations of the renyi, hockey-stick and MMD divergences. All include approximation error: 

[4] W. Kong, A. M. Medina, M. Ribero and U. Syed, "DP-Auditorium: A Large-Scale Library for Auditing Differential Privacy," IEEE Symposium on Security and Privacy (SP), 2024.


- What are the propositions related to Yeom et al. (2018); Jayaraman & Evans
(2019); Steinke et al. (2023). And how is Proposition 1 a generalization of this? 

**Unclear claims and terminology:**
- The use of the term “MCMC samples” is unclear. Are these simply samples from the mechanism, or is there an underlying Markov chain Monte Carlo method being used? The authors should clarify this terminology. 

- The concept introduced in line 51 seems to refer to empirical privacy metrics, which differ from the goal of privacy auditing. Auditing aims to verify whether a mechanism violates its claimed privacy guarantee, rather than assessing the tightness of the bound, as is done in membership inference attacks (MIA) or other inference attacks.


- L.70 the authors say “provides a gain”, a gain in what? Estimation accuracy? 

- The second bullet in “Summary of results” seems a direct consequence of the definition of approximate DP and/or DP definition, i.e.,  finding a set where the ratio is large. The same comment applies to Theorem 4.3 seems to be a direct consequence of the definition of DP. 


**Questionable Experimental Setup**
- Numerical experiment in 5.2 seems a bad comparison since the authors assume access to the distributions, which allows for direct computation of the exact epsilon, rendering the auditing process unnecessary.
- Experiments have limited scope, testing  only for laplace or gaussian mixtures/distributions and limited comparison to previous work.

- Minor:
  - This sentence could be made clearer:
  - “It ensures that the presence of any individual datum will not be revealed from the probability of any outcome.” A more precise statement is that DP ensures that the presence or absence of any record will not be revealed from the outcome of the mechanism. 

- Typos:
  - L145, T: D\in \theta, should be capital \Theta. 
  - L 160: “Subselect the scores by a output set”
  - L. 403 “Chi-squared distributions p and q in Appendix D”

### Questions
1. Can the authors clarify what they mean by the statement in line 72 that the 'effect of output event set choice on the privacy auditing power is distribution-dependent'? This seems to suggest that the impact of the chosen output set on the effectiveness of the audit depends on the specific mechanism's output distribution. However, in a typical auditing scenario, the auditor does not have a priori knowledge of this distribution. How can one determine the need for finding an optimal output set without such knowledge?

2. What do the authors refer to by MCMC samples? (line 64 and later in the manuscript)

3. Could the authors provide a practical scenario where an auditor has access to the probability densities but would choose to sample from them instead of directly computing the probability ratio for auditing?

4. It is known ([3], [4])  that finding the optimal output set can require exponentially many samples in the worst case. Can the authors elaborate on how their proposed method addresses this potential bottleneck?

5. The authors state in line 77 that 'whether and when optimizing worst-case output sets is elusive.' While this is true for arbitrary distributions, there are cases, such as Gaussian mechanisms, where characterizing these sets is possible. This challenge was a key motivation for developing alternative DP notions like Rényi DP, which provide a smoother measure of privacy and avoid the reliance on worst-case output sets with small measures. Could the authors comment on the connection between their work and these alternative DP notions?

6. Figure 2 suggests that DP-sniper has better outcomes while exhibiting higher variance. In practice you could run several tests and selecting the maximum lower bound could yield better results than the suggested approach (that uses white box information). Could the authors explain the advantages of their approach?

7. Why does Figure 3 compare to DP-Sniper if this method is only for pure DP?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a framework for improving the accuracy of differential privacy auditing based on trying to identify which canary scores to keep and which to ignore then computing an empirical DP lower bound. Their algorithm efficiently identifies this set when output distributions are known and approximates it from empirical samples when they are not. Experiments on synthetic and real-world datasets, including black-box DP-SGD training, demonstrate that their approach consistently tightens privacy lower bounds compared to existing techniques. The gains are particularly significant with limited auditing samples or when the output distribution is complex (e.g., asymmetric or multimodal).

### Strengths
* The problem is interesting and important
* The experimental setups considered cover a good range of problem complexities

### Weaknesses
* The manuscript lacks clarity in many places

- The problem statement needs to be more clearly explained, in particular the inner workings of the prototypical auditing experiment. What distribution is the dataset sampled from? How does working with many in- and out-points (usual in MIA) apply to the DP auditing problem where we usually consider two fixed datasets differing in a single point? How does subselection interact with a given auditing function L?

* Focus on pure DP auditing is quite limiting, especially in the context of DP-SGD examples. The manuscript mentions this is for simplicity, but it's unclear from the current presentation whether the methods extend easily or need significant changes.

* Although the manuscript claims "there is no existing consensus regarding the optimal way to incorporate the approximation error with the objective of privacy auditing", there are indeed works that try to incorporate the effect of finite-samples into privacy auditing like [1] and [2]. The manuscript should discuss and compare these methods with the proposed approach.

[1]	William Kong, Andrés Muñoz Medina, Mónica Ribero, Umar Syed: DP-Auditorium: A Large-Scale Library for Auditing Differential Privacy. IEEE SP 2024
[2] NASR, M., SONGI, S., THAKURTA, A., PAPERNOT, N., AND CARLINI, N. Adversary instantiation: Lower bounds for differentially private machine learning. IEEE SP 2021

### Questions
* Is the proposed method only applicable to DP mechanisms for ML? Or can it be applied to more general DP mechanisms? (from the presentation it seems to be the former, but it's unclear why)
* What is the Markov Chain in the MCMC sampling refered to in L64-65?
* In Th 4.3: why is there a greater-or-equal rather than equal (since O_tau is feasible)?
* In Fig 1: why is max(p(o),q(o))/(p(o)+q(o)) a relevant quantity to look at?
* In Eq 7: why are the losses l1 and l2 not shuffled using the same permutation as x1 and x2?
* L340: why is a heuristic designed for auditing with a single training run on a large dataset with many "canaries" appropriate to use when auditing with many runs on a 2-point dataset?
* "Let p and q be two probability distributions for member and nonmember scores in the auditing experiment respectively" - what is the randomness over in this distributions? E.g. dataset sampling, mechanism randomness? Without making this more concrete it is not possible to verify the proofs in the supplementary.

### Soundness
3

### Presentation
1

### Contribution
2
