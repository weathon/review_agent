# Block-Sample MAC-Bayes Generalization Bounds

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
We present a family of novel block-sample MAC-Bayes bounds (mean approximately correct). While PAC-Bayes bounds (probably approximately correct) typically give bounds for the generalization error that hold with high probability, MAC-Bayes bounds have a similar form but bound the expected generalization error instead. The family of bounds we propose can be understood as a generalization of an expectation version of known PAC-Bayes bounds. Compared to standard PAC-Bayes bounds, the new bounds contain divergence terms that only depend on subsets (or \emph{blocks}) of the training data. The proposed MAC-Bayes bounds hold the promise of significantly improving upon the tightness of traditional PAC-Bayes and MAC-Bayes bounds. This is illustrated with a simple numerical example in which the original PAC-Bayes bound is vacuous regardless of the choice of prior, while the proposed family of bounds are finite for appropriate choices of the block size. We also explore the question whether high-probability versions of our MAC-Bayes bounds (i.e., PAC-Bayes bounds of a similar form) are possible. We answer this question in the negative with an example that shows that in general, it is not possible to establish a PAC-Bayes bound which (a) vanishes with a rate faster than $\mathcal{O}(1/\log n)$ whenever the proposed MAC-Bayes bound vanishes with rate $\mathcal{O}(n^{-1/2})$ and (b) exhibits a logarithmic dependence on the permitted error probability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
MAC-Bayes bounds (mean approximately correct)  are a  variation of PAC-Bayes bounds that are not necessarily valid with high probability but bound the expected generalization error.
The study considers „block-sampled“ MAC-Bayes bounds, in which the data is split into block/batches and the divergence term between prior and posterior is defined on block level.
The submitted paper studies the optimization of these „block-sampled“ MAC-Bayes bounds and whether they can be turned into
PAC-Bayes versions which improve on current PAC-Bayes results, which the authors show to be not possible in general.

### Strengths
I found the general topic of the study interesting.
I would like to stress that also the negative result is interesting.

### Weaknesses
* The „impossibility result“ in Theorem 2 can be viewed as a general form of the properly cited results by 

Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan. Formal limitations of sample-wise
information-theoretic generalization bounds. In 2022 IEEE Information Theory Workshop (ITW),
pp. 440–445. IEEE, 2022.

* I was wondering: how are the results related to the results by 

Recursive PAC-Bayes: A frequentist approach to sequential prior updates with no information loss
YS Wu, Y Zhang, BE Chérief-Abdellatif, Y Seldin
Advances in Neural Information Processing Systems 37, 17947-17971

which - in a different setting - consider a split of the training data and incrementally update the PAC-Bayes bound.

* Section 4: I was somehow not convinced by this example that should show the usefulness of the new bounds. The prior leading to (11) depends on the „batch size“ m. I accept that in PAC-Baysian analysis the prior is not necessary some form of „prior belief“ as in stared Bayesian analysis, but rather a tool to get tight performance guarantees. Still, that the prior depends on m does not feel right to me - where should such a prior come from? Could the authors discuss this in more detail?

The fact that the results are better than without blocks-sampling is then also heavily dependent on the prior (end of  Section 4). Why should this prior be used for the  m=n case? Is the prior real „prior belief“  here? If yes, where does it come from for the m-n case? If not - if it is a tool to get tight bounds- why this choice for n=m?


Minor comments:

* Authors forgot to introduce the prior Q_W when introducing PAC-Bayes in equation (1).
* Can the loss in the example 4 be viewed as some known robust loss function?

### Questions
Please see "Weaknesses" above.

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
This paper presents a mean posterior error bound that relates
$$\mathbb{E} \\{ d( \rho\\{ \hat L_n \\}, \rho\\{ \mathbb{E} L_n \\})  \\}$$
to
$$\sum_{j=1}^J \mathbb{E}\\{  D_{\text{KL}}(\rho_j\\|\pi) \\} + J \cdot \text{``per block CGF''}$$
where $\mathbb E$ is over the randomness of the observations, $\rho$ is the data-dependent posterior probability, and $\pi$ a prior probability; importantly, the novelty of the paper lies in $\rho_j$, the posterior probability *after observing the block #j*.

This implies novel generalization bounds that scale as
$$\sqrt{ \frac 1 n  \sum_{j=1}^J \mathbb{E}\\{  D_{\text{KL}}(\rho_j\\|\pi) \\}}$$
where the usual ``complexity term'' is broken down linearly into block components.

### Strengths
I found the block decomposition concept novel, and the authors did convince me that the technique drives tighter bounds.

Some might argue that this is an interpolation between the individual-sample ($m=1$) and the bulk-sample ($m=n$) regimes (borrowing the leave-one-out analysis technique, equality (b), from the former). However, the resulting bounds are tighter as a result of this interpolation, with the "optimal block rate" clearly explored in Section 5.

### Weaknesses
For those outside the PAC-Bayes community, Sections 1-2 do not provide a good introduction to the concepts involved. In particular, *what is $Q_W$*? PAC-Bayes people of course know this is the prior probability; however, even if explicitly mentioned, the role of this prior would usually still be quite confusing for generic readers. The authors did even less in this regard by expending 0 words on $Q_W$, and how the evolution to $P_{W|S_j}$ is to be construed.

Very minor notation issue: $\mathbb E_{P_S}$ vs. $\mathbb E_{S}$.

### Questions
In general, how does the quantity

$$K_{n} := m^{-1}\mathbb{E} \\{ D_{\text{KL}} (\rho_m \\|\pi) \\}$$

evolve as a sequence of $m$? Here $m$ agrees with the definition in your paper (i.e. block size), $\pi$ is a prior probability, and $\rho_m$ is the posterior probability evolved from $\pi$ after seeing $m$ i.i.d. observations. I believe this is important to better conceptualize the benefits of the block decomposition in bounds e.g. (10). Some graphical exploration would be helpful.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the MAC-Bayes bounds. They proposed a family of novel block-sample MAC-Bayes bounds, which generalized the MAC-Bayes bounds in the sense that it aggregates the contributions of a collection of disjoint subsets of the sample (on the MAC-Bayes bound) instead of consider the sample a whole. They proved the convergence of the block-sample bound and showed its superior characteristic power compared with the original PAC-Bayes bounds in a simple application-Gaussian mean estimation. At last, the paper discussed how the block size could affect the convergence and the possibility of transforming the block-sample bound into a high probability form.

### Strengths
* Although I did not verify all the proofs in detail, they appear to be sound overall.
* While not significant, the block-sample bound seems to be a non-trivial generalization of the prior results.

### Weaknesses
I think the authors should put more effort in the presentations of this paper because of the following reasons:
* The authors throw out the definition of PAC-Bayes bounds at the very beginning, however, without giving enough explanation for the term in the expression. For instance, I do not see any description of $Q_W$ and have trouble in understanding what it means. Also, I advise the author to give a concrete example of $I(n,d)$ in addition to just saying it is proportional to $n$ and $d$. Similarly, instead of just saying what $P_{W|S}$ does in general, give a concrete example to demonstrate it.
* The authors introduced the "individual-sample bound" in describing their first contribution but did not even explain what it is.
* From my perspective, the authors did a poor job in motivating the audiences on their contributions. The author only briefly mentioned that their result is a generalization of the MAC-Bayes bound and individual-sample bound (which they did not define), they did not emphasize the significance of their generalization. In particular, the authors did not well explain what advantages this generalization can bring upon the original MAC-Bayes bound in the contribution part. 
* While the author gave an example on mean estimation of Gaussian in Section 4 to demonstrate the usefulness of the block-sample bound compared to the PAC-Bayes bound, I am not convinced by this example due to its overly simplicity. In particular, the PAC-Bayes bound exhibits a poor generalization for this application not because this task is hard. In fact, a simple Hoeffding bound can give a tight error bound on the mean estimation. Therefore, this example is not enough to demonstrate the significance of the generalization. With that being said, I still suggest to move this example into the introduction to give a more fluent and motivative presentation.
* In Section 3, it would be more accessible for the audience if the author could briefly explain the assumption in the main theorem in plain language and sketch their proof idea of the theorem in advance.

I will consider raise my rating if the authors address my concerns.

### Questions
* What does the assumption (3) in the main theorem implies intuitively?

### Soundness
3

### Presentation
1

### Contribution
2
