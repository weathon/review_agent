# Revisiting inverse Hessian vector products for calculating influence functions

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
Influence functions are a popular tool for attributing models' outputs to training data. The traditional approach relies on the calculation of inverse Hessian-vector products (iHVP), but the classical solver ``Linear time Stochastic Second-order Algorithm'' (LiSSA, Agarwal et al. (2017)) is often deemed impractical for large models due to expensive computation and hyperparameter tuning. We show that the three hyperparameters --- the scaling factor, the batch size, and the number of steps --- can be chosen depending on two specific spectral properties of the Hessian: its trace and largest eigenvalue. By evaluating them with random sketching (Swartworth and Woodruff, 2023), we find that the batch size has to be sufficiently large for the LiSSA to converge; however, for all of the models we consider, the requirement is mild. We confirm our findings empirically by comparing to the Proximal Bregman Retraining Functions (PBRF, Bae et al. (2022)).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper considers inverse Hessian-vector products for the purpose of evaluating the influence of training data $(x_m, y_m)$, a data-label pair, on fitted parameters $\theta^*. 

For application to empirical problems, in particular language models, the Hessian is replaced by the Gauss-Newton matrix, and a regularized inverse-Hessian product is considered $(\lambda + H)^{-1}$. Extensive details are provided on the numerical considerations taken in evaluating Hessian-vector products.

An algorithm, LISSA, for estimating the Hessian-vector-product on a training point is considered.  It is effectively a version of minibatch SGD on a quadratic problem (8), where the gradient estimator is specific to this problem instance.  A convergence analysis in expectation and norm-convergence (up to a neighborhood) is provided.  These are taken under a relatively strong norm-type assumption on the minibatch gradients (C.1). 

This norm-type assumption is discussed in theory in the appendix in two settings and also empirically.

A table of hyperparameters are provided for a range of reasonable setups (Table 1), which (appear to be) conforming to the assumption C.1.  These require estimating the trace and operator norm of the hessian, and some discussion of the sketching procedure taken to do this is provided.

The convergence of LISSA is tested with these hyperparameters, to show good performance, and the influence is measured against a second method (PBRF).

Finally some discussion of the value of inverse-Hessian influence (vs gradient-based influence measures) are provided in Section 5.

### Strengths
The paper does well to describe the end-to-end process of using inverse-Hessian influence metric on LLMs and Resnets, for which there are substantial empirical and theoretical questions to address.  It would appear that the cases that the authors considered are evaluated in reasonable level of detail, and that the empirical aspects of the paper should be reproducible.

While the main contributions are about methodology of using inverse-Hessian influences; section 5 provides some additional scientific value on the potential value of using inverse-Hessian influences.  While it is more speculative in nature, it provides some problem setups which might be the setting for future work.

The paper is also well-integrated into existing theory on influence functions, which brings additional value to the paper.

The main theorem seems to provide a suggested starting point for how to pick the parameters in LISSA (with explicit parameters).

### Weaknesses
I find the following overall weaknesses: 

1) The main theoretical contribution of the paper is a convergence analysis for the mini-batch SGD derived.  This is done under a very strong hypothesis, which is effectively a contractivity assumption in norm (C.1) or (9) in Theorem 1.  This makes the main theorem relatively easy to prove, but it also begs the question of showing when this satisfied.   Separately, mini-batch SGD on quadratics are studied to death, and it will be quite difficult to argue that this is truly a new analysis

2) Regarding the contractivity assumption: it is shown that in some sense the assumptions of Theorem 1 necessary, but this sense is effectively adversarial in the test-direction u (B.4Lemma 1).     On the other hand, the positive cases (B3 Motivations for C.1) do not show that C1 is satisfied.  It would substantially improve the paper if the theory covered these two cases.  (If it does, this should be explained).

3) The empirical results of the contractivity (Figure 4) do not consider C.1 but rather the trace.  Trace type conditions are much easier to satisfy, and unfortunately give little insight whether C.1 in fact holds.  A much more desirable theorem (to my taste -- and also more consistent with the experiments) would assume trace-type conditions on H and H^2, and some randomness assumptions on u that lead to the claimed convergence.  There is a lingering sense that C.1 and (9) of Theorem 1 is unverifiable.

4) The empirical conclusions in Section 4 and 5 are a little bit murky, and it's hard to align what appears in the simulations with what appear in the text.

### Questions
Regarding the weaknesses:
Weakness 1): There's a lot of batch-SGD analyses which at first glance may already contain the theorem.  For example:
A) Jain, Kakade, Kidambi, Netrapalli, Sidford. "Parallelizing.." 2018 JMLR
B) Varre, Pillaud-Vivien, Flammarion. "Last iterate convergnece.." 2021 NeurIPS
C) Ma, Bassily, Belkin.  "The power of Interpolation..." 2018 ICML
Can you argue why your analysis is novel, comparing to existing literature like the above?

Weakness 2): Can you show why the theorems apply to the setups in B3? Currently, this is `almost' done but not completely.  In particular the sentence on (l243): ``we find this condition to hold for classification with independent sampling" appears unfulfilled."

Weakness 3): Can you show empirically that C1 occurs, in the norm sense?  (tracial vs norm-type behavior is rarely equivalent in random matrix contexts).

Weakness 4a): Figure 1 is appears to be discussed in the text (l368-370) and 3 cases are explained. Considering Figure 1, why are these the 3 cases, (especially, which illustrates case '3'? Where do you argue that 'PBO _finetunning_ stirs away the model too far for the quadratic approximation to hold'. (and what does that mean?)

Weakness 4b): Figure 3.  Why does the difference picture show what you claim it does?  (In particular why do you average over the 10x10 square, and why are you not measuring the diagonals?). When you write (l464-466). ``As a result the influence similarity between an orignal sentence and a rewritten one appears to be consistently higher than between unrelated sentence."  Can you explain this?

Other questions: 
1) Can you explain precisely how you picked the 'batch' in Table 1, and moreover, what is the general procedure you suggest for selecting LiSSA parameters?  (It seems you _almost_ do this, but being extra-clear would help, especially as regard the batch.)

Minor:
1) (l80-83). ``Influence functions are calculated under the assumption that the optimized parameter $\theta$ of the model delivers minimum to the training loss."  This sentence is mangled. 
2) l409-410.  ``the choice of _dumping_ parameter''
3) l424: ``For example, a plausible interpretation of the directions $v_j$ would be that the top directions correspond to general language coherence and... small eigenvalues could correspond to more specific, informative content."  While the statement about large eigenvalues makes sense, the complementary statement about small eigenvalues does not -- first most small eigenvalues are not interpretable, and are the result of either diffuse (non-one-dimensional) embeddings or undesirable feature distortions.  Perhaps it would be better to just say you can screen out the large eigenvalues which are more about general language coherence.

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
4

### Summary
This paper studies influence functions, which are a popular tool for attributing models' outputs to training data. This reduces to calculation of inverse Hessian-vector products (iHVP) using the classical "Linear Time Stochastic Second-order Algorithm" (LiSSA) method. The authors show that the two specific spectral properties of the Hessian, namely its trace and largest eigenvalue, determines the hyper parameters like the scaling factor, the batch size and the number of steps. So they evaluate these two quantities using random sketching. The conclusion is that batch size has to be sufficiently large for the LiSSA to converge.

### Strengths
- The authors show that the two specific spectral properties of the Hessian, namely its trace and largest eigenvalue, determines the hyper parameters like the scaling factor, the batch size and the number of steps. So they evaluate these two quantities using random sketching. The conclusion is that batch size has to be sufficiently large for the LiSSA to converge. 

- code is open-source shared

### Weaknesses
Overall I think the paper looks good.

### Questions
Since this topic is about hyerparameter tuning, can we connect the proposed theory/results with iterative gradient-based optimization methods? More specifically, I know that for optimization algorithms like GD, its behavior will be influenced by the Hessian near equilibrium (i.e., the condition number of Hessian). So I am curious if this phenomenon is related to what is proposed in this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The submission presents an analysis of the convergence of SGD for solving the inverse Hessian-vector-product, useful in the context of computing influence functions. The submission presents convergence results that depend on statistics of the Hessian and the batch size, and show that using those to set hyperparameters to achieve a desired accuracy leads to good performance, automating the choice of hyperparameters.

### Strengths
That the convergence theory for SGD can automate hyperparameter selection is a nice observation that is likely to be helpful to the community.

### Weaknesses
The main weakness is the first contribution of the submission, the convergence analysis in section 3. The presentation of the results ignores previous work on stochastic optimization and the description of the results lacks clarity. The main novelty would be the boxed condition C.1. However, it is unclear why this bound holds, as there is no theoretical explanation nor empirical validation provided in the main text. See detailed comments in Questions below.

### Questions
**Issues with the theoretical results**
- The submission does not address the large optimization literature on stochastic gradient descent, which has derived many similar convergence guarantees. The rates are not typically written in terms of a linear + constant term, as this is not convergent, and the literature is typically concerned with decreasing step-sizes (see e.g. [Bach and Moulines, 2013](https://arxiv.org/pdf/1306.2119)) or other schemes that can guarantee convergence. But the linear + constant result still appears regularly, for example see Theorem 2.1 in the work of [Needell et al., 2015](https://arxiv.org/pdf/1310.5715), Theorem 1 in the work of [Dieuleveut et al., 2016](https://arxiv.org/pdf/1602.05419), of Theorem 5.8 in the handbook of [Garrigos and Gower](https://arxiv.org/pdf/2301.11235). Given those results, the additional insights provided by the current submission appears small, but I might have missed a point that is specific to the inverse Hessian-product setting. If so, a discussion of existing literature in optimization and highlighting what makes the iHVP problem special would make the submission stronger.

- The text uses the word "converges" loosely, e.g. in corollary 1, "the algorithm converges in $T = \Omega(1/(\eta\lambda))$ steps". Taken literally, this is wrong, as the algorithm is not guaranteed to return $u^t = u^*$ after $T$ steps. I assume this is meant to say that the approximation error is bounded by some quantity after $T$ steps, but it isn't clear from the current writing. Changing the statement to a typical statement of "to achieve an error of $\frac{1}{2}\| u^t - u^*\|^2 \leq ...$, the algorithm should be run with parameters $\eta, B = ...$ for $T \geq ...$ iterations" would make it clearer. If the number of iterations is an upper bound, tt should also be $O$ rather than $\Omega$.

- The condition C.1 seems convenient but it isn't clear from the main text why it is introduced, or why it should hold. Extracting the key argument from the appendix to the main text to provide an explanation for the condition and why it is expected to hold would help readers understand the contribution. That the covariance scales with the trace appears to imply/assume that the covariance and the Hessian are the same, as in the toy quadratic model of [Zhang et al.](https://proceedings.neurips.cc/paper/2019/file/e0eacd983971634327ae1819ea8b6214-Paper.pdf) (see §3.5), and a citation to other people using similar assumptions might help make the case.

- The text "confirms the inverse scaling with batch size". Since the LHS of (C.1) is a variance term, the scaling as $\propto 1/B$ does not seem to need confirmation. The assumption that it is scaling with the trace however does seem to warrant empirical evidence. The scaling with the trace of the Hessian is the less intuitive part of the condition, and adding evidence for the scaling would strengthen the claim. 

**Writing**
The writing would benefit from an additional pass to improve the clarity of the message. Some of the statements appear wrong or misleading and the text contain typos or broken sentences. I only list a few examples below.

- Re: "Koh and Liang (2017) proposed a stochastic iterative approach called [...] (LiSSA, (Agarawal et al. 2017))". A more accurate description would be that Koh and Liang propose to _use_ the LiSSA algorithm of Agarawal et al., rather than propose the algorithm.
- Description such as "Basu et al. (2020) criticize the LiSSA [algorithm], suggesting that it lacks convergence for deep networks" and "we carefully analyze the convergence of the Lissa [algorithm]" should be made more specific. It is not clear whether this is refering to theoretical properties (will the algorithm actually converge to the solution if run sufficiently long) or whether the term "convergence" is used to refer to empirical performance, closer to "suggesting that the method performs poorly on deep networks" and "we analyze the [empirical performance] of the LiSSA algorithm as a function of its hyperparameters". 
- The sentence "SGD is known to work at least as well as the full gradient descent, even in terms of the number of updates (Harvey et al., 2019)" is misleading. Harvey et al. treat the non-smooth convex case, which is significantly different from the quadratic case considered here. The sentence should be removed.
- The equality in Eq. (3) should be replaced by an $\approx$ as it is the result from a truncated Taylor expansion
- L27-28, "interpret the internal computations in an understandable to a human way"
- "Generally speaking this is different from the empirical FIM [...], however, for low noise distributions the two might be used interchangeably (Martens, 2020)". I do not recall Martens making this argument, and it is not clear what is meant by "low-noise distribution".

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Influence function captures how the optimal model weights change with respect to a newly added data point and has the form of an inverse Hessian-vector product. LiSSA is classical solver to compute the influence function which essentially applying stochastic gradient descent to a quadratic objective. This paper proposes conditions to choose the hyperparameters (proximal hyperparameter, batch size, step size) of the LiSSA algorithm for computing the influence functions of a model to a newly added data point. The conditions guarantee the convergence of LiSSA in expectation. The convergence analysis shows that the batch size has to be sufficiently large in order for the LiSSA algorithm to converge.

### Strengths
- The paper is well-organized and clearly written
- This paper gives the conditions on the hyperparameters for the LiSSA algorithm to converge. Such results are missing in the current literature.

### Weaknesses
- The contribution is somewhat limited because the convergence analysis seems to be standard, and the algorithm is only for computing the influence function.
- Some results in section 4 (Figure 1) seem to indicate that, in some situations, the influence function is not a good indicator of how the sensitive the optimal weights are with respect to newly added data points, as the LiSSA influence and PBRF influences are not correlated at all.
- The paper can be strengthened by reviewing more related work. E.g., are there other related algorithms for computing the influence function? Are there other variants of the influence function? Other work that analyze inverse Hessian-vector products (which are listed in the conclusion)? This can show the paper's position within the broader literature

### Questions
- In section 5, the paper claims that the idea of inverse Hessian makes sense because small eigenvalues correspond to more specific and informative content. However, in traditional interpretation, small eigenvalues correspond to noise and carry little to no information. What is it not the case here? Moreover, are there references for the claim that large eigenvalues correspond to general language structure?
- The Hutchinson estimation and randomized sketching are used to estimate Tr(H) and lambda_max. In the experiments, how many Hessian-vector products are needed to get a good estimation?
- In Figure 3, lambda=5 is used, which seems to be large given that the GNH matrix in (7) is already SPSD. How does the results change using a smaller lambda?

### Soundness
2

### Presentation
3

### Contribution
2
