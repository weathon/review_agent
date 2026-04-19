# Leveraging Flatness to Improve Information-Theoretic Generalization Bounds for SGD

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
Information-theoretic (IT) generalization bounds have been used to study the generalization of learning algorithms. These bounds are intrinsically data- and algorithm-dependent so that one can exploit the properties of data and algorithm to derive tighter bounds. However, we observe that although the flatness bias is crucial for SGD’s generalization, these bounds fail to capture the improved generalization under better flatness and are also numerically loose. This is caused by the inadequate leverage of SGD's flatness bias in existing IT bounds. This paper derives a more flatness-leveraging IT bound for the flatness-favoring SGD. The bound indicates the learned models generalize better if the large-variance directions of the final weight covariance have small local curvatures in the loss landscape. Experiments on deep neural networks show our bound not only correctly reflects the better generalization when flatness is improved, but is also numerically much tighter. This is achieved by a flexible technique called "omniscient trajectory". When applied to Gradient Descent’s minimax excess risk on convex-Lipschitz-Bounded problems, it improves representative IT bounds’ $\Omega(1)$ rates to $O(1/\sqrt{n})$. It also implies a by-pass of memorization-generalization trade-offs. Codes are available at [https://github.com/peng-ze/omniscient-bounds](https://github.com/peng-ze/omniscient-bounds).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work introduces a novel information theoretic bound. As is common for bounds of this type, the starting point is a well-known result that states that the generalization error can be bounded in term of the mutual information (MI) between the training set $S$ and the learning algorithm’s output $W$. For SGD algorithms, under Gaussian weight priors and Gaussian noise injection on the final iterate, this boils down to a generalization bound that consists of a flatness term and a trajectory term (see e.g. Neu et al. and Wang et al.). 

Some existing MI bounds of this form have a crucial defect: they increase in the batch size even though small batch SGD generally converges to flatter minima due to larger gradient noise. This defect is purported to be caused by the hidden flatness-dependence of the trajectory term in those MI bounds, though this wrong tendency can be addressed with i.i.d. per-iterate auxiliary noise injections (see Neu et al.). 

The main contribution of this paper is to improve upon existing MI bounds such as Neu et al. and Wang and Mao via a novel technique. More specifically, the authors of this work conduct a more fine grained analysis wherein, rather than choosing the optimal noise covariance based on the expected trajectory $W$, one chooses the best noise covariance for the specific instances of $W$, and then calculates the expected error bound over $W$. By doing so, one can a) sidestep an overfitting issue with estimating the best noise covariance based on (expensive/scarce samples of) $W$; and b) get a numerically tighter bound by virtue of exchanging the $\inf\mathbb E$ with the $\mathbb E \inf$ operator.

This approach, based on the construction of an "omniscient trajectory", is general enough so that it can be applied to other existing methods. Moreover, it is shown to be empirically tighter than existing, comparable bounds, and it is shown to be theoretically optimal (up to constants) for SGD on Convex-Lipschitz-Bounded (CLB) problems for a specific choice of training iterations and step size.

### Strengths
- The proposed approach is sound and very effective in achieving the goal of better modelling the effect of flatness on generalization. For example, it achieves a decrease of the trajectory term on the order of $10^6$ (multiplicative) in the case of CIFAR-10 over the one proposed by Neu et al.. This is a valuable contribution.
- The paper is overall well-written.
- Like the bound of Neu et al., the proposed bound has the right scaling w.r.t. the batch size, while also being numerically tighter.
- In fact, the bound is numerically tight in a very simple setting (MNIST + 2-layer MLP). 
- The omniscient-trajectory technique is to a degree general purpose and can be applied to a whole class of similar bounds.

### Weaknesses
- While the bounds is quite tight for MNIST, it seems that it is numerically vacuous (i.e. larger than 1 for the 0-1 classification accuracy) already for CIFAR10. So while it is numerically tighter than comparable bounds, it does not provide a major qualitative improvement over bounds such as the one by Neu et al.. 
- In my view, even if one lets go of requiring tightness of a generalization bound, the next most important goal is to have the right tendency/correlation w.r.t. to the parameters of the learning setup. However, it is my impression that this bound does not “fix” any of such qualitative defects of existing comparable bounds, whereas, for example Neu et al. "fixes" the scaling w.r.t. batch-size and other flatness bounds improve upon prior ones by enforcing invariance w.r.t. rescaling the parameter vector.
- Empirically, applying Theorem 2 is difficult and requires multiple assumptions/heuristics, e.g.:
    - Crucially, multiple population means have to be estimated from i.i.d. samples of $W$, which is difficult. Given a training set of fixed size $N$, one needs to partition it by averaging of $k$ subsets of size $N/k$, yielding a bound for SGD on training sets of size $N/k$. If one heuristically were to apply that obtained bound to the output of SGD over the full training set $S$, it must be quite loose unless $S$ is highly redundant, which is typically not the case.
    - The bound has a hyperparameter $\lambda$ that regulates a trade-off between different terms in the bound and at the same time must upper bound the spectral norm of the Hessians
    - The bound assumes exactly zero training error at the final output. Thus it does not apply in the case of early stopping, and the application to approximately zero training error outputs requires a small leap of faith.
    - In the case of unbounded losses, one must clip the loss function in order to guarantee the sub-Gaussian concentration required for the analysis.
- The result gives a bound on the expected generalization error. This is qualitatively weaker than a high probability upper bound. Arguably, this is a rather minor weakness since deep learning models tend to be exceptionally stable w.r.t. different initializations and draws of the SGD batches.
- These kinds of bounds apply to the generalization error in terms of the expected surrogate loss, rather than the $0-1$ which is the real measure of interest in classification. I acknowledge that this is essentially unavoidable here in order to link generalization to the optimization trajectories, which are obtained from minimization of the surrogate loss. However, it should be noted that this makes the analysis inherently more loose since the surrogate risk is always an upper bound on the $0-1$ risk (i.e. the test accuracy).

### Questions
- There exist PAC-Bayes based generalization bounds that are non-vacuous even on ImageNet, while yours seems to be vacuous already for CIFAR10. How do these bounds compare conceptually to your type of bound? Perhaps they have in some sense an “unfair advantage”?
- What is the general tendency of your bound to changes in depth, width, training set size, and scaling of the full weight vector? If the tendency is in the wrong direction, is this shared by other bounds based on MI? I am asking because the bound being numerically tighter does not necessarily rule out the fact that it now scales in the wrong directions for some parameters.
- I suggest that you already introduce some of the notation around the equation in line 82. For example, it might not be clear to some readers unfamiliar with these results that $g_t$ denotes the gradients.
- Could you please expand on the difficulties of evaluating the bound in practise, specifically on which estimation steps you consider the most loose ones? How does the hardness/accuracy of estimation compare to that of the bounds of Neu et al. and Wang and Mao, for example in terms of how the training set has to be partitioned to carry out all the required estimations in a sound manner, and the trade-offs induced by the specific choices of such partitions? 
- I would suggest that you include some of the most important aspects of empirical evaluation in the main part of the paper, since this might be considered more important to some readers than some of the more technical/theoretical details.
- Related to the above question: if there are indeed particularly loose estimation steps, do you perhaps have some high level ideas on how to address these inadequacies in future work?
- I see in Appendix C.1 that for your experiments, you evaluate the penalty terms over the test set, which should make the bound incomparable to some of the other competing generalization bounds which work only on the training set. If you disagree with this statement, could you please elaborate? In my view, a fair comparison would require that you set aside a part of the training set for such estimation procedures.

============================================================================

Typos/minor comments:
- Line 34: perhaps "that" instead of "because"
- Line 932 "Hessians" -> "Hessian"

=============================================================================

References:

- Gergely Neu, Gintare Karolina Dziugaite, Mahdi Haghifam, and Daniel M. Roy. Information- theoretic generalization bounds for Stochastic Gradient Descent. In Proceedings of Thirty Fourth Conference on Learning Theory, pp. 3526–3545. PMLR, July 2021. ISSN: 2640-3498.
- Ziqiao Wang and Yongyi Mao. On the generalization of models trained with SGD: Information- theoretic bounds and implications. In Advances in Neural Information Processing Systems, October 2021.

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
This work belongs to the line of information-theoretic generalization bound and improves upon the results of [Neu et al., 2021] and [Wang and Mao, 2023]. In particular, it utilizes geometry-aware perturbation and establishes a novel generalization upper bound that better captures the influence of flatness-related terms and outperforms [Wang and Mao, 2023] by more accurately predicting the empirical observations. Moreover, the authors devise a more powerful auxiliary sequence called "omniscient trajectory" to bound the generalization error of the trajectory of (stochastic) gradien-based methods. As an application of the novel result, the authors derive the optimal minimax rate for GD under the CLB setting.

### Strengths
Overall, this is a solid and well-written paper. The motivation and intuition behind the proposed methodology are well-explained. It provides a refinement for the information-theoretic generalization bound by exploiting the geometric properties of the objective function and by devising a more powerful auxiliary sequence. This allows to better reflect empirical pbeservations and recovers the minimax rate when applying to CLB-based bounds. In this regard, this paper takes a step further towards the understanding of how local geometry might affect the generalization.

### Weaknesses
Despite a rather interesting paper, some concerns arise across different aspects.

First, the major concern comes from the novelty. The paper should be regarded as a refinement over [Wang and Mao, 2021] and the generalization bound is not new and decisively better compared to the prior results. Also, the methodology is also not completely original. For example, the use of anisotropic Gaussian perturbation also appears in previous works like [Neu et al., 2021], as mentioned by the authors themselves.

Second, it seems that the applicability of this result slightly suffers from the form of the iterative update in Sec 2.1 and ingredients like the population Hessian. For instance, the iterative update (it would be great if the authors provided a label for the update) might rule out the use of past gradient information and hence exclude many algorithms. It would be great if the author could provide some intuition to show that these ingredients are indeed necessary.

### Questions
The authors apply their result to CLB problem and obtain the optimal rate. I would be particularly interested if the author could show how their bound will imply a sample rate when the stochastic function is smooth and realiable, which is provably to have a better rate like O(1/n^{2/3}) or O(1/n) than the CLB setting from the recent upper or lower bound result. This might be unrelated to the major topic but I would greatly appreciate if the authors could answer this.

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
3

### Summary
This paper introduces a new approach to developing Information-theoretic Generalization Bounds for Stochastic Gradient Descent (SGD). The core idea is to incorporate an "omniscient" auxiliary trajectory into the proof, which better leverages the flatness of the loss landscape. Previous information-theoretic bounds struggled to align with generalization performance as batch size varied. This paper addresses that limitation, offering bounds that correlate more accurately with generalization across different batch sizes. Additionally, prior bounds for Gradient Descent (GD) on convex-Lipschitz-Bounded (CLB) problems could not achieve the optimal minimax rate of $O(\frac{1}{\sqrt{n}})$, but this paper’s approach successfully reaches this rate.

### Strengths
-The contribution is novel and significant: The "omniscient" auxiliary trajectory is original and allows to overcome 2 limitations of previous information-theoretic generalization bounds: 
1) correlation between the bounds and observed generalization behavior under different batch sizes
2) $O(\frac{1}{\sqrt{n}})$ rate for Gradient Descent on convex-Lipschitz-Bounded problems.

-The paper effectively motivates the problem and provides a useful high-level overview of the main ideas in the introduction.

-The paper includes rigorous proofs for the results, along with experiments that support the theory. However, I have not verified the mathematical details of the proofs.

### Weaknesses
1) The bound requires an additional penalty term to be introduced. This additional term could  worsen the bound in some applications if the benefits to the trajectory term are not high enough. Also, the proof involves two auxiliary trajectories (omniscient and SGLD-like) instead of one, complicating the proof and the bound.

2) In the introduction, it is postulated that the incorrect behavior of the previous bound in regards to varying batch sizes comes from the trajectory term not capturing the flatness more directly. However, in the experiments, the tendency of the new trajectory term w.r.t. learning rate is still incorrect. The overall bound seems to correlate correctly with generalization performance because the additional penalty term behave correctly and because this additional penalty term dominates the bound. It is therefore not clear to me if the paper provides the correct explanation in the introduction about what is ''really'' solving the problem.

### Questions
1) Do you think it would be possible to simplify the approach by using only one auxiliary trajectory (the omniscient one) in the proof?
2) Can you provide more clarifications about the point 2) in the weaknesses above?

### Soundness
4

### Presentation
2

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
This paper studies improved information-theoretic generalization bounds for SGD by more effectively incorporating the concept of flatness. Specifically, the authors point out limitations in previous information-theoretic generalization bounds, which fail to align well with some generalization phenomena observed in deep learning, such as the tendency for flatter minima to generalize better. The authors then derive new information-theoretic generalization bounds by constructing an "omniscient trajectory", which allows for a better capture of the flatness of SGD solutions in the loss landscape. Empirical results demonstrate that these bounds improve upon previous ones. Additionally, the authors extend their analysis by applying techniques from previous works, such as the individual technique, to mitigate limitations of information-theoretic bounds in stochastic convex optimization (SCO) problems.

### Strengths
1. This paper introduces novel information-theoretic generalization bounds for SGD that improve upon previous results, particularly when SGD finds flat minima.

2. Constructing the auxiliary weight process twice, specifically through the omniscient trajectory, is a notable technical contribution in this field.

3. The results provide an explanation for the learnability of some SCO problems, where many previous information-theoretic bounds do not apply (e.g., non-vanishing bounds).

4. The paper is clearly written and easy to follow.

### Weaknesses
1. Some related works, whether cited or uncited in this paper, require further discussion in relation to these results. For example, [R2] also discusses limitations in [R1] and suggests that, if the SGD process can be well-approximated by SDEs (motivated by empirical observations), the auxiliary process could be the SDE approximation of SGD, with perturbed Gaussian noise depending on the training data and current state (i.e. depending on $S$ and $W_{t-1}$ at step $t$). Moreover, while not directly related to the generalization of SGD, [R3], which is missing in this paper, investigates the optimal covariance matrix of Gaussian noise in SGLD. Since these works also consider more complex geometric structures for anisotropic Gaussian noise, and emphasize the importance of alignment between flatness and the covariance of weight distributions for generalization, it would be beneficial to discuss whether the results here are comparable or related to them in any sense, such as [R2, Theorem 3.2].

[R1] Ziqiao Wang, and Yongyi Mao. "On the Generalization of Models Trained with SGD: Information-Theoretic Bounds and Implications." International Conference on Learning Representations 2022.

[R2] Ziqiao Wang, and Yongyi Mao. "Two Facets of SDE Under an Information-Theoretic Lens: Generalization of SGD via Training Trajectories and via Terminal States." The Conference on Uncertainty in Artificial Intelligence 2024.

[R3] Bohan Wang, et al. "Optimizing information-theoretical generalization bound via anisotropic noise of SGLD." Advances in Neural Information Processing Systems 2021.

2. In Section 3.4, assuming that $\nabla\hat{\mathcal{L}}_S(W_T)\approx0$ could limit the applicability of your results in modern deep learning. For example, in some settings, especially for large-scale neural networks, models can perform well without vanishing gradient norms, as noted in [R4]. Can the results here explain generalization in such cases?

[R4] Jingzhao Zhang, et al. "Neural network weights do not converge to stationary points: An invariant measure perspective." International Conference on Machine Learning 2022.

3. In the proof of Theorem 1, particularly in Eq. (B.4), why not let $\Omega(\tilde{w}\_{0:t-1})=\tilde{w}\_{t-1}-\mathbb{E}[g\_t]-\mathbb{E}[\Delta g\_t]$? This choice seems valid if the expectation is taken over a carefully chosen distribution. Such a choice would result in an additional term, $-\mathbb{E}[\Delta G]$ in the trajectory term of Corollary B.1. Could you clarify whether this alternative choice would obtain a tighter bound?

4. In Theorem 3, while I do believe the contribution to information-theoretic generalization bounds here is novel, it seems similar in insight to [R5]: namely, if the learning algorithm is stable, information-theoretic generalization bounds can explain the learnability of these CLB problems. The main distinction in Theorem 3 is that the stability parameter is explicitly expressed in terms of $T$ and $\eta$, as in Bassily et al. (2020), while [R5] simply uses the stability parameter. In other words, both this work and [R5] explain the generalization of stable algorithms (e.g., GD with smooth or non-smooth convex Lipschitz loss) but may not be able to overcome the limitations highlighted in CLB counterexamples in [R6].

[R5] Ziqiao Wang, and Yongyi Mao. "Sample-conditioned hypothesis stability sharpens information-theoretic generalization bounds." Advances in Neural Information Processing Systems 2023.

[R6] Idan Attias, et al. "Information Complexity of Stochastic Convex Optimization: Applications to Generalization, Memorization, and Tracing." International Conference on Machine Learning 2024.

Minor comments:

1. Some citation errors should be corrected: Wang \& Mao (2021) was published in ICLR 2022, not NeurIPS 2021. The evaluated variant of CMI bounds is first introduced in Section 6.2 of the arXiv version of Steinke \& Zakynthinou (2020). Additionally, Hafez-Kolahi et al. (2020) did not use evaluated variants of CMI, so the sentence "the evaluated variants (Hafez-Kolahi et al., 2020; Harutyunyan et al., 2021; Wang \& Mao, 2023b)" is incorrect. A missing reference here is [R7], which applies individual e-CMI bounds to obtain new generalization bounds. Accordingly, the correct sentence should be "the evaluated variants (Harutyunyan et al., 2021; Hellström & Durisi, 2022; Wang & Mao, 2023b)" to reflect $f$-CMI, e-CMI and ld-CMI.

[R7] Fredrik Hellström, and Giuseppe Durisi. "A new family of generalization bounds using samplewise evaluated CMI." Advances in Neural Information Processing Systems 2022.

2. The Hessian notation is inconsistent: in Section 2.2, $\hat{H}\_w$ and $H\_w$ are used to denote the Hessians at $w$, but in Line 218, $\hat{H}\_s$ is used, and in Theorem 2, $\hat{H}\_s(w)$ and $H\_\mu(w)$ are used. Please clarify these notations for consistency.

### Questions
1. How did you estimate the trace of the Hessian (or other matrices) and the inverse of the matrix in your experiments?

2. In Eq. (1), it seems the expression could potentially be simplified by setting $\xi\sim\mathcal{N}(\gamma,\Sigma)$, is this valid?

3. In Eq. (3), what specifically does $H$ represent in $I+H/2\lambda C$? Also, the notation for $H(W)$ is unclear here; you state that "$H(w)$ is defined as $\Delta H(w)$ or $\hat{H}_S(w)$", yet in Appendix B.5, it seems that $H(w)$ should correspond to $\Delta H(w)$. Furthermore, regarding the definition for $C$, does $H(W_T)$ refer to the population Hessian? The notations about Hessian become somewhat confusing due to their inconsistency.

4. In Line 356, the phrase "such $\Delta g_t$ is optimal under ..." appears. Where is the justification for this optimality provided in the text?

4. In the proof of Theorem 3, could you clarify why the CMI term in Eq. (D.4) takes the form $I(W_T+\sum_{t=1}^T\Delta g_t^i;Z_i|Z_{-i})$? I may have overlooked something, but it seems that the construction might no longer require the Gaussian noises.

5. Is there potential to extend your analysis to SGD with momentum? Neu et al. (2021) give a method for extending their analysis to SGD with momentum, though they acknowledge limitations in demonstrating any advantage for SGD with momentum over vanilla SGD using their method.


**I enjoyed reading this paper, and I would be happy to increase my score if the authors could adequately address my concerns.**

### Soundness
3

### Presentation
3

### Contribution
3
