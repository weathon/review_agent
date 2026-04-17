# Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks

- Decision: Accept (Oral)
- Scores: 8, 6, 8, 4

## Abstract
High-dimensional non-convex loss landscapes play a central role in the theory of Machine Learning. Gaining insight into how these landscapes interact with gradient-based optimization methods, even in relatively simple models, can shed light on this enigmatic feature of neural networks. In this work, we will focus on a prototypical simple learning problem, which generalizes the Phase Retrieval inference problem by allowing the exploration of overparametrized settings. Using techniques from field theory, we analyze the spectrum of the Hessian at initialization and identify a Baik–Ben Arous–Péché (BBP) transition in the amount of data that separates regimes where the initialization is informative or uninformative about a planted signal of a teacher-student setup. Crucially, we demonstrate how overparameterization can "bend" the loss landscape, shifting the transition point, even reaching the information-theoretic weak-recovery threshold in the large overparameterization limit, while also altering its qualitative nature.
 We distinguish between continuous and discontinuous BBP transitions and support our analytical predictions with simulations, examining how they compare to the finite-N behavior. In the case of discontinuous BBP transitions strong finite-N corrections allow the retrieval of information at a signal-to-noise ratio (SNR) smaller than the predicted BBP transition. In these cases we provide estimates for a new lower SNR threshold that marks the point at which initialization becomes entirely uninformative.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors consider a teacher student model for finite-width, one-hidden-layer neural networks with quadratic activations (generalizing phase retrieval). 
They study the Hessian of the loss landscape of the student at initialization, and characterize when a certain overlap between teacher weights and leading hessian eigenvector transitions from zero to finite (either in a continuous or discontinuous way). A positive overlap means that the Hessian at initialization contains easily accessible information on the teacher weights.
They compute the sample critical threshold, i.e. the minimum amount of samples M over dimension N such that there is information about the teacher weights in the initialization Hessian, and study the phenomenology as a function of the overparametrisation.

### Strengths
The authors quantify the effects of overparametrisation in a non-convex problem, in particular the effect at initialization. Given the context on spectral initialization and GD dynamics, I find it a nice way to discuss such effects.

The authors provide an application of the recently discussed discontinuous BBP transition, showing that such behavior is not a totally abstract curiosity.

### Weaknesses
The mismatch between finite N simulations of alpha_BBP in the discontinuous case and the theory should be characterized more precisely, even though it is clear that finite size corrections will be hard to "eliminate". 
- Is there a different set of observables (other than that in Figure 3) that could be probed numerically to highlight the transition? Maybe with less important finite size effects? 
- How prohibitive would it be to access finite size corrections from the field theory formalism?

### Questions
It is not apparent to me why one would need p,p* = O(1). What would fail in the derivation for p=O(d), for e.g.?

line 116: I would like to highlight also the following relevant paper https://arxiv.org/abs/2505.17958 where over-parametrized phase retrieval is considered also in an empirical minimization setting (finding the same weak recovery threshold as Maillard et al. 2024) in the complementary setting p^* = O(d), p=O(d) with p^*/d -> 0.

line 143: it is not clear to me why one would normalize the loss by 1/labels. Given that the authors remark that this is an important element of the subsequent analysis, it would be nice to have more intuition here.

line 157: is it "when" or "where"? If the first, when one would expect such Gaussianity to hold?

line 215: it would be nice to have an explicit definition of m here.

line 254-258: is there an intuition for why exponential decay at the bulk boundary induces a first order transition in the overlap? 

line 333 and 351: is it the largest eigenvalue instead of smallest?

line 370 - 400: it is not clear to me how is alpha_0 computed: is it an analytical quantity? A numerical one? Also, it is not clear why it should be a lower bound to the finite-N transition. I suggest clarifying a bit.

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
The paper investigates how overparameterization affects the loss landscape at initialization in gradient-based learning. In particular, it considers a teacher-student setup where both teacher and student are two-layer neural networks with quadratic activations, generalizing previous works on phase retrieval to a multi-index setting. By analyzing the spectrum of the loss Hessian, the authors identify the conditions under which its leading eigenvector becomes informative about the teacher signal, and show that overparameterization bends the landscape, shifting the associated transition toward smaller sample sizes (lower SNR). In the infinite-width limit, the transition reaches the information-theoretic weak-recovery threshold. Finally, the authors investigate finite-size effects and compare them with the high-dimensional predictions.

### Strengths
The paper is theoretically and numerically sound. It addresses the important question of how overparameterization affects the learning landscape, offering novel, quantitative results in the specific setting of quadratic two-layer networks. Its originality lies in extending previous analyses of phase retrieval to a more general framework, providing a detailed characterization of the BBP phenomenology and valuable insights on finite-size effects. Overall, the presentation is clear and connects the findings to known information-theoretic thresholds and spectral methods.

### Weaknesses
1. One main weakness is the lack of a methodological overview in the main text. The technical analysis is confined to the appendix, leaving readers without intuition about the derivation. The paper could strongly benefit from a short but insightful methodological summary in the main section, possibly by shortening the (sometimes redundant) conclusion and/or using the additional page.

2. Some relevant references on the multi-index setting are missing. For instance, the critical SNR $p_\star/2$ has also been derived as the computationally optimal threshold in [1]; spectral method for multi-index models, included the quadratic network studied here, have been investigated and rigorously characterized in [2, 3].

3. The choice of the loss function may appear somewhat *ad hoc* and problem specific. It can benefit from further motivation and discussion.

[1] Troiani et al., "Fundamental limits of weak learnability in high-dimensional multi-index models"

[2] Kovačević et al., "Spectral Estimators for Multi-Index Models: Precise Asymptotics and Optimal Weak Recovery"

[3] Defilippis et al., "Optimal Spectral Transitions in High-Dimensional Multi-Index Models"

### Questions
1. Could you offer some intuition on how the label noise might qualitately affect the BBP phenomenology observed in this work?  

2. In the conclusion, you mention that understanding the interplay between the emergence of a signal in the Hessian and the behavior of gradient-descent dynamics is an open direction. Do you have any preliminary numerical evidence or intuition on whether the BBP transition identified here corresponds to the point where gradient descent (or flow) begins to correlate with the teacher signal?
In particular, do you expect a qualitative behavior similar to the loss of correlation with the informative eigenvector observed in [Bonnaire et al., 2024], or would the overparameterized setting change this picture? Even a qualitative comment on whether such a connection is expected or not would be very interesting, although understandably beyond the main scope of this work.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies how overparameterization changes the geometry of simple neural networks at initialization.
The authors look at a two-layer quadratic network in a teacher–student setup, analyze the spectrum of the Hessian at random initialization, and show that it undergoes a Baik–Ben Arous–Péché (BBP) transition as the sample size (or SNR) increases.
The main finding is that wider networks “bend” the loss landscape: the BBP threshold shifts to smaller SNRs, the transition can become discontinuous, and in the large-width limit the threshold actually reaches the information-theoretic weak-recovery limit.
The work ties together ideas from random matrix theory, spectral initialization, and overparameterized learning, with clear analytical results and convincing numerical checks.

### Strengths
This is a strong theoretical contribution. The idea that overparameterization changes the nature of the BBP transition—and that in the large-width limit one reaches the optimal weak-recovery threshold—is both interesting and novel.It deepens our understanding of why wide models are easier to train, and it connects two previously separate lines of work: loss-landscape curvature and spectral initialization.

Original and timely topic: the interplay between overparameterization and loss-landscape geometry.
Technically clean derivations, connecting to known spectral and phase-retrieval results.
Solid numerical support and a nice discussion of finite-size effects.
Clear writing and good figures that make a rather technical story accessible.
Conceptually important: shows how widening a network effectively reshapes the curvature of the loss, anticipating information about the teacher signal even before training starts.

### Weaknesses
The analysis is limited to quadratic activations, which makes it less clear how general the conclusions are.

The field-theory derivations could be compressed; parts of the appendix are a bit heavy "physics-style"

It would have been nice to see a direct quantitative comparison with actual spectral initialization methods to highlight practical implications.

### Questions
Beyond quadratic activations:
The current analysis focuses on quadratic activations, which make the problem analytically tractable. It would be useful to discuss whether similar BBP behavior should appear for other smooth nonlinear activations (e.g., ReLU, erf). How much of the observed “bending of the landscape” is a consequence of the quadratic structure, and how much would persist in more realistic nonlinear settings where the Hessian couples to the input distribution?

Interpretation of discontinuous BBP transitions:
The discontinuous BBP transitions are a striking result. Can the authors clarify their physical or algorithmic interpretation? Do they correspond to a first-order–like instability in the optimization landscape, or to a sharp onset of alignment during early training dynamics?

Connection to optimization dynamics:
Since the paper analyzes the Hessian at random initialization, one may wonder why this spectral transition should meaningfully predict the onset of learnability for gradient descent in practice, especially at finite width. Are the BBP signatures expected to survive after a few optimization steps, or are they quickly “washed out” as the model moves in parameter space?

Robustness beyond Gaussian inputs:
How sensitive are the predicted thresholds to the Gaussian data assumption? Would structured or correlated inputs (e.g., non-isotropic covariance, nonzero mean) qualitatively alter the BBP critical point or the continuous/discontinuous nature of the transition?

Relation to implicit regularization and flatness:
Could the authors connect their findings to the broader literature on implicit regularization in overparameterized models—such as the bias of gradient flow toward flat minima? Does the observed “bending” of the Hessian spectrum have an analogue in the flatness or margin properties of trained solutions, or suggest a theoretical link between curvature at initialization and generalization in the final model?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper discusses a phase transitional behavior of the Hessian at initialization in the overparametrized setting. They discuss the setup where the teacher and student networks are two-layer neural networks with quadratic activation, and the width of the student is possibly larger than the width of the teacher. In classical BBP transition analysis, it is known that there is a threshold where the SNR is larger than this threshold the largest eigenvector aligns with the true signal. They analyze the threshold in the overparametrized student-teacher setup, and show that 1) transition happens either continuously or discontinuously 2) overparametrization decreases the threshold and makes training easier 3) as the width of the student -> infinity, the threshold becomes optimal. The claims are supported with experiments.

### Strengths
1. An interesting perspective on neural network training by discussing overparametrized phase retrieval - the theory involves machinery from quantum mechanics (of which I did not fully understand), which shows an interesting link between learning theory and physics.

2. The related works are cited extensively and mentioned appropriately in relevant parts of the manuscript.

### Weaknesses
1. Clarification in terminology is needed. 
 - Why would this be a "loss landscape" result? Seems to me that the result is mostly on Hessian "at initialization" - which to me, it is not natural to understand the result as loss landscape result (of course, Hessians and loss landscape are related, but the training dynamics is not discussed).
 - What does "bend the landscape" mean? 
 - What is the SNR that is repeated throughout the paper? I assume it would be alpha = M/N, I am wondering if the terminology SNR is used to express such quantity.
 - Using the term overparametrized could be a little bit misleading, because in general the term is used to state that the number of parameters >= number of data points. 

2. It is a little hard to understand the technical novelty of the paper. What is the technique that is needed to study the overparametrized setting, which is different from previous approaches? Are there any challenges? 

3. The experiments are good in the sense that the discoveries are verified, but the experiments are quite small-scale. $p, p^{*}$ are in the scale of 1,2,3,4. Larger experiments may be helpful. Also, I can see that the threshold becomes smaller when overparametrization occurs, but does that imply that it needs to better training? I don't see direct evidence of it. For instance, it would be helpful if there is an experiment where you apply gradient descent on the actual learning problem and show that overparametrization yields faster convergence/better generalization etc.

4. It would be good if we could see justifications of certain theoretical problem settings. e.g. Why should the activation be quadratic? Why should we train with normalized quadratic loss function?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
