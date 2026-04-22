# Signal Preserving Weight Initialization for Odd-Sigmoid Activations

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Activation functions critically influence trainability and expressivity, and recent work has therefore explored a broad range of nonlinearities.
However, widely used Gaussian i.i.d. initializations are designed to preserve activation variance under wide or infinite width assumptions.
In deep and relatively narrow networks with sigmoidal nonlinearities, these schemes often drive preactivations into saturation, and collapse gradients.  To address this, we introduce an odd-sigmoid activations and propose an activation aware initialization tailored to any function in this class. Our method remains robust over a wide band of variance scales, preserving both forward signal variance and backpropagated gradient norms even in very deep and narrow networks. Empirically, across standard image benchmarks we find that the proposed initialization is substantially less sensitive to depth, width, and activation scale than Gaussian initializations. In physics informed neural networks (PINNs), scaled odd-sigmoid activations combined with our initialization achieve lower losses than Gaussian based setups, suggesting that diagonal-plus-noise weights provide a practical alternative when Gaussian initialization breaks down.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the initialization of neural networks and proposes a new strategy of initialization when using sigmoid-like activation functions, denoted by $f$. The entire study is based on the fixed points of $f$: when $f$ belongs to a given set of sigmoid-like functions, there exists a critical input scaling $\omega > 0$ such that $x \mapsto f(a x)$ has only one fixed point ($0$) when $a < \omega$, and three fixed points ($0, \xi_a, -\xi_a$) when $a > \omega$.

This paper proposes an initialization strategy based on this critical number $\omega$, which depends on the shape of $f$, which is meant to achieve several goals:
1. with well-chosen input scalings ($a > \omega$), the pre-activations are guaranteed not to converge to 0 in probability (Theorem 4.6);
2. the outputs of the network meet a specific requirement (Theorem 4.7).

Finally, some experiments in simple cases (MNIST, Fashion-MNIST) show the superiority of this initialization strategy compared to Xavier, He and Orthogonal.

### Strengths
The main idea is easy to follow.

It is still interesting to design initialization distributions for specific activation functions. Moreover, a complete study of the fixed-point distribution obtained after a large number of layers is still missing in the initialization of neural networks literature.

The proposed initialization distribution can be easily used.

### Weaknesses
## Motivation
**Discussion about former works.** The paper states that it proposes a method that "keeps activation distributions well dispersed across layers, mitigating collapse to zero or saturation." The paper should provide evidence that former initialization strategies fail to address these issues. At least, the paper must explain the limitations of former initialization strategies. Please note that the "Edge of Chaos" works [2, 3] (cited in the paper) already address the issues of "collapse to zero" or "saturation".

## Clarity
This paper contains several notations and terms that are not properly defined. This is a major issue, since it prevents the reader to fully understand some parts of the paper, for instance Theorem 4.7.

**Negative rate $p$.** I did not understand what this "negative rate" is and why it is so important. I only understand that it is related to the output of the model.

**Scalar surrogate $\pi_L$.** Same issue: what is it? What is the link with calibration?

**[Non-critical] Variance $\sigma_z$.** What is it? It seems that this notation comes from [1], but it is not defined in this paper (it should be removed...).

**[Non-critical] Functions $\Phi_m, \Phi_m^{\alpha}$.** These notations should be replaced with $\Phi^m$ and $\Phi^m_{\alpha}$, since they consist of functions $\phi_a$ combined $m$ times. 

**Non-readable figures.** Figure 2 is very difficult to read: the text is too small.

## Significance of Theorem 4.6
I do not understand why Theorem 4.6 is significant whatsoever. It seems that the goal is to show (ii), stating that the probability that the model outputs sth. away from $0$ is bounded by below. However:
1. the "gains" $(a_1, \cdots, a_m)$ are related to the outputs of the neurons of the layers $(1, \cdots, m)$, so the assumption that $(a_1, \cdots, a_m)$ are Gaussian and independent does not hold (even if $a_i$ is Gaussian conditionally to the inputs of layer $i$);
2. the neural network is assumed to have $N_l = 1$ neuron per layer, which is obviously not true.

Overall, the restricted hypotheses of Theorem 4.6 are not discussed: we do not know if they are actually restrictive, and we do not know if the claims hold in practice (with dependent activations and several neurons per layer).

## Preceding work
This papers is entirely based on [1]: same notation, same basic ideas (e.g., rewriting of the propagation used in Eqn. (1)), and same focus on the fixed points. While [1] focuses on $\tanh$, this paper extends the study on [1] to a set of activation functions. This inspiration from [1] is entirely endorsed in the current paper, and there is nothing wrong about that.

However, the significance of the current paper compared to [1] can be discussed. As such, the initialization strategy proposed here is based on only two new contributions:
1. the study of $\tanh$ is extended to a set of $\tanh$-like functions;
2. the tuning of the input scaling is now founded on a heuristic (and not a numerical estimation), which allows us to compute it analytically.

The significance of Theorem 4.6 the clarity of Theorem 4.7 are discussed above.

## References

[1] Robust weight initialization for tanh neural networks with fixed point analysis, Lee et al., ICLR 2024.

[2] Exponential expressivity in deep neural networks through transient chaos, Poole et al., NeurIPS 2016.

[3] Deep information propagation, Schoenholz et al., ICLR 2016.

### Questions
What are the limitations of preceding works (e.g., Edge of Chaos)? Please be precise.

Could you clarify the "negative rate $p$", "$\pi_L$" and Theorem 4.7?

Discuss the hypotheses and the conclusion of Theorem 4.6. Limitation? Possibility to loosen the hypotheses? Does the conclusion hold in practice?

### Soundness
2

### Presentation
2

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
This paper proposes a weight initialization method for feed-forward neural networks (FFNNs) whose activation functions belong to a class referred to as odd-sigmoid functions.
An odd-sigmoid activation function denotes a monotonic, origin-symmetric nonlinearity such as tanh.
Building upon the analysis of fixed points of activation functions (similar to the discussion in Lee et al., 2025), the authors propose a weight initialization strategy that ensures the variance of activations in a layer remains above a certain threshold.
Experiments on MNIST and Fashion-MNIST demonstrate that the proposed initialization leads to higher accuracy during the early training epochs compared to conventional initialization schemes.

### Strengths
* The paper introduces a novel weight initialization method grounded in the fixed-point analysis of activation functions.
* It provides a theoretical analysis of the variance of activations in FFNNs initialized with the proposed scheme.

### Weaknesses
* Some theoretical assumptions discussed in the paper seem to deviate from realistic conditions (see the Questions section below).
* The proposed method is not applicable to activation functions that are not origin-symmetric, such as ReLU.
* The evaluation metrics used in the experiments differ from common benchmarks, making it unclear how practically useful the proposed approach is.

### Questions
* l.057: The paper states that “the proposed initialization reduces reliance on normalization layers such as Batch Normalization, thereby lowering the burden of hyperparameter tuning (e.g., depth/width selection).”
  However, it is unclear how the absence of normalization layers is related to reduced dependency on depth or width selection.

* In Proposition A.2 and Corollary A.3, the coefficient $a_n$ is assumed to be positive.
  However, according to Eq. (1), this assumption does not seem to hold in general. Could the authors clarify this point?

* In Theorem 4.6, the assumption $N_\ell = 1$ is introduced.
  Does this imply that the width of every layer in the FFNN is set to one?
  If so, what general insights can be drawn from this theorem for more realistic FFNNs with $N_\ell > 1$?

* What does the term “width proxy” in Figure 4 represent?

* What learning rate was used for the Adam optimizer?

* Regarding Table 1:

  * Since the initialization involves random sampling, the results should be reported as averages and standard deviations over multiple random seeds.
  * Why are only the first five epochs evaluated? It would be more informative to include results over a standard number of training epochs.

* Do “MLP” and “FFNN” refer to the same architecture throughout the paper?

* For Table 2: Training for only ten epochs with extremely limited data (e.g., 100 or 500 samples) seems unrealistic.
  What kind of practical implication can be drawn from such results?

* In Figure 5, the authors consider cases where even models with Batch Normalization fail to train properly.
  Therefore, the claim that the proposed method enables BN-free training does not appear to be justified based on these results.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a weight initialization method designed for a special class of activation functions called odd–sigmoids. These are functions that are smooth, bounded, strictly increasing and odd. The goal of this initialization is to ensure signal preservation: as activations pass through deep layers, they neither collapse to zero nor saturate, without needing normalization layers like BatchNorm.

### Strengths
The exposition seems quite clear and accessible, the derivations are straightforward, the notation consistent and the ideas and goals quite clear.

### Weaknesses
- The theoretical contribution seem to heavily overlap with results on signal and variance preservation (e.g., Poole et al 2016; Schoenholz et al. 2016; Hayou et al.  2019) within a subclass of activations.  

- The paper seems to only studY forward signal propagation and ignores gradients, Jacobian conditioning, or correlation dynamics, see for example the work by Pennington et al.

- The experiments are limited to MNIST and Fashion-MNIST with vanilla MLPs and quite small depths.  Reported improvements over Xavier, He, or Orthogonal initialization seem modest and could lie within statistical variation?

- From a theory perspective comparison with a number of prior works appears to be missing, particularly Hayou et al. (2019) and Murray et al. (2021) where there appears to be a clear conceptual overlap with regard to analyzing the role of the activation.


Links to some of the papers mentioned.
- https://openreview.net/pdf?id=H1W1UN9gg
- https://proceedings.neurips.cc/paper_files/paper/2017/file/d9fc0cdb67638d50f411432d0d41d0ba-Paper.pdf
- https://proceedings.mlr.press/v97/hayou19a.html
- https://www.sciencedirect.com/science/article/pii/S1063520321001111

### Questions
- It is not clear to me what the analysis of the recursion $x_{n+1} = f(a_n x_n)$ and the ``pitchfork bifurcation'' add beyond established initialization principles such as initialization on the edge of chaos? How does this proposed framework differentiate itself from prior `edge-of-chaos or dynamical isometry analyses?

- Can you contrast and compare your theoretical results with Hayou et al. (2019) and Murray et al. (2021)?

### Soundness
2

### Presentation
2

### Contribution
1
