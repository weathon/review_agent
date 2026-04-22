# On the Convergence Direction of Gradient Descent

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 8, 6, 6, 2

## Abstract
Gradient descent (GD) is a fundamental optimization method in deep learning, yet its asymptotic directional properties remain less understood. In this paper, we prove that if GD converges, its trajectory either aligns toward a fixed direction or oscillates along a specific line. The fixed-direction convergence occurs under small learning rates, while the oscillatory convergence behavior emerges for large learning rates. This result offers a new lens for understanding long-term GD dynamics. Experimentally, we find that this directional convergence behavior also appears in stochastic gradient descent (SGD) and Adam. Furthermore, we discuss how these theoretical findings regarding oscillatory convergence might offer a perspective on the sharpness dynamics observed in the Edge of Stability (EoS) regime. Our work provides both theoretical clarity and practical insight into the behavior of dynamics for multiple optimization methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
Extending to the discrete setting a result known for the gradient flow, the authors characterize how gradient descent, assuming it converges to some local minima, will converge to this limit. They show that, depending on stepsize, it will either converge along a fixed direction, or oscillate along it. Numerical experiments are provided on toy examples to illustrate this result, and also to observe empirically, on a training neural network task, how behave the angle between consecutive iterates for several optimizers. A section is devoted to the connection of their work with the Edge of Stability (EoS) phenomenon.

### Strengths
The presentation is clear, the paper well organized. I think not assuming global smoothness a priori and instead characterizing the convergence according to the eigenvalue of the Hessian at a local minima is an interesting viewpoint, such that the main result (Theorem 1) is interesting.

### Weaknesses
I express severe doubts about the relevance of the connection established between the author's work and the EoS phenomenon, which is presented as a central contribution. 


- In their main result, the authors assume the function is $C^3$, and that gradient descent converges to a strict local minimizer. Because of the $C^3$ assumption, there exists a vicinity of this local minimizer such that the function is strongly convex and $L$-smooth for some $L$, with $L$ as close as we want to $\lambda_n$. 
As gradient descent is assumed to converge to this local minimizer, the algorithm eventually enters this vicinity, and stay inside it. With this regard:

     (i) It is implicitly assumed that asymptotically, the algorithm lies in a strongly convex, $L$-smooth landscape, with $\eta < \frac{2}{L}$. 

     (ii) In this setting, even if the sharpness exhibit oscillations , the loss still decreases monotonically, as it seems to be observed on figure 2.c. This is not consistent with the typical behavior of the EoS regime, where the loss is not monotonically decreasing, but only decreases in the long run.

     (iii) According to [1], a quadratic growth property does not fit the local behavior of neural network losses around minimizers, whose growth is more likely to be sub linear. Such a setting is not covered by the implicit "local strong convexity" assumption of the authors.

For those reasons, it is not clear to me whether this work improves or not our understanding of the EoS phenomenon. I believe the claimed connection should be substantiated more clearly.

[1] Helusive traductionao Ma, Lei Wu, and Lexing Ying. The multiscale structure of neural network loss functions: The
effect on optimization and origin

Minor remarks 
- The labels on figure 1 are very small

### Questions
- On figure 2)a), as the convergence occurs along the y-axis direction, why the sharpness does not converge to 1 instead of 2 ?

- Under equation (5), the justification that $a > b $ seems evasive to me. Could you elaborate ?

### Soundness
1

### Presentation
3

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
This paper studies the asymptotic behavior of gradient descent as it converges to a minimizer, on generic objective functions. The main assumption is that the objective is strongly convex around the minimizer (i.e. the Hessian at the minimizer has no zero eigenvalues).   The paper proves that there exactly two possible modes of convergence: either gradient descent converges along a specific direction (i.e. cosine distance between successive iterates is $\approx 1$), or it oscillates along a particular line (i.e. cosine distance between successive iterates is $\approx -1$).   They prove that the first case happens when $0 < \eta < 2 / (\lambda_1 + \lambda_n)$ and that the second case happens when $2 / (\lambda_1 + \lambda_n) < \eta < 2 / \lambda_n$, where $\lambda_1$ and $\lambda_n$ are the largest and smallest Hessian eigenvalues at the minimizer.

I think the basic intuition comes from the case of a quadratic objective.  Here, the rate of convergence along each eigenvector direction is $1 - \eta \lambda_i$, where $\lambda_i$ is the eigenvalue.  Asymptotically, we will be converging along the eigenvector for which $|1 - \eta \lambda_i|$ is largest.  There are two possible cases: 
  - if $\eta < 2 / (\lambda_1 + \lambda_n)$, then this direction will be the eigenvector corresponding to the _smallest_ eigenvalue $\lambda_n$ and the contraction factor $1 - \eta \lambda_n$ will be positive and we will converge along that particular direction
  - if $\eta > 2 / (\lambda_1 + \lambda_n)$, then this direction will be the eigenvector corresponding to the _largest_ eigenvalue $\lambda_1$, and the contraction factor $1 - \eta \lambda_1$ will be negative and we will oscillate along that line.

The paper basically makes this intuition precise for general objectives.

### Strengths
The paper nicely tackles an interesting problem in a fully rigorous manner.

### Weaknesses
The asymptotic behavior of an optimizer near a minimum may not be that relevant for practical deep learning tasks, where practitioners generally care about the behavior of the optimizer far from any minimum.

Another weakness is that the paper only studies the case where gradient descent converges to a point, whereas it is also possible for gradient descent to wind up in a orbit where it oscillates indefinitely, for example see [Chen and Bruna '23] or [Ghosh et al '25].

Lei Chen and Joan Bruna.  "Beyond the edge of stability via two-step gradient updates."  ICML '23.

Avrajit Ghosh, Soo Min Kwon, Rongrong Wang, Saiprasad Ravishankar and Qing Qu.  "Learning Dynamics of Deep Linear Networks Beyond the Edge of Stability."  ICLR '25.

### Questions
- What happens when the learning rate is greater than $2/\lambda_n$ -- can you prove that convergence is impossible?
- For the experiments in section 3.3, I think you used cross-entropy loss.  For cross-entropy loss, optimizers generally leave EOS before the end of training (see Cohen et al '21, Appendix C), which is what we see here.  I think that if you used MSE loss, you might see the other, oscillatory convergence regime.

### Soundness
4

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
This paper study the convergence in direction of the iterates of gradient descent. The main results of the papers identifies two regimes. If the step size $\eta$ verifies $0<\eta<\frac2{\lambda_1+\lambda_n}$ where $\lambda_1$ and $\lambda_n$ are respectively the smallest and the largest eigenvalues of the Hessian of the objective function $f$ at a critical point, the iterates of gradient descent converge in direction. If $\frac2{\lambda_1+\lambda_n}<\eta<\frac2{\lambda_n}$, the iterates exhibit an alternating convergence direction. The paper then discusses the implications of these results for the edge of stability and provides numerical experiments suggesting that the theoretical findings extend to other optimizer such as SGD with momentum and Adam.

### Strengths
* **S1**: The authors establish convergence direction of gradient descent. The result is sound and novel to my knowledge.

* **S2**: The established results provide insight on the behavior of the sharpness during the optimization trajectory.

* **S3**: Numerical experiments suggest that the theoretical results established for GD may extend to SGD with momentum and Adam.

### Weaknesses
* **W1**: The experiments appear to be based on a single run. Since some of the compared algorithms are stochastic, multiple runs should be conducted. The results should then report the average or median cosine metric along with standard deviations or quantiles to reflect variability and ensure statistical reliability.

* **W2**: The code of the experiments is not provided, hindering reproducibility.

### Questions
* **Q1**: In the experiments with neural networks, how do the learning rates compare with the sharpness? It seems that only the convergence direction occurs. Do SGD and Adam exhibit alternative convergence direction with higher step sizes?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the asymptotic direction of gradient descent (GD) near an isolated local minimum. It proves that, if the GD sequence converges, then its normalized iterates converge either to a fixed direction (for smaller learning rates) or alternate between two opposite directions along a line (for larger but still stable learning rates). The boundary between these regimes is $2/(\lambda_1+\lambda_n)$, where $\lambda_1$ and $\lambda_n$ are the smallest and largest eigenvalues of $\nabla^2f(x^*)$. The authors connect this to the Edge of Stability (EoS) by showing that, via a first-order expansion of $\lambda_{\max}(\nabla^2f(x_k))$, sharpness can oscillate even as the loss converges. Empirically, toy 2D examples illustrate the two regimes, and preliminary experiments suggest similar directional alignment behavior for SGD and Adam on CIFAR-10 via cosine similarity of successive updates.

### Strengths
The paper pinpoints a clean, eigenvalue-driven dichotomy for gradient descent near an isolated local minimum: when $0<\eta<2/(\lambda_1+\lambda_n)$, the normalized iterates converge to a fixed direction, whereas for $2/(\lambda_1+\lambda_n)<\eta<2/\lambda_n$ they alternate between two opposite directions. This result is stated cleanly (Theorem 1) with transparent assumptions and offers an interpretable link to the “edge-of-stability” phenomenon.

The analysis is technically careful: the argument builds an invariant neighborhood around the minimizer, bounds higher-order terms so the smallest-eigenvalue component dominates for sufficiently small $\eta$, and identifies a measure-zero exceptional set of initialization. 

The exposition is clear and visual: Sec. 3.2 connects directional behavior to sharpness oscillations via a local expansion of the largest Hessian eigenvalue, and Figs. 1 and 2 make the two regimes concrete on toy problems. Although preliminary, the empirical section suggests similar directional alignment patterns for SGD and Adam on CIFAR-10, hinting at broader relevance beyond plain GD.

To sum up:
1. The paper has strong mathematical analysis with clear assumptions, statements and proofs.
2. The work includes experimental work that compares the proposed method with other works.
3. The writing is generally clear with nice flow.

### Weaknesses
My only concern is that the empirical validation is limited and largely qualitative: it relies on toy 2D landscapes and a single CIFAR-10 configuration, without uncertainty quantification, ablations, or explicit tracking of the normalized iterate direction toward a limiting vector/pair. Strengthening the evidence with additional toy problems and a broader deep-learning suite, e.g., multiple architectures on CIFAR-10 and CIFAR-100, would make the claims more compelling.

### Questions
1. Can you include more experiments as discussed in Weaknesses?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper establishes the result on direction of GD convergence to the minima (with the $2/(\lambda_1 + \lambda_n)$ as the threshold between existence of direction and direction alternating at every step. A perturbative explanation for sharpness oscillations is given. Experiments conducted to show a presence of oscillations in the NN dynamics under GD/SGD/Adam.

### Strengths
The proof is correct and carefully written, adding the control for higher-order terms; the experiments are clear in what they are measuring and well-explained.

### Weaknesses
Theory:
    - unfortunately, there is no real novelty in the proven theoretical result (Theorem 1), as it is known in the classical iterative methods literature, see e.g. [1], Chapter 4. That is, your result is known for quadratics, the nonlinear extension is a technicality.
    - Considering that this is the main part of the paper (see the details about empirical part below), this lack of novelty constitutes enough of a ground for rejection.
Connection to EoS:
    - The above phenomenon is not really related to EoS. EoS happens also (and specifically!) when dynamics are away from minima - and that’s almost the point of it, as the training continues as the network is in EoS (see original Cohen’s EoS, 2021). Your result is asymptotic near a strict minima, which makes them only connected because both are about oscillations, and that’s it
    - there they are always zero (and negative) eigenvalues present in the spectrum of the Hessian of NN, see [2] or [3]. This makes the bound $2/(\lambda_1 + \lambda_n) = 2/\lambda_n$, therefore eliminating the regime you term “large learning rate regime”
    - Considering that you are discussing the oscillations of top eigenvalue during the EoS oscillations, you should contrast your work with Damian et al. “Self-Stabilization…” (2022), as this is precisely what this work is doing - showing how and why the sharpness oscillates and produced the self-stabilization
Empirics:
    - the empirics are basically the experiments of “A Walk with SGD” of Xing et al. (2018), albeit with Adam added. They are measuring the exact same quantity, showing the oscillations. Therefore, there is no novelty in the experiments either. In particular, the GD experiments show the EoS phenomena (as later described by Cohen et al al. in EoS paper)
    - Moreover, the experiments are unrelated to the theorem you are proving, see the above point about EoS being away from minima — that is, the oscillations start when we are away from minima, and there is no inherent reason why the oscillations in the experiments are the same oscillations that are talked about in the theorem. Therefore, it is unclear how the experiments support your finding.

### Questions
- What is the difference between your experiments and those of Xing et al.?
- How does your sharpness oscillations results compare to those of Damian et al. “Self-Stabilization…”

### Soundness
3

### Presentation
2

### Contribution
1
