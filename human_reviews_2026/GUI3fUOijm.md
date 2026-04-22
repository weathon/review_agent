# Generalization Aware Minimization

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 2, 4

## Abstract
Sharpness-Aware Minimization (SAM) optimizers have improved neural network generalization relative to stochastic gradient descent (SGD). The goal of SAM is to steer model parameters away from sharp regions of the training loss landscape, which are believed to generalize poorly. However, the underlying mechanisms of SAM including whether its bias toward flatter regions is why it improves generalization are not fully understood. In this work, we introduce Generalization-Aware Minimization (GAM), derived by directly applying the goal of guiding model parameters toward regions of the landscape that generalize better. We do so by showing mathematically through a Bayesian derivation that the landscape of expected true (test) loss is a rescaled version of the observed training loss landscape, and that a sequence of perturbative updates in place of SAM's single perturbative update can optimize the expected test loss. We present a practical online algorithm to implement GAM's perturbative steps during training. Finally, we empirically demonstrate that GAM has superior performance over SAM, improving generalization performance on a range of benchmarks. We believe that GAM provides valuable insights into how sharpness-based algorithms improve generalization, is a superior optimizer for generalization, and may inspire the development of still-better optimizers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper developed a new optimizer, GAM, that directly optimizes the expected test loss. This is   different from SAM, which minimizes the so-called flatness instead. The authors argue that the test loss landscape is a rescaled version of the training loss landscape, and GAM is able to approximate the corresponding gradient after a sequence of perturbative updates. The effectiveness of GAM is finally evaluated on several benchmarks.

### Strengths
This paper is easy to follow and the flow of the proposed algorithm is very clear. The idea of transforming the gradient of training loss to the gradient of test loss is really impressive, which I believe deserves further study.

### Weaknesses
While the proposed algorithm is quite impressive, there remains several issues that needs to addressed:

 - My biggest concern is about the quadratic assumption. In most cases, as far as I can tell, this assumption does not hold during the training process. I suggest the authors could provide more experimental validations.
 - In Theorem 2, the authors showed that the transformation of one gradient to another. Can the authors provide more intepretation on why requiring $f^\prime(0)=1$? Moreover, one can notice from Figure 4 that the transformation seems to be not as expected. This could be more severe for high-dimensional distributions, particularly for millions of parameters. What the authors have done to alleviate this issue?
 - In Algorithm 1, how line 13, the gradient with respect to $\gamma$ is implemented?
 - More experimental results should be included. In Table 1, the authors only reported the results on several simple architectures. And the accuracy on ImageNet-1k is too low (practically, it should be at least 70\% even for SGD). And the legend of Figure 2 should be better indicated such as including the value of $\gamma_1$ and using different styles. In brief, more experiments including baselines, computation overhead, hyper-parameter sensitivy should be also be presented.

### Questions
please see Weaknesses.

### Soundness
2

### Presentation
2

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
This paper addresses the unclear mechanism by which the SAM optimizer improves generalization. It proposes a novel optimizer, GAM, which directly optimizes the expected test loss. The authors prove that the true expected test loss is a rescaled version of the training loss. Building on this, GAM employs a sequence of perturbed updates to directly optimize the expected test loss. Furthermore, the authors design a practical online algorithm that outperforms SAM on several benchmarks. This work demonstrates strong theoretical rigor, innovative methodology, and robust empirical support, with significant heuristic value.

### Strengths
- Provides a solid theoretical foundation for understanding the generalization capability of the SAM optimizer.
- Reveals that a sequence of parameter perturbations can transform the gradient of the training loss into that of the test loss, bridging the gap between optimizing training loss and directly optimizing expected test loss.
- Well-structured and easy to read.

### Weaknesses
Although the proposed method is theoretically sound, it incurs substantial computational overhead, limiting its practical applicability.

### Questions
- In equation 11, should the second instance of  $D^t$ be $D^{t-1}$?
- It is unclear how line 8 in Algorithm 1 is derived from equation 11; further derivation details are needed.
- The experimental results in Table 1 lack performance data for CRSAM on the ImageNet dataset; an explanation is required.
- The process for generating Figure 2 is not sufficiently detailed; more methodological description is recommended.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims at better generalization of neural network by directly optimizing on the testing loss landscape. This is done by first theoretically showing the testing loss landscape is a rescaled version of the training loss landscape. Then the authors show such difference can be covered by perturbative steps, arriving at GAM algorithms that using multiple online-adjusted perturbative steps. Experiments show GAM surpasses SGD and SAM. It also implies SAM, as single-step GAM, can be seen as implicitly optimizing directly on the testing loss landscape.

### Strengths
- The paper features theoretical analysis that reveals the connection and difference between the testing and training loss landscape.
- Such connection is used to develop algorithm that optimizes directly on testing loss landscape, which is well motivated.
- Experiments demonstrate that GAM surpasses SGD and SAM with statistical significance.
- SAM is connected to GAM, revealing its implicit role of simulating testing loss landscape. This connection provides a new perspective on understanding SAM.

### Weaknesses
- Gap between theoretical results and aim / practical use: Theorem 1 lay down the theoretical foundation for this paper. However, its conditions are a bit difficult to comprehend at first glance, especially with quadratic approximations involved. For example, the randomness of quadratic loss parameters ($\\tilde{\\theta}, \tilde{M}, \dots$) is abstracted / comes from which part of the whole training? What does the whole training process as a random process, where the loss parameters are involved, look like. Therefore, I suspect that such result cannot be directly applied to the training process. For example (please correct me if I embed Thm 1 into the training process in a incorrect way),  the training process is modeled by the following random process:
  
  $$
    (\bar{X}, \bar{Y}) \overset{\text{training}}{\rightarrow} \theta \underset{\text{training loss}}{\overset{\text{approximating}}{\rightarrow}} (\tilde{\theta}^*, \tilde{M}, \tilde{c})
  $$

  and at the same time

  $$
    \theta \underset{\text{testing loss}}{\overset{\text{approximating}}{\rightarrow}} (\theta^*, M, c).
  $$

  Noting that $(\\tilde{\\theta}^\*, \\tilde{M})$ and $(\\theta^\*, M)$ have a confounder $\theta$, they are correlated and one cannot have $\\theta^\*, \\tilde{\theta}^\* \\perp  M, \\tilde{M}, c, \\tilde{c}$, which is an assumption of Theorem 1. Intuitively, it says the training may bias the parameter to empirical-loss flat minima, where the testing-loss flatness may differ a lot, just like overfitting. I believe it is indeed the case especially when the parameter is explicitly or implicitly (eg, SGD is well-known to have flatness bias) optimized toward flat minima.
- Extra sample use: In Line 5, Algorithm 1, new data is directly sampled from the data distribution to measure how much $\\bar{g}\_{\theta}$ deviates from $g_{\theta}$. I read Appendix E and found no indication about how this data is sampled (so I assumed it is some new data. Correct me if I was wrong) and how much of them is used. If this data is indeed newly sampled, I suspect severely unfair comparison in experiments where SAMs do not use extra data. If it is reused training data, then this data is already used in training and is (over)fitted. Can it estimate the testing data without bias? If it comes from training data partition, then will it hurt data efficiency? 
- Such direct access to testing loss landscape also leaves Theorem 1 not fully exploited. Abstractly, Theorem 1 involves by revealing that the difference between the empirical and testing loss landscape has a rescaling-like structure and Theorem 2 says some perturbative can cover this gap, but how it is covered is by directly comparing the two landscapes instead of exploiting the explicit difference revealed by theoretical results.
- Experiments: The experiments use networks whose architectures may be too old. Also, for deeper networks, GAM seems unable to surpass CRSAM. ImageNet results for CRSAM is also missing.

### Questions
1. In the paper, Theorem 2 helps constructing the true testing loss gradients by moving the parameter toward a new place where the empirical gradients have similar gradients with testing gradients, i.e., simulating the gradient modification by moving where the gradient is computed. What is the motivation behind this simulation? Why not just do some calculation and moving the empirical gradients instead? I found "surprising similarity between the update mechanisms of SAM and GAM" in the Discussion. Is it related to unifying SAM with GAM?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper derives a relationship between the expected test loss and the training loss: they share eigenvectors but the eigenvalues differ, so that the Hessian is a rescaling. From this, they come up with their Generalization-Aware Minimization algorithm that estimates this. On CIFAR-10/100, SVHN and ImageNet, they find that at the same iteration, GAM outperforms SAM and

### Strengths
I'm plausibly unaware of work in the past couple years that could've made this seem more obvious, but the rescaling result seems interesting on its own.

I appreciated adding some intuition and explanation of lines in the algorithm, and the way the assumptions are laid out and justified in §C; note I have not checked the proofs in much detail.

The figures were also good: Figure 1 was very clear and Figure 2 was valuable for further connecting the theory and the outcomes.

### Weaknesses
I'm not sure I would call 3x "manageable" overhead; at the very least, I expect to see compute-matched results somewhere, for those applications where that's more of a constraint than data (like pretraining foundation models). Saying "Future work
could explore approximations or scalable implementations of higher-order derivatives" felt weak to me, given all the work out there on approximating second-order derivatives already.

The settings feel a little unrealistic in 2025, maxing out at a ResNet-50 on ImageNet.

The gains also seem small for that overhead, e.g., 0.7% on ImageNet (when the results are in the 65% range, not near saturation)

### Questions
I was confused by Equation 11; where's the recursion? Shouldn't there be a t-1 like in line 8 of Algorithm 1?

### Soundness
3

### Presentation
4

### Contribution
3
