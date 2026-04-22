# LOTION: Smoothing the Optimization Landscape for Quantized Training

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 8, 0

## Abstract
Optimizing neural networks for quantized objectives is fundamentally challenging because the quantizer is piece-wise constant, yielding zero gradients everywhere except at quantization thresholds where the derivative is undefined. Most existing methods deal with this issue by relaxing gradient computations with techniques like Straight Through Estimators (STE) and do not provide any guarantees of convergence. In this work, taking inspiration from Nesterov smoothing, we approximate the quantized loss surface with a continuous loss surface. In particular, we introduce LOTION, Low-precision Optimization via sTochastic-noIse smOothiNg, a principled smoothing framework that replaces the raw quantized loss with its expectation under unbiased randomized-rounding noise. In this framework, standard optimizers are guaranteed to converge to a local minimum of the loss surface. Moreover, when using noise derived from stochastic rounding, we show that the global minima of the original quantized loss are preserved. We empirically demonstrate that this
method outperforms standard QAT on synthetic testbeds and on 150M- and 300M- parameter language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel quantized training smoothing framework called LOTION, which introduces randomized rounding noise to smooth the quantized loss, addressing the problem of gradient blockage and optimization instability caused by the discontinuity of the loss function in traditional quantization training.

### Strengths
This paper introduces a theoretically grounded approach to quantized training that smooths the quantized loss via randomized noise, effectively mitigating gradient instability and convergence issues in traditional QAT while maintaining conceptual simplicity and clear theoretical interpretability.

### Weaknesses
Although the paper proposes a reasonable smoothing framework from a theoretical perspective, it lacks an in-depth analysis of gradient stability near quantization boundaries. The transitions between sections are somewhat abrupt, the figures are not highly distinctive, and the experimental results focus mainly on loss values without providing richer performance comparisons or ablation studies.

### Questions
1、	Although the paper mentions that absolute maximum quantization can prevent overflow issues, LOTION requires computing or approximating the diagonal entries of the Hessian (e.g., via the empirical Fisher) at each training step. When the model scales to billions of parameters (such as the 150M/300M models mentioned in the paper), is this approach still feasible? Does it lead to lower efficiency compared with other methods?
2、	It is recommended that the authors provide more experimental details and include additional comparative experiments, rather than only comparing loss values.
3、	It is recommended to include an algorithmic flowchart to make the proposed method clearer.
4、	The authors mention that LOTION constructs a continuously differentiable loss function through expectation smoothing, but its gradients exist only “almost everywhere.” Have the authors examined the stability of these gradients near quantization boundaries? Could gradient explosion or vanishing occur in such regions?
5、	The overall writing is fluent, but the authors could reduce repetitive expressions (such as “in particular” and “specifically”) to improve readability.

### Soundness
3

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
3

### Summary
The paper proposes a principled framework, called LOTION, for training quantized neural networks by smoothing the discontinuous loss surface caused by quantization. Instead of relying on heuristic Straight-Through Estimators (STE), LOTION computes the expected quantized loss under randomized-rounding noise, which may make the loss continuous and differentiable almost everywhere while preserving the original global minima. The authors derives a closed-form second-order approximation showing the method acts as a curvature-aware regularizer, stabilizing quantized training. The paper also demonstrates empirically that LOTION outperforms standard QAT and PTQ on both synthetic tasks and large-scale 150M–300M parameter language models at INT4, INT8, and FP4 precision.

### Strengths
1. The paper proposes a new and principled formulation of quantized optimization through loss-level smoothing. It is novel by bridging classical randomized smoothing theory with quantization-aware training.
2. The theoretical part seems solid with clear derivations and a well-justified second-order analysis linking smoothing to curvature-aware regularization.
3. The generality and simplicity of LOTION may have adoption potential across both research and downstream applications.

### Weaknesses
1. While results on 150M–300M language models convincingly support the main claims, the experiments are limited to small parameter sizes and transformer-based architectures. The method’s generality would be more compelling if evaluated on other architectures/models.
2. Although the paper emphasizes that LOTION requires no new hyperparameters, it would be better to have a straightforward training-time cost or memory overhead compared to QAT.
3. Lack of Ablation on Smoothing Strength $\lambda$ and Noise Distribution
4. The paper only reports val loss but not downstream metrics (e.g., perplexity, BLEU)
5. The paper proves that the smoothed loss preserves the global minima of the quantized objective, but it lacks a bound on how closely the smoothed loss approximates the original during optimization

### Questions
1. The authors mentioned that extending LOTION to activation quantization as future work. Then would randomized rounding for activations require dynamic smoothing (per-batch or per-layer), and would this break the differentiability guarantees?
2, Did anuthors observe any stability issues or over-smoothing when $\lambda$ is too high? How sensitive are the results to $\lambda$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the discontinuities introduced during quantization aware training (QAT) that relies on straight-through-estimators (STE) during backward pass. The authors formulate a smoothened loss function that alleviates the discontinuities introduced during QAT. This is achieved by formulating the loss objective as an expectation over randomized rounding operations. Theoretical analysis shows that doing this renders the smooth loss objective almost differentiable, and can be enforced as tractable regularizers. Experiments on synthetic tasks and LLMs shows improved convergence behaviour for the proposed optimization scheme (LOTION).

### Strengths
* Smoothing of quantized loss landscape by approximating with continuous loss landscape, and local minimum convergence guarantees. 
* Improved performance on QAT on wide-range of models.
* Trains the model on the expectation of the quantized loss under randomized-rounding noise
* Theoretical analysis of starting with quadratic losses to arrive at the implied regularization, and then extending it to the general case is convincing.  
* Compelling experiments on synthetic tasks and on LLMs show that the proposed smoothing of loss function helps consistently.

### Weaknesses
* **Randomized rounding versus stochastic rounding:** The authors use stochastic rounding (the more established term in literature) to describe related work, but in their own method description they adhere to randomized rounding. Is this for a specific reason? To the extent I see, they are exactly the same. Or is there a subtlety that the authors want to point out, and clarify?

* **Beyond block-wise quantization:** The formulation of the method is presented for block-wise quantization. Is there a specific reason to limit to these settings? 
* What do the authors mean by this statement:
  > The scale parameters {sB } depend on wi , so the quantization lattice moves as optimization proceeds. 
* **Data-dependent regularizer:** In L239, authors mention "a data-dependent" regularizer; I am unsure how the claim of data-dependence is manifested here?
 * $L_{GN}(w)$, in its final formulation in Eq. 3 is interpreted as a curvature-aware ridge regularizer. And the authors state these components can be obtained "using another backpropagation with sampled labels", or "empiricial Fisher approximation". These are important choices that can have significant implications on the optimisation. Which is the specific choice used in the work; as far as I see, it is the Fisher approximation (L400)? What would be better, and in which settings?
* In Fig 3, why do the gains between QAT and LOTION diminish for more complex networks (as k increases). 
* Why do the differences in performance diminish between INT4 and INT8 levels? Is it because the quantized loss landscape more drastic/discontinuous in INT4 regime?
### Other comments

* LLMs not abbreviated in first sentence in Introduction.
* "underperforms QAT" -> "underperforms compared to QAT"
* $H$ is not defined in its first use in L214. 
* Reporting the wall-clock time for convergence when compared to QAT will be interesting to see.

### Questions
See points above in weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposed a principled smoothing framework with randomized rounding for quantization aware training. The authors conducted  experiments for INT4, INT8, and FP4 quantization for 150M and 300M parameter language models.

### Strengths
It is beyond my capability to find strength in the paper.

### Weaknesses
The writing can be improved.

### Questions
I do not know what questions to ask.

### Soundness
1

### Presentation
1

### Contribution
1
