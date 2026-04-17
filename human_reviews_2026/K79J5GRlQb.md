# Expectation Curvature: Beyond the Hessian in Non-Smooth Loss Landscapes

- Decision: Reject
- Scores: 8, 6, 2

## Abstract
Second-order methods seek to exploit loss curvature, but in deep networks the Hessian often fails to approximate it well, especially near sharp gradient transitions induced by common activation functions.  
We introduce an analytic framework that characterizes curvature of expectation, showing how such transitions generate pseudorandom gradient perturbations that combine into a glass-like structure, analogous to amorphous solids.  
From this perspective we derive: (i) the density of gradient variations and bounds on expected loss changes, (ii) optimal kernels and sampling schemes to estimate both Hessian and glass curvature from ordinary gradients, and (iii) quasi-Newton updates that unify these curvature terms with exactness conditions under Nesterov acceleration.
To probe their empirical role, we implement Alice, a lightweight diagnostic that inserts curvature estimates into controlled updates, revealing which terms genuinely influence optimization.
In this way, our results support further optimization research: they introduce a new theoretical picture of nonsmooth loss landscapes that can catalyze future advances in pruning, quantization, and curvature-aware training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper looks at the "curvature of expectation", i.e., it specifically looks at how activations like ReLUs introduce rapid gradient changes that are not captured by second-order/local curvature methods. (In other words, local curvature for ReLUs is always zero, which doesn't reflect the fact that the gradient changes from 1 to 0 when the origin is crossed.)

The authors define a "glass density" matrix that captures the expected square deviations of gradients (from the second-order model, like a variance) due to these local gradient changes. They prove that this matrix can be optimally estimated using finite differences and Rademacher vectors (similarly to Hutchinson's trace estimator). Under certain modelling assumptions (e.g., a local "floor") they show that this can explain the non-quadratic behaviour of neural networks.

They then derive an optimizer similar to quasi-Newton/Nesterov accelaration that estimates both the diagonal Hessian and the "glass density" using finite differences of 3 gradient estimates ($g^{(0)}$, $g^{(-)}$ and $g^{(+)}$). The resulting optimizer has promising performance characteristics in real-world experiments (although the authors pitch it as as a "diagnostic tool" rather than an optimizer).

### Strengths
* Original theoretical framing of non-smooth/subquadratic networks
* Resulting ALICE algorithm is intuitive and easy to implement/understand
* Rigorous derivation and proofs

### Weaknesses
* Few strong assumptions (the nonnegativity "floor", independence of activations)
* Non-diagonal terms are ignored?

### Questions
* How sensitive are the "glass density" estimates to batch size, perturbation step ($\\lambda$), activation types (e.g., ReLU, Swish, ELU, etc.)?
* Given that ALICE is presented as a "diagnostic tool" rather than an optimizer, why present validation/test results rather than training loss/time until convergence? It seems to me that the latter is more relevant as it doesn't confound the results (i.e., a better optimizer can lead to more overfitting in some cases, which says more about the problem than about the optimizer.).
* Is there a way that the nonnegativity floor assumptions can be empirically validated?
* The main direct application for this work seems to be pruning/compression (because 3 gradient evaluations for optimization is presumably prohibitive). Would it be feasible to perform an experiment that evaluates the impact of using the glass density matrix when pruning weights?

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
This paper presented a new framework for understanding the curvature of expected changes in non-smooth loss landscapes induced by ReLUs. First, the authors derived how gradient discontinuities from perturbations by ReLUs form a glass-like structure. Then, they described an analytical framework for estimating the density of the gradient variations and upper-bounding expected changes in loss. From there, they provided an algorithm to practically estimate Hessian and glass terms. The findings demonstrate the importance of probing the individual contribution of Hessian and glass curvature in training, providing new insights into understanding the non-smooth loss landscapes where the Hessian may fail to approximate the loss curvature sufficiently. 

I recommend tentative acceptance (though clarity must be improved), for two reasons. 1) The paper is well-motivated and provides a theoretically grounded and empirically insightful investigation of curvature in non-smooth loss landscapes. 2) The empirical results effectively demonstrated the theoretical predictions of accuracy improvements and expected bounds.

### Strengths
This paper analyzes a key problem in non-smooth optimization where the Hessian does not approximate the loss curvature due to steep gradient transitions. This is significant as the proposed framework can be generalized to investigate how gradient jumps due to activation sharpness beyond ReLU.
The decomposition into Hessian and glass-like terms is a valuable tool to evaluate the curvature estimations in Hessian-based optimizations. 
The dominance impact of the glass term diagonals estimation has a theoretical interpretation.  
The empirical results comparing with other optimizers are promising.

### Weaknesses
The paper stated the intuition that parameters influence on early layers would be glass dominated vs those closer to output layers are more Hessian dominated. How does this show up in the empirical analyses of ViT/ResNet18? What other constraints of datasets and architecture may contribute to the differences? 
Clarity
Some variables were not declared in the main text or when it was first introduced, impeding readability. E.g., μ in the example in Eq. 1, z^(k) in Eq. 3; ρ and ∗ in Thm. 4. 
Figure 4: missing X-axis labels.
Empirical motivation: a brief description of a Rademacher vector may be helpful for ease of reading.
L300-304 Empirical Questions: the questions were listed in an order following theorems 4 to 6. The experimental results presented in 4.2 were in reverse order. I suggest reordering the results so they follow as listed in the questions.

### Questions
What are the comparisons against (and potentially connections to) other methods that consider loss landscape geometry?
Dauphin et al 2024 (https://arxiv.org/abs/2401.10809); Foret et al 2021 (https://arxiv.org/abs/2010.01412).

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the problem of estimating curvature when the loss in not smooth, e.g. in neural networks with ReLU activations. It quantifies both theoretically and empirically that the gradient of the loss changes, for small displacements of the parameters, faster than expected with a smooth loss. It proposes a new update based on those results, which incorporates also momentum, and tests the new update empirically.

### Strengths
The paper considers a problem that is very interesting to me. It is not clear why second-order optimization methods are supposed to work (and do work) when the loss is non-smooth.

### Weaknesses
Unfortunately, the presentation of the paper is so poor that is very hard to learn anything from it. Theoretical results are so badly explained that is very hard to evaluate their correctness. For most figures, it is not adequately explained what is shown and the conclusions that should be drawn from them.

Here’s a list of major points:

- "Collectively, these boundaries form a gradient glass". It is unclear whether this statement refers to something known, then a reference is missing. or is a new results, then it is not explained clearly what a "gradient glass is", even after looking at Figure 1 (more on that below).

- I don't understand why it is intuitive that early layers have a gradient glass while "parameters closer to the output" have not. I can see that the last layer is often not followed by a ReLU, but what about other layers that are "closer to the output"? Why would they be smoother?

- In Figure 1b, there is no explanation on how are the loss range and gradient range computed. Furthermore, it seems that the gradient stays constant within each domain. Why is it not changing?

-  Figure 2 is quite important because it represents the main motivation of this work. Yet, there is no explanation on how this figure is done. Equation 1 is clear, but it is still unclear how to estimate p from actual data. So we just need to blindly trust that the authors have followed a reasonable procedure for doing that.

- I don't understand what the blue line represents in Figure 3. That is not explained anywhere.

- Theorem 1: “Consider a network with many ReLUs”. What do you mean by “many”?

- Notation used is not explained anywhere. For example, equation (2), I guess y is the pre-activation. What is mu? Is it all parameters of the neural network? Is it only the parameters of the given layer? What is y^(k)? Is it the pre-activation of neuron k? What is z in equation (3)? None of those are defined and/or explained.

- Is there any justification for why pre-activations would be distributed uniformly? Is there any justification for why gradient jumps would be independent and zero-mean?I believe that these assumptions are crucial for deriving most of this work’s theoretical results (for example, the prediction of diagonal dominance), but there is no discussion for why these assumptions are reasonable. I believe they are not.

- The prediction of diagonal dominance, and more generally of Equation 3, can be easily tested by regressing v versus delta. Why wasn't that done?

- Theorem 2 is completely unclear. It is not explained what a kernel is and what is its purpose. Even after looking at the proof in the appendix I still do not understand what this theorem is about.

- It is unclear under what circumstances the bound in Equation (6) holds. It must depend on the size of delta, because for large delta the bound clearly does not hold. However, there is no mention of what are the conditions. Furthermore, it is also not explained what is the expectation in equation (6). Is it an expectation over delta? How is it possible then that the bound then depend on delta?

- Theorem 5 is also unclear. I don't understand how the bound in equation (6) implies this theorem. Equation (6) includes an expectation (uncelar on what), but theorem 5 instead has no expectation.

- “The natural question is how such steps interact with momentum and acceleration schemes." Why would that be a natural question? To me a natural question is whether the many assumptions made in deriving the five theorems hold in any practical circumstance.

- “Do glass curvature terms improve predictive accuracy of loss changes?" This question is very interesting but I do not understand how any of the figures presented answers this question.

- Equation (16) is supposed to be the way to measure the "glass density", however I have no clue on where this equation is coming from, there is no derivation and it does not appear anywhere before section 4.

- I don't understand what Figure 4 is supposed to show. There is no consistent observation, for example one of the curves consistently outperforming others across panels. Also there are four panels but only two are described, a) and b), but the labels are missing on the plots. X axis labels are also missing and not explained in the caption, so we actually don't know what is plotted.

- How do better optimization results with Nesterov acceleration confirm theory? Improvement by Nesterov acceleration are observed basically in all experiments with any optimiser.

- It is also unclear what to make of Figures 5 and 6, there are a bunch of curves on top of each other, there appears to be no significant differences (error bars are not shown), and anyway there are no consistent results across panels.

### Questions
NA

### Soundness
1

### Presentation
1

### Contribution
1
