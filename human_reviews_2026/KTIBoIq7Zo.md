# Why Do We Need Warm-up? A Theoretical Perspective

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Learning rate warm-up - increasing the step size at the beginning of training - has become a ubiquitous heuristic in modern deep learning, yet its theoretical foundations remain poorly understood. In this work, we provide a principled explanation for why warm-up improves training. We rely on a generalization of the $(L_0, L_1)$-smoothness condition, which bounds local curvature as a linear function of the loss sub-optimality and exhibits desirable closure properties. We demonstrate both theoretically and empirically that this condition holds for common neural architectures trained with mean-squared error and cross-entropy losses. Under this assumption, we prove that Gradient Descent with a warm-up schedule achieves faster convergence than with a fixed step-size, establishing upper and lower complexity bounds. Finally, we validate our theoretical insights through experiments on language and vision models, confirming the practical benefits of warm-up schedules.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work theoretically analyzes learning rate phenomena from a theoretical perspective. Specifically, they study a new constraint on the loss landscape geometry that assumes that the the top eigenvalue of the loss Hessian $\lambda$ is bounded by the distance to the minima, i.e., $\lambda \leq H_0 + H_1 (f(w) - f^*)$. Through examples, the authors show that this condition theoretically holds in multiple cases. Next, the authors analyze the convergence of GD under the learning rate motivated by the $H_0-H_1$ condition. Finally, the authors test their learning rate warmup schedule in practice and show it works on par with linear warmup commonly used in practice.

### Strengths
* The paper is clearly written and the main results are easy to follow
* Strong theoretical analysis of learning rate warmup under the $H_0-H_1$ condition
* The authors propose a warmup schedule that performs on with linear warmup.
* The proposed warmup schedule is practical, as it only depends on the current loss and one hparam $C$. This warmup strategy can be perhaps an alternative to the linear schedule used in practice.

### Weaknesses
**$H_0-H_1$ condition**: Definition 3.1 states that the largest eigenvalue of the Hessian $\lambda$ is bounded by $H_0 + H_1 (f - f^*)$. This implies that as the loss decreases the upper bound of $\lambda$ reduces. The authors show that this condition holds in realistic situations (Figures 1, 2, I.1, I.2). However, there are empirical evidence against it. In full batch setting, its well (empirically) established that $\lambda$ increases throughout training [1], which is at odds with the submitted work that claims the exact opposite. In mini-batch setting, its known that $\lambda$ increases during training, albeit with a slower rate.

There can me multiple reasons why this happens, which I detail below:
1. The authors use a proxy for the max eigenvalue (line 262), which may have a different behavior than sharpness
2. Its known that very early in training, $\lambda$ may decrease early in training (Appendix A of [1]), which is rather short (10 steps) compared to the warmup duration (1000 steps). The authors use a very small learning rate (1e-04 for SGD, 1e-07 for Adam), which restricts the training to this $\lambda$ decrease phase. I would request the authors to rerun these experiments with a typical learning rate schedule and check if the decrease in curvature is observed during the entire warmup phase (both with their schedule and linear warmup).
3. For unusually large initializations, $\lambda$ may decrease throughout training [4], which aligns with the results in the paper. I would like the authors to clarify if the experiments in the submitted work are operating in this regime.

**The gap between theory and practice**: The theoretical analysis of the submitted work assumes $\lambda$ decreasing through the warmup phase, which corresponds to the 'natural sharpness reduction' phase described in [3]. However, as mentioned above, in realistic settings, $\lambda$ increases during training. Regardless of whether $\lambda$ increases or decreases, the effect of increasing learning rate is to reduce $\lambda$ [2, 3]. 

This creates a causal gap between the theory and practice: 
1. **Theory**: $\lambda$ decreases during training, so increase $\eta$
2. **Practice**: $\eta$ increases which causes reduction in $\lambda$. 

Furthermore, the proposed schedule does not care about the causality anymore. It increases the learning rate depending on loss change and does not care about whether the curvature increases or decreases. This aligns with the standard linear learning rate schedule used in practice.

[1] Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability, 2021

[2] A Loss Curvature Perspective on Training Instability in Deep Learning, 2021

[3] Why Warmup the Learning Rate? Underlying Mechanisms and Improvements, 2024

[4] Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos, 2023

### Questions
* Line 67: "we provide empirical guarantees". I think there is a typo here. It should be empirical evidence rather than a guarantee.
* Equation 1: where do the constants 10, 20 come from?
* (Comment, not a question) Line 407: convergence is stochastic setting requires interpolation condition, whereas much of the realistic experiments (language modeling) are in underparameterized setting
* Can you check Figure 1, 2, I.1, I.2 for standard learning rates? How long the decrease phase lasts?
* Line 443, how did the authors come this practice schedule. It would be helpful to guide the reader.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates why learning-rate warm-up is so effective in large-scale deep learning and proposes a new theoretical explanation based on a novel $(H_{0},H_{1})$-smoothness framework. Unlike the traditional $(L_{0},L_{1})$ condition, which ties curvature to gradient norm and contradicts empirical behavior in early training, the $(H_{0},H_{1})$ condition bounds curvature by the loss suboptimality, matching observed linear relations between curvature and loss. Building on this, the authors derive a theoretically motivated warm-up schedule and compare it against the standard linear warm-up across experiments on language models and vision tasks (ResNet50, ViT-Tiny). Their results show that both linear and $(H_{0},H_{1})$ warm-up improve convergence over no warm-up, with the proposed schedule performing competitively while offering theoretical justification.

### Strengths
1. **Novel theoretical contribution**: The introduction of $(H_0,H_1)$-smoothness provides a fresh and insightful proxy for curvature, offering a more accurate explanation of warm-up dynamics than prior $(L_0,L_1)$-based analyses.

2. **Balanced theoretical and empirical support**: The paper combines rigorous convergence proofs with clear empirical validation on both language and vision benchmarks, lending credibility to the theoretical claims.

3. **Clear and well-structured presentation**: The exposition is logically coherent and easy to follow, making complex theoretical ideas accessible and enhancing the overall readability of the work.

### Weaknesses
1. **Severe formatting issues**: The manuscript does not comply with the official ICLR template. In particular, it uses an incorrect font throughout and shows evidence of space compression (e.g., between Figures 3 and 4), which detracts from professionalism and readability.
2. **Limited theoretical scope**: While the $(H_0,H_1)$-smoothness analysis shows that warm-up accelerates convergence, it does not establish that warm-up leads to a better final convergence outcome. In principle, similar effects of convergence could be obtained by simply extending training without warm-up. A stronger illustration for the necessity of warm-up still relies on arguments about training instability, which are not captured by this framework.
3. **Missed validation opportunity**: The proposed theory enables regression of $f^\ast$ from empirical training trajectories, yet the paper does not attempt such regression or verify the recovered $f^\ast$ against the true optimum. Including this check would provide a valuable validation of the framework’s practical accuracy.

### Questions
1. **Tightness of the bound**: Figures 1–2 display an almost perfect linear relationship between smoothness and loss suboptimality, whereas the $(H_0,H_1)$ framework only provides an upper bound. Is this near equality theoretically expected for certain model classes, or is it an artifact of the optimization trajectory and estimation method? Clarifying this would strengthen the interpretation of the empirical evidence.
2. **Applicability to attention layers**: Since attention is the fundamental component of Transformer models, does $(H_0,H_1)$-smoothness formally hold for a single attention block? Can the authors provide a proof or at least a theoretical justification beyond empirical observation?
3. **Combining proxies for curvature**: Curvature is inherently a second-order property. $(L_0,L_1)$-smoothness approximates it via a first-order proxy (gradients), while $(H_0,H_1)$-smoothness uses a zeroth-order proxy (loss values). Would a hybrid framework that leverages both first- and zeroth-order information yield a tighter or more general characterization of smoothness?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a generalization of the $(L_0, L_1)$ smoothness, they call $(H_0, H_1)$-smoothness. They show that under that assumption warm-up is preferrable to fixed step size. They show that under some assumptions a few neural networks satisfy this property.

### Strengths
The paper is well written and visually pleasing. Addresses a long standing problem in a clean way. Proofs and results are clear.
I really appreciated reading this.

The math part is well done, results are proved nicely.

### Weaknesses
#### On the Tightness of your Condition

You prove that under your condition LR-warm up is optimal. It is when your condition is tight! Not in general. If on a Linear Shallow network I start from zero, it does satisfy the condition but the Hessian grows towards the solution, it does not go down, thus fixed step size = the maximal step size you would pick it better. 

In general neural networks seem to show the phenomenon of progressive sharpening, which is conceptually the opposite of your condition. Can you please comment on this? Do you see progressive sharpening in the models you train?


#### Explaining Warm-Up?
However, I'm a little dubious that in general warm-up is for convergence purposes. I believe in non-convex cases and large scale ML systems it is for stability. I think it is necessary to comment on that. 

This is not a strong ground for rejection, but I believe one needs to account for/deal with these other explanations. I think the paper is clean and beautiful, just maybe you can conjecture *what else* warm up is needed for. 

For instance, when someone assumes progressive sharpening is happening, warm up is useful to constraint the model to less sharp areas of your landscape. Can you comment on this? Do you see this when applying your theory to neural networks? Do you see this in experiments?

I think my final grade will depend on how you address this. 
Precisely, making some experiments about and reworking the limitations of your work in analyzing the full picture behing warm up are explicit.


Also, I don't think it is sensible to speak about balanced neural networks for practice, because those are the flatter ones within the parameter space. That said I believe your assumption is also satisfied at standard initialization of linear networks. Maybe you can comment on how $H_0$ needs to grow to be satisfied at standard initialization.

### Questions
See weaknesses.

To what extent is progressive sharpening allowed under your property? Until $H_0$ I guess?

### Soundness
3

### Presentation
4

### Contribution
2
