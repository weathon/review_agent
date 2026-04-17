# Flower: A Flow-Matching Solver for Inverse Problems

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
We introduce Flower, a solver for linear inverse problems. It leverages a pre-trained flow model to produce reconstructions that are consistent with the observed measurements. Flower operates through an iterative procedure over three steps: (i) a flow-consistent destination estimation, where the velocity network predicts a denoised target; (ii) a refinement step that projects the estimated destination onto a feasible set defined by the forward operator; and (iii) a time-progression step that re-projects the refined destination along the flow trajectory.  We provide a theoretical analysis that demonstrates how Flower approximates Bayesian posterior sampling, thereby unifying perspectives from plug-and-play methods and generative inverse solvers. On the practical side, Flower achieves state-of-the-art reconstruction quality while using nearly identical hyperparameters across various linear inverse problems. Our code is available at https://github.com/mehrsapo/Flower.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper “FLOWER: A Flow-Matching Solver for Inverse Problems” introduces Flower, a general-purpose solver that leverages a pre-trained flow model to reconstruct signals consistent with given measurements. Flower operates through a three-step iterative process: (1) flow-consistent destination estimation, where the velocity network predicts a denoised target; (2) measurement-aware refinement, projecting the estimate onto the feasible set defined by the forward operator; and (3) time progression, updating the trajectory along the flow path. The authors provide a Bayesian interpretation, showing that Flower approximates posterior sampling, thereby linking plug-and-play (PnP) and generative flow-based solvers within a unified framework. Empirically, Flower achieves state-of-the-art reconstruction quality across diverse inverse problems using nearly identical hyperparameters. Its main contribution lies in bridging Bayesian posterior sampling and PnP methods through a theoretically grounded and practically efficient flow-matching formulation.

### Strengths
The paper:
-  Introduces a novel three-step flow-matching framework (Flower) for inverse problems, bridging generative flow models and plug-and-play (PnP) methods under a unified Bayesian view.

-  Provides solid theoretical grounding with a clear Bayesian justification and strong empirical results showing state-of-the-art/competitive reconstructions across diverse tasks.

- The paper is well-structured and accessible, clearly explaining algorithmic steps and visualizing the process through intuitive figures.

### Weaknesses
-  **More ablation** could be conducted on the number of evaluations performed at each step. The paper currently presents results for Flower1 and Flower5, where Flower5 shows steady improvement over Flower1. An ablation plot illustrating this trend would strengthen the analysis.

-  Although the method claims hyperparameter robustness, the validation is limited to standard imaging tasks. Demonstrating applications to **nonlinear** or **real-world** inverse problems (e.g., MRI, or compressed sensing) would further validate its generality and impact.

### Questions
- Could the authors provide an ablation plot showing the effect of varying the number of evaluations per step (e.g., Flower1-10?) to better illustrate the trade-off between reconstruction quality and computational cost?

- The current method is compatible with the standard Euler sampler used in flow matching. Could the authors discuss whether other ODE or higher-order samplers (e.g., Heun or adaptive solvers) could be integrated into Flower, and what potential advantages or challenges might arise?

- Could the authors elaborate on possible extensions of Flower to nonlinear or real-world inverse problems (e.g., MRI reconstruction, phase retrieval, or compressed sensing)? In particular, what modifications or additional components would be required to handle these more complex forward models?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Flower is a plug-and-play solver for linear inverse problems with pre-trained flow matching models. Given a pre-trained velocity model, Flower solves a linear inverse problem by iterating over three steps: 
1. Estimation of clean image $x_1$ given the current noisy image and velocity model. 
2. Projection of the estimated image $x_1$ into a feasible set with a proximal operator 
3. Estimation of the next noisy image along the flow using the projected $x_1$.  

The correctness of the proposed solver has been explained through Bayesian analysis and the key result shows that Flower samples from an approximate posterior distribution.

### Strengths
1. The core method has been explained intuitively and seems easy to implement. 
2. Computational efficiency: Runtime of Flower is less compared to many other baselines as indicated in Table 4. 
3. Robustness: The method uses same hyper parameters for all the linear inverse problems included in the paper. This simplifies hyper parameter tuning.

### Weaknesses
1. Writing should be more specific. 
    1. Contributions should indicate that this is a solver for linear inverse problems. Statements such as “Thus, Flower can be seen as a plug-and-play inverse solver with general application, with our Bayesian justification establishing a novel link between posterior sampling and the PnP frameworks in generative models.” need to be updated to clarify that this is a solver for linear inverse problems. 
    2. The paper should also indicate that the Bayesian analysis is only valid for sampling from the approximate posterior and not the true posterior. 
2. Only 100 images have been used for quantitative results which is a really small test dataset.

### Questions
1. Estimating $\Sigma_t$ involves computing an inverse of a matrix. How do you efficiently solve this such that the solver itself is light weight?
2. Line 342 states that the validation set for AFHQ-Cat was constructed with 32 images. However, the caption of Table 2 suggests that 100 test images were used. One of these statements need to be updated. 
3. What is $m_t$ in line 715? Is it $\hat{x}_1(x_t)$? If that is indeed the case, it’s more readable to use $\hat{x}_1(x_t)$ instead.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a solver for inverse problems by leveraging pre-trained flow models to produce reconstructions of observed measurements. The proposed solver use pretrained network to predict the target, which is then refined by a proximal step and noise-injection step. The paper provides theoretical analysis showing that the process aproximates bayesian posterior sampling, and provides empirical results on CelebA and AFHQ-cat datasets.

### Strengths
+ The paper proposes a new inverse problem solver that unifies PnP intuition with a Bayesian posterior-sampling narrative.
+ Although I didn't check all the math details presented in the paper, it seems to provide a decent theoretical connection to bayesian posterior sampling using \piGDM-style gaussian approximation.
+ The practicality of this method seems good, with runtime comparable to PnP-Flow and OT-ODE; memory light (tab. 4).

### Weaknesses
- The empirical contribution of this submission, though, seems relatively weak. The main numerical experiments are conducted on two small scale datasets, 100 samples from CelebA and AFHQ-Cat, respectively.
- The overall improvements upon previous methods seem not very significant (in tab. 1 & 2).
- In terms of the limitation of assumptions, i) the paper focuses only on linear problems, and ii) if I understand correctly the proximal step further assumes isotropic prior and measurement operation, which could be oversimplified.
- The most stand-out distinctions from previous flow-based inverse problem solvers are not very clear to me.
- Admittely, I am not very familiar with the current progress of this particular sub-domain. I would therefore also like to hear other colleague reviewers' opinions.

### Questions
Please refer to the weaknesses section for questions if any. Thanks.

### Soundness
2

### Presentation
2

### Contribution
2
