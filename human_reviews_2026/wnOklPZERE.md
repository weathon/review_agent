# FlowOpt: Fast Optimization Through Whole Flow Processes for Training-Free Editing

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
The remarkable success of diffusion and flow-matching models has ignited a surge of works on adapting them at test time for controlled generation tasks. Examples range from image editing to restoration, compression and personalization. However, due to the iterative nature of the sampling process in those models, it is computationally impractical to use gradient-based optimization to directly control the image generated at the end of the process. As a result, existing methods typically resort to manipulating each timestep separately. In this work we introduce FlowOpt - a zero-order (gradient-free) optimization framework that treats the entire diffusion/flow process as a black box, enabling optimization through the whole process without backpropagation through the model.
Our method is both highly efficient and allows users to monitor the intermediate optimization results and perform early stopping if desired. We prove a sufficient condition on FlowOpt's step-size, under which convergence to the global optimum is guaranteed. We further show how to empirically estimate this upper bound so as to choose an appropriate step-size.
We demonstrate the effectiveness of FlowOpt in the context of image editing, showcasing two use cases: (i) inversion (determining the initial noise that generates a given image), and (ii) directly steering the edited image to be similar to the source image while conforming to the target text prompt. 
In both settings, our method achieves state-of-the-art results while using roughly the same number of neural function evaluations (NFEs) as existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a method for image inversion and editing with flow models. The idea is to optimize a $z_t$ (typically $z_T$) to reconstruct the input image. Since it is not feasible to propagate gradients through the entire denoising process, the optimization update omits the Jacobian term. So the update step becomes $z_0^{(i)} - y$, where $z_0^{(i)}$ is the image generated from the current state of the optimization, and $y$ is the input image.
The optimization uses a small learning rate, and the paper shows that if the lr is not small enough, this process does not converge.

### Strengths
- The method is novel, and it is initially surprising that it works. The authors provide an analysis and theoretical justification (but I do have concerns regarding the theoretical part, see weaknesses section).
- The method itself is simple, and the paper presentation is clear.
- The authors performed extensive evaluations against competing methods and the results are plausible (but I do have concerns here, see weaknesses section).
- The limitations of the method are clearly discussed in the Appendix.
- The method's results seem to adhere to the provided edit while staying well aligned with the original image in cases where competing methods fail.

### Weaknesses
### Major Concerns
1. The method requires a relatively large number of NFEs in order to provide an advantage over existing methods (e.g., FireFlow and UniInv) in reconstruction. 

2. The authors present a theorem that guarantees the method's convergence under certain assumptions, however why and if these assumptions hold in practice is not clear. In addition, I think that the proof itself in Appendix F is potentially flawed, as explained next. 
Even assuming the condition holds, for the proof to hold we need to show that there exists some fixed $\kappa > 0$ such that the range in Eq. S8 exists. Otherwise, the limit argument is invalid for this claim. 
Furthermore, there exist many functions for which the condition holds, yet for any fixed $\kappa$ the range doesn't exist. Examples include $\tanh(x)$ where the supremum of $u_1$ and $u_2$ in the expression of $\eta_1$ is infinite and $x^3$ where the infimum of $u_1$ and $u_2$ in the expression of $\eta_2$ is $0$. Both functions satisfy the required condition with $\beta = 1$.

3. While the method compares with relevant inversion-based editing methods, there are also other approaches for text-based image editing, and it is not clear that the general framework (inversion + denoising with a different prompt) is the most effective one. For example, the method is not compared with Flux Kontext or Qwen Image Edit which are the SOTA text-based image editing models.

### Minor Concerns
- Assuming that the Theorem holds, from the results in Appendix C it seems that the convergence is very slow, and in practice the initialization is crucial for the success of the method. It would be interesting to analyze convergence and performance when using other initializations, such as random noise or an interpolation between random noise and the final image. 
- Analysis of performance on few-step models is missing, even though they are potentially strong candidates to benefit from this method. 
- The method seems to support only appearance changes.
- Why ReNoise is not included in the editing results? And no visual results of reconstruction are provided.
- Showing other applications for this optimization framework will strengthen the paper.

### Final Note
Despite these weaknesses, I find the paper overall good. I would be willing to raise my score if the authors address the issues related to the convergence claims and provide a more thorough discussion of the origins of the method's limitations.

### Questions
I would like to see more experiments that empirically support the claim of convergence to a unique solution from different initial conditions. If these cannot be provided, I would suggest the convergence guarantee claims to be removed from the paper.

Methods that involve noise optimization, even gradient-free ones, can often produce inverted latents that don't exhibit properties of typical high dimensional Gaussian samples. Such properties may limit the editability of images generated by these latents (see e.g ReNoise, where the authors try to tackle this issue with regularization during optimization). I would like to see an analysis of the properties of the inverted latents found by this method, which may explain some limitations in editability, and perhaps hint towards a future solution for these limitations.

The limitation for pose editing as presented in figure S16 is counter-intuitive. I would expect that using a larger number of optimization steps would make the edited image deviate less from the original image (as is seen in Figure 4 and Figure S17), and not the other way around.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the task of editing images (and potentially other generative tasks)  using pre-trained flow/diffusion models in a gradient-free manner. The key idea being, treating the entire sampling process as a "black box" instead of tweaking each sampling step individually (which is the case with many existing approaches) and using a zero-order optimisation approach.

### Strengths
The paper is very well written. 
1. This paper presents a clean idea of optimising the whole process rather than per-timestep manipulation. 
2. The paper also presents a theoretical contribution: i.e., a sufficient condition on the step-size for convergence of the opimizer in this setting. 
3. The edits looks visually appealing and demonstrate a good tradeoff between fidelity and edit strength.

### Weaknesses
1. Although the paper compares methods quantitatively and qualitatively, a user study is missing.

2. Paper doesn't really discuss how the Zero-order method performs with increase/decrease in dimension since zero-order methods may suffer from bad convergence with increase in dimension.

### Questions
1. Why have the authors not compared against a gradient-based inversion baseline?

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
5

### Summary
This paper presents FlowOpt, a zero-order (gradient-free) optimization framework for training-free image editing with pretrained diffusion and flow models. Instead of backpropagating through the model or optimizing per timestep.
That said, this work is exactly similar to existing literature, particularly FlowChef, and lacks comprehensive and community-standard evaluations. See details below.

### Strengths
* Theorem 1 provides a sufficient condition on the step size under which the FlowOpt iterations provably converge. This formal analysis of convergence is a valuable addition to flow-based optimization literature, where most prior methods rely on heuristic step-size tuning.

### Weaknesses
The novelty is limited. The proposed zero-order optimization across the full flow process is conceptually identical to FlowChef [1] (ICCV 2025, arXiv Dec 2024), which already introduced a gradient-free control framework with theoretical guarantees and broad task coverage (inversion, editing, and restoration). The main difference, introducing a step-size bound, is a modest theoretical insight rather than something novel or different.

The work lacks comprehensive evaluation on community-standard editing benchmarks such as PIE-Bench [2], which is now widely adopted for fair comparison across inversion-based and inversion-free methods.

The paper doesn’t clarify the conceptual distinction between FlowOpt and FlowChef, despite their almost identical formulations (both optimize the initial latent by approximating the flow trajectory without backpropagation).

[1] “FlowChef: Steering of Rectified Flow Models for Controlled Generations,” ICCV 2025.
[2] “Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code,” ICLR 2024.

### Questions
Can the authors clearly articulate the difference between FlowOpt and FlowChef, both theoretically and empirically?

### Soundness
1

### Presentation
2

### Contribution
1
