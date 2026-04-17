# Training-Free Reward-Guided Image Editing via Trajectory Optimal Control

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Recent advancements in diffusion and flow-matching models have demonstrated remarkable capabilities in high-fidelity image synthesis. A prominent line of research involves reward-guided guidance, which steers the generation process during inference to align with specific objectives. However, leveraging this reward-guided approach to the task of image editing, which requires preserving the semantic content of the source image while enhancing a target reward, is largely unexplored.
In this work, we introduce a novel framework for training-free, reward-guided image editing. We formulate the editing process as a trajectory optimal control problem where the reverse process of a diffusion model is treated as a controllable trajectory originating from the source image, and the adjoint states are iteratively updated to steer the editing process. Through extensive experiments across distinct editing tasks, we demonstrate that our approach significantly outperforms existing inversion-based training-free guidance baselines, achieving a superior balance between reward maximization and fidelity to the source image without reward hacking.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a novel, training-free framework for reward-guided image editing applicable to both diffusion and flow-matching models. The core innovation is formulating the editing process as a trajectory optimal control problem. This approach treats the reverse diffusion process, starting from a source image, as a controllable trajectory. By leveraging the principles of Pontryagin's Maximum Principle (PMP), the authors develop an iterative adjoint-state optimization algorithm that steers the entire trajectory toward a terminal state that maximizes an arbitrary differentiable reward function, while explicitly minimizing the control effort to maintain fidelity to the source image structure.

The method is validated across four diverse editing tasks: Human Preference Optimization, Style Transfer, Counterfactual Generation, and Text-guided Editing. Experiments show that this trajectory-level optimization approach achieves a superior balance between reward maximization and source image fidelity compared to existing inversion-based guided sampling techniques, effectively mitigating issues like "reward hacking" and structural degradation.

### Strengths
Principled Formulation based on Optimal Control (OC): The paper provides a strong theoretical foundation by framing image editing as a trajectory optimal control problem and deriving the iterative update rules using PMP. This approach is superior to heuristic, step-wise guidance methods as it optimizes the entire path, leading to better structural coherence and a more theoretically grounded way to balance editing (reward) and fidelity (control regularization).

Broad Applicability (Generality): The framework is designed to be training-free and is unified under the Stochastic Differential Equation (SDE) framework, making it applicable to both Diffusion Models (SD 1.5) and Flow-Matching Models (SD 3). This generality significantly broadens its impact and utility.

Superior Fidelity-Reward Trade-off: Quantitatively and qualitatively, the method consistently establishes a dominant Pareto front between reward alignment and source fidelity (Figure 4a). In contrast to baselines which often suffer from structural collapse or reward hacking, this method achieves high rewards while maintaining high LPIPS and $CLIP-I_{src}$ (Tables 1, 2, 3, 4).

Mitigation of Common Failure Modes: The trajectory optimization successfully avoids common failure modes of guided sampling methods, such as reward hacking (where GA introduces severe artifacts) and structural degradation (where stepwise guidance corrupts high-frequency details). The user study (Table 5) confirms the superior perceptual quality and faithfulness of the generated edits.

### Weaknesses
Fundamental Dependence on Differentiable Rewards (Critical Limitation): The entire framework hinges on the reward function $r(\cdot)$ being differentiable, as the adjoint state calculation explicitly requires $\nabla_{x_1} r(x_1)$. This strictly limits its application to objectives that are inherently smooth or can be approximated by differentiable proxies, excluding direct application of discrete or non-differentiable objectives (like direct human feedback ratings). This restricts its practical use in true preference learning settings.

High Computational Cost and Efficiency Concerns: The method is computationally expensive. As shown in Table 8, the required time for the optimization is substantially higher (approximately 80% more time than guided sampling baselines) because the iterative process necessitates multiple, full, backwards and forwards passes over the model's trajectory (proportional to $T$ and $N$). While the results are better, the practical deployment cost is a major constraint.

Sensitivity to Optimization Hyperparameters: The method introduces new critical hyperparameters, notably the inversion depth $T$ (where the optimization starts) and the number of iterations $N$. As shown in Figure 5, the performance is highly sensitive to $T$ (controlling the fidelity-edit trade-off) and $N$ (critical for convergence and stability). The need for careful tuning of $T$ and $N$ for every new task or model reduces the "training-free" simplicity.

Simplistic Trajectory Initialization: The paper primarily relies on deterministic DDIM/ODE inversion for the initial trajectory. While the effects of Markovian (stochastic) initialization are discussed, the deterministic approach, while simple, may place the starting trajectory on a rigid path that is far from the optimal reward direction, potentially leading to slow convergence in complex editing scenarios. Further investigation into optimal reward-aware initialization could be beneficial.

### Questions
above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a training-free framework for reward-guided image editing using trajectory optimal control. The authors regard the image editing procedure as the optimal control of a reverse sampling trajectory from a diffusion or flow-matching generative model. Leveraging Pontryagin’s Maximum Principle, the method iteratively updates adjoint states and control terms to steer the generation trajectory, producing edited images that balance reward maximization with fidelity to the source image. Experiments on various tasks such as human preference optimization, style transfer, counterfactual generation, and text-guided editing demonstrate the superior performance compared to baselines.

### Strengths
- Treating the reverse diffusion process as a controllable trajectory and solve the control problem by using Pontryagin’s Maximum Principle is novel to my knowledge.
- The derivation and mathematical formulation is solid and intuitive. 
- The method is training-free, and can be applied on various base models such as SD1.5 and SD3.
- The paper is clear and well-written.

### Weaknesses
- As stated in the paper, the proposed method requires the reward function to be differentiable and the gradient need to be calculated. Would this cause notable extra computational overhead? Authors are expected to provide the comparison of inference time between the proposed method and baselines.
- Generally, I think the experiments are insufficient, which are still not enough to illustrate the effectiveness of the proposed method:
    1. Authors are expected to provide the results on FLUX, which is one of the most powerful image generation foundation models.
    2. Missing baselines. Currently, there emerges a number of image editing methods based on flow-matching-based models such as [1,2,3], authors should compare the image editing performance between the proposed method and these baselines.
    3. The qualitative results are not enough and some flaws still exists (for example, in second row in Figure 3, the ears of the cat are not clear in the image produced by the proposed method). Authors are expected to provide more comprehensive qualitative results, which are important to evaluate the performance.

### Questions
See weakness.

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
3

### Summary
This paper introduces a novel, training-free framework for reward-guided image editing by reformulating the process as a trajectory optimal control problem. Existing methods (like DPS or FreeDoM) calculate the gradient of the final reward $r(x_1)$ with respect to the posterior mean $\hat{x}_{1|t}$. They then push the noisy intermediate image $x_t$ in that direction. The paper's key insight is to treat the entire path from the noisy starting point $x_T$ back to the clean image $x_1$ as a single, adjustable trajectory. The authors use the principles of PMP to develop an iterative adjoint-state update algorithm, achieving good results.

### Strengths
1. The formulation is noval and techinically sound, bypassing the issue of using posterior mean for approximation.
2. The method works with pre-trained models (both diffusion and flow-matching) without requiring any model fine-tuning.
3. The writing is easy to follow and the logic flows reasonable.

### Weaknesses
1. The proposed method is computationally expensive because it requires a high number of model evaluations.
2. The quality of the output depends on hyperparameters like inversion depth ($T$) and the number of iterations ($N$), which requires careful tuning.

### Questions
1. To calculate the gradient through the reward function, I assume it also requires gradient backprop through the decoder and the diffusion model? How much memory cost is it?
2. Is it possible to us RL (e.g., REINFORCE) to adapt the framework to incorporate non-differentiable feedback signals?
3. Since the method is formulated as trajectory optimal control, what guarantees can be provided about finding the global optimum, or are the iterative updates susceptible to finding poor local minima?

### Soundness
3

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
5

### Summary
This paper introduces a training-free, reward-guided image editing framework built atop pretrained diffusion and flow-matching models. Instead of relying on adversarial gradients or retraining reward models, the authors reformulate reward-guided editing as a trajectory optimal control problem, where the reverse diffusion trajectory is treated as a controllable path. Using Pontryagin’s Maximum Principle (PMP) and an iterative adjoint-state optimization, the method updates control signals to steer the entire trajectory toward higher rewards while preserving source fidelity.

### Strengths
* The paper presents a novel and theoretically grounded formulation of reward-guided editing via stochastic optimal control, bridging generative modeling and control theory.
* The adjoint-based iterative optimization offers a principled alternative to heuristic reward gradients, mitigating adversarial guidance effects.
* Experiments cover diverse tasks and datasets and compare against good baselines..
* The method is model-agnostic, applicable to both diffusion and flow models (via ODE & SDE conversions).
The paper is well written and easy to follow.

### Weaknesses
* The proposed iterative adjoint optimization appears computationally demanding. While FreeDoM and DPS operate in O(n), the proposed trajectory-level optimization is $O(n^2)$, limiting scalability.
* A stronger comparison on efficiency–performance trade-offs is needed (e.g., Pareto plots already hinted at in Fig. 4a). It would help to contextualize the computational gains relative to improvement in reward or fidelity.
* It remains unclear how much the improvement arises from multiple refinements versus the control-theoretic formulation itself. Could repeated low-guidance FreeDoM runs approximate similar results?
* Missing analysis of hyperparameter sensitivity, especially the impact of reward weights and guidance terms, both for baselines and the proposed method.
* Grounded text-editing benchmarks (e.g., PIE-Bench [1]) are absent, which limits the community relevance of the text-guided editing evaluation.
* The paper omits discussion of related optimal-control-inspired methods like FlowChef [4], FireFlow [3], and Taming Rectified Flow [2], which are conceptually close in motivation.

[1] “Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code,” ICLR 2024.

[2] “Taming Rectified Flow for Inversion and Editing,” ICML 2025.

[3] “FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing,” ICML 2025.

[4] “FlowChef: Steering of Rectified Flow Models for Controlled Generations,” ICCV 2025.

### Questions
* What is the wall-clock or compute comparison (FLOPs or inference time) between the proposed method and FreeDoM/DPS? 
* Could the authors provide a Pareto frontier (compute vs. performance) plot across baselines to strengthen the argument of practical utility? 
* How does the proposed method relate to ReNO [5], which performs inference-time noise optimization guided by reward models? * Conceptually, is ReNO equivalent to or distinct from gradient ascent in noise space?
* Could the method be extended to VLM-based editing with text prompts? This seems promising given the reward-agnostic formulation.


[5] “ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization,” NeurIPS 2024.

### Soundness
3

### Presentation
3

### Contribution
2
