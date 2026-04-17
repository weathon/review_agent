# Contractive Diffusion Policies

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Diffusion policies have emerged as powerful generative models for offline policy learning, whose sampling process can be rigorously characterized by a score function guiding a Stochastic Differential Equation (SDE). However, the same score-based SDE modeling that grants diffusion policies the flexibility to learn diverse behavior also incurs solver and score-matching errors, large data requirements, and inconsistencies in action generation. While less critical in image generation, these inaccuracies compound and lead to failure in continuous control settings. We introduce **C**ontractive **D**iffusion **P**olicies (CDPs) to induce contractive behavior in the diffusion sampling dynamics. Contraction pulls nearby flows closer to enhance robustness against solver and score-matching errors while reducing unwanted action variance. We develop an in-depth theoretical analysis along with a practical implementation recipe to incorporate CDPs into existing diffusion policy architectures with minimal modification and computational cost. We evaluate CDPs for offline learning by conducting extensive experiments in simulation and real world settings. Across benchmarks, CDPs often outperform baseline policies, with pronounced benefits under data scarcity. Project page: https://contractive-diffusion.github.io

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies a weakness in modern diffusion policies. The authors argue that the iterative nature of the diffusion sampling process introduces compounding errors from two sources 1) inaccuracy of the learned score function, especially in low-data cases, and 2) discretization errors from the ODE solver. In control and robotics, these small and accumulating errors can lead to failed actions.

To solve this, the authors introduce Contrastive Diffusion Policy (CDPs). The core idea is to leverage contraction theory to make the reverse diffusion ODE more stable. The contraction theory establishes the relation between the ODE's Jacobian and the ODE. If this condition is satisfied, it can guarantee that the ODE trajectories converge to a more concentrated target distribution. This property makes the learned ODE more concentrated and the sampling process inherently robust to small perturbations like solver or model errors. 

This paper leverages the theoretical analysis of contraction theory for ODE and modifies the original diffusion or flow policy by forcing the ODE's Jacobian to satisfy the contrastive drifting condition. This is learned along with the original flow objective. For the training details, they use the power iteration method to approximate the largest eigenvalue of the score Jacobian. Then a contraction loss is applied to penalize the model to be more concentrated. The final training loss is a simple weighted sum of the original diffusion loss and the new contraction loss. 

Experiments on D4RL, Robomimic, and real-world Franka robot tasks demonstrate that CDPs outperform the standard diffusion policy baseline, with particularly strong performance in low-data scenarios.

### Strengths
- The paper leverages a solid foundation in contraction theory and gives a formal analysis of the ODE's stability, which is insightful and provides a clear, principled target for regularization
- The paper has a good motivation. Compounding error is a valid issue in continuous control
- The paper includes physical robot experiments in the real world, which demonstrates its robustness in practice

### Weaknesses
- The paper describes the added cost as "minimal" and "negligible." However, the proposed method requires computing a Jacobian-vector product (K=3 or 4 times for the power iteration) for every item in the batch at every single training step. This is definitely more computationally expensive than a standard diffusion loss. A more transparent analysis (e.g., a wall-clock time comparison) would be helpful to quantify this overhead.
- The advantage of diffusion models is to capture complex, multi-modal distributions (like the different ways to perform a task). Forcing the system to be "contractive" (pulling trajectories together) seems intuitively to contradict preserving this diversity. The paper mentions that "excessive penalization could fuel a mode collapse," but does not investigate this trade-off in-depth. A quantitative study on how $\gamma$ affects the variance or modality of the final action distribution would make the paper more complete, especially on a task showing multi-modal behavior.
- The authors acknowledge this limitation of hyperparameter sensitivity. The new contraction loss weight, $\gamma$, is a critical hyperparameter. As shown in Table 4, the optimal value for $\gamma$ varies dramatically across different environments (from 0.001 to 100.0). This high sensitivity could make the method difficult to apply to new tasks without a costly hyperparameter sweep

### Questions
In addition to the questions in the weakness above, I have the following additional questions 

- Regarding Stochastic (SDE) Solvers: The paper focuses on stabilizing the deterministic ODE sampling process (and the experiments appear to use the deterministic ODE sampler). However, stochastic samplers (like DDPM) are well-known to have error-correcting properties due to their Langevin noise term, which also tends to concentrate the final distribution.
Could the authors comment on why an ODE-based solver was chosen over an SDE-based one?
Did the authors experiment with stochastic samplers? If so, how do they perform? Does the inherent noise of an SDE sampler already provide a similar "contractive" effect, making the explicit contraction loss less necessary?

- Regarding Simpler Alternatives (e.g., Data Re-weighting): As an alternative to regularizing the model, could a similar "concentrating" effect be achieved through the training data or loss function? For example, one could identify actions that are "central" or "high-quality" (e.g., near the mean of a mode in the data) and apply a higher loss weight to these samples during training. This would intuitively "tilt" the learned score function to flow towards these more robust actions, without the computational cost of Jacobian-vector products. Could the authors comment on the pros and cons of their proposed method versus such a data re-weighting or distribution tilting approach?

- In Table 2 (Robomimic), the proposed CDP (0.78 avg) underperforms the DP-Unet baseline (0.88 avg) significantly. Could the author provide additional results with CDP + Unet to show the improvements over different architectures?

- I'm not quite sure why the method performs better for low-data regimes. if the score field ε_θ is inaccurate due to sparse data, why does forcing its Jacobian to be contractive (based on this inaccurate field) lead to a better outcome? Is there a theoretical explanation for this? 

*Typo: line 246 J_{\epsilon_\theta}^{sym} seems undefined?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper is improving general diffusion model policies through an additional class of losses that promote contractions in the sampling processes of the diffusion model policy. These losses are denoted Contractive Diffusion Policies (CDPs), which add a Jacobian-based loss so the reverse diffusion dynamics are contractive. The goal of this is to damp solver and score noise and reduce action variance. Experiments are done on D4RL, Robomimic, and a small real-robot setup.

### Strengths
- Simple idea that integrates into standard diffusion-policy training without major architectural changes.
- Well-presented theory around contraction for ODE’s and diffusion (or more generally flow-based) generative models.
- Some positive results on D4RL benchmarks, especially where reducing variance helps.

### Weaknesses
- Contraction seems to fundamentally contradict a central reason practitioners use diffusion model polices over e.g. Gaussian policies, namely multimodality. By pulling trajectories together, CDP can collapse valid modes that diffusion policies are meant to capture. It then seems in this setting that Gaussian policies would need to be benchmarked against, or at the very least explanation provided as to why either (a) multimodality is actually not that important for RL policies, or (b) why the contraction loss does not actually eliminate multimodality when it’s efficient for optimal policies to inhibit this.


- There is no direct evidence that the spectral condition for contraction actually holds during training and sampling. Plots of eigenvalues through various time conditioning would give future readers a better idea of the effect of the contraction loss, and what parts of the diffusion model become more strongly contractive.

- As Jacobian computations are typically expensive, a computation time analysis here seems relevant.


- While some environments show CDP giving a non-negligible boost, none of the robomimic experiments show CDP giving a clear advantage over other methods. Additionally, some appendix figures are difficult to interpret because means and confidence intervals overlap heavily (e.g. Fig. 15). This makes it hard to assess some practical differences.

### Questions
- How much action diversity is lost as the contraction weight increases? Is there a way to reconcile contraction-type losses with max-entropy RL?


- What is the evidence that diffusion model policies suffer from the same score matching and integration errors that have been observed in e.g. image sampling?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel diffusion policy called contractive diffusion policy. Comprehensive numerical experiments are conducted.

### Strengths
1. The approach has strong theoretical foundations. The authors clearly provide conditions for contractive diffusion policy by leveraging the results in ODE theory and diffusion models.
2. The numerical results are very comprehensive. The paper presents results across multiple benchmarks, showing the effectiveness of the proposed approach. 
3. Writing is easy to follow.

### Weaknesses
Although I am familiar with diffusion models theory, I am not familiar with diffusion policy benchmarks. I only have one major concern:
1. In Tang and Zhao (2024), the contractive condition excludes the application of usual diffusion model such as VP SDE. Do you face the same issue in your setup? If yes, what diffusion process are used and how to you compare with usual DDPM-based sampling process?

### Questions
The paper is clear to me. No further questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes to improve diffusion policy in the continuous control setup by inducing contractive behaviors. While the iterative sampling process of diffusion models encourages diverse outputs, it also hinders action generation in fields that require accurate signals (e.g., robotic control). The method introduces a penalty loss term on the score Jacobian to enhance robustness in the sampling process, as it effectively reduces action variance. The authors show that this offline learning method can improve the diffusion policy baseline on several robotic control problems, especially in the limited training data regime.

### Strengths
+ The method is simple and intuitive. The results look promising (especially the real-world experiments).
+ The authors provide an in-depth theoretical analysis of their proposed method for inducing the contraction of diffusion models.
+ The work is positioned well in a detailed discussion of related work.

### Weaknesses
+ I am not fully convinced of the idea of unwanted action variance. Granted, excessive noise in the generated actions can lead to failure in continuous control. But the level of variance that is beneficial varies a lot from task to task, as pointed out in the appendix. It would be better if the authors could come up with a more principled tuning method (e.g., a self-adaptive coefficient). In particular, for tasks that require dynamic control (say, soft-body manipulation), it might be hard to justify the contraction-based regularization.
+ I suspect that it might be harder for the contraction-regularized diffusion model to do transfer learning or further online fine-tuning. Basically, with the reduced action variance, it might be harder to adapt to various action distributions in a post-training manner. While the promise of a large diffusion policy for action generation lies in a foundation model + quick post-training fine-tuning paradigm, I wonder if the proposed approach might hinder this.
+ The Jacobian (or in general Lipschitz) regularization approach has been popularized by a series of works in DNN generalization [1,2,3]. Please add them to the related work section.

I am willing to raise my score after seeing the authors’ response.

[1] Information-theoretic local minima characterization and regularization, ICML 2020

[2] Sharpness-aware minimization for efficiently improving generalization, ICLR 2021

[3] Understanding Gradient Regularization in Deep Learning: Efficient Finite-Difference Computation and Implicit Bias, ICML 2023

### Questions
Does the distillation process (e.g., consistency policy) effectively also induce a contraction? As the reduced sampling steps and the consistency distillation loss might also “pull” nearby diffusion flows closer. I suggest the authors study their proposed loss on distilled diffusion policies as well.

### Soundness
3

### Presentation
3

### Contribution
2
