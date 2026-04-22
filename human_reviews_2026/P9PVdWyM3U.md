# Policy Contrastive Decoding for Robotic Foundation Models

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 8, 4

## Abstract
Generalist robot policies, or robotic foundation models, hold immense potential to enable flexible, general-purpose and dexterous robotic systems. Despite their advancements, our empirical experiments reveal that existing robot policies are prone to learning spurious correlations from pre-training trajectories, adversely affecting their generalization capabilities during inference. To tackle this, we propose a novel Policy Contrastive Decoding (PCD) approach, which redirects the robot policy’s focus toward object-relevant visual clues by contrasting action probability distributions derived from original and object-masked visual inputs. As a training-free method, our PCD can be used as a plugin to improve different types of robot policies without needing to finetune or access model weights. We conduct extensive experiments on top of three open-source robot policies, including the autoregressive policy OpenVLA and the diffusion-based policies Octo and Pi-0. The obtained results in both simulation and real-world environments prove PCD’s flexibility and effectiveness, e.g., PCD enhances the state-of-the-art policy $\pi_0$ by 8.9% in the simulation environment and by 108% in the real-world environment. Our code is publicly available at: https://github.com/pcd-robot/PCD.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Policy Contrastive Decoding (PCD), a training-free plugin for pretrained robotic foundation models. It aims to redirect pre-trained robotic foundation models away from spurious visual correlations toward object-relevant cues. The key idea is: given a visual observation and language instruction, the robot policy is run twice on the original image and on a version where task-relevant objects are masked. It can be applied to both autoregressive and diffusion-based policies, showing its versatility.

### Strengths
1. The method does not need retraining or fine-tuning of a large robot policy, showing practicality in robotics, as VLA training is expensive.

2. The problem that the paper tries to solve is important: reliance on spurious visual correlations leading to poor generalization when distribution shifts happen.

3. The authors evaluate across multiple policies (i.e., autoregressive and diffusion-based) and in both simulation and real-world robot settings. They also include ablations, which help to analyze design choices.

### Weaknesses
1. It lacks some analysis for hyperparameters.

2. It does not include training and implementation details.

3. The paper defines “spurious correlations” formally (Definition 3.1) but then moves to a generic contrastive decoding mechanism without carefully clarifying which types of spurious correlations are mitigated and which are not. For instance, masking the object may remove task-relevant cues if the object is part of the visual context, yet the method tacitly assumes that the object is the only relevant cue and everything else is spurious. This assumption is too strong for many robotics settings.

4. The ablations, while present, focus mainly on hyperparameter $\alpha$, object detection model choice, and inpainting method. There is little ablation on the magnitude of spurious correlation removal, or how performance degrades when masking fails.

### Questions
1. What is the value of $\alpha$ in the main simulation experiment? Is it fixed or adapted to the VLAs?

2. For diffusion-based policies with KDE-PM, why do the authors choose 24 for experiments? Do authors have any time/memory trade-offs for this method compared to standard inference?

3. When the PCD is applied to VLAs, the success rate is improved but can not reach 100%, nor even 80% in most tasks. Why does it fail? Is it because PCD can not fully solve the spurious correlations problem? 

4. How does the performance of PCD degrade when object detection or masking fails (e.g., false negatives, multiple objects, occlusion)? Can authors provide results showing robustness to mask quality or missing objects?

5. It seems that all the experiments were completed from a single perspective. How about multiple perspectives?

### Soundness
3

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
This paper proposes "Policy Contrastive Decoding" (PCD), a novel method to address the issue of "spurious correlations" in robotic foundation models, where policies incorrectly learn from task-irrelevant features like background or lighting instead of the target objects. PCD is a "training-free" and "plug-and-play" approach that works by contrasting the action probability distribution from the original visual input with a distribution derived from an "object-masked" version of the same input. This process, which requires no model retraining, amplifies the policy's focus on object-relevant features. The method is designed to be flexible, using a "Tracking-to-Mask" (Track2Mask) strategy to identify and mask objects automatically, and a "KDE-based Probabilistic Modeling" (KDE-PM) scheme to make it compatible with both autoregressive (like OpenVLA) and diffusion-based (like Octo and $\pi_0$) policies. Experiments in simulation and the real world show that PCD significantly improves the success rates of these state-of-the-art policies, enhancing the $\pi_0$ policy by 8.9% in simulation and 108% in the real world.

### Strengths
- The paper identifies and addresses a critical, known weakness in robotic foundation models: their tendency to learn "spurious correlations" (e.g., relying on background or lighting) rather than the actual task-relevant objects, which hurts generalization.

- The proposed "Policy Contrastive Decoding" (PCD) is a major strength because it is a "plug-and-play" solution. It can be applied to existing, pre-trained policies without requiring any costly retraining, fine-tuning, or access to the model's internal weights.

- The method is not designed for just one type of model. It introduces a "KDE-based Probabilistic Modeling" (KDE-PM) scheme that allows it to work with *both* autoregressive policies (like OpenVLA) and diffusion-based policies (like Octo and $\pi_0$).

- PCD demonstrates significant performance gains on top of three different state-of-the-art policies. It achieves impressive improvements, boosting the strong $\pi_0$ baseline by 8.9% in simulation and, most notably, by 108% in real-world tasks.

- The paper includes extensive experiments to prove its claims, and it is open-sourced.

### Weaknesses
- The method adds significant computational overhead and latency at inference time. The paper notes it "approximately doubles the inference latency per step" and adds a 24% total time cost to real-world tasks. This may influence the performance on highly dynamic tasks,

- The method introduces a new, highly sensitive hyperparameter, $\alpha$, that controls the strength of the correction. Ablation studies show that the best $\alpha$ is different for each model (e.g., 0.2 for $\pi_0$ but 1.0 for Octo), meaning it requires careful, model-specific tuning, which undermines the "plug-and-play" claim.

- The authors admit the method is a patch at inference time, not a fix. It only addresses the symptoms of spurious correlations during testing and does nothing to prevent the model from learning them during training in the first place.

- The real-world experiments, while showing a large relative gain (108%), were conducted on a $\pi_0$ model that was already fine-tuned on 10 demonstrations for each specific task. This makes it unclear how PCD would perform on a general-purpose model in a true zero-shot, real-world setting. Also, the performance gain of 108% in real-world compared to the 8.9% in the simulation is very large, which seems unusual.

- Even with PCD, the absolute success rates in many challenging real-world tasks remain low (e.g., "Stack Cube" improves from only 5% to 10%), suggesting it is not a complete solution for robust, real-world generalization.

### Questions
The proposed Policy Contrastive Decoding (PCD) method, which contrasts outputs from original ($o_i$) and object-masked ($\hat{o}_i$) observations, bears a strong conceptual resemblance to Classifier-Free Guidance (CFG) in diffusion models. In CFG, guidance is achieved by steering the generation process away from a null or unconditional prompt and toward a conditional prompt.

Given this parallel, have the authors considered implementing a more direct CFG-based approach for the diffusion-based policies (Octo and $\pi_0$)? Specifically, one could treat the object-masked input ${\hat{o_i}}$ as the "negative condition" (analogous to the unconditional prompt in standard CFG) and the original observation $o_i$ as the positive condition. The denoising network $\epsilon_{\theta}$ could then be guided using the standard CFG formulation, such as $\hat{\epsilon_{\theta}} = \epsilon_{\theta}(a_i^k, e_i, k) + w(\epsilon_{\theta}(a_i^k, e_i, k) - \epsilon_{\theta}(a_i^k, \hat{e}_i, k))$, where $\hat{e}_i$ is the embedding from the object-masked input and $w$ is a guidance scale.

This CFG-based approach would be a more native and direct way to apply guidance within the diffusion framework, potentially obviating the need for the proposed KDE-based Probabilistic Modeling (KDE-PM) scheme. Could the authors comment on whether they explored this alternative? If so, how did its performance and computational efficiency compare to the proposed PCD method? If not, do they foresee any challenges with such an implementation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Policy Contrastive Decoding (PCD), a training-free method that improves generalist robot policies at inference time by comparing the action distributions from the original observation with those from an object-masked observation. The object mask is obtained using a Track2Mask pipeline. The authors also introduce a KDE-based method to estimate action probabilities for diffusion policies. Experiments show that PCD improves OpenVLA, Octo, and Pi0 on SIMPLER simulation tasks, and Pi0 on a real-world pick-and-place task.

### Strengths
- The method is a training-free approach that can be applied to various generalist policies in a plug-and-play manner.
- The evaluation on both SIMPLER and real-world tasks shows improvement over the base policy.

### Weaknesses
- The method is mainly limited to simple pick-and-place tasks with a clear target object to mask, and it is hard to extend to more complex tasks that need long-horizon planning or involve multiple objects.

- The method introduces additional computation and latency, both from running Track2Mask and from the KDE-PM process. This is discussed in Appendix A.5. It may not be a major issue for the pick-and-place tasks studied in this paper, but it could be a limitation for tasks that require fast inference.

- In Table 1, the performance seems to vary depending on the object annotation strategy, and it is a sensitive design choice for each task and base policy. This requires the user to perform additional tuning to find the best setting.

### Questions
- At a high level, the method seems to be trying a similar idea as classifier-free guidance (CFG) in the diffusion model literature. Can the authors discuss more about this relation? In addition, can the authors try evaluating the diffusion policy with CFG instead of PCD + KDE-PM?  Since there is already the original observation $o$ and the object-masked observation $\hat{o}$, we can simply try a CFG-based action sampling by predicting the noise with a weighted combination of the two predictions
```
eps_uncond = model(x_t, t, \hat{o})
eps_cond   = model(x_t, t, o)
eps = eps_uncond + w * (eps_cond - eps_uncond)
```
- The paper mainly studies the use case for generalist policies, but will this method also work on smaller single-task policies trained on limited data? It might be interesting to train a smaller diffusion policy on the real-world dataset used in the experiments and test it out.
- Why does the performance differ so much between each object annotation strategy? What are the failure cases of Track2Mask?


I’d be happy to reconsider my score once these points are addressed.

### Soundness
3

### Presentation
2

### Contribution
2
