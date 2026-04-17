# Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Vision-Language-Action (VLA) models pre-trained on large, diverse datasets show remarkable potential for general-purpose robotic manipulation. However, a primary bottleneck remains in adapting these models to downstream tasks, especially when the robot's embodiment or the task itself differs from the pre-training data. This discrepancy leads to a significant mismatch in action distributions, demanding extensive data and compute for effective fine-tuning. To address this challenge, we introduce Align-Then-stEer (ATE), a novel, data-efficient, and plug-and-play adaptation framework. ATE first aligns disparate action spaces by constructing a unified latent space, where a variational autoencoder constrained by reverse KL divergence embeds adaptation actions into modes of the pre-training action latent distribution. Subsequently, it steers the diffusion- or flow-based VLA's generation process during fine-tuning via a guidance mechanism that pushes the model's output distribution towards the target domain. We conduct extensive experiments on cross-embodiment and cross-task manipulation in both simulation and real world. Compared to direct fine-tuning of representative VLAs, our method improves the average multi-task success rate by up to 9.8% in simulation and achieves a striking 32% success rate gain in a real-world cross-embodiment setting. Our work presents a general and lightweight solution that greatly enhances the practicality of deploying  VLA models to new robotic platforms and tasks. Our code is released at \url{https://github.com/TeleHuman/Align-Then-Steer}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Align-Then-Steer (ATE), a framework for adapting pre-trained Vision-Language-Action (VLA) models to new embodiments and tasks.
ATE addresses the action-space mismatch problem when transferring VLAs across robots or manipulation domains. It consists of two stages:
(1) Align – construct a unified latent action space by aligning source and target action distributions using a dual-VAE with reverse KL regularization;
(2) Steer – apply latent guidance to direct diffusion or flow-based VLA generation toward the aligned latent manifold during fine-tuning.
This approach enhances adaptation efficiency, preserves pre-trained capabilities, and achieves strong results in both simulation and real-world settings.

### Strengths
1. The motivation is clear and well grounded.

2. The method shows significant improvements in adaptation success rates.

3. The innovation is moderate, but the approach is scalable and extensible.

4. The paper is well organized and easy to follow, with clear figures and mathematical formulations.

### Weaknesses
See questions.

### Questions
1. I recently came across latent action learning in a survey paper [1]. How does the latent action learning proposed in your work differ from that?

2. This question is more discussion-oriented — I also believe latent learning is a promising approach for cross-embodiment transfer. What other potential solutions do you foresee in the future? And compared to [2], what advantages does your method offer?

3. If there is a large gap between pre-training and fine-tuning data, will this alignment still be effective? How robust is the latent alignment when the target robot has a significantly different kinematic structure or action semantics?

4. How does the reverse KL constraint affect convergence? Did you observe mode collapse or over-constrained behavior during training?

5. What is the model scale, and can it run in real time for control tasks?

6. Are there visualizations (e.g., t-SNE plots) showing how the latent distributions align before and after training?

7. Possibly due to the page limit, some ablation studies are placed in the appendix. However, this is not ideal — key experimental results should be presented in the main text.

[1] Towards a Unified Understanding of Robot Manipulation: A Comprehensive Survey. arXiv 2025.

[2] villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models. arXiv 2025.

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
3

### Summary
Align then Steer (ATE) provides a sample-efficient manner to adapt into a new embodiment for pretrained Vision-Language-Actions Models. ATE first trains a VAE that learns a latent space of all actions in the pretraining dataset, $z$, and uses it as a reference for the adaptation dataset. Afterwards, the method then trains another VAE to learn a separate latent for the adaptation dataset. The novelty came from aligning the two distributions together using a reverse KL penalty. Afterwards, for effective policy adaptation, one can use classifier guidance as an additional objective in the adaptation loss.

Via both real-world and simulated benchmarks, ATE achieved considerable improvements in performance across dexterous tasks across various forms of disturbances and difficulties as well as faster convergence.

### Strengths
1. The authors conducted an extensive suite of real-world and simulated experiments, which does bolster the claim that the paper is making. I like that the author made a lot of validation on the robustness of the experiments by checking ATE against various forms of environment perturbations.
2. I believe that this type of alignment procedure in the finetuning stage is also doing something similar to that of RL. I believe that you can follow the same line of thinking proposed in CFGRL [1] and show that this form of classifier guidance in finetuning is a form of policy improvement. 

References:

[1] Frans, K. et al., 2025. “Diffusion Guidance Is a Controllable Policy Improvement Operator”. ArXiv.

### Weaknesses
1. While the paper shows considerable improvement in aggregate, each task is only evaluated with 5 trials (correct me if I’m wrong). If possible, it is better to use a larger number of trials per task (10-20) so that there is less variance in your experiment procedure. 
2. The method section is pretty dense in notation. I believe that the paper can be reformatted for better readability.

### Questions
1. How does ATE compare to other guidance methods such as DSRL or DynaGuide (although their formulations are different)?
2. Can authors also discuss how the sample efficiency of this method stack up compared to the baseline methods across various finetuning set sizes? I believe this can strengthen the claim that ATE allows effective finetuning without large scale data.

Minor questions and remarks:

1. On page 8 (line 410), there is a missing link to the appendix. 
2. The paper is pretty dense in formulae and notations (especially in the preliminary and method section). It would be better to move some of them to the appendix for presentation.
3. In eq. 5, $\mathcal{D}$ is not defined (as compared to $\mathcal{D}_\text{pretrain}$. What does it stand for?
4. On line 266, “we introduce the classifier guidance” should have “the” removed.
5. In figure 3, I think there’s a mismatch between the figure’s numbering system (6w, which I presumed to be 60k) and the description. It would be better to check these typos and correct them as well in the review period.

References: 

[1] Liu, B. et al., 2023. “LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning”. NeurIPS

[2] Wagenmaker, A. et al., 2025. “Steering Your Diffusion Policy with Latent Space Reinforcement Learning”. CoRL.

[3] Du, M. et al., 2025. “DynaGuide: Steering Diffusion Polices with Active Dynamic Guidance”. ArXiv.

### Soundness
3

### Presentation
2

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
This paper proposes two-step framework, namely ATE, on efficient robot learning on robot manipulation tasks. ATE consists of two sub-modules: (i) a dual-VAE module for robot action chunk pre-training and downstream action chunk adaptation, where the embedding space of the latter one is trained to align with the pre-trained VAE. (ii) a steering module for diffusion- and flow-based VLA models. ATE is a plug-and-play framework and could be directly applied to different VLA models. Simulation and real-world experiments demonstrate consistent performance gain w.r.t diffusion- and flow-based models.

### Strengths
1. The proposed method, including the dual VAE and steer module is theoretically sound.
2. The size of ATE is relatively small w.r.t base models like pi_0, yet it brings consistent performance gain over the base models.
3. Extensive experiments make ATE technically solid.

### Weaknesses
I found this paper difficult to follow. Please refer to the questions.

### Questions
Questions:
1.	It seems that there lacks necessary (and straightforward) illustration of how diffusion- and flow-based methods like pi_0 work, and how ATE is trained/integrated with these models. 
2.	What’s the necessity of training two VAEs? Once the first VAE is pre-trained using large-scale datasets like OpenXE, it could theoretically transfer and fit-in to a smaller downstream dataset. The motivation could be more clearly illustrated (even though it is mentioned at line 254).
3.	Is it possible to directly apply the steering module to other latent-based methods (like Q.2)? Or, is it strongly tied to work with the proposed InfoVAE?
Suggestions:
1.	Font size of Figure 1 and Figure 2 is too small to clearly recognize the text.
2.	Line 410: missing reference
3.	Line 457: ATE in inconsistent font.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes ATE, a plug-and-play adaptation framework for vision-language-action (VLA) models. Stage 1 “Align” learns a unified action latent space by training an InfoVAE on pre-training actions and a second InfoVAE on target-domain actions with a reverse-KL regularizer toward the pre-training latent distribution. Stage 2 “Steer” adds a training-time guidance term that pushes generated action chunks toward target-domain actions by penalizing the squared distance between their latent encodings. The method claims minimal architectural changes and low overhead.
The method claims minimal architectural changes and low overhead. It is evaluated on RoboTwin 1.0 (17 tasks), ManiSkill3 (2 tasks), and a real dual-arm RealMan platform (4 tasks + 1 tool-use). 
It achieves 9.8% and 32% success rate improvement on Simulation and real-world setting, respectively. There is also a qualitative claim of smoother, safer motions (force plots).

### Strengths
- Tackles an important and practical problem: efficient adaptation of pretrained VLAs across embodiments and tasks.
- Simple, model-agnostic recipe that integrates with both diffusion and flow matching, with no architectural changes.
- Clear implementation description and algorithm boxes; the method is plausibly easy to reproduce and deploy.
- Writing is good and the paper is easy to follow.
- The improvements look good, and it seems like a good RL alternative.

### Weaknesses
- I feel like the proposed method is more like a self-distillation method which shares similar objective with RL. I wonder whether it can compete with existing offline RL methods. Now the model did not show any other baseline methods besides the model it based on so it is hard to tell.

### Questions
Misc:
Missing Appendix ref in L410

### Soundness
3

### Presentation
3

### Contribution
2
