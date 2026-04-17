# When would Vision-Proprioception Policies Fail in Robotic Manipulation?

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 8

## Abstract
Proprioceptive information is critical for precise servo control by providing real-time robotic states. Its collaboration with vision is highly expected to enhance performances of the manipulation policy in complex tasks. However, recent studies have reported inconsistent observations on the generalization of vision-proprioception policies. 
In this work, we investigate this by conducting temporally controlled experiments. We found that during task sub-phases that robot's motion transitions, which require target localization, the vision modality of the vision-proprioception policy plays a limited role. Further analysis reveals that the policy naturally gravitates toward concise proprioceptive signals that offer faster loss reduction when training, thereby dominating the optimization and suppressing the learning of the visual modality during motion-transition phases. 
To alleviate this, we propose the Gradient Adjustment with Phase-guidance (GAP) algorithm that adaptively modulates the optimization of proprioception, enabling dynamic collaboration within the vision-proprioception policy. Specifically, we leverage proprioception to capture robotic states and estimate the probability of each timestep in the trajectory belonging to motion-transition phases. During policy learning, we apply fine-grained adjustment that reduces the magnitude of proprioception's gradient based on estimated probabilities, leading to robust and generalizable vision-proprioception policies. 
The comprehensive experiments demonstrate GAP is applicable in both simulated and real-world environments, across one-arm and dual-arm setups, and compatible with both conventional and Vision-Language-Action models. We believe this work can offer valuable insights into the development of vision-proprioception policies in robotic manipulation.
Videos and code are available at https://gewu-lab.github.io/GAP/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper claims that in vision-proprioception policies, the proprioceptive modality, which is relatively easier to learn, suppresses the visual modality during the motion-transition phases of a task. The authors propose a gradient adjustment algorithm that effectively addresses this issue, supported by well-designed experiments in both real and simulated environments. In particular, the proposed method first performs an initial segmentation of the task's motion-transition phases using a Change Point Detection (CPD) algorithm on proprioceptive data. It then uses a temporal model, such as an LSTM, to refine this and calculate the probability of each time step belonging to a motion-transition phase. Based on this probability, it then applies dynamic gradient adjustment to regulate learning from proprioception, thereby preventing the suppression of the visual modality. The authors validate the effectiveness of this method across various settings, including simulation, the real world, and with different architectures like conventional policies and Vision-Language-Action models.

### Strengths
1. This paper provides a crucial diagnosis for the inconsistent performance of vision-proprioception policies, a known but previously under-explained problem.
2. The paper's argument is well-structured and easy to follow, building a logical methodology directly from its clear problem diagnosis.
3. The paper validates the effectiveness of the proposed methodology by demonstrating high task success rates throughout its extensive experiments and ablation studies.

### Weaknesses
1. Reliance on Proprioceptive Cues: The reliance on proprioceptive cues for phase detection could be a failure point in tasks where transitions are defined by external, non-physical events (e.g., waiting for a visual cue), as the algorithm may not detect these critical moments.
2. Lack of Direct Evidence for Phase Estimation: The paper's central mechanism, i.e., the phase estimation, is not empirically validated with visualizations. The absence of plots showing the phase changes along with RGB images and/or the estimated transition probabilities makes it difficult to verify that this core component is functioning as intended, independently of the final task outcomes.  The final success rates are not enough to validate the rationale of this proposed method.

### Questions
1. Please include comparison plots (in the supplement) of the loss curves with and without your method for a given task to clearly demonstrate the effectiveness of the proposed approach.
2. Please add some plots to show how the phase changes  (the estimated transition probabilities) along with scene RGB images for validation. 
3. In Figure 4, the regulated policy still shows slightly lower performance than the vision-only policy during some motion transition stages. Could you analyze the reason for this?
4. In Table 4, for the “disassemble” task, the success rate of the FiLM fusion method is significantly lower than other methods. What might be the reason for this specific underperformance?
5. In the real-world bimanual task experiment, why were the policies for the left and right arms trained separately instead of as a single, unified policy?
6. The paper states that for training stability, gradient adjustment is only applied during the early stages of policy learning. Could you provide a more precise analysis of this reasoning, ideally supported by evidence such as training-related plots?
7. To further enhance the paper’s replicability and transparency, the reviewer recommends submitting the source code as a part of the supplementary material.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes GAP, a gradient adjustment algorithm that uses phase-guidance to dynamically modulate proprioception's gradient magnitude during training. This paper identifies a critical issue in vision-proprioception policies: the suppression of visual modality learning during motion-transition phases due to the dominance of more concise proprioceptive signals in optimization. Comprehensive experiments in simulation and real-world settings, across single-arm, dual-arm, and VLA models, demonstrate that GAP significantly improves policy robustness and generalization.

### Strengths
1. A detailed analysis was conducted to investigate the reasons behind the suppressed learning of the visual modality;
2. Phase-guided gradient adjustment offers a principled approach to dynamic modality balancing;
3. Comprehensive evaluation across different environments, platforms, and models.

### Weaknesses
The assessment lacks a direct evaluation of whether the visual modality is better utilized. While performance metrics provide some evidence, incorporating qualitative visualizations or explainable analysis of the GAP-trained model would more effectively demonstrate the method's effectiveness to readers.

### Questions
1. Are the analysis and methods presented in this paper still effective when applied to more advanced policy architectures, such as generalist VLA models built upon imitation learning frameworks with multimodal large language models as their backbone?
2. The paper also includes experiments with a transformer-based VLA (Octo). How did the authors determine parameter attribution and differentiate between parameters belonging to the vision chunk versus the proprioception chunk? Given that Octo's architecture is highly integrated, requiring deep fusion of both observation modalities within the transformer, this distinction seems non-trivial.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper argues that in vision+proprioception policies trained by BC, proprio signals dominate optimization (especially at motion-transition phases where vision should matter for target localization) so the visual pathway under-trains and generalization suffers. They propose GAP, which (i) detects motion-consistent phases via CPD on proprio traces, refines transition probabilities with an LSTM, and (ii) down-scales the proprio branch’s gradients during transitions to re-balance learning. GAP improves success rates across Meta-World, RoboSuite, and several real tasks; it also helps a VLA (Octo) variant.

### Strengths
- The key idea is that during optimization, proprio-encoder parameters are updated as $\omega_s^{j+1} \leftarrow \omega_s^j - \lambda (1 - \rho)\eta \nabla_{\omega_s^j} \mathcal{L}_{\text{BC}}$, where $\lambda$ controls scaling, $\eta$ is the learning rate, and $\rho \in [0,1]$ measures transition likelihood. A higher $\rho$ (likely transition) down-scales proprio gradients, forcing the visual encoder to learn those steps more effectively. GAP improves success rates across simulated and real tasks (Meta-World, RoboSuite) and enhances vision–proprio fusion even in pretrained VLAs such as Octo. The approach is simple yet effective, but relies on proprio-derived phase signals that might misidentify transitions and suppress useful proprio cues.
- The phase-guided reweighting is simple yet principled. It directly addresses the modality competition problem well known in multimodal learning. The LSTM refinement of $\rho_t$ mitigates the discretization artifacts introduced by CPD.  Together, these are reasonable, low-friction additions to standard behavioral cloning pipelines.
- Key findings of this work include (i) vision-proprio policies can underperform vision-only due to optimization bias toward concise proprio signals; GAP reverses this by reweighting gradients at the right time; (ii) transition phases are the crux: improvements concentrate where target localization is required. Intervention plots show reduced degradation after GAP.

### Weaknesses
- Hyperparameters ($\alpha$, $\beta$, $\lambda$) and LSTM size require task-specific tuning, which is a limitation of this work.
- From my understanding, CPD/LSTM finds proprio change points, not visual evidence of new targets. Showing that $\rho_t$ tracks visual uncertainty (e.g., entropy over detectors) would strengthen the claim.
- One minor concern is that the current phase-detection method and the gradient-adjustment strategy both rely on proprioceptive signals (joint positions, velocities, gripper states). Because the change-point detector (CPD) and the transition probability $\rho_t$​ are derived entirely from proprio data, the system may inadvertently bake in proprioceptive priors (already dominate the optimization process). This creates a potential feedback loop: proprio cues define where "transitions" occur, and then proprio gradients are scaled according to those same cues, reinforcing proprio's influence instead of balancing it.

### Questions
### Q1: Does GAP play well with on-policy updates (SAC/BCQ) where exploration changes state visitation?

**Action:** Maybe add a small on-policy appendix experiment, and a disturbance test (pushes/perturbations) to ensure GAP doesn’t over-dampen necessary proprio reliance.

### Q2: Is Octo-VP better because the visual branch actually learned more useful features, or simply because the policy became less overfit to proprioception?

**Action:** Linear-probe vision features pre/post GAP and report OOD splits where only visual scene factors change.

### Q3: Does GAP transfer across arms/cameras?

**Action:** Train on robot A, test on robot B (and different wrist cameras) with no re-tuning. This is hard so not required.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores an efficient strategy for combining vision observations with proprioception observations in robotic manipulation. The authors first empirically show that naïve vision-proprioception policies tend to underutilize vision during motion-transition phases, which degrades performance compared to pure vision policies. To address this, they propose a Gradient Adjustment with Phase-guidance (GAP) method that dynamically scales the proprioceptive gradient updates during training, particularly for parameters related to motion-transition estimation. Experimental results demonstrate that GAP effectively improves both in-distribution and out-of-distribution (OOD) performance.

### Strengths
1. The paper tackles an important and practical challenge in robot learning—how to effectively combine vision and proprioception for efficient and accurate manipulation.
2. The empirical findings are well-motivated, and the proposed gradient adjustment method, GAP, is simple, interpretable, and effective.
3.  Both simulation and real-world experiments demonstrate the effectiveness of GAP. The approach shows consistent gains over both vision-only and vision-proprioception baselines, and the improved OOD performance is particularly promising.

### Weaknesses
Overall, this is a good paper for me, but there are a few minor concerns:
1. The paper lacks a deeper investigation into the form of the gradient adjustment function $(1-p)$ and the choice of key hyperparameters. Since the paper mentions that gradient adjustment is applied only during the early stage of training, it would be beneficial to discuss the rationale for choosing specific hyperparameters ($\alpha $, $\beta$, and the number of stages for applying GAP) and to include an ablation study on parameter sensitivity.
2. GAP adjusts gradients for certain parameters of the network. This may have connections to GradNorm-based balancing methods [1]. For instance, if early in training the gradient magnitude of the proprioceptive branch is larger than that of the vision branch, then GAP might behave similarly to gradient normalization. It would be helpful to clarify whether GAP implicitly aligns with GradNorm principles. Including statistics of gradient norms for both branches and showing how the adjustment parameter $p$ evolves during training would strengthen the analysis.

[1] Chen, Zhao, et al. “GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks.” ICML 2018.

### Questions
1) What is the reasoning behind the choice of GAP hyperparameters ($\alpha $, $\beta$, and stage schedule)? How sensitive is the method to these values?
2) What is the relationship between GAP and GradNorm-based gradient balancing? Would a GradNorm-style adjustment (e.g., scaling by grad/∥grad∥) yield similar improvements?

### Soundness
4

### Presentation
4

### Contribution
3
