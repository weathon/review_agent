# villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Vision-Language-Action (VLA) models have emerged as a popular paradigm for learning robot manipulation policies that can follow language instructions and generalize to novel scenarios. Recent works have begun to explore the incorporation of latent actions, abstract representations of motion between two frames, into VLA pre-training. In this paper, we introduce villa-X a novel Vision-Language-Latent-Action (ViLLA) framework that advances latent action modeling for learning generalizable robot manipulation policies.
Our approach improves both how latent actions are learned and how they are incorporated into VLA pre-training. We demonstrate that villa-X can generate latent action plans in a zero-shot fashion, even for unseen embodiments and open-vocabulary symbolic understanding. This capability enables villa-X to achieve superior performance across diverse simulation tasks in SIMPLER and on two real-world robotic setups involving both gripper and dexterous hand manipulation. These results establish villa-X as a principled and scalable paradigm for learning generalizable robot manipulation policies. We believe it provides a strong foundation for future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents villa-X (vision-language-latent-action), a framework that integrates latent actions into vision-language-action (VLA) models. The core idea of villa-X consists of two components:
- incorporating existing latent action models (LAMs) into a proprioceptive module to obtain more physically grounded latent actions
- jointly modeling latent actions and real robotic actions.

Experimental results demonstrate that (1) the proprioceptive module enables LAMs to extract higher-quality latent actions, and (2) the resulting VLA model significantly outperforms existing VLA baselines in both simulated and real-world environments.

### Strengths
- [S1] villa-X achieves strong performance in both real-world and simulated settings. The authors conduct extensive experiments that convincingly demonstrate the effectiveness of the proposed framework.
- [S2] The paper is clearly written and well-organized. Figures and equations are concise yet informative, effectively illustrating how villa-X operates.

### Weaknesses
- [W1] From a high-level perspective, this work largely follows the structure of the existing framework [1]. The process—first training latent action models (LAMs) with a VQ-VAE-style objective to generate pseudo labels (latent actions) for robot data, and then training the VLA model to predict those latent actions—remains similar. The proposed addition of a proprioceptive module and the joint prediction of latent and robotic actions, while useful, appear somewhat incremental.
- [W2] The paper provides limited discussion on why villa-X performs well across diverse environments and tasks. The experiments primarily report performance improvements without deeper analysis of contributing factors or underlying mechanisms.
- [W3] No statistical significance analysis is presented. The reported performance metrics lack measures of variance or confidence, making it difficult to assess the robustness of the claimed improvements.

**References**

[1] Ye et al., Latent Action Pretraining from Videos. In ICLR, 2025.

### Questions
- [Q1] In Table 1, the performance of LAPA and Go-1 on the WidowX robot appears noticeably poor. Could the authors elaborate on the underlying cause of this phenomenon?

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
This paper introduces a new framework called villa-X, whose core idea is to incorporate latent actions into the pretraining and policy learning of Vision-Language-Action (VLA) models.

### Strengths
1. The system design is simple, effective, and scalable.

2. The experiments are sufficient and comprehensive.

3. The writing is clear and provides a good reading experience.

### Weaknesses
See questions.

### Questions
1. I recently came across latent action learning in a survey paper [1]. How does the latent action learning mentioned in your work differ from that?

2. I also believe that latent learning is a promising approach for solving cross-embodiment transfer. What other potential solutions do you foresee in the future?

3. Equation (3) makes me a bit confused — what exactly is the dataset ID, and why is the context vector ce composed of these two parts?

4. How large is the VILLA-X model in terms of scale?

[1] Towards a Unified Understanding of Robot Manipulation: A Comprehensive Survey. arXiv 2025.

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
4

### Summary
The paper introduces VILLA-X, a Vision-Language-Latent-Action framework that advances robot policy learning by improving both how latent actions are learned and integrated into Vision-Language-Action models. It enhances latent action learning through a proprioceptive forward dynamics module that grounds latent representations in physical robot dynamics, and it introduces a joint diffusion-based policy that conditions robot action generation on latent action planning.

### Strengths
1. The paper is well-structured and clearly written. The problem motivation is sound, the technical approach is explained logically.

2. The evaluation is thorough, encompassing systematic ablations, major simulation benchmarks (SIMPLER, LIBERO), and real-world deployment on two distinct platforms. 

3. The demonstrated capability for zero-shot generalization to novel embodiments addresses a core challenge in the field.

### Weaknesses
1. The technical contributions, while valuable, exhibit limited novelty relative to existing literature. The proposed proprioceptive Forward Dynamics Model (proprio-FDM), which grounds latent actions by predicting low-level states, is conceptually similar to the approach of Nikulin et al. [1], who employ a linear decoder on latent tokens to predict actions. The efficacy of this general principle for grounding has also been previously analyzed by Zhang et al. [2]. Furthermore, the architectural design of separate experts for latent and robot actions (ACT-latent and ACT-robot) bears a strong resemblance to the module separation employed in GO-1.

2. The characterization of the GO-1 baseline may be inaccurate. Based on its open-source implementation, GO-1 does not appear to autoregressively predict latent actions using a next-token-prediction (NTP) loss, but rather uses an L1 loss for latent action learning. Consequently, the description in Section 4.2 and the subsequent analysis in Table 1 could be misleading regarding the true nature of this baseline.

3. The experimental comparisons lack benchmarks against several highly relevant contemporary works, notably IGOR [3] and UniVLA [4]. These methods also focus on cross-embodiment generalization and leverage latent actions learned from web videos, making their inclusion critical for properly contextualizing the claimed advancements of this work.

4. The zero-shot generalization analysis in Section 4.3 would be strengthened by clarifying the training procedure for the world model used for visualization. 



______

[1] Nikulin, Alexander, et al. "Latent action learning requires supervision in the presence of distractors." arXiv preprint arXiv:2502.00379 (2025).

[2] Zhang, Chuheng, et al. "What Do Latent Action Models Actually Learn?." arXiv preprint arXiv:2506.15691 (2025).

[3] Chen, Xiaoyu, et al. "IGOR: Image-goal representations are the atomic control units for foundation models in embodied ai." arXiv preprint arXiv:2411.00785 (2024).

[4] Bu, Qingwen, et al. "UniVLA: Learning to act anywhere with task-centric latent actions." arXiv preprint arXiv:2505.06111 (2025).

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
3

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
The paper presents an approach to learn latent actions, and leverage latent actions when predicting robot actions through joint diffusion. In addition to learning forward and inverse dynamics models, as is done in existing work, the paper proposes to learn an embodiment-conditioned model which predicts future robot states and actions. During inference-time, the action head predicts both latent actions and robot actions given image observations, language instructions, proprioceptive states and embodiment embedding, where an attention mask is used to enforce the factorization of the conditional probability.

### Strengths
- The proposed approach has strong empirical results. In particular, in Table 2, the proposed approach outperforms state-of-the-art VLA models, such as OpenVLA, OpenVLA-OFT, $\pi_0$, and Gr00T. It also outperforms other algorithms such as MoTo and LAPA.

- Using an embodiment-specific embedding is an interesting approach to leveraging diverse robot datasets, where different datasets have slightly different dynamics and action spaces. Ablation study shows that the embodiment embedding has a positive impact on the success rate.

- The ablation study is in general quite thorough, showing the effectiveness of each component of the proposed approach, e.g., latent action model, proprioceptive state prediction, etc.

### Weaknesses
- Although the main design components are validated through ablation studies. Some finer-grained design choices lack discussion and study. See more details in the **Questions** section.

- Figure 3, probing experiment results, is not very easy to understand. Perhaps a plot showing the distribution of error in different intervals, for both w/ pp and w/o pp,  would be more informative. It is also not very convincing why $L_\infty$ (max across all dimensions) is preferred over $L_1$ (summing/averaging over all dimensions). 

- The results in 4.3, although helpful and intuitive, are not very informative scientifically. Since the images are obtained from a “separately trained world model”, it is hard to tell whether the latent action actually encodes the robot behavior or these outcome images just come from hallucination from the world model. It may be good to swap it with another ablation study in the appendix.

- It is unclear why only Gr00t is used as the baseline for real-world experiments, where other state-of-the-art VLA models are left out.

- The tasks evaluated in the experiment are mainly simple pick and place tasks, without much dexterity. 

- Minor: Incomplete sentence in Appendix D.3: "Both models were trained on 10..."

### Questions
- How is $K$, the number of future steps when training FDM and IDM, determined?

- Why is the prediction only for $o_t$ and $o_{t+K}$ when training observation F/IDM in Equation (1), but for the entire sequence when prediction robot states $q_{t+1:t+K}$ and actions $a_{t+1:t+K}$ in Equation (2)?

- In Equation (3), why is the context vector conditioned on dataset ID instead of robot ID. Wouldn’t it make sense for two datasets with the same robot to share an ID?

- Why does the action head predict $(n-1)K$ latent actions and $m$ robot actions? How to choose $n$ and $m$?

- Why is the robot action branch “overly relying on latent actions” harmful? Isn’t the latent action supposed to provide sufficient information for action prediction (Equation (2))?

- What is the action space for the model? How does the gripper command translate to Xhand command in the realworld experiments?

- In Appendix A.1, how is the codebook size of $32$ decided?

- In Appendix B, can you explain how different $\tau$ distribution can be used for latent actions and robot actions? This seems contradictory to (5) where all actions are bundled together during denoising. 

- In Appendix F, is the attention mask ablation referring to the block-wise causal attention mask or random masking of attention during training?

- In Appendix H, why is OpenVLA-OFT not included in Table 9? Especially considering that it is included in Table 2.

### Soundness
3

### Presentation
3

### Contribution
3
