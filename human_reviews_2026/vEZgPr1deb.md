# Latent Action Robot Foundation World Models for Cross-Embodiment Adaptation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Robot action-conditioned video generation models, also known as robot world models, hold great potential for enhancing robotic planning and decision-making. However, the diversity of robot embodiments and action spaces makes it challenging to build models that generalize across different embodiments. We introduce a robot foundation world model, LAC-WM, which operates within a learned unified latent action space shared across diverse embodiments. We explore how this unified action space improves the world model’s performance when adapted to previously unseen robot embodiments. Specifically, we compare LAC-WM to a baseline model EAC-WM conditioned on explicit motion labels. Our results show that conditioning on explicit labels creates disjoint action spaces across embodiments, limiting downstream task performance when adapting to new robots. We evaluate both models on a dexterous manipulation task. The latent action-conditioned model LAC-WM achieves up to a 46.7% improvement in performance over EAC-WM. Crucially, the unified latent action space allows LAC-WM’s downstream performance to scale positively with the number of embodiments used during pretraining. In contrast, the disjoint action space in EAC-WM leads to decreased performance as the number of pretraining embodiments increases. These results highlights the importance of a unified action space for efficient cross-embodiment learning, addressing a key challenge in robotics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces the Latent Action Conditioned Robot World Model to boost cross-embodiment learning, thus tackling the key challenges in today's robot learning field. However, the method itself is not novel at all, with every component having been seen in prior works. Also the experiment is poor, with only one "EAC-WM" baseline and a simple pick-and-place manipulation task. I do not think the draft is ready for ICLR 2026.

### Strengths
- The author highlights that the key challenge of robot learning is the heterogeneity of robotics data, which is crucial and important for building a general embodied intelligence.

### Weaknesses
- In the abstract, "LAC-WM" and "EAC-WM" are abrupt and lacking prior introduction.
- The first two paragraphs of the introduction lack references, while the challenges of cross-embodiment heterogeneity are well studied in many works, as listed in the references below.
- The baseline is inadequate, as only results of Explicit Action Conditioned World Model are presented. 
- There is only one simple manipulation task which is also inadequate.

references:
- [1] Universal actions for enhanced embodied foundation models
- [2] Rdt-1b: a diffusion foundation model for bimanual manipulation
- [3] X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model
- [4] Scaling cross-embodied learning: One policy for manipulation, navigation, locomotion and aviation
- [5] Learning to Act Anywhere with Task-centric Latent Actions

### Questions
As mentioned in weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a world model based on a latent action model, called **LAC-WM**. Recently, previous works have built a unified latent action space form both action and action-less videos by using an Inverse Dynamics Model and a Forward Dynamics Model. In this paper, the authors also introduce action decoding from latent actions to ground-truth actions to mitigate shortcut learning. Furthermore, to address the issue of feasibility--where the model cannot observe future frames during inference--the authors propose an action projector that encodes actions into latent actions. With these approaches, LAC-WM can imagine future scenarios even with unseen objects, and its planning capability outperforms the baselines, including directly using action labels.

### Strengths
- **Action-conditioned world model**
  - This paper introduces an action-conditioned world model based on a latent action approach. Most recent latent action-based approaches require future frames, transferring actions from source videos to predict future frames. However, LAC-WM introduces an action projector to align real actions with the latent action space, enabling the generation of action-conditioned future frames.
- **Planning with VLA**
  - Unlike discrete action spaces such as those in game environments, it is difficult to sample actions in continuous action spaces. To address this, the paper adopts a trained VLA for sampling. This approach appears to be a reasonable way to handle complex real-world tasks.

### Weaknesses
**Major Weakness**
- **Motion Decoder**
  - The authors claim that, to prevent shortcut learning where the latent action is encoded from future frames, they adopt a motion decoder. However, the provided analysis is insufficient. In Fig. 2, although the authors state that the visualization demonstrates a unified latent action space, even without the motion decoder, Agibot and Egodex already appear to be encoded in a unified action space. Moreover, even with the motion decoder, Droid and Egodex seem to form their own clusters in the center.
  - In LAOM [1] and CLAM [2], the models adopt an action decoder to improve robustness to distractors or to decode latent actions into real actions within the same environment. However, in this paper, the motion decoder is adopted to prevent shortcut learning. Since each motion has different properties -- Droid has one arm, Agibot has two arms, and Egodex has human hands -- it may introduce embodiment-specific information rather than effectively preventing shortcut learning. Based on Fig. 3, in terms of movements, there are no significant differences regardless of the motion decoder, and the results already show robustness to distractors, as noted in LAOM.
  - To better align the claimed benefit of the motion decoder with the authors' argument, further experiments such as ablation studies on robot experiments are required.
  - Beyond the motion decoder, the authors also argue that *cross-embodiment augmentation* mitigates the shortcut learning. However, it is difficult to understand how cross-embodiment augmentation achieves that effect without appropriate analyses.
- **Insufficient explanation**
  - The paper provides limited explanation of the proposed method and experimental setups.
  - For the pixel decoder, it is unclear how the embedding $x$ is decoded into the pixel space $I$. To the best of my knowledge, V-JEPA2 [3] does not provide an pixel decoder. Additionally, it is unclear whether the authors used a pretrained visual encoder from V-JEPA2 or trained it themselves.
  - In Tab 3, important details are missing for each setting, such as LAC-WM-DFT.
---

**Minor Weakness**
- **Robot experiments**
  - The provided experimental results are too weak to support the proposed method. Although the authors claim relative improvement, the gains are marginal. Moreover, the performance appears to depend heavily on the VLA.
  - Evaluating on a single task, especially one that is not a real-world task, is limiting.
- **Missing citations**
  - UniSkill [4] also adopts continuous latent actions from in-the-wild videos and demonstrates transferability across different embodiments.

---
[1] Nikulin, Alexander, et al. "Latent action learning requires supervision in the presence of distractors." arXiv preprint arXiv:2502.00379 (2025).

[2] Liang, Anthony, et al. "Clam: Continuous latent action models for robot learning from unlabeled demonstrations." arXiv preprint arXiv:2505.04999 (2025).

[3] Assran, Mido, et al. "V-jepa 2: Self-supervised video models enable understanding, prediction and planning." arXiv preprint arXiv:250

[4] Kim, Hanjung, et al. "UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations." arXiv preprint arXiv:2505.08787 (2025).

### Questions
**Major Questions**
- What is the benefit of using cross-embodiment augmentation and the motion decoder? Is there any in-depth analysis regarding how they mitigate shortcut learning?
  - Does each component affect downstream performance?
- If you fine-tune the IDM and FDM or downstream tasks, can the latent action space remain unified?
  - During fine-tuning, both the IDM and FDM are trained. During pretraining, by using various datasets, a unified latent action space is constructed. However, during fine-tuning, every module is updated, so I wonder whether the unified latent action space becomes biased toward the fine-tuned dataset.
  - When using the action projector to encode latent actions, can these be decoded using the motion decoder?
---
**Minor Questions**
- What are the implementation details? Which vision encoder is used (an open-weight model or manually trained one), and how is the embedding decoded into pixels?
- What is the meaning of each setting or row in Tab 3?
- If you plot latent actions from the same motion (or task) across different embodiments, are they well clustered? Additionally, are they distinguishable from different tasks within the same embodiment?
- Are there any real-world robot experiments?
- Are there any multi-task robot experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes LAC-WM, a Latent Action-Conditioned World Model for robotics, aiming to build a unified latent action space shared across heterogeneous robot embodiments (e.g., human, humanoid, single-arm robots). The model consists of an inverse dynamics model (IDM), a forward dynamics model (FDM), a motion decoder, and an action projector. By pretraining on multiple embodiments (EgoDex, Agibot, Droid) and fine-tuning on an unseen robot (BFA), the authors show that this unified latent space leads to superior cross-embodiment adaptation and higher-quality video rollouts compared to an explicit-action baseline (EAC-WM). Quantitatively, LAC-WM improves planning success rate by up to 46.7% and scales positively with the number of pretraining embodiments.

### Strengths
1. **Clear motivation and well-scoped problem.**
The paper addresses a fundamental issue in robotic foundation models—heterogeneous action spaces across embodiments—which limits generalization. The proposed unified latent action space directly targets this challenge.
2. **Comprehensive experimental evaluation.**
The paper conducts extensive experiments, including (a) latent space visualization (UMAP), (b) cross-embodiment transfer (human ↔ robot), (c) quantitative rollout metrics (PSNR/FVD), (d) planning success rate, and (e) scaling analysis with embodiment number. Results consistently support the central hypothesis.

### Weaknesses
1. **Low absolute task performance.**
Despite relative improvements, the absolute task success rate remains modest (≈0.22). This indicates limited planning robustness, which should be acknowledged.

2. **Computational efficiency and scalability not analyzed.**
The paper does not report training cost, model size, or inference latency—important aspects for foundation model claims.

### Questions
1. Please report results when (i) fixing the total pretraining samples while increasing the number of embodiments, and (ii) fixing the number of embodiments while scaling the total samples. Does LAC-WM still show monotonic gains in UMAP alignment, rollout metrics, and planning success? Any signs of saturation?
2. Have the authors attempted any real-robot experiments or domain randomization studies to evaluate sim-to-real robustness?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes the training of a cross-embodiment latent action space for world models. Unlike conventional methods, which use raw actions as the conditioning input for future frame prediction (referred to as EAC in this paper), the proposed LAC method trains an Inverse Dynamics Model to encode latent actions. A Forward Model is then used to predict the future frame, conditioned on the latent action and the current frame. The framework is trained on a mixed dataset across several embodiments, with the aid of additional motion supervision, which serves a regularization-like role. The proposed training paradigm is shown to be effective in aligning the action space across different embodiments, thereby promoting cross-embodiment adaptation.

### Strengths
1. Using raw actions as the condition for world model generation faces significant challenges due to the strong heterogeneity present in robotics datasets. The heterogeneous action spaces often lead to semantic misalignment, which hinders effective world model training. The idea of utilizing a shared latent action space to replace the original raw actions is promising, and this approach has been widely explored and proven effective in previous works.

2. The introduction of extra motion supervision is shown to be effective in accommodating the shared action space in the experiments presented in this paper. This insight suggests that identifying embodiment-agnostic supervision signals may be a valuable solution to bridge the gap between heterogeneous action spaces and facilitate alignment.

3. The paper presents some interesting and insightful behaviors of the LAC-WM through qualitative results. These observations provide valuable insights into the system's performance and behavior, highlighting the effectiveness of the proposed approach.

### Weaknesses
1. The proposed framework employs an auto-encoder style for training the action space without any explicit distribution constraint. This raises a key question about how actions are sampled from the obtained latent action space, especially since it is not trained as a distribution.

2. As this work aims to build a world model, more qualitative results are needed to demonstrate the quality of the generated frames. The paper could benefit from a deeper exploration of how the generated frames align with actual environmental states, providing a clearer assessment of the model’s effectiveness.

3. Using the distance of image embeddings as a metric to report or select executable actions is potentially unreliable, as it heavily depends on the choice of image encoder. If this approach is used, more ablation studies and analysis should be included to assess how to choose an appropriate image representation and to explore the sensitivity of the method to different encoder architectures.

4. There is a lack of sufficient analysis regarding the trained action space. The authors should provide more case studies or a detailed analysis of the physical meaning of each latent action. This would help clarify how well the latent space captures meaningful and interpretable robotic actions.

### Questions
Which model is used to obtain the image embeddings for the distance calculation in the experiments? Clarification on the choice of model and its impact on the results would be helpful, especially given the potential sensitivity of the metric to the image encoder.

### Soundness
2

### Presentation
2

### Contribution
2
