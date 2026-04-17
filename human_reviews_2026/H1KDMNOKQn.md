# HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
A central objective of manipulation policy design is to enable robots to comprehend human instructions and predict generalized actions in unstructured environments. Recent autoregressive vision-language-action (VLA) approaches discretize actions into bins to exploit the pretrained reasoning and generation paradigms of vision-language models (VLMs). While these models achieve efficient and scalable training, the discretization undermines the continuity required for precise control. In contrast, diffusion-based VLA methods incorporate an additional diffusion head to predict continuous actions, but they rely solely on feature representations extracted from the VLM, without leveraging the pretrained large language model (LLM) as an expert for iterative action generation. To integrate the complementary strengths of autoregressive and diffusion generation, we introduce HybridVLA, which innovatively leverages a shared LLM backbone to perform iterative action prediction through both paradigms. Specifically, a collaborative training recipe is proposed, incorporating diffusion denoising into the next-token prediction process and mitigating interference between the two generation paradigms. With this recipe, we find these two action prediction methods not only reinforce each other but also exhibit varying strengths across different scenarios. Therefore, we design a collaborative action ensemble mechanism that adaptively fuses both predictions, leading to more robust control. HybridVLA outperforms previous state-of-the-art VLA methods by 17\% and 19\% in mean success rate on simulation and real-world tasks, respectively, while demonstrating generalization to unseen configurations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes HybridVLA, a policy architecture that injects diffusion into autoregressive action generation in VLM inference. The proposed method is evaluated on RLBench and shows promising results against other VLA variants. The method is also evaluated on real world single arm / dual arm scenarios.

### Strengths
1. The paper overall is easy to follow.
2. This paper aims to design novel policy architecture that is based on well-established techniques, i.e., diffusion, autoregression, which is a valuable problem and promising direction. 
3. The proposed method is evaluated on different tasks with single-arm and dual-arm real robots.

### Weaknesses
My major concerns are two: unclear motivation and limited novelty. 

- Motivation: The proposed techniques are driven by the motivation to mix diffusion and autoregressive together. Instead of having a diffusion head at the end of the autoregression, we can incorporate diffusion into the autoregression steps. However, I seem to be unable to find a compelling argument or motivating example in this paper, which can convince me why this new formulation can be superior to having a diffusion head only (which is much simpler, and also effective as proven in many works). 

- Novelty: The paper proposed to (1) train the HybridVLA with teacher forcing loss and denoising loss combine, and (2) ensemble the autoregressive and diffusion action output. Both techniques are engineering-driven modifications, lack novelty or general application potential. 

- Missing critical references: CDP[1] and ARP[2]. 
   - Causal Diffusion Policy (CDP) studies the same problem and proposes a similar architecture, yet it was not compared or mentioned in this paper (correct me if I am wrong). The authors need to address this comparison. 
   - Autoregressive Policy (ARP) also supports action-chunk prediction within the autoregressive steps, like the proposed architecture in this paper, which also works with teacher-forcing training (without a collaborative training stage). Moreover, ARP is also evaluated on RLBench (using the same demonstrations for training data) and achieves 84% success rates on 18 tasks, despite being a smaller model. The authors are also recommended to compare with other policies (like 3D diffusion policy) on RLBench. 

[1] CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion (CoRL 2025)

[2] Autoregressive Action Sequence Learning for Robotic Manipulation (RAL 2025)

- The figure in Table 1 is very small, the robot state token and the autoregressive token almost have the same color (when printed), which makes it very difficult to understand.

### Questions
(please see weakness section)

### Soundness
2

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
4

### Summary
This paper proposes HybridVLA, which is a VLA model that unifies both diffusion of continuous actions and autoregression of discrete action tokens under single LLM architecture. They also propose optimized design choices for the model and collaborative action ensemble for synerging action samples from both processes.

### Strengths
- This is one of the first attempts to directly unify diffusion and autoregressive in a VLA model rather than having multiple modules, extending current mainstream VLA design.
- Jointly training diffusion and autoregression improves success rate even without fusion action samples, further improving them with inference time ensemble with confidence threshold.
- The paper reports extensive experiments in simulation and real world robots, and provides meaningful gains over relevant baselines along with multiple experiments and ablations. The results present both practical utility and robustness of the proposed approach.

### Weaknesses
- The approach unifies diffusion and autoregression mainly through joint training and an inference-time ensemble, without introducing a deeper fusion or interaction mechanism between the two processes. It raises a minor concern about whether the two processes are truly leveraging each other’s strengths to the maximal extent. We don’t view this as a major weakness, but rather as a potential area for further exploration.

- Naively averaging diffusion and autoregression outputs in collaborative action ensemble could lead to modal conflict when actions are multi-modal. This conflict is well avoided by setting the confidence threshold very high (>0.9) in this work, but there could be more robust methods to fuse action samples even when confidence is low. For example, by using diffusion modal to filter autoregressive actions during inference, which boosts relative confidence of filtered AR action samples, then fusing them.

### Questions
- What were specific optimization challenges for jointly training diffusion and autoregression with scalable robotics data mixture and model except token sequence formulation design?
- What are failure cases of setting a low threshold and success cases of setting a higher threshold? It would be persuading to see failures with modal conflict in a 'push-t'-like environment with low threshold.
- In the generalization experiments, using height as the axis for spatial generalization may not convincingly demonstrate a robust generalization for robotic tasks, while positional generalization seems to give us more insight.

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
This paper introduces HybridVLA, a method that unifies diffusion-based continuous actions and autoregressive discrete actions within a single LLM backbone. Its core contribution is a hybrid, co-optimized objective that combines diffusion-noise MSE with autoregressive cross-entropy. At inference, it uses few-step DDIM sampling and a confidence-weighted fusion mechanism based on autoregressive token probabilities. The model is pre-trained on cross-embodiment trajectories and then fine-tuned, achieving state-of-the-art performance and strong generalization in real-robot experiments.

### Strengths
1.	It unifies diffusion and an LLM in a single backbone and sequence. This integrates the LLM’s reasoning with diffusion’s fine-grained control, while avoiding error accumulation and interface mismatches in multi-module pipelines.
2.	In the unified representation, the diffusion branch provides continuous latents that capture contact and trajectory details. The LLM organizes high-level intent using context and world knowledge. They learn under the same conditional distribution and fuse at inference based on autoregressive confidence.
3.	On RLBench, it achieves 78% average success and surpasses strong baselines. Real-robot tests show stable generalization to unseen objects, backgrounds, heights, and lighting.

### Weaknesses
While this work presents a promising hybrid framework, its primary weakness lies in the insufficient justification for its core collaborative mechanism and a lack of sensitivity analysis for its key hyperparameters. The paper claims that hybrid training leads to semantic collaboration, but this could more likely be a regularization effect from the complex loss function, rather than genuine mutual reinforcement between the two paradigms.

### Questions
1. How are the weights between diffusion-noise MSE and AR cross-entropy set in the hybrid loss? Please state the basis for weight normalization or dynamic weighting, and report the weight–success-rate curve.
2. How is the confidence threshold for fusion determined and calibrated? Please add a sensitivity analysis for the threshold and the temperature hyperparameter.
3. Table 3 shows that training with the hybrid loss improves performance even when inference uses a single path. How do you establish that this is semantic collaboration?
4. The collaborative fusion relies on a confidence threshold as high as 0.96 to admit AR actions. Please report, per task, the proportion of time steps triggering this threshold, the fusion frequency, and performance under different thresholds. If most steps fall back to pure diffusion, please quantify whether the hybrid mechanism still provides meaningful value.
5. The sequence design places diffusion tokens before AR tokens to prevent leakage, but this also introduces strong causal dependence. When noise is injected into the diffusion branch, does it affect the AR branch and overall decisions?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes HybridVLA, a unified vision-language-action (VLA) model that integrates diffusion-based continuous action generation and autoregressive (AR) discrete action generation within a single LLM backbone. The core idea is a “collaborative training recipe” that inserts diffusion denoising steps into the LLM’s token sequence using special markers (BOD/EOD) so that denoising is treated as iterative “reasoning” steps. A hybrid objective (MSE for diffusion noise prediction + cross-entropy for AR tokens) is optimized jointly, and an inference-time ensemble fuses diffusion and AR actions based on AR token “confidence” with a fixed threshold.  
The model is pretrained on ~760k trajectories and then fine-tuned on RLBench and real world tasks. The results shows that the model can achieves better results compared with the pi0 and CogACT baselines.

### Strengths
The paper is well-written and easy to follow. The idea of combining AR (semantic reasoning, sample efficiency) with diffusion (continuous, precise control) in one model makes sense to me. The unification via token-sequence design and the BOD/EOD markers is conceptually neat, and the hybrid objective plus confidence-based ensemble is practical. Overall I think this is a promising direction to pursue.

### Weaknesses
However, I feel like the current results are not significant enough: 
- In the real world experiments, using a 7B model to compare with a 2.7B pi0 baseline seems unfair to me and I did not see an explanation on it (since the authors also have a 2.7B model). It’s also not clear to me to what extent using different pretraining data would affect the final performance.
- The generalization part, which is critical for robotic applications, follows this setting and only pick one scenario (and they are not the same) for each model. Also HybridVLA does not seem more robust compared to the baseline in many cases (similar performance percentage drop). 
- In the RLBench experiments, while the full HybridVLA 7B model outperform the HybridVLA-dif variant, it is 50% slower, so it is hard to justify whether using a larger diffusion-only model with the same speed can be on par or even surpass the proposed HybridVLA setting.

### Questions
The 0.96 threshold seems adhoc to me, how robust this threshold is (when change backbone / task etc.)?

### Soundness
3

### Presentation
3

### Contribution
2
