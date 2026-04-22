# VELR: Efficient Video Reward Feedback via Ensemble Latent Reward Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Reward feedback learning (ReFL) is effective for both text-to-image (T2I) and text-to-video (T2V) generation with image reward models (RMs). However, image RMs are misaligned with temporal objectives of T2V, motivating ReFL with video reward models. Nevertheless, directly deploying video RMs is impractical due to their large parameter size and the prohibitive cost of memory. To address this, we propose VELR: an efficient framework that employs ensemble latent reward models (LRMs) to predict rewards directly in latent space, bypassing expensive backpropagation through VAE decoders and video RMs. Specifically, we introduce the ensemble technique for the LRM, which enhances capacity, quantifies uncertainty, and mitigates reward hacking. VELR achieves a reduction of up to 150GB in memory, requiring as little as 12.4\% of the memory compared to standard ReFL. Experiments on OpenSora, CogVideoX-1.5, and Wan-2.1 with large-scale video RMs demonstrate that VELR achieves comparable performance as standard ReFL and enables efficient and robust video RM-based ReFL at scales previously unattainable.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes VELR (Efficient Video Reward Feedback via Ensemble Latent Reward Models), an innovative framework to overcome the prohibitive memory cost of applying large-scale video Reward Models (RMs) in Reward Feedback Learning (ReFL) for Text-to-Video (T2V) generation. By training an Ensemble Latent Reward Model (LRM) to predict rewards directly in the latent space, the framework successfully bypasses expensive backpropagation through the VAE decoder and the large video RM. The method achieves a substantial memory reduction (up to 150GB ) while maintaining comparable performance to standard ReFL, making previously infeasible Video RM-based fine-tuning a reality for large T2V models. The work is highly relevant and addresses a critical practical limitation in aligning video diffusion models.

### Strengths
1.	The proposed reward model enables the effective use of powerful, temporal-aware Video RMs (like UnifiedReward and CausalVQA) that were previously computationally inaccessible.
2.	The introduction of the Ensemble LRM is technically sound. 
3.	Experimental results indicate the effectiveness of the proposed method.

### Weaknesses
1.	The primary benefit of moving from Image RMs to Video RMs is the improvement in temporal coherence. The paper critically lacks supplementary video material for the generated samples. Without visual evidence, it is impossible for the reviewer to verify the claimed improvements in temporal consistency and to judge the subjective quality, flicker, and artifacts of the generated videos. This omission severely undermines the empirical claims of the paper.
2.	While the paper compares against LRM-adapted baselines, there is no direct comparison against a dedicated, high-fidelity Image Reward model (running in its original pixel space). Additionally, Since Video RMs primarily prioritize temporal objectives, a convincing comparison is necessary to demonstrate that the VELR framework, despite its efficiency and temporal gains, fully retains the high perceptual visual quality boost provided by state-of-the-art Image RMs. The current results focus heavily on memory and speed, but not enough on the visual quality trade-off (if any) compared to the best image-focused alignment methods.
3.	While motivated, "Truncated Mid Step Setting" introduces a model-specific heuristic for selecting the "mid-step regime" that is dependent on the velocity prediction and noise schedule of the base T2V model (e.g., Wan-2.1 ). This reliance limits the general applicability and plug-and-play nature of the ReFL solution across different diffusion architectures.

### Questions
Please refer to Weaknesses for more details. If the concerns are solved, I will be glad to raise my score rate.

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
This paper addresses the mismatch between image-based reward models and the temporal objectives of text-to-video generation in Reward Feedback Learning (ReFL). The authors propose VELR, a memory-efficient framework that replaces direct video reward modeling with ensemble latent reward models operating in the latent space. By predicting rewards without backpropagation through VAE decoders and full video RMs, VELR substantially reduces memory usage while maintaining comparable performance. The ensemble design is intended to increase capacity, quantify uncertainty, and mitigate reward hacking. Experiments on OpenSora, CogVideoX-1.5, and Wan-2.1 with large-scale video RMs support the claim that VELR enables robust, scalable, and efficient ReFL for text-to-video at previously unattainable scales.

### Strengths
1. The manuscript is clearly written and accessible.
2. The motivation is well articulated; the work explores an effective video latent reward model.
3. The topic is important: latent reward models can reduce RL training costs, particularly for video generation.

### Weaknesses
1. Please report model sizes: the parameter count of the reward model, and the parameter count for the ensemble-based LRM (per model and overall).
2. Additional methodological details would improve clarity. In Section 4.3, when fine-tuning the LRM using outputs from the diffusion model: 
    - Is the dataset the same as that used for LRM pretraining? If only diffusion-generated videos (no real videos) are used, how do you ensure training reliability and avoid distribution shift?
    - For the Truncated Mid Step setting, which intermediate steps are selected and why?
    - For the ensemble of LRMs, are all models trained on the same dataset and loss? If so, how is diversity encouraged to yield distinct rewards for each LRM in the ensemble LRMs?
3. Did you initialize any components from CausalVQA or UnifiedReward pretrained weights?
4. In Figure 3, the proposed method appears to offer limited improvement over the VADER baseline.
5. It would be better to include qualitative comparisons against the baseline Dollar.
6. Please clarify what dimensions the LRM's reward captures: is it decomposed into image quality, temporal consistency, semantic alignment, etc., or is it a single undifferentiated score?

### Questions
Typo on line 77: "ignificantly" should be "significantly".

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
This paper addresses this problem: the prohibitive memory and computational cost of ReFL when using large-scale video reward models. The authors propose VELR, an efficient framework centered around an ensemble of Latent Reward Models. The core idea is to bypass the costly backpropagation through the VAE decoder and the pixel-space vRM by training LRMs to predict rewards directly from the diffusion model's latent representations. The paper claims that this approach, enhanced by ensemble techniques for robustness and a "truncated mid-step" training strategy for efficiency, significantly reduces memory consumption (by up to 150GB) while achieving performance comparable to standard ReFL methods.

### Strengths
- Quantifiable achievements like reducing memory usage to as low as 12.4% of the standard ReFL baseline demonstrate the framework's efficiency. 
- While individual components (LRM, Ensemble, Replay Buffer) are not novel in isolation, their combination into a cohesive system to solve this specific problem works well.

### Weaknesses
1. Limited Methodological Novelty: The concept of LRM was previously introduced, and the use of ensemble learning to improve robustness and OOD generalization is a standard, widely-used technique in machine learning. The novelty is confined to the specific application and combination of these existing ideas.
2. Insufficient Analysis of "Proxy Risk" and Reward Distortion: The LRM is fundamentally an imperfect proxy for the true vRM, which introduces a critical risk: the optimization may be guided by the LRM's prediction errors. The paper identifies "reward hacking" but fails to deeply analyze this "proxy risk". The entire framework's stability rests on the fidelity of the LRM, yet there is no quantitative analysis of the error between the LRM's predictions and the vRM's true scores, nor a discussion of how this error might be amplified during training.
3. Methodological Choices Lack Rigorous Justification: Several key design decisions appear to be based on heuristics rather than thorough empirical or theoretical backing.
     - The "Truncated Mid-Step Setting" is justified by intuitive observations ("early steps are blurry, late steps have minimal effect") but lacks a proper ablation study. The paper does not provide experiments comparing different start points (t_mid) or backpropagation durations (K) to prove that their chosen "mid-step" range is truly optimal or generalizable.
     - The LRM architecture (3D CNN + Transformer) is asserted without ablative evidence. It is unclear if both components are necessary or if simpler architectures could suffice.

### Questions
1. **Proxy Risk**: Could you provide a quantitative analysis of the "proxy error" (e.g., correlation, MSE, or error distribution) between the ensemble LRM and the ground-truth vRM on an out-of-distribution test set? How does your framework ensure that this inherent error is not amplified during fine-tuning, potentially leading the diffusion model towards undesirable optima that only the LRM, not the vRM, finds rewarding?
2. **Truncated Mid-Step Setting**: To validate this strategy, could you provide an ablation study on the choice of the starting step t_mid and the number of backpropagation steps K? How sensitive is the final performance and training efficiency to these hyperparameters, and how does this justify that the "mid-step" is a fundamentally better strategy than, for example, a truncated early- or late-stage update?
3. **Spatio-Temporal Control**: While the LRM architecture is designed to capture spatio-temporal features, can you provide more direct evidence or analysis showing that it is indeed effectively guiding the temporal consistency of the generated videos? For instance, does an LRM trained with this architecture outperform one without the Transformer component in predicting rewards related to long-term consistency?

### Soundness
2

### Presentation
3

### Contribution
2
