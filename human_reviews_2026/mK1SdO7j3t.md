# Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Recently, augmenting vision-language-action models (VLAs) with world-models has shown promise in robotic policy learning. However, it remains challenging to jointly predict next-state observations and action sequences because of the inherent difference between the two modalities. To address this, we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework that handles the modality conflict and enhances the performance of VLAs across diverse tasks. Specifically, we propose a multimodal diffusion transformer architecture that explicitly maintains separate modality streams while enabling cross-modal knowledge sharing. In addition, we propose training techniques such as independent noise perturbations for each modality and a decoupled flow matching loss, which enables the model to learn the joint distribution in a bidirectional manner while avoiding the need for a unified latent space. Furthermore, based on the decoupled training framework, we introduce a sampling method where we sample action and vision tokens asynchronously at different rates, which shows improvement through inference-time scaling. Through experiments on simulated benchmarks such as RoboCasa and GR-1, DUST achieves up to 6\% gains over standard VLA baselines and world-modeling methods, with our inference-time scaling approach providing an additional 2-5\% gain on success rate. On real-world tasks with the Franka Research 3, DUST outperforms baselines in success rate by 10\%, confirming its effectiveness beyond simulation. Lastly, we demonstrate the effectiveness of DUST in large-scale pretraining with action-free videos from BridgeV2, where DUST leads to significant gain when transferred to the RoboCasa benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes DUST, a world-model–augmented VLA framework that resolves modality conflicts by using a dual-stream diffusion transformer with separate yet cross-sharing modality streams, independent noise per modality, and a decoupled flow-matching loss—learning the joint distribution without a unified latent space. A joint sampling method enables test-time scaling with asynchronous evolution of action and vision tokens, yielding up to 6% gains on RoboCasa and GR-1, an additional 2–5% boost from scaling, and a 13% success-rate improvement on real Franka tasks; action-free video pretraining (BridgeV2) further enhances transfer on RoboCasa.

### Strengths
Method: The paper proposes a Dual-Stream Diffusion framework for unified vision-language-action modeling. Two diffusion streams—visual and action—are trained in parallel. This dual-branch design provides a fresh direction for integrating perception and control under diffusion-based generative modeling.

Writing: The paper is well written, the method is clearly presented, and the figures/tables are complete and easy to read.

Experiments: The ablation studies are well conducted and validated the effectiveness of model architechture.

### Weaknesses
The primary concern is how the proposed dual-stream diffusion compares to VLM-based unified understanding–prediction VLAs. In addition to DiT-based image-prediction policies (e.g., PAD, VPP), there now exist VLM-driven unified VLA models (e.g., UP-VLA[1], DreamVLA[2]) that likewise address modality conflicts and improve performance across diverse tasks, with the added benefit of strong semantic generalization. The current evaluation focuses on the former class while omitting the latter, which is increasingly becoming the mainstream baseline for comparison.

Furthermore, the work lacks results on mainstream simulation environments (e.g. Calvin, SimplerEnv, and Libero), preventing a fair comparison with advanced models, including the above methods and the pi0 series.

[1] Zhang J, Guo Y, Hu Y, et al. Up-vla: A unified understanding and prediction model for embodied agent[J]. arXiv preprint arXiv:2501.18867, 2025.

[2] Zhang W, Liu H, Qi Z, et al. Dreamvla: a vision-language-action model dreamed with comprehensive world knowledge[J]. arXiv preprint arXiv:2507.04447, 2025.

### Questions
1. Primary concerns about methods and experiments can be seen in weaknesses.
2. Missing demos on real-world experiments. From the reported details, the real-world evaluation is confined to pick-and-place and primarily in-domain, which limits evidence for semantic and skill generalization. In addition, the real-robot results do not include comparisons against advanced baselines, making it difficult to contextualize performance relative to the current state of the art.
3. The authors mention DUST’s high-frequency inference. How does DUST improve inference efficiency compared to other methods, and is there a concrete speed comparison across approaches?

### Soundness
3

### Presentation
3

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
The paper introduces DUST, a vision-language-action (VLA) framework designed to address modality conflicts through joint world modeling and action prediction. DUST employs a VLM-based modality encoder to extract semantic representations from visual and language inputs, while an MMDiT model conditioned on these features predicts future images and action sequences. This design decouples modality generation while preserving cross-modal knowledge transfer. Experiments conducted on RoboCasa, GR-1 simulated benchmarks, and real-world settings demonstrate that DUST outperforms existing baseline methods.

### Strengths
- Propose the dual-stream multimodal diffusion transformer for action and image prediction, and the ablation process also demonstrated the effectiveness of separately processing different modes of propagation.
- DUST has achieved superior performance over the backbone method in two simulated environments.
- DUST can benefit from pretraining on internet-scale data, as ablation studies show.

### Weaknesses
- **Inadequate ablation analysis**  
  The ablation experiments lack of the using of VLM module. Please supplement the relevant experiments by replacing LLM with a common language model (such as the settings of PAD or MDT).

- **Lack of real world demonstration**  
 The real-world experiments involve relatively simple tasks and scenarios. Moreover, the absence of demonstration videos raises concerns about the model’s real-world performance.

- **Insufficient baselines**  
 Due to insufficient baselines, the results may lack persuasiveness.

### Questions
1. How do other new VLA approach perform on GR-1 benchmark? And supplement some baseline methods (π₀, MDT, Seer, PAD) for the experiment.
2. In real-world experiments, for tasks with low success rates (such as task 4), how do the DUST fail? Please provide the corresponding failure case video

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents DUST, a Dual-Stream Diffusion framework that augments Vision-Language-Action (VLA) models with explicit world modeling.
Unlike prior works that either unify modalities in a shared latent space (PAD, EnerVerse) or separate them with one-way conditioning (Video Policy, FLARE), DUST introduces a dual-stream multimodal diffusion transformer that preserves separate vision and action streams while enabling bidirectional information exchange through shared attention layers.

### Strengths
1. The dual-stream diffusion structure elegantly balances modality decoupling with cross-modal communication
2. The asynchronous denoising method during test-time is useful.
3. The presentation is clear with nice figures, the writing is easy to follow.

### Weaknesses
1. Real-world validation is only conducted on four pick-and-place tasks. Including a broader range of tasks could further verify the effectiveness of the proposed framework. (Given the limited rebuttal phase, the authors do not need to add additional real-world experiments.)
2. The paper lacks direct comparisons with the most relevant baselines, such as PAD, Video Policy, Video Prediction Policy, and UAV.

### Questions
1. Could the authors include the most relevant baselines that also use prediction-based methods to enhance VLA models?

2. The authors may also consider evaluating on more widely used benchmarks such as CALVIN and LIBERO. Since these benchmarks are approaching limit, it is acceptable not to achieve state-of-the-art performance, but the results should at least be comparable to prior advanced methods.

### Soundness
3

### Presentation
3

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
DUST presents a well-designed approach to integrating world modeling into Vision-Language-Action frameworks through a dual-stream diffusion architecture. The decoupled yet interactive modality design is conceptually sound and empirically effective, showing consistent performance gains across simulation and real-world tasks. However, while the results are promising, the methodological novelty appears moderate given the growing body of diffusion-based multimodal policy learning research.

### Strengths
1. I think this paper made a good summary for the world model-based VLAs.
2. Frankly speaking, the number of experiments is quite a lot.

### Weaknesses
1. Overall, the experimental evaluation is required to improved. There are only pick and place tasks.
2. How about the control frequency? 
3. I do not think there are significant difference between (b) and (C) in Figure 1.

### Questions
1. Without more experiments about non-pick-and-place tasks. This paper is not acceptable.
2. Do you have any pretraining stage?
3. You should use pi0 as your baseline model, especially you use a similar architecture to pi0. or another diffusion-based VLA. 


My major concern is the experimental evaluation and unsuitable baselines. Therefore, I did not think this approach is well evaluated.

### Soundness
2

### Presentation
2

### Contribution
2
