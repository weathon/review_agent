# Product of Experts for Visual Generation

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Modern neural models capture rich priors and have complementary knowledge over shared data domains, e.g., images and videos. Integrating diverse knowledge from multiple sources—including visual generative models, visual language models, and sources with human-crafted knowledge such as graphics engines and physics simulators remains under-explored. We propose a probabilistic framework that combines information from these heterogeneous models, where expert models jointly shape a product distribution over outputs. To sample from this product distribution for controllable image/video synthesis tasks, we introduce an annealed  MCMC sampler in combination with SMC-style resampling to enable efficient inference-time model composition. Our framework empirically yields better controllability than monolithic methods and additionally provides flexible user interfaces for specifying visual generation goals.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a training-free, probabilistic framework for controllable visual generation by combining heterogeneous pre-trained "expert" models (e.g., generative models, discriminative VLMs, and physics simulator software). The core of the method is a novel sampling algorithm that draws from the combined "Product of Experts" distribution by interleaving three techniques: Annealed Importance Sampling (AIS) to gradually refine samples from noise, MCMC to ensure fidelity to the generative experts, and Sequential Monte Carlo (SMC) to resample and filter particles. The method is instantiated and evaluated on complex tasks, including graphics-engine-instructed image editing, physics-instructed video generation, and layout-controlled text-to-image synthesis. The results demonstrate superior controllability over baselines, effectively adhering to precise object poses and physics-based motion trajectories while maintaining high visual quality.

### Strengths
- The paper provides a classic and mathematically sound algorithm for combining heterogeneous pre-trained models, including both generative and discriminative experts. The method can generate images and videos with high controllability without extra training.

- The paper demonstrates many instantiations of this framework across different tasks (image editing, video generation, layout control) and expert types (flow models, autoregressive models, VLMs, physics engines), and the proposed method consistently outperforms monolithic baseline approaches.

- The main paper and the appendix provide extensive implementation details, qualitative results, and ablations, which support its claims and reproducibility.

### Weaknesses
- The method is inherently slow due to its iterative nature, requiring $T$ annealing steps, $L$ parallel particles, and $K$ MCMC steps per particle. This makes it less practical for real-world GenAI applications, with image generation taking ~4 minutes and video generation taking 5-30 minutes. In my understanding, the proposed method is more like a proof of concept instead of a feasible path for future visual generation frameworks.

- The use of the physics engine can be "ad-hoc." For instance, the autoregressive video task requires the user to provide a full RGB rendering from the simulator. If a user must **manually** create complex 3D renderings or animations in professional software like Blender to serve as a control signal, the practical cost and effort may be prohibitively high.

### Questions
Please see the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors proposed a sampling method from a joint of generative models with potential discriminators. Their idea is to apply AIS on the product of generative distributions, then interleave the AIS steps with an SMC reweighting using the discriminators when they are presented. The authors evaluated their method both qualitatively and quantitatively on various visual modalities.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed method is general and works well with different expert models across different modalities.
3. The generated results can show better quality against the competing baseline, e.g., in Figure 2 the proposed method preserves the original image more faithfully.

### Weaknesses
1. The proposed method is more heuristic than it appears to be. The main novelty is applying some tweaks to well-established sampling methods when discriminators are involved, which work well on the modern large expert models. At some point, this feels like a glorified classifier-based guidance with no theoretical guarantee provided. Maybe the authors can tone down a bit.
2. The model performance is not always on the better side. This includes both the metrics across the tables and the visual examination such as Figure 5.

### Questions
1. The ablation study is a bit limited. How would the vanilla classifier-free guidance and classifier-based guidance perform with all these expert models? This should serve as the baseline for justifying the necessity of the proposed sampling scheme.

### Soundness
3

### Presentation
2

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
Modern neural models possess complementary knowledge about data like images and videos, but combining this diverse expertise—from generative AI to structured simulators—is still challenging. The authors proposed a probabilistic framework that unifies these heterogeneous models into a single product distribution. For controllable image/video synthesis, the paper introduces an efficient inference-time sampler using annealed MCMC with SMC-style resampling. The proposed method provides superior controllability and more flexible user interfaces than monolithic approaches.

### Strengths
The proposed method achieves visually satisfactory generative results. The paper is well-structured and easy to follow.

### Weaknesses
The performance improvements yielded by the proposed approach is kind of small, compared to the benchmarking algorithms, such as Depth2V for the setting of Object-Centric Simulation Input.

### Questions
Please refer to weaknesses for my main concerns.

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
This paper aims to solve a critical challenge: given models with rich priors and knowledge over shared data domains, how to effectively integrate this diverse knowledge from multiple sources. The paper proposes an annealed MCMC sampler and SMC resampling for better model composition. The qualitative and visual results show success in both image and video generation.

### Strengths
1. This paper is clearly presented, with sophisticated structure and logical flow.
2. It addresses an interesting and significant problem: how to effectively utilize different experts for efficient and controllable visual generation.
3. The paper presents detailed experimental results, showing improvement compared to baselines.
4. It also demonstrates strong visual results with clear corresponding explanations.

### Weaknesses
Method and motivation — Sections 3.1 and 3.2 emphasize that, to sample from the product of a set of generative experts, the paper uses Markov Chain Monte Carlo to iteratively refine samples based on their likelihood under the product distribution (3.1), and employs AIS and SMC to draw samples from the product-of-experts distribution. However, there still exists concern about the intrinsic sharpness of the product-of-experts energy landscape, even with intermediate tempered distributions and particle resampling weighted by discriminative likelihoods.
The overall energy surface can remain highly peaked. There is a chance that particles collapse into narrow regions of high probability, perhaps leading to a drop in effective sample size and poor mixing.
Can the authors provide more evidence that the proposed framework overcomes the structural difficulty imposed by additive log-likelihoods in the PoE formulation?

### Questions
In Table 2, can the authors provide video quality metrics for analysis? It appears comparatively weak in Aesthetic compared with other baselines; a further explanation (visual or textual) would clarify the results.
Can the authors also provide more failure cases where the framework does not perform well (in either image or video quality, efficiency, or controllability)?

### Soundness
3

### Presentation
3

### Contribution
3
