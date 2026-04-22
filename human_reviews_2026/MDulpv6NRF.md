# Epipolar Geometry Improves Video Generation Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Video generation models have progressed tremendously through large latent diffusion transformers trained with rectified flow techniques. Yet these models still struggle with geometric inconsistencies, unstable motion, and visual artifacts that break the illusion of realistic 3D scenes. 3D-consistent video generation could significantly impact numerous downstream applications in generation and reconstruction tasks.
We explore how epipolar geometry constraints improve modern video diffusion models. Despite massive training data, these models fail to capture fundamental geometric principles underlying visual content. We align diffusion models using pairwise epipolar geometry constraints via preference-based optimization, directly addressing unstable camera trajectories and geometric artifacts through mathematically principled geometric enforcement.
Our approach efficiently enforces geometric principles without requiring end-to-end differentiability. Evaluation demonstrates that classical geometric constraints provide more stable optimization signals than modern learned metrics, which produce noisy training signals. Training on static scenes with dynamic cameras ensures metric quality while the model still generalize to various dynamic scenes. By bridging data-driven learning with classical geometric computer vision, we present a practical method for generating 3D consistent videos without compromising visual quality.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes improving video diffusion models by introducing epipolar geometry constraints through preference-based optimization. Although large video generation models trained with rectified flow show impressive visual quality, they often suffer from geometric inconsistency, unstable motion, and artifacts. The method enforces classical geometric principles to stabilize camera motion and reduce distortions—without requiring differentiable rendering.

### Strengths
1. The problem addressed is highly important — achieving geometrically consistent video generation remains a major challenge today.

2. The proposed method is simple yet effective.

3. The generated results show clear improvements compared to the baseline.

4. The project website is beautifully designed. If possible, could you let me know which GitHub template it is based on?

### Weaknesses
1.	How should we evaluate 3D consistency when videos inevitably contain dynamic content? If the presence of motion leads to degraded 3D consistency and such data are consequently filtered out, what would be the impact of suppressing dynamic content generation?

2.	Currently, the method is only compared against a single baseline (the original video generation model). However, I think an important baseline is missing — namely, fine-tuning Wan 2.1 with DL3DV and Real10K using LoRA. My reasoning is that works like DimensionX[1] have already demonstrated that, for tasks such as novel view synthesis from a single image, fine-tuning video diffusion models on such datasets alone can yield geometrically consistent generations. Therefore, in this work, it is difficult to determine whether the improvement in core capability truly comes from the proposed epipolar geometry optimization, or simply because the training and test datasets share the same distribution. I also believe that a visual comparison with this baseline would be important. This is my main concern. If fix this problem, I will raise my score.
[1] DimensionX: Create Any 3D and 4D Scenes from a Single Image with Controllable Video Diffusion

3. My understanding of the data fine-tuning pipeline is as follows: for each prompt, three videos are sampled, their scores are computed, and two of them are selected for DPO fine-tuning. Please correct me if I am mistaken. How long does the scoring process take — is it fast? Why are only three videos sampled instead of more? Why do videos with the same caption show noticeable differences in 3D quality? Can the DPO process use GT videos as the higher-score samples? If GT videos are used as high-score references, how does that differ from directly fine-tuning with GT data?

4. Regarding evaluation metrics, the baseline videos clearly show  geometric distortion, and artifacts. Given the obvious visual differences, why do the metrics in Table 1 — apart from Human Eval — show such limited improvement? Does this imply that the chosen metrics are not appropriate, that the evaluation method has limitations, or that even the optimized results still fail to achieve satisfactory 3D reconstruction quality? How large is the quantitative gap compared to using GT videos directly for reconstruction?

5. Could you elaborate on the setting of Table 5 — for example, what rewards were used and which strategy led to the observed performance differences? I find this comparison particularly interesting.
 
6. Regarding the proposed large-scale preference dataset, what do you think is its actual value? Since all the data are generated by video models, and given that Wan 2.1 (1.3B) still shows a significant performance gap from real data, why do we need these imperfect datasets?

### Questions
Please refer to the weeknesses.

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
4

### Summary
The paper proposes to align pretrained video diffusion models with epipolar geometry via preference-based finetuning to enforce the 3D consistency of the generated video. More specifically, it samples multiple videos per prompt, ranks them using a Sampson epipolar error computed from SIFT/RANSAC correspondences, and applies Flow-DPO with a small temporal-variation penalty to avoid degenerated solutions such as frozen scenes. The experiment results shows that the epipolar geometry based preference alignment leads to better 3D consistency of the generated video.

### Strengths
+The idea of rank-based DPO alignment with classical geometry is interesting. The paper applies Direct Preference Optimization (Flow-DPO) so the model learns from pairwise rankings induced by Sampson epipolar error—no absolute, differentiable reward needed. This leverages DPO’s pairwise nature (relative preferences per prompt) and enables efficient latent-space LoRA finetuning without decoding full videos, which is practically useful for video models. 

+LoRA-based Flow-DPO with an explicit static-penalty term (Eq. 3) is easy to reproduce and mitigates the “frozen scene” failure mode.

### Weaknesses
-Dynamic objects are not handled. The approach assumes static scenes for the reward; moving objects violate a single fundamental matrix and can corrupt the signal.This is important because many real prompts include independent object motion. 

-Dynamics–consistency trade-off. Despite the temporal-variation penalty, dynamic degree drops (Table 2), suggesting a residual tendency to damp motion. The paper frames this as an acceptable trade-off but doesn’t systemically tune λ or add complementary rewards to preserve motion amplitude.

### Questions
1. Can the methods be applied on training videos with dynamic objects? For those videos, as long as we can extract the camera poses, and segment the dynamic objects to rule out those objects during reward calculation, the proposed method could be applied. This will greatly increase the breadth of the application.

2. Building the offline preference set (≈54k videos; ~1,980 GPU-hrs) is non-trivial; discussion of scalability, deduping, and potential bias in prompt expansions would help practitioners.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a method for enhancing the 3D consistency of video generation models by using pairwise epipolar geometry constraints and preference-based optimization without requiring end-to-end differentiability, and reducing unstable camera trajectories and geometric artifacts through mathematically principled geometric enforcement. The paper demonstrates that using classical geometric constraints, epipolar geometry, provides simple but stable optimization signals, leading to improved video generation quality and enhanced 3D consistency. The method is evaluated across several metrics and performs well, including 3D consistency, motion stability, and generalization across dynamic scenes.

### Strengths
- The incorporation of classical epipolar geometry into modern video generation models to improve 3D consistency is interesting.
- The paper presents a thorough evaluation framework that assesses various aspects of video generation quality, which provides a holistic view of the model’s performance and demonstrates the practical benefits of the proposed method. The results show that the use of epipolar geometry reduces geometric artifacts and improves motion smoothness.
- The method also generalizes effectively to dynamic content, despite being trained primarily on static scenes with dynamic cameras.

### Weaknesses
- The performance of the proposed method is limited by the capabilities of the base model itself. The training data is generated by the base model and then sorted. Therefore, the upper bounds of 3D consistency and motion complexity are relatively fixed, even if the input prompts are more complex. Due to the inherent randomness of the base model, generating three videos per caption, I would question the proportion of high-quality samples. I am also curious why this paper does not use labeled real videos as the ground truth to further improve 3D consistency.
- Epipolar geometry is typically only applicable to static scenes. It will fail to work with dynamic objects or non-rigid bodies, which are common in real-world scenarios, thus causing problems when scaling up the data.

### Questions
- What is the impact of the hyperparameter $\lambda$ in the temporal variation loss on the final video quality and 3D consistency? A more detailed analysis would be beneficial to understand the trade-offs between motion stability and 3D consistency.
- Theoretically, epipolar geometry constraints cannot effectively constrain dynamic objects, but the results show that even models trained only on static scenes can generalize to dynamic scenes. I'm curious about the deeper insights into how the model acquires this ability.
- I noticed that most of the dynamic objects in the demo are rigid bodies with relatively small movements. What if we were dealing with more extreme but common real-world scenarios? For example, large motions, non-rigid deformations, or sudden occlusions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper improves video diffusion models by incorporating epipolar geometry into the finetuning process. Using Sampson distance to rank candidate generations, the method performs DPO-based preference alignment to encourage 3D-consistent video generation. The authors evaluate extensively across geometric accuracy, video quality, reconstruction, and human preference, showing meaningful improvements in temporal and geometric stability.

### Strengths
- Combining classical geometry with generative models is a good idea and helps pushing video diffusion models toward physically grounded behavior.
- Extensive evaluation protocol makes the empirical results convincing.
- The paper is well structured, clear, and easy to follow. Also it clearly represents limitations and broader impact discussion in Appendix F.
- Preparing a large-scale geometry preference dataset is a significant effort, and publicly releasing it will be useful to the community.

### Weaknesses
- Dynamic degree decreases, showing that the method improves stability partly by reducing motion amplitude. There is a trade-off between temporal consistency and keeping lively, dynamic motion.
- Although better than baseline, some videos still show flickering, blur, and occasional scene drift, especially in challenging or texture-heavy regions.
- The dataset creation pipeline is computationally heavy and slow (as acknowledged in Appendix F), making the method difficult to scale or replicate.

### Questions
- Instead of relying only on pairwise epipolar distance, could multi-frame geometric reasoning (e.g., triangulation error, bundle-adjustment residuals, multi-view consistency) provide stronger or more stable supervision?
- The pipeline is expensive. Are there surrogate metrics, early stopping heuristics, or active-learning strategies that reduce the number of required generations for ranking?
- Could you present detailed failure examples and categorize them? Understanding failure modes in depth would be helpful for future work.

### Soundness
3

### Presentation
3

### Contribution
2
