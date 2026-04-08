## Human Reviewer 1

### Summary
The paper proposes training-free Diffusion Transformer (DiT) acceleration named AdaCache. The method is motivated by the fact that different videos require different amounts of computation. AdaCache decides whether to skip or recompute the cache during the cache step defined by the schedule, using an introduced rate-of-change metric $c_t$, which is basically L1 distance between residual block features in current and previous cache schedule timesteps.  Authors further augment rate-of-change metric c_t with a Motion Regularization (MoReg) metric to add information about the motion content of the generated video.

Overall, the AdaCache idea has good potential. However, the current version of the manuscript needs significant revision to be accepted for this conference. Please refer to the Weaknesses and Questions sections for more details.

### Strengths
1. Novelty: The idea of adaptive caching seems novel in the field of diffusion model caching.
2. Motivation: The paper provides a clear motivation for AdaCache method.
3. Clearness: The method is simple and easy to understand.

### Weaknesses
1. Method section requires clarifications: 

a. The paper lacks information about the selection of rate-of-change schedule hyperparameters.

b. Lines 286-287 stat that authors observe that unique caching schedules for each layer will make the generations unstable. This important observation requires further explanation and clarification.

2. Experiment results require better presentation:

a. There are concerns regarding the reported speedup and latency. Given that AdaCache is not a deterministic method and inference time for different videos varies, it is incorrect to report just the mean inference time and speedup for all videos. Standard deviation (std) values should be included.

b. Ablation studies are performed on only 32 videos. Considering that AdaCache is not deterministic, the results may have low statistical significance.  

c. AdaCache without MoReg is a general method that could apply to image diffusion transformers. A comparison with prior works on DiT architectures for image generation, such as PIXART-α for T2I generation and DiT-XL for class-conditional image generation on ImageNet, is suggested.

d. The paper missed a comparison with another DiT caching method like FORA [1].

3. The paper lacks Limitation section. It would be interesting to see if there are videos on which AdaCache performs worse than its deterministic competitors.

### Questions
1. My main concerns regarding this paper are related to experiments results presentation and clarification of method hyperparameters:

a. Since method is not deterministic, the results should include std values for inference time and speedup (See Weakness 2a). Moreover, apart of adding std values, I recommend to conduct ablation studies on more than 32 videos (See Weakness 2b).

b. AdaCache without MoReg is not tied to video generation, I recommend to include image generation DiTs caching (See Weakness 2c).

c. I suggest authors to provide information about rate-of-change schedule hyperparameters selection, as it is crucial pert of the proposed method (See Weakness 1a).

2. Additional group of questions is not as important as the main one, but can also help to improve the quality of the manuscript:

a. Statement regarding unique caching schedules for each layer needs clarification (See Weakness 1a). 

b. The paper would greatly benefit from method’s limitations analysis (See Weakness 2).

c. The authors may include FORA [1] in comparisons.

d. It interesting to see how AdaCache performs on MMDiT models such as CogVideoX [2] and SD-3 [3].

e. In line 196, it would be beneficial to explain which features were used for visualization.


[1] Selvaraju, P., Ding, T., Chen, T., Zharkov, I., & Liang, L. (2024). Fora: Fast-forward caching in diffusion transformer acceleration. arXiv preprint arXiv:2407.01425.

[2] Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., ... & Tang, J. (2024). Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072.

[3] Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., ... & Rombach, R. (2024, March). Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper introduces a training-free method called Adaptive Caching (AdaCache) to accelerate video Diffusion Transformers (DiTs). AdaCache is based on the idea that "not all videos are created equal," meaning some videos require fewer denoising steps to achieve reasonable quality. It caches computations through the diffusion process and devises a caching schedule tailored to each video generation to maximize the quality-latency trade-off. Additionally, the paper introduces a Motion Regularization (MoReg) scheme to utilize video information within AdaCache, essentially controlling compute allocation based on motion content. These plug-and-play contributions significantly speed up inference (e.g., up to 4.7× faster on Open-Sora 720p - 2s video generation) without compromising generation quality across multiple video DiT baselines. The code for this method will be made publicly available.

### Strengths
1. Adaptive Caching achieve very good performance even compared with recent PAB paper. I very appreciate it.

2. This approach requires no training and can seamlessly be integrated into a baseline video DiT at inference, as a plug-and-play component.

3. Motion Regularization (MoReg) to allocate computations based on the motion content in the video being generated seems to be very reasonable.

### Weaknesses
1. Regarding the choice of metric, why was the Mean Squared Error (MSE) selected directly? Can the MSE metric truly reflect the actual reduction in features between adjacent steps? Are there alternative metrics that might be more suitable, or can you provide comparisons with other metrics such as the cosine similarity metric or others?

2. Secondly, I'm interested in knowing if the proposed method is compatible with large Text-to-Image (T2I) base models, like FLUX. If it is, what would be the expected impact on the performance metrics?

3. Although above questions exists, I think this is a really valuable paper

### Questions
On the whole, I consider this paper to be well-executed. However, I'm intrigued by the possibility of identifying a metric that could more accurately assess the redundancy in the system. Furthermore, I'm curious about the potential of integrating layer-wise broadcasting dynamically with distillation techniques to enhance real-time video generation capabilities.

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper introduces Adaptive Caching (AdaCache), a plug-and-play, training-free method to accelerate video generation using diffusion transformers.  The authors note that "not all videos are created equal", meaning some videos don’t need as many processing steps to reach high quality. Based on this, they propose an adaptive caching strategy that reduces the computation of denoising steps based on the rate-of-change. The authors also introduce a Motion Regularization (MoReg) scheme to adjust the caching schedule, which can allocate computation based on video motion content for improving the quality-latency trade-off. Experimental results demonstrate that AdaCache significantly speeds up existing video diffusion models.

### Strengths
a. The approach presented is straightforward, and the method section is generally clear and easy to follow.
b. The motivation for the work is reasonable and interesting, i.e., "not all videos are created equal".
c. AdaCache provides a training-free acceleration method that can be applied to existing video diffusion models, achieving significant speedups without additional model training.

### Weaknesses
1. Lines 285-287 mention that using unique caching schedules for each layer makes the generations unstable, but it’s unclear why this is the case. It would help if the authors provided an explanation.
2. Equation 5 introduces a codebook for the caching rate, but it’s not clear what this codebook is or how it’s created. The authors should add more details to clarify this part of the method.
3. While Table 1 shows AdaCache outperforming PAB, the qualitative comparison in Fig. 7 shows a different result. AdaCache seems to lose more visual detail, especially in the details. This raises concerns about its practical quality compared to PAB.
4. In Table 1, AdaCache achieves better VBench results than the baseline. The authors should explain why the accelerated video have better results, especially since the visual quality in Fig. 7 is noticeably worse than the baseline. 
5.  In Table 1, the SSIM of AdaCache-slow on Line 346 appears unusually high.

I’m concerned about the fairness of the experiments. On the OpenSora model, PAB results are based on text-to-video task, while AdaCache is tested on image-to-video task.  I keep my original Rating.

### Questions
1. The primary concern with this paper is the practical effectiveness of AdaCache. More qualitative comparisons are needed to robustly demonstrate AdaCache’s effectiveness in preserving visual quality, especially for the detail generations.
2. The authors need to provide more detailed methodological details, such as the construction and role of the codebook in caching rates.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
5

### Confidence
5

---

## Human Reviewer 4

### Summary
This paper presents AdaCache which accelerates video generation by caching residual computations and devising adaptive caching schedule without requiring re-training. It also introduces MoReg to optimize computation based on motion. This paper demonstrates the effectiveness of AdaCache across various open-source models.

### Strengths
1.A novel adaptive algorithm is designed to control the number of steps for reusing cached features.

### Weaknesses
1. typo need to be corrected:
    1. Line346: SSIM of AdaCache-slow should be 0.7910?
2. The methodology section lacks clarity in certain areas. For example, in Line 276, a pre-defined codebook is mentioned, and the reviewer wants to know how this codebook was pre-defined. Was it manually set? If so, what criteria or method were used to set it? Is there any further detailed explanation regarding it?
3. The qualitative comparison provided by the authors is insufficient and does not adequately demonstrate the superiority of AdaCache. Although AdaCache achieves a higher acceleration speedup, as shown in Figure 7, its results on Open-Sora-Plan (e.g., Rows 1, 2, 6, 7) and Latte (e.g., Rows 2, 4, 6, 7) are significantly worse than those of PAB and the Baseline. Based on the visual quality presented in Figure 7, the reviewer expresses concerns about the stability of AdaCache in terms of visual quality.

---


After reviewing the authors' latest response to Reviewer 5tUr's comments, the reviewer has decided to further lower the score. The primary concerns stem from:

The reliability of quantitative comparison data

(i) In the latest responses (3) and (4), the authors stated: "We follow the original inference settings suggested by the original contributors—in OpenSora, this is image- and text-conditioned generation." This implies that in Table 1, the experimental setup for OpenSora involves image- and text-conditioned generation. However, as far as the reviewer knows, PAB does not use reference images as image conditions by default in OpenSora. Yet, the performance metrics listed for PAB in Table 1—including fidelity metrics such as PSNR and SSIM, as well as reference-free metrics like VBench—are entirely identical to those reported in the PAB paper. The reviewer is uncertain whether these data were directly from the PAB paper. If so, the comparison would be unfair since PAB defaults to operating without image conditioning.

(ii) The sources of the Delta-Dit and TGATE data in Table 1 are unclear. Additionally, the FLOPS and visual quality results are unusual, showing almost no speedup.

Insufficient clarity in the experimental details presented in the paper, which may lead to misunderstandings

(i) The authors conducted experiments on three models: Open-Sora, Open-Sora-Plan, and Latte. Open-Sora-Plan and Latte are text-conditioned, while Open-Sora uses a text- and image-conditioned setup. The reviewer believes that such a unique configuration should be explicitly clarified in the paper.

(ii) In the rebuttal, the authors mentioned that for OpenSora, reducing the timesteps from 100 to 30 leads to a dramatic change in the speedup factor, from 4.5x to 2.24x. This significant variation should be emphasized and analyzed in the paper. However, the reviewer could not find any related discussion in the manuscript.

---
**The final update**

The paper initially received a rating of 6 because the reviewer, while expressing concerns about the reliability and reproducibility of the methodology as well as the rigor of the experimental section, appreciated the motivation behind the work, encapsulated in the statement: "not all videos are created equal." Based on this, the reviewer assigned a rating of 6.

The reasons for the two rating reductions

(1) During the rebuttal period, the author mentioned that modifying the setup from 100 timesteps to 30 timesteps led to a significant change in speedup, decreasing from 4.5x to 2.24x. This detail, however, was neither thoroughly discussed nor emphasized in the paper. It is evident that **the influence of this variable is much greater than that of resolution and video size, which were included in the ablation study**.
The absence of detailed discussion and emphasis on such experimental findings and conclusions not only risks confusing readers but also raises questions about the reliability and reproducibility of AdaCache. As a result, the reviewer lowered the rating for the first time.

(2) The lack of rigor in experimental comparisons

(i) The authors mentioned that the use of image conditions in OpenSora was motivated by an issue raised in the original repository, which stated that "the lack of a reference image leads to inconsistencies in video quality," i.e., a decline in motion consistency and visual quality. This implies that **the introduction of image conditions improves visual quality**.

(ii) The authors further claimed that AdaCache **directly utilized data from PAB because they reproduced PAB and obtained similar quantitative numbers** (with negligible changes). However, to the reviewer’s knowledge, PAB does not employ image conditions. **If the authors introduced image conditions to PAB for a fair comparison but still achieved similar VBench metrics, this appears highly counterintuitive.** If this is the case, what is the rationale behind AdaCache incorporating reference images? It should be noted that synthesizing reference images incurs significant computational overhead.

(iii) Given that a significant portion of the paper, including the experiments in the ablation study, was conducted on OpenSora, the inclusion of image conditions cannot be overlooked. The introduction of image conditions not only affects visual quality but may also impact the L1 distance between features, thereby influencing the caching rate. **This aspect should have been emphasized in the experimental section, yet it is not mentioned even once throughout the paper.**

(iv) To ensure fair quantitative comparisons, AdaCache should follow PAB's experimental setup, exclude reference images, and retest metrics such as VBench, SSIM, and PSNR. 

Due to concerns about the rigor of the experiments, the reviewer downgraded the rating once again.

### Questions
1. The caching rate for the steps following step t is determined based on the rate of feature change between steps t and t+k. The reviewer wonders whether this metric is reasonable. For instance, as shown in Fig. 2, during the early and late stages of sampling, the L1 curve exhibits rapid changes, characterized by a large derivative. Would relying on the differences at earlier time steps to determine the subsequent caching rate introduce errors?
2. Could the authors provide more visual quality comparison results? For example, visualizations of the sampling process under different configurations (fast, slow) and how different video content leads to varying caching schedules, to more intuitively demonstrate the mechanism and effectiveness of the designed caching schedule.
3. PAB demonstrates impressive results in multi-GPU parallel processing. Can the authors' method leverage similar techniques (e.g., DSP) to scale to multi-GPU parallel inference? What would the efficiency be like?
4. The reviewer wants to know the source of the Delta-DIT performance in Table 1 and why there is almost no acceleration.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
3

### Confidence
4