## Human Reviewer 1

### Summary
This paper proposes Corruption2Self, a score-based self-supervised framework for MRI denoising. 

Motivation:

The goal is to learn denoising directly from noisy data (without relying on high-quality labels). Their framework aims to fix the label scarcity and over-smoothing of finer details issues in existing supervised and self-supervised methods respectively. 

Contributions:

The framework comprises a generalized ambient denoising score matching (GADSM) loss followed by reparametrization to improve convergence and detail refinement extension to preserve finer spatial features. Further, the authors extend the framework to incorporate additional MRI contrasts to improve performance. The authors finally claim that their framework achieves state-of-the-art performance among self-supervised methods and comparable performance among supervised methods.

### Strengths
1. The paper comprehensively analyzes and applies existing self-supervised and supervised denoising approaches in the context of MRI

2. The paper is well written, original and provides great detail into the workflow of the Corruption2Self (C2F) framework.

3. Incorporation to reparametrization (Table 4) and extensions to multi-contrast settings to improve denoising.

4. Showcases robustness of methodology on varied noise level estimations compared to true noise (Table 9)

### Weaknesses
### A. Detail refinement extension claim
According to the metrics in Table 1, it is unclear if the detail refinement extension is being effective. The improvements in PSNR / SSIM does not seem notable. It would be helpful to include an error bar (for the table), statistical significance test (to show notability) and visuals to show effectiveness.

### B. Applicability and impact
The paper can cover how their workflow can be used in practice while denoising. The following aspects can add more value-add in terms of real-world impact to the paper.
1. Estimation of $σ_{tdata}$ in real-time during inference.
2. Given that traditional methods such as BM3D work decently well (maybe 1-3 points less performant than C2S), how difficult is it to go about training / deploying C2S in the MRI denoising workflow than using non-learning based methods?
3. Would this methodology potentially result in MRI image acquisition?

### Questions
1. Robustness of C2S (M4Raw):
In the context of matching test-train SNR on M4Raw dataset, the authors claim that 
    1. (Line 327-328) supervised methods such as SwinIR and Restormer perform better when the noise characteristics of train and test data are similar
    2. (Line 328-330) C2S achieves better generalization.
According to 
    1. Table 3 (Results where test data SNR > train data SNR): C2S perform similar to SwinIR for eg.
    2. Table 7 (Results where test data SNR ~ train data SNR): C2S perform similar to SwinIR (although both have higher metrics here).

    It seems that both methods perform similar to each other in both the conditions?
    Are the metrics higher in latter because the labels in Table 7 are noisier than in Table 3? I'm not sure if we can conclude that C2S is more generalized and supervised is not from the above data?

2. I'd be interested to know the intuition behind why reparametrization works. Mathematically, it seems very similar except that we sample ${\tau}$ ~ $U(0, T]$ rather than ${\tau}$ ~ $U(0, T']$ due to $T >> T'$ . Does this approximation help in convergence?

3. Does reparametrization and detail refinement extension help in the case of FastMRI dataset also? Curious as effectiveness for only M4Raw dataset have been reported.

4. How would one estimate `T` (max corruption level) before training the model?

5. Given that complementary pair of contrasts improve denoising, curious to know if all the 3 contrasts can be used (eg. T1, T2 and FLAIR) for instance to further improve denoising?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposed Corruption2Self, a self-supervised MRI denoising method based on ambient diffusion (training diffusion models using corrupted data). The authors proposed an algorithm named Reparametrized Generalized Ambient Denoising Score Matching and showed its superior performance compared to a number of baselines, supervised and self-supervised methods, on M4Raw (containing pairs of real noise and ground truth) and FastMRI datasets (synthetic noisy images).

### Strengths
- The proposed algorithm GADSM has sound math groundings. 
- Authors did extensive experiments to compare the proposed algorithm to a number of baselines and showed its superior performance.
- Authors discussed the application of the algorithm to multi-contrast MRI, which is an overlooked field in MRI denoising.

### Weaknesses
- This paper's originality seems to be limited. The proposed method Reparametrized GADSM is a straightforward extension to ADSM [1]. In addition, authors failed to point out the challenges when applying self-supervised denoising methods in natural images to MRI images. It seems to be that except for the multi-contrast part, the others are natural extensions of techniques that have already been tested on natural images.
- The paper's problem setup is very similar to Noiser2noise [2], both handling Gaussian noise with known sigma. I think it is an important baseline to be tested. 
- The self-supervised method used as baselines in this paper are too outdated. There are a number of newer methods under the category of blind-spot network (J-invariance) such as LG-BPN [3] and PUCA [4], which has more powerful architectures to increase the accuracy in predicting the value in blind spots, therefore better PSNR/SSIM numbers. Even though these methods were proposed to handle noise with spatial correlation (i.e. not pixelwise independent), the architectures can be easily optimized for independent noise (by making the blind spot be just 1 pixel). 
- In Fig. 4, C2S seems to have more blurry results than R2R. Some details seem to be harder to see.
- The dataset shown in Fig. 4 seems to be unfair for supervised method, since the label is very noisy. Authors may want to re-consider the statement that "the potential of self-supervised learning to match or even surpass supervised methods in MRI denoising" (Introduction).
- The multi-contrast experiments are not fair to other baselines such as Noise2noise and R2R, since the other contrasts can be easily included as an extra channel in the model input to boost performance. 
- Is hallucination problem of generative diffusion models a concern here? How can it be addressed?

[1] Daras G, Dimakis AG, Daskalakis C. Consistent Diffusion Meets Tweedie: Training Exact Ambient Diffusion Models with Noisy Data. arXiv preprint arXiv:2404.10177. 2024 Mar 20.
[2] Moran N, Schmidt D, Zhong Y, Coady P. Noisier2noise: Learning to denoise from unpaired noisy data. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2020 (pp. 12064-12072).
[3] Wang Z, Fu Y, Liu J, Zhang Y. Lg-bpn: Local and global blind-patch network for self-supervised real-world denoising. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2023 (pp. 18156-18165).
[4] Jang H, Park J, Jung D, Lew J, Bae H, Yoon S. PUCA: patch-unshuffle and channel attention for enhanced self-supervised image denoising. Advances in Neural Information Processing Systems. 2024 Feb 13;36.

### Questions
- Authors may want to point out clearly how MRI denoising is different from natural image denoising. What are the challenges? Why does it worths special attention? 
- In the multi-contrast experiment, how extra contrasts were used as inputs? They were directly modeled by diffusion models (i.e. a channel of X_0 ... X_T) or as a conditional channel (i.e. diffusion models learn p(target contrast|other contrasts))?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents Corruption2Self (C2S), a score-based self-supervised denoising framework specifically designed for MRI data. C2S uses a reparameterized noise schedule and applies Generalized Ambient Denoising Score Matching (GADSM) to extend traditional score matching approaches to scenarios where only noisy data are available. Key contributions include the introduction of a reparameterized noise level function to stabilize training, as well as a multi-contrast extension to leverage complementary information across MRI contrasts. Experimental evaluations on M4Raw and fastMRI datasets show competitive results for C2S, often surpassing both classical and self-supervised methods in denoising performance metrics.

### Strengths
* Reparameterization of Noise Levels: The proposed reparameterization of noise levels is a noteworthy contribution, offering enhanced training stability and convergence. This allows the model to sample uniformly across the noise range, leading to smoother training curves and better generalization.
* Comparison with Self-Supervised and Supervised Methods: The paper includes extensive quantitative comparisons with self-supervised and supervised denoising models, establishing C2S as a strong self-supervised alternative in terms of PSNR and SSIM.
* Multi-Contrast Extension: Incorporating multi-contrast data is a beneficial approach that leverages complementary MRI contrasts, enhancing structural preservation and improving the quality of denoised images.

### Weaknesses
* Limited Novelty Beyond Classical Denoising Diffusion Probabilistic Models (DDPM): The use of a score-based approach is similar to DDPM without substantial differentiation. Although the reparameterization is innovative, the rest of the framework closely resembles classical score-based diffusion models, raising concerns about the originality of the overall approach.
* Noise Level Estimation Error Not Clearly Specified: While Figure 5 attempts to show robustness to noise level estimation error, the specific impacts and handling of these errors in practical settings remain unclear. Further clarity on this aspect would strengthen the model’s practical applicability.
(Fluctuations in Training Stability: Although the reparameterization claims to stabilize training, Figure 2 indicates some fluctuations, suggesting that stabilization might not be consistent across all noise levels. This could impact reproducibility and model robustness, especially under varying noise conditions.
* Parameter Specification in Equations: The notation for key parameters, such as $\lambda_{out}$ ​and $\lambda_{skp}$​ , lacks clear definitions in the methods section, which could hinder understanding and replication of the proposed framework.
* Effects of Corruption Level (T) in Training: The paper does not provide guidance on the relationship between higher corruption levels (T) and the required number of training iterations to achieve optimal performance. This omission could limit the applicability of the method in datasets with different noise characteristics or levels of corruption.
* Counterintuitive Results in Multi-Contrast Experiments: In the multi-contrast experiments, incorporating T1 contrast data seems to worsen the results. This is surprising since T1 typically offers structure-rich information that could enhance denoising. An analysis of this phenomenon would provide deeper insights into the limitations of C2S in multi-contrast applications.

### Questions
1. Given that this method is closely aligned with score-based DDPM, what aspects of C2S differentiate it from classical diffusion-based approaches, aside from the reparameterized noise level function?
1. How does the noise level estimation error affect denoising quality in practical scenarios? Is there a threshold or method for mitigating significant deviations in noise estimation?
1. How would the training duration change if a higher maximum corruption level $T$ were used? Does this require additional training epochs to reach convergence, as hinted at in the ablation study?
1. Could the authors clarify why adding T1 contrast data in the multi-contrast experiments led to reduced performance? This is counterintuitive, as T1 contrast generally provides valuable anatomical information.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3