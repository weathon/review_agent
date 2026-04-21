# Solving Diffusion ODEs with Optimal Boundary Conditions for Better Image Super-Resolution

- Avg Score: 6.67
- Decision: Accept (poster)
- Scores: 6, 8, 6

## Abstract
Diffusion models, as a kind of powerful generative model, have given impressive results on image super-resolution (SR) tasks. However, due to the randomness introduced in the reverse process of diffusion models, the performances of diffusion-based SR models are fluctuating at every time of sampling, especially for samplers with few resampled steps. This inherent randomness of diffusion models results in ineffectiveness and instability, making it challenging for users to guarantee the quality of SR results. However, our work takes this randomness as an opportunity: fully analyzing and leveraging it leads to the construction of an effective plug-and-play sampling method that owns the potential to benefit a series of diffusion-based SR methods. More in detail, we propose to steadily sample high-quality SR images from pre-trained diffusion-based SR models by solving diffusion ordinary differential equations (diffusion ODEs) with optimal boundary conditions (BCs) and analyze the characteristics between the choices of BCs and their corresponding SR results. Our analysis shows the route to obtain an approximately optimal BC via an efficient exploration in the whole space. The quality of SR results sampled by the proposed method with fewer steps outperforms the quality of results sampled by current methods with randomness from the same pre-trained diffusion-based SR model, which means that our sampling method ''boosts'' current diffusion-based SR models without any additional training.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper propose to steadily sample high-quality SR images from pre-trained diffusion-based SR models by solving diffusion ordinary differential equations with optimal boundary conditions.

### Strengths
The paper seems well-written

### Weaknesses
1 The methods compared in Table 2 are all outdated. It is necessary to compare them with some state-of-the-art real-world super-resolution tasks [1,2].

2 You need to compare with the state-of-the-art diffusion SR method [3], which also appears to have performed bicubic-SR and real SR tasks.

3 From the figures in the appendix, it seems that the visual improvement is not significant.

4 Please provide a comparison of the computational complexity and runtime for all the methods mentioned in the paper to show your effectiveness.

5 Tables 1 and 2 show that the PSNR is not particularly high, indicating that the network's fidelity is not good. Super-resolution tasks not only seek visual improvement but also place great importance on fidelity. Compared methods have better fidelity. Therefore, I suggest the authors work on improving both PSNR (fidelity) and LPIPS, as this would provide stronger evidence of the effectiveness of your method.


[1] Real-esrgan: Training real-world blind super-resolution with pure synthetic data

[1] Knowledge Distillation based Degradation Estimation for Blind Super-Resolution

[2] Diffir: Efficient diffusion model for image restoration

### Questions
see weakness

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper analyzes the characters of boundary conditions (BCs) in the diffusion-ODE sampling process of diffusion-based Super-Resolution (SR) models and finds out that the optimal BC is shared by different LR images approximately. Based on the analysis, the paper further proposes a method of obtaining an approximately optimal BC based on a reference LR-HR set. The derivation of the paper is mathematically complete. Experiments on the tasks of both bicubic-SR and real-SR demonstrate the superiority of the proposed approximately optimal BC.

### Strengths
1. The motivation of analyzing the BCs of diffusion ODEs is intuitive. Different BCs would apparently lead to different SR results. It is vital to find the rule of how BCs affect the results and to propose a method to get a better BC.
2. The analysis of the paper is mathematically complete. The paper proposes the concept of optimal BC and proves that such an optimal BC is common to different LR images. The conclusion is seemingly solid.
3. The experiments are sufficient enough to support the theory. The paper claims that the method of obtaining the approximately optimal BS is not related to the degradation model. Experiments on the tasks of bicubic-SR and real-SR demonstrate the assertion. Further ablation studies show the influence of reference set and the set of BCs.

### Weaknesses
1. The proposed method leverages LPIPS as the implementation of distance measurement function M(·,·). Can we leverage pixel-level metrics like negative PSNR as M? The authors should give more discussions.
2. The paper assumes that the model is well-trained. However, even a “well-trained” model cannot fit the real data distribution precisely. Thus, does such an assumption cause potential inaccuracy of the conclusion?
3. The paper claims that p_θ (y|h_θ (x_T,ϕ)) is approximately uniform. However, what ensures that the model does not have biases when leveraging the “blank token”, which is essentially a placeholder token (that doesn't actually exist)?
4. It seems that the process of calculating the approximately optimal BC is time-consuming. How long does it take?

### Questions
1. The paper only discusses the context of SR (and other low-level tasks in the Sec. 5). But it seems that the theory does not limit the relationship between guidance and generated results. Do the authors think that the method can be leveraged in more general generation tasks such like text-to-image generation?
2. It seems that the improvement on StableSR (in the task of real-SR) is less than the improvement on SR3 (in the task of bicubic-SR). Could the authors discuss the reasons for this phenomenon?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on the randomness in the inverse process of the diffusion model applied to super-resolution tasks, which makes it difficult to ensure the quality of SR results. By solving diffusion ordinary differential equations with optimal boundary conditions, the authors propose an efficient plug-and-play method that enables diffusion models to stably sample high-quality SR images with fewer sampling steps.

Post rebuttal:
I have read the rebuttal and would like to raise my score a little bit.

### Strengths
1.	The proposed method achieves good visualization results with fewer steps in SR tasks. 
2.	The proposed method has good flexibility and can be applied to multiple diffusion models

### Weaknesses
1.	The experiments are insufficient. Although the proposed method has achieved good results on LPIPS, the PSNR values on multiple test sets are very low. The author did not discuss it in depth and did not show the results of SSIM.
2.	The paper stated that the proposed method has fewer parameters and is more efficient than the GAN method, but did not show the corresponding comparison results. Such as comparison of specific parameters or running time.

### Questions
Further supplement and improve the experiment, please refer to the Weaknesses for details.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
