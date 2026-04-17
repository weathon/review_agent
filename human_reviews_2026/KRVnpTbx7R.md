# DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 6

## Abstract
Vector quantization is common in deep models, yet its hard assignments block gradients and hinder end-to-end training. We propose DiVeQ, which treats quantization as adding an error vector that mimics the quantization distortion, keeping the forward pass hard while letting gradients flow. We also present a space-filling variant (SF-DiVeQ) that assigns to a curve constructed by the lines connecting codewords, resulting in less quantization error and full codebook usage. Both methods train end-to-end without requiring auxiliary losses or temperature schedules. In VQ-VAE image compression, VQGAN image generation, and DAC speech coding tasks across various data sets, our proposed methods improve reconstruction and sample quality over alternative quantization approaches.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the non-differentiability of vector quantization (VQ) by proposing Differentiable Vector Quantization (DiVeQ) and a variant, Space-Filling DiVeQ (SF-DiVeQ). The methods enable gradient flow while preserving hard assignments, with a directional reparameterization of the quantization error. Both methods train end-to-end without auxiliary losses or hyperparameter tuning. Key results on VQ-VAE compression and VQGAN generation tasks demonstrate that DiVeQ and SF-DiVeQ consistently outperform existing approaches in reconstruction and sample quality across multiple datasets.

### Strengths
1. DiVeQ introduces a novel directional reparameterization of the quantization error. The proposed techniques are elegantly derived from the reparameterization trick, and allow for true end-to-end training without auxiliary losses or complex schedules.
2. The authors perform a comprehensive evaluation across both image compression and image generation. The experiments are conducted on a wide range of standard, high-resolution datasets, and the proposed methods consistently outperform a diverse set of baselines.
3. The paper's claims are further confirmed by ablation studies, which analyze the impact of key factors like the noise variance, batch size, and learning rate.

### Weaknesses
1. In Page 4 and Figure 2, the NSVQ incurs a higher distortion than the true nearest-codeword assignment with a fixed probability of approximately 0.67. This claim can be extended beyond a 2D plane, where the probability of incurring higher quantization errors rapidly approaches one as the dimension $D$ increases. This can be mentioned to strengthen the argument.
2. In Section 4, the authors claim SF-DiVeQ avoids codebook misalignment (Fig. 4). However, the t-SNE plots show only a single run per method without providing statistical evidence (e.g., multiple runs with variance across seeds) to confirm that SF-DiVeQ consistently avoids misalignment. The authors are suggested to quantify the improvement across runs.
3. The arguments that DiVeQ requires no hyperparameter tuning seem to be overstated. DiVeQ involves the hyperparameter $\sigma^2$, and different values are used for compression ($10^{-3}$) and generation ($10^{-2}$).

### Questions
1. In Section 3.2, SF-DiVeQ is described as quantizing inputs to "random locations on the line connecting two subsequent codewords." Could the authors clarify the ordering of codewords along the space-filling curve?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to improve vector quantization (VQ) in VQ-VAE compression and VQGAN generation. The paper first analyzes the baseline methods and compares them to identify gaps. Then, the paper proposes DiVeQ, which applies reparameterization tricks to the baseline NSVQ. The paper then proposes an improved version of DiVeQ by replacing the codebook replacement method with codebook interpolation, as suggested by another baseline, SFVQ. The paper conducts experiments on VQ-GAN image generation tasks at 256x256 resolution, and the experimental results show that the proposed method achieves better SSIM, PSNR, FID, and LPIPS scores.

### Strengths
* The paper provides an excellent background and related work section and presents a detailed comparison among them.
* The figures are helpful for understanding the idea.
* The experimental results show that DiVeQ achieves better results compared to the baselines.

### Weaknesses
* The contribution of the paper is not strong.

This reviewer believes that the proposed methods are minor extensions of the baselines, and the motivation and experimental results are not insightful enough. The paper proposes two methods, DiVeQ and SF-DiVeQ. DiVeQ is proposed by adding a direction vector to $v \sim N(0,\sigma^2I)$, and SF-DiVeQ is proposed by adding space-filling methods from SFVQ. In addition, although the paper’s title primarily emphasizes the reparameterization trick, the proposed methods are built on an existing baseline that already uses it. Also, since the motivation and experimental results do not present insightful analysis, this reviewer believes that the contribution of the paper is weak.
* The experiments section should be strengthened.

The paper primarily conducts experiments on VQ-GAN, generating 256x256-resolution images. From the experiments, the paper compares image-related metrics such as FID and PSNR. However, this reviewer believes that the experimental results section could be improved by using different models and tasks and by providing more detailed analysis beyond the numerical results. For example, many papers conducted experiments on various models and tasks such as recent stable diffusion models [C1], visual-language-action tasks [C2], and Gaussian splatting [C3]. As these papers are cited in the DiVeQ paper, the reviewer encourages broadening the experiments, as the proposed methods are drop-in replacements for standard VQ layers, as mentioned in the paper.

[C1] Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, and Gang Hua. Designing a better asymmetric VQGAN for stable diffusion. arXiv preprint arXiv:2306.04632, 2023.

[C2] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos. In International Conference on Learning Representations (ICLR), 2025.

[C3] Haishan Wang, Mohammad Hassan Vali, and Arno Solin. Compressing 3D Gaussian splatting by noise-substituted vector quantization. In Scandinavian Conference on Image Analysis, pp. 338–352. Springer, 2025.

### Questions
Please refer to the Weaknesses section. In addition, these are minor questions and points.
* Why are the proposed variants of DiVeQ and SF-DiVeQ in the supplementary materials, while they show better results?
* The baselines could be outdated.
* t-SNE plots are known to present misleading information, especially when compared to other t-SNE plots.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Quantization is a mathematically non-differentiable process, which poses a major challenge for the end-to-end training of neural networks. To overcome this, this paper proposed a differentiable vector quantization (DiVeQ) using the reparameterization trick. The proposed DiVeQ ensures the hard assignment in the forward pass while allowing gradient flow to enable end-to-end training. Its variant, space-filling DiVeQ (SF-DiVeQ), alleviates codebook underutilization and achieves full utilization. The proposed methods excel the existing quantization method in both investigated image compression and generation tasks and can be easily used as a drop-in replacement for the standard vector quantization method.

### Strengths
1. The proposed DiVeQ addresses the gradient collapse problem by modeling quantization as the addition of a simulated quantization error to the input. A key improvement over previous noise-substitution vector quantization (NSVQ) is that the direction of the error vector is aligned with the selected codeword to guarantee a good reconstruction.
2. This paper also provides insights into existing quantization methods and establishes a basic understanding of the limitations. 
3. The proposed SF-DiVeQ addresses quantization error and codebook underutilization by extending quantization from discrete codewords in the codebook to continuous line segments connecting neighboring codewords. SF-DiVeQ achieves full utilization and improved representational efficiency. 
4. A codebook replacement algorithm was also given to enable stable codebook utilization and it can be applied to all investigated existing quantization methods.
5. The proposed methods do not require a specifically designed auxiliary loss or hyperparameter tuning compared to existing methods.
6. The proposed methods demonstrate consistent improvement over existing methods across multiple datasets on image compression and generation tasks. Comprehensive ablations on codebook size, batch size, learning rate, and the hyperparameter of the proposed methods are conducted.
7. The proposed methods can act as a drop-in replacement with minimal changes and the authors have investigated the performance within a residual vector quantization design.

### Weaknesses
1. This paper analyzed the limitations of NSVQ and its poor performance due to the high randomness. For the proposed DiVeQ, the paper highlighted the fundamental theory and the definition of the simulated error and how it improves quantization accuracy by adjusting the hyperparameter. But a more detailed theoretical analysis of convergence properties and gradient flow would be appreciated to provide more insights into the proposed method.
2. The proposed method approximates the quantization process by generating a directional noise vector and one component is sampled from a zero-mean Gaussian distribution. Adjusting the variance can affect the quantization accuracy (see Figure 3). Although the paper states that the variance parameter is not highly sensitive and a small value should be used, two different variance setups were used in the investigated compression and generation tasks. One can assume that some tuning or empirical selection is needed, especially in different domains or tasks. This is the same compared to existing methods, where hyperparameter tuning is needed.
3. The proposed SF-DiVeQ requires a specific and empirical initialization strategy. This introduces extra complexity and might increase implementation overhead if the codebook size is large. The results of SF-DiVeQ without this specific codebook initialization strategy are not discussed.
4. Similar to item #3, the recommended training setup for training SF-DiVeQ is to skip quantization for several initial epochs to stabilize the latent representation and then initialize the codebook vectors using the average of recent latent vectors. This process requires careful tracking of recent latent vectors and defining a specific number of training iterations to capture them, making it less of a drop-in replacement compared to existing methods that allow simple random initialization.
5. The proposed methods are only evaluated on image compression and generation tasks. Further investigation into speech/audio coding would be appreciated to confirm a broader applicability of the proposed methods.
6. The codebook replacement algorithm is used for all investigated methods. The proposed methods demonstrated good performance compared to existing exponential moving averages (EMA) and straight-through estimator (STE) methods by a smaller margin or performed comparably to EMA. This suggests that a well-tuned EMA can also deliver high-fidelity reconstruction. So, maybe, the codebook replacement algorithm plays a more important role in improving the overall performance while the contribution of the proposed DiVeQ is secondary.
7. Throughout the whole paper, the term 'VQ bitrate' is used to represent the codebook size. But bitrate refers to the number of bits used to represent the given bitstream per unit of time. For the image compression task, bitrate is measured in bits per pixel. In the paper, it is a misuse of the term 'bitrate' and it should be 'n-bit codebook' instead of 'VQ bitrate of n'.

### Questions
1. Could the authors provide a theoretical analysis regarding the convergence properties or gradient flow of the proposed methods? 

2. The proposed SF-DiVeQ requires a specific initialization strategy. It would provide more insights if the authors could show the necessity of the pre-training step and the impact of the pre-training step on stabilizing the latent representation and improving the performance. Could the authors quantify the performance degradation when SF-DiVeQ uses the randomly initialized codebook? Is there a big performance drop or is it only marginal? Could the author also provide the extra complexity required by the initialization strategy?

3. For DiVeQ, it is straightforward to the nearest codeword during training. But in SF-DiVeQ, a dithered codebook sampled from two subsequent codewords is used to quantize the input. How do the authors select these two subsequent codewords during training?

4. Two different variance values were used for the investigated tasks. Could the authors explain how and why they chose different values for each task? Does this mean that a variance tuning step is required when applying to other domains, such as speech coding?

5. Could the author give more discussion regarding the strength of the proposed method as a well-tuned EMA/STE with the codebook replacement algorithm can also achieve comparable performance? What do we benefit from the proposed method? Or does the codebook replacement algorithm contribute more to the performance?

6. As mentioned in the weakness part, please change the term used to represent the codebook size. Codebook size is not the same as the bitrate.

7. For the ablation on variance values of the proposed method, the colors in Figure 20 are not easy to distinguish, especially for variances with small values. Could the author adjust it using different line types?

8. A few typo errors are found. 
   For example, on page 9, 'but they are not homogeneously scattered In ST-GS,' probably should be 'but they are not homogeneously scattered. In ST-GS'. 
   On page 16, 'we mask half the the true token indices forcing the'. Should be 'we mask half of the true token indices forcing the'.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work aims to overcome two problems that previous vector quantization works have had.
First, vector quantization uses a codebook that maps latents to the closest codeword, and the mapping operation is not mathematically differentiable.
That is, it suffers from a gradient collapse problem, which means the gradients will not pass through after the codebook in the backward pass.
Second, while training, several codewords remain inactive, and thus, they are not trained well.

To resolve the above problems, the paper proposes the following.
For the gradient collapse problem, the paper uses the reparameterization trick to align the direction of quantization error with the nearest codeword.
By representing a quantized vector with a latent vector and a directional noise and showing that this quantized vector is differentiable to the latent vector and the codeword, the paper defines vector quantization as a differentiable problem.

For inactive codewords, codebook replacement is widely used to replace inactive codewords with new codewords that are near active codewords.
This paper points out that codebook replacement is a heuristic method, and suggests that mapping latent to an interpolation of subsequent codewords, instead of mapping to a single codebook.

The above proposals yield high-quality images while achieving good indicators in both image reconstruction and image generation.

### Strengths
- The gradient collapse problem of vector quantization is a problem that many researchers have been trying to solve by detour for a long time, and the paper proposes to solve this problem directly.
- The paper summarizes well about prior works that try to solve the same problem and their problems.
- The paper explains the problem situation and the proposed method well, through mathematical formulas.
- The proposed method can help generate high-quality images in both image reconstruction and image generation tasks, while achieving better indicators such as FID and LPIPS than other prior works.

### Weaknesses
- The paper proposes SF-DiVeQ to eliminate the need for codebook replacement. However, the reviewer wonders if SF-DiVeQ can really supplant codebook replacement, in case both subsequent codewords are inactive.
- Unlike other prior works, the FID score of the proposed method becomes worse when the VQ bitrate becomes larger. It may imply that the proposed methods are not general.

### Questions
The paper proposes SF-DiVeQ to eliminate the need for codebook replacement. Then, while training VQ with DiVeQ, are inactive codewords replaced by codebook replacement techniques?

### Soundness
3

### Presentation
3

### Contribution
3
