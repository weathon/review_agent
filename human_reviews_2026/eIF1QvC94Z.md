# Turbo-DDCM: Fast and Flexible Zero-Shot Diffusion-Based Image Compression

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 2, 8, 4, 4

## Abstract
While zero-shot diffusion-based compression methods have seen significant progress in recent years, they remain notoriously slow and computationally demanding. This paper presents an efficient zero-shot diffusion-based compression method that runs substantially faster than existing methods, while maintaining performance that is on par with the state-of-the-art techniques. Our method builds upon the recently proposed Denoising Diffusion Codebook Models (DDCMs) compression scheme. Specifically, DDCM compresses an image by sequentially choosing the diffusion noise vectors from reproducible random codebooks, guiding the denoiser’s output to reconstruct the target image. We modify this framework with *Turbo-DDCM*, which efficiently combines a large number of noise vectors at each denoising step, thereby significantly reducing the number of required denoising operations. This modification is also coupled with an improved encoding protocol. Furthermore, we introduce two flexible variants of Turbo-DDCM, a priority-aware variant that prioritizes user-specified regions and a distortion-controlled variant that compresses an image based on a target PSNR rather than a target BPP. Comprehensive experiments position Turbo-DDCM as a compelling, practical, and flexible image compression scheme.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the issues of slow speed and high computational resource requirements of DiffC by proposing the Turbo-DDCM method. As a continuation of DiffC methods based on the RCC theory, TurboDDCM well inherits advantages such as zero-shot capability and achieves improvements in speed and performance.

### Strengths
1. The motivation is reasonably elaborated and well-supported in the subsequent methods and experiments. The time issue is a key concern in the field of image diffusion research based on diffusion architectures, and the authors have properly proposed a new noise reconstruction method to tackle this.
2. The structure of the paper is clear and well-written.

### Weaknesses
1. The authors only conducted time comparisons within the RCC field. It would be valuable to know the speed comparison with diffusion compression methods based on condition introduction and fine-tuning. Specifically, whether RCC-based image compression has time advantages over fine-tuning-diffusion-based image compression, especially given the emergence of many one-step diffusion image compression methods (e.g., StableCodec).
2. Following the first point, for fewer time steps (e.g., 5 steps or even a single step), does the performance of the proposed method degrade significantly?
3. The performance improvement of the final results seems not particularly significant, and the comparisons are insufficient. The baselines selected by the authors appear to be consistent with those of DiffC, but DiffC is a work from ICLR25. It is suggested that the authors include more comparisons with recent works from the past year, such as DiffEIC and StableCodec.
---
ref: 

[1] Li Z, Zhou Y, Wei H, et al. Towards extreme image compression with latent feature guidance and diffusion prior[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[2] Zhang T, Luo X, Li L, et al. StableCodec: Taming One-Step Diffusion for Extreme Image Compression[J]. arXiv preprint arXiv:2506.21977, 2025. (ICCV25 accepted)

### Questions
See weaknesses.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes an efficient zero-shot diffusion-based image compression method, which is based on the Denoising Diffusion Codebook Models (DDCMs) compression scheme. The paper modifies DDCM with Turbo-DDCM and introduces two flexible variants of Turbo-DDCM.

### Strengths
1. The paper is easy to read.
2. The paper proposes a method which has a faster speed than existing zero-shot diffusion-based image compression methods.

### Weaknesses
1. The novelty and contribution are weak. The author just modified the DDCM, but there is no ablation study to analyze the proposed components.
2. The presentation is incomplete. There is a lack of quantitative comparison and ablation study. And there are no tables in the whole paper.

### Questions
1. What are the backbones of all zero-shot diffusion-based methods? For a fair comparison of speed, the proposed method should have the same backbone as these methods.
2. Does the author use Turbo-Lora to accelerate Stable Diffusion 2.1?
3. What is the speed of non-diffusion-based methods? The author should also compare with them.
4. How to process higher resolutions?

Please reply to the Weaknesses and Questions. Based on the author's response, I will adjust my rating.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Turbo-DDCM proposes a new approach to zero-shot diffusion-based compression. In the original Denoising Diffusion Codebook Models (DDCM) approach, at each denoising timestep, the compression algorithm selects the next noise vector to add to the image from a codebook of K noise vectors, y selecting the one which moves the noisy image closest to the target image. In Turbo-DDCM, the authors instead encode a noise vector by selecting a subset of M of those K random vectors, and taking a linear combination of them, with coefficients either -1 or 1. This effectively allows the authors to select from a combinatorially larger set of noise vectors, getting closer to the target image with each step. With this improved guidance, they can steer the diffusion model to produce the target image in many fewer steps. They empirically show that this method allows for zero-shot image compression using diffusion models which is multiple times faster than prior approaches.

### Strengths
This is a clever idea for improving the computational efficiency of noise-vector selection in zero-shot DDPM-based compression. It clearly works quite well.

### Weaknesses
I consider all of these zero-shot methods (DiffC, DDCM, Turbo-DDCM) to be modifications of Theis et al's original DiffC proposal. They all, to various degrees, trade off theoretical elegance/rigor for real-world usability. In the original 2022 algorithm, there's a precisely defined relationship between the diffusion model's log probability of an image and the number of bits needed to compress that image to a certain noise level. Turbo-DDCM achieves state-of-the-art real-world performance, but trades away these theoretical guarantees.

### Questions
The other available zero-shot diffusion compression methods (DDCM, and DiffC) have hyperparameters for speeding up their runtime by trading off against their rate/distortion performance. Primarily this means the number of denoising steps performed. It would be very interesting to see how these methods fare as you decrease the number of denoising steps to match Turbo-DDCM.

It's not obvious to me that the z_t* vectors selected by turbo-DDCM should yield x_{t-1} vectors which are in the same distribution that the diffusion model was trained on? Like the diffusion model was trained to expect specific linear combinations of real images + gaussian noise, at different ratios, for each time step. But when you generate z_t* according to this fancy process, does it still have the same statistics that random noise does, from the perspective of the diffusion model? Like you are selecting these vectors to be correlated with the target image, so I would naively expect them to have different statistics from random noise. But I'm not confident in this assessment.

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
4

### Summary
This paper proposes a fast and flexible zero-shot diffusion-based image compression method (i.e., Turbo-DDCM), which reduces the number of required denoising operations and maintains the performance with an improved encoding protocol. Moreover, Turbo-DDCM presents a priority-aware variant that prioritizes regions of interest and a distortion-controlled variant that compresses an image based on a target PSNR. Experiments are performed on Kodak and DIV2K datasets to investigate the effectiveness of the proposed method.

### Strengths
Turbo-DDCM has a competitive performance with recent methods in terms of the rate-distortion-perception tradeoff, and it achieves up to an order of magnitude speedup over existing methods.

### Weaknesses
1.	The experimental section lacks comprehensive quantitative results, such as BD-rate or BD-PSNR.
2.	The workflow description of the proposed method is not concise enough.
3.	Although the proposed method is faster than the comparison algorithm, it does not offer a performance advantage.
4.	The paper does not analyze the advantages of diffusion-based image compression methods compared to other types of image compression models, such as GAN, CNN, and RNN.
5.	No results tables are provided for better indicating its superiorities over other methods.

### Questions
1.	What is the meaning of the symbol C in Eq. (8)?
2.	The author should provide a detailed explanation of the encoding and decoding process in Figure 2.
3.	What is the meaning of log2(K M) in Eq. (14)?
4.	It is difficult to distinguish the differences among the decoded images in the first row of Figure 1.
5.	In Figure 4, when comparing with other methods, the authors should also provide the results on the DIV2K dataset.
6.	The proposed method should be compared with GAN-based algorithms.
7.	When comparing computational complexity, the authors should also provide GPU memory usage.
8.	The author should analyze the impact of the values of hyperparameters T, K, and C on performance.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a zero-shot algorithm based on a pre-trained diffusion model to achieve image compression at extremely low bit rates. In essence, this method represents an efficient enhancement of the DDCM approach, transforming the original MP method into MULTI-ATOM and proposing a new bit protocol to further reduce the redundancy in the bitstream generated by the MULTI-ATOM method. The paper presents a plethora of experiments and evidence to validate the effectiveness and reliability of the algorithm. The experiments demonstrate that this approach achieves excellent performance metrics in zero-shot diffusion-based schemes.

### Strengths
1. The paper is well-written, with clear expression and concise yet precise explanations of the motivation and methodology.
2. The theoretical explanations are solid, providing a high level of theoretical underpinning for the approach.
3. The experiments are comprehensive, comparing all zero-shot diffusion-based image compression baselines and achieving satisfactory performance even with a several-fold acceleration

### Weaknesses
1. Some modules lack sufficient explanations, especially those borrowed from other articles. Providing necessary explanations can help readers better understand the operational mechanisms of the entire algorithm.
2. The lack of striking innovativeness in the algorithm may be noted, as the entire solution builds upon the pipeline of DDCM and does not surpass DiffC in terms of performance.
3. I believe confining the baselines to the zero-shot domain is inappropriate. While zero-shot algorithms show promise and are worth researching in diffusion-based image compression schemes, all zero-shot algorithms fundamentally sacrifice rapid inference to reduce or eliminate training costs, whereas compression tasks are sensitive to inference latency. Recently, some solutions based on few-step pre-trained diffusion models [1,2] have achieved outstanding performance with minimal latency. Overall, I suggest that the method should be compared against some non-zero-shot diffusion baselines or explore the potential application of this algorithm in fine-tuning diffusion priors, which could enhance the practical contributions and persuasiveness of the paper.

[1] [TCSVT] RDEIC: Accelerating Diffusion-Based Extreme Image Compression with Relay Residual Diffusion

[2] [ICCV 2025] StableCodec: Taming One-Step Diffusion for Extreme Image Compression

### Questions
1. The article's overview of the overall encoding and decoding process is somewhat vague. It would be beneficial to include a pseudocode  to provide readers with a clearer understanding of the algorithm's encoding and decoding processes.
2. The use of ROI in the article for flexible bitrate allocation is commendable. However, the paper lacks details on the application of this technology. For instance, is ROI specific to this algorithm only? Can it be integrated with DDCM or other zero-shot schemes? What is the specific process of implementing ROI? Does ROI introduce additional inference latency? These aspects should be further elaborated.
3. The article showcases the Round Trip Time. What does this specifically refer to? In a compression algorithm, it's essential to separately compare encoding and decoding times.
4. The article employs a 512 central crop for the Kodak dataset, which is relatively uncommon in the compression field where full-size Kodak testing is more prevalent. Is there a specific reason for this approach? Is it related to high GPU memory usage by the algorithm?
5. Testing FID on the Kodak dataset may not be appropriate as Kodak comprises only 24 images, and even when divided into 64-pixel patches, reliable FID metrics may not be guaranteed.
6. CLIC20 is a commonly used dataset in image compression. It would enhance the algorithm's reliability to include CLIC20 in the main experiments.

### Soundness
2

### Presentation
3

### Contribution
3
