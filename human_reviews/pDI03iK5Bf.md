# $\Delta$-DiT: Accelerating Diffusion Transformers without training via Denoising Property Alignment

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
Diffusion models are now commonly used for producing high-quality and diverse images, but the iterative denoising process is time-intensive, limiting their usage in real-time applications. As a result, various acceleration techniques have been developed, though these primarily target UNet-based architectures and are not directly applicable to Transformer-based diffusion models (DiT). To address the specific challenges of the DiT architecture, we first analyze the relationship between the depth of DiT blocks and the quality of image generation. While skipping blocks can lead to large degradations in generation quality, we propose the $\Delta$-Cache method, which captures and stores the incremental changes of different blocks, thereby mitigating the performance gap and maintaining closer alignment with the original results. Our analysis indicates that the shallow DiT blocks primarily define the global structure of images such as compositions, and outlines, while the deep blocks refine details. Based on this, we introduce a denoising property alignment method that selectively bypasses computations of different blocks at various timesteps while preserving performance. Comprehensive experiments on PIXART-$\alpha$ and DiT-XL demonstrate that $\Delta$-DiT achieves a $1.6\times$ speedup in 20-step generation and enhances performance in most cases. In the 4-step consistent model generation scenario, and with a more demanding $1.12\times$ acceleration, our approach significantly outperforms existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a method to accelerate Transformer-based diffusion models  for image generation. It uses the ∆-Cache method to store incremental changes in diffusion blocks, speeding up the generation process.

### Strengths
The paper addresses the timely issue of accelerating diffusion processing and introduces the ∆-Cache method, which shows potential in enhancing the diffusion model's speed. The method is evaluated on two datasets.

### Weaknesses
1- The presentation of the paper is not clear. It is hard to understand the proposed model. For example, ∆-Cache and the DiT blocks are introduced with insufficient context. Even in the abstract, jumps between ideas such as the role of shallow and deep blocks without clear transitions, complicating the flow. In the proposed method section, there are repetitions on the performance of the blocks.  These parts can be compact to improve the presentation. Several parts of the paper are difficult to undrestand, for example, page 2, line 86 -94, page 5 line 237-240, 

2- As I compared the results in Table 1 and Table 3, the performance improvement (speedup and latency) is marginal (Ours (b = 12), Faster Diffusion, PIXART, and TGATE have very close speedup). Yes, it is right that the proposed model has slightly better speed and performance, but it is not feasible. The authors are encouraged to discuss the practical implications of these improvements to better contextualise their importance in real-world applications.

3-  In page 6, line 302, what do you mean by " imaginary parts of a complex number"?

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies cache-based acceleration for Diffusion Transformers (DiTs) and proposes a training-free method, delta-cache. To compensate for the lack of long shortcuts in DiTs, delta-cache stores the incremental changes in feature maps. Furthermore, delta-cache aligns with the denoising property by caching deep blocks during the early stages and shallower blocks in the later stages. Experiments show that delta-cache achieves 1.6x speedup in 20-step generation while improving image quality.

### Strengths
1. The proposed acceleration method is training-free, making it more practical and easy to apply.

2. The idea of reusing the change instead of the block feature is interesting.

3. Figures 2 and 3 analyze the property of different generation stages, which is insightful.

### Weaknesses
1. The choice of bound b and the number of cached blocks Nc is empirical. Figures 6 and 7 show that the results are sensitive to the values of b and Nc. Is there a better optimization method for delta-cache?

2. The method achieves 1.6x acceleration in 20-step generation, which seems insignificant compared to other compression methods such as post-training quantization and time-step distillation. Is it possible to achieve more than 2x acceleration without sacrificing performance? Could the author clarify the advantage of cache-based methods on DiTs?

### Questions
1. The proposed delta-cache is training-free and can be used without training data. The main experimental results in the paper show that caching some feature changes can even improve the generation performance, which seems counter-intuitive. Could the author provide a deeper analysis of this?

Overall, this is an interesting work. The reviewer is willing to adjust the score.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents a framework called Delta-DiT, designed to accelerate the Diffusion Model architecture of DiT. Delta-DiT enables faster denoising operations by caching the differences between specific blocks within the DiT model, eliminating the need for actual computation. The proposed method outperforms existing solvers, such as DPM-Solver, as well as other diffusion acceleration methods like FasterDiffusion.

### Strengths
- This paper is well-written and works effectively with a relatively simple idea. 
- The proposed Delta-DiT can accelerate the existing DiT model by more than 1.6 times without additional training. 
- It is also beneficial that it works in few-step denoising scenarios, such as with LCM.

### Weaknesses
- There is a lack of experimental results demonstrating that Delta-Dit is actually faster than existing methods like DPM-Solver and Faster Diffusion. To compare performance accurately, it should be evaluated on a Pareto curve with latency and FID as axes. The paper only provides a single point of comparison in a table, which may be cherry-picked. 
- There are no experimental results using Unet. While the paper primarily targets DiT, Unet is still widely used in diffusion architecture, making a significant limitation to this work.  
- Error in Related Works: In the case of So et al., they are also targeting DiT. 
- The acceleration results in Table 3 are too small.

### Questions
- Despite applying caching, there seem to be many cases where the FID score actually decreases. What might be the reason for this result? Could this outcome imply an experimental error or a lack of accuracy in the FID metric? Are there experimental results using more recent metrics, such as CMMD [1]? 
- At which exact timestep do you perform computations for caching? Are these computations performed at uniform intervals? 
- What is the unit of the DiT Block when calculating Delta? 
- What is the memory overhead required to run Delta-Diffusion? 

[1] Jayasumana, Sadeep, et al. "Rethinking FID: Towards a Better Evaluation Metric for Image Generation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

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
2

### Summary
This paper proposes a method specifically tailored for transformer-based diffusion models to speed up the inference process. The authors first propose to utilize $\Delta$-Cache to skip several transformer blocks to speed up the inference process. Then the authors discovered the alignment between the transformer blocks and the denoising process e.g., the shallow blocks and the early stage of the diffusion denoising process mostly account for the outline generation. By utilizing this discovery, the authors then propose $\Delta$-DiT to further refine the caching method to improve the generation quality.

### Strengths
1. The paper proposes a simple method for speeding up the generation process of transformer-based diffusion models, which is training-free and thus can be easily applied.
2. The results look good from the numbers on the COCO dataset.

### Weaknesses
1. It seems to me that the alignment ($\Delta$-DiT) does not standout against the baseline method $\Delta$-Cache from Table 1 which makes me a bit concerned about the effectiveness of the alignment. I'm curious have the authors also tried some other ways to apply the $\Delta$-Cache in different blocks/steps.
2. The authors conduct experiments on PixelArt-$\alpha$ and DiT-XL. Could the authors also provide results on more recent models such as SD3 which is also transformer-based?

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3
