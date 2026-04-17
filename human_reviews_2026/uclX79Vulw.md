# FuseBatch: Unlocking the Potential of Diffusion Models in Throughput Perspective

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Diffusion models achieve state-of-the-art image quality but suffer from slow, iterative denoising. Existing acceleration methods focus on reducing the number of iterations, but these approaches are nearing practical limits. To address this, we take a different perspective by improving efficiency through generating multiple images within a single forward pass. We propose **FuseBatch** which fuses multiple inputs into a shared latent, applies the denoiser once, and unfuses the results to recover all outputs. To extend across domains, we introduce **FB-UNet** for pixel-space models and **FB-AE** for latent diffusion models.We further propose **Timestep-Fusion Scheduling (TFS)**, an inference-only strategy that balances throughput and quality, enabling FuseBatch to surpass the baseline at comparable throughput settings. Across DDPM and step-reduction methods (*e.g* DDIM, Flow Matching), we achieve near-multiplicative throughput gains with modest quality trade-offs, demonstrating its compatibility with existing acceleration techniques. Moreover, it scales effectively to high-resolution LDMs where larger fusion factors become attainable, providing a practical and orthogonal path to faster diffusion sampling.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The slow sampling speed of diffusion models is due to progressive denoising, requiring each image to traverse an entire time step. Mainstream acceleration methods focus on reducing the number of steps, but further reduction after a dozen or so steps may significantly degrade image quality, reaching a plateau in marginal gains. This paper proposes a different approach called  FuseBatch. Instead of reducing the time per image, it aims to generate multiple images in a single forward pass, maximizing throughput.

### Strengths
1. The proposed fusebatch idea is interesting. Previous diffusion methods have focused on reducing the sampling step or the time required for each step. This paper proposes a new approach: accelerating the throughput of the entire batch by fusing and then decomposing multiple images during sampling.
2. The proposed method is compatible with DDIM and Flow Matching.
3. The paper is written clearly.

### Weaknesses
1. The paper lacks several essential experiments. The authors only conducted experiments on small-scale class-conditional image generation datasets such as CIFAR-10 and CelebA. As a training-free acceleration method, it is necessary to demonstrate its practical acceleration performance on mainstream text-to-image generation models (such as SDXL).

2. The experiments in the paper lack necessary baseline comparisons. I doubt whether the proposed method can achieve better results than simple training-free approaches such as DeepCache.

3. Can the proposed method be compatible with DPM-Solver?

4. Both the method description and experimental sections of the paper are based on U-Net-based diffusion models. Considering that the current state-of-the-art diffusion models mostly adopt the DiT architecture, it is necessary to demonstrate whether the proposed method can also accelerate DiT-based diffusion models.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes fusing images in a batch to allow a diffusion model to denoise for multiple samples simultaneously. This works very poorly in practice, but performance can be dramatically improved simply by varying the number of images that are fused depending on the progress in the denoising scheduler (less fusion for more important steps, more fusion for less important steps). This method is applied for multiple schedulers for CIFAR and CelebA, primarily in unconditional generation, using fusion-specialized UNet and autoencoder architectures.

### Strengths
S1. This is a novel perspective to approach something that seems intuitively desirable- reduce the tokens/latents processed per image. The idea of fusing and unfusing images is straightforward and elegant. 

S2. The idea of TFS is also intuitive and elegant, simply changing the amount of fusion for different parts of the schedule.

S3. The paper clearly communicates the ideas.

### Weaknesses
W1. The idea is essentially a way to reduce the number of tokens/latents to process per image. While this is done by merging multiple images to a single image, it probably could be compared to simply changing the size of a patch (for DiT) or the downsampling ratio (for UNet). This seems to be the most important ablation missing, especially since we already know that we can simply increase patch size for DiT with some degradation in the quality (for example, from 19.47 to 43.01 FID, difference between DiT-XL/2 and DiT-XL/4 in Table 4 of DiT). This is reminiscent of this paper's Table 2 where for conditional generation, a fuse-batch of 2 (which speeds up computation by 2x, compared to 4x for the DiT) also doubles FID, from 4.07 to 7.56. The paper needs more convincing ablations that this method still works in the class-conditional setting, and either outperforms or is complimentary to simply changing patch size/downsample ratio.

W2. Building off of W1, the class-conditional results seem very unconvincing. My concern would be that the more you attempt to constrain the outputs, the more of a problem it would be that you are combining the denoising. If the classes or text prompts are very dissimilar, it seems this could be especially problematic. The possibility of class/prompt confusion needs to be thoroughly analyzed, and a single table with no TFS result is inadequate.

W3. The experiments are conducted on small-scale datasets in mostly unconditional settings, making it unclear if the method would work in real-world settings with higher resolution images and more conditions/constraints. This weakness is not disqualifying, but it is difficult to evaluate without even ImageNet results, let alone text to image. However, it seems natural that this method will struggle as conditions and resolution increase, so proof is needed that it does not.

W4. It is unclear if this can work in tandem with caching and quantization approaches that can achieve similar speedups, often training free, with 0 loss in performance (as measured by metrics).

### Questions
This is a very interesting idea, and pursues a very compelling avenue for accelerating the diffusion. However, I'm not convinced that it's better than alternatives.

1. How does this compare various approaches for pooling more aggressively on the spatial dimension? 

2. How well does TFS do for class-conditional generation? What sorts of artifacts appear here, considering that K=2 has a much bigger gap than in the unconditional setting?

3. Can this work orthogonally with caching, where SOTA training-free caching methods like TaylorSeer can already support 4x+ acceleration without sacrificing quality?

### Soundness
1

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
4

### Summary
The paper proposes FuseBatch, a framework to accelerate diffusion inference by generating multiple images in one forward pass.
Instead of reducing sampling steps, FuseBatch fuses K inputs into one shared latent, processes it once through the denoiser, and unfuses it back into K outputs. Two variants are introduced: FB-UNet for pixel-space diffusion models and FB-AE for latent diffusion models.
A Timestep-Fusion Scheduling (TFS) policy further adjusts the fusion factor K across timesteps to balance throughput and fidelity.
Experiments on DDPM, DDIM, Flow Matching, and LDMs show roughly 2–4× throughput improvement with moderate FID degradation.

### Strengths
1. Provides a fresh, orthogonal acceleration direction by increasing per-pass yield instead of reducing denoising steps.

2. Fusion/unfusion modules are clearly described and easy to integrate; minimal parameter overhead (<0.3%).

3. Comprehensive experiments on multiple samplers and datasets.

4. Compatible with existing step-reduction and latent diffusion frameworks.

5. Timestep-Fusion Scheduling (TFS) helps recover quality at high fusion factors.

### Weaknesses
1. Lack of intuition on why fusion works: In general, the paper focuses on how to implement the fusion/unfusion modules but does not clearly explain why such fusion should work intuitively. The proposed “sum + small conv + index encoding” seems highly lossy, and the paper provides no solid intuition or theoretical reasoning to justify why multiple latent signals can be linearly combined and later disentangled effectively.

2. Quality drops quickly as K increases (e.g., FID from 3.4 → 18.2 at K=4 in Table 1), and TFS mainly mitigates this empirically.

3. Limited novelty — the main idea resembles multi-input multiplexing or mixup-style encoding.

4. Evaluation remains small-scale (CIFAR-10, CelebA); no tests on modern large text-to-image models (e.g., SDXL, Flux).

5. Missing runtime and memory benchmarks for fusion/unfusion overhead.

6. The text-conditioning mechanism is briefly mentioned but not clearly described.

### Questions
1. Why should additive or convolution-based fusion preserve separable semantic information?

2. How are text conditions handled during fusion—are prompts fused or injected independently?

3. What is the runtime and memory overhead of fusion/unfusion modules relative to one denoiser pass?

4. Can the method extend to transformer-based diffusion models like Flux or SD3?

5. How sensitive is performance to the index encoding scheme or kernel size?

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
This paper introduces FuseBatch, a new approach to improving the throughput of diffusion models by enabling the simultaneous generation of multiple images in a single forward pass. Instead of focusing on reducing the number of denoising iterations, FuseBatch works by fusing multiple input images into a shared latent space, processing them together, and then unfusing them to recover individual outputs. The framework is adaptable to both pixel-space and latent-space diffusion models, with specialized variants like FB-UNet and FB-AE. Additionally, the paper proposes Timestep-Fusion Scheduling (TFS), a strategy that dynamically adjusts the number of fused samples at different timesteps to balance computational efficiency with output quality. The method is demonstrated to provide significant throughput improvements across several popular diffusion models while maintaining comparable sample fidelity. Experiments show that FuseBatch can scale effectively to high-resolution settings, offering a scalable solution for faster image synthesis without drastic sacrifices in quality.

### Strengths
1. This paper introduces FuseBatch, a novel framework that significantly boosts the throughput of diffusion models by generating multiple images within a single forward pass, which is a unique approach compared to traditional methods that focus solely on reducing inference time per sample.
2. The paper proposes FB-UNet and FB-AE, two tailored architectural solutions for pixel-space and latent-space models, respectively. These designs ensure that FuseBatch can be applied across different types of diffusion models, providing a scalable and versatile solution for faster image generation.
3. The introduction of Timestep-Fusion Scheduling (TFS) is an innovative method to dynamically adjust the fusion factor across timesteps, which optimizes the trade-off between throughput and image quality, addressing the inherent challenges in balancing efficiency and fidelity during diffusion processes.
4. Extensive experiments on popular diffusion models (DDPM, DDIM, and Flow Matching) and datasets (CIFAR-10, CelebA) demonstrate that FuseBatch provides substantial throughput gains with minimal quality loss, showcasing its effectiveness in real-world scenarios.

### Weaknesses
1. Quality Degradation at Larger Fusion Factors: While FuseBatch increases throughput by fusing multiple images in a single forward pass, this results in quality degradation as the fusion factor K increases. For example, when K=4, there is a notable increase in FID, suggesting a deterioration in image quality. The paper mentions TFS as a strategy to balance throughput and quality, but it is unclear how TFS performs when K is further increased beyond K=4, particularly in models requiring high fidelity. Hence, I think it would be useful to explore and quantify the threshold at which FuseBatch and TFS become ineffective or result in unacceptable quality trade-offs. Especially when dealing with high-resolution models or tasks that demand intricate details.
2. Potential for Overfitting at High Fusion Factors: FuseBatch works by increasing the number of images generated per forward pass, which boosts throughput. However, as the fusion factor increases, the model might overfit or exhibit artifact generation due to the merging of too many samples. In cases where the fusion factor is large, the model might lose important details specific to individual images. I was wondering if the author could investigate this potential risk and provide further analysis on the behavior of FuseBatch at higher fusion factors, particularly in scenarios where intricate image details are crucial. 
3. Limited Discussion on Model Complexity and Training Overhead: While FuseBatch introduces lightweight fusion and unfusion modules to increase throughput, the paper doesn’t provide a detailed analysis of their impact on model complexity or training overhead. Specifically, there is no mention of how these modules affect training time or convergence when applied to larger models or datasets. If the additional complexity increases training time or model size, this could limit FuseBatch’s scalability. Further exploration of the cost-benefit trade-off between throughput gains and model overhead would be beneficial.
4. Insufficient Analysis of Negative Impact on Output Diversity: FuseBatch fuses multiple images into a shared latent space to improve throughput. However, when generating multiple samples simultaneously, there is a risk of reducing the diversity of the generated outputs. This could be particularly important in tasks where sample diversity is key, such as in creative image generation, design, or anomaly detection. The paper evaluates FuseBatch in terms of quality and throughput, but it does not consider how increased throughput might impact the diversity of the generated images, especially when generating a larger number of images per forward pass. I think it would be useful to explore how FuseBatch affects output diversity to ensure its applicability in tasks where maintaining variety in generated samples is essential.
5. Unclear Impact on Model Robustness in Low-Data Regimes: FuseBatch improves throughput by processing multiple images at once, which can be beneficial in high-data regimes. However, in low-data settings, such as few-shot learning or anomaly detection with limited samples, there may be concerns about the framework’s ability to generalize or maintain robustness with fewer training samples. The paper evaluates FuseBatch using standard datasets with abundant data, but does not explore how the model would perform in data-scarce environments. It would be useful to investigate FuseBatch's performance in low-data scenarios to assess its ability to generalize or avoid overfitting when data is limited.

### Questions
Please refer to the questions and suggestions in the “Weaknesses” part.

### Soundness
3

### Presentation
3

### Contribution
3
