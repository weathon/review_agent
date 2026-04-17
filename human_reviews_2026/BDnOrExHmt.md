# PixNerd: Pixel Neural Field Diffusion

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 8

## Abstract
The current success of diffusion transformers are built on the compressed latent space shaped by the pre-trained variational autoencoder(VAE). However, this two-stage training paradigm inevitably introduces accumulated errors and decoding artifacts. To avoid these problems, researchers return to pixel space modeling but at the cost of complicated cascade pipelines and increased token complexity.
Motivated by the simple yet effective diffusion transformer architectures on the latent space, we propose to model pixel space diffusion using a large-patch diffusion transformer and employ neural fields to decode these large patches, leading to a single-stage streamlined end-to-end solution, which we coin as pixel neural field diffusion transformer (**PixNerd**). Thanks to the efficient neural field representation in PixNerd, we achieve **1.93 FID** on ImageNet 256x256 and nearly **8x lower latency** without any complex cascade pipeline or VAE. We also extend our PixNerd framework to text-to-image applications. Our PixNerd-XXL/16 achieves a competitive 0.73 overall score on the GenEval benchmark and 80.9 overall score on the DPG benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces PixNerd, a single-stage, end-to-end diffusion transformer that operates directly in pixel space. It aims to solve the trade-off between VAE-based latent models (which brings 2-stage training and is not elegant) and traditional pixel-space models (which are computationally expensive or rely on complex cascades).

The core technical contribution is a novel decoder. Instead of using a simple linear projection on large patches (which loses detail), the diffusion transformer's final features are used to predict the weights of a small MLP (a "neural field head") for each patch. This patch-specific MLP then decodes the final velocity for each pixel within that patch by taking the pixel's local coordinates as input. This approach allows the model to be computationally efficient (using large patches) while retaining high-frequency detail (via the neural field decoder).

### Strengths
This paper proposes an alternative approach besides latent diffusion models. It has several strengths: 

1. Elimination of VAE: The single-stage, end-to-end pipeline avoids the pre-trained VAE, removing its associated decoding artifacts and 2-stage training.

2. Efficient Pixel-Space Modeling: The proposed "neural field head" is an effective method for combining the computational efficiency of large-patch transformers with the high-fidelity representation of neural fields.

3. Strong Benchmark Performance: Achieves a 1.93 FID on ImageNet 256x256, which is state-of-the-art for pixel-space models and 3. competitive with top-tier latent-space models. It also demonstrates superior performance on text-to-image generation tasks.

4. Flexible Resolution: The coordinate-based nature of the decoder allows for training-free scaling to arbitrary resolutions at inference time, a significant practical benefit.

### Weaknesses
1. New Artifacts: While VAE artifacts are removed, the authors note in the appendix (A.4) that the model can introduce its own "blurry or unnatural artifacts."

2. Architectural Complexity: The design trades the complexity of a VAE for the complexity of the neural field head. Ablation studies (4.2) show performance is sensitive to this head's depth, channel count, and normalization strategy.

3. Potential Training Instability: The paper discusses the necessity of specific normalization strategies (Fig 5a, A.9) to "ensure training stability" and prevent "loss spikes" during long training runs, suggesting a sensitive training dynamic.

4. The main insight of this paper is to use NeRF MLP Head to replace the traditional MLP Head. Does it induce some specific designs other than the original NeRF method?

### Questions
1. Why does PixNerd have lower latency than the traditional latent diffusion models? Any insights here?

2. On text-to-image generation task, the parameters of PixNerd is 1.2B + 1.7B. What modules do these two parameters correspond to?

3. For NeRF MLP depth ablation in figure 5 (c), does the total number of layers is the same? For PixNerd-L/16 with 4 MLP layers, whether it use 22 transformer layers or it use 20 transformer layers? Further, given the same total layer number, which is the best ratio for transformer layers and NeRF MLP layers? 

4. How about the latency comparison of transformer layers and NeRF MLP layers?

5. Based on figure 5 (d) DCT-Basis is better than Sin/Cos coordinate-encoding. Any insights for this?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper argues that current latent diffusion models are limited the VAE which introduces encoding and decoding errors. This paper proposes to directly model pixel space with large patch sizes and learn neural fields to decode large patches. The new approach avoids the two-stage training process of the latent diffusion. The paper reports state-of-the-art results on ImageNet 256x256 with a FID score of 1.93 and a nearly 8x reduction in latency compared to existing methods. The authors also demonstrate the model's effectiveness in text-to-image generation, achieving competitive scores on the GenEval and DPG benchmarks.

### Strengths
- Novel method. The proposed PixNerd architecture, which combines a large-patch diffusion transformer with a neural field decoder, is a novel and interesting approach to pixel-space diffusion modeling. 

- Strong results. This paper reported a competitive FID score of 1.93 on ImageNet. Moreover, the proposed framework can be applied to text-to-image generation and achieves a competitive 0.73 overall score on the GenEval benchmark and 80.9 overall score on the DPG benchmark.

- Efficiency. This paper shows a significant latency reduction compared to both pixel diffusion and latent diffusion methods.

### Weaknesses
- While the results on ImageNet at 256x256 resolution are quite competitive (see Table 1), the results at 512 resolution are not so convincing (see Table 6). The authors are encouraged to explain the performance degradation. The authors are also encouraged to provide comparisons at a even higher resolution like 1024 or 768.

-Unclear latency comparison. The abstract mentions an 8x latency improvement but does not specify which models were used for comparison. According to Table 1, PixNerd did not achieve 8x latency improvement compared to latent diffusion methods.

### Questions
How does the performance of PixNerd, in terms of both image quality and latency, scale as the image resolution increases (e.g., 512x512, 1024x1024)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces PixNerd, a novel end-to-end pixel-space diffusion transformer that replaces the traditional linear projection head with a neural field decoder. Its main goal is to bring the efficiency and visual fidelity of latent diffusion transformers into the raw pixel domain, without relying on VAEs or multi-scale cascades that introduce complexity and latency.

### Strengths
S1. Originality and Significance

- I highly commend the paper’s central motivation and direction. While most recent efforts focus on reducing computational cost by operating entirely in latent space, this work takes the opposite yet equally important perspective to explore how to lower cost directly in pixel space. This inversion of the conventional design philosophy is both insightful and original, addressing a long-standing challenge in diffusion modeling.

S2. Experimental Quality and Clarity

- I also deeply appreciate the thorough and well-executed ablation studies. The paper demonstrates an exceptional level of experimental rigor, allowing readers to form a fair and comprehensive understanding of the proposed method’s behavior. In my view, this kind of meticulous empirical investigation exemplifies what a well-written paper should strive for.

### Weaknesses
W1. Justification for architectural necessity.

- Given that recent advances in diffusion distillation (e.g., one-step or few-step distilled diffusion models) can achieve nearly identical performance to full diffusion models with drastically reduced sampling steps, it is unclear why PixNerd needs to exist as a separate model. Could the authors clarify whether PixNerd itself can serve as a teacher model for distillation? If not, then PixNerd would likely exhibit much higher latency compared to diffusion + distillation pipelines. Moreover, the claimed 8× speedup still appears slower than VAE + diffusion pipelines, which challenges the practical advantage of the method. If the authors could convincingly argue in this point, then I will flip my assessment.

W2. Missing baselines and quantitative comparison.

- In Table 1, it would significantly strengthen the evaluation to include representative generative baselines such as StyleGAN [1], CDM [2], Simple Diffusion [3], VDM++ [4], and PaGoDA [5].
In particular, the paper should provide quantitative comparisons with one-step models (either GAN or diffusion + distillation) and explicitly report latency differences to position PixNerd more clearly within the current landscape of efficient generative models.

[1] Sauer, Axel, Katja Schwarz, and Andreas Geiger. "Stylegan-xl: Scaling stylegan to large diverse datasets." ACM SIGGRAPH 2022 conference proceedings. 2022.
[2] Ho, Jonathan, et al. "Cascaded diffusion models for high fidelity image generation." Journal of Machine Learning Research 23.47 (2022): 1-33.
[3] Hoogeboom, Emiel, Jonathan Heek, and Tim Salimans. "simple diffusion: End-to-end diffusion for high resolution images." International Conference on Machine Learning. PMLR, 2023.
[4] Kingma, Diederik, and Ruiqi Gao. "Understanding diffusion objectives as the elbo with simple data augmentation." Advances in Neural Information Processing Systems 36 (2023): 65484-65516.
[5] Kim, Dongjun, et al. "Pagoda: Progressive growing of a one-step generator from a low-resolution diffusion teacher." Advances in Neural Information Processing Systems 37 (2024): 19167-19208.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces PixNerd, a end-to-end pixel-space diffusion model that matches the performance of SOTA latent diffusion model without relying on pretrained VAE. PixNerd achieves this by using implicit neural field to replace the traditional decoding head of DiT. PixNerd achieves competitive performance on ImageNet and text-to-image generation tasks.

### Strengths
- The paper is well-written with clear structure. 
- The paper present comprehensive comparisons against current SOTA models both qualitatively and quantitatively. PixNerd matches or exceeds the performance of comparable methods on ImageNet and text-to-image tasks
- The ablation studies are conducted systematically to evaluate each design choice.

### Weaknesses
- The training memory usage is almost doubled compared to that of latent diffusion counter part. 
- PixNerd's performance at higher resolutions (512×512) does not scale as strongly as at 256×256. For example, PixNerd is better than SiT-XL on ImageNet256 but falls behind on ImageNet512. Does this imply that the gains from the neural field head diminish at higher resolutions?
- Minors:
	- The citation of Rectified flow seems missing. 
	- Table 4 should specify that the comparison is reported on ImageNet256 for clarity.

### Questions
- What is the main source of additional training memory and how can this be optimized? 
- Why is DCT-Basis encoding better? What about the other popular alternatives like RoPE?
- I notice that PixNerd (512x512) is finetuned from PixNerd(256x256) and its performance on ImageNet512 is not as impressive as PixNerd(256x256). What's the reason for not training PixNerd (512x512) from scratch? 
- Is PixNerd compatible with representation alignment techniques like REPA? Since it operates directly in pixel space, would this alignment even be more effective?

### Soundness
3

### Presentation
3

### Contribution
3
