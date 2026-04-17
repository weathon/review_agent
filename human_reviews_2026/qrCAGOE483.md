# LucidFlux: Caption-Free Universal Image Restoration via a Large-Scale Diffusion Transformer

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 4

## Abstract
Universal image restoration (UIR) aims to recover images degraded by unknown mixtures while preserving semantics—conditions under which discriminative restorers and UNet-based diffusion priors often oversmooth, hallucinate, or drift. We present LucidFlux, a caption-free UIR framework that adapts a large diffusion transformer (Flux.1) to restoration with minimal parameter overhead. LucidFlux introduces a lightweight \emph{dual-branch conditioner} that injects signals from the degraded input and a lightly restored proxy to respectively anchor geometry and suppress artifacts. A timestep- and layer-adaptive modulation schedule routes these cues across the backbone’s hierarchy, yielding coarse-to-fine, context-aware updates that protect global structure while recovering texture. To avoid the latency and instability of text prompts or VLM captions, we enforce \emph{caption-free semantic alignment} via SigLIP features extracted from the proxy. A scalable curation pipeline further filters large-scale data for structure-rich supervision. 

Across synthetic and in-the-wild benchmarks, LucidFlux consistently surpasses strong open-source and commercial baselines across seven metrics, with clear visual gains in realism, detail, and artifact suppression. Ablations confirm that, for large DiTs, when, where, and what to condition—rather than scaling parameters or relying on text prompts—is the key lever for robust, prompt-free restoration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes LucidFlux, an image restoration framework that adapts a large diffusion transformer (Flux.1) for restoring images degraded by unknown mixtures, without relying on captions. The key contributions include a lightweight dual-branch conditioner that anchors geometry and suppresses artifacts by injecting signals from both the degraded input and a lightly restored proxy. Additionally, a timestep- and layer-adaptive modulation schedule enables context-aware, coarse-to-fine restoration. To ensure semantic alignment without text prompts, LucidFlux leverages SigLIP features from the proxy. The framework also introduces a scalable data curation pipeline for structure-rich supervision. Together, these innovations enable robust and efficient restoration with minimal parameter overhead.

### Strengths
1. The proposed method is simple and easy to follow.
2. The proposed method achieves state-of-the-art performance on real-world super-resolution datasets.

### Weaknesses
1. The usage of the term "Universal image restoration" is not accurate. In the low-level vision field, "Universal image restoration" usually refers to all-in-one image restoration tasks [1,2,3] in most papers. However, in my opinion, this paper mainly addresses real-world image restoration tasks.

2. In terms of model architecture, the proposed method adopts an identical dual-branch design as DreamClear, showing limited structural innovation.

3. I believe the proposed dataset is an important contribution of this paper. However, it is unclear whether it will be open-sourced.

4. The experiments comparing the proposed method with commercial models are not meaningful, as many of the models used for comparison are not super-resolution models.

5. This paper uses FLUX as the base model, which has significantly more parameters and computational cost compared to previous works. The authors need to provide inference efficiency comparisons with prior methods.


 References

[1] Controlling Vision-Language Models for Universal Image Restoration. ICLR, 2024.

[2] Universal Image Restoration Pre-training via Degradation Classification. ICLR, 2025.

[3] Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model. CVPR, 2024.

### Questions
Refer to Weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents LucidFlux, a new approach to universal image restoration. The framework leverages a large-scale diffusion transformer  to restore real-world images degraded by various unknown distortions. A key contribution of this work is the development of a caption-free semantic alignment method, which replaces traditional text-based approaches with SigLIP-based semantic features derived from a lightly restored proxy image. In addition, the method incorporates a dual-branch conditioner and a timestep- and layer-adaptive modulation schedule, which together help to enhance both the structural integrity and texture details of the restored images. The framework also includes a scalable data curation pipeline to filter large scale image datasets, ensuring high-quality supervision throughout the training process.

### Strengths
1. LucidFlux introduces a caption-free semantic alignment method that reduces reliance on external captions. The model employs timestep- and layer-adaptive modulation to ensure efficient recovery of both fine textures and global structure. Additionally, it introduces a scalable and publicly documented data curation paradigm, ensuring high-quality supervision for large-scale image restoration.

2. The method demonstrates state-of-the-art performance across both synthetic and real-world benchmarks, consistently surpassing commercial models in both quantitative and qualitative evaluations. It offers valuable insights to the community regarding performance between commercial models and the open-source domain.

3. The paper provides a well-structured and systematic explanation of the proposed framework. The design rationale behind key components is clearly articulated, offering convincing insights into how they work in concert to enhance restoration quality.

### Weaknesses
1. The idea of LucidFlux is similar with DreamClear, which affects its novelty. The authors shall clarify the differences between them.

2. Some details of compared methods are missing. Clarifying these settings would improve the transparency and comparability of the results.

3. The contribution of caption-free is not so clear. Some experimental results with the same Flux.1 backbone can be useful.

### Questions
1. The core design of the dual-branch conditioner appears to be functionally equivalent to the paradigm used in DreamClear [1]. What is the key methodological difference that sets this work apart and establishes its novelty?

2. Although the paper mentions that experiments were conducted at a resolution of 1024, there is a lack of detail regarding the settings for comparison models, especially for super-resolution techniques. Important parameters, such as the upscale factor, are not clearly defined. 

3. Although the paper claims that LucidFlux outperforms commercial models, the comparison includes Gemini-NanoBanana and Seedream 4.0, which are not specialized image restoration models. The authors do not clarify why these models were included in the comparison—whether they are considered to have performance comparable to specialized restoration models, or if there were no other commercial restoration models available for comparison. Providing more context on this choice would help clarify the relevance of these comparisons.

4. Beyond fully freezing the backbone, have you explored partial unfreezing (e.g., selected cross‑attention or upper layers)? Please report impacts on perceptual metrics and PSNR/SSIM, as well as stability and compute/memory trade‑offs.

5. Your curation uses fixed thresholds (e.g., Laplacian variance 150–8000, Sobel flat‑patch threshold ≈800 with 50% image‑level ratio, CLIP‑IQA Top‑20%). Could you report sensitivity to these choices (e.g., Top‑10/30%, different Sobel cutoffs) and their effect on metrics/semantic coverage?

6. For a clean attribution of the caption‑free design, can you add a comparison on the same Flux.1 backbone with (i) ideal GT captions and (ii) inference‑time VLM captions, and report both quality and latency?


[1] Ai Y, Zhou X, Huang H, et al. DreamClear: High-Capacity Real-World Image Restoration with Privacy-Safe Dataset Curation[J]. Advances in Neural Information Processing Systems, 2024, 37: 55443-55469.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose LucidFlux, a Flux.1 diffusion transformer adapted to the universal image restoration task. 

Starting from a frozen Flux.1 backbone, four trainable modules are added which are trained on 1.36 million LQ-HQ image pairs. These image pairs are synthesized using the Real-ESRGAN degradation pipeline, starting from 342k HQ images which the authors in turn obtain by applying a proposed filtering pipeline to 2.9 million candidate images.

The proposed LucidFlux model obtains impressive qualitative results and performs well compared to recent open- and closed-source methods.

### Strengths
- The paper is well written overall, components of the proposed approach are explained well in Section 3. I found the paper quite interesting and enjoyed reading it overall.
- LucidFlux seems to perform well compared to recent open- and closed-source methods.
- LucidFlux achieves seemingly very impressive qualitative/visual results.

### Weaknesses
- Incorrect citation formatting across basically the entire paper. For example, line 69: "Discriminative restorers based on CNNs and Transformers Dong et al. (2016); Liang et al. (2021); Zamir et al. (2022) perform well on..." --> "Discriminative restorers based on CNNs and Transformers (Dong et al., 2016; Liang et al., 2021; Zamir et al., 2022) perform well on...". This makes parts of the paper quite difficult to read, but it's a simple fix.
- Some details regarding the data collection and model training are missing.

### Questions
- In Section 3.4 you write "First, we collect 2.3M images from the Internet". This is quite vague indeed? Could you give more details on how and from which sources you collect these images?
- Perhaps I'm missing something obvious, but what loss is used to train the model?




Minor things:
- Duplicate reference of "Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study"?
- "MLLM" should perhaps be defined when it's used in the abstract and at the beginning of Section 3.
- In Figure 2, $\phi_{LQ}$ and $\phi_{LRP}$ could perhaps be added as output of the two Dual-Branch Conditioners blocks?
- Line 218: LQ and LRP might not need to be defined again here?
- Line 223: "These parameters effect" --> These parameters affect"?
- In equation 4, I would probably consider changing the circle symbol to something else.
- "T2I" is defined at the start of Section 3.4, but not at the start of 3.3.
- The "Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models" reference is missing year and venue.
- "IQA" should probably be defined somewhere.
- Line 453: "and three scores are enlarged after adding caption-free" --> "and all three scores are improved after adding caption-free"?
- Line 464: "and the identical IQA metrics are used in the open-source comparisons" --> "and the identical IQA metrics are used as in the open-source comparisons"?
- Line 482, "by large DiTs.Across real": typo.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes LucidFlux, a caption-free UIR framework that adapts the large diffusion transformer Flux.1 without relying on image captions.

### Strengths
1. The paper is well-written, the results are great,  and the experiments are solid.
2. A lightweight dual-branch conditioner is incorporated that injects signals from both the degraded input and a lightly restored proxy.
3. A novel timestep- and layer-adaptive modulation schedule then routes these conditioning cues throughout the backbone's hierarchy. 
4. To ensure semantic alignment without the latency and instability of text prompts or MLLM captions, caption-free alignment is enforced using SigLIP features extracted from the proxy image.

### Weaknesses
1. It is claimed that "Universal image restoration (UIR) aims to recover images degraded by unknown mixtures while preserving semantics—conditions under which discriminative restorers and UNet-based diffusion priors often oversmooth, hallucinate, or drift." However, there are lots of DiT-based diffusion priors in UIR. This main challenge is not solid.
2. As for the first contribution, "a lightweight dual-branch conditioner", it seems very similar to DiffBIR. The main difference is that DIffBIR only uses a lightly restored proxy as conditions.
3. As for the second contribution, "timestep- and layer-adaptive modulation", it is also a common technique for injecting conditions.
4. "A SigLIP-based module preserves semantic consistency without prompts or captions." The motivation is great but the contribution is minor.

### Questions
1. The paper positions itself by claiming that UNet-based diffusion priors in UIR often lead to oversmoothing, hallucination, or semantic drift. However, numerous recent and successful UIR methods utilize DiT-based architectures. Could the authors clarify the specific limitations of these existing DiT-based priors to more solidly justify the necessity of their proposed approach and distinguish it from this relevant body of work?
2. The proposed dual-branch conditioner appears functionally similar to the conditioning mechanism in DiffBIR, which also utilizes a restored image proxy. Could the authors elaborate in detail on the fundamental technical and conceptual differences between their conditioner and prior arts, clarifying the specific novelty and advantages introduced by their dual-branch design?
3. The timestep- and layer-adaptive modulation is presented as a key contribution. Given that adaptive feature modulation is a well-established technique for injecting conditions in diffusion and other generative models, what is the specific innovation in your implementation? Please clarify how your method differs from and advances upon this common paradigm.
4. Why not compare DiffBIR across all experiments?

Minor Suggestion:
1. It is recommended to use \citep rather than \cite.

### Soundness
4

### Presentation
4

### Contribution
2
