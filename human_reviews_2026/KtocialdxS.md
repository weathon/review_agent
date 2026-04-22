# LowDiff: Efficient Diffusion Sampling with Low-Resolution Condition

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Diffusion models have achieved remarkable success in image generation but their practical application is often hindered by the slow sampling speed. Prior efforts of improving efficiency primarily focus on compressing models or reducing the total number of denoising steps, largely neglecting the possibility to leverage multiple input resolutions in the generation process.
In this work, we propose LowDiff, a novel and efficient diffusion framework based on a cascaded approach by generating increasingly higher resolution outputs.
Besides, LowDiff employs a unified model to progressively refine images from low resolution to the desired resolution. With the proposed architecture design and generation techniques, we achieve comparable or even superior performance with much fewer high-resolution sampling steps. LowDiff is applicable to diffusion models in both pixel space and latent space.
Extensive experiments on both conditional and unconditional generation tasks across CIFAR-10, FFHQ and ImageNet demonstrate the effectiveness and generality of our method. Results show over 50\% throughput improvement across all datasets and settings while maintaining comparable or better quality. On unconditional CIFAR-10, LowDiff achieves an FID of 2.11 and IS of 9.87, while on conditional CIFAR-10, an FID of 1.94 and IS of 10.03. On FFHQ 64×64, LowDiff achieves an FID of 2.43, and on ImageNet 256×256, LowDiff built on LightningDiT-B/1 produces high-quality samples with a FID of 4.00 and an IS of 195.06, together with substantial efficiency gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper explores cascaded diffusion for more efficient generation. The paper trains cascaded diffusion into a single model instead of separate models. For UNet, this is achieved by utilizing different resolution scale of the network with resolution embedding. For transformer, this is achieved by only using resolution embedding. The paper evaluates the results on CIFAR-10, FFHQ, and ImageNet. The paper shows that the method can achieve better or similar performance in FID/IS while using less effective NFE.

### Strengths
1. The experiment section is extensive as the authors have demonstrated the performance on four established settings, CIFAR unconditional, CIFAR conditional, FFHQ, and ImageNet.
2. The experiment covers both transformers and UNet.
3. The result does show speedup without quality degradation. In some cases, it achieves better quality.

### Weaknesses
1. The paper lacks novelty and innovation in 2025. The cascaded diffusion approach is well established. The only novelty is to train it into a single network. For transformer models, the architecture is trivial. Many existing t2i/t2v models already pre-train on multiple resolutions and aspect ratios through NaViT architecture, so the resolution embedding is most likely not needed. For unet, the paper's proposed architecture is also straightforward and not very novel.

2. Prior work [1] has shown that it is possible to turn existing t2i/t2v models into resolution-cascaded generation while training-free. This again diminishes the novelty of the method proposed.

[1] Training-free Diffusion Acceleration with Bottleneck Sampling

### Questions
1. Table 11 and 12 show LowDiff achieves better results than CDM. This is surprising. Is CDM using the exact architecture as the LowDiff counterpart? Can authors explain the results?

2. Can you specify the hyperparameters, such as the inference CFG scale used for producing all the tables? For example, for Table 4, is the result guided or not guided? It is different from the report in LightningDiT paper. How does the author obtain the FID values?

3. For the ImageNet256 experiment trained on the latent space, is the downsampling/upsampling performed at the pixel space or the latent space? Please specify in the paper.

4. Table 1, can you add model size as a column for easier comparison?

5. Recent methods, such as MeanFlow [1],  can achieve one-step generation. This competes with cascaded diffusion, as one-step generation makes cascaded generation meaningless. The paper's use of LightningDiT makes it hard to compare with established works on ImageNet-256px. But you can still compare on CIFAR-10 against those one-step methods.

6. Minor formatting issue: Should use \citep{} instead of \cite{} in most places.

[1] Mean flows for one-step generative modeling

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
summary:

The paper proposes a diffusion-based framework aimed at accelerating the sampling process through a cascaded architecture. The main idea involves allocating more inference steps to lower-resolution stages and fewer steps to higher-resolution stages, thereby reducing overall computational cost while preserving generation quality. Additionally, the model introduces module sharing across different resolutions—reusing components learned at lower resolutions during higher-resolution synthesis—to improve both performance and parameter efficiency. Experimental evaluations are conducted on standard benchmarks such as CIFAR-10, FFHQ, and ImageNet.

### Strengths
Strengths:
1. The sharing mechanism seems different.

### Weaknesses
Questions:
1. The paper is not clearly presented and suffers from a lack of logical coherence. It is unclear what specific problem the authors aim to solve. For instance, while they claim to accelerate sampling via a cascaded framework, it remains ambiguous how their approach fundamentally differs from or improves upon existing cascaded methods. The explanation of how the proposed framework achieves faster sampling or better performance is insufficient. The authors should clearly articulate: (i) the core challenge they are addressing, (ii) how their method solves it, and (iii) the resulting benefits. Simply applying a cascaded structure is not novel; the paper needs deeper analysis to justify its contributions.

In our view, the authors present a new sharing mechanism from existing cascaded methods. However, their writting always focus on how the cascaded framework accelerates the sampling, not on why they adopts a new sharing mechansim, on what's benefits of the sharing mechansim.

2. The use of a cascaded framework for efficient image generation is well-established in the literature. The current work does not introduce significant architectural innovations or new acceleration techniques beyond this standard paradigm. As such, the claimed contributions appear largely rooted in the application of an existing strategy rather than a meaningful advancement in methodology. To strengthen the novelty, the authors should provide a detailed comparison with prior cascaded approaches and highlight specific technical distinctions.


3. Figure 4(a) presents a confusing design: the low-resolution generation stage appears to take features from the higher-resolution branch as input. This creates a cyclic dependency that contradicts the natural feed-forward flow of a cascaded system. Such a design raises concerns about the feasibility and implementation of the sampling process. The authors must clarify the direction of information flow and resolve this apparent inconsistency.

4. The paper lacks comparisons with established cascaded diffusion models such as PixelFlow and RelayDiffusion. These methods are directly relevant to the paper’s theme of accelerated sampling via multi-stage generation. Without benchmarking against such baselines, it is difficult to assess the true effectiveness and competitiveness of the proposed approach.

### Questions
see above

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The proposed work aims to improve the efficiency of the diffusion models using a cascaded multi-resolution approach, whereby the image is generated gradually from low to high res. The approach, coined LowDiff, uses a unified model with shared weights for better parameter-efficiency. Evaluations are conducted on a series of controlled benchmarks, i.e. CIFAR, FFHQ and Imagenet-256.

### Strengths
- The paper is generally easy to follow
- Good results compared with the baseline on the selected datasets (i.e. CIFAR, etc)
- Adequate ablation studies that explore the impact of the proposed component and multi-res architecture

### Weaknesses
- The concept itself, while interesting, is not really novel. First, the idea of progressive training and generation is a quite old concept, as early as 2017 (see for example: Progressive growing of gans for improved quality, stability, and variation, Karras et al, ICLR 2018). Secondly, this concept also resembles parts of the large body of work on training free high resolution image generation, that use the low resolution noise as guidance in various forms.

- The comparisons are performed on small and constrained datasets, on ultimately - low resolution images. It's unclear how this approach would scale.

- Furthermore, the comparison with state-of-the-art are (1) somewhat outdated and (2) don't include different approaches that aim to reduce latency.

- No combination of the current approach with other orthogonal directions, to understand if the gains stack.

- How is this method positioned given that flow models are able to generate high quality samples in very few steps?

### Questions
- L152-154: Convolutions are by design resolution-invariant, so its' a bit unclear why this is the argument made for preferring transformers. Could  the authors perhaps elaborate?

- Table 3: given that lower FID values are better, using 0 for i guess what means the model didn't work is a bit confusing. Perhaps this could be improved?

- Perhaps I missed this, but what is the impact of weight sharing cross-stages?

### Soundness
2

### Presentation
2

### Contribution
2
