# Pixel-Level Residual Diffusion Transformer: Scalable 3D CT Volume Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Generating high-resolution 3D CT volumes with fine details remains challenging due to substantial computational demands and optimization difficulties inherent to existing generative models. In this paper, we propose the Pixel-Level Residual Diffusion Transformer (PRDiT), a scalable generative framework that synthesizes high-quality 3D medical volumes directly at voxel-level. PRDiT introduces a two-stage training architecture comprising 1) a local denoiser in the form of an MLP-based blind estimator operating on overlapping 3D patches to separate low-frequency structures efficiently, and 2) a global residual diffusion transformer employing memory-efficient attention to model and refine high-frequency residuals across entire volumes. This coarse-to-fine modeling strategy simplifies optimization, enhances training stability, and effectively preserves subtle structures without the limitations of an autoencoder bottleneck. Extensive experiments conducted on the LIDC-IDRI and RAD-ChestCT datasets demonstrate that PRDiT consistently outperforms state-of-the-art models, such as HA-GAN, 3D LDM and WDM-3D, achieving significantly lower 3D FID, MMD and Wasserstein distance scores.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a transformer-based 3D diffusion model for medical CT generation that integrates a patch-wise Local Denoiser and a global Diffusion Transformer within a two-stage residual framework. A predictor–corrector (“hot diffusion”) sampling strategy improves generation stability and diversity. Comprehensive ablation studies validate each component and analyze computational efficiency. The progressive training scheme reuses a pretrained low-resolution model and fine-tunes lightweight residual modules at higher resolutions, achieving competitive performance with reduced training cost.

### Strengths
1. The progressive training strategy from low to high resolution is clearly defined and achieves comparable performance with reduced computational cost.

2. The combination of local and global components provides a balanced design for modeling fine details and overall structure.

3. The model attains competitive quantitative results and produces visually consistent high-resolution 3D CT volumes.

### Weaknesses
1. Visual inconsistency in generated samples:
In Figure 5, the second and fifth examples under Ours show black voids in the lower-left region, indicating low-density areas that are absent in the real data. The authors should clarify whether these artifacts result from the “hot diffusion” sampling strategy, which may disrupt local density continuity or anatomical consistency.

2. Extra dataset evaluation:
Additional experiments on CT datasets such as CT-ORG are recommended to further validate the model’s generalization.

3. Potential limitation of frozen local denoiser:
The local denoiser remains frozen during high-resolution training. It would be valuable to include an ablation comparing frozen versus fine-tuned local modules to assess potential performance degradation at higher resolutions.

### Questions
1. What is the potential to extend this framework to conditional generation, such as generating raw CT volumes from segmentation maps or other structural priors?

2. In Figure 5, several artifacts are visible. Could the authors explain their causes and further evaluate the influence of the parameter k on generation quality through additional qualitative or quantitative analysis?

3. The local denoiser is frozen during high-resolution training. Could this design lead to performance degradation at higher resolutions? An ablation comparing frozen and fine-tuned local denoisers would help clarify this effect.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces PRDiT, a two-stage diffusion transformer framework for synthesizing high-resolution 3D CT volumes directly at voxel level. A predictor–corrector diffusion sampling method and a progressive low-to-high-resolution training strategy improve sample fidelity and efficiency. Experiments on LIDC-IDRI and Rad-ChestCT show clear advantage over other methods.

### Strengths
1. The decomposition of diffusion into local + global residual branches is elegant and addresses the long-standing trade-off between local detail and global coherence in 3D image synthesis; 
2. The low-to-high-resolution reuse strategy reduces training cost
3.Figures 4–5 demonstrate noticeably sharper bone edges, smoother organ boundaries, and fewer artifacts relative to baselines
4. Strong reproducibility section (datasets, configs, metrics) and detailed appendices on architecture, hyperparameters, and inference time

### Weaknesses
The idea of splitting local/global branches is incremental relative to prior hierarchical or multi-scale diffusion models. The architectural originality lies mainly in combining them via residual refinement rather than introducing new attention or tokenization mechanisms.

Reported mean ± std over 3 seeds is small; given large variance in 3D generation, stronger statistical support or significance tests would enhance credibility.

No mention of hallucination risk or downstream misuse

### Questions
n/a

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
I reviewed the paper, a two-stage diffusion architecture for high-resolution 3D medical image synthesis. The model combines a local MLP denoiser, which captures fine-grained voxel details on overlapping patches, with a global residual Transformer, which ensures anatomical consistency across the whole volume.   
The authors also propose a predictorcorrector sampling scheme that reintroduces controlled noise to stabilize generation, and a low-to-high-resolution scaling strategy that allows efficient 256 training guided by a pretrained 128 backbone.

They evaluate their method on LIDC-IDRI and Rad-ChestCT, showing improved FID and MMD over several baselines (HA-GAN, 3D-LDM, WDM-3D). The generated CT volumes appear sharper and more realistic, while training remains computationally feasible. Overall, the work presents a solid engineering improvement that makes transformer-based diffusion models more practical for 3D medical data synthesis.

### Strengths
I find the paper technically solid. The proposed two-stage design separating local voxel-level denoising from global residual refinement is clearly explained, and supported by ablation studies showing that each component contributes to performance. The predictorcorrector sampling strategy is effective, improving image quality with minimal computational overhead. I also appreciate the scaling approach that leverages a pretrained low-resolution model to enable efficient 256 synthesis, which addresses an important computational bottleneck for 3D diffusion models.

The paper includes comprehensive experiments on two public CT datasets, reports multiple quantitative metrics (FID, MMD, and provides consistent improvements over established baselines. Figures illustrate sharper structural detail and realistic textures. I also value the inclusion of limitations and future work, as well as the clear structure and readability of the paper.

### Weaknesses
While the method is good, I find the novelty somewhat limited. The combination of a local denoiser and a global Transformer is conceptually straightforward and resembles existing hierarchical or hybrid DiT approaches. Similarly, the proposed sampling resembles previously known stochastic sampling methods, though applied here in a slightly modified form.

The evaluation focuses mainly on generative metrics such as FID and MMD, which can  be unstable for 3D medical data. There is no downstream or clinical validation (e.g., segmentation or detection performance using synthetic data), which makes it difficult to judge real-world utility. Baselines are also somewhat limited and some stronger 3D diffusion models or DiT variants are missing, especially at higher resolutions

Finally, several implementation details are underspecified, such as the exact memory-efficient attention mechanism, data split protocol, and reproducibility of high-resolution experiments. Overall, the paper’s contribution feels more like a solid engineering refinement than a major conceptual breakthrough.

### Questions
- Could you clarify which memory-efficient attention variant is used in the global DiT (e.g., FlashAttention, windowed, or block-sparse)? How much does it contribute to scalability compared to a vanilla DiT?
- Predictor sampling: Can you provide pseudocode or additional explanation of the schedule? Did you explore adaptive or variable numbers of corrective steps (k > 2), and how stable is training when increasing k?
- Baselines: Why were recent efficient 3D diffusion or DiT variants  excluded from comparison? Would your method still outperform these stronger models, especially at 256 resolution?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Generating fine-grained 3D CT data is an extremely challenging problem. This paper addresses this challenge by employing a pixel-level residual diffusion transformer. The chosen topic is highly relevant and represents a significant area of research. The methodology utilizes a coarse-to-fine strategy and includes comparisons with various established models.
The authors demonstrated superior performance against existing models by comparing metrics such as 3D DIF, MMD, and W-Score on two generated datasets of $128^3$ resolution. However, a significant drawback is the lack of evaluation by a medical imaging expert from a clinical image generation standpoint.
Observing Figure 4, 5, the model appears to capture the overall coarse shape, but there still seem to be significant problems with the finer details. Specifically, as the layers deepen, internal structures within the lungs appear to vanish, and the soft tissue contrast does not accurately reflect reality.
The overall comparison lacks benchmarking against the current State-of-the-Art (SoTA) DiT (Diffusion Transformer) model. The paper mentions that DiT models suffer from "unstable dynamics and optimization difficulties," yet it fails to provide any comparative experiments to support this claim or justify the exclusion.

### Strengths
Generating fine-grained 3D CT data is an extremely challenging problem. This paper addresses this challenge by employing a pixel-level residual diffusion transformer. The chosen topic is highly relevant and represents a significant area of research. The methodology utilizes a coarse-to-fine strategy and includes comparisons with various established models.
The authors demonstrated superior performance against existing models by comparing metrics such as 3D DIF, MMD, and W-Score on two generated datasets of $128^3$ resolution. However, a significant drawback is the lack of evaluation by a medical imaging expert from a clinical image generation standpoint.

### Weaknesses
Observing Figure 4, 5, the model appears to capture the overall coarse shape, but there still seem to be significant problems with the finer details. Specifically, as the layers deepen, internal structures within the lungs appear to vanish, and the soft tissue contrast does not accurately reflect reality.

### Questions
The overall comparison lacks benchmarking against the current State-of-the-Art (SoTA) DiT (Diffusion Transformer) model. The paper mentions that DiT models suffer from "unstable dynamics and optimization difficulties," yet it fails to provide any comparative experiments to support this claim or justify the exclusion.

### Soundness
2

### Presentation
2

### Contribution
2
