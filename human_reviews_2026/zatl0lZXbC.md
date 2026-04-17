# HNDiff: Haze-Noise Diffusion for Image Dehazing

- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
Existing diffusion-based methods have recently made significant progress in image dehazing. However, they typically neglect the physics of haze formation and reconstruct clean images from pure Gaussian noise, thereby limiting their restoration potential. To address this issue, we propose Haze-Noise Diffusion (HNDiff), a novel diffusion framework that embeds the atmospheric scattering model as an inductive bias. By grounding diffusion in physical principles, HNDiff ensures that the restoration aligns more closely with underlying mechanisms of haze formation. In its forward process, we introduce joint haze-noise diffusion with a haze-aware noise scheduler, which progressively adds both haze and noise to an image. Essentially, the scheduler adapts noise levels according to haze density, meaning that regions with heavier haze receive stronger noise injection to encourage content generation, while clearer regions receive lighter noise to better preserve details, which directly links the forward degradation process with the physics of haze. In the reverse process, we then derive a physically consistent dehazing-denoising process that simultaneously removes haze and noise to restore a clean image in a manner aligned with the forward degradation process. To further enhance practicality, we propose Latent HNDiff, which compiles clean latent priors that can be seamlessly integrated into existing dehazing networks to boost performance.  Extensive experiments show that our work significantly improves leading dehazing backbones and achieves state-of-the-art results on benchmark datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces HNDiff, a diffusion-based image dehazing framework that embeds the ASM into both the diffusion processes. Instead of reconstructing clean images from pure Gaussian noise, HNDiff models haze formation by progressively adding both haze and noise in a physically grounded forward process and removes them in the reverse process. The method features a haze-aware noise scheduler that adapts noise injection based on haze density. To improve computational efficiency and generalization, a Latent HNDiff variant is proposed, operating in latent space as a prior generator integrated into standard dehazing networks through a FGM. Extensive experiments across five benchmarks and three baselines show consistent improvements in fidelity metrics.

### Strengths
1. Integrating the ASM as an inductive bias provides a theoretically consistent link between physical haze formation and diffusion modeling.

2. The haze-aware noise scheduler adapts the diffusion process spatially, aligning generative noise injection with haze density and improving detail recovery.

3. Comprehensive experiments on multiple datasets and architectures demonstrate consistent performance gains and strong generalization.

### Weaknesses
1. The three-stage training (ground-truth prior pretraining, latent diffusion optimization, joint fine-tuning) adds significant implementation difficulty and training time. The quantitative gains (e.g., +0.5 dB PSNR on average) are relatively small considering the model’s complexity and multi-stage training strategy.

2. The paper provides strong empirical results but lacks ablation or convergence analysis explaining how ASM integration affects the diffusion dynamics mathematically.

3. A method that uses a latent diffusion to provide prior to help the image restoration performance has become relatively common. This work appears to be just an implementation and improvement of this type of pattern under an atmospheric scattering model. The diffusion  process designed for mist-noise and ASM originally would have brought better interpretability, but using the potential diffusion model seems to obscure this advantage.

### Questions
1. How does the latent diffusion representation compare to full-image diffusion in terms of interpretability and potential loss of fine spatial information?

2. Is the improvement mainly due to the physical prior or the additional parameters introduced by the latent modules and FGM?

3. I am more curious about whether the haze-noise diffusion pattern designed by this paper can be applied to image-to-image diffusion models rather than latent diffusion models. This seems to be an interesting exploration.

4. More experiments and comparison on RTTS dataset is recommended to validate the out-of-domain generalization performance.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Haze-Noise Diffusion (HNDiff), a diffusion-based image dehazing framework that integrates the atmospheric scattering model as a physical prior. Unlike conventional methods that add only Gaussian noise, HNDiff employs a haze-aware noise scheduler that adapts noise levels based on haze density. The authors also propose Latent HNDiff, which leverages clean latent priors to enhance existing dehazing networks.

### Strengths
1. HNDiff integrates the physical model into the diffusion process, mimicking the formation of haze by highlighting its spatially varying characteristics.
2. The latent formulation reduces computational cost while improving restoration accuracy and visual fidelity.

### Weaknesses
1. The paper introduces the Haze-Noise Diffusion process, claiming to incorporate the physical mechanism of haze formation as an inductive bias for improved dehazing. While the methodology is detailed, the authors fail to clearly justify the necessity of this inclusion. Specifically, the paper lacks a **clear, intuitive explanation** to help the reader understand why this specific combination of haze and noise addition in Eq.(2), is more effective.
2. The manuscript introduces several distinct components, including HNDiff framework, Haze-Aware Noise Scheduler, and Latent HNDiff with FGM, yet it does not establish a clear hierarchical structure or narrative focus among them. This lack of prioritization weakens the overall coherence of the work, making it difficult for readers to discern which element represents the paper’s primary innovation or how these components collectively address a unified research question.

### Questions
1. What does HANS stand for in Figure 2?
2. The manuscript claims that the proposed model achieves effective dehazing within only four diffusion steps. Given such an extremely limited number of iterations, what meaningful dehazing features are actually learned? How these few steps capture haze-related representations and improve restoration quality?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors propose the HNDiff (Haze-Noise Diffusion) framework, which integrates the Atmospheric Scattering Model (ASM) as an inductive bias into the diffusion process. Through a bidirectional pipeline of "joint haze-noise diffusion" and "physically consistent dehazing-denoising", it addresses the issues of low restoration fidelity in traditional diffusion-based dehazing methods—specifically, their neglect of the physical principles of haze formation and reliance on pure Gaussian noise.   Additionally, the proposed Latent HNDiff can serve as a prior generation network, enabling flexible integration with existing dehazing backbones. Experiments demonstrate that HNDiff consistently enhances the performance of three mainstream dehazing models (FocalNet, ConvIR, and SGDN) across five benchmark datasets.   Moreover, it outperforms schemes that merely scale up network size while requiring fewer parameters and lower computational complexity, verifying its effectiveness and generalization.

### Strengths
Physical grounding: Embedding ASM into diffusion is a critical strength—unlike conventional diffusion methods that ignore haze’s structured, spatially varying nature, HNDiff’s forward/reverse processes align with how haze forms in the real world.  This ensures more realistic restorations and better generalization to real-world haze.

Latent HNDiff’s ability to enhance existing dehazing backbones (FocalNet, ConvIR, SGDN) without reengineering them makes it a valuable tool for researchers and engineers. The FGM module’s lightweight design (pooling + MLP) ensures minimal overhead, and the three-stage training balances stability and performance.

The paper’s results are convincing across metrics and datasets.

### Weaknesses
The paper mentions the haze estimator uses a simplified U-Net. However, it does not provide: (a) the exact architecture of the U-Net (e.g., number of layers, channel counts, skip connections); (b) the loss function used to train the estimator; (c) The performance of the haze estimator in extremely dense haze scenarios.

The paper has verified the consistency between HNDIFF and the Atmospheric Scattering Model (ASM) under uniform haze scenarios. However, haze in real-world environments is often non-uniformly distributed (e.g., gradient haze where density increases with depth). It remains unclear how the current haze-aware noise scheduler adapts to such scenarios.

### Questions
Does the implicit haze residual modeling (via continuous accumulation) effectively capture non-uniform haze density variations?
How do the estimators avoid mistaking image details (e.g., leaf textures, building edges) for haze residuals? For instance, a dense forest may have texture densities similar to light haze.

### Soundness
3

### Presentation
3

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
In this paper, the authors proposed a new framework, termed HNDiff, for image dehazing task. Its latent version can be integrated into existing dehazing networks to boost performance. The experimental results show that it can be successfully applied into classical models (e.g., FocalNet, ConvIR, and SGDN).

### Strengths
In summary, HNDiff represents an advancement in diffusion-based image dehazing by embedding physical principles into the diffusion process. The theoretical derivation looks highly rigorous.

### Weaknesses
1. The gt prior $Z_{gt}$ is generated from the hazy image $I_H$ and its clean counterpart $I_0$. What is the function of  $I_H$ here? From my perspective, $I_H$ will only bring about negative impacts.
2. The visualization results in Figure 5 are not reliable. The authors merely selected the first channel, which is prone to be biased. A relatively more reasonable approach would be to average each latent over the channel dimension or adopt other statistically meaningful methods. By the way, the dimensions of $Z_{0:4}$ are not identical to $I_H$. If any resize operations are applied here, clearly explanations should be provided.
3. The entire training process requires three stages, which is quite complex and may cause robustness issues. During the stage 2, only a latent-prior loss is employed? The estimated haze and noise are not supervised?
4. The idea of haze-aware noise scheduler (dynamically adjust the noise level according to haze density) has been explored before.
5. The HANS in Figure 2 is not defined before it is used.
6. Experiments on real hazy images (e.g., RTTS) are missing. In Table 1, SOTS-outdoor is more suggested than SOTS-indoor. The visual results on other models besides FocalNet, ConvIR, and SGDN are not provided. The visual result of HNDiff in Figure 13 contains some artifacts.

### Questions
Please refer to the weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2
