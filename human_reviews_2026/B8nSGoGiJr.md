# Noise and anatomy-guided diffusion model for realistic CT image synthesis

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Generative models, specifically Diffusion Models (DMs), have been quite successful in generating high-quality images. However, DMs rely on large-scale training data. In medical imaging, more specifically for computed tomography (CT), these models struggle in accurately reconstructing anatomical structures due to limited training data. This can cause the wrong depiction of organs, which can impact clinical treatment. Some existing models, although guided by anatomical structures, ignore dose-dependent noise, which is critical in real-world scenarios. To tackle this challenge, we propose a novel diffusion model, namely NA-Diff, which is guided by noise from different dose levels and anatomical structures, leveraging a dual conditional diffusion framework. To facilitate large-scale training of DMs on complex structured CT data, we transform natural images emulating realistic CT noises and leverage them for pre-training, followed by fine-tuning on small CT data. Extensive experimental results demonstrate that NA-Diff generates high-fidelity and noise-aware CT images, effectively delineating the organ-of-interest and bridging the gap between synthetic and real CT.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles diffusion models’ limitations in CT synthesis—poor anatomical fidelity, ignored dose-dependent noise—by proposing NA-Diff, a dual conditional diffusion model guided by dose-level noise and anatomical structures. It uses pre-training on CT noise-emulated natural images and fine-tuning on small CT data to mitigate domain shift and data scarcity, with two variants: Sep-Diff, which adopts a three-stage design with separate conditioning on noise and anatomy, and Cat-Diff which uses a two-stage concatenated conditioning strategy.

Contribution:
1.Proposed NA-Diff, a dual conditional framework using noise maps and anatomy masks for CT synthesis.
2.Constructed pre-training datasets by emulating CT noise on both CT and natural images across varying dose levels.
3.Mitigated the reliance on scarce CT data through pre-training on noise-emulated natural images.
4.Demonstrated the superiority of NA-Diff through extensive quantitative evaluations using image quality and segmentation metrics.

### Strengths
1.The paper is well-structured, and the figures clearly illustrate the overall framework of the proposed method.

2.The references are comprehensive, relevant, and demonstrate a thorough review of related work.

3.The experimental evaluation is fairly comprehensive, comparing different methods in terms of image quality assessment (IQA), generative performance, and downstream segmentation results.

### Weaknesses
1.The methodological novelty is limited. The paper is built upon the DiT architecture, with only modifications to the conditioning setup and repeated serial training.

2.There is a lack of rigorous theoretical justification or supporting evidence to ensure the authenticity and reliability of the generated data.

3.The experimental results are relatively weak. The quality of the synthesized CT images is moderate. As shown in Figure 4, the generated CT images suffer from detail loss and exhibit noticeable blurriness. The performance of synthetic data in the segmentation task is modest. As shown in Table 6, incorporating synthetic data leads to marginal improvements in model performance.

4.Organ-level limitation. The current study focuses solely on liver structures in CT images and has not been extended to other organs. Further evaluation is needed to verify the model’s generalizability across different anatomical structures [1].

[1] Ma, Jun, et al. "Fast and low-GPU-memory abdomen CT organ segmentation: the flare challenge." Medical Image Analysis 82 (2022): 102616.

### Questions
1.How does the paper ensure the authenticity and reliability of the generated CT data?

2.In Table 2, it is unclear whether NA-Diff-D uses the noise condition. According to Table 1, NA-Diff-D is a model pre-trained based on noise-conditioned training, which implies that it indeed includes the noise condition. This should be clearly stated in Table 2 for consistency.

3. What are the display window settings used for CT image display? It is recommended to indicate them in the figures.

### Soundness
2

### Presentation
3

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
Diffusion models typically require large-scale real-world datasets for effective training; however, medical imaging datasets are limited in size, making it challenging to fulfill such data demands. Moreover, existing generative approaches often overlook dose-dependent noise and anatomical structures in CT images. To address these issues, this paper proposes a novel Diffusion Transformer–based approach for CT synthetic. The method first conducts pre-training on natural images perturbed by CT noise, and then performs fine-tuning on CT guided by dose-dependent noise and anatomical structure. As a result, the model is able to generate high-fidelity and noise-aware synthetic CT scans.

### Strengths
1.	The paper effectively mitigates the challenge of limited medical imaging size by the pre-trained diffusion model on a large set of CT noise–emulated natural images.
2.	Guided by dose-dependent CT noise modeling and anatomical structural during fine-tuning, the method is able to generate synthetic CT images that reflect real clinical characteristics.
3.	Experimental results demonstrate clear improvements in the fidelity and realism of the generated CT images, validating its effectiveness.

### Weaknesses
1.	The novelty of the approach is somewhat incremental, as it mainly adapts existing diffusion architectures with noise and anatomy fine-tuning strategies
2.	The paper does not clearly articulate the motivation behind generating noise-aware CT images. From the perspective of downstream medical tasks, higher-dose and lower-noise CT scans yield better performance for diagnosis. So, why modeling noise variations is necessary? Additional explanation or downstream validation would help clarify this design.
3.	The experiments are somewhat limited in terms of dataset diversity and baseline comparisons. It could include results on more CT datasets and assess the impact of utilizing different natural image datasets for pre-training.

### Questions
1 In the original DiT framework, the final outputs are predicted noise term and covariance matrix. But in Equation (6), the output $\hat{x_0}$ is a denoised image. Additionally, the forward process described in Equation (4) seems to deviate from the AdaLN-Zero formulation in the original DiT. More explanation of the modifications would improve clarity.

2 In the Introduction, Contributions 2 and 3 appear similar, both focusing on CT noise-emulated natural images. Why not consolidate them into a single contribution?

3 Equations (7) and (8) introduce dose-dependent noise estimation, but the paper does not provide sufficient explanation about its physical basis or formulation. Could the authors explain why this type of noise is dose-dependent?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces NAG-DPM (Noise- and Anatomy-Guided Diffusion Probabilistic Model), a physics-informed diffusion framework for accelerated MRI reconstruction. The key idea is to guide the reverse diffusion process with two complementary priors: Noise Guidance: An adaptive noise-estimation module dynamically adjusts the diffusion variance to match the k-space noise level, improving robustness to acquisition noise. Anatomy Guidance: A U-Net–based anatomical prior (trained on fully sampled references) constrains the diffusion trajectory toward anatomically consistent reconstructions. A two-stage training scheme jointly learns both priors. Experiments on fastMRI Knee and Brain (T2, FLAIR) datasets show that NAG-DPM outperforms both physics-driven networks (MoDL, VarNet) and existing diffusion-based reconstructions (Score-MRI, ADiffRecon) in PSNR/SSIM while maintaining higher stability under noise corruption and adversarial perturbations.

### Strengths
The joint noise-anatomy guidance effectively balances fidelity and denoising, addressing a major limitation in existing diffusion MRI reconstructions

Visual results show sharper anatomical details with reduced ringing and over-smoothing

Embeds data-consistency steps directly into the reverse diffusion loop, ensuring consistency with measured k-space data.

The theoretical motivation for adaptive variance modulation is sound and aligns with diffusion score calibration

Diffusion models are notoriously slow; introducing adaptive noise estimation cuts sampling steps by ~40% while improving stability. This is a meaningful engineering contribution.

### Weaknesses
While the dual-guidance formulation is elegant, the diffusion core (DDPM) and U-Net prior are largely standard. The novelty lies more in integration than in new diffusion mathematics.

No discussion of scalability to 3D volumes or real-time applications. Experiments focus exclusively on 2D Cartesian MRI. Validation on 3D or non-Cartesian acquisitions would make the contribution more general.

Despite adaptive variance, inference remains costly (≈6–10 s per slice vs. <1 s for MoDL).

### Questions
na

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces NA-Diff for CT image synthesis guided by both noise and anatomical conditions. NA-Diff leverages the emulated CT noise and natural-image pretrained weights to overcome limited CT data.

### Strengths
1. Combining dose-level noise and anatomy guidance sounds well-motivated in medical image synthesis, especially at limited data regime.
2. The methodology is clearly presented.

### Weaknesses
1. The paper mainly focuses on liver-mask guided generation. Including more challenging anatomy structures, for example, smaller organs like pancreas, may better demonstrate the generalizability of the method.
2. What is the value of noisestep to generate the noise map that still maintains the original image structures? Does this value require manual selection?
3. In Table 6, considering the trivial improvement, including the confidence interval may be necessary.
4. Although in Table 7, there is a 1% Dice improvement compared to Seg-Diff, the additional computational cost needs to be considered.

### Questions
Will the code be released?

### Soundness
2

### Presentation
2

### Contribution
2
