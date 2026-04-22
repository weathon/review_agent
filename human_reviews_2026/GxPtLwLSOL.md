# GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 6, 4

## Abstract
Image super‑resolution (SR) is fundamental to many vision systems—from surveillance and autonomy to document analysis and retail analytics—because recovering high‑frequency details, especially scene-text, enables reliable downstream perception. scene-text, i.e., text embedded in natural images such as signs, product labels, and storefronts, often carries the most actionable information; when characters are blurred or hallucinated, optical character recognition (OCR) and subsequent decisions fail even if the rest of the image appears sharp. Yet previous SR research has often been tuned to distortion (PSNR/SSIM) or learned perceptual metrics (LPIPS, MANIQA, CLIP‑IQA, MUSIQ) that are largely insensitive to character-level errors. Furthermore, studies that do address text SR often focus on simplified benchmarks with isolated characters, overlooking the challenges of text within complex natural scenes. As a result, scene-text is effectively treated as generic texture. For SR to be effective in practical deployments, it is therefore essential to explicitly optimize for both text legibility and perceptual quality. We present GLYPH‑SR, a vision–language‑guided diffusion framework that aims to achieve both objectives jointly. GLYPH-SR utilizes a Text-SR Fusion ControlNet (TS-ControlNet) guided by OCR data, and a ping-pong scheduler that alternates between text- and scene-centric guidance. To enable targeted text restoration, we train these components on a synthetic corpus while keeping the main SR branch frozen. Across SVT, SCUT‑CTW1500, and CUTE80 at ×4 and ×8, GLYPH‑SR improves OCR F1 score by up to +15.18 percentage points over diffusion/GAN baselines (SVT ×8, OpenOCR) while maintaining competitive MANIQA, CLIP‑IQA, and MUSIQ. GLYPH‑SR is designed to satisfy both objectives simultaneously—high readability and high visual realism—delivering SR that looks right and reads right. We provide code, pretrained models, the synthetic corpus with generation scripts, and an evaluation suite to support reproducibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper treats scene-text SR as a bi-objective problem: visual quality + text legibility. It uses a dual-branch Text-SR Fusion ControlNet guided by OCR text/positions and a ping-pong scheduler that alternates text-centric and image-centric guidance. It reports big gains in OCR F1 on SVT/CTW/CUTE80 while keeping perceptual metrics competitive.

### Strengths
1. Clear goal (make text readable, not just “look sharp”).
2. Comprehensive evaluation with OCR metrics and perceptual IQA.

### Weaknesses
1. Limited analysis of trade-offs (e.g., when text gets clearer, what happens to non-text textures?).
2. No multilingual or curved-text stress test.
3. Sensitivity to OCR detector quality is not studied.

### Questions
1. How robust is the method to OCR detection errors?
2. Can the approach handle dense, multi-language street scenes?
3. How is the ping-pong schedule chosen; can it be learned?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a framework GLYPH-SR, which utilizes textual information mining from the LQ to guide the denoise process through the designed TS-ControlNet. This paper also introduced ping-pong scheduler to control the condition injection strength along the denoising trajectory for better balance between visual and textual restoration. Extensive experiments demonstrate the competitive performance of the proposed framework.

### Strengths
Strengthens:

1. Proposed the GLYPH-SR framework with TS-ControlNet, allowing fine-grained control over both glyph-level details and scene-level realism. Furthermore, a ping-pong scheduler is introduced to dynamically balance visual fidelity and text legibility during the denoise process.
2. Constructed a factorized synthetic corpus separating text degradation from global image degradation, enabling controlled finetunning and clear ablation analysis.
3. Analyzed the trade-off between SR metrics and OCR metrics, which is essential for the evaluation of the proposed framework and similar methods.

### Weaknesses
Weaknesses:

1. The novelty could be further improved. 
    - The text branch of the proposed TS-ControlNet continues to adopt the plain ControlNet structure, without any specific modifications for its text-focused role. Introducing task-oriented designs could potentially further improve its performance.
    - Although the paper considers the trade-off between SR and OCR metrics, it does not introduce a unified metric to evaluate both scene reconstruction and text restoration quality.

2. The experiments are insufficient. The paper lacks comprehensive evaluations on related real-world image super-resolution task.

3. Some minor writing issues. e.g. "paragraphStep" in line 273

### Questions
1. Does the proposed framework only support English text? How does it perform on non-Latin characters?

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
4

### Summary
This paper introduces GLYPH-SR, a novel vision-language model (VLM)-guided latent diffusion framework designed to address the dual-objective problem in image super-resolution (SR): achieving high perceptual quality and high-fidelity scene-text recovery. The core of the method is a dual-branch Text-SR Fusion ControlNet (TS-ControlNet) that integrates scene-level captions with OCR-derived text strings and their spatial positions. A key innovation is the "ping-pong" scheduler, which dynamically alternates between text-centric and image-centric guidance during the diffusion denoising process. The model is trained on a carefully constructed synthetic corpus that factorizes glyph quality and global image quality perturbations. Extensive evaluations on SVT, SCUT-CTW1500, and CUTE80 benchmarks at up to 8× scaling demonstrate that GLYPH-SR achieves significant improvements in OCR F1 scores (up to +15.18 percentage points) while maintaining competitive performance on perceptual metrics (MANIQA, CLIP-IQA, MUSIQ) against a strong suite of diffusion and GAN-based baselines.

### Strengths
1. The paper compellingly argues that text legibility is a critical yet overlooked aspect of SR in practical applications. It provides a clear analysis of the systemic biases (metric and objective) in prior work that lead to text hallucination or conservative restoration, effectively framing the need for a dual-objective approach.

2. The proposed TS-ControlNet architecture and the binary ping-pong scheduler are elegant and effective solutions for fusing semantic text cues with global image priors without disrupting the pre-trained diffusion backbone. The design allows for targeted text restoration through fine-tuning a relatively small number of parameters.

3. The construction of a four-partition synthetic dataset is a significant methodological contribution. It enables the precise disentanglement of text restoration from general SR, providing a clean signal for training the text-specific components.

4. The paper provides an extensive empirical evaluation across multiple datasets, scale factors, and a wide range of state-of-the-art baselines. The dual-axis evaluation protocol, reporting both OCR metrics and perceptual quality metrics, is thorough and appropriate for the claimed contributions.

5. The ablation studies on guidance components, the scheduler policy, and the sensitivity analysis to upstream OCR/VLM errors are systematic and provide valuable insights into the model's behavior, strengths, and limitations.

6. The qualitative results (Figures 1, 4, 5, 12, 13) are highly effective. They clearly demonstrate GLYPH-SR's superior ability to reconstruct legible, accurate text in challenging scenarios (e.g., 8× scaling) where other methods fail, providing strong visual support for the quantitative claims.

### Weaknesses
1. The related work, experiment  section (Section 2/4) and lacks a thorough discussion of several recent and highly relevant works that also leverage VLMs, text prompts, or diffusion models for text-aware image restoration. Notable omissions include, but are not limited to:

    a) Zhang et al. (2024), "Diffusion-based Blind Text Image Super-Resolution"

    b) Chen et al. (2024), "Image Super-Resolution with Text Prompt Diffusion" / "Universal Image Restoration with Text Prompt Diffusion"

    c) Zhang et al. (2024), "ConsisSR: Delving Deep into Consistency in Diffusion-based Image Super-Resolution"

    d) Bogolin (2025), "Text-Aware Image Restoration with Diffusion Models"

    e) Xiaoming et al. (2024), "Enhanced Generative Structure Prior for Chinese Text Image Super-Resolution"

    This gap weakens the contextualization of the paper's novelty and leaves the reader uncertain about how GLYPH-SR specifically advances the field beyond these concurrent efforts. A more comprehensive survey and a clearer delineation of contributions are needed.

2. While Figure 14 is provided, the discussion of failure cases is somewhat brief. A deeper analysis is warranted, particularly regarding: (a) the root cause of text hallucination in non-text regions (e.g., is it due to over-reliance on text guidance or errors in the initial OCR?); (b) the model's tendency to enhance only the most salient text instances; and (c) a critical failure mode not explicitly discussed: what happens when the upstream VLM/OCR fails to detect or correctly recognize severely degraded text in the LR input? This scenario is highly probable in real-world applications and likely breaks the method's core premise.

3. As shown in Table 6, GLYPH-SR's computational footprint (13B+ parameters, ~43GB VRAM, ~38s/inference) is substantial, limiting its practical deployability compared to faster baselines. The discussion on potential efficiency improvements (Section C.4) is preliminary and speculative. A more concrete analysis or preliminary results from, for example, a distilled VLM, would strengthen the paper's practical impact.

4. Minor Typos and Presentation: L273. The authors should carefully check the content.

### Questions
1. The sensitivity analysis in Table 5 uses simulated OCR errors. How does GLYPH-SR perform on real-world low-quality images where the initial OCR (providing S_TXT) is inherently noisy or incomplete? Can you provide results on a wild dataset with poor ground-truth OCR to demonstrate robustness?

2. Could you provide more details on the tuning of critical hyperparameters like the control scale s_CTRL and the CFG scale ω? Were they empirically tuned, and what are the observed trade-offs between text fidelity and image quality at different values? Are there failure modes associated with extreme values?

3. Have you conducted any experiments with a smaller or quantized VLM (e.g., a distilled version of LLaVA-NeXT) to reduce computational cost? If so, what was the corresponding drop in OCR F1 and perceptual scores? This would greatly inform practical applications.

4. The paper rightly notes the misalignment of traditional metrics (Fig. 7). To further substantiate the perceptual claims, have you considered or conducted a user study to quantitatively assess human preference between GLYPH-SR and key baselines regarding both overall image quality and text readability?

5. The evaluation focuses on standard Latin scripts. What are the prospects or any preliminary results for GLYPH-SR on multilingual text, complex scripts (e.g., Chinese, Arabic), or handwritten text? Does the current design have inherent limitations for such scenarios?

6. Based on the analysis of Figure 14, what specific architectural or training modifications (e.g., incorporating a text-region segmentation mask, adding a localization loss, or using a more robust text detector) do you envision could mitigate the issues of hallucination in non-text regions and incomplete enhancement of multiple text instances?

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
4

### Summary
This paper proposes GLYPH-SR, a vision–language guided diffusion framework for text image super-resolution, aiming to jointly optimize image perceptual quality and text legibility.
The core technical contributions include:
1. Bi-objective formulation and dual-axis protocol that treats SR as joint optimization of image and text fidelity.
2. Text-SR Fusion ControlNet, which fuses OCR-derived textual cues and image captions.
3. A ping-pong scheduler alternating text-centric and image-centric guidance during diffusion denoising.
4. A synthetic factorized corpus that decouples text and image degradation for targeted training.

### Strengths
1. The bi-objective view of SR (visual realism + text fidelity) is intuitive and important for practical use, addressing the neglected fact in most STISR works.
2. The TS-ControlNet + ping-pong scheduler combination is intuitive for the target of optimizing image perceptual quality and text legibility jointly.
3. The four-way partition synthetic corpus fits the claimed objective of joint optimization for training purpose.

### Weaknesses
1. Most baselines in experiments are not SR methods specialized for scene text image, except DiffTSR. In addition, methods like DiffTSR are not built for restoring a full scene text image, but for cropped image that only contains a single textline. The comparison could be unfair.

2. Despite most baselines were not built for scene text image, the proposed GLYPH-SR still can not outperform them consistently, even in terms of OCR accuracy.

3. As mentioned in Sec. C.3, the restoration performance rely heavily on the OCR result at the beginning of the procedure. Though strong VLM is applied, the OCR result could still be wrong under severe degradation, otherwise the super-resolution is unnecessary.

4. The dataset used for training and **evaluation** is not specifically built for image super-resolution. Even in the evaluation, the LR images seem to be generated by manually downsampling. The lack of real-world scenario in evaluation made it less convincing.

5. The OCR accuracy on LR/HR image is not reported in the tables, which made it harder to see the improvements.

### Questions
1. How was DiffTSR applied to this task? Were the full scene text images directly fed to DiffTSR, or fed after cropping?

2. Why is this paper named "GLYPH-SR" ? The Text-guidance in ControlNet only contains the recognized text and detected position. It seems that this work has nothing to do with glyph.

3. What is the OCR accuracy on the original image/downsampled(HR/LR) image?

4. Have the authors consideder applying other methods to balance the image and text guidance instead of the binary ping-pong policy? (e.g. through a dynamic learnable parameter)

### Soundness
2

### Presentation
2

### Contribution
2
