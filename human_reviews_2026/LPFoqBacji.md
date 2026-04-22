# RIFLE: Removal of Image Flicker-Banding via Latent Diffusion Enhancement

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Capturing screens is now routine in our everyday lives. But the photographs of emissive displays are often influenced by the flicker-banding (FB), which are alternating bright–dark stripes that arise from temporal aliasing between a camera’s rolling-shutter readout and the display’s brightness modulation. Unlike moiré degradation, which has been extensively studied, the FB remains underexplored despite its frequent and severe impact on readability and perceived quality. We formulate FB removal as a dedicated restoration task and introduce Removal of Image Flicker-Banding via Latent Diffusion Enhancement, RIFLE, a diffusion-based framework designed to remove FB while preserving fine details. We propose the flicker-banding prior estimator (FPE) that predicts key banding attributes and injects it into the restoration network. Additionally, Masked Loss (ML) is proposed to concentrate supervision on banded regions without sacrificing global fidelity. To overcome data scarcity, we provide a simulation pipeline that synthesizes FB in the luminance domain with stochastic jitter in banding angle, banding spacing, and banding width. Feathered boundaries and sensor noise are also applied for a more realistic simulation. For evaluation, we curate a paired real-world FB dataset with pixel-aligned banding-free references captured via long exposure. Across quantitative metrics and visual comparisons on our real-world dataset, RIFLE consistently outperforms recent image reconstruction baselines from mild to severe flicker-banding. To the best of our knowledge, it is the first work to research the simulation and removal of FB. Our work establishes a great foundation for subsequent research in both the dataset construction and the removal model design. Our dataset and code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The work tackles flicker-banding, alternating bright/dark stripes when photographing emissive displays (OLED/LED/CRT, etc.) with rolling-shutter cameras due to temporal aliasing with PWM/scan refresh. 

To overcome the lack of paired data, the authors (i) devise a simulation pipeline that injects FB in the luminance channel with stochastic jitter in angle, spacing, width, feathered boundaries, and sensor noise; and (ii) collect a real-world test set by capturing short-exposure vs. long-exposure pairs from fixed viewpoints, aligned at pixel level.

RIFLE is a one-step latent diffusion restorer (PiSA-SR backbone) augmented with: a Flicker-Banding Prior Estimator (FPE) (ResNet) that predicts band attributes (angle/spacing/width) and injects them into the diffusion network; a Masked Loss (ML) that weights supervision toward banded regions while preserving background fidelity

### Strengths
This work tackles a timely and practical problem that is clearly distinct from moiré and common in everyday captures, and it is the first I have seen to set out a coherent task definition, dataset, and baseline specifically for flicker banding. The luminance channel simulation is thoughtfully engineered, with stochastic jitter and feathered transitions, so the artifacts closely match real cases and the pipeline scales well. The approach is task aware: a lightweight prior estimator (FPE) plus a masked loss is a simple and effective recipe, and the ablations show their complementary roles. Results consistently lead on a real world benchmark, visibly reducing banding under both mild and severe conditions while remaining competitive with strong diffusion and transformer baselines. Overall, this is a well rounded and complete paper, and if the authors release the data and code it would likely be helpful to the community.

### Weaknesses
1. The method feels like a careful integration of known pieces (one-step diffusion, masked loss, small prior estimator) without a single, clearly novel idea specific to flicker-banding. But that doesn’t mean the paper is without value. I think solving a worthwhile problem with a simple approace and doing it with a complete, well-structured pipelin, is still commendable. So I’d like to see how other reviewers evaluate the problem definition.
2. Assumptions like luminance-only masking and mostly straight stripes may fail with curved or multi-frequency banding, subpixel-dependent chroma effects, or high-nit LED walls. Show out-of-distribution tests and discuss failure modes. Can the entire process now handle more of these related small issues?
3. The real-world benchmark is small and not diverse enough in displays, PWM frequencies, camera models, shutter speeds, and ISO. Add stratified tests and more scenes to show breadth.

### Questions
See weaknesses. And I admit that I am not an expert on this topic, so I would like to see other reviewers' opinions on this topic.

### Soundness
3

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
4

### Summary
This paper introduces RIFLE, a diffusion-based method designed to alleviate flicker-banding (FB) distortions in individual images. These visual anomalies originate from temporal aliasing, a consequence of the interplay between a camera's rolling shutter and the brightness modulation of displays. The authors make three principal contributions: a methodology for creating synthetic FB-corrupted images for training purposes, a fresh, paired real-world dataset for evaluating performance, and the RIFLE model itself. RIFLE is built based on the PiSA-SR baseline by incorporating a Flicker-banding Prior Estimator (FPE) and a Masked Loss (ML) function.

### Strengths
* **Valuable Problem:** The paper deals with flicker-banding removal, which is a real and under-explored issue in computational photography. This actually affects everyday users.
* **Novel Dataset:** The authors put together what they say is the first paired real-world dataset for FB removal. That’s a big deal for the community—it gives everyone a solid benchmark to build on.
* **Realistic Simulation Pipeline:** They designed a simulation pipeline that creates realistic FB artifacts by working with the luminance channel and adding random jitter. This is key for training data-driven models.

### Weaknesses
The work takes on an important, often overlooked problem and brings a useful new dataset to the table. But RIFLE, the approach the authors introduce, doesn’t really break new ground, and the experiments don’t do a good job proving it works. When you look at the breakdowns, the main piece—FPE—barely makes a difference. Plus, the numbers aren’t shown clearly or consistently, which just makes it harder to figure out how well the method actually performs.

**Major Weaknesses:**
*   **Restricted Methodological Originality and Unproductive Elements:** The RIFLE model is emphasized as a substantial contribution; however, its novelty compared to the PiSA-SR baseline is minimal. The method consists of two enhancements:

    *   **Masked Loss (ML):** This involves a conventional weighted loss function that assigns a greater weight to the areas affected by banding. Considering that the background (non-banded) area is already noticeably similar to the ground truth, this design choice appears straightforward and simple, rather than a significant innovation.
    *   **Flicker-banding Prior Estimator (FPE):** This is the main component, a ResNet-15 architecture trained to predict three banding attributes (width, spacing, angle). However, the paper's component analysis shows the near-ineffectiveness of this element. The baseline (PiSA-SR) achieves a PSNR of 22.12. The addition of the FPE yields a minor improvement of 0.03 dB (PSNR of 22.15). Conversely, incorporating the simple ML alone results in a PSNR of 22.28. The complete RIFLE model (ML+FPE) attains a PSNR of 22.30, suggesting that the entire 0.18 dB improvement over the baseline is due to the uncomplicated masked loss, rather than the prior estimator. This finding directly contradicts the paper's central technical claim.

*   **Contradictory and Vague Quantitative Results:** A notable inconsistency exists between the numerical results presented in Table 1 and Table 2.

    *   Table 1 presents results on the "real-world flickering-banding datasets." Here, RIFLE achieves a PSNR of 20.66, whereas the PiSA-SR baseline achieves a PSNR of 20.57.
    *   Table 2 presents the ablation study, which, according to Figure 7's caption, is also performed "on our Real-Flicker dataset." Here, RIFLE (as "ML+FPE") achieves a PSNR of 22.30, and the baseline (PiSA-SR) achieves a PSNR of 22.12.

    These two tables, supposedly reporting on the same dataset, exhibit a significant and unexplained difference of approximately 1.6-1.7 dB for the same models. It is unclear whether Table 2 uses a smaller subset of the data or if a more profound methodological mistake exists. This discrepancy makes it impossible to assess the model's true capabilities or the validity of the component analysis.

*   **Inherently Limited Method Design:** The simulation pipeline's crucial observation involves applying banding exclusively to the luminance (Y) channel, based on analysis in Appendix C. While clever, this design choice introduces a critical constraint. The authors acknowledge in their limitations (Appendix E) that the model is primarily intended for "gray-scale flicker" and struggles with colored stripes. This is a major disadvantage, as the paper's own survey in Appendix A acknowledges that technologies like OLEDs and projectors do generate chromatic artifacts.

    Furthermore, the FPE is designed to predict a single banding angle. This design lacks the ability to handle the complex or curved banding patterns prevalent in real-world captures, a limitation also conceded by the authors. As a result, the method is ill-suited for many real-world instances of the very problem it claims to solve.
* Weak Ablation Studies and Comparison: no runtime, model complexity comparison.

**Minor Weaknesses:**
*   **Exaggerated "First Work" Assertion:** The paper repeatedly claims to be the "first work" on FB simulation and removal. This claim is debatable. While it may be the first to coin the term "Flicker-Banding" and leverage a diffusion model, the underlying phenomenon is a known artifact resulting from rolling shutters interacting with modulated light sources. The related work section is brief and quickly dismisses comparable work, such as single-image deflicker techniques, which also attempt to eliminate flicker-induced spatial distortions from a single frame.

### Questions
* Can authors explain why do Table 1 and Table 2 look so different even both claim to use the real-world dataset? As mentioned above, the PSNR numbers in Table 2 are higher by about 1.6 or 1.7 dB.
* Aslo, as stated above, FPE is reported to improve performance by 0.03dB, while Masked Loss gives bigger jump of 0.16dB. Can you explain why FPE is important for RIFLE model if its effect is so small?
* Can authors provide more comparison of runtime complexity, I doubt that this will consume a lot of time, which is not really phones friendly, which contradict with the original purpose of removing flickers when capturing by phone.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates flicker-banding artifacts (FB) in screen photography, a type of image degradation. This degradation, characterized by bright and dark stripes, arises from the temporal aliasing between display brightness modulation and the camera's rolling shutter mechanism. To address this problem, the paper proposes an FB simulation data generation pipeline and a latent diffusion-based image restoration framework RIFLE.

### Strengths
The paper proposes a reasonable data generation pipeline that successfully generated a corresponding dataset. Furthermore, an effective method for removing FB was introduced.

### Weaknesses
1. Regarding the dataset, an analysis would be beneficial. Given that 400+ paired patches constitute a relatively small scale, does this dataset adequately cover all possible scenarios where flicker-banding might occur?
2. Currently, all the visualization cases you've presented are screen captures. Screen captures are typically affected by a complex combination of moiré patterns and flicker artifacts. While current moiré removal methods aim to directly produce clean, high-quality images. However, RIFLE, based on your description, only removes flicker artifacts. Consequently, the visualization results still appear to be affected by moiré patterns. This raises the question: Does this imply that your method is a 'degraded version' of moiré removal methods? Furthermore, for an image degraded by screen capture, wouldn't applying a moiré removal method be more appropriate? You lack examples demonstrating scenarios where only flicker artifacts are present without moiré interference. Simply showcasing screen-capture scenarios is insufficient to fully demonstrate the necessity and unique value of your method.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes RIFLE (Removal of Image Flicker-Banding via Latent Diffusion Enhancement), a diffusion-based method to remove flicker-banding (FB)—alternating bright-dark stripes caused by temporal aliasing between camera rolling-shutter and screen brightness modulation. Unlike moiré patterns, FB remains underexplored despite its significant impact on image readability. To address the lack of datasets, the authors create a realistic simulation pipeline to synthesize FB artifacts and construct a paired real-world dataset for rigorous evaluation. RIFLE introduces two novel components: a Flicker-Banding Prior Estimator (FPE) that predicts banding parameters, and a Masked Loss (ML) targeting restoration in banded regions. Experimental results demonstrate that RIFLE outperforms recent image reconstruction baselines in both quantitative metrics and visual quality on real-world FB datasets, making it the first comprehensive attempt at tackling FB using neural networks

### Strengths
- Problem framing. Clearly formulates FB removal as distinct from moiré and global deflicker, with a concise physical explanation and taxonomy of patterns.
- Real‑world benchmark: paired captures with short/long exposure and pixel‑level alignment; evaluation limited to screen regions to avoid metric saturation
- Well‑motivated masked loss focusing on banded areas and a lightweight prior estimator (FPE) that predicts angle/spacing/width and conditions the diffusion network.

### Weaknesses
- Benchmark scale and diversity. The real test set is small and limited to static screens; this constrains claims about generalization across displays (PWM frequencies/duty cycles/scan schemes), camera models, and capture settings.
- Baselines. Comparisons are to general reconstruction/editing methods (MAT, InvSR, PiSA‑SR, Step1X). Stronger task‑proximity baselines are missing: (i) simple frequency‑domain notch filtering aligned to the estimated banding angle; (ii) line‑pattern destriping/deraining networks; (iii) rolling‑shutter correction plus tone‑mapping heuristics.
- Evaluation scope. Since FB impairs readability, an OCR‑based metric (accuracy on cropped text regions) or a user study on legibility would be very persuasive. The current reference metrics can be insensitive when the non‑banded regions dominate (the paper notes saturation; Sec. 5.2), yet no downstream/readability metric is provided.

### Questions
Simulation assumptions. Banding is applied only to luminance, with justification deferred to the supplement. Some PWM/scan schemes can induce color‑dependent artifacts (sub‑field driving). Please quantify the luminance‑only assumption on real data, and report a domain‑gap analysis between simulated and real FB.

### Soundness
3

### Presentation
2

### Contribution
2
