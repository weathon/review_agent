# Decoupling Global Structure and Local Refinement: Blueprint-Guided Scroll Generation with Direct Preference Optimization

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Existing methods for generating long scroll images, often fail to maintain global structural and stylistic consistency, resulting in artifacts like content repetition. To address this, we propose the Dual-Resolution Scroll Generation with Preference Optimization (DRSPO) framework. Our approach decouples global composition from local refinement by first generating a low-resolution (LR) blueprint to establish a coherent overall structure. This LR blueprint then guides a high-resolution (HR) feature to render fine-grained details. We further enhance generation quality by incorporating Direct Preference Optimization (DPO) at both stages, and we introduce a novel theoretical adaptation to apply preference tuning directly to the region-based generation process. Experimental results demonstrate that our method produces high-quality long scroll images with reasonable global structure and fine-grained details.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the task of generating long images using diffusion models.
It proposes a two-stage method: generating a low-resolution (LR) "blueprint" image, upsampling this blueprint into a high-resolution (HR) image. A key component is fine-tuning the models using Direct Preference Optimization (DPO). The proposed method (DRSPO) demonstrates superior results compared to previous methods like MultiDiffusion and SyncDiffusion.

### Strengths
This appears to be the first work to apply DPO to enhance the quality of long scroll image generation.

### Weaknesses
- I think the task of generating long scroll images is a somewhat solved problem. However, the paper compares its method only against relatively weak baselines.

- We may generate directly with state-of-the-art models like FLUX.1-dev (which is mentioned in the paper) or, generate a 512x2048 image with a FLUX or Stable Diffusion 3.5 and then using an off-the-shelf super-resolution model (e.g., ESRGAN). This alternative pipeline might produce comparable or even superior results.

- The validity of the evaluation metrics (HPS v2, PickScore, CLIP Score) is questionable, as these off-the-shelf models may not be reliable for evaluating long (1024x4096), non-square images.

### Questions
While FLUX is included in the quantitative results (Table 1), it is notably absent from all qualitative comparisons. What is the actual qualitative performance of FLUX on this task?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the Dual-Resolution Scroll Generation with Preference Optimization (DRSPO) framework, which decouples global composition and local detail refinement for long scroll image generation. DRSPO first generates a low-resolution (LR) global blueprint and then produces high-resolution (HR) details based on this overall structure. By doing so, it successfully generates images that are globally coherent while maintaining fine-grained local fidelity.

### Strengths
This paper's strengths are as follows.

(1) By employing a low-to-high-resolution generation strategy, the method successfully achieves both global structural consistency and local detail refinement.

(2) The paper improves generation quality for long scroll images by introducing Direct Preference Optimization (DPO).

(3) The application of DPO to high-resolution generation is theoretically supported (though with certain approximations), and the authors explicitly discuss the limitations of these approximations, linking them to future research directions.

### Weaknesses
This paper's weaknesses are as follows.

(1) Although the model is trained using DPO that optimizes for the Aesthetic Score, the same metric is also used for evaluation, effectively leading to reward hacking. Moreover, the method’s performance on this score is worse than existing methods, raising doubts about whether the proposed approach truly improves generation quality.

(2) Compared with MAD, the overall performance is similar or even inferior, despite MAD requiring no fine-tuning or DPO training. This raises concerns about the cost-effectiveness of the proposed method given its additional training overhead.

(3) In Table 1, the color distinction among “1st”, “2nd”, and “3rd” is difficult to see, which reduces the readability of the results.

### Questions
My questions about this paper are as follows.

(1) Have the authors tried fine-tuning directly on long scroll images rather than adapting pretrained models?

(2) How is the Aesthetic Score computed for long scroll images? Is it calculated locally and then aggregated, or is there a global evaluation?

(3) From the results in Table 2, the effect of DPO seems limited. Given that DPO requires additional training cost for diffusion model fine-tuning, one would expect more significant improvements. Why does DPO seem less effective than the control in this setting?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
- This paper propose the Dual-Resolution Scroll Generation (DRSPO) framework, which decouples the process by first creating a low-resolution blueprint for global structural coherence, then refining it with high-resolution features for local detail.

- To enhance quality, we integrate Direct Preference Optimization (DPO) at both generation stages and introduce a novel theoretical adaptation for applying preference tuning to region-based generation.

- Experimental results confirm that our method effectively produces high-quality long scroll images with consistent global structure and fine-grained details, overcoming issues like content repetition.

### Strengths
- The paper provides formulas, ablation experiments, and visualization results, though they are not very satisfactory.
- Compared with the FLUX model

### Weaknesses
The methods compared in this paper are already too outdated. Additionally, the FLUX model has not undergone specific optimization for such tasks, making the comparison unfair. In my view, this paper is not suitable for the ICLR conference. Furthermore, from the visualization results, it can be seen that the results here are not advanced, with numerous artifacts and glitches. The small images also fail to show more details clearly. Therefore, I believe this paper still requires further improvements. In particular, the method mentioned in the paper is no longer novel in the field of text-to-image generation.

### Questions
The methods compared in this paper are already too outdated. Additionally, the FLUX model has not undergone specific optimization for such tasks, making the comparison unfair. In my view, this paper is not suitable for the ICLR conference. Furthermore, from the visualization results, it can be seen that the results here are not advanced, with numerous artifacts and glitches. The small images also fail to show more details clearly. Therefore, I believe this paper still requires further improvements.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper discusses an approach for generating big images similar to MultiDiffusion but guided by a low resolution image. "Direct Preference Optimization" is involved.

### Strengths
1.	The fundamental idea looks correct – generate a low-resolution image and then diffuse at high resolution may result in larger results.
2.	A dataset is explicitly collected to align with the goal.

### Weaknesses
1.	The idea of generating low resolution image and then diffusing again is extensively discussed even long before MultiDiffusion and similar works. This part shouldn’t be seen as a contribution of this work.
2.	The DPO part is formulated with high verbosity but in essence they can be achieved as a simple target like some common score guidance like [github.com/vicgalle/stable-diffusion-aesthetic-gradients]. I also do not find Eq 13-19 as solid contributions of this work.
3.	The experiments seem to compare to wrong targets. This paper is a cascaded image generator and it should at least compare to multi stage generating. For example, generate low resolution images and then diffuse it again using MultiDiffusion at some denoising level at high resolution. And also other native high resolution generators like opensource PixArt family (and even close sourced Flux pro 4k etc) as well as some other cascaded generators like stable cascade. The current MultiDiffusion results look misleading to me.
4.	Why use canny image? Why not directly use low resolution image for other “upscaling” types of upscaling signals?

### Questions
See weaknesses

### Soundness
2

### Presentation
3

### Contribution
2
