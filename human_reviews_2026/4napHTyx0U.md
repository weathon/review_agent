# EDIF: Editing via Dynamic Interactive Tuning with Feedback

- Decision: Reject
- Scores: 4, 6, 0, 4, 4

## Abstract
Although text-guided image editing (TIE) has advanced rapidly, most prior works remain object-centric and rely on attention maps or masks to localize and modify specific objects. In this paper, we propose a method of Editing via Dynamic Interactive Tuning (EDIF) that adaptively trades off source-image structure and instruction fidelity in difficult scene-centric editing settings. Unlike object editing, scene-centric editing is challenging because the target cannot be clearly localized, and edits need to preserve global structure. To cope with the limitation of TIE systems that typically use a unified conditioning signal and ignore the block-wise variation in the internal behavior of the model, we show that inside the model, the source-image condition and the text-prompt embedding act with layer-dependent directions and strengths. We also demonstrate both empirically and the oretically that the editing state can be diagnosed using the source image signal-to-noise ratio and VLM logits, which indicate whether the edited image faithfully reflects the intended editing prompt. By constructing a Pareto line between these two objectives, EDIF adaptively modulates the source-image and editing-text conditions, guiding each denoising step to stay close to this line for balanced optimization. Extensive experiments on ImgEdit, EmuEdit-Bench, and Places365 show that EDIF achieves state-of-the-art performance in various scene-editing scenarios, including indoor and outdoor environments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents EDIF, a method for scene-centric text-guided image editing that dynamically balances the influence of the source image and the textual editing instruction during diffusion-based generation. Unlike traditional object-centric editing, EDIF targets more complex scene-level edits where global coherence and spatial structure must be preserved. The authors observe that image and text conditions have layer-dependent effects within the diffusion model, and propose to diagnose the editing state via source-image SNR and VLM logits. EDIF constructs a Pareto trade-off curve between fidelity to the original image and adherence to the textual edit, adaptively modulating the denoising process to stay close to this optimal balance.

### Strengths
* Introduces an adaptive modulation mechanism using SNR and VLM feedback to control the editing strength during diffusion.

* Provides both empirical and theoretical analysis on layer-wise condition influence and editing diagnostics.

* Experimental results on multiple benchmarks indicate reasonable quantitative and qualitative improvements.

* Conceptually clear in balancing fidelity vs. edit strength via a Pareto trade-off framework.

### Weaknesses
* The idea of adaptive conditioning or feedback tuning in diffusion-based editing is conceptually similar to prior guidance or attention-control methods.

* The paper’s focus on “scene-centric editing” is somewhat narrow and not well contextualized with real-world applications.

* Poor presentation quality: Figures are low-resolution, making qualitative evaluation difficult. In addition, formatting inconsistencies and citation errors (e.g., line 53) deviate from the ICLR template, giving the impression of a hastily prepared submission.

* Overall contribution-to-effort ratio is moderate; the framework is incremental rather than groundbreaking.

### Questions
* Further clarification on the novelty is appreciated.
* The paper claims that source-image and text conditions have layer-dependent effects. Can the authors provide quantitative or visual evidence (e.g., attention maps, activation statistics) to substantiate this observation?
* The concept of “scene-centric editing” is not clearly defined. What specific criteria distinguish it from conventional object-centric editing, and how does EDIF handle transitions between the two types?
* Since the paper targets practical scene editing, could the authors include comparisons with strong or commercial baselines (e.g., DALLE-3, Firefly, EmuEdit) to demonstrate the competitiveness and real-world applicability of EDIF?

### Soundness
2

### Presentation
1

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
This paper proposes a novel text-guided image editing method for scene-centric settings, called Editing via Dynamic Interactive Tuning (EDIF). Unlike existing methods, EDIF constructs a Pareto line between the source image to edited image ratio ($SNR_{src}$) and VLM logits, aiming to strike a balance between preserving the structure of the source image and accurately reflecting the intended edits. The editing process of EDIF is informed by an ablation study that analyzes the block-wise influence of conditions by zeroing out either the image condition or the text condition in specific blocks. This work provides new insights into training-free image editing, and the experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed method is novel and interesting, offering new insights into training-free image editing.

2. The results demonstrate strong performance in scene-centric editing.

### Weaknesses
1. Some parts of the presentation are confusing. In line 267, it states, "we first obtain the predicted clean image." Does this imply that EDIF requires one complete editing iteration before adjusting the edits in subsequent iterations? However, other descriptions suggest that EDIF can be completed during denoising iterations. Could the authors clarify this point?

2. The number of adjustments needed during the editing process is not discussed. What is the typical editing time for each image?

3. Some content requires further clarification. The process for constructing an effective Pareto line and determining when the editing is satisfactorily completed is unclear. Additionally, it is not specified which blocks should be adjusted during the editing process.

4. There are typos present, such as a citation error in line 053 and "Pareto frint" in line 249. Some citations also appear incorrect. For example, the citation "Xu et al., 2023a" in line 307 cannot be found in the reference list.

### Questions
1. The prompt decomposition transforms free-form prompts into key concept prompts by incorporating keywords such as "add" or "make." However, what about cases where the scene editing only requires the removal of elements?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper does not follow ICLR's official template. I suggest a desk rejection.

### Strengths
/

### Weaknesses
/

### Questions
/

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focused on scene-centric editing setting, and proposed Editing via Dynamic Interactive Tuning (EDIT) to achieve adaptive trade-off between source-image structure and instruction fidelity. This paper pointed out the block-wise variation inside the diffusion models, i.e., both the image condition and text condition functions independently. This paper used the signal-to-noise ratio and VLM logits to diagnose the editing state, and then using them to adaptively modulate the source-image and editing-text condition to achieve balanced editing results.

### Strengths
1. This paper focused on the scene-centric editing, which is more challenge compared with the object-centric editing in existing works and is an important research direction.
2. This work proposes a reasonable method to achieve dual optimization of source preservation and prompt fidelity

### Weaknesses
1. The writing and organization of this paper are inadequate and require substantial revision. There’re many typos such as the “?” citation in line 53.
2. The method is plug-and-play, but the fact that it was only tested on Kontext weakens the generalizability of the study. Can this method be used on other image editing base models besides Kontext?
3. The quantitative experimental results of this work did not show significant improvement.
4. User studies need to include more users and explain the specific rating criteria.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes EDIF to address the trade-off between source-image structure preservation and instruction fidelity in scene-centric text-guided image editing. The author shows that the source image condition and the embedding act with layer-dependent directions. Therefore, this paper uses source SNR and VLM logits to diagnose the editing state. Extensive experiments on three benchmarks demonstrate the effectiveness of EDIF.

### Strengths
1. This work is clearly expressed and easy to understand.
2. This work provides extensive experimental results and comprehensive comparisons with multiple baselines.

### Weaknesses
1. The increased time consumption of EDIF compared to baseline was not mentioned.
2. The function differences between different blocks are largely based on empirical observations, which lack theoretical support. Can these function differences be extended to other image editing methods?

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
