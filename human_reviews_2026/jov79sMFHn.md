# Nano3D: A Training-Free Approach for Efficient 3D Editing Without Masks

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
3D object editing is essential for interactive content creation in gaming, animation, and robotics, yet current approaches remain inefficient, inconsistent, and often fail to preserve unedited regions. Most methods rely on editing multi-view renderings followed by reconstruction, which introduces artifacts and limits practicality. To address these challenges, we propose \textbf{Nano3D}, a training-free framework for precise and coherent 3D object editing without masks. Nano3D integrates FlowEdit into TRELLIS to perform localized edits guided by front-view renderings, and further introduces region-aware merging strategies, Voxel/Slat-Merge, which adaptively preserve structural fidelity by ensuring consistency between edited and unedited areas. Experiments demonstrate that Nano3D achieves superior 3D consistency and visual quality compared with existing methods. Based on this framework, we construct the first large-scale 3D editing datasets \textbf{Nano3D-Edit-100k}, which contains over 100,000 high-quality 3D editing pairs. This work addresses long-standing challenges in both algorithm design and data availability, significantly improving the generality and reliability of 3D editing, and laying the groundwork for the development of feed-forward 3D editing models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Motivated by FlowEdit and TRELLIS, the authors propose Nano3D, a training-free framework for precise and coherent 3D object editing without requiring masks. The method first uses a closed-source image generation model to edit the front view of the input image (e.g., removing, adding, or replacing objects). It then performs 3D editing on both geometry and appearance, and leverages an XOR operation between sparse structures to effectively filter the regions that require editing. This pipeline ultimately achieves relatively high-fidelity 3D edits.

### Strengths
The paper is very clearly written, and the method is easy to understand (except for Section 4.2).

The visual results are impressive, significantly enhancing the credibility of Nano3D.

The authors curate the first large-scale 3D editing dataset, Nano3D-Edit-100k, which will be valuable to the research community.

The ablation studies thoroughly demonstrate the role of each component, and the contribution of each module is both intuitive and theoretically well-motivated.

### Weaknesses
The method relies on FlowEdit, which takes the front-view edited image as input. Does this imply that editing content on the back side of an object is not supported?

The authors use a fixed threshold τ = 100 to generate the edit mask. However, different editing tasks involve regions of vastly different scales—for example, a subtle edit like replacing human ears with elf ears should likely use a much smaller τ. This suggests that τ is a task-dependent hyperparameter requiring manual tuning, which limits practical usability.

The mask generation relies on geometry-induced changes in voxel occupancy. Consequently, the method cannot handle pure texture edits that preserve geometry (e.g., changing the color or pattern of an object without altering its shape).

Possibly due to my limited familiarity with FlowEdit, I found the descriptions in Sections 4.2 and 4.4—regarding the integration of FlowEdit and TRELLIS—to be overly brief, which created some difficulty in fully understanding the technical details.

### Questions
Please refer to the Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Nano3D is a training-free framework that enables precise and coherent 3D object editing without masks. It combines FlowEdit with TRELLIS for localized edits and introduces Voxel/Slat-Merge strategies to maintain structural consistency between edited and unedited regions.
Additionally, the authors present Nano3DEdit-100k, a large-scale dataset of over 100,000 high-quality 3D editing pairs, laying the foundation for future feed-forward 3D editing models.

### Strengths
[1] The paper clearly communicates its main claims and presents a valuable contribution in the form of zero-shot 3D editing.

[2] The quality of the presented results is also qualitatively impressive.

### Weaknesses
1. The proposed system shows a strong dependency on **FlowEdit** and **Trellis**, but the paper fails to clearly convey the fundamental nature of the models being utilized.  
   - In this system, *FlowEdit* functions as an **image editor**. What matters is not merely that FlowEdit is used, but rather that **an image editing module** is employed — and its inputs and outputs should be clearly formulated.  
   - The key insight readers should gain from this paper is the use of a **3D representation**, not simply the use of Trellis. Regardless of whether Trellis or another model is used, it is crucial to **define its input, output, and equations** clearly. In fact, in Section 3.2, there is only a brief mention of `z_i`, with no explanation of what `p_i` or `C` represent.  

2. In Section 4.3, the explanation of **Voxel-Merge** is difficult to follow, as it assumes readers already know all the terms and omits formal definitions, which significantly reduces readability. Specifically, what does `s(i)` refer to?  

3. **Voxel-Merge** is a proposed method, yet the explanation provided in Lines 203–210 is insufficient.  

4. The claim that *"We obtain voxels that are fully consistent before and after editing"* is not a factual statement but rather an assertion. Isn’t the method still **dependent on thresholding**?  

5. Overall, the paper gives the impression of **assembling multiple existing models** to perform a single task. I would encourage the authors to describe their ideas in a **more generalized and conceptual form**. As it stands, the paper reads more like a **technical report of an empirical study** than a research paper presenting a novel framework.

### Questions
My questions are in the weakness

### Soundness
1

### Presentation
3

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
This paper introduces a training-free 3D editing framework based on current powerful generative models like TRELLIS and Nano-Banana. In contrast to prior methods relying on 2D multi-view editing, they adapt FlowEdit into TRELLIS thus mitigating the problem of multi-view inconsistency to some extent. To enhance consistency, the authors propose Voxel/Slat-Merge strategy to preserve unedited regions via connectivity analysis. The framework enables a pipeline for constructing Nano3D-Edit-100k, the first large-scale 3D editing dataset. The paper provides evaluations on qualitative visuals, quantitative metrics (e.g., Chamfer Distance, DINO-I, FID), and user studies to demonstrate superior performance over baselines like Tailor3D, Vox-E, and TRELLIS. The work aims to advance 3D editing toward feedforward models, mirroring the evolution in 2D image editing.

### Strengths
1. **Stable and efficient data generation pipeline:** It effectively adapts FlowEdit from 2D to 3D editing within TRELLIS, demonstrating that pretrained generative priors can enable training-free, inversion-free edits. 
2. **Significant Dataset Contribution:** Nano3D-Edit-100k fills a critical gap in 3D editing data, with automated construction using VLMs for diverse instructions and quality filtering. The comparison highlights the semantic alignment, making it a valuable resource for future supervised models.
3. **Comprehensive and extensive evaluation.**

### Weaknesses
1.  **Limited novelty**: it is appreciated to smartly integrate pretrained models into such framework. but the novelty is really limited, which introduces some concerns about the contribution to the research community.
2. **Limited edit operations**: now it only supports static add/removal/replacement, can’t support deformation and so on.
3.  **The claim of “without mask” is not convincing**. As in the voxel/slat-merging stage, it requires a difference map and a threshold to preserve original structure. This is actually still a mask, which could be easily got because the property of supported editing operations.

### Questions
1. Generally I like the whole pipeline. But why is the framework called “Nano3D”? What is the motivation behind this? I can’t really decode the meaning behind this. And I don’t think it’s appropriate to include Nano because of Nano-Banana.
2. How sensitive is Nano3D to the choice of front-view rendering? Could incorporating multiple views in FlowEdit guidance improve robustness for complex occlusions or asymmetric objects?
3. Nano3D-Edit-100k balances 10 categories, but what is the distribution of edit types (add/remove/replace) and instruction complexities? Could releasing per-category metrics help assess dataset quality?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Nano3D, a training-free 3D object editing pipeline that plugs FlowEdit into TRELLIS’s first-stage voxel generator and then preserves untouched content via region-aware merging at both voxel and SLAT levels (Voxel-Merge / SLat-Merge). Nano3D reports the best quantitative performance among baselines. A user study (n = 50) also found that participants prefer Nano3D across prompt alignment, visual quality, and shape preservation.  Lastly, the paper further introduces Nano3D-Edit-100k, a 100k-pair dataset built with an automated pipeline.

### Strengths
* **Simple and effective approach**: This paper mainly focuses on keeping non-edited regions, and the proposed approach is simple but effective. Compared to directly extending FlowEdit to TRELLIS, analyzing issues (such as changing connected regions) makes sense, and the Voxel & SLat merge helps to resolve this.
* **Impressive empirical results**: Nano3D leads on CD/DINO-I/FID (Table 1) and is preferred in the user study (Table 2); ablations show Voxel-Merge and SLat-Merge contribute additively. From the qualitative results, the non-edited parts are clearly maintained without harming the generation quality.
* **3D Editing Dataset**: They specify a practical 100k-pair pipeline, achiving better alignment than 3D-Alpaca. Such a dataset and tool may be helpful for future work.
* The overall writing and illustrations are clear to follow and understand.

### Weaknesses
* The method uses "input 3D object’s front view" as a source condition. Thus, for the edits on the unseen part, it may lead to problems. In addition, while merging helps preserve the rest, the 3D editing quality may hinge on the quality of that 2D edit. It is good to see that the paper itself notes many failures originate in the 2D editing stage, but there’s no controlled study of viewpoint sensitivity (e.g., edits specified from non-frontal views or multi-view edits). A better analysis of the method's robustness will help to understand the boundary of the method.
* The method highly relies on the Trellis representation, which makes it hard to extend. Specifically, the input must be the object style of Trellis and is generalizable to complex scenes or objects.
* The compared baselines are too old and narrow. For example, Vox-E is a work from ICCV2023. For an informative comparison, more up-to-date works should be considered, such as PrEditor3D (CVPR 2025) or recent works.

### Questions
* In Figure 5, the scale of the rendered object seems to change compared with Trellis (smaller). Is it because of the method or the plotting issue? In addition, why does Trellis show some degeneration on the objects compared with Nano3D?

### Soundness
3

### Presentation
3

### Contribution
3
