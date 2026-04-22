# Real-VAS: a Realworld Video Amodal Segmentation dataset

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Amodal video object segmentation is fundamentally limited by the absence of datasets that combine real-world complexity with precise ground-truth annotations. To address this, we present Real Video Amodal Segmentation (Real-VAS), a new large-scale, zero-shot evaluation dataset. We introduce a novel data generation pipeline that composites two clips of real-world video, enabling the creation of pixel-perfect amodal ground truth without relying on human estimation or expensive 3D reconstruction. Our dataset is structured into two challenging scenarios: dynamic Occlusion, created by compositing moving objects, and a unique Container category featuring complex, physically constrained interactions. These container scenarios—On Surface Containment, Articulated Containment, and Mobile Containment—allow us to generate precise ground truth by simulating an object's motion based on its container's tracked transformation. As a result, Real-VAS provides a diverse and challenging benchmark for evaluating amodal segmentation models on realistic video with the precision of synthetic data. The dataset and our generation code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents Real-VAS, a large-scale zero-shot evaluation dataset for video amodal segmentation, aiming to solve the key limitations of existing datasets, such as the sim2real gap in synthetic data and inaccurate human-estimated annotations in real-world data. It develops two automated data generation pipelines to produce pixel-perfect ground truth without relying on expensive 3D reconstruction or manual annotation. The dataset includes novel Containment scenarios and adopts three metrics (mIoU_fo, mIoU_ffo, mIoU_occ) to evaluate SOTA models, demonstrating its effectiveness as a challenging VAS benchmark.

### Strengths
1. The dataset generation method combines real-world video clips with automated ground-truth generation. It bridges the sim2real gap of synthetic datasets and avoids the inaccuracy and high cost of manual annotation for real-world data .

2. The paper proposes mIoU_fo and mIoU_ffo to solve the misleading issue of standard mIoU, making VAS model evaluation more accurate.

3. The Real-VAS dataset contributes to VAS research.

### Weaknesses
1. Limited Containment scenario coverage: The Containment pipeline relies on a "snugly fit" physical constraint, restricting scenario diversity.

2. Incomplete related work: The paper omits an earlier paper [1] that focuses on video amodal segmentation and was published prior to all the methods cited in its related work section. 
In the related work section, the authors only discuss their contributions related to synthetic datasets and fail to mention this earlier paper in the method part. This undermines the comprehensiveness of the literature review.

3. Data generation heavily depends on tools like SAM2 (for segmentation) and CoTracker3 (for tracking); errors in these tools may propagate to the dataset’s ground truth, reducing its reliability

[1] Self-supervised amodal video object segmentation.

### Questions
1. In the Occlusion pipeline of Real-VAS, a mean motion score S (calculated via CoTracker3) and a predefined threshold $\tau$ are used to filter dynamic occluders and occludees. How was the specific value of threshold $\tau$ determined, and were ablation studies conducted to verify its rationality in ensuring both dataset quality and scenario diversity? 

2. What technical challenges need to be addressed to extend the Containment pipeline to such loose interaction scenarios, and are there any preliminary solutions?

### Soundness
2

### Presentation
2

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
This paper introduces the Real Video Amodal Segmentation (Real-VAS) dataset, designed to balance data collection scale with real-world fidelity. It also presents two data generation pipelines: an occlusion pipeline, which blends two video clips and leverages Language Segment-Anything and SAM 2 to construct occlusion scenarios, and a containment pipeline, which uses SAM 2 and CoTracker3 to track objects before and after occlusion. Evaluations on Real-VAS show that video-based approaches outperform image-based ones, and that training on similar data could benefit performance.

### Strengths
1. The integration of multiple classic and state-of-the-art computer vision algorithms and models to enhance real-world fidelity provides a promising direction for addressing realistic data scarcity in the amodal segmentation domain.
2. The dataset and code will be open-sourced.

### Weaknesses
This paper should be rejected due to two key weaknesses:
1. **The claim of "pixel-perfect amodal ground truth for real-world video" is too weak**
   - The occlusion pipeline uses soft alpha blending, which makes the occluder semi-transparent. This is unrealistic and not truly *amodal*, as amodal segmentation requires full occlusion.
   - The containment pipeline, which applies the container's transformation (tracked by CoTracker3) to the occludee, does not produce pixel-perfect ground truth either.
2. **The evaluation is limited**
   - The evaluation experiments provide no novel insights. Simply showing that video models perform well on video domains, or that models trained on containment data perform well on containment domains, offers little value to the community.  
   - Some methods perform very poorly on Real-VAS, but the paper does not analyze why or how to improve them. Even for a dataset paper, there should be evidence showing how this dataset can inspire new approaches and advance amodal segmentation by revealing new insights.
   - Several standard models, such as [PCNet-M (CVPR 2020)](https://xiaohangzhan.github.io/projects/deocclusion/) and [AISFormer (BMVC 2022)](https://uark-aicv.github.io/AISFormer/), are missing from the evaluation. Including them would better demonstrate the dataset's reliability as a zero-shot amodal segmentation benchmark.

Things to improve the paper that did not impact the score:
1. **Figure 1:** Use the same example throughout the occlusion pipeline for better clarity.  
2. **Table 1:** It would be helpful to include all related datasets (e.g., TAO-Amodal) and highlight key differences such as *bbox vs mask* to provide a clearer comparison supporting the claim of being large-scale.  
3. **Table 5:**  
   - Ensure consistent capitalization between "mIOU" and "mIoU."
   - Add citations for each method.  
   - Clearly separate image-based and video-based methods.

### Questions
One key question the authors need to address:
* The arXiv paper ["Track Anything Behind Everything: Zero-shot Amodal Video Object Segmentation"](https://arxiv.org/abs/2411.19210) was explicitly **cited** in the *Related Work: Amodal Segmentation Methods* section and referred to multiple times throughout the paper for its method **TABE**, but its dataset, **TABE-51**, was intentionally omitted.  
* For example:  
  * This paper's **Figure 4** example and that paper's **Figure 7** (bottom-left) example are almost identical - two people talking in front of a building.  
  * This paper's **Figure 1** alpha-blend example and that paper's **Figure 3** example are almost identical - a person walking through a door.  
  * This paper's **Table 2** is exactly the same as that paper's **Table 2**.

What is the difference between this work and TABE-51? Is this work built on top of it?

Other questions:
- More details about the dataset: e.g., how many samples are from in-house sources and how many are web-crawled? What is the size of the containment and occlusion subsets, respectively?  
- As a data generation pipeline designed for scalability, what are the inference speed and cost to produce a single data point?

### Soundness
2

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
3

### Summary
This paper introduces Real-VAS, a real-world video amodal segmentation dataset for zero-shot evaluation. It generates pixel-perfect ground truth using automated compositing and container-tracking pipelines, covering dynamic occlusion and containment scenarios. Real-VAS combines realism and precision, providing a challenging benchmark for testing physical reasoning and occlusion understanding.

### Strengths
The paper demonstrates strong originality by introducing a novel real-world video amodal segmentation dataset (Real-VAS) that overcomes the long-standing gap between synthetic precision and real-world realism. Its quality is supported by a well-designed automated pipeline that produces pixel-perfect annotations without manual estimation. The clarity of presentation, with detailed figures and methodological descriptions, makes the contribution easy to follow. Finally, its significance lies in establishing a challenging benchmark that can drive progress in zero-shot amodal segmentation and physical reasoning in vision models.

### Weaknesses
Although Real-VAS introduces an important real-world benchmark, its dataset scale remains limited. The relatively small number of videos may restrict the statistical robustness of model evaluation and the diversity of object interactions. To strengthen its impact, the authors could expand the dataset with more varied scenes, objects, and motion patterns, or release additional data splits to enhance coverage and generalization testing.

### Questions
Have the authors compared model performance on Real-VAS with results on other real or synthetic amodal segmentation datasets? In other words, if a model performs well on Real-VAS, does that success transfer to other datasets, or is Real-VAS capturing distinct challenges? Such comparisons would help clarify how well the benchmark aligns with or extends existing datasets in terms of generalization and real-world difficulty.

### Soundness
4

### Presentation
4

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
This paper explores the construction of a real-world amodal video dataset (implied as Real-VAS). To build this dataset, the authors first identify potential occlusion events from static camera videos, then leverage physical law-based models as constraints, overlaying the cropped mask sequence of one object onto another to realize the construction of amodal videos. The paper provides a detailed introduction to the dataset construction method and conducts ablation studies to validate its effectiveness. This work not only enriches the data resources for the field of amodal video segmentation but also offers a feasible technical framework for real-world amodal dataset construction, holding certain significance for the development of the field.

### Strengths
1. The authors constructed amodal videos through clip overlapping from the perspective of ensuring the laws of real physical motion, which features high reliability. This approach may provide some insights to the development of this field and robotics.
2. It is very reasonable to use models such as Depth Anything V2 and Generative Omnimatte to ensure physical plausibility.
3. The writing is good and the paper is easy to follow.

### Weaknesses
1. For the REAL-VAS CONTAINMENT part, the motion of objects is mainly controlled by CoTracker3, which raises a serious concern: for containment data, is the trained model actually fitting CoTracker3? What advantages does this have compared with directly combining CoTracker3 with ordinary VOS/VIS models? Directly combining CoTracker3 seems to be more robust.
2. All videos are sourced from static cameras, which limits the diversity of videos, even though simulated camera motion has been added.
3. The proposed Real-VAS is described as a "large-scale" dataset, but the total number of videos is only 400, which is not a large scale. In addition, for containment-type data, fixed cameras and scenes are required for shooting, and the cumbersome production process seems unsuitable for scaling up.
4. There is a lack of evaluation of the constructed data as a training set.
5. The analysis in Table 3 is very valuable, but please add a brief introduction to TABE and Diffusion VAS.

### Questions
How is the motion of objects inside the container modeled? For example, in the third row of Figure 2, when an object is put into a paper bag, how is the speed at which the object falls to the bottom of the paper bag determined? And as the paper bag moves, will the object move inside the container? This is more reflective of the laws of physics but seems to have not been explored.

### Soundness
3

### Presentation
3

### Contribution
2
