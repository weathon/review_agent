# P$^3$-SAM: Native 3D Part Segmentation

- Decision: Reject
- Scores: 4, 8, 6, 4

## Abstract
Segmenting 3D assets into their constituent parts is crucial for enhancing 3D understanding, facilitating model reuse, and supporting various applications such as part generation.
However, current methods face limitations such as poor robustness when dealing with complex objects and cannot fully automate the process.
In this paper, we propose a native 3D point-promptable part segmentation model termed P$^3$-SAM, designed to fully automate the segmentation of any 3D objects into components.
Inspired by SAM, P$^3$-SAM consists of a feature extractor, multiple segmentation heads, and an IoU predictor, enabling interactive segmentation for users.
We also propose an algorithm to automatically select and merge masks predicted by our model for part instance segmentation.
Our model is trained on a newly built dataset containing nearly 3.7 million models with reasonable segmentation labels.
Comparisons show that our method achieves precise segmentation results and strong robustness on any complex objects, attaining state-of-the-art performance.
Our code will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a native 3D point-promptable part segmentation model named P3-SAM, aiming to achieve fully automatic and precise segmentation of arbitrary 3D objects. The core contribution lies in constructing a native 3D part segmentation dataset containing nearly 3.7 million models, based on which a segmentation model with excellent performance was trained. Experimental results show that this method surpasses existing methods on multiple benchmarks. Nevertheless, there is still room for improvement in the paper regarding the innovativeness of its methodology, the details of its data processing, and the depth of its experiments.

### Strengths
1.  **Pioneering Native 3D Dataset:** The core contribution is a massive native 3D part segmentation dataset (nearly 3.7 million models), which directly resolves the 2D-3D data gap and avoids issues from 2D-lifting.

2.  **State-of-the-Art Performance and Robustness:** The model demonstrates convincing state-of-the-art performance across multiple benchmarks and shows exceptional robustness on diverse data modalities, including non-watertight meshes, watertight meshes, and point clouds.

3.  **Complete Automatic Segmentation Pipeline:** The paper introduces a full end-to-end automatic segmentation pipeline, using Farthest Point Sampling (FPS) for prompt generation and Non-Maximum Suppression (NMS) for mask filtering, enabling segmentation without human intervention.

### Weaknesses
1.  **Limited Methodological Innovation:** The paper's state-of-the-art performance is largely attributed to its large-scale, novel 3D dataset rather than breakthroughs in the model architecture itself. The model appears to be more of a strong engineering implementation than a novel theoretical contribution, which may cast doubt on the superiority of its design.

2.  **Restricted Model Interactivity:** The model's support for only single-point prompts limits its application in complex scenarios where objects might require multiple points or bounding boxes for precise definition. The paper fails to justify the exclusion of other interaction methods or discuss how the model handles ambiguous single-point prompts.

### Questions
1.  **Questions on Model Generalization:** The model is primarily trained on synthetic 3D data; its generalization performance on real-world 3D scanned data is unclear.

2.  **Request for Specific Limitation Examples:** The paper states the model lacks an understanding of spatial volume. This raises the question of whether it would fail to segment real-world point clouds with complex internal structures or closely adjacent parts. Visual failure cases are requested to illustrate this limitation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces $P^{3}$-SAM, a 3D point-promptable part segmentation model that is natively trained on 3D data. It is a solid contribution to the field by moving away from 2D-lifting or 2D data engine approaches and building a truly native 3D solution. It also introduces a large-scale 3D part dataset of 3.7M models, which is a significant contribution to the community.

### Strengths
1. The curation of the 3.7M 3D model dataset with native 3D part annotations is a significant contribution in terms of number. The data curation details in Appendix A.3.1 are reasonable, and the handling of non-watertight and watertight data sounds novel to me.
2. The motivation to build a 3D native solution is meaningful and reasonable. It addresses real limitations in existing 2D-lifting and 2D data engine methods, such as 2D-3D data gaps, 3D consistency issues, and computational overhead from multi-view rendering.
3. $P^{3}$-SAM outperforms the SOTA methods like PartField in the benchmark.

### Weaknesses
1. PartObj-Tiny contains only 200 shapes across 8 categories, which is insufficient to demonstrate the performance of a model convincingly. Although there is another evaluation on PartNetE dataset having 1,906 shapes, evaluating on larger datasets like 3DCOMPAT++ [1] will make the claims much stronger.


[1] 3DCOMPAT++: A comprehensive dataset for 3D object understanding with fine-grained part annotations

### Questions
1. Given weakness 1, would it be possible to evaluate the performance on the car and airplane categories (68 meshes in total) from 3DCOMPAT++ [1] and compare the results with PartField? It will also be great if the evaluation could be performed on both fine-grained parts and coarse parts.
2. PartField shows some emergent cross-shape consistency when exploring the features. $P^{3}$-SAM is trained on a larger (maybe better) dataset, could you also give some qualitative and quantitative (if possible) comparison on a small scale with PartField?  

[1] 3DCOMPAT++: A comprehensive dataset for 3D object understanding with fine-grained part annotations

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
This paper proposes a 3D point feature based segmentation method that leverages the speed and high quality segmentation. It uses a single prompt point, two stage segmentor and IoU classifier to automatically and accurately segment complex 3D objects into their constituent parts. The first stage produces multi-scale mask predictions, while the second stage refines them using global contextual features. The IoU predictor then evaluates and selects the best mask, enabling robust and fully automated part segmentation.

### Strengths
- Native 3D formulation without lifting based on 3D point-based feature, avoid multi -view rendering or 2D-to-3D lifting which reduced the domain gasps and improved the robustness.
- Large-scale and high-quality 3D dataset with automatic part-level annotations. It is a strong generalization across watertight and non-watertight  objects.
- Efficient two-stage segmentation as mentioned in paper only 8s for a full and interactive segmentation.

### Weaknesses
- High probability of removing normals while training (0.3)? With the probability of 0.3 it may remove normals during data augmentation and improves robustness to missing information. But it could weaken the model’s geometric precision. 
- Lack of volumetric and semantic understanding because the training data does not contain labels and might have weak boundary prediction. The dataset was created using automatic geometric merging based on area thresholds and adjacency rules. This may merge distinct but small parts and remove fine structural details.
- The automatically generated segmentation using geometric heuristics may introduce bias and remove some fine-grained parts which are hard to apply on real world data.  The dataset has removed the imbalance parts, it might be hard to apply on moderately complex and regular structure.

### Questions
- How do you validate the created dataset, since the algorithm combines small parts and merges adjacent meshes automatically? Was the manual inspection involved to ensure that the resulting part annotations remain accurate and do not over-merge distinct components?

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
This paper proposes a novel method for 3D part segmentation by training a 3D-native part segmentation model. The model follows the paradigm of the 2D SAM, taking a 3D point as an input prompt and outputting three masks along with an IoU prediction. It is trained on 3.7 million 3D models with reasonably accurate segmentation labels.

### Strengths
1. The paper addresses an important task and is well-motivated and technically sound.

2. It presents a method for cleaning a large-scale 3D dataset and obtaining part annotations. The proposed model is one of the first to be trained with such large-scale 3D native part supervision.

3. The paper is well-written and easy to follow.

4. The proposed method appears to outperform previous baselines across multiple setups.

5. The paper provides ablation studies with respect to the network architecture.

### Weaknesses
1. **Method novelty and comparison with existing work.**
   From a methodological perspective, the high-level idea of the proposed model and training scheme closely follows that of 2D SAM, and thus lacks substantial novelty or conceptual breakthrough. Moreover, similar attempts to extend 2D SAM to 3D already exist (e.g., *PointSAM*). The paper should include a more detailed discussion comparing its network design and training scheme to 2D SAM and PointSAM, highlighting the specific differences, motivations, and potential advantages.

2. **Dataset curation and its scientific value.**
   One of the main contributions of this paper appears to be the curation of a large-scale part-annotated 3D dataset by leveraging the existing connectivity and grouping information embedded in raw 3D meshes. While the authors briefly describe the process of creating, filtering, and cleaning the dataset in the supplementary, I believe the main paper should devote more space to this aspect. Specifically, it would benefit from a deeper quantitative and qualitative analysis of data quality and its impact on model performance.

   Additionally, since the cleaning process involves several critical steps, it would be valuable to include analyses or ablation studies illustrating the effect and necessity of each step. This would help readers understand whether the current pipeline can reliably produce high-quality part annotations, and what the remaining challenges are in the data curation process.

   I acknowledge that the proposed network architecture is not unique and could be replaced by alternative designs. Therefore, more insights into data curation would significantly strengthen the paper and benefit future research, as the core challenge in this field lies more in *data quality and scalability* than in model design. Currently, the paper emphasizes the model side while underrepresenting the data aspect.

3. **Further experiments on data scale and quality.**
   I would like to see additional experiments exploring the influence of data quantity and quality, such as scaling curves with varying training data sizes. The current baselines are trained with significantly less data than the proposed method, which makes it unclear whether the improvements stem from the model architecture or simply from the larger training dataset. It would be important to evaluate whether the proposed model still outperforms when trained on the same amount of data as the baselines.

### Questions
1. **Data release.**
   Will you release the data processing scripts and the processed datasets with part annotations to facilitate future research and reproducibility?

2. **Post-processing hyperparameters.**
   What hyperparameters are involved in the post-processing stage (e.g., NMS)? How were these parameters determined? How do you control the granularity of the “automatic segmentation,” and how do these parameters affect the final results? Including an ablation study on these factors would strengthen the paper.

3. **Title clarification.**
   Would you consider revising the current title? There are already several works doing similar things,  *“Native 3D Part Segmentation,”* and *P³-SAM* does not appear to be the first in this line.

4. **Connectivity information.**
   Could you elaborate on how connectivity information is incorporated into *P³-SAM*?

5. **Hierarchical part segmentation.**
   Even after reading the supplementary material, the description of the *Hierarchical Part Segmentation* module remains unclear. Could you provide more details on how this module operates? For example, how do you obtain part segmentations with very fine granularity?

6. **Watertightness influence.**
   I do not fully understand why the distinction between watertight and non-watertight meshes affects the network, given that the model takes point clouds as input. Could you clarify this effect?

7. **Normal quality impact.**
   If the input normals are of poor quality (e.g., flipped faces or inconsistent directions), how does that affect the final segmentation performance?

8. **Clarification on Figure 6.**
   For Figure 6, could you explain the version without flood fill? Why it looks different than other subfigures?

9. **Runtime and efficiency.**
   Regarding the reported timing, does it correspond to the inference time for a single-point prompt, or to the runtime of the automatic segmentation process that samples multiple point prompts? How many point prompts are typically used for automatic segmentation, and how does this number affect the overall runtime and efficiency?

### Soundness
3

### Presentation
3

### Contribution
3
