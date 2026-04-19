# GeVLM: 3D Object Grounding with Geometry-enhanced Vision Language Model

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 5, 5

## Abstract
Understanding 3D scenes with point cloud data in tasks such as object referencing, question-answering, and captioning poses significant challenges to vision language models (VLMs), due to the complexity of integrating both linguistic and spatial information. While existing methods have mapped point cloud features into LLM space to enable 3D scene comprehension, they often overlook viewpoint information and the relative spatial distance between objects, this can lead to confusion in interpreting spatial descriptions and grounding objects. This paper presents a geometry-enhanced vision LM (GeVLM) to address these challenges. Specifically, we propose viewpoint-consistent position encoding (VCPE) and distance-aware cross-entropy (DACE) loss, which enhance the model's ability to interpret relative spatial relationships agnostic to camera viewpoint and incorporate distance information in the label space. We additionally introduce the DetailedScanRefer dataset, which provides identifiers and spatial annotation for each object mentioned in the referencing description to further emphasize spatial relationships. GeVLM demonstrates significant improvements over the Chat-3D v2 baseline, particularly with 4.0\% and 2.7\% absolute increase in Acc@0.25 and Acc@0.50 respectively on the ScanRefer benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper aims to solve a challenging 3D object grounding task. The paper proposes GeVLM to grasp view-point information and the relative spatial distance between objects. Besides, to facilitate the research community, the paper introduces a new dataset named DetailedScanRefer, which provides identifiers and spatial annotation for each object mentioned in the referencing description to further emphasize spatial relationships.

### Strengths
1. Focusing on positional relationships is a very worthy research problem for 3D tasks.
2. Experiments validate the performance of the proposed method by comparing some advanced algs.

### Weaknesses
1. The way to integrate view-point information seems too incremental and has limited novelty. It simply utilize position embedding then use self-attention.
2. The gain of VCPE and the designed DetailedScanRefer dataset in the ablation study are too small to prove the effectiveness of the method.
3. The paper lacks qualitative experimental analysis to prove that VCPE and DACE have learned the expected position information.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper points out that many 3D LLM models often do not consider the viewpoint information and the relative spatial distance between objects. It introduces a method for 3D scene understanding using LLM that mainly focuses on incorporating 3D viewpoint information, improving positional encoding, implementing distance-aware cross-entropy loss, and enhancing the dataset's quality. The model trained using the proposed method improves upon the baseline by clear margins.

### Strengths
1. This paper notices an interesting aspect that other 3D LLM may overlook. The corresponding method is specifically tailored to tasks and successfully tackles the identified problems, as shown by the ablation studies.
2. The proposed method is orthogonal to the existing methods and can be used to improve other methods.

### Weaknesses
1. The improvement on the Scan2Cap dataset is weak, although the author incorporates extra information for captioning, and the proposed method may have a negative impact. Some recent baselines, for example, LEO[1] and LL3DA[2], are not compared. In line 453, the caption ability is mentioned, but no results are provided.
2. The improvement on SQA, a benchmark focusing on viewpoint, is weak. Since this paper mainly focuses on improving the view understanding of the LLM, the results are not compelling enough.
3. In line 418, 'the task prioritizes object semantics over spatial location, further diminishing the effectiveness of the DACE loss.' the paper does not give any evidence to support this claim.
4. Although the method surpasses the baseline, it also uses extra annotations. No ablations can be used to decide which aspect contributes to the performance gain. For example, the setting in which only VCPE DACE is used and no detailed ScanRefer annotations are added.

[1] https://arxiv.org/abs/2311.12871
[2] https://arxiv.org/abs/2311.18651

### Questions
1. Is the view information only used during training? Since we can not access the ground-truth view information, this hinders the wide application of the proposed method.
2. Some matrices, such as the ROUGE-L and EM on the ScanQA dataset, are not reported.
3. Why are "near" and "far" considered viewpoint-related questions with potential ambiguities in line 519? The distance between objects will remain constant from the given viewpoint.
4. Why is 'Chat-3D v2' marked with * in Tables 1, 3, and 4 but not in Table 2? Does this denote reproduced results?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes to enhance 3D LLM by incorporating information about 3D viewpoints and relative spatial distances. Besides, they create the DetailedScanRefer dataset with grounding annotations for each object described. The experimental results show the improvement over baseline methods.

### Strengths
1. The motivation for incorporating viewpoint and spatial distance is reasonable.
2. The DetailedScanRefer dataset with fine-grained grounding annotations is valuable for future research in 3D object grounding.
3. The experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The VCPE module's reliance on camera viewpoints as an additional input may pose challenges for real-world applications due to the difficulty of acquiring such data.
2. The spatial distance-aware loss function appears to overlook semantic similarities between objects. Objects that are spatially close but semantically different from the target should incur a greater penalty than those that are semantically similar but spatially distant. A more effective approach might involve incorporating both spatial and semantic distances into the cross-entropy loss.
3. The utility of the newly created dataset for enhancing grounding performance is questionable, as indicated by the ablation results in Table 4. Additional evidence supporting the dataset's importance would be beneficial, along with strategies to enhance its robustness, especially considering potential errors in object ID assignments by GPT-4o.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work begins by emphasizing the importance of viewpoint in grounding and introduces a geometry-enhanced vision-language model. It includes the VCPE and DACE modules, which address ambiguity caused by varying viewpoints and the positional information neglect in cross-entropy loss, respectively. Additionally, this work utilizes GPT-4 to generate the DetailedScanRefer dataset to support training, enabling the model to achieve substantial improvements over the baseline.

### Strengths
1. The motivation of this paper is clear, enhancing LLM-based 3D visual grounding models from the perspectives of viewpoint and relative spatial relationships, which aligns well with human intuition.
2. This paper introduces the integration of viewpoint information into the LLM-based paradigm and designs a loss function that better aligns with the task. Additionally, it presents a fine-grained dataset, rich in content, which makes a commendable contribution to the community.
3. The method proposed in this paper achieves significant improvements over the baseline.

### Weaknesses
1. The experimental section lacks a sufficiently comprehensive comparison. In addition to the baseline and other LLM-based models, comparisons should also be made with models specifically designed for this task to better demonstrate the overall performance of the proposed model in this context.
2. Since this method uses viewpoint information as model input, it is important to clarify whether other comparison models also include or utilize this information to ensure fairness. This is particularly relevant for works like ConcreteNet (“Four Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding,” ECCV 2024), which already explores viewpoint information. A comparison with this model's use of viewpoint information would be valuable.
3. From the ablation study shown in Table 4, it is observed that most of the performance gains are concentrated in the DACE module. It remains unclear whether similar improvements could be achieved using only "world + DetailedScanRefer + DACE." The necessity of incorporating viewpoint information is not well substantiated. Additionally, it appears that DetailedScanRefer does not contribute to performance improvement; clarification on this would be helpful. I would also like to see whether DACE remains effective without DetailedScanRefer.
4. There is no quantitative metric has been provided to assess the quality of the dataset generated by GPT-4 and Mask3D.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
