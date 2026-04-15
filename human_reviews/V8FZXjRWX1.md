# ReBaR: Reference-Based Reasoning for Robust Human Pose and Shape Estimation from Monocular Images

- Decision: Reject
- Scores: 5, 5, 6, 5, 5

## Abstract
This paper introduces a novel method, ReBaR (Reference-Based Reasoning for Robust Human Pose and Shape Estimation), designed to estimate human body shape and pose from single-view images. ReBaR effectively addresses the challenges of occlusions and depth ambiguity by learning reference features for part regression reasoning. Our approach starts by extracting features from both body and part regions using an attention-guided mechanism. Subsequently, these features are used to encode additional part-body dependencies for individual part regression, with part features serving as queries and the body feature as a reference. This reference-based reasoning allows our network to infer the spatial relationships of occluded parts with the body, utilizing visible parts and body reference information. ReBaR outperforms existing state-of-the-art methods on two benchmark datasets, demonstrating significant improvement in handling depth ambiguity and occlusion. These results strongly support the effectiveness of our reference-based framework for estimating human body shape and pose from single-view images.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a 3D human reconstruction system. PARE is used as a baseline, and there are several technical contributions to make it better, which include 1) introducing Transformer, 2) utilizing body features (global feature), and 3) relative depth estimation. Experimental results demonstrate the effectiveness of the proposed system.

### Strengths
Experiments are extensively conducted and demonstrate the effectiveness of the proposed system.

### Weaknesses
1. Technical contribution is limited. Table 4 clearly shows that there are noticeable performance improvements from the baseline, but what the authors did is not very novel actually. Utilizing the body feature (global feature) increases the performance a lot, which shows that using both local and global features is necessary. This makes sense, but is kind of expected.

2. I’m not sure why we need the relative depth loss function. The 3D joint angles already provide relative depth information. Why we need this?

3. Writings and notations are not clear. There are a couple of words that represent the same thing: reference feature and body feature. All of them are not very clear. Why not just call them a global feature, which is more intuitive and easy to get? Texts for description (for example, AGE in L_AGE) should be written with $L_\text{AGE}$. Also, the authors used $R$ to represent a set of real numbers, which should be $\mathbb{R}$.

### Questions
1. There are two ‘GAP’ in Figure 2. What does it mean? Does it mean global average pooling? The main manuscript says nothing about GAP.
2. Hand4Whole in Table 1 has wrong reference. It is from Moon et al (CVPRW 2022).
3. How is the part segmentation GT obtained? The authors rendered SMPL meshes by considering visibility as well? For example, the occluded body part has GT part segmentation?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this paper, the authors propose a reference-based reasoning network, known as ReBaR, for human pose and shape estimation, with a particular focus on handling occluded scenarios. As a single image-based method, ReBaR utilizes reference features for part regression reasoning to address challenges like occlusions and depth ambiguity. The paper presents extensive experiments to validate the effectiveness of the proposed method, and qualitative comparisons with previous state-of-the-art methods highlight significant improvements in handling depth ambiguity and occluded scenarios.

### Strengths
1. The paper addresses a more challenging setting of depth ambiguity and occluded scenarios, which is a commendable and important research direction.

2. ReBaR consistently outperforms existing state-of-the-art methods on multiple datasets, including AGORA, Human3.6M, 3DPW, and 3DPW-OCC.

3. The authors provide extensive qualitative comparisons and analyses, which effectively support the superiority of the proposed method over previous state-of-the-art techniques.

### Weaknesses
1. This paper should be better organized and improve the writing:

    The readability can be greatly enhanced if the author can further polish this paper, making long sentences and some of transactions more smooth. 

    There are many typos that need to be corrected. For example: 
       Last paragraph of the introduction: “datasets Then” should be “datasets. Then”.
      Section 3.3:   “ambiguity.Next” should be “ambiguity. Next”
     , etc…

2. Figures and tables should be clearer and better aligned with the text. For example:

    In Fig 2, the loss components should be more distinguishable. As indicated in the left corner, the auxiliary loss is marked as orange. However, all the losses are colored in orange. 

    The relationships between elements in the figure and the main text should be clarified. 

    The attention-guided encoders (AGE) are not indicated in Fig 2.

    It's unclear which components in Fig 2 are associated with the body-aware regression module.

    Please provide an explanation for the GAP component in Fig 2.

     It is suggested that the authors provide an overall architectural figure and detailed figures closely related to each section. Otherwise, readers need to retrieve Fig 2 frequently when reading the text. 

    Table 1 would be better positioned adjacent to Section 4: Comparison to the State-of-the-Art. 
Other improvements, such as those mentioned above, should also be addressed to enhance the paper's organization and clarity.

3. While the proposed method achieves excellent performance, it appears to be relatively complex. It would be beneficial to provide insights into the training process. Is the proposed method trained in multiple stages, such as initially training for 2D/3D pose estimation and subsequently training for the final mesh? Additionally, there are multiple loss components involved; it would be helpful to clarify whether these losses are optimized jointly or separately during different training stages.

4. The proposed method benefits from supervision by ground truth body/part segmentation maps, which some other methods do not utilize. To better understand the impact of this additional annotation, it would be valuable to conduct an ablation study. This would help ensure a fair and comprehensive comparison with other methods.


5. Given the potential computational cost of the proposed method, reporting the total parameters and actual inference times for each frame would be valuable information for readers and practitioners.

### Questions
see weakness

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel method, named ReBaR, for human pose and shape estimation from single-view images. It includes two main modules: an attention-guided encoder and a body-aware regression module. The first module extracts features from both body and part regions using an attention-guided mechanism. Then, the second module uses a two-layer transformer to encode body-aware part features for each part and a one-layer transformer for per-part regression. Experimental results on two benchmark datasets show the effectiveness of the proposed method.

### Strengths
-The proposed method considers attention-based feature learning from both body and part regions which is reasonable for occlusion handling.

-The proposed method regresses additional relative depth to the torso plane, which helps reduce depth ambiguity.

-The experimental results show better performance than baseline methods.

### Weaknesses
1. The proposed method significantly outperforms the baseline method PARE in Table 1 and only slightly outperforms PARE in Table 2. Why does this happen?

2. To reduce depth ambiguity, the 3D GT labels such as SMPL parameters and 3D joints also provide the depth information of the human body. If they are accurately regressed, the depth information can be well estimated. Why is the relative depth loss L_{RD} still needed?

3. Some training details are unclear. For example, 1) what are the weight parameters for each loss? Please specify each weight parameter in the training; 2) considering there are so many losses, how to balance these losses in training? 3) how to generate the body/part attention map (M_p, M_B); 4) since some training sets may not have GT SMPL labels, how to ensure supervision? 5) The loss functions in Equation 3 are somewhat confusing. Please consider the consistency throughout the paper (e.g., L_{BS} and L_{PS} in Figure 2 appear to have the same meaning as L_{b_seg} and L_{p_seg} in Equation 3).

4. Considering transformers used in the architecture, how does the training/inference time of the proposed method compare to the baseline methods (PARE and CLIFF)?

5. What are the limitations and failure cases of the proposed method? A discussion is suggested to provide inspiration for future research directions.

6. Typos. 
(1) On page 4, “R^{6840x3}” should be “R^{6890x3}”.
(2) On page 7, “Table 7 shows the result where our method still performs …” should be “Table 1”.
(3) On page 9, “MJE Depth reduced from 58.2 to 48.5” should be “… 58.2 to 47.5”

### Questions
Please refer to the weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
Authors introduced the novel method:  reference-based reasoning for robust human 3D mesh reconstruction task. Authors proposed to extract features from both body and part regions using attention-guided mechanism. Then they proposed the reference based reasoning by inferring the spatial relationships of occluded parts with the body. Experiments are conducted on two benchmarks and demonstrated the goo d performance.

### Strengths
Authors are tackling the challenging problem: 3D human mesh reconstruction in challenging cases when the severe occlusion exists in RGB images.
The idea of encoding part regions and global regions together looks interesting.

### Weaknesses
Less qualitative results for targetted scenario: Though authors insist that they tackle the problem of severe occlusion and depth ambiguity; while the results presented in the main result is not showing the results for that. Especially, the figure in the 5th row of the Fig. 4 is neigher occluded nor confused in 3d depth. Other results are also not including challenging cases. I think authors need to show the specific cases they are intending.
Also, the compared algorithms are not sufficient enough to validate the effectiveness of the method. The ProPose algorithm (CVPR’23) is shown in the Fig. 1; while it is not compared in the Figure 4. Also, in terms of PAMJE, the proposed method is not clearly the SOTA in Table 2.
Explanations are also not clear, especially, the usage of the losses in Sec. 3.4. In Sec. 3.2 authors explained their attention is learned using part-segmentation maps; while in Sec. 3.4, seemingly all the losses are applied to the entire network. I think authors need to clarify the aspect. Also, some `hat’ notations in Sec. 3.4 are wrongly used.

### Questions
Did authors made the part segmentation loss only applied to the attention learning? or other losses are also applied to learn it?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
To address the problem of occlusion and depth ambiguity, this paper proposes ReBaR, which learns reference features for part regression reasoning. This results show that it can outperform baseline methods, especially in the evaluation of depth.

### Strengths
1. The motivation is clear and interesting. 
2. Experiments are extensive and achieve state-of-the-art results.

### Weaknesses
1. The approach section contains too many details and lacks motivations and general pictures. Also, the pipeline is blurry.
2. Is PARE your baseline? If so, I am interested in whether REBAR and PARE have the same experimental setting and the comparison in FLOPS and PARAMS, especially for the PARAMS in the part branch.
3. It seems not very clear to me why your method achieves better results in depth with your network design.

The core question is why does your method perform better, especially in depth compared with PARE? Is that because the increase of data or parameters or the network design.

### Questions
Please see weakneeses

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
