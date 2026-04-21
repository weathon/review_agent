# IMDPrompter: Adapting SAM to Image Manipulation Detection by Cross-View Automated Prompt Learning

- Avg Score: 5.75
- Decision: Accept (Poster)
- Scores: 5, 6, 6, 6

## Abstract
Using extensive training data from SA-1B, the Segment Anything Model (SAM) has demonstrated exceptional generalization and zero-shot capabilities, attracting widespread attention in areas such as medical image segmentation and remote sensing image segmentation. However, its performance in the field of image manipulation detection remains largely unexplored and unconfirmed. There are two main challenges in applying SAM to image manipulation detection: a) reliance on manual prompts, and b) the difficulty of single-view information in supporting cross-dataset generalization. To address these challenges, we develops a cross-view prompt learning paradigm called IMDPrompter based on SAM. Benefiting from the design of automated prompts, IMDPrompter no longer relies on manual guidance, enabling automated detection and localization. Additionally, we propose components such as Cross-view Feature Perception, Optimal Prompt Selection, and Cross-View Prompt Consistency, which facilitate cross-view perceptual learning and guide SAM to generate accurate masks. Extensive experimental results from five datasets (CASIA, Columbia, Coverage, IMD2020, and NIST16) validate the effectiveness of our proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper introduces IMDPrompter, a novel approach for image manipulation detection (IMD) that leverages the Segment Anything Model (SAM). It addresses the challenges of manual prompt reliance and cross-dataset generalization by proposing a cross-view automated prompt learning paradigm. This includes components like Cross-view Feature Perception, Optimal Prompt Selection, and Cross-View Prompt Consistency, which enhance SAM's ability to generate accurate masks for detection and localization without manual guidance. The method is validated across five datasets, demonstrating its effectiveness in both in-distribution and out-of-distribution image manipulation detection and localization.

### Strengths
The primary strength of this paper lies in its application of the Segment Anything Model (SAM) to the field of image manipulation detection (IMD), introducing a cross-view automatic prompt learning paradigm. This paradigm significantly enhances the automation and generalization across datasets in IMD tasks through components such as automated prompt generation, optimal prompt selection, and cross-view prompt consistency. Furthermore, the paper demonstrates the effectiveness of the proposed method through extensive experiments across five different datasets, testing not only the model's in-distribution performance but also its robustness in out-of-distribution scenarios. These contributions not only advance the technology of image manipulation detection but also provide a powerful new tool for the field of multimedia forensics.

### Weaknesses
1. The technical details about the Cross-View Consistency Enhancement Module (CPC) implementation are insufficient. The paper does not clearly explain how different consistency loss formulations impact the results, and there is no visualization or analysis demonstrating how cross-view consistency is maintained throughout the detection process.
2. The comparison baselines are relatively outdated, missing important recent work from 2024 and other foundation model-based approaches. This limits the comprehensiveness of the comparative analysis and makes it difficult to assess the method's standing against the latest advancements in the field.
3. The paper shows limited discussion of model robustness to distribution shifts and lacks experiments demonstrating how the model adapts to real-world scenarios where data distribution varies. There is no clear mechanism described for handling domain adaptation.
4. The multi-view fusion analysis is incomplete. The paper lacks detailed ablation studies that quantify the individual contribution of each view, and there is no discussion of the computational trade-offs associated with different view combinations.
5. The paper lacks justification for selecting SRM, Bayer, and Noiseprint as noise perspectives. There is no analysis of their specific effectiveness for different types of image manipulations, nor any comparison with other potential noise perspectives that could potentially achieve similar or better results.

### Questions
1. Can you provide more detailed technical information about the consistency loss formulation in CPC? It would be helpful to see visual examples of how CPC affects the prompt generation process and its impact on the final detection results.
2. Why were certain recent methods excluded from the comparison? Have you considered comparing with methods published in 2024 or other SAM-based approaches?
3. How does IMDPrompter handle images from different domains in practical applications? What mechanisms could be added to make the model more adaptive to distribution shifts?
4. Can you provide comprehensive ablation studies showing the contribution of each view and how different view combinations affect the model's performance? This should include an analysis of the computational overhead associated with each additional view.
5. Could you explain the rationale behind choosing SRM, Bayer, and Noiseprint specifically as noise perspectives? Have experiments been conducted with other noise perspectives, and if so, what were the results?

### Soundness
2

### Presentation
3

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
This work introduces IMDPrompter, a novel cross-view prompt learning framework based on the Segment Anything Model (SAM) to automate detection and localization in image manipulation tasks, overcoming SAM’s reliance on manual prompts and enhancing cross-dataset generalization through cross-view perceptual learning techniques.

### Strengths
1. Innovative Application: The paper applies SAM  to the underexplored area of image manipulation detection, which extends SAM's use case beyond traditional segmentation tasks. The proposal of a cross-view automated prompt learning paradigm with IMDPrompter is unique and addresses the challenges specific to manipulation detection tasks.

2. Automation of Prompt Generation: One of the standout innovations is the elimination of SAM's reliance on manual prompts. The introduction of modules like Optimal Prompt Selection (OPS) and Cross-View Prompt Consistency (CPC) strengthens SAM’s utility by automating the prompt generation, potentially making SAM more accessible for manipulation detection.

3. Robustness and Generalizability: IMDPrompter's multi-view approach—integrating RGB, SRM, Bayer, and Noiseprint—demonstrates enhanced generalization, particularly on out-of-domain datasets. The ablation studies further substantiate the contributions of each module, which supports the validity of the multi-view and prompt-learning design.

4. Strong Experimental Validation: The model shows significant improvements in image-level and pixel-level metrics across multiple datasets (CASIA, Columbia, IMD2020, etc.), indicating its robustness. The experimental setup includes various metrics (AUC, F1-scores), highlighting the model’s strengths compared to prior approaches.

### Weaknesses
1. Complexity and Computational Cost: IMDPrompter’s architecture includes multiple modules (OPS, CPC, CFP, PMM), each introducing additional computational overhead. The increased complexity may impact the model’s efficiency, potentially limiting real-world deployment, especially for large-scale or time-sensitive tasks. Maybe we can provide computational complexity analysis or runtime comparisons with existing methods.

2. Limited Modality Discussion: While IMDPrompter is tested on a range of datasets for image manipulation, it would be beneficial if the authors discussed the potential application of this approach to other domains or modalities, such as video manipulation, to establish broader applicability. You can discuss particular challenges or modifications that would be needed to apply IMDPrompter to video manipulation detection.

3. The current framework provides limited insight into how each view or prompt contributes to the final decision-making process. Adding visualizations or more detailed explanations on how SAM’s interpretability might translate into manipulation detection could improve the work’s practical value.  For example, you can add the ablation studies showing the impact of each view, or visualizations of the learned prompts for different types of manipulations.

4. Reliance on Specific Views: The model’s reliance on SRM, Bayer, and Noiseprint views may limit its utility across other manipulation detection types that do not exhibit these specific signal properties. Further exploration into the model's adaptability to new types of data without these views might be necessary. I think you can discuss or demonstrate how IMDPrompter might be adapted to work with different types of views or features. Maybe you can also add an experiment with a subset of the current views to assess the model's flexibility.

### Questions
Real-World Deployment: Has IMDPrompter been tested in real-world settings, where variations in lighting, compression, and manipulation styles may further challenge the model’s robustness?

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
3

### Summary
The paper addresses the application of the SAM to the domain of image manipulation detection. The paper proposes IMDPrompter, a cross-view automated prompt learning paradigm that extends SAM's capabilities for IMD tasks. The proposed method is evaluated on five datasets (CASIA, Columbia, COVER, IMD2020, and NIST16), demonstrating significant improvements over existing state-of-the-art methods in both in-distribution and out-of-distribution settings.

### Strengths
The paper is well-motivated. The authors use SAM to address the challenges in the image manipulation detection domain. The introduction of automated prompt learning and integration of semantic-agnostic features is innovative and extends the applicability of SAM to a new domain.

The design is straightforward and reasonable. Apart from RGB images, using multiple views provides more information for IMD.

Five datasets and ablation studies demonstrate the effectiveness of the proposed IMDPrompter.

### Weaknesses
1. The proposed method introduces several additional modules, encoders, and views, which increase the complexity of the model.

2. Although the combination of four additional modules increases the performance, the authors do not provide the mechanism or reason behind why they chose views of SRM, Bayar, and Noiseprint, and which type of image processing information would help the IMDPrompter generate a more precise mask.

3. There is no CFP in Figure 2, but it is described in the caption. 

4. There are no details of the architecture of the SRM/RGB/Bayar/Noiseprint encoder and its computational cost. 

5. While the paper acknowledges that relying solely on RGB information is insufficient for cross-dataset generalization, there is limited discussion on scenarios where the proposed semantic-agnostic views might also fail, such as advanced manipulation techniques that bypass noise-based detectors.

6. The number of abbreviations is too many, which may interrupt the experience and hinder understanding for readers not familiar with the notation.

7. It would be better to bold the best results in Table 3,4,5,6.

### Questions
Please see the weakness part 2, 3, 4, 5, 6, 7.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an automated prompt learning SAM-based method for image manipulation detection. Multiple views, such as RGB, SRM, Bayar, and Noiseprint, are utilized and integrated to generate auxiliary masks and bounding boxes for SAM. Meanwhile, many modules, such as Cross-view Feature Perception and Prompt Mixing modules, are proposed for mixing features for the Mask Decoder. Extensive results demonstrate the effectiveness of the proposed method.

### Strengths
1. The authors utilize SAM's zero-shot capabilities for image manipulation detection.
2. Multiple views, such as RGB, SRM, Bayar, and Noiseprint, are utilized and integrated to generate auxiliary masks and bounding boxes for SAM
3. Many modules, such as Cross-view Feature Perception and Prompt Mixing modules, are proposed for mixing features for the Mask Decoder. 
4. Extensive results demonstrate the effectiveness of the proposed method.

### Weaknesses
1. Can the authors clarify why SAM is employed in image manipulation detection? The authors utilize four different views to generate masks and then bounding boxes, which serve as prompts for the Mask Decoder. To the best of my knowledge, the output masks of the Mask Decoder are significantly dependent on prompt accuracy. It is essential that the generated masks are sufficiently accurate. Can the authors provide the accuracy metrics for both the generated masks and the output masks from the Mask Decoder? Additionally, is it necessary to employ SAM?  
2. Are the Mask Decoder and Prompt Encoder kept frozen during the training process?
3. In the PMM module, the feature F_{SAF}, which is encoded from images, is resized to the same shape as the output of the Prompt Encoder, F_{opt}, which is encoded based on coordinates. Could the authors elaborate on the motivation for this approach? The fusions of image embeddings and coordinate embeddings appears inconsistent.  
4. The authors claim that the proposed OPS and CPC enhance alignment across views. Ideally, they utilize the CPC loss function to achieve prompt consistency. However, However, this claim lacks convincing evidence. Can the authors provide details on how the two proposed modules contribute to improved alignment?
5. The proposed method is trained only on the CASIAv2 dataset, while several other studies, such as CAT-Net v2 [1], TruFor [2], and UnionFormer [3], utilize additional datasets for training. In Table 1 and Table 2, the metrics are based on the CASIAv2 dataset without additional datasets. Can you explain why the method is only trained on the CASIAv2 dataset?
6. Table 2 presents numerous NaN values for Sensitivity, Specificity, and F1 scores related to TruFor. Can the authors provide the corresponding metrics?
7. Can you provide more recent methods for comparison? such as UnionFormer [3].

[1] Kwon, Myung-Joon, et al. "Learning jpeg compression artifacts for image manipulation detection and localization." International Journal of Computer Vision 130.8 (2022): 1875-1895.

[2] Guillaro, Fabrizio, et al. "Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.

[3] Li, Shuaibo, et al. "UnionFormer: Unified-Learning Transformer with Multi-View Representation for Image Manipulation Detection and Localization." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
