# Learning Spatial-Semantic Features for Robust Video Object Segmentation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Tracking and segmenting multiple similar objects with distinct or complex parts in long-term videos is particularly challenging due to the ambiguity in identifying target components and the confusion caused by occlusion, background clutter, and changes in appearance or environment over time. In this paper, we propose a robust video object segmentation framework that learns spatial-semantic features and discriminative object queries to address the above issues. Specifically, we construct a spatial-semantic block comprising a semantic embedding component and a spatial dependency modeling part for associating global semantic features and local spatial features, providing a comprehensive target representation. In addition, we develop a masked cross-attention module to generate object queries that focus on the most discriminative parts of target objects during query propagation, alleviating noise accumulation to ensure effective long-term query propagation. The experimental results show that the proposed method sets new state-of-the-art performance on multiple data sets, including the DAVIS2017 test (\textbf{87.8\%}), YoutubeVOS 2019 (\textbf{88.1\%}), MOSE val (\textbf{74.0\%}), and LVOS test (\textbf{73.0\%}), which demonstrate the effectiveness and generalization capacity of the proposed method. We will make all the source code and trained models publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors focus on the problem of video object segmentation in long-term tracking and complex environments. To improve the model's robustness, a spatial-semantic network block is proposed to integrate semantic information with spatial information for video object segmentation. Additionally, a discriminative query mechanism is developed to capture the most representative region of the target for better target representation learning and updating. The proposed method achieves state-of-the-art results on most VOS dataset.

### Strengths
This paper addresses key issues in current applications of VOS methods: long-term tracking, occlusion, and object representation changes. 
1.	The proposed approach utilizes semantic and spatial information from upstream pre-trained models, enriching the target's semantic and detailed information. It is novel in the field of VOS. 
2.	The paper proposes a discriminative query generation mechanism to provide the model with more distinctive target information, which is validated on LVOS datasets.
3.	The proposed method is validated on various VOS datasets and achieves state-of-the-art results.

### Weaknesses
The paper does not have obvious weaknesses, but there are still some issues. 
1.	In Table 1, why is there no separate ablation study for the spatial block and semantic block? Please provide this part of the experiment.
2.	Some detail issues: In Figure 3, the blue results are not clearly marked and the position of * is not aligned.

### Questions
1、	How many trainable parameters and total parameters does the model have?
2、	Why does "DepthAnything" achieve the best results?
3、	If the backbone of the Cutie model is replaced with ViT, would it achieve good results?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel spatial-semantic block that effectively integrates semantic information with spatial features, resulting in a more comprehensive representation of target objects, especially those with complex or distinct parts. By utilizing a pre-trained Vision Transformer (ViT) backbone without the need to retrain all parameters, the proposed method significantly enhances the efficiency of video object segmentation (VOS).

Additionally, the development of a discriminative query mechanism marks a substantial advancement in the field. This mechanism prioritizes the most representative regions of target objects, thereby improving the reliability of target representation and query updates. This is particularly advantageous in long-term video scenarios, where appearance changes and occlusions can lead to noise accumulation during query propagation.

The authors also highlight the importance of learning comprehensive target features that encompass semantic, spatial, and discriminative information. This holistic approach effectively addresses challenges related to appearance variations and identity confusion among similar-looking objects in long-term videos, making it a valuable contribution to the VOS community.

Finally, extensive experimental results demonstrate that the proposed method achieves state-of-the-art performance across multiple benchmark datasets, including DAVIS 2017, YouTube VOS 2019, MOSE, and LVOS.

### Strengths
This paper presents a spatial-semantic modeling method and a discriminative query mechanism that significantly enhance the model's performance. Extensive experiments have been conducted to demonstrate the effectiveness of the model, and several visual examples are provided to clearly illustrate the results at different processing stages. Additionally, the final results showcase the model's considerable potential.

### Weaknesses
Writing Style:
1. The writing language is not concise enough, with many long sentences that significantly reduce readability. This is particularly evident in the introduction, such as on the second page: "We construct a Spatial-Semantic Block comprising a semantic embedding module and a spatial dependencies modeling module to efficiently leverage the semantic information and local details of the pre-trained ViTs for VOS without training all the parameters of the ViT backbone."

Image Details:
1. In Figure 2, there are N spatial-semantic blocks, but N is not specified later in the paper.

Method:
1. In Figure 2, the argmax operation in the distinctive query propagation is non-differentiable. Will this prevent the gradient from being propagated through the model?

2. If the introduced ViT backbone is not fine-tuned, will its performance degrade on the new dataset? A comparison experiment between freezing and not freezing the parameters is needed here.

3. The number of different queries should be related to the number of targets. However, using 8 queries yields better results. When faced with more than 8 targets, can 8 queries adequately represent the different targets?

4. In Table 3, there are two XMem entries, one of which is not referenced. It is unclear what the unreferenced entry represents, and why it lacks FPS results needs to be clarified.

5. Table 3 lacks a comparison of Joint Former results trained on the MEGA dataset. Please provide the results for Joint Former trained on the MEGA dataset in detail. If the original Joint Former was not trained on this dataset, can it be trained and then compared for performance?

6. The spatial-semantic block consists of two parts: first, the global feature cls token is fused with the semantic features, and then further enhanced through Deformable Cross Attention. It is necessary to separately validate the effects of directly fusing the features versus applying Deformable Cross Attention for further enhancement.

### Questions
1.In Figure 2, there are N spatial-semantic blocks, but N is not specified later in the paper.

2.In Figure 2, the argmax operation in the distinctive query propagation is non-differentiable. Will this prevent the gradient from being propagated through the model?

3.If the introduced ViT backbone is not fine-tuned, will its performance degrade on the new dataset? A comparison experiment between freezing and not freezing the parameters is needed here.

4.The number of different queries should be related to the number of targets. However, using 8 queries yields better results. When faced with more than 8 targets, can 8 queries adequately represent the different targets?

5.In Table 3, there are two XMem entries, one of which is not referenced. It is unclear what the unreferenced entry represents, and why it lacks FPS results needs to be clarified.

6.Table 3 lacks a comparison of Joint Former results trained on the MEGA dataset. Please provide the results for Joint Former trained on the MEGA dataset in detail. If the original Joint Former was not trained on this dataset, can it be trained and then compared for performance?

7.The spatial-semantic block consists of two parts: first, the global feature cls token is fused with the semantic features, and then further enhanced through Deformable Cross Attention. It is necessary to separately validate the effects of directly fusing the features versus applying Deformable Cross Attention for further enhancement.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the complex task of tracking and segmenting multiple similar objects in long-term videos, where identifying target objects becomes challenging due to factors like occlusion, cluttered backgrounds, and appearance changes. To tackle these issues, the authors propose a new framework for robust video object segmentation, focusing on learning spatial-semantic features and generating discriminative object queries.

The framework introduces a spatial-semantic block that combines global semantic embedding with local spatial dependency modeling, which enhances the representation of target objects by capturing both broad context and fine details. Additionally, a masked cross-attention module refines object queries, concentrating on the most distinctive parts of target objects and reducing noise accumulation over time. This approach aids in effective long-term query propagation, a critical factor for high-performance tracking over extended sequences.

The experimental results are strong, showing state-of-the-art performance across several benchmarks.

### Strengths
This paper’s S3 algorithm for Video Object Segmentation (VOS) demonstrates notable strengths:

1.Spatial-Semantic Integration: By combining semantic embedding with spatial dependency modeling, it effectively captures complex object structures without requiring extensive ViT retraining.

2.Discriminative Query Mechanism: The adaptive query approach improves target focus and reduces noise in long-term tracking, enhancing robustness.

3.Extensive Validation: State-of-the-art results on multiple benchmarks highlight its strong generalization across datasets.

### Weaknesses
1.This paper claims to address the challenges of long-term tracking and segmentation. However, as far as I know, memory mechanisms are crucial for tackling these challenges in long-term tracking and segmentation, yet the authors do not seem to have conducted ablation experiments on the number of frames in the memory bank.

2.I believe that the ablation study on the number of queries is insufficient with only 8, 16, and 32 as tested values. A wider range of query counts should be explored to more thoroughly validate the effectiveness of the proposed method.

### Questions
See weakinesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on video object segmentation. The authors analyze the existing challenges like structural complexity, occlusion, and dramatic appearance changes, and correspondingly propose spatial-semantic feature augmentation as well as discriminative query association. The ablation studies and visualizations verify the effectiveness of each module.

### Strengths
1. The motivation is clear and the architecture makes sense. Integrating high-level semantics and low-level spatial cues is promising in video object segmentation.
2. The experiments are thorough and the ablation studies can well reflect the effectiveness of each module.

### Weaknesses
1. The method is complicated. What is the advantage of using spatial offsets with deformable convolution compared to simple position encodings?
2. The second row of Figure 3(a) seems strange. With semantic feature augmentation, the feature maps can well highlight the desired object instance. Adding spatial cues on the contrary suppresses the emphasis on the target instance but enhances object instances with the same semantics.
3. Compared to SAM2, which designs a memory to prompt the segmentation of new frames, what is the advantage of this architecture?

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2
