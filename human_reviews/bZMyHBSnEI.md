# Deep Equilibrium Multimodal Fusion

- Decision: Reject
- Scores: 5, 5, 5, 6

## Abstract
Multimodal fusion integrates the complementary information present in multiple modalities and has gained much attention recently. Existing fusion approaches exhibit three key elements for informative multimodal fusion, *i.e.*, stabilizing unimodal signals, capturing intra- and inter-modality interactions at multi-level, and perceiving modality importance in a dynamic manner. The current fusion methods mostly suffice only one of these conditions, without considering all three aspects simultaneously. Encapsulating these ideas, in this paper, we propose a novel deep equilibrium (DEQ) method for multimodal fusion via seeking a fixed point of the dynamic multimodal fusion process and modeling feature correlations in an adaptive and recursive manner, which naturally consolidates the three key ingredients for successful multimodal fusion. Our approach encodes and stabilizes rich information within and across modalities thoroughly from low level to high level and dynamically perceives modality importance for efficacious downstream multimodal learning, and is readily pluggable to various multimodal frameworks. Extensive experiments on four well-known multimodal benchmarks, namely, BRCA, MM-IMDB, CMU-MOSI, and VQA-v2, involving a vast variety of modalities, demonstrate the superiority and generalizability of our DEQ fusion. Remarkably, our DEQ fusion consistently achieves state-of-the-art performance on these benchmarks. The code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focus on improving multimodal fusion by developing a dynamic fusion framework adaptively model the cross-modality interactions hierarchically. The authors propose DEQ fusion which recursively executing nonlinear projections on modality-wise features and the fused features until the equilibrium states are found. Experimental results on various benchmarks can support their findings.

### Strengths
1. Multimodal fusion is one of the most fundamental problem in multimodal fusion, thus it is worth to design novel fusion mechanism.
2. The proposed method is novel and well motivated to me. To the best of my knowledge, most existing multmodal fusion methods tend to fuse information from multiple source in a static manner. The proposed method fuses features in a dynamic and recursive manner, which is new and interesting.
3. The proposed method are evaluated on various multimodal benchmarks, including multi-omics analysis, image-text classification, audio-text sentiment analysis and visual question answering. I appreciate the extensive experimental results. Additionally, the authors claim that the proposed DEQ fusion is readily pluggable to existing multimodal frameworks, which is very promising.

### Weaknesses
1. Computational cost: Though recursively fusing multimodal information is quite novel and have a chance to get better performance, I wonder if the proposed method is more time-consuming that its counterparts? It seems such comparisons is lack in the main paper and supplementary. Given this point, I encourage the authors to share more explanations about this.
2. Scalability: As the authors claim that DEQ fusion is readily pluggable to existing multimodal frameworks, I think it is deserving to further clarify how to combining DEQ fusion into existing multimodal fusion methods. For example, some pseudo code code will be very appreciated and make the proposed method easier to follow.
3. Motivation: The authors claim that DEQ fusion is an unified framework that looks into three aspects simultaneously including stabilizing and aligning multimodal signals, integrating interactions across modalities from multi-level, dynamically perceiving information. In my view, many attention-based multmodal fusion methods may also achieve the aforementioned points. In section 3.3, the authors say that DEQ fusion is not depend on any unimodal feature extraction or preprocessing methods. However, there exists some attention-based fusion methods which may also independent on unimodal feature extraction methods. For example, a simple self-attention module or MMTM[1]. Could the authors give some further clarifications?

[1] Joze, Hamid Reza Vaezi, et al. "MMTM: Multimodal transfer module for CNN fusion." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

### Questions
Please refer to weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The article presents an approach involving dynamic multi-modal fusion, introducing a weight-tied architecture to amalgamate distinct modality features and derive the unified representation simultaneously. By seeking the equilibrium state, this method facilitates the acquisition of stabilized intra-modal representations and fosters interactions across different modalities.

### Strengths
+ This study concentrates on the development of an innovative multi-modal fusion method, which endeavors to attain a state of equilibrium among features, markedly distinguishing itself from prior fusion processes.
+ The explanation in this article clearly and meticulously depicts its fusion architecture.

### Weaknesses
- The phrase "every level" in the introduction implies a comprehensive integration of cross-modality interactions throughout the multi-modal fusion process. However, given the paper’s focus on fusion of features, which is traditionally associated with late fusion, there seems to be a discrepancy. The paper apparently does not delve into early or middle fusion strategies. To reconcile this, one could interpret “every level” as referring to different stages or aspects within the late fusion process itself, although this may require clarification from the authors for a precise understanding.
-  The ablation studies conducted on the BRCA and CMU-MOSI datasets highlight the significance of the DEQ, a component not originally introduced by this work, overshadowing the impact of f_{\theta} and f_{fuse}. This raises concerns regarding the efficacy of the designed architecture in fully capitalizing on the potential for interaction among modalities. It suggests a need for further investigation and possibly a reevaluation of the architecture to ensure optimal performance.
-  The introduction categorizes three ways ‘stabilizing and aligning..., integrating..., dynamically perceiving...’ but provides limited insights, necessitating to explain the reason behind. It should be clarified that how each way contributes to better multi-modal learning. For instance, the rationale behind the need for a multi-modal model to eliminate redundancy is not clear. A thorough analysis is essential to elucidate why previous research has concentrated on these specific approaches, helping to strengthen the foundational knowledge and context for the study. 
-  Furthermore, it is recommended that the authors provide a more comprehensive explanation regarding the advantages of this design in stabilizing intra-modal representations. A thorough exploration of the key difference compared to previous research, particularly in terms of enhancing stability, would enhance the reader's understanding. 
-  While the proposed architecture aims to achieve stable intra-modal representations, its application is limited for it only targets to the fusion of features. This raises the question of whether the method could also encompass the learning of stable uni-modal encoders, fostering the acquisition of even more robust features. Exploring this avenue could potentially enhance the method’s applicability and effectiveness.

### Questions
1. It is advisable to provide detailed explanations for the dimensions of each vector and matrix utilized in the study, like $z$ in Related work. 
2. Given that Multi-bench encompasses a diverse array of fusion strategies, it is crucial for the authors to specify which particular method was employed on MM-IMDB dataset. 
3. The vectors $\alpha_i$ ought to be represented in bold to maintain consistency with standard mathematical notation.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a deep equilibrium (DEQ) method for multimodal fusion by seeking a fixed point of the dynamic multimodal fusion process and modeling feature correlations in an adaptive and recursive manner.

### Strengths
(1)	This method innovatively combines multimodal fusion with DEQ framework to iteratively achieve multi-level multimodal fusion while retaining single-modal information
(2) 	The experiments proves the effectiveness of the method, and the ablation experiment is relatively complete. The weight visualization in Figure 3 dynamically perceives modality importance for efficacious downstream multimodal learning, which is intuitive.

### Weaknesses
1. The method in this paper is compared with the weight-tied method, which shows that the method in this paper can converge. This is obvious because the method optimizes fθ by the formula z* = fθ(z*,x), and does not impose such a constraint on the weight-tied method with a finite number of layers, and the weight-tied method certainly cannot converge.
2. In the original DEQ paper, DEQ is proposed for memory efficiency, and the effect is similar to that of weight-tied, and it would be better if the article gave a comparative experiment with the weight-tied method. 
3. Some of the expressions in the paper are unscientific and abstract, such as the sentence on page 3:’Our fusion design is flexible from the standpoint that fθi(·) can be altered arbitrarily to fit multiple modalities. It could be better expressed as ‘Our fusion design is flexible from the standpoint that fθi(·) can be altered arbitrarily to fit multiple level features’.
4. The drawing is not intuitive.

### Questions
When the equilibrium state is reached, why an informative unified representation in a stable feature space for multimodal learning be obtained? What is the relationship between these two? The paper does not give proof.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed the Deep Equilibrium Multimodal Fusion (DEQ) algorithm for multimodal fusion. DEQ seeks a fixed point of the dynamic multimodal fusion process and models the feature correlations in an adaptive and recursive manner, which allows the DEQ algorithm to capture the complex dynamics of interactions between modalities. Extensive experiments demonstrated the effectiveness of DEQ.

### Strengths
(1) An interesting paper, the proposed DEQ method for multimodal fusion could be a new perspective in the field. By achieving equilibrium, the model could handle complex interactions between different types of data, potentially leading to better performance in tasks that require a comprehensive understanding of multimodal information. It is also a nice contribution to stability and robustness in the learning process for multimodal data.

(2) The experimental results are promising.

### Weaknesses
(1) DEQ models can be complex and require significant computational resources for training and inference. The search for a fixed point can sometimes lead to difficulties in convergence, especially in dynamically changing environments/contexts. There may be challenges in generalising the fixed-point approach to different types of multimodal data or applications.

(2) The paper may lack extensive evaluation against challenging applications, which is crucial to establish its real-world effectiveness. For example, I wonder how good the results of DEQ are in medical image fusion (e.g. CT, MRI, PET, etc.).

### Questions
Some related work is missing. For example the below [1] [2].

[1] Channel Exchanging Networks for Multimodal and Multitask Dense Image Prediction, TPAMI, 2022.
[2] 'Equivariant Multi-Modality Image Fusion' (Zhao et al, 2023).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
