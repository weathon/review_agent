# Dense Representation Learning for a Joint-Embedding Predictive Architecture

- Avg Score: 6.00
- Decision: Reject
- Scores: 6, 8, 5, 5

## Abstract
The joint-embedding predictive architecture (JEPA) recently has shown impressive results in extracting visual representations from unlabeled imagery under a masking strategy.
However, we reveal its disadvantage lies in the inadequate grasp of local semantics for dense representations, a shortfall stemming from its 
masked modeling on the embedding space and the consequent in less discriminative or even missing local semantics.
To bridge this gap, we introduce Dense-JEPA, a novel masked modeling objective rooted in JEPA, tailored for enhanced dense representation learning.
Our key idea is simple: we consider a set of semantically similar neighboring patches as a target of a masked patch.
To be specific, the proposed Dense-JEPA (a) computes feature similarities between each masked patch and its corresponding neighboring patches to select patches having semantically meaningful relations, and (b) employs lightweight cross-attention heads to aggregate features of neighboring patches as the masked targets.
Consequently, Dense-JEPA learns better dense representations, which can be beneficial to a wide range of downstream tasks.
Through extensive experiments, we demonstrate our effectiveness across various visual benchmarks, including ImageNet-1K image classification, ADE20K semantic segmentation, and COCO object detection tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an improved version of the JOINTEMBEDDING PREDICTIVE ARCHITECTURE (JEPA) that brings significantly better performance on downstream dense prediction tasks and slight improvements on classification task. The paper proposes masked semantic neighboring strategy to select more semantic similar patches as prediction target, and proposed a Local Aggregation Target module to construct learning targets.

### Strengths
1. The overall method, Dense-JEPA, obtains better performance in comparison with I-JEPA and MAE with fewer pre-training epochs on both dense prediction tasks and classification tasks.
2. The overall modification is simple yet effective.

### Weaknesses
1. Some symbols and equations make the paper a bit hard to follow and might be redundant. For example, 
    1. The symbols in equation 2-5. Adding symbols of features in Figure 2 might make it easier to read and understand. Further more, providing structure figures like those of transformer blocks, indicating what the $Q$, $K$ and $V$ features of $h_{\theta}$ and %h_{\hat{\theta}}$ would also make it quicker to understand. 
    2. The subscript of $s_I$ and $x_t$ are wierd in $s^{LAT}_{i} = h_{\hat{\theta}}({x_j}_{j\in P_i}, x_t)$ of Eq.4. What is the relationship between subscript $t$ and $i$? Can we understand it as $s^{LAT}_{t} = h_{\hat{\theta}}({x_j}_{j\in P_i} , x_t)$
2. The motivation of the Local Aggregation Target module might require further explanation. Why does adding such a module improve performance? The modification proposed in this paper is simple, thus, more explanation, analysis, and insights would be benificial for the community to understand it better.

### Questions
1. The reviewer does not quite understand the structure of $h_{\theta}$. In Eq. 4, it seems that $s_x$ and $x_c$ are different embeddings and the Table 7 studies $h_{\theta}$ using cross-attention. However, from Figure 2, the definition of $s_x$ around Eq. 2 ('... given the output of the context encoder, sx,...' and the definition of $x_c$ around Eq. 4 ('where $s_x$ denotes context embeddings and $x_t$, $x_c$ denote the averaged embeddings from all patches in the target encoder and only unmasked patches in the context encoder, respectively'), the reviewer feel that $s_x$ and $x_c$ are the same thing.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces Dense-JEPA, a method for self-supervised representation learning from images that builds upon the I-JEPA approach. I-JEPA is designed to predict missing patch representations in the representation space. Dense-JEPA offers two straightforward extensions to improve the target representations of masked patches.

- Dense-JEPA initially retrieves patches that are 'semantically' similar to a target patch within a specified local neighborhood.
- It subsequently generates an aggregate representation for a target patch employing cross-attention mechanisms on the 'semantically' similar patches. Symmetrically, it also generates locally aggregated representations for the predicted patches and matches them to the new aggregated target.

The paper provides a comprehensive evaluation of Dense-JEPA across a wide range of tasks, including ImageNet classification, ADE20K/COCO segmentations/detections, DAVIS video object segmentation, and Clevr object counting and depth prediction. The results consistently demonstrate that Dense-JEPA outperforms I-JEPA and other masked image modeling approaches when using similar model sizes.

Moreover, this work underscores the significance of careful consideration regarding the selection and computation of targets when conducting masked image modeling in the representation space.

### Strengths
The paper presents a simple but novel extension to the I-JEPA framework.  It conducts a thorough empirical assessment across a diverse range of tasks, consistently demonstrating improvements over the I-JEPA baseline and other approaches to masked image modeling.

Additionally, this paper highlights the importance of thinking about the target representation in masked-modelling tasks which is a valuable insight for the community.

### Weaknesses
Although the paper is generally well-written, there are certain sections where clarity could be improved. Specifically, I found the annotations in section 3.3 somewhat confusing. It appears that the cross-attention operation is performed, with the query being the average patch representation of the target x_t (or context (x_c), and the key/value being the semantically similar patches {x_j}_{j\in P_i} (or the entire context s_x). However, it's not entirely clear how distinct representations are obtained for each spatial location, given that the query is shared across all locations? Could you please clarify the computations performed by the local-aggregation layer?

Additionally, while the findings are well-supported in the explored settings, it remains uncertain whether they would generalize to other learning frameworks, such as masked-image auto-encoders, or if they hold true for larger-scale models. It would be valuable to include a section in the manuscript discussing these limitations.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes Dense-JEPA, a new self-supervised learning method based on I-JEPA, that additionally incorporates the notion of local aggregated regions, and prediction at the region level. Dense-JEPA is evaluated on a wide range of segmentation and classification tasks, and compares favourably on most of the tasks against I-JEPA.

### Strengths
1) The gains on segmentation tasks over I-JEPA is significant, and indicates that grouping patches into semantically meaningful regions helps learning better local representations. This is demonstrated on a set of different tasks of different nature.

2) The paper is well written and easy to follow, and the method is clearly described. Only the main figure could be improved, see below.

### Weaknesses
1) The results are not surprising, dense self-supervised learning has been explored extensively in the literature, and the same conclusion and gains on segmentation tasks have already been observed. DenseCL [1] uses a contrastive loss function to match local vectors. VICRegL [2] combines local and global aggregation and uses a similar nearest neighbour search for semantically similar patches. ODIN [3] computes segmentation masks online and aggregates the patches corresponding to the same regions. These papers are not mentioned or cited, I would therefore recommend to do a more extensive literature review, and highlight the novelty within the dense SSL literature.

2) There is no comparison with DINO on most of the tasks. The DINO family of methods [4,5] has shown that dense loss functions might actually not be required to learn very good dense features, and offers the best performance on local tasks as of today. For example, on DAVIS-2017, DINO reports 61.8 J&M with a ViT-S/16, while the best number reported in the paper is 58.3 with a much larger ViT-L backbone. On classification on ImageNet, a DINOv2 ViT-L model is over 86 points. The gains over I-JEPA are interesting, but the overall performance is very far from the state-of-the-art. MAE is not really a good baseline as it performs very poorly on frozen tasks, which are standard to evaluate SSL methods.

3) The gains over I-JEPA are probably not worth the complexity. The paper is missing a study on the tradeoff in terms of compute compared to I-JEPA. In terms of memory, running time and data efficiency.

4) Figure 2 is very unclear. There is not much additional information compared to the similar figure in the I-JEPA paper, and the explanation for “Local aggregation head” and “Masked Semantic Neighboring” is only in the text. I would recommend clarifying the Figure and emphasizing the difference with I-JEPA.

5) The visualization of Figure 1 is not convincing. I-JEPA patches already contain highly semantic information and Dense-JEPA clearly brings a bias towards neighboring patches, which is not necessarily a good thing as we would like to let the system pick patches that are far away but semantically very similar.



[1] Dense Contrastive Learning for Self-Supervised Visual Pre-Training, Wang et al, CVPR 2021

[2] VICRegL: Self-Supervised Learning of Local Visual Features, Bardes et al, NeurIPS 2022

[3] Object discovery and representation networks, Henaff et al, ECCV 2022

[4] Emerging Properties in Self-Supervised Vision Transformers, Caron et al, ICCV 2021

[5] DINOv2: Learning Robust Visual Features without Supervision, Oquab et al, 2023

### Questions
Why do you use cross-attention to do the aggregation ? Have you tried max-pooling or other types of pooling ? Do you consider that it is an essential component of the method ?

The transformer blocks in the encoder are performing self-attention between every pair of tokens, which might already have the effect of grouping patches using a distance in the embedding space. Have you thought about that and do you think that your method theoretically brings something ?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes Dense-JEPA, a novel masked modeling objective for JEPA, enhancing dense representation learning.  Extensive experiments on image classification, semantic segmentation and object detection demonstrate Dense-JEPA effectiveness.

### Strengths
The experiments are comprehensive and the results are good.

### Weaknesses
1. The novelty can be improved. It seems that Dense-JEPA is just an improvement on I-JEPA, especially Figure 2 is so similar to figure 2 in I-JEPA. The main points proposed by the paper are not well presented.
2. The descriptions of MSN and LAT are ambiguous. From section 3.2, MSN is just using the presentation extracted by target encoder to find the most similar neighbor patches, which is prepared for LAT to aggregate from these similar patches, so MSN could not be regarded as an individual module.

### Questions
Without LAT, does MSN actually influence the training procedure (Table 5, the third row)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair
