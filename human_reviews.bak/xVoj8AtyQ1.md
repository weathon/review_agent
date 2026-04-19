# Vision Transformer with Irregular Attention

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 3

## Abstract
Compression of Transformer is a natural request that arose in the computer vision community. Apart from quantization that hardly rely on hardware, sparsification is another way to remove redundant parts, usually based on mask training or sparsity regularization. We propose the novel compressed structure of multi-head self-attention (MHSA) mechanism called Irregular Attention (IAtt). IAtt is built on BTD-LL1 tensor decomposition and is aimed at sparsifying pre-trained Vision Transformer by pruning query and key (QK) contraction dimension in MHSA block. We derive the algorithm of rank selection procedure for BTD-LL1 based on the structure of fusion layer obtained from CP decomposition of original MHSA kernels. In order to improve the compression ratio with least possible quality loss we introduce the fine-tuning schemes that yield each head its own sub-optimal rank for QK in the IAtt. We validated the proposed scheme for DeiT architectures on ILSVRC-2012 dataset. Our results show that IAtt has better performance compared to original MHSA compressed by SVD. It indicates that attention heads have non-uniform importance and require different QK contract dimensions.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper mainly focuses on the compression of Multi-Head Self-Attention (MHSA) mechanism. This paper introduces a novel compressed structure for MHSA named Irregular Attention (IrrAtt). This structure is built upon the BTD-(L,L,1) and aims to sparsify pre-trained Vision Transformers by pruning the query and key (QK) contraction dimension in the MHSA block. This paper also presents an algorithm for rank selection based on the structure of the fusion layer derived from the CP decomposition of original MHSA kernels. The main goal is to achieve better compression ratios without compromising the quality of the model.

### Strengths
1. Irregular Attention (IrrAtt) provides a new perspective on compressing the Multi-Head Self-Attention mechanism, especially for computer vision tasks. It is applicable to pretrained Transformer models, and holds substantial research value in the field of Machine Learning system and model deployment.
2. The Vision Transformer can be significantly sparsified by using BTD-(L,L,1) tensor decomposition for constructing IrrAtt, which results in a more compact model. It theoretically holds certain feasibility, and is thoroughly discussed.
3. The proposed rank selection algorithm, derived from the fusion layer structure, enables each attention head to have its optimal rank for the Query-Key (QK) contraction. 
4. The experiments have validated that the proposed method can achieve a good balance between performance maintenance and model compression.

### Weaknesses
1. The paper presents a number of conceptual and technical difficulties, and the ambiguous explanations make it challenging for readers to understand the paper. The introduction of the methods is not detailed enough, making it hard to grasp the true contributions of the author.
2. It may be necessary to carefully adjust the rank for the QK contraction in the IrrAtt for each attention head in order to achieve optimal performance.
3. The proposed rank selection algorithm, derived from the fusion layer structure, may result in extra computing overhead, particularly when the model is being trained.
4. Although validation on a single dataset can provide valuable insights for research, it's beneficial for the model's robustness and generalization to be validated on multiple datasets.
5. The paper did not conduct ablation experiments on various modules of the method, such as initialization, making it difficult for me to judge its true effectiveness.
6. There are several mistakes in the article, such as the MHSA formula in Equation 2. Please clarify or provide references if my understanding is incorrect. English abbreviations appearing for the first time in the paper should be followed by their full names or explanations.

### Questions
1. Are the comparison methods TruncAtt and CP SlimAtt first introduced in this paper? If not, please indicate the reference. If they are, there is a lack of comparison with existing methods, making it difficult to evaluate the level achieved by the proposed method.
2. How is the compression ratio controlled to reach the target compression ratio in this method? Are the significant discrepancies in the FLOPs and params CR values in Table 1 due to different target compression ratios set by the comparison methods?
3. The current version leaves me doubting its reproducibility.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a mechanism called Irregular Attention to compress the retrained Vision Transformer, which is built based on BTD-(L,L,1) tensor decomposition. The proposed method automatically determines the own rank of attention weight under the constraint of the total ranks of heads.

### Strengths
The paper proposes a method which has the same computation complexity as TruncAtt and performance similar to CP SlimAtt. 

The paper provides quantitative results to prove the effectiveness of the proposed method.

### Weaknesses
The paper is not very well written. For example, the abbreviation "CP" has made multiple prior appearances without prior explanation, only being clarified in Section 4.2, which may confuse the reader.

The paper evaluates the effectiveness of the proposed method based on the experimental results on DeiT and ILSVRC-2012 dataset, which is not comprehensive. The paper should conduct experiments on more datasets to have a more solid conclusion.

### Questions
Please refer to the weakness part.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to compress Transformers. It proposes the Irregular attention build on top of BTD-(L,L,1). It sparsifies pre-trained vision transformers by pruning the query and key contract dimensions in the MHSA. A fine-tuning scheme is also introduced to improve the performance. The proposed irregular attention is validated for DeiT on ILSVRC-2012 dataset. Experiments show the better results of the proposed method.

### Strengths
The studied problem of compressing vision transformers is quite important for the community.

The idea of diversing the importance of multi-attention heads in transformers is reasonable.

Using BTD-(L,L,1) in this problem is new.

### Weaknesses
The paper only evalutes on one dataset and one vision transformer. It needs more validations to support the paper's arguments.

The organization of experiments can be improved. Instead of presenting the results, it is better also to analyze the results and give the readers insights about the improvement if possible.

It seems that the paper did not compare to other compression methods. The methods compared in Table 1 are the preliminaries for the proposed one, however, many related works described in Section 5 are not compared.

The contributions listed in the end of Section 1 are not significant enough for an ICLR paper.

The importance of the obtained results and the derived method need to be further strengthed.

### Questions
Is there any evidence to support the assumption of (5)?

The way of presenting Algorithm 1-3 should be revised to improve its readability.

It lacks an overview of the proposed irregular attention, and how it can be used in existing vision transformer architectures.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes Irregular Attention (IrrAtt) for the compression of multi-head self-attention (MHSA) in vision transformers. IrrAtt is built on top of BTD-(L,L,1) tensor decomposition and is aimed at sparsifying pre-trained vision transformers by pruning the query and key contraction dimension in the MHSA block. The proposed IrrAtt is validated for the DeiT architecture on the ILSVRC-2012 dataset.

### Strengths
The compression of vision transformers is an important research problem. This paper is well-motivated. The writing is professional and convincing.

### Weaknesses
The comparison with existing methods is very limited, i.e., only three methods are compared. Considering that there has been a very large literature on the compression and efficient design of vision transformers, such a limited comparison cannot demonstrate the effectiveness of the proposed method.

The proposed IrrAtt is only validated for the DeiT architecture. Considering that there have been many popular transformer architectures such as Swin Transformer, PVT, and MViT, the only validation for DeiT cannot demonstrate the effectiveness of the proposed method.

There is no ablation study in this paper. This paper has many designs and components (Eq. (1) – Eq. (7), Alg. 1 - Alg. 3), and it is important and necessary to evaluate each of these designs and components. Recently, ablation study is also a necessary part of computer vision papers, especially for deep learning papers.

Will the code be released? This is not mentioned in the paper. This is important to ensure the reproducibility of the paper.

### Questions
Please see the above weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
