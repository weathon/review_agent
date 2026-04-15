# Contextual Vision Transformers for Robust Representation Learning

- Decision: Reject
- Scores: 6, 5, 5, 5

## Abstract
We introduce Contextual Vision Transformers (ContextViT), a method designed to generate robust image representations for datasets experiencing shifts in latent factors across various groups. Derived from the concept of in-context learning, ContextViT incorporates an additional context token to encapsulate group-specific information. This integration allows the model to adjust the image representation in accordance with the  group-specific context. Specifically, for a given input image, ContextViT maps images with identical group membership into this context token, which is appended to the input image tokens. Additionally, we introduce a context inference network to predict such tokens on-the-fly, given a batch of samples from the group. This enables ContextViT to adapt to new testing distributions during inference time. We demonstrate the efficacy of ContextViT across a wide range of applications. In supervised fine-tuning,  we show that augmenting pre-trained ViTs with our proposed context conditioning mechanism results in consistent improvements in out-of-distribution generalization on iWildCam and FMoW. We also investigate self-supervised representation learning with ContextViT. Our experiments on the Camelyon17 pathology imaging benchmark and the JUMP-CP microscopy imaging benchmark demonstrate that ContextViT excels in learning stable image featurizations amidst distribution shift, consistently outperforming its ViT counterpart.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces Contextual Vision Transformers (ContextViT), a method to address structured variations and distribution shifts in image datasets. It leverages context tokens and token inference models to enable robust feature representation learning across groups with shared characteristics. The paper provides evidence of ContextViT's effectiveness through experiments in gene perturbation classification and pathology image classification.

### Strengths
- The paper introduces a novel method, ContextViT, to address structured variations and distribution shifts in image datasets. It brings a unique perspective to the problem of improving feature representations for vision transformers.
- The paper is well-written and provides clear explanations of the methodology, experiments, and results.
- ContextViT is extensively evaluated in different tasks, showcasing its effectiveness in improving out-of-distribution generalization and resilience to batch effects.

### Weaknesses
- How to chose and define the "in-context" prompt is unclear.
- While the paper is well-structured and well-written, it would be beneficial to include more detailed comparisons with related work to highlight the novelty of the proposed approach.
- In the "Out-of-Distribution Generalization (Pathology Images)" section, it's not entirely clear what "linear probing accuracy" means and how it relates to out-of-distribution generalization. A more in-depth explanation of this metric would improve the clarity of the paper.

### Questions
- Are there any specific use cases or domains where ContextViT is particularly well-suited, and are there any limitations or scenarios where it may not perform as effectively?
- Could the authors provide more insights into how ContextViT's approach to handling structured variations and distribution shifts could be applied in practical applications outside of the ones discussed in the paper?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a Contextual Vision Transformers (ContextViT) based on ViT. ContextViT is designed for adapting ViTs to OOD data with varying latent factors. This work is inspired by in-context learning and prepends tokens to input sequences for alleviating model performance. This paper finds out that standard context tokens might not be able to generalize to unseen domains, therefore it proposes a context inference network that estimates context tokens from input images. The proposed method is evaluated with cell-imaging and histopathology datasets and achieves performance improvements under distribution shifts.

Pros:

- This paper is well-written and easy to follow.
- Figure 1 is well drawn to illustrate the overall idea of this work.
- Layer-wise context conditioning is well-motivated and makes sense.

Cons:

The novelty of this work is limited.
- The intrinsic difference between this work and visual prompting [1] is unclear. It seems that visual prompting can also fit this OOD scenario. 
- The key idea of this work is similar to [2], which also uses a network to predict the context/domain tokens.
- The comparison in experiment section is insufficient.
- Lack of visualization of the learned context token, which shows the difference of context tokens of different groups.

The paper is simple and effective, but its novelty is unfortunately limited, and analysis for the insight of this approach is absent.

[1] Jia, Menglin, et al. "Visual prompt tuning." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
[2] Zhang, Xin, et al. "Domain Prompt Learning for Efficiently Adapting CLIP to Unseen Domains." arXiv preprint arXiv:2111.12853 (2021).

### Strengths
- This paper is well-written and easy to follow.
- Figure 1 is well drawn to illustrate the overall idea of this work.
- Layer-wise context conditioning is well-motivated and makes sense.

### Weaknesses
The novelty of this work is limited.
- The intrinsic difference between this work and visual prompting [1] is unclear. It seems that visual prompting can also fit this OOD scenario. 
- The key idea of this work is similar to [2], which also uses a network to predict the context/domain tokens.
- The comparison in experiment section is insufficient.
- Lack of visualization of the learned context token, which shows the difference of context tokens of different groups.

The paper is simple and effective, but its novelty is unfortunately limited, and analysis for the insight of this approach is absent.

### Questions
-

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes ContextViT to address the distribution shift between different datasets. ContextViT uses a context inference model taking the dataset as input to get a context embedding for the dataset, and predicts the label conditioned on the context embedding (token). It also makes this process layer-wise to capture different-scale distribution shift.

### Strengths
The paper presents a method to mitigate the distribution gap between different datasets. Based on their experimental results, the proposed method, ContextViT, has the ability to improve the performance under distribution shift.

### Weaknesses
- The paper mentioned that the proposed method applies the concept of in-context learning in vision transformer. However, in my opinion, in-context learning is a kind of few-shot learning, which predicts based on the (data, label) pair of a few samples, unlike the usage of all the dataset-c data (or a batch of the data) in this paper. The method looks like a summarization of the dataset information and then makes the prediction based on that summarization.

- The method requires a lot of distribution-c data at the inference stage and increases the inference overhead. 

- The oracle-context model is very similar to some prompt tuning works, like Visual Prompt Tuning & Prompt Learning for Vision-Language Models, but these works are not discussed in the paper.

### Questions
Please see weaknesses.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces an improved ViT where some group-specific context information from the sub-groups in datasets is collected and generated from those images in a group. The network generates the context token from those images and appends them to image patch embeddings. Their experiments show some improvement of this ViT on some group-specific datasets.

### Strengths
The idea of capturing context information from the datasets is interesting.
The writing of this method is clear and easy to follow.
The experiments demonstrate the efficiency of their proposed framework on both the dataset with the same distribution and other datasets with different distributions.

### Weaknesses
The view and impact of this paper are limited. It seems the method focuses on improving the performance of the datasets that contain several distinct groups. Although the authors demonstrate improvements on some specific datasets, the improvement in general image tasks is still unclear. It is suggested to widely evaluate their framework on other popular datasets and tasks or extend related techniques to improve the capability of transfer learning from one task to some other tasks. It should also be compared with more related works.


Despite the proposed contextual learning paradigm, the technical contributions in this paper are limited and not novel enough.


Some unclear presentations:

1. Figure 1 is unclear and somehow misleading. The source of the context (where those images come from) and the function (input, output) of the inference model should be labeled. I strongly suggest redoing this figure.

2. The end of page 5 is missing.

3. Table 2 looks messy and should be redesigned.

### Questions
What would the performance be if we want to apply this framework to a large dataset that was combined with several small datasets?

If we don't know the sub groups of the data, is there anyway to benefit from the proposed framework?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
