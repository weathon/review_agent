# Prototype-based Regularization Learning For Text-Video Retrieval

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
This work addresses the Intra-Inter Conflict (IIC) dilemma in text-video retrieval, _i.e._, (a) intra-category variance, refers to category-consistent instances that display substantial distributional disparity, and (b) inter-category similarity, where instances belonging to different categories exhibit distributional coupling. Through an analysis of the learned feature and recalled samples of current models, we posit this conflict stems from the appearance bias issue, _i.e._, the matching process is dominated by superficial semantics shared across samples, which undermines the contribution of discriminative semantics. To this end, we propose Prototype-based Regularization Learning (PRL), which regularizes the semantic boundaries of features through a set of prototypes, so as to maximally compel the model to learn compact and distinctive representations for text-video retrieval task. Specifically, PRL performs within- and cross-instance clustering in the embedding space to assign informative prototypes to instances with similar categories.  Moreover, a Prototype Discriminating Loss (PDL) is proposed that makes semantically correlated instances self-organize around their respective prototype while maintaining separation across different ones, and a Prototype Projection Loss (PPL) is devised to align video and text features by adaptively projecting prototypes into a shared semantic manifold, thereby fostering cross-modal correspondence.
Extensive experiments on five datasets demonstrate that the proposed model-agnostic strategy significantly boosts the performance of existing models, _e.g._, improving TempMe, X-Pool, and CLIP4Clip by +6.5%, +3.1%, and +5.0% of SumR on the MSR-VTT dataset.
Code available at: https://anonymous.4open.science/r/PRL-200D.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper identifies a key problem in text-video retrieval called the Intra-Inter Conflict (IIC) dilemma: Intra-category variance: Videos/texts of the same category (e.g., "drinking") have inconsistent feature representations. Inter-category similarity: Videos/texts from different categories (e.g., "drinking" vs. "pouring") have entangled features. 

To solve this, the authors propose Prototype-based Regularization Learning (PRL), a model-agnostic framework that uses clustering to find category-level "prototypes." It then uses two loss functions: (i) Prototype Discriminating Loss (PDL), which clusters similar instances around their prototype and pushes different prototypes apart. (ii) prototype Projection Loss (PPL), which aligns video and text prototypes in a shared semantic space.

### Strengths
**Clear Problem Defination**: This paper provides a clear analysis and nomenclature (IIC) for a widespread but often overlooked issue in retrieval models.

**Model-Agnostic and Flexible**: PRL is designed as a plug-and-play component that can be integrated into various existing models (e.g., CLIP4Clip, X-Pool) without major architectural changes.

**Effectiveness**: The method demonstrates significant and consistent performance improvements across five different datasets and multiple strong baseline models.

### Weaknesses
There are relatively few references, lacking some famous methods in this field. 

I will consider increasing the score if other reviewers do not have so much concern.

### Questions
How does this work compare to vision-language-model for embedding such as VLM2Vec, LLaVE and UNITE? These VLM-based retrieval methods are good at semantic matching.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to solve the Intra-Inter Conflict (IIC) dilemma in text-video retrieval. Through analysis, the authors find that the root cause of this problem is "appearance bias", where the model tends to focus on superficial semantics shared across samples while ignoring truly discriminative semantics.

To address this issue, the paper proposes a framework called "Prototype-based Regularization Learning" (PRL). PRL is a model-agnostic and orthogonal module that can be plugged-and-played into existing TVR models.

The core mechanisms of PRL include:

Clustering-based Prototype Mining, In each training mini-batch, K-means clustering is performed on video features to construct Video Prototypes, and paired text features are grouped into Text Prototypes.
1.  Prototype Discriminating Loss (PDL):** This loss function pushes different prototypes apart at the "Category-level" and encourages instances (hard negatives) within the same cluster to repel each other at the "Instance-level".
2.  Prototype Projection Loss (PPL): This loss function is used to address the ambiguity of text descriptions (e.g., "a person is walking"). It enhances cross-modal correspondence by pulling all text features within a cluster closer to their corresponding "cross-modal prototype".

Experimental results show that PRL improves the performance of six mainstream TVR baseline models (e.g., CLIP4Clip, X-Pool, TempMe) on five benchmark datasets, including MSR-VTT, MSVD, and DiDeMo. For example, on MSR-VTT, PRL brings a +6.5% and +5.0% SumR improvement to TempMe and CLIP4Clip, respectively. The method is also shown to be generalizable to image-text retrieval tasks.

### Strengths
1.  As a "plug-and-play" regularization module, PRL has high practical value. It does not propose a completely new, large-scale model, but rather regularizes the embedding space of existing models by mining prototypes, forcing the model to learn more compact and distinctive representations.
2.  High computational efficiency: PRL's biggest highlight is its efficiency. Experiments (Table 4) show that it introduces minimal training time (e.g., +0.59min) and memory overhead (e.g., +3MB) while adding 0 learnable parameters.

### Weaknesses
1.  The motivation of the paper is based on video retrieval, but the problems raised are widely present in clip-based retrieval tasks, which does not highlight the necessity of video retrieval as the main task.
2.  PRL's prototype mining is based on K-means performed on each mini-batch. If the data distribution within a mini-batch is biased, the quality of the mined prototypes may degrade. The paper does not discuss how to ensure clustering stability or analyze the impact of different Batch Sizes on prototype quality.
3.  The video prototypes are mined through *unsupervised* K-means clustering and do not use true category labels.

### Questions
1.  The paper claims that the model tends to focus on superficial semantics shared across samples. However, in the example in Figure 1 (bottom right), the top-5 retrieval results are all related to "sports," which aligns with the category. The unsupervised clustering operation does not seem to directly solve this issue. The authors need to provide statistical results with PRL , consistent with Figure 1 (top right).
2.  Some papers [1] state that CLIP focuses on cross-modal differences rather than intra-modal differences. Does this phenomenon lead to clustering instability? The authors need to statistically analyze the clustering results for visual and text modalities separately and compare their similarities and differences.
3.  CLIP exhibits  "bag-of-words" [2]. The reviewer is concerned that clustering might amplify this problem. Please provide an explanation.
4. Video retrieval pays more attention to the temporal problem, and the authors are asked to explain the necessity of using clustering methods in video tasks rather than image retrieval tasks.

[1] CLIP Adaptation by Intra-modal Overlap Reduction
[2] When and why vision-language models behave like bags-of-words, and what to do about it?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Prototype-based Regularization Learning (PRL) for text-video retrieval. The method addresses the Intra–Inter Conflict (IIC) dilemma—how to maintain compactness within categories while ensuring clear separability between categories in joint video–text embedding spaces.
PRL introduces a clustering-based prototype mining approach with two new regularization losses: 1) Prototype Discriminating Loss (PDL) for semantic separation, and 2) Prototype Projection Loss (PPL) for adaptive cross-modal alignment.

The framework is model-agnostic and can be easily integrated into existing retrieval models. Experiments across five major datasets and multiple architectures show consistent performance gains. The paper also includes ablation studies and visualizations to support its claims.

### Strengths
1.  The paper gives a strong motivation for the IIC dilemma, supported by both visual and analytical explanations (see Figure 1, Page 2). The empirical analysis convincingly demonstrates drift and entanglement issues in current approaches.

2.  PRL is plug-and-play, requiring minimal architectural changes and maintaining efficiency.

### Weaknesses
1.  The paper does not sufficiently compare PRL with recent prototype-based or IIC-focused methods. Several closely related works [a-c] are absent from both the discussion and experiments. This omission weakens the paper’s claim of novelty and superiority.

[a]  Zeng Z, Wang S, Xu N, et al. Pan: Prototype-based adaptive network for robust cross-modal retrieval[C]//Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval. 2021: 1125-1134.

[b]  Liu Y, Chen Q, Albanie S. Adaptive cross-modal prototypes for cross-domain visual-language retrieval[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021: 14954-14964.

[c]  Li H, Song J, Gao L, et al. Prototype-based aleatoric uncertainty quantification for cross-modal retrieval[J]. Advances in Neural Information Processing Systems, 2023, 36: 24564-24585.

2.  The clustering and assignment process in Section 2.2 is underspecified. It is unclear how prototypes and cluster memberships are maintained across batches or epochs, and how issues like empty clusters or ties are handled. Some loss definitions (e.g., Eq. 5) lack clarity on normalization and contrastive sampling, especially for small clusters.

### Questions
1. How does batch-wise K-means handle non-uniform batch composition? Are prototypes re-initialized each batch or tracked across epochs?

2. Have you tried soft or probabilistic clustering (e.g., GMM)? Why exclusively use hard K-means?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work studies the task of text-video retrieval with a new method called prototype-based regularization learning. The proposed method is motivated by a new point termed as intra-inter conflict dilemma, where the category-consistent instances that display substantial distributional disparity and the  instances belonging to different categories exhibit distributional coupling. The proposed method is applied on a couple of methods and see some performance boost.

### Strengths
* This work study the complex representation space of the multimodal data and aims to deliver more discriminative representations for retrieval, which is encouraging. 

* The proposed method is applied on a number of methods and extend to text-image retrieval, which show some generalization ability. 

* The introduction and method are well-written.

### Weaknesses
* Despite the new concept of the Intra-Inter Conflict (IIC) dilemma is interesting, there are some concerns on it. It seems that the proposed dilemma lacks generalizability. Fig.1 shows the observation on MSRVTT and upon DisCoVLA. It is unclear if this dilemma is specific to the DiscoVLA or MSRVTT. More empirical evidence from other datasets or preliminary methods is needed to justify the proposed claim. 

* The performance boost of the proposed method is generally marginal, doubting the necessity of the proposed challenge and the effectiveness of the proposed method.  

* This work suggests that the semantics from the same category should be  more similar, while the ones from different categories should be less similar. I'm not sure whether this assumption always holds considering the complex behavior of the representation space. 

* Besides, I'm not sure whether this assumption is always consistent with the contrastive learning objective. More empirical evidence and justifications are required to admit this intuition.

### Questions
* How the category in the proposed motivation is defined? To what extent some concepts can be regarded as the same category, for example, whether two dogs with different breeds belong to the same category? Or whether the two verbs that describe the same action belong the same category? Since this lay the foundation for the motivation and the proposed method, a more rigorous (or quantitative) definition of it is expected beyond the conceptual illustration.  

* On some dataset, such as ActivityNet, the proposed method does not benefit but fails with CLIP3Clip, a failing case analyzation would be helpful. 

* How to determine the number of the prototypes across different methods and datasets? Why more count does not lead to more fine-grained representation learning and consistent improvement?

### Soundness
3

### Presentation
3

### Contribution
3
