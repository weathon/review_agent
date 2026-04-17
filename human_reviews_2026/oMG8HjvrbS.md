# Theoretical refinement of CLIP by utilizing linear structure of optimal similarity

- Decision: Reject
- Scores: 6, 4, 6, 2, 8

## Abstract
In this study, we propose an enhancement to the similarity computation mechanism in multi-modal contrastive pretraining frameworks such as CLIP. 
Prior theoretical research has demonstrated that the optimal similarity metrics between paired modalities should correspond to the pointwise mutual information (PMI) between the two modalities.
However, the current implementations of CLIP and its variants fail to fully utilize the underlying linear structure of PMI.
We therefore propose KME-CLIP, which leverages this structure through the inner product in a reproducing kernel Hilbert space.
We theoretically prove that our method can approximate PMI with arbitrary accuracy and empirically demonstrate that our approach overall outperforms the standard CLIP formulation across several retrieval and classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies that under certain condition - pointwise mutual information (PMI) possesses a linear structure in an inner product of L2 norm which original CLIP fails to exploit. The authors proposes a KME-CLIP that projects embedding into a Kernel Hilbert space where they can exploit those PMI information.

### Strengths
1. Finding that under certain condition - pointwise mutual information (PMI) possesses a linear structure in an inner product of L2 norm which original CLIP work fails to exploit and thereby proposing a solution - is a novel contribution
2. The paper is concise and well written

### Weaknesses
1. I think a bit more insight on the related work is helpful. For example, this work does not refer a related work https://arxiv.org/abs/2409.13079 and how this might be different
2. The results in Table 1 shows the authors approach is better in most benchmarks compared to CLIP or WPSE. But it is hard to understand the actual impact without some quantitative example.
3. The paper has novel contribution but not clear how impactful the contribution is. What would it take application builders to reject CLIP and start using KME-CLIP?

### Questions
Please see the weakness

### Soundness
2

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
This paper proposes KME-CLIP, an enhancement to CLIP's similarity function. The core idea is that the optimal similarity metric, Pointwise Mutual Information (PMI), is not well-approximated by CLIP's standard cosine similarity. The authors' key insight is that under a conditional independence assumption, the exponential of PMI can be expressed as an inner product in an $L^2$ space. To exploit this, KME-CLIP represents image and text embeddings as kernel mean embeddings in a Reproducing Kernel Hilbert Space. Theoretical results and experiments verifies their advantage towards current methods.

### Strengths
The paper provides sufficient theoretical proof that KME-CLIP can approximate PMI with arbitrary accuracy. The method theoretically and empirically outperforms standard CLIP and WPSE on different tasks.

### Weaknesses
1. Assumption 2 is quite hard to hold for real data distribution. It is highly unlikely that a single latent topic $z$ is sufficient to make a complex image and its detailed textual description completely independent.
2. The motivation of the paper could be problematic. It is built on the current methods' non-linear approximation of $exp(PMI)$, including CLIP's. However, Proposition 2 indicates that the CLIP similarity can be written as a one-point version of the proposed KME-CLIP. This violates the initial motivation. The effectiveness may only come from the introduction of multiple embeddings.
3. The proposed similarity calculation requires $O(m_X m_Y)$ kernel computations for each pair in the $B \times B$ similarity matrix. This leads to higher computational and hardware requirements as shown in Appendix Table 7.

### Questions
When switching to WPSE's similarity function in the current framework, how would the model perform?

Please also see Weaknesses.

### Soundness
2

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
3

### Summary
This paper introduces KME-CLIP, a novel method that refines the CLIP framework by theoretically identifying and exploiting the linear structure of the exponential Pointwise Mutual Information (exp(PMI)) in a Reproducing Kernel Hilbert Space (RKHS). The authors provide theoretical guarantees on the approximation capability of their method and demonstrate consistent, albeit sometimes modest, empirical improvements over baselines across a range of standard multimodal tasks.

### Strengths
1. The paper presents a well-formulated and mathematically grounded connection between PMI, RKHS, and contrastive learning objectives.
2. The authors clearly delineate differences and relationships between CLIP, WPSE, and their proposed model.
3. KME-CLIP shows small but stable improvements in retrieval and zero-shot classification across a variety of datasets.
4. The exposition is clear and professional.

### Weaknesses
1. The empirical methodology closely follows existing work (especially WPSE), with only a minor modification in similarity computation.
2. The observed performance gains are small (typically 1–3%), raising questions about practical significance relative to computational overhead.
3. The paper exclusively uses Gaussian kernels without justifying why other positive definite kernels (e.g., Laplacian) are not considered, nor exploring their impact on performance.
4. The paper lacks a detailed evaluation of training/inference cost or memory footprint, which is critical for kernel-based approaches.

### Questions
1. Why is the linear classification performance not improved despite a superior PMI approximation? Could post-RKHS projection features or modified feature extraction pipelines address this gap?
2. How sensitive is the method to the choice of kernel and its parameters? Are there scenarios where alternative kernels outperform Gaussian kernels?
3. Can KME-CLIP be efficiently applied to large-scale pretrained CLIP models without full retraining?
4. For practical deployment, what is the recommended point set size to balance performance gains and inference cost? Is there a heuristic to determine this size for different datasets/tasks?
5. Does the RKHS similarity mitigate or exacerbate the “modality gap” observed in prior CLIP analyses?
6. Could the RKHS log-inner-product formulation be combined with in-modal self-supervision (e.g., as in SLIP)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to replace the point-wise contrastive loss of CLIP with a dense contrastive loss calculated using the idea of PMI. 

The authors show that this leads to better results (using the same evaluation as CLIP). 

The method is rigorously motivated by theoretical analysis.

### Strengths
- The paper is well presented and the idea is valid.
- Figure 1 is a clear illustration of the method. Using the intermediate features of the encoders rather than just the output features is innovative.
- The authors include a Section 3.4 comparing their method with the baselines. This is helpful for readers to compare with existing literature.

### Weaknesses
The main issue is that this paper is substantially similar to [Uesaka et al. ICLR 2025]
 - The authors acknowledge this in Section 3.4.2. The distinctions between the current work and [Uesaka et al. ICLR 2025] are minimal.
 - Please compare Figure 1 of the two papers - I think the idea being presented is identical, with minor differences in implementation.
 - The theoretical contributions are different.

I am ok with acceptance if the above concern is not valid after discussion.

Secondary concerns:
- The paper only compares against CLIP and WPSE. There have been many improvements to the CLIP loss. [Dong], for example.
   - While these other variants of CLIP do not try to approximate PMI, they should be used as baselines.


[Dong] Don et al. CVPR 2023. MaskCLIP: Masked Self-Distillation Advances ContrastiveLanguage-Image Pretraining.

### Questions
- In the introduction, what do you mean by " It is theoretically known that the similarity function optimal in terms of the contrastive loss corresponds to the pointwise mutual information (PMI), defined as log(p(x, y)/(p(x)p(y))) for two datapoints x and y from different modalities (Oord et al., 2018; Zhang et al., 2023)."?
How do we defined the optimal similarity function here?
I think more discussion about why we want to use PMI instead of other contrastive losses would be helpful before delving into the details of how to optimize PMI better. Thank you.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper derives a theory-backed similarity metric for comparing two modalities in CLIP-type models. The main idea is to use approximate pointwise mutual information (PMI), which is implemented via kernel mean embeddings utilizing the linear structure of exp(PMI). The results show consistent improvement over similar CLIP-type methods.

### Strengths
1. The derivation is neat and backed by theory. The insight of the linear structure of PMI with the conditional independence assumption makes sense, and this seems to make a major difference in practice -- comparison against WPSE.
2. Significant and consistent improvements against WPSE and CLIP in various retrieval and zero-shot classification tasks.
3. Writing is clear, and the arguments are adequately motivated.

### Weaknesses
1. The low modality gap assumption in Section 4.2 may not be realistic [1]. It would be better to clarify why this assumption is needed and what happens in scenarios where this is violated.
2. Related to this, would the proposed method yield a lower modality gap? 
3. How does the choice of kernel function affect the results or approximation quality?

[1] Liang, Victor Weixin, et al. "Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning." Advances in Neural Information Processing Systems 35 (2022): 17612-17625.

### Questions
1. The proposed method does not seem specific to the hypersphere. Could other geometric spaces be used to obtain features? 
2. Would it make sense to consider this similarity metric under the paradigm of accepting the modality gap similar to [2]?

[2] Ramasinghe, Sameera, et al. "Accept the modality gap: An exploration in the hyperbolic space." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Soundness
4

### Presentation
3

### Contribution
3
