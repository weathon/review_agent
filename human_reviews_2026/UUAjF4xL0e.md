# Distributional Vision-Language Alignment by Cauchy-Schwarz Divergence

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 10

## Abstract
Vision-language alignment is crucial for various downstream tasks such as cross-modal generation and retrieval. Previous multimodal approaches like CLIP utilize InfoNCE to maximize mutual information, primarily aligning pairwise samples across modalities while overlooking distributional differences.  In addition, InfoNCE has inherent conflict in terms of alignment and uniformity in multimodality, leading to suboptimal alignment with modality gaps. To overcome the limitations, we propose CS-Aligner, a novel framework that performs distributional vision-language alignment by integrating Cauchy-Schwarz (CS) divergence with mutual information. CS-Aligner captures both the global distribution information of each modality and the pairwise semantic relationships. We find that the CS divergence seamlessly addresses the InfoNCE's alignment-uniformity conflict and serves complementary roles with InfoNCE, yielding tighter and more precise alignment. Moreover, by introducing distributional alignment, CS-Aligner enables incorporating additional information from unpaired data and token-level representations, enhancing flexible and fine-grained alignment in practice. Experiments on text-to-image generation and cross-modality retrieval tasks demonstrate the effectiveness of our method on vision-language alignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that the widely used InfoNCE in CLIP learning ignores distributional differences and has inherent conflict in terms of alignment and uniformity, which result in suboptimal alignment and modality gap. To tackle these limitations, the paper proposes CS-Aligner, that integrates Cauchy-Schwarz divergence with InfoNCE, leverages unpaired data and incorporates distributional token-wise alignment. Experiments on text-to-image generation and cross-modality retrieval tasks demonstrate the effectiveness of CS-Aligner

### Strengths
1. The paper proposes a novel framework, CS-Aligner, to reduce modality gap for better vision-language alignment.
2. The idea of using Cauchy-Schwarz divergence is ingenious, which can simultaneously tackle the two limitations: ignoring of distributional differences and alignment/uniformity confliction. Authors also provide comprehensive theoretical analysis to support it.
3. Experiments on several multi-modal/cross-modal tasks show the effectiveness of CS-Aligner.

### Weaknesses
1. The Cauchy-Schwarz divergence constitutes the core contribution of this work. It is therefore essential to thoroughly compare it against other distribution distance measures—such as KL/JS divergence and the Wasserstein distance. While the authors discuss their differences in the Appendix, they provide no quantitative comparisons.

2. A main ablation table should be included to quantitatively evaluate the effectiveness of each component in CS-Aligner. Moreover, the ablation study should extend beyond text-to-image generation to include more tasks, which would help comprehensively assess the impact of individual components.

3. The authors should clarify how the visualizations in Figure 1 were derived, particularly regarding the dataset and experimental settings used. For qualitative analysis, it would be beneficial to also present results on other tasks—such as cross-modal retrieval and image captioning—to better demonstrate the method's general applicability.

4. Regarding Table 7, why larger lambda leads to worse results? How about the sensitivity of lambda and sigma on other tasks?

Minor issue: In Line 431, cross-model -> cross-modal

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents CS-Aligner, a novel framework for vision-language alignment that combines Cauchy–Schwarz (CS) divergence with mutual information (InfoNCE). The central idea is to extend alignment beyond individual pairs by introducing distributional alignment, which effectively resolves the inherent tension between the alignment and uniformity objectives in InfoNCE. While InfoNCE encourages matched pairs to come closer (alignment) and all samples to spread apart (uniformity), this dual objective often leads to suboptimal multimodal representation. CS-Aligner addresses this issue by aligning both the pairwise semantics and the global distributions of visual and textual modalities, resulting in more coherent and consistent cross-modal embeddings.

### Strengths
1.  Clearly identifies InfoNCE’s limitations—its conflation of alignment and uniformity objectives and dependence on strong negatives that weaken cross-modal consistency.

  2.  Introduces CS divergence as a novel, theoretically grounded approach for distributional alignment, effectively addressing the shortcomings of sample-level contrastive methods.

### Weaknesses
1.  The experiments are mainly focused on t2i and i2t tasks, which are not sufficient to verify the effectivess. 

  2. The ablation studies in Appendix H.1 are insufficient and unconvincing. The experimental settings are not clearly defined. Furthermore, the paper fails to adequately explain the core concepts of InfoNCE and CS-Aligner, and does not justify the significant performance degradation observed when using only InfoNCE. While these experiments are in the appendix, it is essential to treat them with the same seriousness as those in the main text.

  3. The performance of the CS divergence is sensitive to the choice of kernel and bandwidth, but the paper does not provide a robustness analysis to assess this sensitivity.

### Questions
1. It would be better to include more multimodal tasks to verify the effectiveness, as the proposed method is designed for multimodal alignment.

2. Can CS-Aligner generalize to other modality pairs (e.g., audio–text)?

3. Are there scenarios where distributional alignment might unintentionally blur modality-specific distinctions？

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces CS-Aligner, a novel vision-language alignment framework that combines Cauchy-Schwarz divergence with InfoNCE to address the alignment–uniformity conflict. It enables both global distribution-level and fine-grained token-level alignment, supports unpaired and multi-caption data, and achieves superior performance on generation and retrieval tasks, particularly in low-resource scenarios.

### Strengths
1. The authors integrate Cauchy-Schwarz divergence with mutual information (InfoNCE) and provide a theoretical analysis of how it alleviates the alignment–uniformity conflict, addressing limitations in InfoNCE-based alignment.
2. Leveraging distribution-level alignment, the method naturally supports unpaired data and multi-caption supervision, offering greater flexibility and scalability for real-world applications.
3. The use of lightweight adapters (e.g., Adapter, LoRA) enables efficient alignment with CLIP and LLMs, significantly reducing training cost and making the method practical for resource-constrained settings.

### Weaknesses
1. Although the method is well-motivated, the theoretical analysis of convergence, generalization, or stability of CS-Aligner remains shallow and largely empirical.
2. The KDE-based CS divergence may introduce additional computational overhead and sensitivity to kernel bandwidth selection, which is not systematically analyzed.
3. Some comparisons (e.g., Eclipse, LLM2CLIP) reuse decoders from other works, which may not fully isolate alignment quality from downstream generative components.
4. The benefits of using unpaired data or token alignment are demonstrated qualitatively, but quantitative ablations are limited and lack statistical rigor.

### Questions
1.	How sensitive is the CS divergence estimation to kernel bandwidth (\sigma) and the choice of kernel function?
2.	Does the KDE-based divergence scale efficiently to very large multimodal datasets (e.g., LAION-400M)?
3.	Can the proposed distributional alignment be combined with optimal transport or contrastive regularization for improved interpretability?
4.	How does token-level CS alignment affect semantic disentanglement across layers? Is it stable under random seed variation?
5.	Does the framework ensure mutual consistency between global (distributional) and local (pairwise) alignment objectives?
6.	Could the approach be extended to temporal or 3D multimodal tasks (e.g., video–text, audio–image)?
7.	Given its use of CS divergence, can the model handle multimodal imbalance (e.g., text richer than image features)?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
This paper tackles an important but overlooked problem in InfoNCE used in CLIP when aligning text features and image features. The authors revealed that InfoNCE
has inherent conflict in terms of alignment and uniformity in multimodality and demonstrated in Figures 1 and 2 that this is the cause of the modality gaps. Therefore, in this paper, a new method called CS-Aligner was proposed that combines InfoNCE with Cauchy-Schwarz (CS) divergence. Two standard experiments in image-text alignment are done to validate the proposed CS-Aligner: text to image generation and image-text retrieval. In the text to image generation, it has been experimentally demonstrated that the proposed CS-Alignment can achieve better performance than previous large-scale methods and small-scale alignment (Table 1). Its generality has also been validated by using different datasets (Table 2). In the image-text retrieval task, it was shown that CS- Aligner outperforms previous approaches such as Long-CLIP (Zhang et al., 2025) and LLM2CLIP (Huang et al., 2024).

### Strengths
The paper is described in detail. The proposed method is mathematically solid. This paper gives important insight to the InfoNCE loss and proposes an important method that would give great impacts to the multimodal research community. 

Experimental results both in the main body and supplementary are convincing enough.

### Weaknesses
The proposed method has some parameter sensitivity as shown in Table 7 in the supplemental material. I do not think the method is parameter-independent as the authors claim. 
- Could you show soma sample images to show the method is robust to parameter changes, in addition to showing FID?
- Is there any method that can automatically find optimal parameter settings rather than relying on users’ empirical optimization?

Some minor modification proposals (no need to reply)
- A missing period after eq. (21)
- A missing period, a missing eq. number, and erroneous “?” in lines 911-913
- “??” in line 1065 should be modified

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
4
