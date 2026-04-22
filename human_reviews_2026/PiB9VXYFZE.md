# UniFast-HGR: Scalable and Efficient Maximal Correlation for Multimodal Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
This paper introduces UniFast-HGR, an efficient and scalable framework for estimating Hirschfeld-Gebelein-Rényi maximal correlation in multimodal learning. The method addresses computational bottlenecks in traditional HGR and Soft-HGR approaches, which suffer from $O(K^3)$ complexity due to covariance matrix inversion and limited scalability to deep architectures. UniFast-HGR incorporates three key innovations: replacing covariance with cosine similarity to avoid matrix inversion, removing diagonal elements to mitigate self-correlation bias, and applying $\ell_2$ normalization as a variance constraint for improved stability. These improvements reduce computational complexity to $O(m^2K)$ while maintaining bounded correlation scores. The OptFast-HGR variant further accelerates computation by simplifying normalization steps, achieving dot-product-level efficiency with minimal accuracy loss. Experimental evaluations across benchmark datasets validate the framework's ability to balance computational efficiency with accuracy, establishing it as an effective solution for addressing contemporary deep learning challenges.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes the UniFast HGR framework, which optimizes HGR maximal correlation estimation through three key improvements: variance constraints, replacing covariance with cosine similarity, and removing the diagonal. The OptFast HGR variant is also introduced, further improving computational efficiency and making it suitable for large-scale data processing.

### Strengths
1. It supports multimodal extensions, capable of handling feature pairs from two or more modalities, and demonstrates strong performance in tasks such as emotion recognition, remote sensing image classification, and segmentation.

 2. The framework also shows robustness to missing modalities or incomplete labels.

3. OptFast HGR improves computational efficiency, suitable for large-scale multimodal data.

### Weaknesses
1.The paper mainly compares with HGR, Soft-HGR, and CCA series methods, which are relatively old. Are there no more recent works in this area?

2. Memory consumption is not discussed in detail. Although computational complexity is reduced, storing all modality distribution vectors and computing cosine similarities and trace terms may incur significant memory overhead, especially when training large-scale, high-dimensional models.

### Questions
1.Have the authors compared UniFast HGR or OptFast HGR with more recent correlation-based or multimodal fusion methods?

2. How significant is the memory overhead when handling high-dimensional multimodal inputs or large batch sizes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes UniFast-HGR, a computationally efficient variant of the Hirschfeld–Gebelein–Rényi (HGR) maximal correlation estimator for multimodal learning. Building upon the Soft-HGR framework, the authors reformulate correlation estimation using only cosine similarity while introducing a variance constraint. The proposed method claims to reduce computational complexity from $O(K^3)$ to $O(m^2K)$, where K denotes the feature dimension and m the minibatch size. Empirical evaluations on tasks such as image classification and remote sensing semantic segmentation reportedly demonstrate higher accuracy than prior HGR-based approaches.

### Strengths
- The paper introduces a cosine-similarity-based approximation of the HGR correlation, offering a more lightweight computation pipeline than the covariance-based Soft-HGR.
- The proposed UniFast-HGR shows improved performance over Soft-HGR and other correlation-based baselines on Berlin, Vaihigen, and Globe230k datasets.

### Weaknesses
# [W1] Marginal Impact of the Claimed Computational Cost Reduction

The paper claims that UniFast-HGR reduces the computational complexity from $O(K^3)$ to $O(m^2K)$. While Figure 1 includes a runtime comparison, this experiment is performed only on MNIST, a toy-scale dataset with trivial feature extractors. In realistic multimodal systems, the computational bottleneck arises from forward passes of the encoders f(x) and g(y), not from the correlation computation itself. Thus, demonstrating speedup on MNIST does not provide evidence that the proposed method improves efficiency in practical deep learning scenarios.

Furthermore, under typical large-scale configurations—such as CLIP [1], where m = 32,768 and K = 1,024—the proposed $O(m^2K)$ formulation would actually increase the computational cost, because the complexity grows quadratically with respect to the minibatch size. The paper provides no runtime measurements under realistic multimodal training settings (e.g., contrastive pretraining or large-scale multimodal fusion), making it unclear whether UniFast-HGR yields any efficiency benefits when training modern large models.


# [W2] Limited Experimental Scope (Tables 1,2)

Tables 1 and 2 evaluate UniFast-HGR exclusively on multimodal remote-sensing datasets (e.g., HSI–LiDAR). These datasets are small in scale (e.g., Vaihingen contains only 33 tiles; Globe230k ≈ 230k patches) and highly domain-specific. Such experiments do not demonstrate that the method generalizes to general multimodal learning.

If the paper intends to position UniFast-HGR as a broadly applicable multimodal framework, it must include scalable experiments on widely adopted multimodal benchmarks—e.g., COCO Captions for retrieval evaluation or LAION-400M for large-scale pretraining. Without validation on datasets at million-scale or beyond, the claimed generality remains unsubstantiated.


# [W3] Inconsistent and Unreliable Experimental Results (Tables 3,4, and 12)

Tables 3, 4, and 12 lack essential training details (e.g., batch size, optimizer, number of training epochs), making it impossible to evaluate reproducibility or fairness. Additionally, several reported baselines deviate significantly from established results.

- In DINOv2 [2], a ViT-L/14 achieves 86.3% top-1 accuracy on ImageNet-1K under linear evaluation, whereas this paper reports only 81.8%, suggesting misconfiguration or under-training.
- For IEMOCAP, the MultiEMO model [3] achieves a Weighted-F1 of 72.84, yet all baselines reported in Table 12 (CCA, Deep CCA, Soft CCA, Dot Product, Cosine Similarity, Soft-HGR) fall far below this.

These discrepancies raise concerns regarding experimental rigor and whether baselines were fairly tuned. As a result, the reliability of the claimed improvements remains unclear.


[1] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PmLR, 2021.

[2] Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." arXiv preprint arXiv:2304.07193 (2023).

[3] Shi, Tao, and Shao-Lun Huang. "MultiEMO: An attention-based correlation-aware multimodal fusion framework for emotion recognition in conversations." Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2023.

### Questions
# 1. Runtime evaluation at scale

Figure 1 only reports runtime on MNIST. Can you provide runtime measurements under large-scale settings (e.g., ImageNet-1K, multimodal fusion training) to verify whether UniFast-HGR yields efficiency benefits when encoder forward cost dominates?


# 2. Generalizability beyond remote sensing

Do the authors plan to evaluate UniFast-HGR on widely-adopted multimodal benchmarks (e.g., COCO Captions, LAION-400M) to support the claim of general multimodal applicability?


# 3. Reproducibility of Tables 3, 4, 12

Could you provide full training details (batch size, optimizer, epochs) and explain why the reported baselines (e.g., DINOv2 linear evaluation, MultiEMO Weighted-F1) differ significantly from known results?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes UniFast-HGR, a scalable surrogate for HGR maximal correlation that replaces covariance with cosine similarity under unit-variance constraints and removes diagonal terms to avoid self-correlation bias. An OptFast variant further improves efficiency. The method is simple to integrate and the experiments are extensive, indicating consistent accuracy gains with markedly lower cost.

### Strengths
1. Experiments are extensive and the gains are clear, with consistent improvements and strong speedups on diverse tasks.
2. The objective is broadly applicable and can be readily integrated into a wide range of encoders and training pipelines.

### Weaknesses
1. Notation and assumptions contain inconsistencies or under-specified definitions.
2. Lacking details of the experimental settings, such as key hyperparameters, the runtime environment and so on.

### Questions
1. Will you release the source code to facilitate exact reproduction?

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
2

### Summary
- This paper proposes UniFast-HGR, an efficient and scalable framework for estimating the Hirschfeld–Gebelein–Rényi (HGR) maximal correlation in multimodal learning.

Traditional HGR and Soft-HGR methods suffer from high computational complexity (O(K^3)), instability in covariance matrix inversion, and limited scalability to deep architectures.

UniFast-HGR introduces three key innovations:

1.Replacing covariance with cosine similarity — avoiding matrix inversion and reducing complexity to O(m²K).

2.Removing the diagonal elements of the correlation matrix — mitigating self-correlation bias.

3.Applying ℓ₂ normalization as a variance constraint — improving numerical stability and boundedness.

Additionally, an accelerated variant OptFast-HGR is introduced, further simplifying normalization steps to achieve dot-product–level efficiency with minimal accuracy loss.

Extensive experiments are conducted across multiple domains—image classification, remote sensing segmentation, multimodal emotion recognition, and large-scale multimodal retrieval (ImageNet, COCO, InternVid)—demonstrating consistent improvements over baselines such as CCA, Soft-HGR, CKA, and dCor.

### Strengths
- High originality: The substitution of covariance with cosine similarity and the removal of diagonal elements represent a mathematically elegant and computationally efficient rethinking of the HGR formulation.
- Significant computational gains: Complexity is reduced from O(K³) to O(m²K), enabling application in large-scale deep networks.
- Theoretical grounding and stability: Variance constraints via ℓ₂ normalization stabilize training and maintain bounded correlation scores.
- Extensive empirical evaluation: Results span small- to large-scale benchmarks and demonstrate consistent superiority over strong baselines.

### Weaknesses
- Insufficient ablation analysis:The paper introduces three improvements (cosine substitution, diagonal removal, ℓ₂ normalization), but their individual contributions are not clearly isolated through ablation studies.
- OptFast bias not fully analyzed:The paper notes a “slight bias” but does not quantify it or provide formal bounds or convergence guarantees.
- Limited discussion of nonlinear architectures:Experiments mainly use CNNs and ViTs; there is no analysis on transformer-based multimodal encoders or generative models.
- Writing and clarity:Sections 2.2–2.3 are notation-heavy, and some derivations could be complemented with intuitive figures or algorithmic flowcharts for readability.

### Questions
- On diagonal removal:Can the authors provide a formal justification or spectral/information-theoretic rationale for excluding the diagonal? How does this behave under non-unit variance or noisy feature conditions?
- On scalability and modality extension:How does UniFast-HGR scale with more than two modalities (e.g., 3+)? Does computational complexity grow linearly or quadratically with modality count?
- On OptFast bias:Is there an analytic upper bound on the estimation bias introduced by OptFast-HGR? Could adaptive normalization or bias correction mitigate this?

### Soundness
2

### Presentation
2

### Contribution
3
