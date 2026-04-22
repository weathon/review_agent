# LAGUNA: LAnguage Guided UNsupervised Adaptation with structured spaces

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
Unsupervised domain adaptation remains a critical challenge in enabling knowledge transfer of models across domains. Existing methods struggle to balance the need for domain-invariant representations with preserving domain-specific features. This is often caused by alignment approaches that impose the projection of different domain samples close to each other in latent spaces, despite drastic differences. We introduce LAnguage GUided domaiN Adaptation with structured spaces, a novel approach that shifts the focus from aligning representations in absolute coordinates to aligning the relative positioning of equivalent concepts in latent spaces. LAGUNA defines a domain-agnostic structure on the geometric relationships between class labels in the language space and guides the adaptation, ensuring that the organization of samples in visual space reflects reference inter-class relationships while preserving domain-specific characteristics. 
Remarkably, LAGUNA surpasses previous work in 18 different adaptation scenarios across four diverse image and video datasets with average accuracy improvements of up to +3.32% on DomainNet, +5.75% on GeoPlaces, +4.77% on GeoImnet, and +1.94% mean class accuracy improvement on EgoExo4D. Finally, we also show empirically that LAGUNA competes with MLLMs 100x its size in complex adaptation scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LAGUNA (LAnguage-GUided domAiN Alignment), a framework for cross-domain visual recognition that aligns representations through language-defined semantic geometry. It first constructs a fixed language anchor space from class names, then trains a language supervisor to produce pseudo-labels and maintain this structure. In the final stage, domain-specific visual anchors and a cross-domain attention mechanism align visual features with the language geometry, while a volume regularization term preserves anchor diversity.

### Strengths
Using language both as a semantic reference space and as a source of pseudo-supervision offers an elegant way to bridge domains and modalities in the absence of target labels.

### Weaknesses
1. While the paper avoids reliance on large-scale vision-language models at inference, it does not adequately compare against methods that do use such models (e.g., CLIP, ALIGN, or recent prompt-tuning approaches in zero-shot or few-shot settings).
2. The three-stage pipeline introduces substantial training overhead. While each component is well motivated, the cumulative complexity may hinder adoption, particularly in settings with limited computational resources or high domain variability.

### Questions
1. The method relies on relative encodings (cosine similarities to semantic anchors) instead of directly aligning embeddings. Why is this preferable in practice? Did the authors compare it to direct matching of latent vectors (e.g., using contrastive or prototypical alignment)?
2. The class-name-based anchors A are fixed and assumed to encode reliable semantic geometry. Given that sentence embeddings may not reflect task-specific distinctions (e.g., action granularity), how sensitive is the framework to inaccuracies in these language embeddings?
3. The use of log-determinants of Gram matrices is elegant but potentially unstable or sensitive in high-dimensional spaces. Are there numerical issues during training? How does performance vary with or without this regularization term? Additionally,  can the authors provide intuition or theoretical reasoning for why this specific form of volume preservation is optimal over, say, minimizing pairwise collapse or encouraging orthogonality?
4. The method relies on aligning relative encodings rather than absolute feature vectors. Is there a formal justification or theoretical advantage to this approach in terms of generalization, invariance, or robustness across domains?
5. Given that multiple objectives (structure-preserving loss, cross-entropy, volume regularization) are optimized jointly across language and visual modalities, how are gradient conflicts handled? Was gradient interference an issue during training?

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
4

### Summary
This paper proposes a language-guided approach to address the unsupervised domain adaptation (UDA) problem by aligning source and target domains in relative spaces. It first highlights the advantages of relative representations over absolute alignment. Then, it introduces a method that aligns source and target feature spaces with a language-based reference structure to achieve domain-invariant yet discriminative representations. Experimental results across 4 benchmarks demonstrate the effectiveness of the proposed method.

### Strengths
1. The investigation comparing absolute and relative alignment (fig. 1) is interesting. It shows that domain-specific characteristics such as color or texture, though differing across domains, can still be useful for classification within each domain.

2. The idea of using language to construct a reference structure is interesting as language provides a semantic space agnostic to visual variations.

3. The paper is clearly written and easy to follow.

### Weaknesses
1. While the paper argues that it is important to preserve domain-specific representations in the aligned space and provides a two-point example (lines 52–70), I am still concerned that such domain-specific representations inherently reflect domain shift and should ideally be minimized. Could you do some experiments to justify and empirically support your motivation.

2. In fig. 2, it would be helpful to indicate which components are learnable using visual icons, rather than only marking the frozen parts. Additionally, in Stage 3, $\mathbf{r}_i^{y^s}$ appears to be mistakenly written as $\mathbf{r}_i^{y^t}$. Moreover, this figure (on page 3) can only be fully understood when read together with the corresponding explanation on page 4,5, since readers may not immediately know what $\mathbf{r}_i^{g^t}$, $\mathbf{r}_i^{g^s}$, $\mathbf{r}_i^{y^s}$ and $\mathbf{r}_i^{z^t}$represent.

3. The paper does not clearly describe how the domain-specific anchors $A_t$ and $A_s$ are initialized. Could you please explicitly state their initialization strategy and further clarify why these anchors need to be learnable rather than fixed.

4. Since the proposed method involves caption generation for both source and target domains, it may introduce additional computational overhead compared to traditional UDA approaches. Could you provide a time or efficiency analysis (e.g., training and inference time, caption generation cost) to demonstrate the practicality of your method relative to existing UDA baselines.

5. Regarding the benchmark evaluation, the paper does not include comparisons with more recent methods published this year, even though it is already October. For example, I found a recent work [1] that also uses language guidance to address the UDA problem. It would strengthen the evaluation if the authors could include this method in their benchmark for a fairer and more up-to-date comparison.

6. In line 390, there is a typo: the improvement value should be 0.78 instead of 0.87.

[1] Litrico, Mattia, et al. "TRUST: Leveraging Text Robustness for Unsupervised Domain Adaptation." arXiv preprint arXiv:2508.06452 (2025).

### Questions
Please see the above weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents LAGUNA (LAnguage Guided UNsupervised Adaptation), a new unsupervised domain adaptation framework that replaces traditional absolute feature alignment with concept-level relational alignment. Instead of forcing cross-domain samples to collapse into a shared latent space — which may destroy domain-specific structure — LAGUNA uses language to define a domain-agnostic semantic geometry over class labels, and then constrains visual features so that their inter-class relationships respect this reference geometry. By aligning the relative positioning of equivalent concepts rather than raw coordinates, LAGUNA preserves domain-specific characteristics while enforcing consistent semantics across domains, showing that semantic alignment can be achieved without projecting all domains into a single shared space.

### Strengths
This paper presents LAGUNA (LAnguage Guided UNsupervised Adaptation), a new unsupervised domain adaptation framework that replaces traditional absolute feature alignment with concept-level relational alignment. Instead of forcing cross-domain samples to collapse into a shared latent space — which may destroy domain-specific structure — LAGUNA uses language to define a domain-agnostic semantic geometry over class labels, and then constrains visual features so that their inter-class relationships respect this reference geometry. By aligning the relative positioning of equivalent concepts rather than raw coordinates, LAGUNA preserves domain-specific characteristics while enforcing consistent semantics across domains, showing that semantic alignment can be achieved without projecting all domains into a single shared space. 


Through its architectural design and corresponding experiments, this paper demonstrates that semantic alignment can be achieved without enforcing representations from different domains to reside in a shared space, thereby allowing each domain to retain its unique characteristics while maintaining consistency with a common structural framework.

### Weaknesses
1. It is hard to say that this paper is a piece of work of UDA due to the usage of captions. Moreover, for low-quality captions, the ablation study in this paper only examines the effect of using a weaker caption generation model (i.e., replacing BLIP-2 with BLIP-1). However, it does not evaluate the robustness of the proposed method under more challenging cases, such as when captions omit the target category or include incorrect class information.

2. The key point of this paper is leveraging the conception space to bridge the source and target domains. Namely, the conception space is treated as a proxy space of the latent domain-invariant space. It is not a new idea. The differences from prior work are not clearly stated.  

3. For Eq. (8)–(9), the paper lacks theoretical justification or empirical evidence explaining the rationale behind grounding gti and gsi to the structure of source anchors As instead of At. The underlying principle and the performance advantage of this design choice should be further clarified and supported.

4. In Section “3 METHOD,” the notation for the language-based, domain-agnostic reference structure A is inconsistent with the symbol *A used in Fig. 2, which may cause confusion in understanding the formulation and should be unified for clarity.

5. Given the stated goal “to ensure that source and target visual representations are grounded in the structure imposed by A, while still being free to capture domain-specific nuances,” it would be valuable to include experiments that explicitly analyze and visualize how As and At reflect this balance between adherence to the shared reference structure A and preservation of domain-specific characteristics.

### Questions
Refer to the weaknesses above.

### Soundness
3

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
The paper tackles the challenge of unsupervised domain adaptation (UDA), aiming to improve model generalization across domains without labeled target data. Existing alignment methods often overemphasize domain invariance, leading to the loss of domain-specific nuances.

To address this, the authors propose LAGUNA (LAnguage Guided UNsupervised Adaptation) — a novel framework that leverages the geometric relationships between class labels in language space as structural guidance for adaptation. Instead of forcing latent feature alignment in absolute space, LAGUNA focuses on preserving relative inter-class relationships while maintaining domain-specific features.

Extensive experiments across four diverse datasets (DomainNet, GeoPlaces, GeoImNet, EgoExo4D) demonstrate consistent and notable performance gains, with accuracy improvements of up to +5.75% compared to prior methods.

### Strengths
1. The paper addresses an important problem in unsupervised domain adaptation.

2. The idea of leveraging language-model–based structure is novel and well-motivated.

3. The experiments are comprehensive, covering multiple datasets and baselines.

4. The paper is clearly written and easy to follow.

### Weaknesses
1. The ablation study does not sufficiently demonstrate the contribution of each component. It would be helpful to see how the model performs when the vision–language model (VLM) is fine-tuned on the target domain (both images and captions). Currently, the authors only compare their method with several off-the-shelf VLMs without any fine-tuning, which limits understanding of how much improvement comes from the proposed adaptation versus general fine-tuning effects.

2. The choice of embedding models is not clearly justified. The paper uses a Sentence-Transformer for label embeddings and BERT for caption embeddings, but it is unclear why different pre-trained models were chosen instead of using a unified one. An ablation study comparing different embedding backbones (e.g., alternative Sentence-Transformers, BERT variants, or even word2vec) would strengthen the rationale for this design choice.

### Questions
1. Clarification on pseudo-label stabilization: It is unclear how the pseudo-labels are updated and stabilized during training. Are they dynamically refined across iterations, or fixed after initial assignment? Since no ground-truth labels are available, how do the authors ensure that the optimization process converges to a meaningful or near-optimal solution rather than reinforcing noisy labels?

2. Rationale for the regularization term (λ₃): The paper sets λ₃ to 0.001, which appears relatively small compared to the other loss weights, suggesting this term may have limited influence. It would be helpful to report model performance with λ₃ = 0.1 or λ₃ = 0 to evaluate the sensitivity of the method to this parameter and clarify the necessity of the corresponding regularization term.

### Soundness
3

### Presentation
3

### Contribution
3
