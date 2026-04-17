# Hystar: Hypernetwork-driven Style-adaptive Retrieval via Dynamic SVD Modulation

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Query-based image retrieval (QBIR) requires retrieving relevant images given diverse and often stylistically heterogeneous queries, such as sketches, artworks, or low-resolution previews. While large-scale vision--language representation models (VLRMs) like CLIP offer strong zero-shot retrieval performance, they struggle with distribution shifts caused by unseen query styles. In this paper, we propose the Hypernetwork-driven Style-adaptive Retrieval (Hystar), a lightweight framework that dynamically adapts model weights to each query’s style. Hystar employs a hypernetwork to generate singular-value perturbations ($\Delta S$) for attention layers, enabling flexible per-input adaptation, while static singular-value offsets on MLP layers ensure cross-style stability. To better handle semantic confusions across styles, we design StyleNCE as part of Hystar, an optimal-transport-weighted contrastive loss that emphasizes hard cross-style negatives. Extensive experiments on multi-style retrieval and cross-style classification benchmarks demonstrate that Hystar consistently outperforms strong baselines, achieving state-of-the-art performance while being parameter-efficient and stable across styles.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces Hystar, a lightweight framework for Query-based Image Retrieval (QBIR) that addresses the challenge of handling stylistically diverse queries (e.g., sketches, artworks) which cause distribution shifts and performance drops in standard Vision-Language Models (VLMs) like CLIP. Extensive experiments show that Hystar achieves state-of-the-art performance on multi-style retrieval and cross-style classification tasks, offering superior parameter efficiency and stable performance across various query styles.

### Strengths
1. Well-Motivated Problem and Thorough Literature Review: The paper adeptly identifies and tackles the critical, yet underexplored, challenge of style-diversified retrieval in QBIR. The authors provide a comprehensive overview of existing paradigms, such as LoRA-based PEFT methods and those relying on style-cluster priors, and convincingly argue their limitations in generalizing to unseen query styles. This strong motivation establishes a clear and valuable niche for their work.

 2. Innovative and Principled Methodology: The core technical contribution is both novel and well-designed. The use of a hypernetwork to dynamically generate LoRA matrices is a clever adaptation. More importantly, the decision to restrict the hypernetwork's output to singular-value perturbations (∆S) is a key insight. This approach not only enhances training stability but also explicitly promotes generalization across diverse and unseen styles by focusing adaptation on a compact, semantically meaningful parameter space.

 3. Compelling and Multi-faceted Empirical Validation: The authors provide rigorous experimental evidence to support their claims. Through comprehensive comparisons against strong baselines (e.g., FreestyleRet, VPT), they convincingly demonstrate the superiority of their hypernetwork-driven paradigm. Furthermore, the inclusion of t-SNE visualizations offers an intuitive and powerful qualitative analysis, effectively illustrating how Hystar learns more discriminative and style-invariant feature representations.

### Weaknesses
1. Terminological Imprecision: The term "Vision-Language Models (VLMs)" is used to describe CLIP. However, in contemporary literature, "VLM" often refers more specifically to models that include a text decoder for generative tasks (e.g., GPT-4V, LLaVA). CLIP, being a dual-encoder model designed primarily for contrastive learning and representation embedding, is more accurately categorized as a vision-language representation model. Adopting this more precise terminology would enhance conceptual clarity.

 2. Insufficient Justification for Architectural Choices: The rationale behind using a hypernetwork to generate dynamic components for attention layers while using static offsets for MLP layers, as a specific form of decoupling, requires deeper analysis. The paper would benefit from a more thorough discussion comparing this design choice against other potential decoupling strategies. A theoretical or empirical justification for why this particular split (dynamic attention vs. static MLP) is optimal for the task of style adaptation would significantly strengthen the methodological argument.

 3. Limited Scope of Ablation Studies: The ablation studies, while validating the proposed components, could be more comprehensive by including a wider range of Parameter-Efficient Fine-Tuning (PEFT) baselines. For instance, comparing against other adaptive tuning methods like (IA)³ or Adapter-based approaches would provide a clearer picture of Hystar's performance gains relative to the broader landscape of efficient adaptation techniques, not just the ones it directly outperforms.

### Questions
please see above

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses the style distribution shift problem in query-based image retrieval (QBIR) (such as queries with styles different from those in the training data, like sketches and artworks), and proposes the Hystar framework. The Hystar framework includes two main components: 1. Hypernetwork-driven Dynamic SVD Modulation: A hypernetwork generates singular value perturbations (ΔS) in the attention layer based on the query style, achieving dynamic adaptation for each input. Simultaneously, static singular value offsets are used in the MLP layer to ensure cross-style stability.
2. StyleNCE Loss Function: A contrastive loss weighted by optimal transport theory highlights indistinguishable cross-style negative samples, improving the model's robustness to style differences.

### Strengths
1. The motivation of this work is good, which clearly points out the performance degradation problem of VLM under style distribution shift and emphasizes the static limitations of existing PEFT methods (such as LoRA and VPT).
2. The experiment was comprehensive and convincing.
3. The proposed method is efficient in parameters: Only a few parameters need to be fine-tuned.

### Weaknesses
1. This work emphasizes parameter efficiency, but lacks a quantitative analysis of the forward computational overhead of the hypernetwork (relative to static PEFT) (e.g., the percentage increase in inference time). This is important for evaluating its feasibility in real-time retrieval scenarios.
2. Appendix A provides an intuitive theoretical explanation of SVD modulation (based on spectral norm constraints). However, further exploration is possible, for example, analyzing from the perspectives of generalization theory or manifold learning why SVD modulation is particularly effective for style adaptation.
3. This work mainly focus on retrieval and zero-shot classification. Further validation of Hystar's generalization ability on other style-sensitive tasks can be achieved, such as stylized image generation guidance and cross-style visual question answering.

### Questions
1. This paper chose DINOv2 as the style feature extractor. Were other features considered (e.g., features from dedicated style analysis networks)? Do different feature choices significantly impact performance?
2. How does Hystar handle styles that are completely new, extremely abstract, or bizarre in the training data? How does the model respond when the same query image contains a mixture of multiple styles?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces HySTAR, a novel framework for query-based image retrieval (QBIR) designed to handle significant variations in query style (e.g., sketches, artwork, low-resolution images). The core problem it addresses is the poor performance of large vision-language models (VLMs) like CLIP when faced with stylistic distribution shifts.

The paper's two main contributions are:
- Hypernetwork-driven Dynamic PEFT: A parameter-efficient fine-tuning (PEFT) method where a small hypernetwork dynamically generates input-specific modulations. These modulations are applied to the singular values (SVD) of the attention layers in a frozen VLM, conditioned on a style feature vector.
- StyleNCE Loss: A contrastive loss function that uses optimal transport (OT) to re-weight hard negative samples, specifically designed to improve semantic alignment across different styles- first to do so for Image Retrieval. 

The authors claim that this combination of dynamic, style-adaptive modulation and a style-focused loss function achieves state-of-the-art results on the DSR and DomainNet benchmarks.

### Strengths
- Relevant Problem: The paper tackles a well-known and practical limitation of large pre-trained models: their lack of robustness to domain and style shifts. The application to cross-style image retrieval is a challenging and valuable research area.
 - Novel Methodology: The primary technical idea—using a hypernetwork to predict dynamic PEFT parameters based on the input's style—is a creative and novel contribution. Moving from static PEFT (like LoRA or VPT) to an adaptive, input-conditioned PEFT is an interesting research direction. The specific choice to modulate singular values (SVD) is also an unconventional and intriguing approach.
 - Well-Motivated Loss Function: The proposed StyleNCE loss is a strong contribution. The intuition that cross-style retrieval creates many "hard negatives" (e.g., a sketch of a cat vs. a photo of a tiger) is sharp. Using optimal transport to systematically identify and up-weight these hard samples is a well-founded and logical approach to improving fine-grained, cross-style discriminative power.

### Weaknesses
Despite the novel ideas, the paper suffers from several major weaknesses, primarily in the experimental design and the clarity of the method's description.
- Unfair Experimental Comparison (Major Flaw): The central claim of HySTAR's superiority is built on a confounded experiment. As shown in Figure 1, HySTAR's hypernetwork is conditioned on features from DINOv2, a powerful, separate vision model. The baseline methods (CLIP, BLIP, LoRA, VPT, FreestyleRet) do not have access to these supplementary features. The performance gains attributed to HySTAR's architecture (dynamic SVD modulation) may, in large part, be coming from the simple fact that it is a multi-model system using rich DINOv2 features that the baselines lack. This is a critical confounding variable that makes the primary results in Tables 1, 2, and 3 uninterpretable.
- Missing Architectural and Implementation Details: The paper is missing key details required for understanding and reproducing the method.
  + Hypernetwork Architecture: The hypernetwork is defined in Eq. 2 as a simple 2-layer MLP, but its dimensions (input, hidden, and output) are never specified.
  + Style Feature Extraction: It is unclear how the style feature $z$ is extracted from DINOv2. Is it the [CLS] token of the input image? An average pool of patch tokens? This is a crucial, undefined component of the model.
- Insufficient Ablation Studies: The ablation study in Table 4, while showing the benefit of the "Hyper" and "Static" modules, is insufficient. There are no ablations on the hypernetwork design itself (e.g., depth, width) to justify the chosen (and unstated) architecture.
- Limited Theoretical Justification for SVD: The paper's justification for using SVD modulation is thin. Appendix A argues that it provides stability by bounding the spectral norm of the weight update. However, it provides no theoretical or empirical intuition as to why modulating singular values is semantically the correct approach for style adaptation. Why is this superior to other low-rank updates like LoRA ($W_0 + BA$), especially when LoRA is more computationally straightforward (it doesn't require SVD on-the-fly)? The connection between "visual style" and "singular values of attention weights" is never established.
- Errors and Poor Presentation in Results:
   + Error in Table 2: As noted by the user, the paper incorrectly highlights its own score as "best" in Table 2. For the "Infograph" style (last column), the Top 1 accuracy for CLIP is 60.3%, which is higher than HySTAR's 59.3%. Yet, HySTAR’s score is bolded.
    + Weak Visualization (Figure 7): The figures are extremely hard to read. It is also one of the most prominent results showcasing the comparison between baseline method and your method for retrieval across various styles. Also, it is not clear what is the baselines comparison? Is it zero shot CLIP? Or FreestyleRet? FreestyleRet has better retrieval results if so, than what is demonstrated in the comparison.

### Questions
- Weakness 1): Can you please address the experimental confound of using DINOv2? To fairly evaluate HySTAR as a PEFT method, could you provide an ablation comparing the baselines (VPT, LoRA) to a version of HySTAR that does not use DINOv2, and instead derives its style vector $z$ from the VLM backbone itself (e.g., from CLIP)?
- Re: (Weakness 2): What are the precise architectural details of the hypernetwork (input, hidden, and output dimensions)? And how exactly is the style feature $z$ extracted from the DINOv2 model?
- Re: (Weakness 3): Beyond stability, what is the theoretical or empirical justification for SVD modulation being a superior method for style adaptation compared to other low-rank updates like LoRA?
- Re: (Weakness 4 / Figure 7): Please clarify what the baselines are and make the figures readable to verify the claims of superior consistency.

### Soundness
2

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
4

### Summary
This paper presents Hystar, a dynamic multi-style retrieval framework designed to address challenges in Query-based Image Retrieval (QBIR) under significant style variations. Hystar achieves a balance between adaptability and stability in Parameter-Efficient Fine-Tuning (PEFT) by combining two key components: (1) hypernetwork-driven dynamic modulation and (2) static singular-value calibration. To tackle the limitations of standard contrastive losses that treat all negatives equally, the paper introduces the OT-weighted StyleNCE loss. This loss leverages Optimal Transport (OT) theory (via Sinkhorn iterations) to reweight negatives by difficulty: it amplifies the contribution of hard negatives while maintaining balanced batch-wise contributions, enabling effective capture of cross-style discrepancies. Extensive experiments on datasets including DSR and DomainNet demonstrate that Hystar consistently outperforms strong baselines.

### Strengths
* The use of singular-value modulation (SVD-based) is conceptually elegant and ensures stable, low-rank updates.
* The StyleNCE loss introduces optimal transport weighting to emphasize hard cross-style negatives. It meaningfully improves robustness under distribution shifts and has solid theoretical motivation.

### Weaknesses
Many of the key techniques presented in this paper build upon prior research. The overarching design strategy of freezing most parameters while learning only small-scale incremental terms follows the mainstream PEFT framework, whose foundational works include [1] and [2]. Similarly, the OT-weighted StyleNCE loss is grounded in Optimal Transport (OT) theory and Sinkhorn iterations, both of which are well-established mathematical and algorithmic tools as detailed in [3].

[1] Lingam, Vijay Chandra, et al. "Svft: Parameter-efficient fine-tuning with singular vectors." Advances in Neural Information Processing Systems 37 (2024): 41425-41446.

[2] Wang, Zhiwu, et al. "Singular Value Fine-tuning for Few-Shot Class-Incremental Learning." arXiv preprint arXiv:2503.10214 (2025).

[3] Jiang, Ruijie, Prakash Ishwar, and Shuchin Aeron. "Hard negative sampling via regularized optimal transport for contrastive representation learning." 2023 International Joint Conference on Neural Networks (IJCNN). IEEE, 2023.

### Questions
The proposed method does not appear to be limited to the Style-adaptive Retrieval task and could potentially be applied to similar tasks, such as image or video retrieval. Given the limited number of existing methods for direct comparison in the Style-adaptive Retrieval task, has the author considered evaluating the proposed method on these alternative tasks?

### Soundness
2

### Presentation
3

### Contribution
2
