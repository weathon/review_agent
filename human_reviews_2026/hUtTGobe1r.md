# Decoupling Primitive with Experts: Dynamic Feature Alignment for Compositional Zero-Shot Learning

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Compositional Zero-Shot Learning (CZSL) investigates compositional generalization capacity to recognize unknown state-object pairs based on learned primitive concepts. Existing CZSL methods typically derive primitives features through a simple composition-prototype mapping, which is suboptimal for a set of individuals that can be divided into distinct semantic subsets. 
Moreover, the one-to-all cross-modal primitives matching neglects compositional divergence within identical states or objects, limiting fine-grained image-composition alignment. In this study, we propose EVA, a Mixture-of-Experts Framework for Semantic Variant Alignment. Specifically, we introduce domain-expert adaption, leveraging multiple experts to achieve token-aware learning and model high-quality primitive representations. To enable accurate compositional generalization, we further present semantic variant alignment to select semantically relevant representation for image-primitives matching. 
Our method significantly outperforms other state-of-the-art CZSL methods on three popular benchmarks in both closed- and open-world settings, demonstrating the efficacy of the proposed insight.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the Compositional Zero-Shot Learning (CZSL) problem, which aims to recognize unseen combinations of state and object primitives.
Existing methods usually assume a single static prototype for each primitive (e.g., old, young), ignoring the semantic polysemy that arises across different contexts (e.g., old man vs. old book). This leads to semantic entanglement and limited generalization.
To address this, the authors propose EVA (Expert-based Variant Alignment), a Mixture-of-Experts framework that explicitly models semantic variability at the primitive level.
It consists of two main components:
(1) Domain-Expert Adaption, which inserts lightweight LoRA-based MoE adapters into a frozen CLIP backbone to dynamically route tokens to specialized experts, enabling context-aware primitive representations; and
(2) Semantic Variant Alignment, which selects semantically relevant feature variants from both text and image perspectives to achieve fine-grained cross-modal matching.
Experiments on MIT-States, UT-Zappos, and C-GQA datasets demonstrate that EVA consistently outperforms state-of-the-art methods in both closed- and open-world settings, achieving up to +3.5% (closed-world) and +2.2% (open-world) AUC improvements while remaining efficient and interpretable.

### Strengths
1. Clear motivation addressing primitive polysemy in CZSL.
2. Strong empirical results across three benchmarks.
3. Lightweight and interpretable design built on frozen CLIP.

### Weaknesses
1. The paper lacks a deeper analysis of why MoE improves compositional classification — explanations remain intuitive without quantitative evidence of expert complementarity or routing diversity.
2. The observed gain may stem from implicit feature reparameterization rather than genuine semantic disentanglement.
3. No study of routing stability or expert utilization balance; potential risk of expert collapse.

### Questions
1.	Are the experts activated evenly during training, or does routing collapse occur?
2.	Can the authors provide quantitative or visual evidence showing that different experts learn complementary semantics?
3.	Would EVA still work without explicit state/object labels in open-vocabulary or unsupervised settings?
4.	How consistent are the improvements across different vision–language backbones (e.g., BLIP, SigLIP)?
5.	Does the MoE module truly enhance semantic reasoning, or mainly increase representational capacity?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the problem of primitive polysemy in Compositional Zero-Shot Learning (CZSL) and proposes an expert-based dynamic feature decoupling framework called EVA (Expert-based Variant Alignment). The authors point out that primitives can have different meanings under different semantic contexts (e.g., “old” in “old man” vs. “old building”), which are difficult to model using a single embedding. As a result, traditional attribute–object composition approaches cannot fully capture semantic variations.

EVA mainly consists of two core modules:

1.	Domain Expert Adaptation: Inserts a Mixture-of-Experts (MoE) module into the CLIP encoder. Through a token-level routing mechanism, it selects the appropriate expert to model semantic variations across contexts.

2.	Semantic Variant Alignment: Designs a cross-modal dynamic alignment strategy that combines global and local feature matching to achieve fine-grained consistency between visual and textual semantic spaces.

Experiments on standard CZSL datasets — MIT-States, UT-Zappos, and C-GQA — show that EVA outperforms several recent methods (e.g., GIPCOL, Troika, IVLP) under both closed-set and open-set settings. The authors claim EVA achieves more stable semantic generalization and better interpretability.

### Strengths
1.	The paper proposes a logically coherent and complete MoE + alignment framework, effectively integrating expert modeling with semantic disentanglement.

2.	Experiments cover multiple mainstream CZSL datasets, and the reported performance exceeds several existing methods on certain metrics, showing credible results.

### Weaknesses
1. The work is essentially an architectural extension of existing CLIP-based compositional alignment methods.
2. The core mechanisms (expert routing and alignment strategy) lack quantitative validation.
3. The paper does not analyze computational cost, parameter scale, or inference latency, making it difficult to assess practicality.
4. Some figures and text repeat information, reducing clarity.
5. Differences between variants are small, making it unclear how much each module contributes.

### Questions
1. Does the expert module experience imbalance during training? Have you introduced a load balancing loss?

2. What is the performance drop if the alignment module is removed or only the MoE structure is retained?

3. Does the EVA framework rely on CLIP’s pretrained semantic structure? Would it still work with non-CLIP backbones?

4. Can you provide more interpretability results (e.g., clustering distributions of experts across semantic attributes) to demonstrate the effectiveness of semantic decoupling?

### Soundness
2

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
5

### Summary
This paper addresses Compositional Zero-Shot Learning (CZSL) focusing on the challenge that primitive features (states, objects) vary semantically across contexts. The authors propose EVA (Expert-based Variant Alignment), a MOE framework that learns context-aware primitive representations and performs fine-grained cross-modal alignment with carefully designed traning objective. EVA introduces two key components:(1) Domain-Expert Adaption, which employs MoE adapters in image and text encoders to capture diverse semantic facets of primitives through dynamic token routing; and(2) Semantic Variant Alignment (SVA), which aligns fine-grained visual and textual variants via both text-to-image and image-to-text matching. Comprehensive quantitative and qualitative experiments on the three most commonly used CZSL datasets  demonstrate that EVA achieves superior performance  in both closed- and open-world settings. The analyses further show that different experts specialize in distinct semantic aspects, validating the effectiveness of the proposed design.

### Strengths
1. The paper introduces an MoE-based framework to address the semantically heterogeneous of primitives in CZSL. By dynamically routing tokens to domain-specific experts, the proposed approach effectively captures context-dependent primitive semantics.The proposed model achieves SOTA performance in both closed-world and open-world settings without any suffix modules.


2. The proposed SVA module overcomes the limitation of previous all-to-one alignment  by introducing a global-to-local cross-modal alignment from both image and text perspectives. This design enables more accurate and context-sensitive alignment of visual and textual primitives, leading to better compositional discrimination. 

3. The paper is well-structured and easy to follow, with thorough quantitative and qualitative evaluations.

### Weaknesses
1.  Figure 2(c) does not clearly illustrate how the SVA module works, and Figure 2(a) lacks explicit description for the text-to-image alignment pathway $\mathcal{L}_s^v ,\mathcal{L}_o^v$.
2. The paper does not include an ablation study on the number of experts.
3. The paper lacks deeper analysis of expert specialization, such as overall statistical distributions or further exploration of text experts.

### Questions
1. Figure 7 shows that the token load across text experts is highly unbalanced. Do these experts correspond to particular semantic groups or patterns? Could you provide further analysis, and would introducing a load-balancing loss help?

2. In the SVA module, the image-to-text alignment require computing affinities of experts. How is this objective applied or adapted to the baselines reported in Table 2?

3. Table 8 reports efficiency analysis on UT-Zappos, but this dataset has relatively few state and object categories. Since the efficiency is sensitive to the number of primitives, could you report or discuss results on other datasets and under open-world settings?

4. Could you provide a more detailed analysis of the vision experts? For example, showing how the same expert allocates patches across different images?

5. It would be valuable to include a more detailed investigation of text experts. For instance, analyzing the impact of using separate experts for states, objects, and compositions.

6. Which dataset was used for the qualitative analyses in Figures 7–8?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a mixture-of-experts framework for semantic variant alignment (EVA) to solve compositional zero-shot learning. The proposed method introduces two main modules: domain-expert adaptation and semantic variant alignment. The extensive experiments on three benchmarks demonstrate significant improvements over previous SOTA.

### Strengths
1) This paper introduces MoE paradigm to model the heterogeneous nature of primitive concepts.
2) The experimental results are promising.

### Weaknesses
1) The novelty is incremental. Specifically, many CZSL methods enable the primitive embeddings to dynamically adapt to diverse semantic variants by fine-grained learning[3], clustering-based prototypes[1], distribution learning[2] et al. The introduction of MoE to facilitate the adaptation of features is sound but this is merely a simple application without promoting the development of CZSL.
2) Some parts of the paper are difficult to understand. Terms like "domain-expert adaption," “in-domain knowledge”, "variant-based method" are used with little prior explanation. 
2) The inclusion of a balancing coefficient $\alpha$ is also empirically motivated rather than theoretically justified, and it is unclear how sensitive results are to this choice.
3) In lines 258-269, the feature variants V is not described clearly, and the selection of state/object image features from feature variants lacks rigorous mathematical support or analysis. In addition, A_S \in R^{N_e+1}, why i\in [0,N_e] in equation (11)? 
4) In lines 267-268, is {v_i}_{i=0}^{N_e} feature variants or semantic variants?
5) In lines 262-263, why is A_v computed based on V and image features f_c, while A_s and A_o are computed based on V and text features t_s/t_o?
[1] LEARNING CLUSTERING-BASED PROTOTYPES FOR COMPOSITIONAL ZERO-SHOT LEARNING
[2] Prompting Language-Informed Distribution for Compositional Zero-Shot Learning
[3] Leveraging sub-class discimination for compositional zero-shot learning

### Questions
Addressing weaknesses and improving writing quality.

### Soundness
3

### Presentation
3

### Contribution
2
