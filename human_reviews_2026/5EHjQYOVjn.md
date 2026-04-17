# Align Your Query: Representation Alignment for Multimodality Medical Object Detection

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Medical object detection suffers when a single detector is trained on mixed medical modalities (e.g., CXR, CT, MRI) due to heterogeneous statistics and disjoint representation spaces. To address this challenge, we turn to representation alignment, an approach that has proven effective for bringing features from different sources into a shared space. Specifically, we target the representations of DETR-style object queries and propose a simple, detector-agnostic framework to align them with modality context. First, we define modality tokens: compact, text-derived embeddings encoding imaging modality that are lightweight and require no extra annotations. We integrate the modality tokens into the detection process via Multimodality Context Attention (MoCA), mixing object-query representations via self-attention to propagate modality context within the query set. This preserves DETR-style architectures and adds negligible latency while injecting modality cues into object queries. We further introduce QueryREPA, a short pretraining stage that aligns query representations to their modality tokens using a task-specific contrastive objective with modality-balanced batches. Together, MoCA and QueryREPA produce modality-aware, class-faithful queries that transfer effectively to downstream training. Across diverse modalities trained altogether, the proposed approach consistently improves AP with minimal overhead and no architectural modifications, offering a practical path toward robust multimodality medical object detection.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a framework that aligns internal object-query representations with modality context, allowing one unified detector to perform well across diverse imaging types. The proposed framework makes the detector “aware” of each modality through lightweight modality tokens, text embeddings such as “tumor in MRI” or “aortic enlargement in CXR.” In a short pretraining stage called QueryREPA, the model aligns its internal object queries with these tokens using contrastive learning, producing modality-aware query representations. During detection, a Multimodality Context Attention (MoCA) mechanism appends the modality token to the query set so the model can attend to modality context while predicting bounding boxes. Tested on a large mixed-modality dataset built from seven public sources, the method consistently outperformed strong baselines like DINO and Deformable DETR, improving average precision by 3–4 points with almost no extra computation, and yielding notably sharper and more accurate lesion localization.

This paper shows that a simple, modular alignment strategy, combining lightweight text-based modality tokens, a short contrastive pretraining, and a self-attention fusion step, can make standard detectors far more robust in multimodality medical settings.

### Strengths
- The paper is clearly written, with a logical flow from motivation to method and experiments. The figures (especially the MoCA and QueryREPA schematic) effectively illustrate the pipeline.

- The challenge of building a single detector that generalizes across medical imaging modalities (CXR, CT, MRI, pathology) is practically relevant and timely for generalist medical AI.

- The approach achieves modest but consistent AP gains across all tested modalities and object scales, demonstrating robustness and reproducibility of the improvement.

- The paper includes thorough training details, dataset descriptions, and implementation notes (appendices), supporting easy replication.

### Weaknesses
The core ingredients, contrastive representation alignment, modality tokens, and self-attention fusion, are not fundamentally new. Similar strategies already exist, such as in Grounding DINO. The conceptual novelty is thin; it’s a tidy engineering blend rather than a new paradigm.

Although described as “detector-agnostic,” the method still assumes a query-based transformer architecture. Extending it to CNN-style or diffusion-based detectors would likely require non-trivial adaptation.

The proposed “representation alignment” operates only at a coarse level, matching query and modality-token embeddings without ensuring semantic disentanglement or true feature sharing across modalities. As a result, the model may still overfit modality-specific cues rather than learning transferable visual structure.

Although, the authors do not explicitly claim reasoning ability, the scope remains limited to detection only. The model learns context alignment, not semantic reasoning, it cannot interpret relationships between findings (e.g., “lesion near the diaphragm”) or answer higher-level questions, and therefore remains a low-level detector.

### Questions
1. The paper combines contrastive alignment, modality tokens, and self-attention fusion, all of which have been explored in prior works (e.g., Grounding DINO, REPA). Could the authors clarify what is fundamentally new in this combination beyond engineering integration?

2. The alignment between query and modality-token embeddings seems to operate at a coarse global level. Have you examined whether the model actually learns shared semantic subspaces across modalities (e.g., via visualization or clustering metrics)?

3. Your framework focuses on aligning modality-aware object queries for improved detection across heterogeneous medical imaging types. Given the recent progress of unified multimodal models such as Qwen-VL and GPT-4V, which can already perform vision–language reasoning and coarse localization, how would your approach compare to adapting such a large multimodal foundation model for structured object detection (e.g., by adding a query-alignment or visual token–to–query head)? In your view, what are the trade-offs between starting from a domain-specific alignment framework like MoCA + QueryREPA versus extending a general VLM into a precise medical detector in terms of localization accuracy, data efficiency, and scalability?

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
5

### Summary
The paper proposes Align Your Query, a detector-agnostic framework for multimodality medical object detection that introduces Multimodality Context Attention (MoCA) to fuse text-derived modality tokens with object queries and QueryREPA contrastive pretraining to align query representations.

### Strengths
1.The paper is clearly written and well organized, making the technical background, proposed methods, and experimental results easy to follow.

2.The idea of aligning DETR-style object queries with text-derived modality tokens is conceptually interesting and demonstrates moderate empirical gains in multimodality medical detection.

### Weaknesses
1.The technical novelty is limited, as the work largely adheres to existing DETR-style frameworks and primarily adds lightweight extensions rather than proposing a fundamentally new detection paradigm.

2.The proposed QueryREPA pretraining introduces additional complexity without clear evidence that it significantly improves beyond standard multimodal contrastive alignment methods.

3.The evaluation lacks sufficient analysis on large-scale or real clinical datasets, limiting confidence in the method’s generalizability and real-world impact.

4.The paper’s ablation and visualization results, while supportive, do not thoroughly isolate the individual contributions of MoCA and QueryREPA, making the source of performance gains somewhat ambiguous.

### Questions
1.How does the proposed method go beyond prior DETR-style architectures to offer genuine methodological novelty?

2.Can the authors provide clearer evidence that QueryREPA offers distinct benefits over standard contrastive pretraining approaches?

3.Have the authors considered evaluating on larger or clinically diverse datasets to better demonstrate real-world generalization?

4.Can the authors expand the ablation studies to more clearly separate the effects of MoCA and QueryREPA?

### Soundness
2

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
3

### Summary
The paper proposes a DETR-based object detection framework tailored for multimodal medical imaging, introducing two key components, MoCA  and Query REPA, that reportedly enhance the performance of existing detection models in medical scenarios.

### Strengths
1. The proposed method requires no modification to the underlying detector architecture, incurs negligible computational overhead, and is compatible with other approaches. 
2. Experiments span multiple medical imaging modalities (e.g., X-ray, MRI, ultrasound), demonstrating the model’s generalizability. Both quantitative results and qualitative visualizations are provided, offering clear and intuitive evidence of effectiveness.

### Weaknesses
### **Major Weaknesses**

1. The paper mentions “heterogeneous statistics and disjoint representation spaces” at the beginning of the abstract, but the main text does not further elaborate on whether the proposed method addresses this issue, nor does it explain how it does so.
2. Regarding the paper’s two core contributions, MoCA and QueryREPA, although the authors describe them in a very elaborate manner, I do not perceive them as particularly novel. But my familiarity with DETR-based methods is limited, so I would like to hear the opinions of other reviewers on this matter.
3. Proposition 1 appears rather trivial. It is widely recognized that increasing the batch size can enhance the effectiveness of the InfoNCE loss.

### Minor Weaknesses

1. The baselines are somewhat outdated. The paper should include more recent and competitive baselines.
2. It seems that each modality uses a separate, dedicated image encoder. The authors should provide clearer clarification on this point.
3. Suggest to cite the following paper:
    
    Liu, Y., Xi, S., Liu, S., Ding, H., Jin, C., Zhong, C., ... & Shen, Y. (2025). Multimodal Medical Image Binding via Shared Text Embeddings. *arXiv preprint arXiv:2506.18072*.

### Questions
The authors need to address the issues I raised in the Weaknesses section, primarily clarifying why the proposed method works and articulating the core novelty of the approach.

I will base my final scoring decision on the authors’ responses to these points, as well as feedback from other reviewers.

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
3

### Summary
The motivation of this paper is to generate representations that generalize across a mixed training corpus of multi-modality medical data using ideas from representational alignment. They focus on object detection problems and mix modality tokens in with object query tokens to allow the attention mechanism to vary by modality.

### Strengths
This alignment strategy can work with different types of encoders, providing flexibility. 

Clear performance gains from using this method compared to and on top of related work. Shows that improvement is not dependent on specific implementation details unrelated to the proposed method .

Minimal computational overhead by design.

### Weaknesses
QueryREPA adds pretraining before end-to-end detection training, but this is a noted limitation. 

The modality tokens are produced via a fixed prompt and encoded with off-the-shelf general or biomedical CLIP variants. This approach captures coarse modality and class semantics but may lack granularity for subtle medical image distinctions. The study does not analyze failure cases for rare or complex conditions and doesn't examine potential limits of using static text encoders for domains with volatile or ambiguous language (pathology subtypes, COVID lesions, etc.). Are better prompts available? How does tokenization performance degrade as class granularity increases or when new modalities are added?

Needs better evaluation against prior work involving representation alignment applied to similar problem settings (either natural or medical data domains), or more thorough explanation in the related work section of why prior works are not comparable.

### Questions
See previous sections.

### Soundness
4

### Presentation
4

### Contribution
3
