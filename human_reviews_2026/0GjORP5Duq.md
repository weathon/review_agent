# RACA-CLIP: Relation-Aware Compositional Alignment for CLIP

- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Vision-Language Models (VLMs) such as CLIP excel at broad multimodal tasks, yet struggle with compositional reasoning. Despite capturing coarse correlations, they often act like “bags-of-words” missing fine-grained structures such as object–attribute bindings and inter-object relations. We attribute this to: (i) limited compositional diversity in large-scale image–text data, and (ii) contrastive objectives that emphasize global alignment over grounded structure. To address this, we propose a hierarchical fine-grained alignment framework that explicitly bridges visual and textual components at the object, attribute, and relation levels. Unlike prior work relying on parsers, we leverage scene graph annotated datasets for structured supervision, requiring no extra labeling. We introduce a hierarchical fine-grained loss to complement standard contrastive learning by grounding entities and relations across modalities. Experiments on compositional benchmarks SugarCrepe, What’sUp, and Cola show large gains in capturing nuanced structure, while preserving performance on standard vision-language tasks. RACA CLIP method improves compositional reasoning accuracy by +24.86% on SugarCrepe, +5.7% on What’sUp, and +4.76 on Cola, offering a simple yet effective path toward stronger, human-like compositional understanding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the persistent challenge of compositional reasoning in vision–language models such as CLIP. 
It proposes RACA-CLIP, a structured contrastive learning framework that integrates scene-graph supervision to align visual and textual representations at the object, attribute, and relation levels.
The method achieves consistent gains on compositional reasoning benchmarks and maintains competitive zero-shot performance on ImageNet and retrieval tasks.

### Strengths
1. The paper propose a structured contrastive framework that integrates scene-graph representations into CLIP models, which aligns image regions with corresponding text descriptions via IoU-based multi-positive matching.
2. Comprehensive experiments on compositional benchmarks demonstrate the effectiveness with substantial gains and analysis. The inclusion of weight interpolation analysis and representation-level statistics strengthens interpretability.

### Weaknesses
1. Although RACA-CLIP claims improved compositional reasoning, the evaluation relies mainly on accuracy gains on compositional benchmarks.  There is no causal or probing analysis that clearly isolates whether improvements come from structure-aware learning or simply data augmentation with GBC scene graphs.
2. The paper uses ViT-B and LoRA fine-tuning, but doesn’t examine whether the approach scales gracefully to larger models (e.g. ViT-L).
3. Qualitative analysis would help verify whether the claimed improvements correspond to better interpretability rather than just numeric gains.

### Questions
How sensitive is RACA-CLIP to the quality and noise of scene graph annotations?

### Soundness
3

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
4

### Summary
This paper proposes RACA-CLIP, a structured contrastive learning framework designed to improve compositional reasoning in vision-language models (VLMs), specifically focusing on relation-aware and attribute-grounded alignment. The method introduces region-level contrastive learning with IoU-weighted alignment between detected objects and caption spans, as well as a triplet supervision mechanism over structured ⟨subject, relation, object⟩ units, leveraging scene-graph annotations to provide fine-grained grounding signals. The approach preserves the dual-encoder architecture of CLIP while injecting relational inductive bias into the learned embeddings. Experiments across five compositional benchmarks show consistent improvements over CLIP and other enhanced baselines, including large gains in SugarCrepe’s Add and Swap settings (+16.24 and +24.86), while retaining — and occasionally improving — zero-shot recognition and retrieval performance. Ablation studies and controlled analyses suggest that the performance gains stem from improved binding between objects, attributes, and relations, rather than from memorization or dataset artifacts. Overall, the paper contributes an impactful improvement to a key weakness of modern contrastive VLMs.

### Strengths
The paper identifies fundamental limitations of contrastive VLMs such as CLIP, citing the lack of structural inductive bias and over-reliance on global alignment that ignores object–attribute bindings and inter-object relations. This motivation is compelling and well-supported by prior analyses.

The proposed hierarchical alignment introduces region-level IoU-weighted contrastive learning and relation-aware triplet supervision (⟨s, r, o⟩), explicitly modeling the compositional structure of images and captions in a way that complements standard CLIP training.

The method leverages scene-graph annotated datasets to enable alignment supervision without requiring additional labeling efforts, improving practicality and efficiency.

### Weaknesses
The approach requires accurate object, attribute, and relational supervision. The authors acknowledge that performance may degrade with lower graph fidelity, but no robustness evaluation is provided.

The method introduces computational overhead (region features + triplet losses), yet there is no cost or latency analysis of fine-tuning or inference, which affects real-world use cases.

Graphs primarily capture physical/spatial relations; more abstract or high-level reasoning (e.g., intent, affordance, temporal actions) is not evaluated. Compositional generalization outside benchmarks like SugarCrepe and What’sUp remains unclear.

### Questions
Do the improvements correlate with real-world compositional benchmarks outside synthetic evaluations (SugarCrepe-like)?

Can the method be applied to generative multimodal models (e.g., aligning decoder-token grounding)?

### Soundness
3

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
5

### Summary
This paper proposes to use scene-graph as a way to augment training in CLIP for enahnced compositional understanding capacity. The method uses existing scene-graph dataset to supervise region-aware contrastive learning, with improvement shown in the downstream benchmarks.

### Strengths
The paper is clearly written and easy to follow. The proposed method is well explained, particularly around lines 264 and 298. On the selected benchmark, it demonstrates improved performance compared to the baseline.

### Weaknesses
1. Limited novelty:
The idea of using scene graphs as external supervision has already been shown to improve performance [1,2,3]. Compared to these works, this paper introduces very limited novelty. The claim that “scene graph annotated datasets are leveraged for structured supervision without additional labeling” is misleading, as these datasets are still manually annotated—only pre-processed differently. Using such well-annotated data for contrastive supervision is not a particularly novel or interesting contribution.

2. Incomplete evaluation:
The evaluation is insufficient. While the paper reports results on SugarCrepe, it does not evaluate on other established benchmarks such as Winoground or MMVP, which would better demonstrate generalization and fine-grained reasoning improvements.

3. Insufficient related work review:
Prior studies [1,2,3] have already explored using scene graphs to enhance fine-grained understanding through contrastive learning, yet this paper provides very limited discussion of them. Additionally, works like [1,4] also achieve strong results on SugarCrepe and should be included in the comparison table for completeness.



[1] Huang, Yufeng, et al. "Structure-clip: Towards scene graph knowledge to enhance multi-modal structured representations." Proceedings of the AAAI conference on artificial intelligence. Vol. 38. No. 3. 2024.
[2] Herzig, Roei, et al. "Incorporating structured representations into pretrained vision & language models using scene graphs." arXiv preprint arXiv:2305.06343 (2023).
[3] Li, Liunian Harold, et al. "Grounded language-image pre-training." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[4] Zhang, Le, Rabiul Awal, and Aishwarya Agrawal. "Contrasting intra-modal and ranking cross-modal hard negatives to enhance visio-linguistic compositional understanding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
How do authors justify the novelty of the method compare to [1,2,3]?

### Soundness
2

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
2

### Summary
This paper proposes RACA-CLIP, a structured contrastive framework that enhances CLIP’s compositional reasoning by integrating scene-graph-based supervision. This paper introduces region-level IoU-weighted alignment and relation-aware triplet losses to better capture object–attribute bindings and inter-object relations. Trained on the graph-based captioning dataset, RACA-CLIP achieves large improvements on compositional benchmarks

### Strengths
The model demonstrates robust performance on several compositional benchmarks, outperforming CLIP, NegCLIP, and TripletCLIP while maintaining strong zero-shot and retrieval accuracy.

This paper clearly explains the global-only limitation of CLIP and motivates structured alignment with strong intuition and references.

### Weaknesses
The method heavily depends on scene-graph annotations and LLM-based triplet extraction, which may introduce noise or inconsistencies and limit scalability to unstructured web data.

The computational cost of extracting scene graphs and aligning multiple fine-grained objectives isn’t discussed. Could you please discuss it?

If the impact on broader downstream tasks, such as visual question answering or image generation, this paper will be more comprehensive.

### Questions
Refer to Weaknesses

### Soundness
3

### Presentation
3

### Contribution
4
