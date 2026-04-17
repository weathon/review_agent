# Interaction-aware Representation Modeling With Co-Occurrence Consistency for Egocentric Hand-Object Parsing

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6, 6

## Abstract
A fine-grained understanding of egocentric human-environment interactions is crucial for developing next-generation embodied agents.
One fundamental challenge in this area involves accurately parsing hands and active objects.
While transformer-based architectures have demonstrated considerable potential for such tasks, several key limitations remain unaddressed:
1) existing query initialization mechanisms rely primarily on semantic cues or learnable parameters, demonstrating limited adaptability to changing active objects across varying input scenes;
2) previous transformer-based methods utilize pixel-level semantic features to iteratively refine queries during mask generation, which may introduce interaction-irrelevant content into the final embeddings;
and 3) prevailing models are susceptible to ``interaction illusion'', producing physically inconsistent predictions.
To address these issues, we propose an end-to-end Interaction-aware Transformer (InterFormer), which integrates three key components, i.e., a Dynamic Query Generator (DQG), a Dual-context Feature Selector (DFS), and the Conditional Co-occurrence (CoCo) loss. 
The DQG explicitly grounds query initialization in the spatial dynamics of hand-object contact, enabling targeted generation of interaction-aware queries for hands and various active objects.
The DFS fuses coarse interactive cues with semantic features, thereby suppressing interaction-irrelevant noise and emphasizing the learning of interactive relationships.
The CoCo loss incorporates hand-object relationship constraints to enhance physical consistency in prediction. Our model achieves state-of-the-art performance on both the EgoHOS and the challenging out-of-distribution mini-HOI4D datasets, demonstrating its effectiveness and strong generalization ability.
Code and models are publicly available at https://github.com/yuggiehk/InterFormer.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents InterFormer, a transformer-based model for egocentric hand–object segmentation. It introduces three key components, Prototypical Query Generator (PQG), Dual-context Feature Selector (DFS), and Conditional Co-occurrence (CoCo) loss to improve interaction-aware representation and reduce physically inconsistent predictions. The model achieves state-of-the-art results on EgoHOS and mini-HOI4D datasets.

### Strengths
1. This paper provides clear figures to clarify the motivation and methodology. 
2. The method achieves competitive performance on the reported benchmarks.

### Weaknesses
1. Limited novelty and over-engineered design. The paper mainly achieves performance gains through stacking multiple existing components, relying on a cascaded U-Net-style decoder to generate priors and a sequence of post-processing steps. The overall method appears as an incremental engineering integration rather than a conceptually novel contribution.
2. Inconsistent motivation and contribution. The authors claim that existing query initialization methods “lack adaptability to diverse categories of contacting objects,” and thus propose the Prototypical Query Generator (PQG). However, PQG only guides the network toward hand–object contact regions via similarity-based subregion selection and has little to do with modeling category diversity.
3. Misleading use of the term “Prototypical”. The proposed Prototypical Query Generator (PQG) constructs queries by combining interaction priors, subregion Top-N selection, and a learnable bias，yet the connection to 'prototypical' concepts appears tenuous. 
4. Lack of model efficiency analysis. The method adds multiple stages and operations (e.g., Top-K feature selection), yet provides no analysis or comparison of model efficiency in terms of computational cost, or inference speed. 
5. Insufficient diversity of datasets. The experiments are mainly conducted on mini-HOI4D and EgoHOS, which are highly similar in structure and data distribution. Incorporating additional datasets such as Ego-IRGBench[1] and VISOR-NVOS[2] is recommended. 
[1] ANNEXE: Unified Analyzing, Answering, and Pixel Grounding for Egocentric Interaction. CVPR, 2025.
[2] Learning to segment referred objects from narrated egocentric videos. CVPR, 2024.
6. Poor writing quality. For instance, the notation system is inconsistent (L362), the references to Leonardi et al. (2024) and EgoHOS (L44) are unrelated, the citation for mini-HOI4D (L383) appears incorrect, and the claim "most existing methods" (L206) lacks supporting citations. 
7. Limited practical applicability and broader impact. While the study is technically well-executed, its potential influence on future research directions appears modest.

### Questions
See weakness.

### Soundness
2

### Presentation
2

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
This paper presents InterFormer, a transformer-based framework for egocentric hand–object parsing that introduces interaction-aware representation learning. It integrates four key modules: the Interaction Prior Predictor (IPP) to capture contact boundaries, the Prototypical Query Generator (PQG) to create dynamic interaction-grounded queries, the Dual-context Feature Selector (DFS) for fusing local and global context, and the Conditional Co-occurrence (CoCo) loss to enforce physical consistency. The method achieves superior performance on EgoHOS and mini-HOI4D benchmarks

### Strengths
The proposed IPP–PQG–DFS–CoCo framework is non-trivial and well-structured, explicitly modeling physical contact and interaction awareness within a transformer-based segmentation pipeline.

By introducing boundary-guided priors (IPP) and the Conditional Co-occurrence loss, the method embeds physical and causal constraints that significantly reduce implausible  errors.
The paper is well written and easy to follow.

### Weaknesses
Incomplete SOTA coverage. The authors do not discuss or compare to HOIST-Former (CVPR 2024), a recent and stronger transformer-based approach that explicitly models spatio-temporal hand–object relations. This omission weakens the manuscript’s literature coverage and contextualization.

Insufficient treatment of occlusion. High occlusion is a core challenge in egocentric settings, yet the paper does not analyze how the proposed method handles heavily occluded hands or objects. In particular, there is no discussion or visualization showing how the transformer attention mechanism behaves on occluded regions or how IPP/PQG mitigates occlusion-induced errors.

Limited out-of-distribution evaluation. The model is trained in a fully supervised manner, which can limit generalizability. This concern is reinforced by the absence of evaluations on large, diverse egocentric benchmarks such as EPIC-KITCHENS and Ego4D. The authors should either justify this omission or provide cross-dataset results.

Shallow comparison to large multimodal / foundation models. Although the paper includes MLLM baselines (e.g., ANNEXE, Care-Ego), the comparison and accompanying analysis are brief. The manuscript lacks an exploration of why InterFormer outperforms these models or where MLLMs might still be advantageous (for example, open-vocabulary categories or rare classes). A detailed error analysis contrasting InterFormer and MLLM failures would strengthen the evaluation.

Weak theoretical motivation for design choices. Some architectural decisions are insufficiently justified. For example, the PQG selects the top-N similarity subregions using a fixed partition—why this fixed pooling strategy rather than a learned selection mechanism? Provide either theoretical intuition or ablation results showing sensitivity to NNN and partition size

### Questions
Refer weaknesses

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
This paper presents InterFormer, a claimed novel transformer-based framework for Egocentric Hand-Object Segmentation (EgoHOS) task which is now a popular egocentric tasks under CV area. The authors address key limitations in SOTA: inflexible query initialization, reliance on noisy pixel-level features, and physically inconsistent predictions; by incorporating 3 major components: (i) a Prototypical Query Generator (PQG) that creates adaptive queries by fusing learnable parameters with interaction-relevant context ; (ii) a Dual-context Feature Selector (DFS) that refines feature representations by combining semantic and interactive cues ; (iii) a Conditional Co-occurrence (CoCo) loss that enforces physical plausibility by penalizing object predictions when corr. hand is not detected. The proposed model is evaluated on the EgoHOS and mini-HOI4D datasets, where it achieves good performance. This is an overall good paper, only lacking in theoretical analysis and qualitative illustrations.

### Strengths
1. The paper points to a key area, i.e. handling contact regions with a prior predictor for interactions.
2. Instead of a one size fits all - the approach of handling specific SOTA technique drawback by the most optimal way is a fresh look -- PQG, DFS, and CoCo loss.
3. CoCo loss is a simple solution to handle the interaction illusion problem.
4. Codebase will be released if accepted and appendix section helps some clarifications.
5. Comparison with extensive SOTA establishes good empirical results.

### Weaknesses
1. Instead of an end-end pipeline, the proposed modular approach may lead to complexity in time, space and a cut in error backprop.
2. The dependency of the later modules on IPP, makes the contextual training necessary for IPP for generalization.
3. Anonymous codebase was expected for validity along with more supplementray to show qualitative results and failure cases.
4. Certain variables like CocoLoss params were found by ablation on specific datasets, thereby reducing generalization in lack of non-empirical justification. Hard thresholds can cause problems in other datasets.
5. Real life deployment on a actual glass should have proved robustness.

### Questions
1. The figures could be explained with a simple end to end example instead of just architecture.
2. Equation on Coco loss is straight-forward. any weights on the combined loss?
3. How actually is interaction-relevant context extracted and how to map perception to this, in order to represent context?
4. Any specific reason for DFS, there can be multiple parallel feature selector methods.

### Soundness
3

### Presentation
3

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
Authors propose a new architecture which explicitly models interaction by using explicit hand-object boundaries, and the results features are used along with pixel level features to extract hand-object interactions. The proposed approach also claims to address the problem of interaction illusion where the model incorrectly predicts objects being interacted with under implausible scenarios (hand not visible generally). Experiments demostrate improved performance over state of the art approaches on in-domain and out of domain datasets.

### Strengths
1. The idea to explicitly model interaction for extract hand-object segmentation masks under interaction is highly intuitive.
2. The experimental results support the claim of improving performance over SOTA on various datasets; qualitative evidence is also shown to strengthen that claim.
3. Paper tries to address the problem of interaction illusion which forces the model to learn causality over correlation (weakly speaking).

### Weaknesses
1. The paper uses a lot of ambiguous and vague terms - "enrichment", "structural priors", "task-relevant", "preliminary coarse interactive representations". I urge the authors to not use them since it causes confusion and distracts from understanding the actual core contributions. I would recommend a proper rewriting of the paper.
2. The idea to use interaction cues (from Zhang et al., 2022, EGOHOS) is useful, but it is important to highlight if other methods lack this supervision. If they do, then the comparison is not fair - it is still useful to compare but important to disambiguate the role of architectural changes vs supervision. Some ablations do cover the impact of just introducing interaction loss (IPP) - focusing on mAcc in Table 4, most of the improvement is actually from IPP. Can authors retrain any baseline by adding another head that predicts hand-object boundaries, then measure how much is the improvement.
3. Can authors show qualitative comparison on the models training with and without CoCo loss, how much does that help with interaction illusion. If authors do claim improvement, it would be useful to do a small quantitive study comparing the prominence of this issue in other methods vs the proposed one.
4. Equations 7, 8, 9 are unnecessarily complex, please just write $I_{N_{lh}<\tau} \cdot N_{lo}$, this is simpler to read which means penalize interaction when the hand is not visible.

### Questions
1. Please clearly focus on 2-3 contributions - multiple small contributions are harder to validate. For example why is the number of pixels being minimized instead of the cross entropy loss directly by penalizing the prediction probability corresponding to object undergoing interaction.
2. Why is $\tau$ in pixels instead of normalized coverage in the image? Moreover hands can be much closer to the camera where pixel based coverage might not be enough and could be normalized based on the hand size (GT hand mask can provide that).

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
To solve inflexible query initialization, pixel-level feature noise, and "interaction hallucinations" of Transformer-based EgoHOS methods, this paper proposes InterFormer. It integrates three key components: Prototype Query Generator (PQG) for adaptive queries, Dual Context Feature Selector (DFS) for noise reduction, and Conditional Co-occurrence (CoCo) Loss for physical consistency. The model achieves SOTA on EgoHOS in-domain, out-of-domain, and mini-HOI4D, with core contributions in explicit interaction-aware modeling and improved accuracy/generalization.

### Strengths
- Targeted Problem-Solving: It accurately identifies three critical EgoHOS pain points (e.g., interaction hallucinations harming agent safety) and designs solutions accordingly, ensuring tight problem-solution alignment.
- Logical Component Design: PQG, DFS, and CoCo Loss form a cohesive pipeline (query generation -> feature processing -> result optimization), with progressive and rigorous logic.
- Comprehensive Experiments: It outperforms baselines on out-of-domain/OOD datasets (+5.09%/+11.4%) and uses ablation experiments to verify each component’s effectiveness, ensuring credible conclusions.
- Reproducibility & Openness: The paper promises reproducibility and open-source availability, which are crucial for facilitating further research.

### Weaknesses
Section 3.4 mentions that the presence of a hand is a fundamental prerequisite for any hand-object interaction; when the right hand is not detected in the prediction results, current models may incorrectly classify the interacting object as being 'operated by both hands' despite the absence of one hand (the right hand). In terms of results, the CoCo Loss designed based on this premise has indeed achieved good performance in tests. However, this assumption has a critical flaw: the fact that a hand is not detected in the frame does not mean the hand is absent from the current interaction. If hastily determine whether a hand is involved in the interaction based solely on whether it is detected in the frame, the inferences drawn will violate physical common sense in more realistic and complex scenarios (e.g., large changes of viewpoint , or when the perspective can only observe one of the hands).

### Questions
See Weakness.

### Soundness
2

### Presentation
4

### Contribution
3
