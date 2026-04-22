# Collaborative Unpaired Multimodal Learning for Image Classification

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 2, 2, 8, 4

## Abstract
Multimodal learning typically requires expensive paired data for training and assumes all modalities are available at inference. Many real-world scenarios, however, involve unpaired and heterogeneous data distributed across institutions, making collaboration challenging. We introduce Unpaired Multimodal Learning (UML) as the problem of leveraging semantically related but unaligned data across modalities, without requiring explicit pairing or multimodal inference. This setting naturally arises in collaborative scenarios such as satellite imagery, where institutions collect data from diverse sensors (optical, multispectral, SAR), but paired acquisitions are rare and data sharing is restricted. We propose a collaborative framework that combines modality-specific projections with a shared backbone, enabling cross-modal knowledge transfer without paired samples. A key element is post-hoc batch normalization calibration, which adapts the shared model to each modality. Our framework also extends naturally to federated training across institutions. Experiments on multiple satellite benchmarks and additional visual datasets show consistent improvements over unimodal baselines, with particularly strong gains for weaker modalities and in low-data regimes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses Unpaired Multimodal Learning (UML), where institutions have semantically related but unaligned data across modalities without explicit pairing or requiring multimodal inference at test time. The authors propose a collaborative framework combining modality-specific projections with a shared backbone, enabling cross-modal knowledge transfer.
1. The technical approach centers on modality-specific projections \$ f_k(\cdot) \$ mapping inputs to a shared latent space \$ Z \$, followed by a shared backbone \$ g(\cdot) \$ for classification, complemented with post-hoc batch normalization calibration.
2. The framework includes a natural extension to federated learning, where only the shared backbone parameters are aggregated, preserving modality-specific projections locally.

### Strengths
1. The paper tackles a less-studied, practically motivated problem setting, especially relevant for satellite imagery. The provided motivating example is appreciated, though further clarification would be helpful.
2. The application of modality-specific batch normalization calibration within a shared network is well-addressed and represents a sensible approach to modality heterogeneity.
3. The comparisons made with multi-task learning and federated learning frameworks are insightful. This work may also be interpreted as a form of knowledge distillation or cross-modal transfer learning.
4. The limitations section is commendable for its thoroughness and transparency.

### Weaknesses
1. The introduction’s clarity could be improved. Rather than clarifying the contributions, it leads to more questions. In particular, it is unclear how the model learns both a shared representation and modality-specific uniqueness without any explicit pairing information and it is unclear what unpaired yet semantically coherent data samples across modalities look like.
2. The theoretical formulation is confusing and, at times, appears contradictory. The assumptions of unpaired modalities, semantic coherence, and a shared label space are neither rigorously formulated nor validated. Specifically, the interplay between these assumptions seems paradoxical, as the presence of semantic coherence suggests a form of alignment contradictory to the stipulated absence of a correspondence mapping. Here is a foundational contradiction that raises concerns about the validity of the core assumptions. 

Specifically, the paper posits the existence of unpaired modalities—that is, no explicit correspondence function \$ \varphi : X_j \to X_k \$ exists between modalities (Key Constraint, lines 116-118)—while simultaneously assuming semantic coherence expressed as shared semantic space mappings \$ \psi_k : X_k \to S \$ for each modality (Assumption 1, lines 125-129). This assumption implies the existence of a relation between representations of different modalities in a common latent space \$ S \$.

By composing these mappings, one can construct a mapping between modalities:

$$
\varphi = \psi_k^{-1} \circ \psi_j: X_j \to X_k,
$$

contradicting the initial claim of the absence of such a correspondence function. This contradiction suggests that semantic coherence, as formulated, logically implies some form of alignment between modalities, thereby undermining the claim of unpairedness. The authors should clarify this apparent paradox and explicitly differentiate between the notions of semantic coherence and modality pairing, potentially revisiting assumptions or refining definitions to avoid this logical inconsistency.

3. Greater concreteness in distinguishing between modality pairing and semantic coherence would strengthen the paper. Clear definitions and examples, especially in the context of satellite imagery (e.g., SAR versus optical sensors), would clarify these foundational concepts.
4. The theoretical underpinnings explaining why a shared backbone can effectively learn transferable representations across unpaired modalities are limited.
5. The heavy focus on satellite imagery benchmarks constrains the generalizability claims; extending evaluation to other modalities or domains would enhance impact.
6. The architectural design choices lack justification and broader exploration. The brief mention of ResNet-18 and reliance on Appendix details do not suffice. A more comprehensive evaluation of architectures would be valuable and also enhance applicability beyond the very specific problem space this work is tested on.
7. Core algorithmic novelty is limited; the shared backbone and batch norm calibration strategies align with existing practices from multi-task and federated learning literature. The main contribution is primarily in problem formulation, which is also unclear.

### Questions
*Clarifications:*

1. The authors mention in lines 128-129 that SAR and optical sensors observe the same phenomena, hence they are semantically coherent but not aligned. What would aligned or paired data look like between these sensors in practice? To my understanding, observing the same phenomena from different sensors is paired data. Furthermore, these lines state that the assumption of semantic coherence does not hold for text-to-image pairs. Does this imply that captions and images cannot share semantic correlation? Could you please elaborate on these points to clarify the distinction between semantic coherence and alignment?
2. Could the authors provide citations for the ISCA baseline referenced in line 312? It would be helpful if baseline descriptions included references.
3. Regarding Table 3, could you clarify how the paired data training is performed?
4. In what practical scenarios would institutions share models but not data samples? Further elaboration on this would help ground the application.
 
*Thought Experiments and Extensions:*

5. Theoretical Justification: Can you provide theoretical analysis of why the shared backbone learns transferable representations across unpaired modalities? What conditions guarantee effective knowledge transfer?
6. Scalability Analysis: How does performance scale with the number of modalities (K>2) and severely imbalanced data distributions across institutions?
7. Architecture Sensitivity: How sensitive is the method to the projection/backbone capacity split? Could you provide principled guidelines for this design choice?
8. Beyond Satellite Imagery: Can you demonstrate the generalizability of your method beyond satellite imagery? Experiments or motivating examples in medical imaging or robotics would strengthen the work.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper defines Unpaired Multimodal Learning (UML) as the task of leveraging semantically related but unaligned data across multimodality. The authors also propose a lightweight framework for UML and test it across 3 satellite benchmarks and additional visual datasets.

### Strengths
- The paper is well written and easy to follow
- The idea is simple and shows some effectiveness in the experiments
- Providing analysis and ablation study

### Weaknesses
- The motivation could be clarified further. It is not entirely clear whether the authors assume the presence of different modalities or multiple similar datasets from different institutions. Regarding the proposed approach, its novelty appears somewhat limited, as it primarily involves training a model with multiple convolutional blocks for each subdataset and applying separate batch normalization layers at the end. Additionally, the reviewer would like to clarify whether the Unimodal results reported in Table 3 were obtained using the entire dataset or only the corresponding subdataset.
- The benchmark includes several datasets that are quite similar and share the same set of labels, which reduces realism. In two out of three cases, the results show only a slight improvement over those obtained using the stronger modality (stronger dataset).
- Assumption 2 is quite **restrictive** and not realistic. Sharing label space does not always happen between institutes. Additionally, Assumption 1 also does not support the motivation of the paper. Each modality uses different modality-specific projections, which lead to capturing different features. 
- The key constraint also seems to be unnecessary.

### Questions
Please refer to Weaknesses.

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
3

### Summary
The paper defines Unpaired Multimodal Learning (UML): collaborating across semantically related but unaligned modalities to improve unimodal inference—no paired examples and no multimodal inputs at test time. 
It proposes a simple framework with modality-specific projections feeding a shared backbone trained under a common supervised objective; after training, it performs post-hoc BatchNorm (BN) recalibration per modality using cumulative moving averages, and optionally extends to federated learning by aggregating only the shared backbone (FedAvg) while keeping per-modality heads local. 
Experiments on BigEarthNet-MM, SEN12MS, and EuroSAT S1-RGB show consistent gains over unimodal baselines and a margin over adapted DA baselines and ISCA; ablations indicate BN recalibration is critical, and gains are largest in low-data regimes.

### Strengths
1. Problem framing & practicality. The UML setting is clearly scoped and practically motivated. The storyline circulates about the unified objective: improve each modality’s classifier without paired data or multimodal inference.

2. Method simplicity. The approach is a lightweight multi-task-style sharing of a backbone with per-modality stems, followed by post-hoc BN recalibration. This is straightforward to adapt and show consistent performance gain via authors’ experiments.

3. Federated extension is clean and relevant. The problem of UML is close to the practical setting that FL literature is addressing. Aggregating only the shared backbone via FedAvg is a natural design choice for privacy and heterogeneity.

### Weaknesses
1. Assumption 2 about Shared Label Space might be a bit restrictive. In practice, collaborators may not share identical label sets: some clients lack classes (vacant classes / label skew), or taxonomies differ; these cases are studied in open-set, partial-set, and universal domain adaptation [1,2]. 

2. Limited novelty beyond well-known components. The core recipe—shared backbone (multi-task-like) + per-modality stems + BN recalibration—is straightforward. The post-hoc BN step echoes established BN statistics re-estimation/PreciseBN/EvalNorm ideas [3]; thus, while the combination for UML is useful, the conceptual novelty is modest for ICLR.

3. Baselines could be stronger/deeper.
(a) Since the benefit hinges critically on BN, comparisons to BN-centric alternatives (e.g., EvalNorm, AdaBN, test-time BN variants) would clarify whether plain post-hoc BN recalibration alone explains most gains.
In the federated setting, a direct comparison to FedBN (keeping BN local) seems particularly relevant; the paper cites related work but does not report head-to-head numbers.

4. Dataset curation choices may simplify the problem. The paper rebalance(s) and convert(s) originally multi-label datasets into single-label classification, reducing realism. Moreover, the “fine-grained per-band as a client” setup, while stress-testing scalability, is not a typical deployment scenario.

5. Theoretical support is high-level. The paper offers intuition but no formal guarantees about when collaboration helps vs. hurts (negative transfer), or how BN drift relates to modality imbalance. A bit more formalization (even simplified) could elevate the contribution.

References:
[1] Garg, Saurabh, Sivaraman Balakrishnan, and Zachary Lipton. "Domain adaptation under open set label shift." Advances in Neural Information Processing Systems 35 (2022): 22531-22546.
[2] Ma, Xinhong, Junyu Gao, and Changsheng Xu. "Active universal domain adaptation." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
[3] Singh, Saurabh, and Abhinav Shrivastava. "Evalnorm: Estimating batch normalization statistics for evaluation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors introduce a federated, multi-modal representation learning algorithm (Alg. 1 and 3) that allows the training a deep neural network without explicit pairing of modalities. Experiments have been conducted on Sentinel-1/2 Earth observation data to demonstrate a boost in classification accuracy of up close to 12% (Fig. 2b). Leveraging a modality-specific reset and adjustment of batch normalization parameters post training (Alg. 2) is reported as key ingredient for the gains in performance. A ResNet-18(-like) model architecture serves as backbone.

### Strengths
The paper resembles a sound piece of technical work with clear, numerous, and systematic experiments (Sect. 4). Appendices support the findings with further details for reproducibility (Apps. A, E, and F). I appreciate the detailed discussion on limitations (l366-l377). The work is a valuable contribution in the realm of multi-modal representation learning for Earth observation data with focus on practical implementation when data providers need to retain privacy of their own data and data-specific embedding models.

### Weaknesses
The proposed methodology (Fig. 1) very closely resembles the generic setup of (Geospatial) Foundation Models where modality-specific tokenizers get utilized to perform correlation learning in a joint embedding space. Unfortunately, the authors do not reference recent trends in this field to place their work into such a perspective.

### Questions
What is the license the code will get published under?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript addresses practical challenges such as data heterogeneity and inter-modal collaboration difficulties by proposing collaborative unpaired multimodal learning for image classification.  The proposed method combines the modality-specific predictors and the shared backbone in a collaborative framework, enabling effective cross-modal knowledge transfer without requiring paired samples.  Extensive experimental results demonstrate the effectiveness and superiority of the proposed approach.

### Strengths
1. This manuscript considers a more practical multimodal learning scenario，i.e., unpaired multimodal data, and the proposed method can be effectively applied to various distributed frameworks.

2. This manuscript exhibits originality, as it effectively exploits the complementarity among multimodal data without the need to construct paired training samples.

3. The manuscript presents extensive experiments, and the comprehensive results demonstrate the effectiveness of the proposed method.

### Weaknesses
1. In Section 3.2, the authors mention “larger batch sizes $ B \gg b$” and “$( E_{\text{cal}} \ll E )$”, but the rationale behind these specific settings is no1. In Section 3.2, the authors mention “larger batch sizes $ B \gg b$” and “$( E_{\text{cal}} \ll E )$”, but the rationale behind these specific settings is not clearly explained.

2.  It is unclear whether Centralized Unpaired Multimodal Learning and Post-hoc Batch Normalization Calibration are two independent stages within the proposed framework.

3.  In Section 3.3, the authors state that “the common classification objective provides a supervisory signal that aligns the feature spaces without requiring explicit sample correspondences.”  While a supervisory signal can indeed align feature spaces, it seems more appropriate to describe this process as aligning class-level knowledge rather than merely aligning the feature spaces.

4.  The manuscript lacks a comprehensive investigation of the methods proposed in 2025, which should be supplemented.t clearly explained.

### Questions
Please refer to Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
