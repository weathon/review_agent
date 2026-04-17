# SiMO: Single-Modality-Operable Multimodal Collaborative Perception

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Collaborative perception integrates multi-agent perspectives to enhance the sensing range and overcome occlusion issues. While existing multimodal approaches leverage complementary sensors to improve performance, they are highly prone to failure—especially when a key sensor like LiDAR is unavailable. The root cause is that feature fusion leads to semantic mismatches between single-modality features and the downstream modules. This paper addresses this challenge for the first time in the field of collaborative perception, introducing **Si**ngle-**M**odality-**O**perable Multimodal Collaborative Perception (**SiMO**). By adopting the proposed **L**ength-**A**daptive **M**ulti-**M**od**a**l Fusion (**LAMMA**), SiMO can adaptively handle remaining modal features during modal failures while maintaining consistency of the semantic space. Additionally, leveraging the innovative "Pretrain-Align-Fuse-RD" training strategy, SiMO addresses the issue of modality competition—generally overlooked by existing methods—ensuring the independence of each individual modality branch. Experiments demonstrate that SiMO effectively aligns multimodal features while simultaneously preserving modality-specific features, enabling it to maintain optimal performance across all individual modalities. The implementation details can be found in [https://github.com/dempsey-wen/SiMO](https://github.com/dempsey-wen/SiMO).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SiMO (Single-Modality-Operable Multimodal Collaborative Perception), a novel framework designed to enhance the robustness and reliability of multi-agent collaborative perception (MACP) systems against sensor failures. The central problem SiMO addresses is that existing multimodal fusion methods in MACP are fragile, similar to a "series circuit,” failing completely when a critical sensor like LiDAR is unavailable.SiMO's solution functions like a "parallel circuit," allowing the system to operate with any single, effective modality. It achieves this through two main innovations: 1) Length-Adaptive Multi-Modal Fusion (LAMMA): This plug-and-play fusion module first aligns features from different modalities (LiDAR and Camera) and then integrates them using additive fusion to maintain a consistent feature space before and after fusion. 2) Pretrain-Align-Fuse-RD (PAFR) Training Strategy: This strategy addresses modality competition to ensure that each individual sensor branch is comprehensively and independently trained, thus strengthening the performance of the single-modality operations.

### Strengths
1. High Robustness Against Modal Failures: SiMO explicitly address the complex challenge of dynamic, heterogeneous modal failures in Multi-Agent Collaborative Perception, which is important for real-world safety-critical deployment of V2V.
2. The core idea of aligning and fusing features using addition to ensure semantic consistency between pre- and post-fusion feature spaces is straight-forward and effective.
3. Proposed LENGTH-ADAPTIVE MULTIMODAL FUSION (LAMMA) can handle varying input modalities. With RD training, the performance of camera-only collaboration gets better, indicating an alleviation in modality competition.

### Weaknesses
1. This framework necessitates multiple stage training, which increases the training complexity.
2. Late fusion metrics are not reported.
3. The results on DAIR-V2X is not persuasive. While authors attribute the poor performance to the single-camera sensor and out-of-date camera-based detector, it would be encouraging to implement a more modern camera detector to validate the SiMO's effectiveness.

### Questions
no questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work presents the single-modality-operable multimodal collaborative perception framework, which is the first attempt to address modality failure in the context of multi-agent collaborative perception. The proposed SiMO employs Length-Adaptive Multi-Modal Fusion to effectively leverage the remaining modal features during modality failures while preserving semantic consistency across modalities. To reduce modality bias in modality competition, SiMO further proposes a “Pretrain–Align–Fuse–RD” training paradigm. The paper provides thorough analyses and extensive experiments, validating the effectiveness of the proposed approach.

### Strengths
1. This work is the first to address the problem of modality failure in the context of multi-agent collaborative perception, and it proposes an effective and practical solution through the SiMO framework.
2. The paper is supported by extensive experiments, complemented by insightful visualizations and in-depth data analyses.
3. The manuscript is clearly written and logically structured and the technical content is easy to follow.

### Weaknesses
1. The paper does not clearly explain the unique challenges of modality failure in the multi-agent collaborative perception (MACP) setting compared to the single-agent case. Similar modality failures could also appear in single-agent multi-modality scenarios. The authors should clearly explain why existing single-agent modality-robust methods are inadequate in the MACP context, and ideally include comparative experiments to substantiate this claim.

2. The training procedure is complex. It requires four separate end-to-end training stages, resulting in a large training overhead. The authors should provide a detailed analysis of its training robustness and efficiency.

3. The description of “heterogeneous model failure” in Table 2 is unclear, making it difficult to understand what modalities the ego vehicle and neighboring agents respectively possess in this setting.

### Questions
1. In the training process, Step 2 is sequential, aligning LiDAR first and then camera. This actually favors camera performance. My question is : when more modalities need to be aligned, how would this ordering be determined?

2. What is the exact input and what are the parameters used for the t-SNE visualization? Please clarify the implementation details.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes SiMO (Single-Modality-Operable Multimodal Collaborative Perception), a multimodal collaborative-perception framework that (1) aligns modality BEV features into a unified semantic space, (2) fuses them with a Length-Adaptive Multi-Modal Fusion (LAMMA) module that is intended to gracefully degrade to self-attention when a modality is missing, and (3) applies a Pretrain–Align–Fuse–RD (PAFR) training schedule to avoid modality competition (pretraining branches, training aligners, freeze branches, then fuse + random-drop fine-tuning). Experiments on OPV2V-H and V2XSet (plus DAIR-V2X discussion) show that SiMO maintains reasonable detection performance with LiDAR failure / camera-only operation while prior multimodal fusion baselines collapse.

### Strengths
The paper is, to the best of my knowledge, the first work in collaborative perception that explicitly tackles multimodal perception failure caused by missing modalities, with a particular focus on ensuring operability when only RGB inputs are available.

It identifies the inconsistency between pre-fusion and post-fusion features as the main reason for performance collapse during modality failure, and introduces LAMMA, a length-adaptive fusion module designed to maintain semantic consistency under varying modality combinations.

Through extensive experiments on multiple V2X collaborative datasets, the paper shows that each single-modality branch can function independently while maintaining competitive multi-modal performance, indicating that SiMO achieves strong robustness and communication efficiency under degraded sensory conditions.

### Weaknesses
The real-world validation is limited, and the camera results are inconsistent. On DAIR-V2X, the camera-only performance remains poor (the authors attribute this to the limitations of single-view LSS). This weakens SiMO’s claim of addressing single-modality operability in practice. The paper should (a) include stronger single-view camera baselines, or (b) tone down the claims regarding the single-view camera setting.

Comparisons with stronger or more recent modality-failure methods adapted to multi-agent settings (e.g., MetaBEV/UniBEV, contrastive alignment approaches, or simply stronger camera-only backbones) are recommended.

The scalability to more than two modalities and a large number of agents is unclear. How does the method perform with three or more modalities? Since the complexity varies with the number of connected queries, this should be discussed and, if possible, demonstrated experimentally.

Pose errors and asynchronous agents are not studied, yet collaborative perception is known to be sensitive to pose noise and temporal misalignment. The robustness to pose calibration errors or time delays should be evaluated or at least discussed (as prior works such as HEAL have shown this sensitivity).

The evaluation conducted solely on simulated datasets is not sufficiently convincing.

### Questions
In Figure 8, BM2CP’s feature visualizations are shown — why not visualize the features of your own method?

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
This paper proposes SiMO to address a critical yet underexplored challenge in multimodal collaborative perception: system failure under partial modality loss. SiMO introduces two key components: i) Length-Adaptive Multi-Modal Fusion: a fusion module that handles variable numbers of input modalities via attention mechanisms and maintains semantic consistency before and after fusion through additive combination. ii) Pretrain-Align-Fuse-RD training strategy: a staged training protocol that mitigates modality competition by pretraining and aligning modality-specific branches independently before joint fusion, followed by fine-tuning with random modality dropout (RD). Experiments show that SiMO achieves state-of-the-art performance in full multimodal settings while maintaining strong single-modality performance.

### Strengths
-	The paper tackles robustness to sensor failure in collaborative perception, which is a realistic and safety-critical issue that is rarely explored in existing multimodal collaborative perception literature.
-	The method is plug-and-play and demonstrated on two backbone frameworks (AttFusion and Pyramid Fusion), suggesting broad applicability.
-	Strong empirical validation: the paper presents comprehensive experiments across homogeneous/heterogeneous modality failures.

### Weaknesses
-	While RD improves robustness as a data augmentation strategy, the 0.5 dropout probability is not justified theoretically or empirically. A sensitivity analysis would strengthen this design choice.
-	While LAMMA fusion is designed to handle missing modalities, how does it differ from simpler alternatives—such as replacing the missing modality’s feature with a zero tensor in existing fusion schemes?

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
