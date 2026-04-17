# Sigma: Semantically Informative Pre-training for Skeleton-based Sign Language Understanding

- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Pre-training has proven effective for learning transferable features in sign language understanding (SLU) tasks. Recently, skeleton-based methods have gained increasing attention because they can robustly handle variations in subjects and backgrounds without being affected by appearance or environmental factors. Current SLU methods continue to face three key limitations: 1) weak semantic grounding, as models often capture low-level motion patterns from skeletal data but struggle to relate them to linguistic meaning; 2) imbalance between local details and global context, with models either focusing too narrowly on fine-grained cues or overlooking them for broader context; and 3) inefficient cross-modal learning, as constructing semantically aligned representations across modalities remains difficult. To address these, we propose **Sigma**, a unified skeleton-based SLU framework featuring: 1) a sign-aware early fusion mechanism that facilitates deep interaction between visual and textual modalities, enriching visual features with linguistic context; 2) a hierarchical alignment learning strategy that jointly maximises agreements across different levels of paired features from different modalities, effectively capturing both fine-grained details and high-level semantic relationships; and 3) a unified pre-training framework that combines contrastive learning, text matching and language modelling to promote semantic consistency and generalisation. Sigma achieves new state-of-the-art results on isolated sign language recognition, continuous sign language recognition, and gloss-free sign language translation on multiple benchmarks spanning different sign and spoken languages, demonstrating the impact of semantically informative pre-training, and the effectiveness of skeletal data as a stand-alone solution for SLU. We will release the code upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors present a model and framework for sign language recognition that tackles three challenges in the current scientific literature in this domain. They ground skeletal data in textual contexts, they balance local and global context critical for sign language understanding, and they align different modalities. An early fusion method based on cross-attention allows the model to learn from visual encoding and text modalities, and alignment is learned to maximize agreement between these modalities.

Using as little as 10 pre-training and 10 fine-tuning epochs, the authors are able to achieve competitive results on WLASL2000 and CSL-Daily.

In summary, this paper presents a new method for pre-training and fine-tuning models for ISLR, CSLR, and SLT, showing the importance of cross-modal training.

### Strengths
The paper is clearly situated within a relevant body of work, and the necessary background knowledge is explained in sufficient detail.
The overall presentation of the paper is good: it is well structured and explains the relevant aspects. The tables and figures in the table support the main text well. The design choices made by the authors are clearly grounded and informed.

Content-wise, the paper tackles an important limitation of recent work in SLR and SLT, that is: that most models incorporate only visual cues and neglect the linguistic aspects of the task at hand. By including a model and alignment for global context and text, these authors advance the state-of-the-art in these domains.

### Weaknesses
The paper is slightly lacking in terms of how to apply this model. In particular, it is unclear to me how this model can be used for ISLR, CSLR, and SLT. See also: "Questions".

The paper introduces three new methods at the same time. This sometimes makes it difficult to follow. However, at the end, the impact of the individual contributions is measured and reported.

The paper could also benefit from expanding the related works section to include a wider array of papers. Currently, the scientific literature overview is somewhat limited. For example, more excellent work has been done on keypoint-based SLR and SLT that is not mentioned here.

### Questions
I would consider using SLPT instead of SLP as the abbreviation for Sign Language Pre-training, as SLP is often used for Sign Language Production.

Please check your citation commands, as references appear in the text as "Author (Year)" even when "(Author, Year)" would have been appropriate.

Can you elaborate on how your model is used at inference time? During fine-tuning, textual inputs are used. Can these be discarded for inference? If the SGT decoder is removed from the architecture, how are logits predicted for ISLR/CSLR and how are tokens generated for SLT?

### Soundness
3

### Presentation
4

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
The paper proposes Sigma, a unified, skeleton-based pre-training framework for Sign Language Understanding (SLU). It aims to solve three key limitations in existing methods: weak semantic grounding, where models fail to connect low-level motion patterns to linguistic meaning ; an imbalance between modeling fine-grained local details and high-level global context ; and inefficient cross-modal learning

### Strengths
1. Unified framework: this paper presents a single, unified framework that is fine-tuned to achieve state-of-the-art performance on all three major SLU tasks (ISLR, CSLR, and SLT), demonstrating its versatility and strong generalization capabilities.

2. Architectural Novelty: The proposed components (SignEF and HAL) are not arbitrary but are specifically designed to solve clearly articulated problems in SLU, namely the local-global imbalance and weak semantic grounding.

3. Strong results: Sigma achieves new state-of-the-art results on multiple diverse benchmarks, including WLASL2000, CSL-Daily, How2Sign, and OpenASL, surpassing previous methods in recognition accuracy, WER, and translation (BLEU/ROUGE) scores.

4. Efficiency: this work effectively utilizes skeleton-based data, which is robust to background and subject variations, requires significantly less storage, and is more computationally efficient than RGB-based methods, making it a more scalable solution.

### Weaknesses
1. The expression of Eq.2 is not clear. For example, what's the meaning of n and k here? Additionally, s and c are not defined here. This hinders the readers from understanding the meanings.
2. The intuition of dividing works into sub-word level units is strange. As stated in the paper, “curiosity” is split into“curios” and “ity,”. However, "curios" doesn't have any concrete meaning in English, and in sign language the actor doesn't express meanings in a sub-word level. The recombination of different sub-word results in misclarification of the meaning which is hardly understood by human beings, let alone aligning with visual features expressed by the actor. Thus, in the perspective of aligning texts with visual features, the alignment should be performed in the word level instead of the sub-word level. The effect of this design is also not ablated in the experiments.
3. The experssion of Algorithm 1 is not clear. In line 7, performing softmax operation will not diminish the dimension L. Besides, the dimension of performing summation in line 8 seems wrong. Besides, does the Cluster-wise sign-to-text similarity have a ground-truth for supervision? Or just enforce it to approximate 1.0?
4. In HAL, why omitting temperature scaling could be stated as fine-grained local alignment?
5. The design of SGT encoder is borrowed directly from previous works.
6. The effects of different components are not ablated in the experiments, e.g. the effects of HAL, early fusion and so on.

### Questions
See above. I will adjust my score according to the authors response.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The proposed Sigma framework offers a predictable combination of early fusion, hierarchical alignment, and multi-task pre-training—techniques already commonplace in multimodal learning. While it claims to address known SLU limitations (weak semantics, local-global imbalance, cross-modal gaps), the solutions lack novelty and largely rehash standard contrastive and alignment strategies.

### Strengths
The sole merit of Sigma lies in its marginal SOTA improvements across recognition and translation benchmarks. These gains—while consistent—are small and stem from routine integration of early fusion, hierarchical alignment, and multi-objective pre-training, not from conceptual innovation.

### Weaknesses
In a mature field, such incremental performance uplifts, absent methodological novelty or deeper insight, offer limited scientific value. The work is technically sound but contributes little beyond leaderboard chasing.

### Questions
(1) Skeleton inputs are de facto standard in SLU. Beyond re-emphasizing robustness to appearance, what specific insight or technical leap distinguishes Sigma from the dozens of prior skeleton-based models?

(2) The paper claims "sign-aware early fusion" resolves weak semantic grounding. This connection is not intuitive—fusing modalities earlier does not inherently bridge motion patterns to linguistic meaning. Provide concrete evidence (e.g., ablation on semantic alignment metrics) showing how this mechanism grounds low-level kinematics in high-level semantics.

(3) Are the proposed modules (early fusion, cluster-based hierarchical aggregator, etc.) plug-and-play improvements for any skeleton-based SLU backbone, or are the gains architecture-dependent? Without cross-model validation, claims of broad applicability remain unsubstantiated.

### Soundness
2

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
The paper presents Sigma, a unified pretraining framework for sign language understanding (SLU), designed to address the long-standing challenge of developing scalable and generalisable models that perform effectively across multiple SLU tasks. To this end, the authors introduce SignEF, HAL, and SGT, which collectively tackle the challenge and deliver impressive results across several benchmark evaluations.

### Strengths
The paper presents a unified pre-training framework for skeleton-based sign language understanding that integrates sign-aware early fusion (SignEF), hierarchical alignment learning (HAL), and multi-task pre-training within a single architecture. The design of SignEF is particularly novel, as previous studies in sign language research have seldom explored the potential benefits of cross-modal interaction, typically adopting architectures that fuse features only after unimodal extraction. In contrast, SignEF introduces a new and promising direction for future research. Moreover, the introduction of a local cluster-wise contrastive learning mechanism is also innovative. By employing a cluster aggregator to integrate features, this mechanism compresses feature representations and better aligns the sign and text spaces, leading to improved performance. Overall, this paper provides a creative perspective on integrating pose information with linguistic context, demonstrating that pose data alone can serve as a powerful foundation for SLU.

The paper is technically sound and supported by extensive empirical validation. A series of ablation studies thoroughly examine the effects of core components, benefiting from comprehensive experimental results and a simple architectural design. These ablations provide transparent justification for the architectural choices and enhance reproducibility. Moreover, the experiments cover multiple benchmarks and downstream tasks (ISLR, CSLR, SLT), demonstrating consistent performance gains across all settings.

### Weaknesses
1. Regarding the choice of the pre-trained language model, while the paper employs mT5 Base as the textual backbone, a relatively modest choice given the availability of larger mT5 variants, it would be beneficial to clarify the motivation behind this selection and whether Sigma can be readily extended to larger backbones to enhance reproducibility and future extensions.

2. It would be informative to report the additional computational and memory costs incurred by SignEF and HAL compared with the baseline.

3. The overall structure of the manuscript is well organized; however, certain bold and italicized expressions appear somewhat unconventional and could be refined to enhance the readability and stylistic consistency of the text. For instance, the references to SignEF and HAL in Lines 160–161, as well as WLASL2000 in the header of Table 18, may be appropriately revised.

4. Sigma is built exclusively around pose data, which the authors highlight as “clean and privacy-preserving”. However, this design choice completely excludes the rich RGB information, which could result in the loss of important visual cues. Could the authors further discuss when RGB information becomes particularly important?

5. It would be helpful if the authors could discuss the scalability and multilingual ability of Sigma.

### Questions
Please refer to the Weakness section above.

### Soundness
3

### Presentation
3

### Contribution
3
