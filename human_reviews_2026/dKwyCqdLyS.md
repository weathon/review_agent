# MIRACL: A Robust Framework for Multi-Label Learning on Noisy Multimodal Electronic Health Records

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
Multimodal Electronic Health Records (EHRs), comprising structured time-series data and unstructured clinical notes, offer complementary views of patient health. However, multi-label prediction tasks on multimodal EHR data, such as phenotyping, are hindered by potential label noise, including false positives and negatives. Existing noisy-label learning methods, often designed for single-label vision data, fail to capture real label-dependencies or account for the cross-modal, longitudinal nature of EHRs.
To address this, we propose MIRACL (\textbf{M}ultimodal \textbf{I}nstance \textbf{R}elabelling \textbf{A}nd \textbf{C}orrection for multi-\textbf{L}abel noise (MIRACL\footnote{\url{https://github.com/anon-coder-def/MIRACL}})), a novel framework that systematically addresses these challenges.  Notably, MIRACL is the first framework designed to explicitly leverage longitudinal patient context to resolve more challenging multi-label noise scenarios. To achieve this, MIRACL unifies three synergistic mechanisms:
(1) a difficulty- and rank-based metric for robust identification of noisy instance-label pairs, 
(2) a class-aware correction module for robust label refinements, promoting the recovery of real label-dependencies, and
(3) a patient-level contrastive regularization loss that leverages both cross-modal and longitudinal patient context to correct for noisy supervision across different visits.
Extensive experiments on large-scale multimodal EHR datasets (MIMIC-III/IV) demonstrate that MIRACL achieves state-of-the-art robustness, improving test mAP by over 2\% under various noise levels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MIRACL, a framework designed to address label noise in Electronic Health Records (EHRs) by combining difficulty-based noise detection, label correction, and patient-level contrastive learning. The model incorporates both structured (e.g., time-series medical records) and unstructured (e.g., clinical notes) data, with a focus on multi-label learning. Experiments on the MIMIC-III and MIMIC-IV datasets show that MIRACL outperforms state-of-the-art methods.

### Strengths
- The framework combines difficulty-based noise detection, label correction, and patient-level contrastive learning, which makes it a strong contender for noisy label problems in healthcare.
- By handling both structured and unstructured data, MIRACL significantly enhances the utility of EHRs, which are inherently multimodal in nature.

### Weaknesses
- The simplified label noise types (e.g., random label flipping) used in the experiments may not capture the dependencies or temporal patterns of noise typically observed in healthcare data. I believe the authors should redesign the way noise is introduced into the data.

- There is a lack of related baselines, such as ACTLL [1], STW [2], SREA [3], and SIGUA [4], for comparison with MIRACL on the MIMIC-III and MIMIC-IV datasets.

- The paper lacks more benchmark datasets, such as the eICU Collaborative Research Database [5].

- The figures in the paper need to be polished. There are some issues with the lines and arrows.

[1] Dynamical Label Augmentation and Calibration for Noisy Electronic Health Records.

[2] CTW: Confident Time-Warping for Time-Series Label-Noise Learning.

[3] Estimating the electrical power output of industrial devices with end-to-end time-series classification in the presence of label noise.

[4] Sigua: Forgetting may make learning with noisy labels more robust.

[5] The eICU Collaborative Research Database, a freely available multi-center database for critical care research

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes MIRACL, a framework designed to address multi-label noise in multimodal Electronic Health Records (EHRs). It proposed to utilize three modules to mitigate label noise: a multi-faceted selection metric, a class-aware correction module, and a patient-level contrastive regularization loss. Experimental results on MIMIC-III/IV datasets show that MIRACL achieves state-of-the-art robustness under various noise settings, demonstrating its effectiveness, particularly in high-noise regimes.

### Strengths
1. **Important and Practical Topic**: The multi-label noise in healthcare is important, and this topic has the potential to make an impact in the real world.
2. **Comprehensive Framework for Multi-Label Noise**: MIRACL is presented as the comprehensive framework unifies three synergistic mechanisms to achieve robustness: class-wise sample selection, label correction, robust contrastive learning.
3. **State-of-the-Art Performance in Phenotyping**: Extensive experiments on MIMIC-III and MIMIC-IV Phenotyping (PHE) datasets demonstrate that MIRACL achieves state-of-the-art robustness

### Weaknesses
1. **Limited Novelty of The Proposed Solution**: It is more like a mixture of many existing methods for sample selection, label correction, and contrastive learning. 
2.  **Limited Effectiveness of Selection Metric**: As shown in Table 4, the Correction w/ Loss only, w/ Mems only, w/ Rank only do not perform significantly worse than MIRACL.
3. **Not Good Presentation**: 
(1) For example, visits and patients are not defined in problem formulation. In Line 193, $\Delta^{(t)}_f$ is not defined. 
(2) In Line 204, why can instance-level prediction dynamics be used to capturing inter-label dependencies? 
(3) In Line 215, what is 2 x L? 
(4) In Line 255,  $\mathbb{E}$ usually refers to the expectation and can be easily misunderstanded here. 
(5) In Line 268, how to compared the correlation matrix with a threshold?

### Questions
In Line 433, the statement that the performance curve shows a clear upward trend as $λ_{cons}$ increases from 0.0 to 0.05 lacks evidence.

### Soundness
3

### Presentation
1

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
This paper tackles multi-label noise in multimodal EHR data combining structured time-series and unstructured notes. The proposed framework, MIRACL, unifies three components: (1) difficulty- and rank-based detection of noisy labels, (2) class-aware label correction, and (3) patient-level contrastive regularization leveraging longitudinal context. Experiments on MIMIC-III/IV show consistent gains under various noise settings.

### Strengths
- The paper addresses an important and realistic problem of noisy labels in multimodal, longitudinal EHRs.

- The framework is well-structured and empirically validated under multiple corruption types and noise levels.

- The clinical multimodal scenario is meaningful and underexplored.

### Weaknesses
- The claim of being “the first to leverage longitudinal context” appears overstated given prior work on temporal or contextual consistency.

- The added difficulty of multimodality vs. single-modality noise learning is not clearly explained.

- The literature review and motivation are weak; for example, why existing methods (e.g., BalanceMix) are not applicable is not convincingly argued.

- The three modules borrow heavily from prior methods, lacking clear methodological novelty.

- The baseline comparison is insufficient, omitting recent multimodal fusion and noisy-label learning approaches.

### Questions
1. What unique challenges does multimodal label noise introduce beyond the single-modality case?

2. How is longitudinal context quantitatively used in the contrastive module?

3. Minor typo: Figure 2: “Atrraction” -> “Attraction.”

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces MIRACL, a multimodal EHR framework for handling multi-label noise, integrating three complementary components: a sample selection strategy based on difficulty and ranking metrics combined with GMM; a label correction mechanism designed for different sample categories (clean/uncertain/noisy); and patient-level contrastive regularization utilizing patient longitudinal context. This framework addresses multiple noise issues and achieves consistent performance improvements on the MIMIC-III/IV datasets.

### Strengths
I find the most interesting merit of this paper lies in its design for forgetting and remembering multi-labels, particularly how MIRACL tracks the learning and forgetting behavior of each multi-label during every training epoch. Additionally, the ranking design aids in understanding label credibility.

### Weaknesses
The overall writing structure of the paper lacks clarity. I have several comments:

1. I find that the paper's claimed longitudinal design has fundamental issues. For example, it pulls all visits of the same patient closer together but does not model temporal ordering. True longitudinal modeling should involve sequential trajectories: $\text{Visit}_1 \rightarrow \text{Visit}_2 \rightarrow \text{Visit}_3$.

2. There is a logical inconsistency in the paper. The premise states that MIMIC data may contain noise (motivation), yet the experimental setup assumes MIMIC is clean, and the conclusion claims that the method can handle noise. This reasoning is not self-consistent.

3. Consider the following situation. For instance, take Patient A: Visit~1 in~2020 shows Diabetes= 0 (healthy), whereas Visit~2 in~2023 shows Diabetes=1 (newly diagnosed). However, MIRACL's approach of forcing $h_{\text{visit1}} \approx h_{\text{visit2}}$ seems unreasonable.

4. Although the memorization and forgetting design in Section 3.3.1 is interesting, how do you ensure this design is effective or accurate for multi-label classification across patients with varying numbers of visits? Especially for patients with only 1 or 2 visits.

5. The positive sample definition is too coarse. It does not distinguish between visit time intervals---1 day later vs. 3 years later. It does not consider disease state changes---for example, should visits for acute conditions (such as pneumonia) be connected? It does not consider label consistency---what if two visits have completely different labels?

6. The synergy with label correction is unclear. How do contrastive learning and label correction mutually reinforce each other? The paper does not elaborate on this in depth.

7. The notation is inconsistent, with $Y$, $\tilde{Y}$, $\hat{Y}$, and $Y_{\text{corr}}$ used interchangeably.

8. According to the paper's experimental setup, are all different visits of the same patient in the training set? If so, new visits in the test set could leverage historical information from training, which appears to be data leakage. What about temporal splitting? The data should be split temporally (e.g., training on pre-2020 data and testing on post-2021 data).

### Questions
1. Is GMM fitting sensitive to initialization?
2. Does the bimodal assumption still hold for extremely imbalanced datasets (e.g., certain rare disease labels)?
3. How does patient-level contrastive learning operate when handling new patients with only a single visit record?

### Soundness
2

### Presentation
2

### Contribution
2
