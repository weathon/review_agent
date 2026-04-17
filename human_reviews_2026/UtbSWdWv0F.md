# CogMoE: Signal-Quality–Guided Multimodal MoE for Cognitive Load Prediction

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 0

## Abstract
The poor and variable quality of physiological signals fundamentally constrains reliable cognitive load (CL) prediction in real-world settings. In safety-critical tasks such as driving, degraded signal quality can severely compromise prediction accuracy, limiting the deployment of existing models outside controlled lab conditions. To address this challenge, we propose CogMoE, a signal-quality–guided Mixture-of-Experts (MoE) framework that dynamically adapts to heterogeneous and noisy inputs. CogMoE replaces conventional modality-based fusion with a quality-aware gating mechanism that integrates EEG, ECG, EDA, and gaze according to their estimated signal quality, shifting the basis of multimodal modeling from modality identity to signal quality. The framework operates in two stages: (1) quality-aware multimodal synchronization and recovery to mitigate artifacts, temporal misalignment, and missing data, and (2) signal-quality-specific expert modeling via a cross-modal MoE transformer that regulates information flow based on signal quality. To further improve stability, we introduce CORTEX Loss, which balances task accuracy, quality-aware representation refinement and expert utilization under noise. Experiments on CL-Drive and ADABase demonstrate that CogMoE outperforms strong baselines across all modality combinations and sequence lengths, consistently delivering improvements across diverse signal-quality conditions. Our code is publicly available at https://github.com/shahaamirbader/CogMoE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript presents a potentially valuable idea emphasizing signal quality during multimodal fusion but currently falls short in exposition, motivation, and experimental rigor, evaluation.

### Strengths
CoGMoE / CogMoE

### Weaknesses
- Manuscript readability and presentation require improvement; the method is not explained in sufficient technical detail to assess reproducibility or novelty.

- Terminology is inconsistent (e.g., CoGMoE vs CogMoE); a single canonical name and notation should be used throughout.

- The problem statement and its significance are not clearly articulated; it is difficult to judge the scope and impact of the contribution.

- Related work discussion is insufficient: prior techniques and their specific limitations are not described in depth, leaving motivation for a new method unclear.

- Evaluation is limited in scope, both in metrics and experiments, preventing strong conclusions about effectiveness and generalizability.

### Questions
- Justify the claim: “However, the fundamental bottleneck is not the lack of sensors or models, but the variable quality of physiological signals.” What empirical evidence or literature supports this assertion?

- Provide motivation: Why was the decision made to “shift the focus of multimodal fusion to signal quality, the true bottleneck in practical CL prediction”? How does this design choice address limitations of prior fusion methods?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the critical challenge of real-world cognitive load (CL) prediction, where physiological signal quality is often poor and variable. The authors propose **CogMoE**, a novel Mixture-of-Experts (MoE) framework that, unlike traditional models, dynamically routes data based on signal quality rather than modality. The two-stage framework first performs quality-aware synchronization and recovery of multimodal signals (EEG, ECG, etc.). Then, a cross-modal transformer routes representations to one of three experts specialized for different quality regimes: a High Fidelity Expert (HFE) for clean signals, a Noise Resilient Expert (NRE) for noisy ones, and a Contextual Refinement Expert (CRE) for recovered data . The model is trained with a novel **CORTEX Loss**, which jointly optimizes for task accuracy, balanced expert utilization, and quality-aware objectives. Experiments on the CL-Drive and ADABase datasets show state-of-the-art performance, with accuracy gains of up to 13% and 9.5%, respectively.

### Strengths
1. Novelty of Core Concept: The paper's central idea of a "signal-quality-guided" MoE is highly novel and directly targets the primary bottleneck of real-world physiological sensing.
2. Purpose-Built, Validated Experts: The three experts (HFE, NRE, CRE) are thoughtfully designed for specific quality regimes (clean, noisy, recovered) . Ablation studies (Table 8) rigorously prove that all three are necessary for robust performance.
3. Specialized Loss Function: The CORTEX Loss is a key strength, intelligently designed with auxiliary losses ($\mathcal{L}_{noise}$, $\mathcal{L}_{refinement}$) to enforce expert specialization and a regularizer ($\mathcal{R}_{gate}$) to prevent expert collapse, which is empirically validated (Table 9).

### Weaknesses
1. Supervision Dependency: The framework relies on supervised labels, which are expensive and difficult to obtain at scale for physiological data, limiting practical scalability.
2. Limited Domain Evaluation: While the framework is general , the experiments are confined to two driving datasets (CL-Drive and ADABase). Its effectiveness in other noisy-sensor domains (e.g., healthcare) is not empirically demonstrated.

### Questions
1. CORTEX Loss Sensitivity: The CORTEX Loss introduces several new hyperparameters (e.g., $\gamma, \lambda$, and the $\beta$ schedule). How sensitive is the model's performance and, critically, its expert balance, to the precise tuning of these values? Given that removing the gating regularizer entirely causes catastrophic collapse (Table 9), how difficult is it to find a stable $\beta$ schedule?
2. Real-Time Feasibility: Table 5 lists an inference time of 114ms. Does this end-to-end time include the "Quality-Aware Synchronization and Recovery" stage? Operations like CWT-based alignment and nuclear norm minimization for completion  can be computationally expensive. Is the entire pipeline fast enough for real-time deployment in a safety-critical context?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes CogMoE, a multimodal framework for cognitive load prediction that is explicitly guided by signal quality rather than modality identity. The framework contains a pre-processing stage for feature extraction, time frequency synchronization, multi-modal data recovery. followed by a signal-quality-specific MoE modeling stage. The proposed framework shows improved performance on several benchmarks.

### Strengths
The paper is clearly written: the motivation is well articulated, and the model design follows directly from it. Shifting MoE routing from modality identity to measured reliability is an interesting idea and a clever fit for physiological data. The experimental coverage is solid, spanning EEG, ECG, EDA, and gaze.

### Weaknesses
1. While the signal-quality–specific MoE is an interesting and domain-appropriate idea—especially given the authors’ argument that aligned EEG, ECG, EDA, and gaze often reflect overlapping aspects of the same cognitive state—a direct, controlled comparison against alternative routing rules (e.g., modality-based experts) is needed to substantiate the claimed benefit.

2.The final loss comprises multiple components and may be brittle without careful balancing. How sensitive is the model to weight coefficient lambda, gamma, beta, and how many total combination of these are searched?

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper proposes CogMoE and outlines 4 main contributions: 1) Quality-Aware Multimodal Synchronization and Recovery module, 2) three signal-quality-specific expert modeling, 3) CORTEX Loss, and 4) experiments (though this is generally not considered as a contribution of novelty of the work). In experiments on two public datasets, the authors show consistent gains over baselines.

### Strengths
The paper studies handling, evaluating signal quality, and interpolating missing signal using other modalities, which is where many otherwise strong models fail in the wild. Reporting results across multiple modality combinations is useful for practitioners who must decide which sensors to deploy, and the paper makes an effort to tie modeling choices to operational constraints (noisy segments, dropouts, cross-sensor lag). Conceptually, using expert pathways specialized for “clean” vs “noisy/reconstructed” inputs is an interesting way to avoid generic fusion algorithms and may be broadly applicable beyond cognitive-load prediction.

### Weaknesses
The paper reads rushed, and the contribution is not crisp. Several choices are taken as givens without ablation (the ablation section is one paragraph). For example, why CWT for representing a signal? Given that CWT is a sifting process, the high-frequency data will be lost, and the data is normalized in a sense. The method CMWT representation for alignment/recovery is introduced in the work, but does not compare to reasonable alternatives (e.g., STFT features, time-domain DTW variants, learned alignment modules). The custom loss also appears as a package rather than with per-term deltas, which makes it hard to attribute gains or evaluate the necessity of each element. On evaluation, the paper discusses “alignment” and “robust fusion” but does not analyze how the number of signal sources and their measured quality jointly affect accuracy; instead, it relies on aggregate tables. 

Minor: Acronyms are introduced inconsistently (sometimes before definition, sometimes multiple times). 

The work likely contains a solid idea, but it needs clearer writing, a sharper statement of what is novel, and targeted ablations to make the methodology and its advantages unambiguous.

### Questions
No further questions at this stage.

### Soundness
2

### Presentation
2

### Contribution
2
