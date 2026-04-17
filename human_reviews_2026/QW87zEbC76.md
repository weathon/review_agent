# Toward Stable Brain-Computer Interfaces: Revealing and Addressing Prediction Fluctuations in EEG-Based BCIs

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Brain-Computer Interfaces (BCIs) are increasingly used in areas such as neurofeedback and mental healthcare, where reliable real-time feedback is essential. While deep learning (DL) has greatly improved Electroencephalography (EEG)-based BCIs by boosting accuracy in tasks like emotion recognition, attention detection, and workload assessment, current models often suffer from \textit{temporal instability}. Predictions fluctuate erratically across consecutive windows, contradicting the slow-changing nature of cognitive states and producing inconsistent feedback that undermines user engagement. Existing metrics and post-processing methods fail to capture or resolve this issue effectively. We address this gap through three contributions: (1) a systematic study of prediction fluctuations across datasets, tasks, and representative models; (2) two new stability metrics, Frequency-weighted Spectral Entropy (FSE) and First-Order Difference Standard Deviation (FDS), that directly measure temporal irregularities; and (3) TRin (Temporal Robustness integrated BCI), a fluctuation-aware training framework combining stability-driven losses with curriculum learning. Experiments on three public datasets show that TRin consistently reduces fluctuations while improving accuracy. By introducing stability as a core evaluation dimension, this work provide a new way for more robust and effective real-time BCIs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study focuses on the temporal instability of predictions in EEG-based BCIs, where consecutive outputs fluctuate despite high accuracy. It systematically characterizes prediction instability across datasets and models, proposes two new temporal-stability metrics (FDS and FSE), and introduces a new training framework (TRin) that integrates temporal regularization with a dynamic training strategy. Experimental results show that TRin effectively reduces prediction fluctuations and improves classification performance across multiple EEG datasets and architectures.

### Strengths
1. The paper squarely targets temporal instability of EEG-BCI predictions—an issue that is rarely foregrounded yet highly consequential for online, closed-loop use. By reframing “prediction stability” as a first-class objective (rather than a post-hoc smoothing concern), the work opens a new axis for evaluating and improving BCI systems.

2. The authors introduce interpretable, quantitative stability metrics that disentangle accuracy from temporal jitter, and they propose a principled training scheme to directly improve stability at the model level. 

3. The objectives, metrics, and loss terms are clearly specified, with intuitive explanations and well-chosen ablations. A rigorous setup—cross-subject evaluation, consistent window/stride, multiple architectures, and statistical testing—makes the reported gains credible.

### Weaknesses
**High overlap setting.** The temporal overlap is very large (~95% overlap), which induces strong sample correlation, magnifies the apparent advantage of any smoothing-leaning method, and can overstate statistical significance. Part of the reported gains may therefore stem from resampling/overlap rather than the proposed objective.

**Narrow base model coverage.** 
The current baselines skew toward classic or mid-capacity models, leaving open whether improvements hold with stronger backbones and standard temporal-consistency approaches. To strengthen the evidence, integrate the regularizer into more advanced architectures—for example, EEG foundation models—and report results under a matched latency budget.

### Questions
1. Real-time BCIs need both stable outputs and quick response. Temporal regularization or heavy overlap can add delay and blur change points. This may improve FSE/FDS on slow-changing labels but cause the model to lag or suppress too much at sudden changes, hurting control in closed-loop use. Could you include a step-change experiment to quantify responsiveness under abrupt changes?

2. Without comparisons to standard consistency and simple smoothing methods, it’s hard to tell whether the gains come from your method or from generic smoothing. Could you provide matched-latency, equal-tuning comparisons against standard consistency/smoothing methods (e.g., Temporal Ensembling).

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
4

### Summary
This paper proposes TRin (Temporal Regularization integrated network), a training framework for EEG-based BCIs that aims to improve the temporal stability of decoding outputs. TRin combines a temporal regularization loss and a dynamic curriculum to enforce smooth and consistent predictions over time.
New stability metrics (FSE and FDS) are introduced to quantify prediction consistency, and experiments on multiple EEG datasets show improved decoding accuracy and stability over baselines.

### Strengths
- This paper is well-written and intuitively illustrated.
- Experiments across multiple datasets show consistent gains in both accuracy and stability, highlighting the practical potential of the proposed framework.
- The introduction of the FSE and FDS metrics provides a quantitative extension to prior evaluation practices, enabling more systematic analysis of prediction stability in EEG decoding.

### Weaknesses
- The ablation study lacks key variants such as temporal loss only or TRin without curriculum, making it unclear which component drives the observed improvement. Moreover, the number of evaluated models is limited, reducing confidence in the generality of the results. Clarification through fine-grained ablation or sensitivity analysis is needed.
- The datasets used in this study have trial-level constant labels that do not capture within-trial dynamics. As a result, variations in model outputs could reflect unlabeled but meaningful state transitions, rather than noise. This mismatch between label granularity and the definition of “stability” weakens the conceptual foundation of the proposed framework.
- Figure 1 illustrates how unstable predictions could lead to inconsistent feedback, but this remains only a conceptual example without empirical validation. No experiments evaluate TRin’s impact on real-time inference latency, adaptation speed, or feedback consistency.

### Questions
- Q1. The paper claims that prediction fluctuation undermines user engagement and neurofeedback performance, but provides no supporting evidence or references. Could the authors cite prior work or empirical findings that demonstrate this effect?
- Q2. Could “smooth-but-wrong” predictions be over-rewarded under constant-label conditions, since the stability metrics penalize fluctuation rather than correctness?
- Q3. Were the results and visualizations obtained from all subjects or only selected examples? If only a subset was shown, how consistent are these patterns across subjects?

### Soundness
2

### Presentation
3

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
This paper identifies prediction fluctuations in deep learning-based EEG brain-computer interfaces as a critical problem that undermines real-time feedback and user engagement. To address the temporal instability, work proposes metrics: Frequency-weighted Spectral Entropy (FSE) and First-Order Difference Standard Deviation (FDS) as quantitative metrics beyond standard accuracy measures. It further proposes a temporal robustness integrated training framework (TRin) combining a temporal regularisation loss with a dynamic training strategy to jointly improve classification accuracy and reduce erratic temporal fluctuations. Empirical validation across 3 paradigms and datasets with different families of model architectures demonstrates consistent performance gains.

### Strengths
The paper addresses a fundamental and underexplored aspect of temporal stability, crucial for real-time neurofeedback and clinically relevant applications.

The novel metrics FSE and FDS provide meaningful evaluation of prediction smoothness and instability, overcoming limitations of conventional metrics that degenerate under constant trial labels.

The temporal regularisation loss is theoretically grounded, representing a Gaussian random walk prior, which aligns with the slow and continuous cognitive dynamics suggested by the neuroscience literature.

The dynamic clip-based data shuffling and hyperparameter scheduling effectively preserve temporal dependencies while maintaining generalisation capacity.

Empirical results show consistent accuracy improvements alongside a significant reduction in prediction fluctuations, supporting the utility of the approach.

### Weaknesses
Although the temporal regularization is motivated by slow cognitive state changes, the paper does not provide direct neurophysiological validation that enforced stability corresponds to true neural dynamics rather than smoothed noise or artifacts. This gap raises questions about the interpretability of improvements.

The high overlap(95%) between input windows means that consecutive model inputs share most of the data, potentially inflating stability metrics and masking true rapid neural transitions. This challenges whether the regularisation improves actual neural state modelling or merely smooths highly correlated inputs.

Testing with lower overlaps or variable window lengths would clarify how much of the temporal stability gain is due to the loss function versus inherent redundancy.

The risk of suppressing meaningful transient neuromarkers by over-regularizing predictions is noted but not deeply examined or quantitatively analyzed.

### Questions
The following are the questions or recommendations that would help me better interpret the work:

Can the authors provide or integrate direct neurophysiological validation linking temporal stability metrics to known neural state dynamics or markers?

Have experiments been conducted with lower or variable sliding window overlaps to disentangle the effects of input overlap from the temporal regularization itself?

What are the trade-offs between temporal regularisation during training and simpler post-hoc label smoothing in terms of accuracy, stability, latency, and computational complexity? Benchmarking would help understand and quantify the contributions.


How does the model’s temporal stability behave during abrupt neural or cognitive state transitions that may challenge the Gaussian random walk prior assumption? Are there paradigms where this could be verified, such as motor imagery, where the neuromarkers are often ERD/ERS?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to address the gap in the BCI field, which is that all metrics focus on classification outcomes but not on the stability of classifications temporally. Authors mention 3 contributions: 1) a systematic study, 2) two new stability metrics, and 3) the TRin framework. Experiments are reported on three public datasets. The paper is clear and readable with sufficient motivation for the gap in the BCI field.

### Strengths
The paper provides sufficient motivation and is well-explained. It is also well written (except for minor issues with the style of introducing acronyms and the clarity of Figure 2), with explicit definitions for the stability metrics. The experimental setup covers multiple datasets and model families and helps interpret how temporal regularization leads to improved model performances.

### Weaknesses
The core ideas of (a) penalizing drastic (unexpected) output probability changes, (b) curriculum over time sequence, and (c) stability metrics are reasonable but not clearly distinguished from known practices (e.g., temporal smoothness penalties on logits, clip training, frequency-domain regularization). The paper would benefit from a crisper novelty outline: what is new versus standard temporal smoothing, what prior “temporal-consistency” losses do not cover.
The main comparative analysis to TRin is MSE loss, even though the task is classification, and the paper’s own baseline uses cross-entropy. MSE on logits is an unstable objective. The more relevant baselines are (i) CE + output smoothing, (ii) CE + temporal smoothing penalty on logits, and (iii) label-smoothing under CE. In particular, CE + a simple moving average of posteriors is a strong, practical baseline for real-time systems; Fig. 4’s qualitative results suggest that such post-processing might close much of the perceived gap without changing training. 
The paper’s own metrics (FDS/FSE) emphasize continuity. They will reward any approach that smooths predictions. That’s fine, but to demonstrate broader utility, I would also expect analyses that: (i) hold latency constant when comparing against moving-average post-processing, (ii) evaluate under abrupt ground-truth changes, and (iii) report false-transition lag. The current results lean toward “smoother is better” without stress-testing responsiveness.

### Questions
Please map TRin’s pieces (loss form, curriculum learning, metrics) to prior temporal-consistency/smoothing literature, and clarify what is new in formulation or guarantees beyond standard smoothness penalties.

Can you share why MSE is used as a comparison to TRin and not CE? Annecdotally, how would the results change, and how would you compare your framework's performance with other metrics beyond "smoother is better"?

### Soundness
3

### Presentation
3

### Contribution
2
