# Entropy Never Lies: Signed Entropy Integral Unmasks Mislabeled Data

- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Mislabeled samples in training datasets severely degrade the performance of deep networks, as overparameterized models tend to memorize erroneous labels. We address this challenge by proposing a novel approach for mislabeled data detection that leverages training dynamics. Our method is grounded in the key observation that correctly labeled samples exhibit consistent entropy decrease during training, while mislabeled samples maintain relatively high entropy throughout the training process. Building on this insight, we introduce a signed entropy integral (SEI) statistic that captures both the magnitude and temporal trend of prediction entropy across training epochs. SEI is broadly applicable to classification networks and demonstrates particular effectiveness when integrated with contrastive language-image pretraining (CLIP) architectures. Through extensive experiments on three medical imaging datasets---a domain particularly susceptible to labeling errors due to diagnostic complexity---spanning diverse modalities and pathologies, we demonstrate that SEI achieves state-of-the-art performance in mislabeled data identification, outperforming existing methods while maintaining computational efficiency and implementation simplicity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper addresses the problem of distinguishing mislabelled (noisy) samples from valuable clean samples that are difficult to classify. Motivated by medical data, it demonstrates that both mislabelled and clean samples exhibit high prediction entropy during training, which makes them difficult to distinguish. However, it is also observed that they have different training dynamics. To make use of this information, the paper proposes defining a 'signed entropy' statistic for self-supervision by introducing a sign function on top of the Shannon entropy. Additionally, the signed entropy integral is introduced to leverage the training dynamics across the entire training trajectory (rather than just one iteration). Finally, a data-driven threshold is used to separate clean samples from mislabelled ones. This approach was evaluated using three noisy label learning medical datasets and showed promising results.

### Strengths
The idea is simple yet effective. It demonstrates its effectiveness with noisy and difficult-to-clean labels. 

The paper is very well written and easy to follow. The proposed method is intuitively clear and well motivated. 

The ablation studies effectively support the claim that the proposed sign is necessary for the entropy function. 

In theory, the proposed method could be applied to any type of data, not just medical data.

### Weaknesses
The paper is based only on medical datasets. There are standard benchmarks for noisy label learning to demonstrate generalisation, such as Clothing1M and WebVision, as well as small-scale datasets. 

The paper is based purely on synthetic noisy scenarios. This is another clear limitation that needs to be addressed using real-world noise. 

It would be helpful to make comparisons with entropy-based loss variations or temporal entropy integration. There are many methods for modifying standard entropy loss. For example, 'Learning from Training Dynamics: Identifying Mislabelled Data Beyond Manually Designed Features' (AAAI 2023) and 'Efficient Adaptive Label Refinement for Label Noise Learning' (Neurocomputing 2025). 

The method can be computationally expensive due to the integration over training time. Presenting large-scale datasets would demonstrate this in practice. 

The presented threshold is automatic, but is based purely on heuristics. This implies the need for particular tuning for different types of data. This is another reason for evaluation on standard benchmarks.

### Questions
Why was the real-world noise experiment skipped? 

Also, why are standard noisy label learning benchmarks missing?

### Soundness
2

### Presentation
3

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
This paper introduces SEI, a signed-entropy–based training-dynamics metric to detect mislabeled samples, and reports state-of-the-art results on three medical imaging datasets. The core problem tackled is that overparameterized networks tend to memorize noisy labels, degrading performance; detecting and filtering mislabeled data during training is therefore critical. The key idea is that correctly labeled samples show a consistent drop in prediction entropy across epochs, whereas mislabeled samples maintain relatively high entropy; SEI integrates (signed) entropy over time to capture both magnitude and trend, with the sign reflecting label–prediction consistency. The implementation is simple and the method plugs into standard training loops. The claimed conclusion is that SEI is a simple, efficient, and broadly applicable metric that achieves SOTA mislabeled-data identification without architectural changes or complex training procedures.

### Strengths
- The paper is clearly written and has a well-structured presentation. Figures and tables are clean and directly support the claims; the narrative is easy to follow.
- The empirical section is strong within the chosen three datasets: ablations and compatibility checks against other SOTA training pipelines.
- The method is simple and practical. No architectural changes needed; it integrates into existing training workflows with low overhead. This makes it work well with CLIP fine-tuning and, in principle, with standard classifiers.
- The use of entropy trajectories makes the decision process intuitive and easy to reason about. This gives the approach an interpretability angle.

### Weaknesses
- Limited novelty relative to prior entropy-based signals. The signed and temporal integration is a neat spin, but the conceptual jump may be incremental given prior work on entropy/uncertainty and training-dynamics signals.
- The paper has a narrow domain scope. All results are on medical imaging; it’s unclear whether the approach generalizes to broader CV/ML settings. This also challenges the impact of the SOTA results and brings forward a question about how competitive is SEI when evaluated in standard benchmarks for label noise.
- Similarly to the last point, the evaluation relies too much on synthetic noise. If the evaluation relies mainly on synthetic label noise, the conclusions may not carry over to real-world noisy datasets; this gap is well-documented in the literature.
- SEI depends on training dynamics; thresholds and rankings might vary with epoch budget, LR schedules, data augmentation, label smoothing, or heavy regularization. This could be addressed with further discussion or ablation studies exploring the different design decisions made.

### Questions
- Beyond CLIP, how does SEI perform with standard CNNs/ViTs trained from scratch or with supervised pretraining? Did the authors observe any meaningful differences in how different architectures present different entropy values/tendencies in noisy datasets?
- How sensitive is SEI ranking to training length (early vs late epochs), LR schedules, strong augmentation, label smoothing, and weight decay? Is there a recommended epoch window for a stable ranking?   
- Samples that are intrinsically hard or minority-pattern but correctly labeled may retain higher entropy and risk being flagged as noisy (class imbalance/long-tail scenarios). Do you have analyses showing that SEI does not systematically filter rare but correctly labeled patterns (e.g., minority subtypes)? Any per-class or per-subpopulation error analysis?
- In which regimes does SEI struggle? Have you explored more extreme noise rates or class imbalances?

### Soundness
3

### Presentation
4

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
The paper propose a metric called SEI for identifying mislabeled data, considering both entropy dynamics and label-prediction consistency. The metric is motivated by the observation that correctly labeled samples exhibit consistent entropy decrease during training, while mislabeled samples maintain relatively high entropy throughout training. Experiments on multiple benchmarks shows competitive performance.

### Strengths
The approach is simple, architecture-agnostic, and easy to integrate. Experiments demonstrate strong performance across multiple benchmarks.

### Weaknesses
my major concern is that the method is build upon empirical observation on a handful of dataset and noise setting. While the presented results seems promising, it is unclear when SEI is expected to succeed or fail.

More discussion and intuition is needed to explain why the proposed SEI works. For example, for high capacity network, the model has the potential to remember the wrong label, this might directly impact the label-prediction alignment pattern (one of the main component in the SEI).

Furthermore, because the experiments rely on controlled, artificially generated noise, it is unclear whether the training-dynamics patterns SEI exploits are specific to these noise models or generalize to real world applications.

### Questions
It seems odd to me to frame inter-observer variability as ‘label noise.’ In my view, such variability is more about uncertainty, not necessarily wrong labels. How should SEI handle inter-observer variability?

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
3

### Summary
This paper proposes a simple yet effective method, Signed Entropy Integral (SEI), for detecting mislabeled samples in training datasets through analysis of training dynamics.  
The core insight is that correctly labeled samples exhibit a steady decrease in prediction entropy during training, whereas mislabeled samples maintain persistently high entropy due to model–label inconsistency.  

SEI functions as an automatic, architecture-agnostic indicator of label noise.  
A self-calibrating threshold is introduced, leveraging auxiliary-class pseudo-errors to distinguish clean from mislabeled samples without requiring external supervision.  

Experiments conducted on three medical imaging datasets demonstrate that SEI achieves state-of-the-art performance for mislabeled data detection under both symmetric and confusion-calibrated noise conditions.  

Overall, the paper presents a clear and well-motivated study with good empirical support.  It provides valuable insight into leveraging entropy trajectories for robust label noise detection, supported by transparent presentation and convincing experimental validation.

### Strengths
**1. Clear Motivation and Conceptual Simplicity**

The paper addresses a well-motivated and underexplored problem—detecting mislabeled samples through training dynamics—and proposes a simple yet effective solution with clear conceptual grounding.

---

**2. Architecture-Agnostic Design and Practical Usability**

SEI operates without architectural modifications and shows strong generalization across both CNN (ResNet) and Transformer (CLIP) backbones.  
Its plug-and-play nature and minimal computational cost make it practical for real-world noisy-label scenarios.

---

**3. Interpretability and Diagnostic Insight**

The method provides an intuitive and interpretable perspective on model behavior under label noise.  
By linking entropy evolution to label correctness, it offers diagnostic insight into how neural networks react to mislabels during training.

### Weaknesses
**1. Sensitivity to Training Configuration**

Since SEI depends on entropy evolution across training, its stability under different optimization schedules remains unclear.  
Variations in training duration, learning rate, or early stopping may affect the computed SEI values, raising concerns about reproducibility.

---

**2. Lack of Statistical Significance Analysis**

Although the reported results show consistent improvements, no variance measures or statistical tests are provided.  
Without reporting standard deviations or confidence intervals, it is difficult to determine whether the gains are statistically meaningful.

---

**3. Lack of discussion on the stability and generality of the thresholding strategy.**

The auxiliary-class–based threshold still depends on a manually defined sampling ratio \( N / (K + 1) \).  
Its robustness under data imbalance, limited samples, or multimodal settings has not been validated.

---

**4. Limited Domain Generalization**

All experiments are conducted on medical imaging datasets.  
It remains uncertain whether SEI’s entropy patterns generalize to other domains such as natural image or text classification, where label noise and model calibration behave differently.

### Questions
**1. Sensitivity to Training Configuration**

It is unclear how sensitive SEI is to training hyperparameters such as the number of epochs or the learning rate.  
Since the method relies on entropy trajectories over training, the stability of SEI under different optimization schedules (e.g., early stopping or extended training) should be clarified.

> **Question:**  
> How sensitive is SEI to training duration and learning rate choices?

---

**2. Lack of Statistical Significance Analysis**

Although the experimental tables are comprehensive, they lack statistical tests such as confidence intervals or t-tests.  
Without measures of variability, it is difficult to judge whether the observed performance improvements are statistically significant or within the range of random fluctuation.

> **Questions:**  
> - Could the authors report standard deviations or statistical tests (e.g., t-test, CI) to validate the robustness of SEI’s improvements?  
> - How consistent are the results across multiple random seeds?

---

**3. Stability of the Thresholding Strategy**

The proposed auxiliary-class–based adaptive threshold depends on the ratio \( N / (K + 1) \), which remains a manually defined proportion.  
It is unclear whether this strategy is stable under data imbalance or when applied to multimodal tasks.

> **Questions:**  
> - How sensitive is the SEI threshold to the number or sampling ratio of auxiliary-class samples?  
> - Have the authors tested how performance changes when this ratio is varied?  

---

**4. Domain Generalization of the Entropy Trend**

The paper primarily focuses on medical imaging datasets.  
However, it remains uncertain whether the same entropy evolution patterns hold in other domains such as natural images, where model calibration and noise characteristics differ.

> **Question:**  
> For non-medical tasks, would SEI exhibit similar entropy trajectories, or is this behavior domain-specific?

### Soundness
3

### Presentation
3

### Contribution
3
