# A robust PPG foundation model using multimodal physiological supervision

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Photoplethysmography (PPG), a non-invasive measure of changes in blood volume, is widely used in both wearable devices and clinical settings. Although recent work has explored PPG foundation models using large-scale intensive care unit (ICU) datasets, these efforts often assume the need for clean and high-quality signals. In contrast, we argue that the inherent noise and variability in ICU datasets can be harnessed to build more robust and generalizable representations. To address this, we propose a PPG foundation model that leverages accompanying electrocardiogram and respiratory signals in ICU datasets to select contrastive samples during pretraining. Our approach allows the model to retain and learn from noisy PPG segments, improving robustness without requiring multimodal inputs at inference. Our model, pretrained on 3x fewer subjects than existing state-of-the-art approaches, achieves performance improvements of up to 36\% in classification and 42\% in regression on 14 out of 15 diverse downstream tasks, including stress and heart rate prediction. Our results demonstrate that multimodal supervision can leverage clinical data to enable the development of robust, unimodal foundation models for both clinical and consumer-level data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a robust PPG foundation model that leverages ECG and respiratory signals to guide sample selection during contrastive learning. The paper emphasizes that by integrating complementary biosignal modalities, the proposed approach effectively mitigates the limitations of unimodal, morphology-based contrastive targets, resulting in substantially improved robustness, generalization, and downstream task performance.

### Strengths
- The paper’s focus on multi-modal supervision is well-motivated, particularly given that health is inherently multi-modal while most existing foundation models remain unimodal.
- In general, the ablation and case studies offer meaningful insights and contribute to understanding the model’s behavior and robustness.
- The paper is clearly written, well-organized, and easy to follow.

### Weaknesses
**Method:** The proposed approach employs five key physiological metrics from ECG and RESP: HR, RMSSD, RR, RA, and RV to guide multi-modal supervision during pre-training, enabling the model to learn corresponding representations in the latent space. However, many of the downstream tasks evaluated (e.g., HR, BP, and activity recognition) are directly related to these same input measures. This raises concerns about potential task overlap and limited generalization. How can this design choice be justified? If the input features and downstream tasks are closely aligned, it becomes unclear whether the learned representations truly generalize beyond the pre-training objectives. Perhaps, consider evaluations on tasks that are novel and not previously explored. 

**Experimental Design**: 
There are two major concerns with the experimental design: **(1) the choice of baseline** and **(2) the use of derived metrics for training**.

First, the experiments rely heavily on PaPaGei as the sole baseline. While PaPaGei is a reasonable point of comparison, it cannot be the only one. The key issue is that PaPaGei is trained exclusively on PPG signals, whereas the proposed model leverages multi-modal supervision, including ECG and respiratory signals. Comparing a unimodal foundation model with a multi-modal supervised one introduces an inherent imbalance and may not provide a fair assessment of performance gains.

Second, the rationale for using derived physiological metrics (HR, RMSSD, RR, RA, and RVT) during pre-training needs stronger justification, especially since these metrics are closely correlated with several downstream tasks. It remains unclear whether their inclusion in pre-training offers benefits beyond what could be achieved by incorporating them at the linear probing stage alongside the learned embeddings. More fundamentally, how would a simple baseline model trained directly on these five derived features perform on the same downstream tasks? Addressing this question would help clarify the true contribution of the proposed approach.

**Minor:**  
Figure 3 — The comparison of UMAP plots for heart rate may not be meaningful, as the proposed approach uses heart rate as part of its pre-training objectives, whereas the baseline models do not. Consequently, it is expected that the proposed model exhibits a clearer gradient structure in the latent space, which limits the interpretive value of this comparison.

**Ablation Study:** It would be valuable to analyze the individual contribution of each computed metric derived from the co-recorded signals. Specifically, examining the effect of using HR, RMSSD, RR, RA, and RVT during pre-training, either by incorporating one metric at a time or by comparing groups of metrics (e.g., ECG-based vs. respiratory-based). This could provide deeper insights into which modalities or features most influence model performance.

**Open-Source**: The models and code are not publicly released, even though the proposed approach is trained and evaluated on open-source datasets. This limits the reproducibility of the work and weakens its overall contribution, particularly given that other open-source PPG foundation models already exist [1, 2].

Overall, the main contribution of this work appears to be the inclusion of additional biosignal modalities during contrastive pre-training, which improves the performance of a PPG foundation model. While this idea is interesting, the contribution is not sufficiently significant, as prior studies have already explored unimodal versus multimodal representations in similar contexts [3, 4]. Importantly, given the limitations in the experimental design, methodological justification, and lack of open-source release discussed above, I lean toward a weak reject recommendation.

[1] Pillai, A., Spathis, D., Kawsar, F., & Malekzadeh, M. (2024). Papagei: Open foundation models for optical physiological signals. _arXiv preprint arXiv:2410.20542_.

[2] Saha, M., Xu, M. A., Mao, W., Neupane, S., Rehg, J. M., & Kumar, S. (2025). Pulse-ppg: An open-source field-trained ppg foundation model for wearable applications across lab and field settings. _Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies_, _9_(3), 1-35.

[3] Zhou, Y., Khasentino, J., Yun, T., Biradar, M. I., Shreibati, J., Lai, D., ... & Hormozdiari, F. (2025). Applying multimodal AI to physiological waveforms improves genetic prediction of cardiovascular traits. _The American Journal of Human Genetics_.

[4] Ezzameli, K., & Mahersia, H. (2023). Emotion recognition from unimodal to multimodal analysis: A review. _Information Fusion_, _99_, 101847.

### Questions
- Are there other ECG and RESP metrics that can be used during contrastive pre-training?

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
This paper introduces a PPG foundation model pretrained on multimodal ICU data, where synchronized ECG and respiratory signals guide contrastive learning to derive robust and generalizable PPG representations. Compared with prior single-modality approaches (e.g., PaPaGei), the model leverages noise and signal variability more effectively, achieving substantial performance gains on 14 out of 15 downstream tasks across six unseen datasets.

### Strengths
1. This study leverages ECG and respiratory signals to enhance PPG representation learning, demonstrating a solid understanding of the physiological mechanisms underlying PPG.
2. It evaluates both cross-subject and within-subject settings, providing a comprehensive view of the model’s generalization performance.
3. The proposed model achieves significant performance improvements over PaPaGei across multiple downstream tasks.

### Weaknesses
Key Concerns:
1. The study compares the proposed model only with PaPaGei, which limits the comprehensiveness of the evaluation. It is recommended to include additional self-supervised or pretrained baselines (e.g., SimCLR, PulsePPG) to strengthen the experimental validity.
2. The paper claims that the inherent noise and variability in ICU data can be leveraged to improve model robustness. However, the proposed pretraining approach relies on five contrastive objectives computed from synchronized ECG and respiratory signals. It remains unclear how the authors ensure that these auxiliary signals remain reliable when the PPG signal quality deteriorates (e.g., due to patient motion). As a result, this claim may be somewhat overstated, since the method appears to focus more on multimodal assistance for PPG representation learning rather than on directly addressing signal quality issues.
3. Although the proposed model performs well on downstream tasks, concerns remain regarding its cross-dataset generalization. The model is pretrained solely on the MIMIC dataset, while larger and more diverse publicly available datasets such as MESA or VitalDB could have been incorporated to build a more comprehensive pretraining corpus. Although the authors mention that sleep or anesthesia data may contain relatively stationary signals, incorporating more heterogeneous datasets could capture a wider range of physiological patterns and improve the model’s robustness and generalization.
4. The method employs only derived features from ECG and respiratory signals instead of the raw multimodal inputs, which may constrain the model’s ability to capture complex temporal dependencies.

Minor Concerns: 
1. Although the number of subjects used is one-third of that in PaPaGei, the total number of data segments is comparable, thus the claim of higher data efficiency is not entirely justified.
2. The method employs only derived features from ECG and respiratory signals instead of the raw multimodal inputs, which may constrain the model’s ability to capture complex temporal dependencies.
3. Table 1 does not report the number of subjects and total samples for each downstream dataset, which makes it somewhat difficult to fully assess data balance and generalization stability.

### Questions
1. Could the authors clarify whether the auxiliary signals (ECG and respiration) remain reliable for contrastive supervision when the PPG signal quality is low, for instance due to patient motion?
2. Would it be possible to include additional self-supervised baselines such as SimCLR, BYOL, or PulsePPG for a more comprehensive comparison?
3. Could the authors elaborate on the rationale for using derived metrics instead of full multimodal inputs, and whether any experiments were conducted to validate this design choice?
4. It would be helpful to provide statistics on the number of subjects and total samples for each downstream dataset, to better contextualize task scale and model performance.
5. Incorporating downstream tasks related to cardiac arrhythmias (e.g., atrial fibrillation) could further demonstrate the model’s ability to handle abnormal cardiac patterns.

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
4

### Summary
The paper proposes a PPG foundation model pretrained on noisy ICU PPG by using ECG and respiration only during pretraining to compute HR, RMSSD, breathing rate, breathing amplitude, and RVT. A rank-n-contrast loss then pulls PPG embeddings closer when the targets are similar, aiming to learn noise-robust representations.

### Strengths
Clear, physiologically grounded supervision: Using ECG/RESP to form targets avoids brittle PPG morphology extraction, yet preserves unimodal inference. The target set (HR, RMSSD, RR, RA, RVT) is plausible for 10-s windows and filtered for physiological ranges.

### Weaknesses
- The contribution of the method is trivial and uses the common infoNCE loss. The authors only consider the physiological parameters to decide the positive and negative pairs.
- The robustness of HR/RMSSD/RR/RA/RVT estimation on 10-s windows (detectors, failure handling, thresholds) is crucial; more explicit error rates/quality filters would strengthen claims about the stability of the metric space.
- The final backbone checkpoint is chosen by VitalVideos systolic BP probe performance for practicality. This could inadvertently bias toward that dataset/task; a small sensitivity analysis (random/earliest/best-avg across a subset) would help.
- The final backbone is ~28.8M params vs PaPaGei’s ~5–5.7M. The architecture ablation shows gains even when the architecture differs, but fully disentangling capacity from supervision remains tricky without equal-capacity baselines for all comparisons.
- No fine-tuning heads or end-to-end adaptation are reported; it’s unclear how the model behaves under modest supervised finetuning, which is typical in practice.

### Questions
Please see the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2
