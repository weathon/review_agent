# Privacy-Aware Video Anomaly Detection through Orthogonal Subspace Projection

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Video anomaly detection (VAD) is central to modern surveillance, yet most existing methods optimize for accuracy while overlooking critical ethical concerns such as privacy and transparency. For deployment in real-world settings, VAD should not only detect anomalies reliably but also respect fundamental privacy principles. We propose the Orthogonal Projection Layer (OPL), a lightweight architectural module that suppresses task-irrelevant variations, including background clutter and noise, to produce representations focused on anomaly-relevant cues. Faces, unlike other cues such as gait or body pose, are highly sensitive biometric identifiers: they uniquely reveal identity, are tightly regulated by data protection laws, and pose immediate risks of misuse. To address the privacy risks inherent in human-centered anomalies, we extend this idea to the Guided OPL (G-OPL). Using only weak supervision from face-presence indicators, G-OPL selectively removes facial attributes while retaining non-identifying human features needed for anomaly detection. A cosine alignment loss ensures that facial information is systematically captured and neutralized, without requiring identity labels or adversarial training. We further introduce a privacy-aware evaluation framework that jointly assesses anomaly detection accuracy, privacy preservation, and interpretability. Our analysis uncovers how projection layers filter sensitive information, why this improves transparency, and under what conditions ethical design also enhances robustness. Extensive experiments confirm that embedding ethical constraints directly into model design strengthens privacy protection while maintaining, and in some cases improving, anomaly detection performance. These results position projection-based architectures as a principled path toward trustworthy and deployable VAD systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses privacy concerns in video anomaly detection (VAD) by proposing Orthogonal Projection Layer (OPL) and Guided OPL (G-OPL) to suppress task-irrelevant nuisances and privacy-sensitive facial information from intermediate representations. The method uses QR decomposition for stable orthogonal projection and weak supervision (face presence signals) to guide the removal of sensitive attributes. Three novel privacy metrics (SSC, ARD, PD/FPD) are introduced to quantify privacy preservation. Experiments on five benchmarks show the method can maintain or improve detection performance while reducing identity leakage, with notable gains on ShanghaiTech (+8.4% AUC) and UCSD Ped2 (+7.1% AUC).

### Strengths
- The three complementary metrics (SSC, ARD, PD/FPD) provide the first systematic quantification of privacy leakage in VAD, validated against ArcFace-based identity retrieval.

- QR decomposition avoids adversarial training instability while maintaining differentiability. The geometric alignment loss is elegant and interpretable.

-  Projection matrices QQ^T offer geometric visualization of removed information, valuable for trust and auditability.

### Weaknesses
## Main Concerns:

- The paper removes facial features regardless of their relevance to anomalies, which contradicts practical scenarios where faces may be essential for detection (e.g., unauthorized access, aggressive behavior, surveillance evasion). The authors acknowledge designing G-OPL to suppress even task-relevant facial features, yet provide no analysis of when this is appropriate vs. harmful. This appears optimized for benchmark scores rather than real-world utility. 

## Minor Issues:
- Results vary dramatically: ShanghaiTech (+8.4%), CUHK Avenue (+3.5%), yet the paper doesn't explain why. Critical missing analyses: (a) What percentage of anomalies contain faces in each dataset? (b) Are faces correlated with or orthogonal to anomaly labels? (c) For MSAD with pre-blurred videos, what is G-OPL actually removing from pre-extracted features?

### Questions
- When should G-OPL be applied?
- Why do performance gains vary across datasets?

### Soundness
4

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
5

### Summary
This paper proposes to handle privacy-preserving video anomaly detection by using orthogonal projection layers to project out sensitive face data from VAD features. The authors propose multiple variants of the method with different properties, demonstrating their success on different scenarios and in different configurations. Notably, this approach does not require adversarial optimization or direct private attribute labels.

### Strengths
1. Achieving privacy-preservation while mitigating the use of unstable adversarial training is advantageous. 
2. The proposed method is very flexible in its deployment and does not add much overhead to the existing VAD models.
3. Handling privacy-preservation in the latent space is a natural improvement to existing methods.
4. The proposed ARD metric appears to be a useful measure of utility variance from a baseline model.

### Weaknesses
1. The proposed form of privacy preservation is reliant wholly on mitigating similarity to localized facial features, which is a weak notion of privacy. There are many other attributes that could be considered private. It is unclear if this method could extend to handle multiple attributes simultaneously.
2. The main privacy-utility result Table 3 is largely empty. Lines 340-341 claim that this is the first comprehensive privacy analysis for VAD across datasets, yet the privacy analysis is only conducted on the proposed method. The metrics should be computed for prior methods/baselines as well.
3. Tables 1 and 2 simultaneously compare many variations of the proposed method (backbone features, backbone model), making direct comparisons a bit unclear. Some variants perform better in some cases, while other variants are better in different cases. More analysis should be provided here.
4. The proposed privacy metrics are specific to the training method and likely will not generalize well to methods that don't explictly optimize to project out facial features, even if a method is generally more privacy-preserving. While useful for measuring the performance of this method and finding ideal layer placements, they don't appear generally valuable to the community.
5. The method appears difficult to optimize since optimal configurations change based on the dataset trained on. It would be better to find a setting that works well (even if not the best possible) across all datasets.

### Questions
1. Can the metrics be fairly applied to prior methods? Especially the privacy-preserving ones.
2. Lines 463-466 claim advantages of the FPD metric over classifier probe-based approaches. Doesn't FPD use a classifer probe? Please clarify this.
3. Could the OPL be added to existing privacy-preserving approaches to improve their performance while maintaining their privacy-preservation?
4. How does this method perform under existing privacy-preservation metrics (like VISPR used in SPAct/TeD-SPAD)?

#### Suggestions:
1. The findings in light blue blocks are verbose and often basically just a repeat of a prior paragraph. These should be condensed into impactful statements/findings instead of paragraphs.
2. TeD-SPAD uses I3D, not Swin-T.

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
This paper focuses on privacy-aware video anomaly detection (VAD). Existing VAD methods often capture irrelevant or nuisance features that are unnecessary for detecting anomalies. To address this issue, the authors propose a method based on an Orthogonal Projection Layer (OPL), which learns a nuisance subspace and projects feature representations away from it. Additionally, a guided version of OPL is introduced to explicitly remove identity-related information. The paper also presents new privacy-aware evaluation metrics—namely, Sensitive Subspace Capture (SSC), Anomaly Retention Distance (ARD), and Privacy Decay (PD/FPD). Extensive experiments are conducted on five VAD datasets, supported by both quantitative results and qualitative visualizations.

### Strengths
1. The overall method is intuitive, conceptually simple, and easy to follow.
2. Plug-and-play design: the proposed OPL/G-OPL modules seems compatible with existing VAD frameworks.
3. The approach is evaluated across multiple benchmark datasets, demonstrating robustness and general applicability.
4. The paper introduces new privacy metrics (SSC, ARD, PD/FPD) that  quantify the privacy–utility trade-off.

### Weaknesses
1. The approach requires heavy hyper-parameter tuning, including layer-specific placement and the number of projection modules. Its performance appears sensitive to configuration choices and may not generalize well to unseen domains without retuning.
2. Table 3 feels incomplete, as results for the baseline models are missing. It would be useful to include the proposed metrics for those baselines to better show how the method performs relative to them and to demonstrate its general effectiveness.
5. The figures are quite hard to follow, especially Figures 1, 5, and 6. It would help if the authors can clarify what each element represents or improved the captions so readers can better understand the result.
6. The evaluation is limited to two backbone models (RTFM and MGFN). Demonstrating results on additional VAD architectures would strengthen the generality of the proposed approach.
7. The proposed metrics, such as Sensitive Subspace Capture (SSC), are very specific to face-related privacy. Because of this narrow focus, they may not be broadly useful or directly applicable to future works

### Questions
**Performance discrepancy:**
In Table 1, the results with G-OPL appear are lower than those with OPL alone, and a similar trend is observed in Table 2 for the MGFN model. Could the authors clarify why the guided version sometimes underperforms the unguided variant?

**Improvement on blurred-face data:**
For MSAD, the paper reports improvement with G-OPL even though the faces are blurred in the input videos. How does the face-guided mechanism provide a performance gain when explicit facial information is largely unavailable?

**Figure-3(a)**
There is a noticeable performance drop at 𝑘=64, followed by an improvement at k=128. Could the authors clarify why performance recovers at larger k?

### Soundness
2

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
3

### Summary
This paper addresses an important and timely topic in privacy-aware video anomaly detection. The proposed projection-based approach removes sensitive facial information while keeping features useful for detecting anomalies. The idea is original and well motivated, filling an ethical gap that is often overlooked in this area. Some parts could be clearer and the experimental validation could be stronger.

### Strengths
- The topic is interesting and important in the age of AI.
- This paper proposed some privacy-aware metrics, which are important for privacy-aware performance evaluation.
- The paper is well-written and easy to follow.

### Weaknesses
- It is not very clear why the proposed method can protect privacy. As shown in Fig. 2, the input to the model is still images with sensitive information. The authors need to explain clearly why removing sensitive features is helpful if input images are not anonymized.
- In the abstract, the authors mentioned that "Faces, unlike other cues such as gait or body pose, are highly sensitive biometric identifiers". This is only partly true. Gait, some sometimes body pose, are also sensitive biometric identifiers.
- Fig. 1 is not very easy to understand. In the caption, the authors mentioned that "G-OPL, guided by face presence, isolates sensitive biometric cues", but this is not easy to observe from Fig. 1
- It is not clear whether the design of evaluation metrics presented in Section 3.3 has considered the effect of video length.
- Overall, the compared methods are a bit outdated. More methods published in 2024 and 2025 should be compared.

### Questions
- In the abstract, the authors mentioned that "Faces, unlike other cues such as gait or body pose, are highly sensitive biometric identifiers". The authors should give more explanations about this sentence, because some sometimes body pose, are also sensitive biometric identifiers. 
- Fig.1 should be improved. Why does Fig. 1 show "G-OPL, guided by face presence, isolates sensitive biometric cues"?
- The motivation of this paper should be more clear. The authors need to explain clearly why removing sensitive features is helpful if input images are not anonymized.
- Has the design of evaluation metrics presented in Section 3.3 considered the effect of video length? More details are needed.

### Soundness
2

### Presentation
3

### Contribution
3
