# Lightweight Spatio-Temporal Modeling via Temporally Shifted Distillation for Real-Time Accident Anticipation

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Anticipating traffic accidents in real time is critical for intelligent transportation systems, yet remains challenging under edge-device constraints. We propose a lightweight spatio-temporal framework that introduces a temporally shifted distillation strategy, enabling a student model to acquire predictive temporal dynamics from a frozen image-based teacher without requiring a video pre-trained teacher. The student combines a RepMixer spatial encoding with a RWKV-inspired recurrent module for efficient long-range temporal reasoning. To enhance robustness under partial observability, we design a masking memory strategy that leverages memory retention to reconstruct missing visual tokens, effectively simulating occlusions and future events. In addition, multi-modal vision-language supervision enriches semantic grounding. Our framework achieves state-of-the-art performance on multiple real-world dashcam benchmarks while sustaining real-time inference on resource-limited platforms such as the NVIDIA Jetson Orin Nano. Remarkably, it is 3-7$\times$ smaller than leading approaches yet delivers superior accuracy and earlier anticipation, underscoring its practicality for deployment in intelligent vehicles.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces a lightweight spatio-temporal framework for real-time traffic accident anticipation, particularly suited for deployment on resource-constrained edge devices. The key contributions are: (1) a novel temporally shifted distillation strategy that enables a student model to learn predictive temporal dynamics from a frozen image-based teacher, eliminating the need for a video pre-trained teacher and making it ideal for small datasets and low-resource settings; (2) a hybrid architecture that combines RepMixer spatial encoding with an RWKV-inspired recurrent module for efficient long-range temporal reasoning, achieving low computational complexity; (3) a masked memory strategy that simulates occlusions and partial visibility to improve robustness under real-world conditions.

### Strengths
1. The paper innovatively redefines temporal learning through temporally shifted distillation, removing the need for video-pretrained teachers and enabling real-time performance under constraints.
2. Comprehensive experiments and ablation studies show improvements in mAP and mTTA with fewer parameters.
3. Clear structure, well-explained components, and visual aids make the paper accessible, though some concepts remain high-level.

### Weaknesses
1. The stitching method of the spatio-temporal scanning pattern seems somewhat arbitrary. There might be continuous information between feature sub-blocks that is not utilized. Is this an existing design approach?

2. In lines 231-236, I don't understand why only KV is involved in the calculation and not the Q parameter.

3. In lines 241-260, can the masked part be considered multi-stage, requiring multiple training iterations? Does this introduce higher training costs compared to other models? Is this reflected in the paper?

4. Lines 264-269: Are there applications on larger-scale datasets? I would like to know how this lightweight model performs under larger-scale conditions. Even if the performance isn't strong, can you provide the model's applicability boundary?

5. Table 4: The performance on the CCD dataset seems close to overfitting. Has the lightweight version been applied to other methods? Also, are there additional evaluation metrics? Are the two metrics used sufficient?

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a lightweight dash-cam accident-anticipation model that distills future-aware temporal cues from a frozen MobileCLIP teacher using a windowed RWKV student and masked memory. It addresses the problem of anticipating accidents several seconds earlier than prior work while staying real time on edge devices with only RGB inputs. Experiments show that the porposed method obtains 75.3% mAP and 4.04s mTTA on DAD with 26M params, beating heavier SOTA (in Table 4).

### Strengths
- The reviewer found the idea of distilling from an image-only MobileCLIP into a temporal RWKV student for accident anticipation to be interesting.
- Interestingly, temporal-only distillation outperforms spatial-only (74.1% vs 71.2% mAP in Table 2), showing future supervision actually drives anticipation.
- Experiments are comprehensive: Table 1 (RWKV depth), Table 2 (distillation parts), Table 3 (mask ratio), Figure 4 (qualitative attention) and writing is mostly clear and structured.

### Weaknesses
- The temporally shifted distillation still does not show a direct comparison to distilling from a video teacher., so it is unclear how much is due to the shift itself.
- Experiments miss an ablation on modern, purely temporal CNN/Transformer baselines that work on DAD/CCD datasets even though Table 4 already cites many frame-level systems.

### Questions
- Please respond to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a lightweight framework for accident anticipation using temporally shifted distillation (TSD).
A student model learns to predict future cues by aligning its current features with a teacher’s representation from later frames.
The design combines a RepMixer backbone with an RWKV-based temporal block for efficiency.
Experiments on DAD and CCD show improved early anticipation and real-time performance on edge devices.

### Strengths
1.	The idea of learning temporal prediction through shifted distillation is original and intuitively appealing.

2.	The method achieves a good balance between accuracy and efficiency, which makes it suitable for real-world deployment.

3.	The paper is clearly written, and the model architecture and ablation design are well presented.

### Weaknesses
1.	The analysis of the temporally shifted distillation is limited.
Table 2 only reports overall performance, so it is difficult to tell whether the model actually learns to anticipate future events or only captures static correlations.
Additional experiments showing temporal behavior, such as feature alignment across time, would strengthen the claim.

2.	The mechanism itself is not deeply analyzed.
It would be useful to evaluate how different time shifts affect learning or whether the model improves its understanding of future frames during training.

3.	Since the teacher model processes only single frames, it is unclear how much of the predictive ability comes from the distillation process compared with the student’s temporal module.
More explanation or comparative evidence would clarify this point.

### Questions
Have the authors tried using a video-based teacher that already has temporal understanding? It would be interesting to see whether the proposed shift is still necessary in that setting.

### Soundness
4

### Presentation
3

### Contribution
3
