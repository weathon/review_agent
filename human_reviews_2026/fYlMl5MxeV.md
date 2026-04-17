# Calibration-Free Defense Against Backdoor Attacks in the Wild

- Decision: Reject
- Scores: 8, 2, 2, 6

## Abstract
The widespread adoption of pre-trained neural networks from unverified sources has heightened concerns about backdoor attacks. These attacks cause networks to misbehave on inputs containing specific triggers while maintaining normal performance otherwise. Existing methods typically rely on pruning, operating under the assumption that backdoors are encoded in a small set of specific neurons. This approach, however, is ineffective on large-scale models where phenomena like polysemanticity make isolating malicious neurons without harming model performance difficult. Furthermore, pruning-based methods are impractical as they require unavailable calibration data to determine critical thresholds, limiting their deployment in real-world scenarios. We introduce Calibration-free Model Purification (CMP), a novel, completely data-free defense that avoids pruning entirely. CMP leverages a self-distillation framework guided by our discovery of a systematic "prediction skew" as the fundamental mechanism for backdoor transfer during knowledge distillation. It employs a dual-filtering system that counteracts this skew, preventing the student model from inheriting the teacher's malicious behavior. On the challenging ImageNet dataset, CMP reduces attack success rates to near-zero across diverse attacks while preserving clean accuracy, outperforming existing methods. Our work presents the first scalable, threshold-free defense, offering a practical solution for real-world AI security.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Exisiting backdoor defense methods rely on pruning with the assumption that backdoors are encoded in a small set of specific neurons. This paper argues against this assumption for tis ineffectiveness on large-scale models and proposes Calibration-free Model Purification. It avoids purning entirely and leverages a self-distillation framework guided by the discovery of a systematic "prediction skew". A dual filtering system is employed. CMP reduces attack success rates on diverse attacks while preserving clean accuracy.

### Strengths
- The motivation of the work is very interesting. The authors identify two fundamental limitations from exisiting defenses, including assumption on backdoors encoded in a small subset of identifiable neurons and unavailability due to requring training resources form threshold calibration. Then a calibration-free model purification method is proposed. I think these observations are helpful to this area.
- The proposed CMP method employs two filters. They are reasonable and easy to implement. Across different backdoor attacks, the proposed method shows consistent better performances.
- The authors also show experiments on OOD data. This shows the generalizability of the proposed method.

### Weaknesses
- I did not find major weaknesses of this paper. Overall, I think the paper is well-motivated and presented.

### Questions
- In Figure 1, should the label for Teacher model in 2. Mismatch Filter be the Target Label?
- In Sec. 3.2, it mentions augmentations cause benign images to exhibit trigger-like characteristics. Which augmentation operators are evaluated? Does this conclusion apply to a set of augmentations or some specific augmetation operators?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes a data-free backdoor defense method, CMP, which utilizes a self-distillation framework to prevent a student model from learning the malicious behaviors of a teacher model. The results indicate that CMP is effective for backdoor defense. However, the experimental evaluation is incomplete: the main performance evaluation was conducted solely on ImageNet, whereas the analysis for the key motivation in the main text used CIFAR-10. Additionally, the paper lacks a discussion on adaptive attacks.

### Strengths
1，	The knowledge distillation process can cause a student model to inherit the backdoor behaviors of its teacher. Addressing this issue, this work proposes a more robust and secure distillation framework.
2，	This work utilizes "Model Inversion" to synthesize data from the model itself, requiring no additional clean datasets.

### Weaknesses
1，	The notion that backdoors can transfer from a teacher to a student model via knowledge distillation has been discussed in prior works [1] [2]. However, in the "RELATED WORKS" section, the authors claim to have "uncovered an unprecedented mechanism that allows backdoors to transfer." It is recommended that the authors review and clarify their novelty claim here.
2，	This work finds that data augmentation can cause benign images to exhibit trigger-like characteristics. Although the authors demonstrate this phenomenon in Table 2, they do not provide a more detailed analysis or discussion. It is suggested to add visualizations or other more interpretable analyses.
3，	In its related work on 'Backdoor Defense', this paper discusses pruning-based defenses more extensively. However, the core method of this work is self-distillation. It is recommended that the authors include more summary of similar methods [1-6].
4，	What are the time and computational costs of model inversion, and what is the visualization of the synthetic image samples? 
5，	In Section 3.2, the authors design an experiment on the CIFAR-10 dataset to demonstrate the "prediction skew" phenomenon. However, in the experimental evaluation in Section 5, the authors do not provide the defense performance results of their method on CIFAR-10.

### Questions
1，	The paper states in Section 5.1 that the CMP method synthesizes 1000 images per class, which implies that for the ImageNet-1K dataset, approximately 1,000,000 synthetic images need to be generated. What is the time cost of this process?
2，	The paper lacks a discussion on adaptive attacks. If an attacker anticipates that the defender will use distillation for purification, they could design a novel backdoor to bypass this defense. Notably, [7] proposed Anti-Distillation Backdoor Attacks (ADBA), which are specifically designed to survive the knowledge distillation process and successfully transfer from the teacher model to the student.

The citation of weakness and questions are listed below.
[1] Yao Z, Zhang H, Guo Y, et al. Reverse backdoor distillation: Towards online backdoor attack detection for deep neural network models[J]. IEEE Transactions on Dependable and Secure Computing, 2024, 21(6): 5098-5111.
[2] Hong J, Zeng Y, Yu S, et al. Revisiting data-free knowledge distillation with poisoned teachers[C]//International Conference on Machine Learning. PMLR, 2023: 13199-13212.
[3] Yoshida K, Fujino T. Disabling backdoor and identifying poison data by using knowledge distillation in backdoor attacks on deep neural networks[C]//Proceedings of the 13th ACM workshop on artificial intelligence and security. 2020: 117-127.
[4] Hu C, Teng X, Xing W, et al. Distill To Detect: Amplifying Anomalies in Backdoor Models through Knowledge Distillation[C]//ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025: 1-5.
[5] Bie R, Jiang J, Xie H, et al. Mitigating backdoor attacks in pre-trained encoders via self-supervised knowledge distillation[J]. IEEE Transactions on Services Computing, 2024, 17(5): 2613-2625.
[6] Li X, Gao Y, Liu M, et al. A New Data-Free Backdoor Removal Method via Adversarial Self-Knowledge Distillation[J]. IEEE Internet of Things Journal, 2024.
[7] Ge Y, Wang Q, Zheng B, et al. Anti-distillation backdoor attacks: Backdoors can really survive in knowledge distillation[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 826-834.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes CMP (Calibration-Free Model Purification) — a data-free, threshold-free defense against neural backdoor attacks. The key insight is that backdoor persistence during knowledge distillation arises not merely from specific neurons but from a systemic prediction skew, where the poisoned teacher model over-assigns probability mass to the target class even on benign inputs. CMP adopts a self-distillation framework using synthetic images generated via model inversion.

### Strengths
- The paper is well-organized and clearly written.
- The paper introduces a new causal explanation of backdoor transfer through prediction skew, shifting the perspective from neuron-level pruning to distribution-level bias correction.
- The dual-filter design is conceptually novel and operationally elegant, effectively removing backdoors without any calibration data.

### Weaknesses
- The “prediction skew” phenomenon, though empirically compelling, lacks a formal definition or an information-theoretic grounding. No bound is provided linking skew magnitude to ASR reduction. A KL-based or mutual information analysis would strengthen the theoretical foundation.
- Experiments are restricted to moderate-size architectures (ResNet-18, small ViT/DeiT). The paper claims scalability to larger models but does not report training cost, memory, or wall-time for full ImageNet distillation. This is critical for real-world viability.
- The attacker is assumed fixed; CMP’s robustness against adaptive threats is not explored.

### Questions
- Why is detaching the second-highest logit the most effective choice?
- How sensitive is CMP to the quality or diversity of DeepInversion-generated samples? Does the defense remain stable if inversion produces low-quality or partially mode-collapsed data?
- Please report distillation runtime, GPU hours, and resource overhead relative to standard fine-pruning or NAD.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Calibration-Free Model Purification (CMP), a fully data-free and threshold-free defense mechanism against backdoor attacks in deep neural networks. CMP leverages a self-distillation framework with two novel components — the mismatch filter and skew filter — to prevent the transfer of backdoor behaviors from a poisoned teacher to a clean student model. It operates without requiring calibration data or pruning, achieving robust results on ImageNet-scale experiments across diverse attacks. The authors claim CMP is the first practical, scalable, and data-free defense adaptable to both CNNs and transformers.

### Strengths
- The idea of removing the backdoor without calibration data is novel and practically important. The paper contributes new insights into the mechanisms of backdoor transfer during knowledge distillation. 
- The method is well-motivated with analysis experiments in section 3.2, which also makes the presentation better for understanding.
- The extensive experiments verified the effectiveness of CMP. Its architecture-agnostic design strengthens its broader impact.

### Weaknesses
- While the prediction skew is identified as the key phenomenon, the paper offers limited formal or mathematical justification for why the dual-filtering scheme guarantees elimination of such skew under different attack types. Especially for the skew filter, which relies heavily on the assumption of the second-largest probability class as the backdoor class, it lacks empirical evidence.
- Despite the synthesized data is from the previous technique (i.e., model inversion), it lacks visualization or explanation of how different they look compared to the real image. And more experiments are needed to validate the effectiveness of CMP using different model inversion techniques, illustrating the influence of the synthesized data.
- The required LLM statement is not included in both the main text and appendix.

### Questions
- The appendix is submitted separately, making it hard to find. As the author guide (https://iclr.cc/Conferences/2026/AuthorGuide) encourages submitting a single file (paper + supplementary text), can you concatenate them?
-  Why do the OTBR (fixed) and (oracle) version performs the same in Table 4? Does it indicate that the calibration data is not important for an effective OTBR?

### Soundness
3

### Presentation
3

### Contribution
3
