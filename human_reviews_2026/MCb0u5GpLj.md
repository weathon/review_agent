# Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Pre-trained vision-language models (VLMs) such as CLIP have demonstrated strong zero-shot capabilities across diverse domains, yet remain highly vulnerable to adversarial perturbations that disrupt image-text alignment and compromise reliability. 
Existing defenses typically rely on adversarial fine-tuning with labeled data, limiting their applicability in zero-shot settings. 
In this work, we identify two key weaknesses of current CLIP adversarial attacks—lack of semantic guidance and vulnerability to view variations—collectively termed semantic and viewpoint fragility. To address these challenges, we propose \textsc{Self-Calibrated Consistency (SCC)}, an effective test-time defense. SCC consists of two complementary modules: 
\textit{Semantic consistency}, which leverages soft pseudo-labels from counterattack warm-up and multi-view predictions to regularize cross-modal alignment and separate the target embedding from confusable negatives; and
\textit{Spatial consistency}, aligning perturbed visual predictions via augmented views to stabilize inference under adversarial perturbations.
Together, these modules form a plug-and-play inference strategy. 
Extensive experiments on 22 benchmarks under diverse attack settings show that SCC consistently improves the zero-shot robustness of CLIP while maintaining accuracy, and can be seamlessly integrated with other VLMs for further gains. 
These findings highlight the great potential of establishing an adversarially robust paradigm from CLIP, with implications extending to broader vision-language domains such as BioMedCLIP. Code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Self-Calibrated Consistency (SCC), a test-time defence mechanism that enhances the adversarial robustness of pre-trained CLIP without requiring retraining. The authors identify two central weaknesses in existing test-time defences: semantic drift and viewpoint fragility. This paper addresses them through two complementary modules: Semantic Consistency, which regularises cross-modal alignment using pseudo-labels from multi-view counterattack warm-up, and Spatial Consistency, which enforces multi-view prediction agreement to stabilise adversarial recovery. The proposed SCC operates in a plug-and-play fashion, improving robustness while maintaining clean accuracy across 22 benchmarks, including medical datasets and BioMedCLIP variants.

### Strengths
- The paper identifies three weaknesses in existing test-time defences: semantic drift, view sensitivity, and hard-negative dominance, supported by both empirical and analytical evidence (Sec. 3.1). These findings motivate the proposed method, which appears technically sound.
- SCC introduces a semantic consistency loss anchored by soft prototypes derived from multi-view pseudo-labels. The integration of spatial consistency through multi-view logits averaging (Eq. 9) provides an interesting and effective variance-reduction mechanism.
- The paper presents extensive evaluations across 22 benchmarks (16 general-domain and 6 medical datasets), showing consistent robustness under multiple attack budgets and outperforming previous test-time defences.
- The proposed test-time defence operates without adversarial fine-tuning, making it computationally efficient and practically attractive for real-world applications.

### Weaknesses
- Although the authors claim to evaluate adaptive attacks using PGD and CW, these are standard white-box attacks and not adaptive to the defence mechanism. More rigorous adaptive-attack analysis is needed to assess true robustness.
- The evaluation scope is appreciated, but currently limited to two attack types and CLIP-based classification tasks. Expanding experiments to include more attacks, model architectures, and downstream tasks would strengthen the work.
- The paper’s theoretical component lacks depth. While the propositions are conceptually interesting, the analysis of the three vulnerabilities remains mostly heuristic. The first claimed contribution in the introduction lacks clear supporting evidence.
- It is unclear whether baselines are re-implemented or drawn from released checkpoints. Clarifying this, along with reporting results averaged over multiple random seeds, would improve reproducibility and fairness of comparison.
- SCC’s soft prototype $t_{\text{soft}}$ (Eq. 7) depends on text embeddings and warm-up steps in CLIP’s cosine space. The method’s applicability to generative or non-contrastive VLMs is not discussed and should be addressed.
- Several notations (e.g., $\hat{t}_{y^{*}}$) appear without prior definition or explanation, which slightly reduces clarity.

### Questions
- Could the authors conduct experiments with truly adaptive attacks, where the attacker knows SCC’s mechanism and incorporates it into the attack optimisation, to quantify how much the defence can be compromised?

- Could the authors also explore additional attack methods, such as  [1–3], as well as non-CLIP models and tasks beyond classification, to further validate the generality of SCC?

[1] Croce, F., & Hein, M. (2020, November). Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning (pp. 2206-2216). PMLR.\
[2] Zhang, J., Yi, Q., & Sang, J. (2022, October). Towards adversarial attack on vision-language pre-training models. In Proceedings of the 30th ACM International Conference on Multimedia (pp. 5005-5013).\
[3] Huang, H., Erfani, S. M., Li, Y., Ma, X., & Bailey, J. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP. In Forty-second International Conference on Machine Learning.

### Soundness
3

### Presentation
2

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
This paper addresses the significant vulnerability of pre-trained VLMs like CLIP to adversarial attacks. The authors identify the limitations of existing defenses, such as adversarial fine-tuning, which are often computationally expensive, require labeled data, and suffer from poor generalization. To overcome these issues, the paper introduces SCC, a novel and effective test-time defense that requires no model retraining.

### Strengths
1. The authors evaluate their approach on multiple benchmarks and conduct extensive ablation studies.

2. The experimental setup is described in a detailed and thorough manner.

### Weaknesses
1. Lack of Evaluation Against Adaptive Attacks: The current experiments rely on standard white-box attacks (PGD, CW) which are unaware of the SCC defense mechanism. A strong adversary could potentially design an adaptive attac that incorporates the entire SCC process (including the pseudo-label generation and multi-view optimization) into its own loss function to bypass the defense.

2. Although the paper makes a significant contribution to improving the adversarial robustness of CLIP and related VLMs, all experiments are conducted solely on the CLIP model. The absence of results on other CLIP-like models (e.g., OpenCLIP, EVA-CLIP) may limit the contribution of the proposed methods.

3. On the Performance vs. Efficiency Trade-off: SCC introduces an iterative optimization process at inference time. Parameters like the number of warm-up steps, optimization steps S, and the number of views L all impact both the final robustness and the inference latency.  For instance, in a time-sensitive task, what is the expected drop in robustness if optimization steps S is reduced to just one or two steps?


4. Insufficient discussion and comparison with recent test-time defense methods, e.g. [1, 2, 3]

[1] CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP, CVPR 2025.

[2] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models, CVPR 2025.

[3] On the Zero-shot Adversarial Robustness of Vision-Language Models: A Truly Zero-shot and Training-free Approach, CVPR 2025

### Questions
Please refer to the questions raised in the Weaknesses section above

### Soundness
2

### Presentation
2

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
The paper investigated two main weaknesses of CLIP adversarial attacks: lack of semantic guidance and vulnerability to view variations. To defend against VLM attacks, the paper introduces self-calibrated consistency as a test-time defense. This defense mechanism introduces semantic consistency based on soft pseudo-labels and multi-view predictions to regularize cross-modal alignment and further separate target embeddings from hard negatives. The proposed spatial consistency also enforces consistency among perturbed visual predictions and their augmented counterparts for viewpoint stabilization. The proposed self-calibrated consistency can serve as a plug-and-play defense to boost adversarial robustness without retraining VLMs. Experiments across diverse benchmarks and scenarios demonstrate the performance improvement of the proposed self-calibrated consistency method.

### Strengths
1. The motivation is well illustrated with empirical analyses across diverse datasets (see Figure 1).
2.  Experiments across different datasets show the efficacy of the proposed method in terms of clean and robust accuracy.
3. Theoretical analyses are given to justify the effectiveness of the proposed semantic consistency method. The proofs seem to be correct.

### Weaknesses
1. The paper is not organized well. The authors introduce the concept of counterattack in the Introduction (the first section), yet the formal definition of test-time counterattack is given in Section 3.1. I also find it hard to understand the definition of $\hat{z}$, is it logit?
2. The authors mentioned that the paper evaluates adaptive attacks, but not too much information is given regarding adaptive attacks. What if the adversarial attackers know the original images instead of the counter-attacked ones?
3. The robustness evaluation is primarily on PGD and CW attacks. AutoAttack [a] with both white-box and black-box attacks should be evaluated.
4. FARE evaluates robustness also across diverse downstream vision-language tasks. It seems that the proposed method can also be transferred, yet the paper focuses on image classification only.
5. The robustness evaluation is merely on low perturbation radii. Is it possible to have a stronger attack?


[a] Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks. (ICML 2020)

### Questions
1. What would happen if the VLM is under text-level attacks (e.g., Bert Attack [b]) or joint image-text attacks (e.g., Co-Attack [c])?
2. Can authors provide some visualizations of the counter-attacked images?
3. Is it possible to create an adaptive setting that includes the "counterattack" loop in the adversary generation scheme?

[b] BERT-ATTACK: Adversarial Attack Against BERT Using BERT (EMNLP-2020)
[c] Towards Adversarial Attack on Vision-Language Pre-training Models (ACMMM 2022)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper primarily focuses on defense against adversarial attacks. This paper proposes Self-Calibrated Consistency (SCC), a training-free defense that improves the adversarial robustness of CLIP-style Vision–Language Models (VLMs) by enforcing semantic consistency, using a short counterattack with multi-view aggregation to form a soft textual prototype. In the meantime, the proposed method enforces spatial consistency, optimizing a single corrective perturbation that is shared across augmented views to stabilize predictions. Experiments across both natural and medical image datasets demonstrate the improved robustness of the proposed method with a minor drop in clean accuracy. Furthermore, the additional inference cost is also lightweight.

### Strengths
The paper crisply diagnoses why test-time counterattack-style defenses fail: semantic drift toward hard negatives and view sensitivity, and thus designs self-calibrated consistency to directly counter those issues.

The proposed self-calibrated consistency method is training-free and plug-and-play.
For experiments, robustness gains are large and consistent across natural-image datasets and medical sets, while clean accuracy is essentially unchanged

Ablations and sensitivity plots are informative with a clear algorithm pipeline.

### Weaknesses
Experiments focus on zero-shot classification, while the claims about broader VLM use (retrieval, open-vocab detection/segmentation) are mentioned rather than demonstrated. I do suggest the authors explore other VLM applications in addition to classification.

Note that the hyperparameters are tuned by grid search, while practical guidance beyond the reported defaults is limited.
Attack coverage is narrow: results are mostly under PGD-10 and a small CW budget. There’s no AutoAttack, Square, or transfer/black-box evaluation despite citing those attacks.

If I understand correctly, the "multi-view" mechanism relies only on horizontal flips and low-variance Gaussian noise, which are weak augmentations. Robustness under more realistic geometric/photometric changes isn’t examined.

Medical evaluation reports only accuracy/robust accuracy. Clinically meaningful metrics (AUC, sensitivity/specificity) and per-dataset breakdown under higher budgets are missing.

### Questions
Can you report adaptive evaluations where the attacker knows self-calibrated consistency?

How sensitive is self-calibrated consistency to the assumed attack radius used for the corrective perturbation?

How much variance do you observe across seeds due to noise-based views? Error bars should be given.

Do gains transfer beyond zero-shot classification to other downstream vision-language tasks?

### Soundness
3

### Presentation
2

### Contribution
3
