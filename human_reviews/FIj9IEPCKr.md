# Proxy Denoising for Source-Free Domain Adaptation

- Decision: Accept (Oral)
- Scores: 8, 8, 8, 8

## Abstract
Source-Free Domain Adaptation (SFDA) aims to adapt a pre-trained source model to an unlabeled target domain with no access to the source data. Inspired by the success of large Vision-Language (ViL) models in many applications, the latest research has validated ViL's benefit for SFDA by using their predictions as pseudo supervision. However, we observe that ViL's supervision could be noisy and inaccurate at an unknown rate, potentially introducing additional negative effects during adaption. To address this thus-far ignored challenge, we introduce a novel Proxy Denoising (__ProDe__) approach. The key idea is to leverage the ViL model as a proxy to facilitate the adaptation process towards the latent domain-invariant space. Concretely, we design a proxy denoising mechanism to correct ViL's predictions. This is grounded on a proxy confidence theory that models the dynamic effect of proxy's divergence against the domain-invariant space during adaptation. To capitalize the corrected proxy, we further derive a mutual knowledge distilling regularization. Extensive experiments show that ProDe significantly outperforms the current state-of-the-art alternatives under both conventional closed-set setting and the more challenging open-set, partial-set, generalized SFDA, multi-target, multi-source, and test-time settings. Our code and data are available at https://github.com/tntek/source-free-domain-adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Proxy Denoising (ProDe) to improve Source-Free Domain Adaptation (SFDA) by addressing noisy predictions from Vision-Language (ViL) models. The authors propose a proxy denoising mechanism based on proxy confidence theory to correct these noisy predictions and guide adaptation toward a domain-invariant space. They further enhance this process with mutual knowledge distillation regularization. Experiments demonstrate that ProDe outperforms existing methods across various SFDA settings, including closed-set, open-set, partial-set, and generalized scenarios.

### Strengths
*	The paper is well-written, well-organization, and easy to follow.
*	The insight on rectifies the inaccurate predictions of ViL models significantly contributes to SFDA settings.
*	The paper introduces a novel ProDe method, which effectively corrects ViL model predictions through the use of a proxy confidence theory, offering a reliable approach to prediction refinement.
*	The mutual knowledge distillation regularization is a strong addition, enabling the model to capitalize on refined proxy predictions with improved efficiency.
*	The authors evaluate the proposed method through extensive experiments, including challenging partial-set, open-set, and generalized SFDA settings, demonstrating the versatility of the method.

### Weaknesses
*	Overall, this paper is of high quality, with clear motivation and novel insights. The proposed ProDe method also demonstrates strong performance in the SFDA scenario. It would be interesting to see additional results in a similar scenario, such as test-time domain adaptation.

### Questions
*	Could the authors provide more detailed descriptions for Fig. 1? Additional explanations would help readers better understand the key ideas of the paper.
---
I would consider increasing my score if the authors address the concerns raised by me and other reviewers.

### Soundness
3

### Presentation
4

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
The authors tackle Source-Free Domain Adaptation (SFDA), where a pre-trained model adapts to an unlabeled target domain without access to source data. While Vision-Language (ViL) models show potential for SFDA, they often generate noisy predictions, an issue the authors investigate in this context. To address it, they propose Proxy Denoising (ProDe), a novel method that leverages proxy confidence theory to correct the ViL model’s predictions and introduces mutual knowledge distillation to make better use of these refined predictions. Extensive experiments on standard benchmarks demonstrate that ProDe outperforms prior methods across conventional closed-set, as well as partial-set, open-set, and generalized SFDA settings. The authors intend to release their code.

### Strengths
The paper effectively addresses the important problem of source-free domain adaptation (SFDA), where models must adapt to new target domains without access to labeled source data—an increasingly relevant setup in practical scenarios where source data may be proprietary or sensitive. Demonstrating state-of-the-art performance on standard SFDA benchmarks, the proposed method showcases its robustness and potential impact in the field. The authors employ mutual knowledge distillation to synchronize knowledge, reducing noise and enabling reliable knowledge transfer, and they incorporate category balance regularization to prevent “category collapse,” ensuring balanced treatment of each class. Through extensive analysis, including feature distribution visualizations and thorough ablation studies, the paper provides deep insights into model behavior and validates the effectiveness of each component. Furthermore, the paper is well-documented, with a clear presentation of the experimental setup, benchmark datasets, and evaluation metrics, enhancing reproducibility and accessibility.

### Weaknesses
### Ambiguity and Misleading Terminology in Domain Invariance Claims:
The authors claim that their method moves toward a “domain invariant space”  $D_v$ starting from  $D_t$ , but this terminology is misleading and theoretically problematic. If a domain-invariant space were achievable, there would be no need for adaptation across other domains, as all target domains would align seamlessly with this invariant space. However, the results suggest that domain shifts still impact the model and hence there is a need for adaptation for every domain, which contradicts the premise of invariance. For example, if the model were genuinely invariant after training on a source domain (e.g., Ar), it should perform equally well across all other domains (e.g., Cl, Pr, Rw) without further adaptation. This contradiction suggests that  $D_v$  is not genuinely invariant, but rather biased towards the target domain.

It would be beneficial if the authors could redefine the term “domain invariant space.” They might consider an alternative term, such as “target-aligned space,” which more accurately reflects the observed need for per-domain adaptations.

###	Unsupported Assumption Regarding  $e_{VI}$  and Invariant Space Error:
On line 164, the authors assert that  $e_{VI}$  could be ignored, implying that the error between the vision-language model’s space and the purported domain-invariant space is negligible. This assumption is dubious without further justification. The vision-language model’s embedding space may indeed diverge from the so-called invariant space, leading to substantial misalignment and error. Ignoring  $e_{VI}$  risks undermining the model’s robustness in handling domain shifts.

I suggest the authors provide empirical evidence for dismissing  $e_{VI}$ and validation of $d_I^0 \approx d_V^0 \gg e_{VI}$ by conducting experiments across a range of scenarios like different domain adaptation settings.

###	Oracle Configuration:
There is a flaw in the Oracle experiment setup, particularly in the Cl-to-Ar scenario (Line 410). The Oracle is incorrectly trained on the source domain (Cl) rather than the target domain (Ar). If this was a typo, I recommend correcting it. Otherwise, if this is intentional, it would be helpful to clarify the rationale for training on the source domain and claiming it to be an oracle.

###	Over-Reliance on a Single Vision-Language Model (CLIP):
The authors rely solely on CLIP for their experiments, neglecting to evaluate the method’s effectiveness with other vision-language models (e.g., LLaVA, Llama). This limitation raises concerns about the generalizability of the approach. Vision-language models have varied architectures and domain alignment properties, and the performance may vary significantly across models. Without results on other models, it is unclear if the proposed method is tailored specifically to CLIP or if it can generalize to other ViL models.

I suggest that the authors expand their experiments to include other vision-language models, such as LLaVA and Llama. Reporting these results would provide valuable insights into the generalizability of ViL models. 

###	Generalization Claims to SFDA Settings:
The authors claim that their approach can generalize to broader SFDA settings, yet they do not provide any insights, or experimental results for critical scenarios like source-free multi-target domain adaptation (SF-MTDA [1])  and source-free multi-source domain adaptation (SF-MSDA [2]). 

To enhance the rigor of their claims, I recommend that the authors either include experimental results for SF-MTDA and SF-MSDA scenarios or provide a detailed discussion on potential limitations or adaptations necessary for these settings. This addition would clarify the scope and limitations of the method’s applicability.


### References

[1] Kumar, Vikash, et al. "Conmix for source-free single and multi-target domain adaptation." Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 2023.

[2] Ahmed,Miraj, et al. "Unsupervised Multi-Source Domain Adaptation Without Access to Source Data."  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

### Questions
It would be great if the authors could clarify the doubts above especially those related to the theory and domain invariance space.

# UPDATE (After Discussion Period):
The discussion addresses all my concerns, and I appreciate the detailed responses and clarifications provided. Consequently, **I am increasing my ratings**. The responses demonstrate a strong understanding of the core issues, effectively addressing ambiguity, unsupported assumptions, and the generalization of claims. The inclusion of additional experiments, such as evaluations with OpenCLIP and broader SFDA settings, further solidifies the robustness and adaptability of the proposed method. The effort to provide comprehensive results, insightful explanations, and necessary corrections is commendable. Thank you for your diligence and thoroughness in addressing these points.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The previous methods using ViL for pseudo-supervision can generate noise, which introduces negative effects that have been overlooked. This paper proposes a ProDe method that first introduces a proxy confidence theory, which vividly analyzes and explains the sources of noise in ViL predictions, and specifically designs a denoising mechanism to correct ViL's predictions. Additionally, it introduces a mutual information extraction method to achieve knowledge synchronization between the ViL model and the target model.

### Strengths
The paper is based on proxy confidence theory and designs a reliable denoising algorithm to reduce the prediction noise of ViL, addressing an important issue that has been neglected in the use of ViL for pseudo-supervision. This may facilitate subsequent related work.

### Weaknesses
Although the ViL model is obtained based on a large dataset, for specific source and target domains, the ViL model can approximate the domain-invariant space. The validity of this assumption requires further theoretical support.

### Questions
I would like to know if using ViL for pseudo-supervision is suboptimal when the distance between the source domain space D_S (or training dataset D_Tt) and D_I is closer than the distance between D_V and D_I. Additionally, the formulas (5) and (6) in the paper contain many hyperparameters; is tuning these hyperparameters a challenge?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses Source-Free Domain Adaptation (SFDA) in terms of utilizing Vision-Language Models (VLMs) for supervision. Specifically, the authors argue that prior works that utilize VLMs for SFDA treat their predictions as the ground truth without considering the potential noise in their predictions. To alleviate this issue, the authors propose Proxy Denoising (ProDe)—an SFDA framework that corrects the supervisory VLM’s predictions before target adaptation. Extensive experiments on various Domain Adaptation benchmarks demonstrate the effectiveness of the approach compared to prior works in multiple domain adaptation settings. Moreover, analysis experiments substantiate the intuition behind the proposed method.

### Strengths
- **Presentation-** The paper is written well overall and conveys the central ideas of the work quite effectively. The paper is easy to follow and understand. Additionally, the authors have presented experiments on a wide range of benchmarks and settings.

- **Novelty-** The authors claim that the paper is the first to analyze the inaccurate predictions of the teacher VLM in the context of SFDA and propose a method to alleviate the same.

- **Results-** The authors present extensive experiments across several domain adaptation benchmarks and comparisons with several prior works that do and do not use VLMs for training (although whether some of the comparisons are fair is a question, see Weaknesses for details).

### Weaknesses
### (a) Concerns with Proxy Confidence Theory
- Theorem 1 provides a relation between the confidence of the VLM’s predictions and the confidence of the source model and the current training model. This is based on the approximation of the VLM’s predictions to a Gaussian distribution and further expressing this in terms of the confidence of the VLM’s predictions.
- However, the intuition behind how a Gaussian distribution is considered for the VLM’s predictions is not entirely clear. Moreover, the conversion in Eq. 2 also seems unclear, in terms of how the conversion is possible. Essentially, it would be better if the authors could explain L185-188 in more detail.

### (b) Fairness of comparisons
- The authors present comparisons of their proposed method ProDe with prior SFDA works and with works utilizing VLMs. Based on the implementation details provided in the supplementary, it appears that the prior SFDA works that do not utilize VLMs make use of ResNet-50 or ResNet-101 depending on the difficulty of the dataset.
- Can these comparisons of ProDe with prior SFDA works be considered fair? ProDe uses supervision from a VLM that has been pre-trained on WiT-400M while the SFDA works consider an ImageNet pre-trained ResNet-50 or ResNet-101. There is a massive difference in the models being used for adaptation. Although the student model is a vision-only backbone in ProDe, it is supervised by a VLM during target adaptation.
- The authors need to discuss these differences in the experiment settings to provide a more complete picture of the results. Additionally, the authors should present the comparisons in a fair setting, i.e., similar supervisory signals or backbones should be used in both the proposed method and the prior works.

### Questions
- How does the proposed method ProDe perform in a multi-source or multi-target domain adaptation setting? Can the authors present these results on OfficeHome or DomainNet?
- Why have the authors used DomainNet-126 rather than the full DomainNet dataset? Given that DomainNet is the most challenging domain adaptation benchmark among the chosen datasets, it is an important result.
- Following the discussion on fair comparisons in the previous section, can the authors present results in a fair setting? Eg. the authors can present results of prior SFDA works by using the CLIP vision encoder as the initialization rather than the ImageNet pre-trained backbones. Another possibility could be a baseline that provides VLM supervision to prior SFDA works.
- Additionally, if the above is not possible, could the authors present results with ViTs rather than ResNet-50 / ResNet-101 for a more fair comparison? Given that the results with VLMs use ViT-B, the prior works should use a backbone with a similar capacity.

### Soundness
3

### Presentation
3

### Contribution
3
