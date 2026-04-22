# Combinational Backdoor Attack against Customized Text-to-Image Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 8, 2, 4

## Abstract
Recently, Text-to-Image (T2I) synthesis technology has made tremendous strides. Numerous representative T2I models have emerged and achieved promising application outcomes, such as DALL-E, Stable Diffusion, Imagen, etc. In practice, it has become increasingly popular for model developers to selectively adopt personalized pre-trained text encoders and conditional diffusion models from third-party platforms, integrating them together to build customized (personalized) T2I models. However, such an adoption approach is vulnerable to backdoor attacks. In this work, we propose a \textbf{C}ombinational \textbf{B}ackdoor \textbf{A}ttack against \textbf{C}ustomized \textbf{T2I} models (CBACT2I) targeting this application scenario. Different from previous backdoor attacks against T2I models, CBACT2I embeds the backdoor into the text encoder and the conditional diffusion model separately. The customized T2I model exhibits backdoor behaviors only when the backdoor text encoder is used in combination with the backdoor conditional diffusion model. These properties make CBACT2I more stealthy and controllable than prior backdoor attacks against T2I models. Extensive experiments demonstrate the high effectiveness of CBACT2I with different backdoor triggers and backdoor targets, the strong generality on different combinations of customized text encoders and diffusion models, as well as the high stealthiness against state-of-the-art backdoor detection methods. The code is available at: https://anonymous.4open.science/r/COM_backdoor-2404/.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes CBACT2I, a novel combinational backdoor attack targeting customized text-to-image (T2I) models. CBACT2I combines the backdoor encoder and the backdoor conditional diffusion model to build a backdoor text-to-image model, the malicious behavior emerges only when both compromised components are assembled together. This feature makes the attack stealthier and harder to detect. The paper conducted extensive experiments on various models and datasets, demonstrating the attack effectiveness and the stealthiness against existing defense mechanisms.

### Strengths
1. New attack surface: This paper introduces a novel backdoor attack in text-to-image models, considering the combinations of text encoders and conditional diffusion models.
2. High effectiveness and generality: This paper conducted comprehensive experiments to demonstrate the attack effectiveness with different backdoor triggers and backdoor targets the strong generality on different combinations of customized text encoders and diffusion models.
3. Defenses discussion: This paper conducted extensive experiments to demonstrate the attack stealthiness against existing defense mechanisms, such as ONION, T2Ishieldis and UFID.

### Weaknesses
1. The ASR for “style backdoor target” depends on a simple classifier. The style-ASR is computed via a ResNet-18 trained by the authors (98% acc.), which may introduce bias. Since GPT4o-as-a-judge is introduced in the case study in the real-world scenario, it is suggested also employ GPT4o to judge the ASR of “style backdoor target”.
2. The idea of using CBACT2I for secret information hiding is interesting. However, there is no experimental validation for the "secret hiding" application. The authors should provide some experimental results.

### Questions
1.The authors compute style-ASR with a ResNet-18 (≈98% acc.), which may bias results. Since GPT-4o is already used as a judge in your real-world case study, could authors also report GPT-4o–based ASR for the style backdoor target?
2. Could the authors provide quantitative experimental results to demonstrate the effectiveness of their approach in the secret information hiding application?
3. For the “pre-set image” backdoor attack, could the authors include additional similarity metrics (e.g., LPIPS) to more comprehensively measure attack effectiveness?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper implements a Combinational Backdoor Attack by simultaneously optimizing the text encoder and the noise denoising module in the text-to-image model.

### Strengths
This work focuses on the backdoor attack in text-to-image tasks, which is a significant security threat, and proposes a novel threat scenario: "Combinational Backdoor Attack."

### Weaknesses
***1. Unclear Threat Model***

The threat model is somewhat confusing. I understand that the authors aim to jointly tamper with two components (the text encoder and the UNet) to enhance the stealthiness of the backdoor attack. However, this setup raises several concerns:
(1) How often does such a co-usage scenario occur in real-world settings? As far as I know, on open-source platforms like CivitAI, personalized fine-tuning of text encoders is rare; most community models focus on VAE or UNet modifications (please correct me if I am mistaken).
(2) Why improving stealthiness necessarily requires backdooring multiple components? This approach seems more like an application of existing methods in a new setting rather than a fundamentally new methodological contribution.

***2. High Similarity to Related Works***

**The proposed method appears overly similar to prior works [1,2].** In particular, Section 4.3 is highly similar to [1] (see Eq. (1) in both papers), and Section 4.4 is highly similar to [2] (see Eq. (4) here with Eq. (7) in [2]). **Given these overlaps, the novelty of the contribution is questionable.**

***3. Insufficient Evaluation Against Backdoor Defenses***

The experimental evaluation does not include comparisons with recent text-to-image backdoor defense methods[3,4,5], which are essential to validate the claimed stealthiness.


[1] Struppek L, Hintersdorf D, Kersting K. Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis[J]. arXiv preprint arXiv:2211.02408, 2022.

[2] Zhai S, Dong Y, Shen Q, et al. Text-to-image diffusion models can be easily backdoored through multimodal data poisoning[C]//Proceedings of the 31st ACM International Conference on Multimedia. 2023: 1577-1587.

[3] Wang Z, Zhang J, Shan S, et al. Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models[J]. arXiv preprint arXiv:2504.20518, 2025.

[4] Zhai S, Li J, Liu Y, et al. Efficient Backdoor Detection on Text-to-image Synthesis via Neuron Activation Variation[C]//ICLR 2025 Workshop on Foundation Models in the Wild.

[5] Xu Y, Zhong N, Li G, et al. Fine-grained Prompt Screening: Defending Against Backdoor Attack on Text-to-Image Diffusion Models[J].

### Questions
While Eq. (4) seems to be designed only for generating “specific images” backdoor, it remains unclear how the method achieves “specific styles” backdoor as claimed in Section 4.5. 

Do authors utilize the loss function of Eq. (4) to obtain a “specific styles” backdoor?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a new security vulnerability in customized text-to-image pipelines, where users mix pretrained text encoders and diffusion models. The authors propose CBACT2I, a combinational backdoor that injects separate triggers into the encoder and decoder: each component appears benign on its own, but together they activate malicious outputs when a triggered prompt is used. The attack preserves normal functionality, works across different encoder–decoder combinations, and evades existing defenses. Experiments show high attack success with strong stealthiness. The work is well-motivated and reveals an overlooked, practical threat in modular T2I model development.

### Strengths
1. This paper proposes a novel attack, where the backdoor can only be triggered when the text encoder matches the diffusion model.

2. The experiments are both sound and comprehensive, which demonstrates the effectiveness of the proposed method as well as its robustness.

3. The proposed method is straightforward, simple, and effective.

4. Good writing, easy to follow.

### Weaknesses
1. **Scope of generalization.** Experiments focus on a few open-source diffusion models, and all of them are variants of stable diffusion model family; transferability to other architectures, tokenizers, or deployed commercial stacks (closed-source encoders/decoders) is not shown. I therefore recommend more experiments on different text encoders and diffusion models, including the newest SD models and the earliest LDM, whose text encoder is based on BERT.

2. **Limited defense evaluation.** Only a few detectors (ONION, T2IShield, UFID) are evaluated; the paper lacks study against preprocessing (normalization), model-editing defenses, or newer detection methods tailored for modular pipelines. Moreover, I also suggest that the author examine how the fine-tuning would affect the injected backdoor.

3. **Confusing attack significance.** The proposed combinational design indeed improves stealth, but it also substantially reduces the probability of accidental triggering: only a very specific combination of a poisoned encoder, a poisoned text encoder, a backdoored diffusion model, and a triggered prompt will activate the backdoor. This raises important questions that the paper does not sufficiently justify: *What's the point of conducting such an attack?* and, as it seems only the attack can trigger the backdoor, *Why does the attacker need to attack himself?*, and as a result, *What real-world significance does this attack actually have?*  Particularly, I do not fully agree with the author on the discussion in Sec 6, where the attack targets in the real-world scenario were described as producing bias, harmful, and advertisement contents. Since the attacker has full access to the original model, they can always obtain an exclusive model for these malicious tasks by just fine-tuning it. As for the positive phase, I also doubt that the proposed method can have any advantage over existing watermarking methods or backdoor methods.

### Questions
see in weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
