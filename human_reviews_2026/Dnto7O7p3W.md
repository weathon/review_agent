# Evading Protections Against Unauthorized Data Usage via Limited Fine-tuning

- Decision: Reject
- Scores: 4, 6, 2, 4, 2

## Abstract
Text-to-image diffusion models, such as Stable Diffusion, have demonstrated exceptional potential in generating high-quality images. However, recent studies highlight concerns about the use of unauthorized data in training these models, which can lead to intellectual property infringement or privacy violations. A promising approach to mitigating these issues is to embed a signature in the model that can be detected or verified from its generated images. Existing works also aim to fully prevent training on protected images by degrading generation quality, achieved by injecting adversarial perturbations onto training data. In this paper, we propose RATTAN, which effectively evades such protection methods by removing the protective perturbations from images and catastrophically forgetting such learned features in a model. It leverages the diffusion process for controlled image generation on the protected input, preserving high-level features while ignoring the low-level details utilized by the embedded pattern. A small number of our generated images (e.g., 10) are then used to fine-tune marked models to remove the learned features. Our experiments on four datasets, two different IP protection methods, and 300 text-to-image diffusion models reveal that while some protections already suffer from weak memorization, RATTAN can reliably bypass stronger defenses, exposing fundamental limitations of current protections and highlighting the need for stronger defenses.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces RATTAN, a framework that bypasses protections designed to prevent unauthorized data use in text-to-image diffusion models. Existing defenses like Anti-DreamBooth and DIAGNOSIS embed imperceptible adversarial patterns or “semantic coatings” into training images to prevent a model from using the data or detect whether a model has used protected data. RATTAN leverages the diffusion process itself to regenerate protected images, retaining their high-level semantic features while removing low-level protection patterns. Additionally, RATTAN shows it only needs around 10 images for fine-tuning the diffusion model to effectively remove the embedded pattern.

### Strengths
1. Technical Simplicity and Effectiveness:  Uses diffusion’s own reconstruction process to remove embedded patterns, requiring minimal computation and data.
2. Comprehensive Evaluation: RATTEN is tested on multiple datasets, models, and protection schemes and show effectiveness among them.
3. Writing is clear and easy to understand.
4. Sufficient ablations to support the arguments and experiment choices.

### Weaknesses
1. **Novelty Concern:**  
   The core of the proposed method involves using an SDEdit-like process on protected images, which resembles existing purification methods such as DiffPure and Regeneration Attack. Could the authors elaborate on the technical differences, especially compared to DiffPure? Also, does RATTAN’s advantage over DiffPure primarily stem from using a stronger, pretrained diffusion model as the purifier?

2. **Lack of Theoretical Analysis:**  
   RATTAN provides strong empirical results; however, theoretical analysis is absent. Is there any justification or intuition for why RATTAN can effectively guarantee protection removal? DiffPure provides a theoretical bound for its SDE process — can RATTAN offer a tighter bound or a similar theoretical analysis?

3. **Baselines:**  
   For non-traceable IP protection, the results are mainly based on Anti-DreamBooth. However, Anti-DreamBooth is not the current state-of-the-art protection method. I encourage the authors to compare RATTAN with more recent and advanced protections, such as Mist v2 or Glaze.

4. **Loss of Details:**

    As the author stated, RATTAN would remove the low-level details in the image (Fig.3) . However, low-level details can matter in some cases, especially for IP-related personalization. Can the author provide quantified analysis about the loss of visual details, for example, report PSNR/SSIM and compare it to baselines?


My major concern lies in the technical similarity between RATTAN and DiffPure, while there is very little discussion or direct comparison. Although the paper includes empirical visual and numerical comparisons, it remains unclear why RATTAN outperforms DiffPure by such a large margin. This raises a fundamental question about the paper’s positioning:  
does RATTAN propose a novel purification method, or is it rather an enhanced adaptation to new scenarios.

Mist v2: https://arxiv.org/abs/2310.04687 

Glaze: https://nightshade.cs.uchicago.edu/ | https://arxiv.org/abs/2302.04222

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method to evade IP protection mechanisms for diffusion models, such as DIAGNOSIS and Anti-DreamBooth. The approach combines controlled image regeneration with model fine-tuning and aims to remove protection signatures while preserving the semantic content and visual quality of the generated images.

### Strengths
1.The method achieves competitive results and clearly demonstrates the ability to reduce the detectability of protected models under various settings.

2.The approach is model- and data-agnostic, which makes it relatively easy to apply in practice.

### Weaknesses
1.The DIAGNOSIS evaluation lacks a detailed comparison of image quality metrics between the proposed method and the baselines. Although the authors provide an explanation, including additional metrics would offer a more comprehensive and objective assessment of whether the method preserves generative quality while evading detection.

2.The paper could provide a deeper analysis or discussion of the mechanism underlying the method’s effectiveness, particularly explaining how the controlled regeneration and fine-tuning steps remove protection patterns. Additional interpretability analyses would further clarify this point.

3.The paper should also evaluate the proposed method against more recent protection and verification approaches, such as DiffusionShield [1], FT-Shield [2], and the verification framework in [3], to better demonstrate its robustness and effectiveness.

[1] DiffusionShield: A Watermark for Data Copyright Protection Against Generative Diffusion Models.

[2] FT-Shield: A Watermark Against Unauthorized Fine-Tuning in Text-to-Image Diffusion Models.

[3] Towards Reliable Verification of Unauthorized Data Usage in Personalized Text-to-Image Diffusion Models.

### Questions
Please refer to the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper shows that existing protection methods based on adversarial attacks for unauthorized data usage in text-to-image diffusion models can be circumvented through a two-stage process: (1) controlled image generation that preserves high-level features while removing low-level protective patterns, and (2) fine-tuning the protected model on a small set (as few as 10) of these regenerated images.

### Strengths
1, The delivery is intuitive and clear.

### Weaknesses
1, The motivation of this work is unclear. While [A] has indicated the unreliability of attack-based protection, there is no need to present even stronger evasion for this protection. Focus should be put on better IP protection methods.

2, The baseline choice is questionable. According to [A], the most robust protection under its evasion should be Mist [B]. And the strongest attack-based protection is ACE [C] currently. However, these two protection methods are not even mentioned in the reference.

3, The performance in evading Anti-DB is not persuasive enough. Rattan still suffers performance degradation compared to original.

Reference:

[A] Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI

[B] Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models via Adversarial Examples

[C] Targeted Attack Improves Protection against Unauthorized Diffusion Customization

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

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
This paper targets copyright-protection methods that embed imperceptible perturbations into images so that any model trained on them can later be detected. The authors propose RATTAN, a method that “washes” a marked (i.e., backdoored) model so it no longer reveals unauthorized data use. The attack assumes the adversary can obtain a small set of perturbed samples with captions, and has full access to the marked model and a strong clean diffusion model. By applying partial noising to each protected image and then using text-guided denoising, the attacker regenerates proxy images that preserve high-level semantics but remove the protection signal, and then lightly fine-tunes the marked model on these proxy pairs. Under these assumptions, the paper shows that as few as 5–10 seed samples can eliminate detection signatures and evade attribution defenses.

### Strengths
- The problem of copyright protection and unauthorized data use in generative models is timely and important.
- The paper is clearly written and well structured; the proposed method is conceptually simple and easy to understand.
- The experiments are fairly comprehensive, including ablations on key hyperparameters and components.

### Weaknesses
* **(Main weakness) The method feels trivial and of limited applicability because it relies on strong assumptions.** I usually appreciate simple but elegant methods, but here the paper fundamentally answers this question (I'll use backdoor to represent the perturbation for simplicity): **if I already have a set of ground-truth backdoor examples (i.e., I basically know what the backdoored data look like) and I have a (strong) clean auxiliary model, how can I purify a backdoored model?** The solution is straightforward: use the clean model to obtain correct labels(i.e., images here) and then fine-tune/unlearn on the backdoored model.

  People usually discuss two distinct challenges: **(1) data filtering** (i.e., how to find and remove a few poisoning samples in a huge dataset, possibly with a clean model), and **(2) post-training purification** (i.e., I don't have the knowledge of a clean model, but I can access a small set of clean/mixed/poisoned examples and need to purify my backdoored model. **This paper, however, assumes full access to both ground-truth backdoor data and the white-box access to the backdoored model, plus a strong clean proxy model.** That is a much stronger adversary than the two canonical settings above. Therefore, the authors need to:
    * Provide justification for the assumed threat model and clarify the realistic scenarios.
    * For the “access to a set of backdoor data” assumption (even if only 5–10 seeds), run ablations for more realistic data knowledge. For example: if the dataset has N samples and only 1%/3%/5%/10% are poisoned, what happens if RATTAN applies Controlled Image Generation to all images? Does the overall utility drop substantially?
    * For the “(strong) auxiliary model” assumption, Table 3 has an ablation but the SD variants tested are all fairly strong. Evaluations on weaker models (e.g., pre-SD Latent Diffusion, GLIDE-small) to show how attack success degrades with weaker generators would be more informative.

* Only two baselines are directly related to specific watermark removal tasks; DiffPure (2022) looks out of date. There exist more recent backdoor-removal and copyright-infringement-attack/defense works that should be included in baselines and discussed the related-work part.

### Questions
In Table 1 (Naruto), the protection method only achieves 70% detection rate, and in Figure 11 (column 4), even the “triggered” images do not appear to contain any copyright-relevant visual content. Can the protection methods still flag a model as “copyright-violating” when the generated outputs show no visible copyrighted content? A randomness measure (e.g., variance across sampling runs, prompts, or seeds) would help show how stable the detection performance is.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how to bypass existing IP protection and provenance-tracking methods for text-to-image diffusion models. The authors propose a method called RATTAN, which uses a two-stage process: (1) partially noising and regenerating protected images with a pretrained diffusion model to remove protective signatures, and (2) fine-tuning the target model with a small set of these regenerated samples to erase “protection-related” features. Experiments on multiple datasets and protection schemes (including DIAGNOSIS and Anti-DreamBooth) show that RATTAN can reduce detection accuracy to nearly 0% while maintaining image quality metrics such as FID.

### Strengths
- The topic is highly relevant and timely. Evaluating the real-world robustness of IP protection and tracing methods for T2I diffusion models is an important problem.
- The method design is overall intuitive and technically coherent. The idea of using partial diffusion regeneration followed by few-shot fine-tuning makes sense conceptually.
- The writing is generally clear and easy to follow.

### Weaknesses
- The scope of the paper is confusing. Although the abstract and introduction claim to target both perturbation-based and watermark-based protections, the organization is inconsistent. For instance, Figure 1 seems entirely focused on watermark-based defense, while the threat model states “The defender can detect whether a model has been trained on unauthorized data by inspecting its generated images,” which applies only to watermark-based schemes, not perturbation-based ones like Anti-DreamBooth and Glaze. Perturbation-based methods aim to disrupt training itself rather than post-hoc detection, so the paper’s unified framing is conceptually shaky.
- The idea itself is quite simple. It is essentially a two-step process similar to SDEdit or DiffPure: add noise and regenerate. The only difference is the detailed settings (e.g., hyperparameters and the prompt settings), as well as the subsequent fine-tuning. In essence, the method relies on the pretrained diffusion model’s own generative prior to "repaint" the protected images. For perturbation-based protections like Glaze, where the perturbation aligns an artwork’s latent space with an unrelated one, how can RATTAN ensure that the regenerated images preserve the true artistic style rather than hallucinating one from the pretrained model’s prior knowledge? This is especially problematic for personalized styles unseen during pretraining (which is also the main target of personalized generation). Furthermore, RATTAN seems quite similar in principle to DiffPure and Regen, so why do those baselines perform much worse? The paper should provide more analysis on what exactly accounts for the performance gap.
- Many statements and baselines appear outdated. The authors claim that “MIAs are less effective for T2I diffusion models (Duan et al., 2023),” but more recent work has shown strong MIAs even for fine-tuned diffusion models, with accuracy close to 100% [1,2]. The baselines are also weak. Only DIAGNOSIS is considered for verification-based defenses. Although some additional methods are tested in the appendix, they are said to be ineffective on diffusion even without any modification, which doesn’t meaningfully validate RATTAN’s superiority. More recent watermarking defenses (e.g., FT-Shield [3], SIREN [4], z-Watermark [5]) and stronger perturbation-based defenses (e.g., Mistv2 [6]) should be included. Similarly, all experiments are limited to SD v1.4 and LoRA fine-tuning. It would be important to test whether RATTAN still works for stronger models such as SDXL and more advanced fine-tuning paradigms.
- The degradation of image quality introduced by RATTAN might be underestimated. From Figures 3 and 4, the regenerated samples often lose fine-grained semantic details (e.g., Pokémon eyes, teeth, and textures). For applications sensitive to stylistic precision and detail fidelity, this trade-off could be unacceptable. Since the paper’s evidence is purely empirical, it is unclear how to balance protection removal against generation quality, or how robustly this trade-off generalizes.


Ref:

[1]: Zhai, S., Chen, H., Dong, Y., Li, J., Shen, Q., Gao, Y., ... & Liu, Y. (2024). Membership inference on text-to-image diffusion models via conditional likelihood discrepancy. Advances in Neural Information Processing Systems, 37, 74122-74146.

[2]: Wu, X., Hua, Y., Liang, C., Zhang, J., Wang, H., Song, T., & Guan, H. (2024, June). Cgi-dm: Digital copyright authentication for diffusion models via contrasting gradient inversion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 10812-10821). IEEE Computer Society.

[3]: Cui, Y., Ren, J., Lin, Y., Xu, H., He, P., Xing, Y., ... & Tang, J. (2025). Ft-shield: A watermark against unauthorized fine-tuning in text-to-image diffusion models. ACM SIGKDD Explorations Newsletter, 26(2), 76-88.

[4]: Li, B., Wei, Y., Fu, Y., Wang, Z., Li, Y., Zhang, J., ... & Zhang, T. (2025, May). Towards reliable verification of unauthorized data usage in personalized text-to-image diffusion models. In 2025 IEEE Symposium on Security and Privacy (SP) (pp. 2564-2582). IEEE.

[5]: Huang, J., Guo, Z., Luo, G., Qian, Z., Li, S., & Zhang, X. (2024). Disentangled Style Domain for Implicit $z$-Watermark Towards Copyright Protection. Advances in Neural Information Processing Systems, 37, 55810-55830.

[6]: Zheng, B., Liang, C., & Wu, X. (2023). Targeted Attack Improves Protection against Unauthorized Diffusion Customization. ICLR 2025.

### Questions
- The paper mentions “300 text-to-image diffusion models.” in the abstract. How is this number obtained? The experiments appear not to involve so many models?

- In Table 3, detection accuracy increases again when using all samples instead of a small subset. Why does this happen?

### Soundness
3

### Presentation
3

### Contribution
2
