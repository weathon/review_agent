# Empirical Robustness of Pixel Diffusion Undermines Adversarial Perturbation as Protection against Diffusion-based Mimicry

- Decision: Reject
- Scores: 4, 2, 8, 4

## Abstract
Diffusion models have demonstrated impressive abilities in image editing and imitation, raising growing concerns about the protection of private property. A common defense strategy is to apply adversarial perturbations that can mislead a diffusion model into generating bad-quality images. However, existing research has almost entirely focused on latent diffusion models while overlooking pixel-space diffusion models. Through extensive experiments, we show that nearly all attacks designed for latent diffusion models, as well as adaptive attacks aimed at pixel-space diffusion models, fail to compromise the latter. Our analysis suggests that the weakness of latent diffusion models arises mainly from their encoder, whereas pixel-space diffusion models exhibit strong empirical robustness to adversarial perturbations. We further demonstrate that pixel-space diffusion models can serve as an effective purifier by removing adversarial patterns generated for latent diffusion models and preserving image integrity, which in turn allows them to bypass most existing protection schemes. These findings challenge the assumption that adversarial perturbations provide reliable protection for diffusion models and call for a reevaluation of their role as a protection mechanism.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines existing methods that protect images from unauthorized editing by generative diffusion models through adversarial perturbations. Based on extensive experiments, the authors find that these protection mechanisms are largely effective only against Latent Diffusion Models (LDMs), while proving ineffective against Pixel-space Diffusion Models (PDMs). Their empirical analysis attributes this robustness gap to the encoder in LDMs, which amplifies perturbations, whereas PDMs perform denoising directly in pixel space, preserving a higher distributional overlap. Leveraging this observation, the authors propose PDM-Pure, a simple purification framework built upon strong PDMs (implemented via SDEdit with DeepFloyd-IF). This demonstrates that it can effectively remove protective adversarial perturbations and bypass existing defense methods.

### Strengths
- The proposed research gap is potentially impactful, urging the community to rethink the image protection manners against Pixel-space
Diffusion Models (PDMs).
- Clear empirical separation between LDMs and PDMs. The paper evaluates multiple architectures (U-Net and DiT), datasets/resolutions, and budgets. Consistently, attacks devastate LDMs but have a negligible effect on PDMs. The breadth of models/perturbations tested strengthens the central empirical claim.
- The paper proposes a simple, practical purifier with strong results. PDM-Pure is appealingly minimal—SDEdit with a strong PDM—and yet defeats several state-of-the-art protections. Showing cross-task recovery (inpainting, textual inversion, LoRA) is particularly compelling for the security threat narrative (i.e., protections are not merely degraded; they can be bypassed).
- Writing and presentation. The exposition is clear; the figures communicate well; the listed attack variants and experimental knobs are presented in reasonable detail.

### Weaknesses
- **Empirical robustness $\neq$ robustness**. The paper frames PDM robustness almost as a property of the pixel space itself. But the evidence is empirical and limited to a set of popular PDMs, datasets, and attack templates. Robustness guarantees are not established; the Lipschitz bound sketch is suggestive but not a model-tight, formal theoretical guarantee.
- **Attack surface is slightly narrow.** While the authors implement many known ingredients (PGD/SDS/EoT/feature-layer attacks), several ideas are not explored, for example: Non-$\ell_\infty$ budgets (e.g., spectral/patch/structure aware constraints) that are still imperceptible but couple to denoiser receptive fields.
- More Experiments can be added to further strengthen the claim: (1) Different noise budgets and norms: Beyond $\ell_\infty$ up to 16/255;
visibility-matched $\ell_2$; structured perturbations. (2) PDM denoiser feature analysis against attacks:  Analysis of the statistical characteristics of the latent space of the PDM’s denoiser against different attack methods is suggested to further justify the robustness. (3) Human studies: Perceptual integrity and style preservation after purification (especially for artists).
- **Security framing is slightly narrow.** Even if adversarial-noise protections erode after the proposed PDM-Pure, alternate defenses (e.g., robust watermarking/fingerprints) remain. The paper would be stronger by situating results within a broader countermeasure, clarifying what is and isn’t undermined. For example, can your PDM-Pure remove the watermark proposed in [1]? 

### Minor:
- **Slight overgeneralization in the conclusions.** Statements like “no existing attacks have proven effective in attacking PDMs, which means no protection can be achieved by fooling a PDM-based image editor” on PDMs read stronger than the evidence warrants.
The experiments are solid but cannot support universality claims.

[1] Wen, Yuxin, et al. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust, NeurIPS 2023

### Questions
- According to the paper [1], The Attack 7 is to craft the adversarial perturbation in a pre-trained VAE latent space to create perceptual subtle perturbation and the attacking loss is defined by the feature distance extracted by the PDM’s denoising network. But in the Appendix F you stated that the feature is extracted by the middle-layer of VAE. Did you implement their method correctly? Could you provide the pseudocode and parameters of your implementation for clarification? 
- The attack budget is defined as pixel-wise difference, do you think there are other perceptual metrics can be the budget to stress test the PDMs? For example, maybe when the pixel-wise difference is high, the perceptual loss is still at a reasonable level, and the attack might still be successful. How will you justified that your empirical robustness is not realized by relative low attack budget tested?

[1] Shih et. al. Pixel Is Not a Barrier: An Effective Evasion Attack for Pixel-Domain Diffusion Models, AAAI 2025

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on an important gap on adversarial robustness of pixel-space diffusion models (PDMs). Empirical results show that successful adversarial attacks on Latent Diffusion Models (LDMs) cannot work well on PDMs. The authors also provide some theoretical analysis. A simple defense technique has also been proposed to bypass existing protecting schemes.

### Strengths
1. This paper highlights that PDMs exhibit notable robustness against adversarial attacks, in contrast to LDMs. The authors support this claim with both empirical results and theoretical analysis, providing convincing evidence for their findings.
2. The paper introduces a reasonable and effective technique to bypass perturbation-based protection schemes. The proposed approach is well-motivated and demonstrates practical utility in overcoming existing defenses.

### Weaknesses
1. Simple Method. The proposed approach is straightforward; however, it does not introduce a novel or effective attack method for PDMs. Additionally, the contribution of PDM-Pure appears limited. Given the significant findings presented in the paper, it would be valuable for the authors to explore ways to enhance the adversarial robustness of LDMs, such as by modifying their architectures or tuning their parameters.
2. Lack of In-depth Theoretical Analysis. From Lines 313 to 315, the authors primarily incorporate previous empirical results into their theoretical analysis, which limits the rigor of the theoretical framework. To strengthen the conclusions presented in Line 320, the authors are encouraged to provide more detailed and comprehensive theoretical results.
3. Lack of Important Baseline. The paper only presents a few bad cases for Diffpure in Figure 17, which is insufficient for a comprehensive evaluation. Moreover, detailed results for Diffpure are missing from Table 3. 
4. Minor proofreading issues. 
    (1) Inappropriate citation format: The citation formats in Line 313 and Line 361 should be checked, particularly the usage of \citet and \citep.
    (2) Spelling errors: There are several spelling issues, such as “textcollorbule” in Line 221 (which should be properly formatted in LaTeX), “Moresults” in Line 239 (which may be intended as “More results”), and “since” in Line 320 (which should be capitalized as “Since”).

### Questions
1. Can the authors propose a new method to attack PDMs? If not, can they enhance the PDM-Pure algorithm by considering at least one additional metric (e.g., FID score, inference time, etc.), as suggested in Weakness 1?
2. Can the authors strengthen the theoretical analysis, as discussed in Weakness 2?
3. Can the authors provide detailed experimental results for Table 3, as highlighted in Weakness 3?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work shows that private images protection using current adversarial perturbation methods is not a valid protection. The authors analyze the differences of LDMs and PDMs, and found that the generations of the latter does not suffer from adversarial perturbations in protected images, allowing them to bypass many proposed protections. The authors propose to use this PDMs ability to build a strong adversarial perturbation watermark-purifier, showing that it succeeds in removing protections, with much smaller caveats in generative quality compared to previous purifiers.

### Strengths
This work seems significant. While removing watermarks is not an exactly a hard task, it is interesting to see why adversarial perturbations could compromise generation ability of LDMs and not PDMs. The writing is well organized, and easy to follow. The authors present some interesting insights and a wide range of attacks and purification methods comparison. Small thing, but the authors are also helping the reader with information where it would be good to read a pdf and zoom-in image to see the qualitative difference.

### Weaknesses
- The figures are referenced far from where they are in the paper, it is a bit unordered
- The limitations were not discussed
- directions or other methods of private image protection against proposed purifier are not discussed, only stated as potential future work

### Questions
What are the limitations?
Is your purification method costly compared to other purifiers?
What are the possible directions to create a better defenses as mentioned near the conclusion as future work? Do you have any experiments on that?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper compares latent diffusion models (LDMs) and pixel-space diffusion models (PDMs) under "adversarial perturbations" designed to protect copyright. Under a unified PGD-style diffusion attack, the authors report that LDMs degrade substantially while PDMs remain largely unaffected. Motivated by this, they propose PDM-Pure: running SDEdit in pixel space with a strong PDM to "purify" protected images and restore editability. The paper provides (i) attack-side comparisons across budgets and metrics for LDM vs PDM, and (ii) a purification table where multiple published protections (SDS/Photoguard/Mist/AdvDM/etc.) are evaluated for SDEdit using FID. 

Overall, the paper shows that PDMs exhibit strong empirical robustness to adversarial perturbations that reliably compromise LDMs, the authors attribute the gap to encoder vulnerability, and propose PDM-Pure against adversarial perturbations.

### Strengths
- Clear and repeatable LDM–PDM gap under a unified attack protocol

- Useful mechanistic intuition (PDMs exhibit significantly stronger empirical robustness compared to LDMs, with some possible direction of explanations)

- Practical and simple PDM-Pure purifier

- Purification evaluation covers many published perturbations

- Relevance to real editing pipelines (inpainting/TI/LoRA)

### Weaknesses
Writing & Presentation:
- Table 3 and Figure 6 should be adjusted for clearer flow.

Theory, Methods & Experiments:
1. The inference "no existing attacks have proven effective against PDMs, therefore no protection can be achieved by fooling a PDM-based image editor" does not follow. The reason current protective perturbations primarily target LDMs is that LDMs are more advanced and efficient, and thus far more widely used in academia and industry than PDMs.

2. The claim that "LDMs are more vulnerable than PDMs" is mainly built on a unified PGD-style attack; however, representative published perturbations (e.g., SDS/Photoguard/Mist/Glaze/AdvDM) are not tested in table1. This weakens the persuasiveness of the central claim.

3. The methodological novelty is limited. PDM-Pure is essentially SDEdit in pixel space; the contribution leans more toward a systematic empirical study and a paradigm recommendation, and it requires broader and more rigorous evidence.

3. For “THE ENCODER IS VULNERABLE,” the experimental support should include corresponding ablation studies to strengthen the causal interpretation. (Also, to my knowledge, some published work has already noted this point.)

4. The goal of adversarial perturbation is to protect data, while image generation encompasses multiple task types (e.g., identity learning, style mimicry, image editing). However, the key Table 3 evaluates only image editing and reports only FID. How does PDM-Pure perform on other task types? (Not only the visual comparison, but also the quantitative results)
Need more experiments on additional image-generation tasks and including more metrics in the key table to validate the method’s effectiveness.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
