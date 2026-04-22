# Discrete Latent Features Ablate Adversarial Attack: A Robust Prompt Tuning Framework for VLMs

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
While adversarial fine-tuning can enhance the robustness of vision-language models (VLMs), such approaches are computationally expensive. Adversarial prompt tuning has emerged as a practical alternative. However, existing methods are limited by their reliance on vulnerable continuous image features. To mitigate the vulnerability in the feature representation, we propose **DEFEAT** (**D**iscrete Lat**E**nt **F**eatur**E** based **A**dversarial **T**raining), a robust prompt tuning framework for VLMs.
Specifically, the DEFEAT method introduces a perturbation discrete shield module that reconstructs discrete latent features and designs a logits fusion strategy, substantially reducing the discrepancy between clean and adversarial image representations.
Moreover, the DEFEAT method integrates prompt tuning with adversarial training while applying regularization from learnable prompts to hand-crafted prompts, further enhancing the adversarial robustness.
Extensive experiments across 15 datasets validate the effectiveness of the proposed  DEFEAT method among existing adversarial prompt tuning methods. The official code is available at https://github.com/cheny02/DEFEAT-ICLR2026.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes DEFEAT, a robust prompt-tuning framework for CLIP-like VLMs that discretizes latent grid features with a VQ-VAE module, and fuses logits from discretized and original branches, coupled with prompt-alignment regularization to hand-crafted prompts. The central claim is that feature discretization reduces clean->adv feature shift, yielding substantially higher robust accuracy and harmonic mean scores under few-shot, cross-dataset, and domain-generalization settings. The paper provides PGD evaluations, AutoAttack and CW, and a defense-aware adaptive attack targeting the fusion branches. Overall, the method is compact (frozen encoders; train *PerturbShield* + prompts) and the empirical coverage is broad, though statistical reporting and adaptive-attack rigor need tightening.

### Strengths
- Clear discretization motivation with reduced feature shift
- Simple modular defense (VQ-VAE grid reconstruction + projection + fusion + prompt regularization)
- Broad empirical coverage with consistent robustness/H gains; AA/CW and adaptive checks provided
- Ablations clarify each component’s role; µ improves robustness

### Weaknesses
- Adaptive attack lacks explicit BPDA/ST gradients through quantization, risking masking
- AA/CW only on subset; AA configs (targeting, restarts, seeds) unspecified in main text
- Threat model focused on ℓ∞; no ℓ2/spatial in main paper
- No multi-seed CIs; missing runtime and codebook specs

### Questions
- What are the typical failure cases where discretization/fusion underperform (e.g., fine-grained textures, clutter, small objects), and what mitigation directions look promising?
- Are the reported robustness gains consistent across multiple seeds, and do they remain visible under paired comparisons against the strongest baseline?
- How stable are results to reasonable prompt variations (e.g., templated vs hand-crafted phrasing), and do trends remain when prompts are slightly perturbed?

### Soundness
3

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
4

### Summary
This paper presents a novel and well-executed framework for improving the adversarial robustness of VLMs through prompt tuning. Extensive evaluation on 15 datasets under few-shot, cross-dataset and domain-generalisation settings.

### Strengths
1. Introduces a novel application of latent feature discretization to adversarial defense for VLMs.

2. Comprehensive evaluation across 15 datasets, covering adversarial few-shot classification, cross-dataset, and domain generalization.

### Weaknesses
1. This paper presents a well-structured framework. however, the underlying techniques for the individual components are not entirely novel.

2. The paper makes a significant contribution to improving the adversarial robustness of CLIP. However, a limitation of the current study is that all experiments are exclusively conducted on the CLIP model. To broaden the applicability and contribution of the proposed methods, it would be highly beneficial to include experimental results on other relevant models, such as OpenCLIP and EVA-CLIP. Furthermore, the empirical evaluation focuses exclusively on image classification.The absence of experiments on other applications limits the demonstrated scope and real-world applicability of the DEFEAT method.

3. The practical utility of the DEFEAT method is not fully clear due to the lack of analysis on its computational cost relative to baseline methods. 

4. The discussion could be further enriched by considering recent advancements in test-time defenses. if the authors could investigate whether combining DEFEAT with pre-processing defenses [1] or test-time defenses [2,3] could lead to synergistic effects, potentially achieving a "1+1>2" outcome.

[1] Diffusion Models for Adversarial Purification, ICML 2022

[2] CLIP is Strong Enough to Fight Back: Test-time Counterattacks towards Zero-shot Adversarial Robustness of CLIP, CVPR 2025.

[3] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models, CVPR 2025.

### Questions
Please refer to the questions raised in the Weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes DEFEAT, a robust prompt tuning framework for vision-language models (VLMs) that enhances adversarial robustness by leveraging discrete latent features. The method employs a VQ-VAE-based Perturbation Discrete Shield (PerturbShield) to map both clean and adversarial features into shared discrete representations, combined with logits fusion and prompt alignment strategies. Experiments across multiple datasets demonstrate significant improvements in adversarial robustness while maintaining competitive clean accuracy.

### Strengths
- The method is novel, introducing discrete latent representations into the robust prompt tuning framework, which provides a fresh and intuitive perspective on defending against adversarial perturbations.
- The experimental results are strong, showing substantial improvements in robustness accuracy compared with existing adversarial prompt tuning methods.
- The paper provides rich visualizations of feature distributions, which help illustrate and explain how the proposed method mitigates adversarial effects.

### Weaknesses
- The paper lacks a theoretical justification for why clean and adversarially perturbed features can be mapped to the same discrete representation within the VQ-VAE framework.
- The approach involves a large number of hyperparameters (\alpha, \beta, \lambda, \mu), making the method relatively hard to tune and potentially sensitive to configuration choices.

### Questions
Please refer to the weaknesses.

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
The authors propose DEFEAT (Discrete LatEnt Feature based Adversarial Training), a new robust prompt tuning framework for VLMs. The core hypothesis is that discretizing latent features can effectively mitigate adversarial perturbations. The authors conduct extensive experiments on 15 datasets across adversarial few-shot classification, domain generalization, and cross-dataset generalization settings. The results show that DEFEAT achieves state-of-the-art robustness and a better accuracy-robustness trade-off compared to existing adversarial prompt tuning methods like APT and FAP.

### Strengths
1. A Novel and Insightful Defense Mechanism: The core idea of using a VQ-VAE to create discrete latent feature representations as an adversarial defense is a novel contribution in the prompt-tuning space. The design is particularly insightful for choosing to discretize the grid features ($I_{patch}$) rather than the single, global class token ($I$).

2. Comprehensive experiments: Extensive experiments across 15 datasets are conducted to verify the effectiveness of the DEFEAT method. The paper provides good ablation studies that clearly demonstrate why the proposed components are necessary.

### Weaknesses
Heavy Reliance on a Pre-Trained Robust Backbone:  As the authors report in Section C, DEFEAT requires a robust visual backbone (like TeCoA) to function. When applied to a standard, non-robust CLIP model, the PGD attack can still pose significant threats.

### Questions
Since all tested adversarial prompt tuning methods fail without a robust backbone, is there any possible way to adapt your method under this scenario?

### Soundness
3

### Presentation
3

### Contribution
3
