# Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Differential Attention (DA) has been proposed as a refinement to standard attention, suppressing redundant or noisy context through a subtractive structure and thereby reducing contextual hallucination. While this design sharpens task-relevant focus, we show that it also introduces a structural fragility under adversarial perturbations. Our theoretical analysis identifies negative gradient alignment—a configuration encouraged by DA’s subtraction—as the key driver of sensitivity amplification, leading to increased gradient norms and elevated local Lipschitz constants. We empirically validate this Fragile Principle through systematic experiments on ViT/DiffViT and evaluations of pretrained CLIP/DiffCLIP, spanning five datasets in total. These results demonstrate higher attack success rates, frequent gradient opposition, and stronger local sensitivity compared to standard attention. Furthermore, depth-dependent experiments reveal a robustness crossover: stacking DA layers attenuates small perturbations via depth-dependent noise cancellation, though this protection fades under larger attack budgets. Overall, our findings uncover a fundamental trade-off: DA improves discriminative focus on clean inputs but increases adversarial vulnerability, underscoring the need to jointly design for selectivity and robustness in future attention mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a theoretical and empirical study of Differential Attention (DA) — a variant of self-attention that combines two attention maps through subtraction (A₁ − λA₂). The authors reveal that while DA improves task-relevant selectivity, its subtractive structure may amplify adversarial sensitivity due to negative gradient alignment between branches. The results show that DA often yields higher attack success rates and larger local Lipschitz estimates, confirming the theoretical fragility. Interestingly, stacking DA layers can partly cancel noise and improve robustness under small perturbations, though this advantage diminishes under stronger attacks.

### Strengths
1. The paper firstly provides a systematic theoretical treatment of how DA’s subtractive structure influences robustness.

2. Empirically, the work is thorough and reproducible, covering multiple model families (ViT, CLIP) and three attack methods, with reasonable ablations on λ and model depth.

3. The paper is well-written with a clear logical flow. The mathematical analysis is clear and connects gradient alignment, Lipschitz continuity, and adversarial fragility in a consistent framework.

### Weaknesses
1. The experimental design does not fully satisfy the standard of transfer-based robustness evaluation. The authors describe results on five datasets, but each model is trained and tested on the same dataset, without evaluating cross-dataset or transfer performance.

2. The “depth-dependent robustness” claims also remain partially qualitative.

### Questions
1. See in W1.

2. The paper theorizes depth-dependent robustness (ᾱ L̄DA < 1), but empirically, only ASR and per-layer Lipschitz values are shown. How is ᾱ estimated or validated?

3. The adversarial settings focus on small ϵ, would the observed depth advantage persist under larger perturbations or different norms.

4. It would be valuable to include comparisons with adversarially trained or robustly pretrained baselines.

5. Whether adversarial examples transfer between DA-based and standard-attention models more effectively than vice versa?

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
4

### Summary
This paper systematically investigates the robustness of the Differential Attention (DA) mechanism under adversarial perturbations. The authors find that although DA effectively suppresses redundant information and enhances the model’s focus on critical features through the subtraction of two attention maps, this subtractive structure leads to negative gradient alignment. As a result, it amplifies input perturbations and increases the local Lipschitz constant, making the model more vulnerable to adversarial attacks. To address this, the paper proposes the “Fragile Principle,” which theoretically reveals the structural cause of DA’s amplified sensitivity. Extensive experiments on ViT, DiffViT, CLIP, and DiffCLIP models further validate this phenomenon. In addition, the authors observe a “depth-dependent robustness crossover” property of DA: shallow models tend to be more sensitive, whereas deeper stacks can exhibit partial robustness under small perturbations. Overall, this work uncovers a structural trade-off between selective enhancement and robustness degradation in DA, providing both theoretical insights and practical guidance for designing future attention mechanisms that balance focus and robustness.

### Strengths
1.The first systematic study on the robustness of Differential Attention (DA).
Previous research has primarily focused on the advantages of DA in reducing attention hallucinations and enhancing focus. In contrast, this work is the first to introduce the “Fragile Principle” from the perspective of adversarial robustness, revealing that the subtractive structure of DA theoretically amplifies sensitivity to input perturbations.
2.Rigorous theoretical derivation and comprehensive experimental design.
The paper presents a series of formal results (Lemmas and Theorems 1–5) that systematically derive the gradient amplification mechanism and Lipschitz bounds of DA, providing solid mathematical foundations for the Fragile Principle. The authors conduct extensive comparative experiments on ViT, DiffViT, CLIP, and DiffCLIP models across multiple datasets (CIFAR-10/100, Tiny ImageNet, MSCOCO, ImageNet-1k), covering various adversarial attack methods including PGD, CW, and AutoAttack.
3.Consistent and well-validated results.
Theoretical predictions—such as negative gradient alignment, increased Lipschitz constants, and depth-dependent robustness trends—are consistently supported by empirical evidence.

### Weaknesses
1.Overreliance on local linear assumptions in theoretical analysis.The paper primarily analyzes the sensitivity of DA through gradients and local Lipschitz constants. While such local linear approximations are common, they do not always accurately capture the global behavior of deep nonlinear networks.
2.Limited robustness evaluation.The current experiments mainly assess robustness under conventional ℓ∞ and ℓ₂ attacks. However, in real-world applications, more complex perturbations—such as natural distortions, semantic attacks, and physical noise—are often more representative and challenging.
3.Lack of mechanistic interpretation.The experiments observe a “depth-dependent robustness crossover” phenomenon, but the paper only describes it empirically without explaining its underlying causes in terms of gradient propagation or feature space dynamics.

### Questions
1.Theoretical analysis is primarily based on local linear approximations and single-layer isolation assumptions.These assumptions may not hold under deep networks with nonlinear coupling, and the paper provides insufficient discussion of their impact and applicability. It is recommended to explicitly clarify the conditions under which the local linear assumption remains valid.
2.The observed phenomenon of “shallow fragility and deeper robustness to small perturbations” lacks mechanistic explanation.The paper is encouraged to provide a more detailed interpretation or theoretical reasoning for this behavior.
3.More ablation studies are needed to pinpoint the source of the fragility issue.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the adversarial robustness of the Differential Attention (DA) mechanism. It concludes that while DA improves discriminative focus on clean inputs, it inherently amplifies adversarial sensitivity, and that stacking DA layers can partially mitigate this effect through depth-dependent noise cancellation.

### Strengths
[1] The authors hypothesize and theoretically analyze that while DA helps suppress redundant context and enhances task-relevant selectivity, it may amplify sensitivity to input perturbations due to negative gradient alignment between its branches. The biggest problem with this paper is the lack of sufficient comparative methods, failing to demonstrate the advantages of this method compared to others.

[2] The paper formalizes this effect as the Fragile Principle and connects it to increased gradient norms and local Lipschitz constants.
Theoretical derivations are provided, along with extensive experiments using ViT/DiffViT and CLIP/DiffCLIP models on multiple datasets (CIFAR-10/100, Tiny ImageNet, COCO, ImageNet).

### Weaknesses
[1]The literature review section and comparative baselines did not adequately discuss other studies on attention robustness.  The paper does not benchmark DA against attention variants designed for robustness. The biggest problem with this paper is the lack of sufficient comparative methods, failing to demonstrate the advantages of this method compared to others.

[2] The text repeatedly mentions "gradient opposition" and "perturbation amplification." Providing visualizations of gradient fields or attention maps (e.g., adversarial heatmap comparisons) would significantly enhance its persuasiveness. 

[3] Similar analyses on gradient sensitivity and Lipschitz bounds for attention already exist (e.g., Kim et al., 2021; Dasoulas et al., 2021).

[4] Quantitative examples are lacking (e.g., attention map visualization, perturbation trajectory, or gradient heatmap could be more informative).

[5] Is the proposed differential attention mechanism applicable to existing large language models?

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies the robustness properties of Differential Attention (DA), a variant of standard attention that subtracts two attention maps. It provides a theoretical analysis linking DA’s subtractive operation to gradient alignment and local Lipschitz bounds that govern adversarial sensitivity. The authors derive formal expressions showing that negative gradient alignment amplifies perturbations while increased network depth can attenuate this effect. Experiments on ViT and DiffViT models trained on CIFAR-100 and Tiny-ImageNet evaluate attack success rates under PGD, CW, and AutoAttack. Further experiments on CLIP and DiffCLIP with ViT-B/16 backbones test sensitivity on ImageNet and COCO using cosine-similarity-based adversarial losses. Results show that DA-based models are generally more sensitive to small perturbations, with the difference diminishing as network depth increases.

### Strengths
-   Provides a clear theoretical analysis of Differential Attention and its connection to adversarial sensitivity.
    
-   Conducts a well-controlled experimental study across multiple depths, datasets, and attack types, systematically isolating the role of model depth.
    
-   Demonstrates consistent empirical evidence that sensitivity effects are most pronounced in shallow models and diminish with increasing depth.

### Weaknesses
-   **Shallow-depth focus limits practical impact.** The paper’s main vulnerability appears in 1–4 layer ViTs, which are not used in practice. By D = 12 the effect largely disappears, so the controlled finding is interesting but of limited relevance to standard transformer deployments.
    
-   **Vision-only experiments are low-resolution / small-scale.** The ViT / DiffViT results are reported on CIFAR and Tiny-ImageNet. These datasets use lower-resolution images and limited context. Full ImageNet-1k experiments on realistic ViT-B/16 (or comparable) models are necessary to validate transfer to real-world settings.
    
-   **Adversarial attack suite for transformers is incomplete.** The evaluation relies on PGD / CW but omits transformer-aware attacks. Stronger image-side attacks such as TI-FGSM [1],  token-level attacks (Token Gradient Regularization) [3], and patch-focused (P-IFGSM) [2] attacks should be included. Also test higher budgets (e.g., ε = 16/255 as an upper bound) for ImageNet-scale robustness checks.
    
-   **VLM attack protocol is weak relative to prior work.** For CLIP/DiffCLIP the paper minimizes cosine similarity to a single prompt. Robust evaluation should use contrastive, multi-prompt, and multimodal attack frameworks (e.g., VL-Attack [4] / PromptAttack–style contrastive attacks and adapted AutoAttack variants) to provide a stronger, field-standard baseline.


1.  **Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu.** _Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks._ In **CVPR**, 2019.  
    
2.  **Lianli Gao, Qilong Zhang, Jingkuan Song, Xianglong Liu, and Heng Tao Shen.** _Patch-wise Attack for Fooling Deep Neural Networks._ In **ECCV**, 2020.  
    
3.   **Jianping Zhang, Yizhan Huang, Weibin Wu, and Michael R. Lyu.** _Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization._ In **CVPR**, 2023.  
    
4.  **VL-Attack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models.** In **NeurIPS**, 2023.

### Questions
-   For the PGD experiments, were the attacks performed with **random restarts** to avoid local minima and confirm consistency across runs?
    
-   For **DiffCLIP**, how might **prompt-based or contrastive multi-prompt attacks** affect the observed robustness trends?
    
-   Beyond **classification**, are there other tasks (e.g., retrieval, captioning, grounding) where demonstrating the Differential Attention sensitivity effect might be more informative or representative?
    
-   Have the authors considered evaluating on **natural adversarial benchmarks** such as _NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples_ [1] to complement synthetic PGD or patch attacks?

1.NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples, NeurIPS 2024 (DnB Track)

### Soundness
2

### Presentation
3

### Contribution
2
