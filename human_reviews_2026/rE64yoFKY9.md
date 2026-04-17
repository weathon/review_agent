# Boosting Targeted Adversarial Transferability: A Generative Approach Guided by Core Target Samples

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Adversarial examples generated on one model can often be transferred to other unseen models, but achieving high targeted transferability remains challenging due to overfitting—especially under single-surrogate constraints. In this work, we propose BAT, a generative approach that Boosts targeted Adversarial Transferability by training the generator to align its outputs with a curated set of high-confidence \textit{core target samples}. These samples—either selected from real data or synthesized from noise—serve as guidance across both output and feature spaces. To mitigate overfitting without requiring multiple surrogates, BAT employs an ensemble of frozen discriminators derived via pruning from a single pretrained surrogate model. BAT is applicable whether both the generator's training (source) and the evaluation images come from the target models’ training domain or exhibit a domain shift; it remains effective even without real target-class images during training. Extensive experiments on ImageNet-1K show that BAT notably outperforms existing $\ell_{\infty}$-constrained targeted attacks. We also provide theoretical bounds that reveal how ensemble size influences transferability, aligning with observed empirical trends.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
he paper introduces BAT, a generative targeted-transfer attack that aligns adversarial examples to a small “core” set of high-confidence target samples in both output and intermediate feature spaces. To overcome single-surrogate overfitting, BAT builds a self-ensemble of frozen discriminators by pruning a single pretrained model, avoiding extra training while injecting boundary diversity. Three variants handle data availability: BAT-BS (best real target images), BAT-CS (confidence-crafted target references), and BAT-CN (references synthesized from noise), covering both P=Q and P≠Q regimes. On ImageNet-1K and a Painting domain shift, BAT yields notably higher targeted TSR than strong iterative and generative baselines, with a simple theory linking transferability to ensemble size and its diminishing returns.

### Strengths
- Clear, modular method: dual-space alignment to high-confidence core targets plus pruned-ensemble discriminators to mitigate single-surrogate overfitting.
- Broad applicability: supports domain match and shift and target-data-guided vs. target-data-free training (BAT-CN works without real target images).
- Strong empirical gains: consistent TSR improvements over state of the art across many victim architectures, including evaluations against robustly trained models.
- Ablations isolate contributions of ensemble size and core-set selection and show monotonic improvements from baseline to full BAT variants.
- Theory offers lower/upper bounds explaining the role of ensemble size and why gains saturate, matching empirical trends.

### Weaknesses
- Statistical reporting is light: no multi-seed confidence intervals or seed-paired comparisons; robustness claims could be sensitive to randomness
- Threat-model breadth could be expanded: emphasis on $\ell_\infty$ at a fixed $\epsilon$; limited coverage of ℓ2 or structure-preserving perturbations in the main text
- The pruning-diversity assumption is plausible but under-analyzed: lack of explicit diversity metrics (e.g., gradient/decision disagreement) vs. multi-surrogate ensembles
- While theoretical support for a tighter $\ell_2$ constraint than $\ell_\infty$ holds, empirical testing would more strengthen the claim
- Limited discussion of compute/latency for training and inference, especially as |Ds| and core-set size scale

### Questions
- How stable are the reported gains across multiple seeds, and can you provide confidence interval range comparisons vs. the strongest baseline?
- Can you quantify ensemble diversity induced by pruning (e.g., gradient cosine, boundary disagreement) and relate it to TSR?
- Do the main trends persist for a second norm (ℓ2) and a simple spatial/color threat without heavy retuning?
- What are the compute/latency costs for training and inference as |Ds| and core-set size vary, and where is the best accuracy/efficiency frontier?

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
This paper presents BAT,  a generative attack framework for targeted adversarial transfer under the constraint of a single surrogate. BAT is based on aligning both output and intermediate feature spaces of generated adversarial examples to those of a curated set of high-confidence core target samples. The key contribution is the creation of a diverse, frozen discriminator ensemble by pruning a single pretrained surrogate, thereby avoiding the need for multiple trained surrogates. The approach is evaluated in both domain-matched and domain-shift scenarios and supports both target-data-guided and target-data-free settings. Through extensive experiments and theoretical analyses, BAT is shown to improve targeted transferability over prior methods.

### Strengths
1. This work has a comprehensive evaluation  under both no-shift and domain-shift settings  across diverse architectures.

2.  Detailed ablations (Figure 5, Table 6) help EXPLAIN where performance gains come from,  diversified discriminators, core target selection.

3.  The section 3 and algorithm 1 and 2 in the appendix and clear and easy to understand.

4. The self-ensemble is simple and effective.

### Weaknesses
1. The description of the generator architecture is under-specified.  Since it is an important component of the proposed work,  the authors should provide a detailed description.

2. It is easy to convert some untargeted transferable attacks (e.g., admix-DT ) to the targeted ones. The authors should compare with them


[1] Xiaosen Wang, Xuanran He, Jingdong Wang, and Kun He. Admix: Enhancing the transferability of adversarial attacks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16158–16167, 2021.

### Questions
1. Augmentation strategies can also be viewed as a form of self-ensemble. Could data-level self-ensemble via various augmentations achieve better performance compared to the model-level self-ensemble achieved by pruning? if data-level self-ensemble and model-level self-ensemble are combined, can the TSR be further improved?

2. While the paper presents an interesting framework for boosting targeted transferability, I remain concerned about the level of novelty. The idea of ensembling surrogates to improve transferability is well established (e.g.,[2]); the use of feature-space alignment is also explored (e.g., [3]);  The notion of target samples as anchors is also explored (e.g., TTAA). So is this work a recombination of existing techniques into the generative targeted-transfer setting? I am not sure whether I understand correctly.


[2] Hung-Jui Wang, Yu-Yu Wu, and Shang-Tse Chen. Enhancing targeted attack transferability via diversified weight pruning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2904–2914, 2024a.

[3] Zhipeng Wei, Jingjing Chen, Zuxuan Wu, and Yu-Gang Jiang. Enhancing the self-universality for transferable targeted attacks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12281–12290, 2023.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes BAT, a generative method to improve targeted adversarial transferability by training a generator to align perturbations with curated high-confidence core target samples, guided in both output and feature spaces. To reduce overfitting without multiple surrogates, BAT uses an ensemble of frozen discriminators obtained by pruning a single pretrained surrogate, and it remains effective under domain shift and even without real target-class images.

### Strengths
1. Multiple BAT variants are developed and evaluated, offering a clear view of design choices and their impact. 
2. Against competitive baselines, BAT demonstrates consistent gains, supporting the claimed effectiveness.
3. On ImageNet-1K, BAT surpasses existing ℓ∞-constrained targeted attacks and is supported by theoretical bounds linking ensemble size to transferability, matching empirical trends.

### Weaknesses
1. The rationale for why randomly pruning a frozen surrogate yields effective discriminator ensembles is under-explained; a deeper analysis or ablation would clarify mechanism and sensitivity.
2. The preliminaries section is cluttered and difficult to follow; tighter structure and clearer notation would improve readability.
3. Criteria for identifying “high-confidence” target samples are insufficiently specified.

### Questions
1. What mechanism makes randomly pruned, frozen discriminator ensembles effective for targeted transfer?
2. How well do the theoretical bounds track empirical gains with ensemble size, does BAT generalize across model families beyond the surrogate, and what are the main failure modes?

### Soundness
3

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
This paper proposes BAT (Boosting Adversarial Transferability), a generative framework that enhances targeted adversarial transferability by training generators to align outputs with curated high-confidence core target samples. The method employs an ensemble of frozen discriminators derived via pruning from a single pretrained surrogate model, eliminating the need for multiple surrogates. Theoretical bounds are provided to explain how ensemble size influences transferability. Experiments demonstrate consistent improvements in targeted success rates across various settings, including no domain shift, domain shift, and against robust models.​

### Strengths
Innovative Single-Surrogate Ensemble Approach: The paper introduces a novel method to create diverse discriminator ensembles through pruning of a single pretrained model, eliminating the need for multiple distinct surrogate models while still achieving strong transferability. This addresses a key limitation of existing methods that require access to multiple models.​
Theoretical Foundations with Practical Insights: The work provides rigorous theoretical analysis establishing lower and upper bounds on targeted transferability, revealing how ensemble size trades off with performance. This theoretical framework aligns well with empirical observations and offers valuable guidance for practical implementation.​

### Weaknesses
1. Your experimental evaluation primarily focuses on transfer scenarios using ResNet and DenseNet architectures. However, the increasing adoption of vision transformers (ViT) in computer vision applications necessitates a more comprehensive analysis of cross-architecture transferability. Could you please conduct additional experiments to address: (1) ResNet-to-ViT transferability across multiple ViT variants (e.g., ViT-Base, ViT-Large, DeiT); (2) ViT-as-surrogate transferability to traditional convolutional networks.

2. Universality under domain shift: Beyond the single Painting↔ImageNet experiment, could you provide additional results on other cross-domain pairs (e.g., Sketch, Photo, synthetic datasets) to substantiate that the “core-sample + self-ensemble” strategy remains robust across a wider spectrum of distributional discrepancies?

### Questions
See weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
3
