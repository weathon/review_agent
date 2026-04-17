# Elucidating Guidance in Variance Exploding Diffusion Models: Fast Convergence and Better Diversity

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Recently, the conditional diffusion models have shown an impressive performance in many areas, such as text-to-image, 3D, and video. To achieve a better alignment with the given condition, guidance-based methods are proposed and become a standard component of diffusion models. Though the guidance-based methods are widely used, the theoretical guarantee mainly focuses on the variance-preserving (VP)-based models and lacks current state-of-the-art (SOTA) variance-exploding (VE) models for conditional generation. In this work, for the first time, we elucidate the influence of guidance for VE models and explain why VE-based models perform better than VP models in the context of Gaussian mixture models from classification confidence and diversity perspectives. For the classification confidence, we prove the convergence rate for the confidence w.r.t. the strength of guidance $\eta$ for VE models is  $1-\eta^{-1}(\log \eta)^2$, which is faster than $1-\eta^{-e^{-T}}(\log \eta)^{2 e^{-T}}$ result for VP models ($T$ is the diffusion time). This result indicates that the VE models have a stronger ability to align with the given condition, which is important for the conditional generation. For the diversity, previous works show that when facing strong guidance, VP models tend to generate extreme samples and suffer from the modal collapse phenomenon. However, for VE models, we show that since their forward process maintains the multi-modal property of data, they have a better ability to avoid the modal collapse facing strong guidance. 
The simulation and real-world experiments also support our theoretical results.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This article studies how classifier guidance affects classification confidence and mode collapse in Variance-Exploding (VE) and Variance-Preserving (VP) SDEs. The authors derive convergence rates for classification confidence, showing it scales as $1 - \eta^{-1} (\log \eta)^{2}$ for VE-SDEs and $1 - \eta^{-e^{-T}} (\log \eta)^{2 e^{-T}}$ for VP-SDEs, where $\eta$ is the guidance strength and $T$ is the diffusion time. They also analyze the distinct mode collapse behaviors in these two SDE classes. Theoretical claims are verified through numerical experiments on Gaussian mixtures and the MNIST dataset.

### Strengths
- **Originality**: This work provides a novel theoretical analysis of how guidance strength affects the classification confidence of VE-SDEs and VP-SDEs, a relationship not previously established in the literature.
- **Quality**: The study is of high quality, featuring a well-organized introduction, a clearly motivated methodology, rigorous theoretical derivations, and well-designed experiments that substantiate the claims.
- **Clarity**: The theoretical findings are well motivated, which makes it easy to follow.
- **Significance**: The research addresses a significant scientific problem by providing a theoretical foundation for the observed performance differences between VP-SDEs and VE-SDEs, which is crucial for their effective application.

### Weaknesses
- The claim in Section 5 that VE-SDE outperforms VP-SDE on Gaussian mixtures under strong guidance is not fully substantiated, as the presented analysis itself is independent of the guidance strength.
- While the analysis of diversity is a core contribution, the study lacks empirical experiments that directly evaluate the diversity of generated samples. This omission weakens the central claims regarding the differential behavior of VE-SDE and VP-SDE.
- The difference in classification confidence is presented as the primary reason for VE-SDE's superior performance. However, this may not be the sole factor. The authors should strengthen their argument by demonstrating a strong correlation between their theoretical metric (classification confidence) and established generative quality metrics like FID.
- The numerical experiments, while well-executed on MNIST, would benefit from validation on more complex and diverse image datasets (e.g., CIFAR-10, FFHQ). This would strengthen the generalizability of the conclusions and demonstrate the broader applicability of the theoretical findings.

### Questions
- The study would be strengthened by a more comprehensive analysis that directly links the theoretical guidance strength, 
$\eta$, to the empirically observed behaviors of mode collapse in both VE-SDE and VP-SDE. A quantitative discussion bridging the derived convergence rates and the diversity of generated samples would be particularly valuable.
- While the theoretical analysis of classification confidence is insightful, its practical importance would be more compelling if supported by empirical evidence. We recommend conducting experiments that demonstrate a clear correlation between the proposed classification confidence metric and established generative performance metrics, such as FID.
- The conclusions are currently based on experiments with the MNIST dataset. To demonstrate the broader applicability and robustness of the findings, it is crucial to include experiments on more challenging and modern image datasets, such as CIFAR-10, CelebA, or FFHQ.
- In Line 198, does $m_{t} = 1$ rather than $m_{t} = 0$ for VE-SDE?

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
3

### Summary
The manuscript provides a theoretical analysis on the guidance of Variance Exploding (VE) diffusion model, explaining why VE can achieve state-of-the-art performance in conditional generation tasks. Authors analyze guidance from two perspectives: classification confidence and diversity and prove that though VE’s error is larger than VP without guidance but converges faster with guidance, making it quickly achieve high confidence while maintaining multi-modal structure. Theoretical results are supported the experiments on MNIST, showing VE models achieve higher confidence and accuracy with less distortion under strong guidance.

### Strengths
1. Meaningful research questions that address key puzzles within the community;
2. Rigorously logical arguments and solid proofs of theorems.

### Weaknesses
1. The experiments seem to be a little weak. Though experiments on MNIST support the derived conclusion, it is too simplistic and lacks diversity. Despite producing the desired results, the evidence remains insufficiently strong. I believe that it would be beneficial if more experiments were included, such as the CIFAR dataset. 
2. The assumption of identity variance may be overly restrictive. Even if the marginal distributions are the same, differences across conditional distributions may manifest not only in their means but also in substantial variations in their variances. It would be valuable to examine whether the authors' analysis remains valid under scenarios where the variances of the conditional distributions differ significantly. Similarly, this concern warrants experimental validation. The authors could consider conducting experiments on datasets where conditional distributions exhibit considerable divergence in variance.
3. In Section 5, the authors present a qualitative analysis of the diversity. Though logically coherent, it would be beneficial if authors incorporate quantitative metrics related to diversity—such as the Inception Score—into their experiments. By doing so, the theoretical conclusions derived from the analysis could be further supported from an empirical perspective.
4. Typo: the Gaussian distribution should be expressed as $\mathcal{N}$ in line 260.
5. The assumption seems strong in this paper. The proofs are given under the Mixture of Gaussian distributions rather than a general target distribution. So it is hard to say its practical meaning for complex high-dimensional generation tasks.

### Questions
Can the authors give more explanation about the meaning of classification confidence and why they formulate it as Eq. 3?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies guidance for variance-exploding (VE) diffusion models, contrasting them with variance-preserving (VP) models. In a Gaussian-mixture setting, it claims two main results: (i) classification-confidence convergence faster in VE models with guidance strength $\eta$ increasing; (ii) VE’s forward process preserves multi-modality. Simulations and MNIST experiments qualitatively support these findings.

### Strengths
The paper studies how guidance influences classification confidence and diversity, which are exactly the two key properties practitioners trade in practice.

Using Gaussian mixture models yields closed-form expressions for scores, enabling detailed proofs and clear diagnostics. The results appear correct and also complements existing study.

The mode-preservation rationale for VE is intuitive and well-motivated.

### Weaknesses
The main results focusing on 2-component Gaussian mixture, with a remark at the end promising the direct extension to multiple components. I am ok with Gaussian mixture assumption, however, if the analysis is directly applicable to multiple components, it is recommended to present the multiple-component results.

Many results in the paper is similar to Wu et al., both in setting and presentation. The novelty beyond the VE setup is not very clearly stated.

Experiments are illustrative rather than conclusive. Yet, this is a minor weakness as the majority of the paper is theoretical.

### Questions
Theorem 4.2 mainly compares upper bounds and claims that VE is better.

Are there quantitative results in Section 5 to support the intuition? In fact, although VE keeps the mean unchanged, with larger and larger noise added, the three components also become indistinguishable.

Mode collapse is more often used than modal collapse.

I hope the authors can clarify their technical contributions and discuss the results for general finite-component Gaussian mixture models. I am willing to increase my rating if the responses are satisfactory.

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
3

### Summary
This paper offers a theoretical and empirical investigation into how guidance operates in variance-exploding (VE) diffusion models. While previous studies have primarily focused on variance-preserving (VP) models, the authors aim to explain why VE-based models, such as EDM, tend to perform better in conditional generation tasks when combined with classifier or classifier-free guidance.

The paper provides analytical results showing that VE diffusion models achieve faster convergence of classification confidence under guidance and preserve greater diversity across samples. The authors derive explicit convergence rates with respect to the guidance strength parameter $\eta$. For the case of VE and VP, the authors proposed and proved their rates, and the result suggests that VE models align more efficiently with the guidance signal.

The authors further argue that VE processes preserve multimodal structure throughout the forward diffusion, which prevents the mode collapse that often occurs in VP models under strong guidance. Experiments on Gaussian mixture models and MNIST support these claims, demonstrating that VE-guided diffusion maintains higher classification confidence and sample diversity compared to VP-guided baselines.

### Strengths
The motivation is clear and relevant. Understanding why VE models perform better under guidance is important, as these models dominate practical diffusion-based generative systems such as EDM and SMLD. The paper fills a gap in theoretical understanding. The mathematical analysis is rigorous and well structured. The derivations of convergence rates are transparent and carefully derived under clearly stated assumptions. The extension of the analysis from VP to VE models is technically sound. The study deepens understanding of guidance in diffusion models and could influence future work on improving conditional generation or training strategies in VE-based systems.

### Weaknesses
The paper’s theoretical results are derived under restrictive assumptions, primarily Gaussian mixture models. While this setting makes the analysis tractable, it limits the generality of the conclusions. Real-world data are rarely Gaussian, and it remains uncertain how the theoretical findings extend to complex, high-dimensional distributions used in practical diffusion models.

The discussion of diversity preservation is mostly qualitative and lacks a formal definition or measurable bound. A quantitative diversity metric or theoretical result would make the argument more convincing. The empirical evaluation, though consistent with the theory, is limited to small datasets such as MNIST and synthetic Gaussian mixtures. The experiments do not show whether the same patterns appear in large-scale or multimodal diffusion models used in realistic image or text-to-image generation tasks.

Finally, the contribution is solid but still kind of incremental in scope. The paper primarily adapts established VP analyses to the VE case rather than introducing an entirely new theoretical framework.

### Questions
1. The theoretical analysis assumes Gaussian mixtures. How do you expect the conclusions to generalize to non-Gaussian or multimodal real-world data distributions, such as those seen in text-to-image models?
2. Does the faster convergence rate hold for both classifier and classifier-free guidance, or is it specific to one type?
3. Could larger-scale experiments on more complex datasets confirm that VE-guided diffusion retains higher diversity and faster convergence beyond the toy examples?

### Soundness
3

### Presentation
3

### Contribution
3
