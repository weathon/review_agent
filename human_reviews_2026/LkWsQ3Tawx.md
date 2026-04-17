# Untraceable DeepFakes via Traceable Fingerprint Elimination

- Decision: Accept (Poster)
- Scores: 2, 2, 6, 4

## Abstract
Recent advancements in DeepFakes attribution technologies have significantly enhanced forensic capabilities, enabling the extraction of traces left by generative models (GMs) in images, making DeepFakes traceable back to their source GMs. Meanwhile, several attacks have attempted to evade attribution models (AMs) for exploring their limitations, calling for more robust AMs.
However, existing attacks fail to eliminate GMs' traces, thus can be mitigated by defensive measures.
In this paper, we identify that untraceable DeepFakes can be achieved through a multiplicative attack, which can fundamentally eliminate GMs' traces.
Therefore, by leveraging the structural prior from content-coupled fingerprints, we design a multiplicative attack framework that instills an explicit inductive bias into the adversarial model, guiding it to eliminate fingerprints within DeepFakes, thereby evading AMs even enhanced with defensive measures. This framework trains the adversarial model solely using real data, applicable for various GMs and agnostic to AMs.
Experimental results demonstrate the outstanding attack capability and universal applicability of our method, achieving an average attack success rate (ASR) of 97.08\% against 6 advanced AMs across 12 GMs.
Even in the presence of defensive mechanisms, our method maintains an ASR exceeding 72.39\%.
Our work underscores the potential challenges posed by multiplicative attacks and highlights the need for more robust AMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a "multiplicative attack" framework aimed at generating untraceable deepfakes that evade forensic attribution models. The core idea is to apply pixel-wise multiplicative perturbations to the generated image to suppress identifiable model fingerprints while preserving visual fidelity. The authors present a theoretical justification suggesting that such perturbations make attribution statistically difficult, and propose an optimization-based implementation that enforces perceptual similarity through the combination of multiple loss terms. Extensive experiments are conducted across multiple models, showing that the proposed method significantly reduces attribution accuracy compared to existing attacks while maintaining high image quality, especially under adversarial training.

### Strengths
- The paper studies an important and timely problem, i.e., untraceable deepfakes.
- The experiments are comprehensive to a large extent and cover multiple generative and attribution models.
- The paper is well written and easy to follow.
- Code is provided, which should be encouraged. Although in the README the paper states the code is "submitted to AAAI".

### Weaknesses
My main concerns focus on the paper’s theoretical claims about the "multiplicative attack" and the supporting propositions. Currently, the theoretical claims made in this paper do not actually support the strong claims regarding multiplicative attacks being uniquely powerful.

- The paper presents its attack as multiplicative, $x' = x \odot W$, but then immediately defines the adversarial residual as $\Delta x = x \odot (W - 1)$ and constrains $|\Delta x|_2 \le \Delta$. This is algebraically identical to writing $x' = x + p(x)$ with $p(x) = x \odot (W-1)$. In other words, the "multiplicative" formulation is just an input-dependent additive perturbation in disguise. Because all of the theorems and propositions are stated and proved in terms of $\Delta x$ (the residual) and its $\ell_2$ geometry, the results apply equally well to additive attacks $x' = x + p$. Therefore, the paper gives no theoretical justification for the claim that multiplicative attacks are somehow fundamentally superior to additive attacks. In fact, since the theorems themselves are modeled under an $\ell_2$ norm constraint, they appear to fit additive models better than the proposed multiplicative model. At best, the paper demonstrates properties of $x + p(x)$ models, not anything special about strict multiplicative structure.

- Theorem 1 is essentially: assume $F$ is differentiable at $x$ and the Jacobian $J(x)$ has a non-zero direction, then there exists a small $\Delta x$ (with $|\Delta x|_2 \le \Delta$) that changes $F(x)$. I have two concerns regarding this theorem.
  - The assumptions already imply the conclusion. Requiring a non-degenerate $J(x)$ is equivalent to assuming the model is sensitive to some input direction, so the "existence" of a perturbation that changes the output is tautological. This is hardly a new theoretical insight, but rather a trivial existence. 
  - The authors provide no bound on how large the change in the attribution decision will be, nor any probabilistic results about success across AMs. Changing $F(x)$ in the sense of its vector-valued output does not imply flipping the attribution label, nor does it imply removal of a fingerprint. Thus, it is not guaranteed that the attribution decision will be changed and thus the fingerprint removed.

- Proposition 1 and 2 are also problematic and not specific to multiplicative models. The paper argues that multiplicative attacks are "statistically non-invertible" and supports this with a per-pixel linear-Gaussian model $x'_j = x_j W_j + \eta_j$ and a Cramér–Rao lower bound $\mathrm{Var}(\hat W_j) \ge \sigma^2/(N \mathbb{E}[x_j^2])$. This derivation is problematic for multiple reasons. 
  - The noise $\eta$ seems to be artificially introduced. In the attack model the adversary deterministically crafts $x'$; there is no physical measurement noise generated by the attacker. The paper only introduces $\eta$ later so that standard CRLB formulas can be applied. It is unclear why such noise is present and why it would follow the Gaussian distribution. More confusingly, why is $\eta$ assumed to be zero in Proposition 1 but Gaussian in Proposition 2? The decision is confusing, and the specific form seems to be chosen solely to make the proof go through. 
  - CRLB results are standard and also apply equally to additive input-dependent perturbations. The same Fisher-information calculation would hold for $x' = x + p(x) + \eta$ , meaning the bounds are not specific to multiplicative attacks. Thus, the CRLB-based proposition does not establish a unique advantage for multiplicative attacks and only restates that parameter estimation is harder in the presence of noise.

- Finally, the theoretical analysis appears weakly connected to the practical method. The proposed implementation does not rely on an $\ell_2$ constraint but instead uses perceptual loss. The idea of erasing fingerprints by using methods like sampling, transformations, and Gaussian blurring is already explored in prior works such as TraceEvader and StatAttack [1], and given the above concerns, the presented theory does not seem to explain the empirical success of the proposed attack. The rationale behind the design remains unclear. Although the paper claims to require no access to any AMs, the design in fact embeds many human priors about AMs into the attack process.

- The authors wrote "Moreover, using a network to invert images imprints its fingerprints onto the recovered content, further degrading defense efficacy". If this is true, then a trivial attack would be to simply pass the generated image through any unrelated image-to-image model, which would already remove the fingerprint.

- The citation format is incorrect. When citing a work rather than the authors, the paper should use the ``\citep{}`` command (which produces "(Ouyang et al., 2024)") rather than ``\cite{}`` command (which produces "Ouyang et al. (2024)").

Ref:

[1]: Hou et al., Evading DeepFake Detectors via Adversarial Statistical Consistency. CVPR 2023.

### Questions
- Why is $w_i \in \\{0.5, 0.3, 0.2\\}$? Why are $\beta_1, \beta_2, \beta_3$ set as $\\{0.5, 0.1, 0.4\\}$? What happens when these values are changed?
- I also do not understand why the threat model restricts the attacker to having "no access to or information about any AMs". In realistic scenarios, attackers can easily access various AMs as their papers and codes are mostly opensourced. If access to some AMs could enable the attacker to design stronger attacks, then that would be definitely be better.

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
5

### Summary
This paper attempts to develop a defense strategy against attribution attacks. First, it reviews the limitations of existing approaches, which mainly focus on adding perturbations to images without addressing the removal of model fingerprints, the key indicators used for model attribution. Then, the authors introduce a framework for generating multiplicative perturbations, which constitutes the main claimed contribution of this work. However, the paper’s novelty appears limited, and its motivation and overall logic are not clearly articulated. The anonymous code provided in abstract is dead. Further details are discussed below.

### Strengths
The overall writing quality is clear and well-organized, making the paper easy to follow. The visual demonstrations are also well-designed and effectively support the presented ideas, helping readers better understand the proposed method and its outcomes.

### Weaknesses
The primary claim of this paper is that existing approaches depend solely on additive perturbations applied to images, which are insufficient to remove the underlying model fingerprints responsible for attribution. To overcome this limitation, the authors propose an adversarial network incorporating multiple constraint loss terms that consider spatial, spectral, and perceptual aspects.

However, my main concern is that these constraints appear to focus on controlling visual distortions rather than directly addressing the removal of model fingerprints. While the paper provides some theoretical justification, it is not strong enough to convincingly substantiate the central claim.

In addition, the experimental section does not include attribution accuracy results on clean images, making it difficult to assess the inherent challenge of the task or the true effectiveness of the proposed approach. This omission substantially undermines the paper’s contribution. Moreover, although the authors highlight the strength of their method in black-box defense settings, the corresponding results are missing from the table, which further adds to the confusion and raises doubts about the completeness of the evaluation.

Overall, the motivation and contribution of this work remain unconvincing, and in its current form, the paper does not meet the standards expected for acceptance at ICLR.

### Questions
NA

### Soundness
1

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a universal, black-box multiplicative attack to make DeepFakes untraceable by eliminating model fingerprints rather than merely confusing them with additive perturbations. The method trains an image-to-image network on real images augmented with sampling and transformation operations to mimic generative fingerprints; at test time it applies a pixelwise multiplicative mapping to DeepFakes to suppress attribution cues while preserving perceptual quality. Experiments across 12 GMs and 6 attribution models report high attack success and robustness against adversarial training and inversion defenses.

### Strengths
1. Clear articulation of why additive attacks preserve fingerprints; frequency-domain evidence and defensive degradation back this claim.
2. Simple but clever training using sampling/transform ops to mimic GM artifacts without GM access, which is broad applicability.

### Weaknesses
Theory makes local (first-order) arguments and per-pixel noise assumptions. Some modern AMs often include non-linear, patchwise, or frequency pipelines and heavy pre-processing, have you tested its performance on such models?

### Questions
See weaknesses.

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
This paper presents a universal black-box attack method for untraceable deepfakes, that trains an adversarial model solely using real data and applicable for various generative models and agnostic to attributiom models. The experimental results show that the attack success rate of the proposed method is not only high across a wide range of attribution models, but also effective against defense mechanisms.

### Strengths
1.[effectiveness] The proposed method achieves better attack performance than the previous methods, with some loss on SSIM and LPIPS.

### Weaknesses
1.[clarity] In Figure 1, only GAN models are analyzed for the spectral property. Do diffusion models show any similar property? This paper also includes diffusion models apart from GANs, so the diffusion model cannot be missed in this figure.

2.[clarity] This method is effective. However, as a black-box method, its efficiency is also important for discussion. How does the proposed method compare with the previous methods in terms of compute?

3.[ablation] The proposed method balances three parameters in L352 (please also add equation numbers -- this is a typesetting issue). They control perceptual, spatial, and spectral loss functions. In Table 1, I noted that the proposed method's SSIM and LPIPS results are far from the best. If we tune the parameters to improve the SSIM and LPIPS to match with TraceEvador, how does the proposed method perform? Namely, if the proposed method still outperforms TraceEvador at the same SSIM/LPIPS trade-off, the proposed method is strongly better. Otherwise we do not have enough evidence to claim this as a real improvement. At the mean time, the impact of the three parameters is not discussed in this paper.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
