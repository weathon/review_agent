# Pro-Trans: Progressive Tensor Ring with Attention Guided Local Smoothing Regularization

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The generalization of adversarial defense methods remains a critical open challenge, and optimization-based adversarial purification methods employing tensor network representations have recently shown strong potential. However, such tensor-based defense methods operate solely on the given input without relying on prior knowledge, which inevitably leads to overfitting to adversarial perturbations. Moreover, their iterative optimization procedures incur substantial computational overhead during inference. In this paper, we propose Pro-Trans, a novel tensor-based adversarial purification method that integrates progressive tensor ring with attention guided local smoothing regularization. Specifically, our progressive tensor ring avoids redundant upsampling operations, thereby reducing computational overhead and accelerating convergence. In addition, the proposed regularizer adaptively applies varying degrees of local smoothing regularization across different regions, thereby suppressing perturbations while mitigating semantic loss. Experimental results show that Pro-Trans consistently outperforms existing methods across diverse adversarial settings on CIFAR-10, CIFAR-100, and ImageNet, achieving state-of-the-art performance while maintaining low computational cost. The code will be available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Pro-Trans, a tensor network-based adversarial purification method that combines Progressive Tensor Ring (PTR) and Attention-Guided Local Smoothing Regularization (AGLSR). The key idea is to avoid redundant upsampling operations in coarse-to-fine optimization by progressively adjusting optimization objectives and trainable cores, while using attention masks to adaptively apply smoothing regularization. The authors evaluate their method on CIFAR-10, CIFAR-100, and ImageNet against various adversarial attacks. The method shows improved robustness accuracy and reduced computational cost compared to TNP, achieving 8.4% RA improvement on ImageNet and 4x speedup.

### Strengths
The paper addresses a relevant problem in adversarial defense and demonstrates clear empirical improvements over the baseline TNP method.

The proposed PTR design is intuitive and achieves the stated goal of reducing computational overhead. The convergence analysis shows PTR converges faster and more stably than existing tensor networks.

The experimental evaluation covers multiple datasets and diverse attack scenarios including cross-dataset, cross-threat, and cross-attack settings, which helps demonstrate generalization capability.

The paper is well-organized with clear figures and comprehensive appendix including implementation details and additional visualizations.

### Weaknesses
The theoretical understanding of why the proposed method works is severely lacking. The authors acknowledge this limitation but provide no theoretical analysis or even intuitive explanations for why PTR achieves more stable convergence or why AGLSR effectively balances perturbation suppression and semantic preservation. For a venue like ICLR, this is a major weakness.

The novelty is limited and primarily consists of engineering improvements. PTR essentially removes interpolation-based upsampling from TNP, which is a relatively straightforward optimization. Using attention masks to guide regularization is not a novel concept. The paper lacks fundamental insights into adversarial purification.

The experimental setup has several concerns. First, using only 512 randomly selected images for evaluation is insufficient for drawing strong conclusions. Second, different methods use different rank settings (e.g., rank 14 for Pro-Trans vs rank 10 for TNP on CIFAR), which raises fairness concerns. Third, the reliance on robust classifiers in some experiments introduces confounding factors.

The method's complexity is concerning. It requires multiple components (PTR, AGLSR, attention extraction from classifier) with multiple hyperparameters. However, there is no sensitivity analysis for key hyperparameters like $\alpha$ in Eq. 6, the number of stages, or rank settings. The dependence on downstream classifier attention also limits the method's independence.

Some experimental results are poorly explained. In Table 1, when using robust classifier, SA drops significantly from 91.99% to 87.69% while RA increases. This trade-off deserves thorough discussion. Why do different datasets require vastly different rank settings (14 for CIFAR vs 50 for ImageNet)?

The paper claims "first coarse-to-fine Progressive Tensor Ring for AP" but PuTT (Loeschcke et al., 2024) also uses coarse-to-fine strategy. The distinction is not sufficiently clear.

The identity tensor initialization in Eq. 3 seems arbitrary. Why Gaussian for first $d$ cores and identity for others? What if we initialize differently?

The AGLSR formulation in Eq. 6 directly multiplies attention mask with image $M \odot Y_d$ before computing TV. This may not be well-justified since TV should measure smoothness of the image itself, not attention-weighted image.

### Questions
Can you provide theoretical analysis or at least intuitive explanations for why PTR converges faster and more stably than TNP? What specific properties of avoiding interpolation-based upsampling lead to these improvements?

How sensitive is the method to the hyperparameter $\alpha$ in Eq. 6? Can you provide ablation studies showing performance across different $\alpha$ values?

Why does using robust classifier decrease SA so much (Table 1)? Is this trade-off acceptable? How does Pro-Trans perform without robust classifier compared to TNP without robust classifier?

Can you justify the fairness of comparing methods with different rank settings? What happens if TNP and Pro-Trans use the same rank?

How do you determine the rank settings for different datasets? Why is rank 14 optimal for CIFAR but 50 for ImageNet?

In Eq. 6, why multiply $M \odot Y_d$ before computing TV rather than computing $M \odot TV(Y_d)$? Have you tried alternative formulations?

Can you extend the evaluation to the full test set rather than 512 images? The current sample size is too small for strong statistical conclusions.

What happens if attention masks are computed from a different classifier? How dependent is the method on the specific classifier architecture?

Can you clarify the relationship between PTR and PuTT more precisely? What makes PTR the "first coarse-to-fine Progressive Tensor Ring"?

### Soundness
2

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
3

### Summary
The paper proposes Pro-Trans, a tensor–network (TN) based adversarial purification method that combines a Progressive Tensor Ring (PTR) optimizer with an Attention-Guided Local Smoothing Regularizer (AGLSR). PTR implements a coarse-to-fine schedule without interpolation-based upsampling, aiming to cut compute and stabilize convergence. AGLSR derives a spatial mask from intermediate activations of the downstream classifier and applies stronger TV smoothing in low-saliency regions to suppress perturbations while preserving semantics. Experiments on CIFAR-10/100 and ImageNet under AutoAttack (ℓ∞, ℓ2) and PGD+EOT report improved robust accuracy (RA) with notably lower per-image purification time compared to prior TN defenses. On ImageNet, Pro-Trans reduces time from ~8s (TNP) to ~3s while improving RA (≈42.8→≈51.2).

### Strengths
- Benchmarks include AutoAttack (ℓ∞=8/255, ℓ2=1.0), PGD+EOT, and multiple classifiers (ResNet-50, WRN-28-10), with tables that compare against AT and prior AP/TN baselines.

- The attention mask helps recover natural accuracy that is lost by naive smoothing, while keeping most of the robustness gains. 

- PTR removes redundant upsampling and shows smoother, faster convergence than PuTT/QTT/QTR; the paper backs this with loss curves and reconstruction metrics.

### Weaknesses
- PTR is chiefly an optimization schedule/design choice rather than a new TN representation; the paper itself notes the lack of a deeper theoretical account of why robustness improves.

- AGLSR requires access to downstream activations to build the attention map; this tight coupling invites end-to-end adaptive attacks that differentiate through the purifier+classifier. 

- Adaptive attacks. PGD+EOT is included, but purifiers typically require a more systematic BPDA/EP-style evaluation that backprops through TV and the PTR steps; without that, robustness claims remain somewhat optimistic. 

- The paper ablates the attention mask qualitatively, but it would help to: (i) test different mask sources (e.g., using a mismatched classifier), (ii) vary α and the TV form, and (iii) show sensitivity to PTR ranks and stage lengths—especially under matched compute to TNP.

### Questions
See weakness as above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the limited generalization of adversarial purification defenses by proposing a novel coarse-to-fine Progressive Tensor Ring method, building upon tensor-based defense strategies. The approach significantly enhances the adversarial robustness of models.

### Strengths
This work presents a novel tensor network-based adversarial purification method, described in substantial detail, which significantly improves the computational efficiency of traditional TN approaches in AP tasks.

### Weaknesses
1. The defense methods compared in Tables 1–4 vary significantly and lack consistency, which hinders the ability to draw unified conclusions. In particular, it is difficult to assess the specific impact of PTR on SA performance based on the presented comparisons.
2. In Section 4.1, it is mentioned that 512 images were randomly selected for testing. Could author clarify the rationale behind this specific sample size? Additionally, it would be helpful to provide further details regarding the selection process and whether all experiments were conducted exclusively on this subset. The limited sample size raises concerns about potential randomness in the evaluation.
3. Based on the experimental results, PTR appears to suffer from significant overfitting to the distribution of adversarial examples. This is evidenced by a notable improvement in both quantitative and qualitative performance after reconstruction, coupled with a clear decline in the SA metric.
4. I have observed that in Table 5, the performance of the method by Nie et al. shows a noticeable decline on CIFAR-100 compared to its results on CIFAR-10. Could you provide details on how these results were obtained?
5. The ablation study appears to be overly simplistic, as several key components of the method have not undergone thorough ablation analysis.
6. There appears to be an inconsistency in the evaluation of reconstructed image quality. While Figure 5 suggests a noticeable degradation in the visual quality of Pro-Trans reconstructions, Tables 8 and 9 only report quantitative results for the higher-quality PTR outputs. This selective reporting of results makes it difficult to draw meaningful conclusions about the actual performance of the Pro-Trans method.
7. The efficiency analysis appears incomplete, as it only includes comparisons with the TN method. To provide a more comprehensive evaluation, comparisons with other adversarial purification approaches—such as those proposed by Yoon et al. and Nie et al.—should be added.

### Questions
See weakness.

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
3

### Summary
This paper presents Pro-Trans, a tensor network–based adversarial purification framework designed to improve both robustness and computational efficiency. The method introduces a Progressive Tensor Ring (PTR) that performs coarse-to-fine optimization within a fixed tensor topology, avoiding interpolation-based upsampling and its associated instability. In addition, an Attention-Guided Local Smoothing Regularizer (AGLSR) is proposed to adaptively smooth different image regions based on attention maps extracted from the downstream classifier, thereby removing perturbations while preserving semantic information.
Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet show that Pro-Trans consistently improves robust accuracy compared with previous tensor-based defenses such as TNP, while also significantly reducing inference time (about three seconds per image instead of eight).

### Strengths
1.	The progressive optimization design within a fixed tensor ring is a thoughtful contribution. It effectively removes the instability and heavy computation of interpolation-based upsampling found in previous coarse-to-fine tensor models. The reasoning behind the design choices is well-motivated and consistent throughout the paper.
2.	Pro-Trans achieves competitive or superior robustness on multiple datasets while maintaining lower computational cost. The convergence analysis and ablation studies provide convincing evidence that PTR improves both stability and speed.
3.	The adaptive local smoothing guided by classifier attention is an elegant way to address over-smoothing and semantic loss. The visualizations clearly show the benefit of using attention masks for targeted denoising.
4.	The experiments are comprehensive, covering different datasets, attacks, and generalization settings (cross-dataset, cross-threat, and cross-attack). The evaluation protocol aligns with the RobustBench standards, and the ablation studies are informative.

### Weaknesses
1.	While the method is well-engineered, it still builds directly on the coarse-to-fine tensor purification framework (TNP). The main innovation, namely progressive optimization without upsampling and adaptive smoothing—are important improvements but not entirely new conceptual directions.
2.	The paper lacks a theoretical explanation of why progressive unfreezing of tensor cores leads to faster and more stable convergence. Similarly, the role of attention-guided total variation regularization could be discussed more analytically.
3.	The robustness results are reported on 512-image subsets due to computational cost, which reduces the statistical strength of the conclusions. It would be reassuring to include mean and standard deviation across multiple runs or a larger evaluation sample.
4.	The mathematical sections are dense and assume strong familiarity with tensor network notation. A brief intuitive explanation or a small illustrative figure might make the ideas more accessible to a broader audience.

### Questions
•  How are gradients and adaptivity handled in the attention-guided purifier under white-box attacks?
•  How is the progressive optimization schedule chosen, and is it robust to different settings?
•  Does the method generalize to different or unseen classifiers producing the attention maps?

### Soundness
3

### Presentation
3

### Contribution
3
