# Divergence-Free Neural Networks with Application to Image Denoising

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
We introduce a resource-efficient neural network architecture with zero divergence by design, adapted for high-dimensional problems. Our method is directly applicable to image denoising, for which divergence-free estimators are particularly well-suited for self-supervised learning, in accordance with Stein's unbiased risk estimation theory. Comparisons of our parameterization on popular denoising datasets demonstrate that it retains sufficient expressivity to remain competitive with other divergence-based approaches, while outperforming its counterparts when the noise level is unknown and varies across the training data.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a theoretically grounded framework for constructing neural networks that are divergence-free for image denoising. The authors propose a resource-efficient parameterization that represents the network output as a structured combination of conservative fields. The proposed divergence-free property is used to simplify SURE loss, avoiding the instability of Monte Carlo approximations and the expressivity limitations of blind-spot methods. The experimental results show that the proposed method achieves better performance against other divergence-based approaches

### Strengths
+ The main strength of this paper is its rigorous mathematical analysis.
+ The method for constructing the sparse, parameterized skew-symmetric matrices and sharing the scalar potential network is reasonable. It successfully makes the theoretical representer theorem computationally tractable for high-dimensional data like images.

### Weaknesses
- The experimental evaluation is focused only on divergence-based self-supervised methods (MC-SURE, Noise2Self, UNSURE). The field of self-supervised (or unsupervised single-image) denoising is much broader and has developed significantly. The paper is missing crucial comparisons to other state-of-the-art methods that are not based on SURE, which makes it impossible to assess the true competitiveness of the proposed approach, e.g., [1-3]
[1] High-quality self-supervised deep image denoising, NIPS 2019
[2] Iterative denoiser and noise estimator for self-supervised image denoising, ICCV 2023
[3] Positive2Negative: Breaking the Information-Lossy Barrier in Self-Supervised Single Image Denoising, CVPR 2025
- The entire framework is demonstrated only for additive white Gaussian noise. While this is a standard setting for theoretical exploration, its practical use is unexplored. The paper itself acknowledges this limitation but does not provide any discussion or preliminary results on how the divergence-free constraint might perform with more complex, realistic noise models (e.g., Poisson-Gaussian, spatially correlated, or signal-dependent noise). The heavy reliance on SURE makes the direct extension non-trivial
- The proposed parameterization is complex, though more efficient than the full model. This significant overhead could be a barrier to practical adoption, and this trade-off is not sufficiently emphasized in the results

### Questions
- The paper argues that the proposed method is more expressive than blind-spot methods because it doesn't ignore the central pixel. However, blind-spot methods are typically blind to the noise distribution, whereas DivFree's theoretical advantage requires knowing the noise level. Could you discuss this trade-off between expressivity and the need for prior information? How does your method perform if the provided noise level is not very precise

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
3

### Summary
SURE needs divergence. To get rid of the divergence, they introduce a constant divergence field S_DC (Eqn 10). Section 4 is about how to generate such a divergence field. The core trick is by means of a universal approximation (Theorem 1). Realization is given in Section 4.2.

### Strengths
Solid contribution.

The coverage of the prior work is excellent.

Intuition of the field construction can be done better. But basically I get the idea that they want to ensure symmetry so that the field is valid (Eqn. 13 and Eqn 14).

Empirical results are okay. This is a piece of theoretical contribution.

### Weaknesses
This is not a criticism but a general comment. Will not contribute to my judgment of the paper.

SURE is a proxy for MSE, which assumes that the underlying noise is iid Gaussian. But if I know that the noise is iid Gaussian, I can literally just simulate training data and train my model in a supervised way. The cost of doing that is null. Even if we say that the noise level is unknown, worst case I just train a larger model with more data augmentation. So I have a hard time to convince myself that SURE is the right way to go. 

I can appreciate unsupervised learning, but it must be some bizarre unknown noise type that I cannot easily calibrate and know from my camera. But for this case, SURE doesn't seem to be the right approach to go.

So I am not sure about the utility down the road. Perhaps denoising is not a good showcase?? No need to address this in the rebuttal. Just some thoughts to share with the authors.

### Questions
No major concerns.

### Soundness
2

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
4

### Summary
This paper proposes a new class of divergence-free denoisers by interpolating the constraints of two prior classes of divergence-free denoisers and also a method for constructing denoisers that belong to the new class directly, without the need to use additional loss functions during training like prior methods. The authors show improved performance with known noise levels, and claim this is due to closer adherence to the divergence-free constraint, as well as comparable performance in the unknown noise level setting.

### Strengths
The idea of examining a class of objects (in this case, denoisers) in between two other existing classes in the literature is very well motivated. I could not find any issues with the theory, which is built on existing solid results in a reasonable way. The results also seem consistent with the understanding of the method presented in the paper, namely that in the known noise-level regime their more constrained method achieves higher performance than the less constrained baseline which does not enforce zero divergence strictly.

### Weaknesses
I'm not sure about the resource efficiency claims. It seems that mostly the case rests on using far fewer terms than necessary as a basis to represent the divergence-free denoiser. I think the paper would be stronger if the authors could show what the exact influence of the number of components, K', is on the denoising performance and how close their denoiser is to being truly divergence free. My understanding of the theory is that it only guarantees a divergence-free field if the number of basis terms is n(n+1)/2, where n is input size, but far fewer basis terms are used for the sake of computational tractability. Additionally, the authors should compare the runtime of the various methods, since it appears that their denoiser requires effectively K'+1 network evaluations, combining forward and backward passes.

### Questions
How does the following depend on K'?
- denoising performance
- runtime
- divergence 

I would be willing to upgrade my score if the authors could provide a more thorough empirical validation of their method, in particular with respect to the major approximation that causes deviations from theory, the number of basis components of their divergence-free denoiser field.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a **Divergence-Free Neural Network (DivFree)** designed for self-supervised Gaussian image denoising.  
The network is parameterized with skew-symmetric bases so that ∇·f(x)=0 holds exactly, thereby eliminating the need for Monte-Carlo divergence estimation used in SURE-based losses.  
The authors claim this leads to more stable training and improved performance compared to MC-SURE and UNSURE.  
Experiments are limited to grayscale Gaussian noise (σ=15–50), with an additional “unknown σ” case that is not truly blind.

### Strengths
- Mathematically consistent derivation of divergence-free vector fields using a skew-symmetric representation.  
- Conceptually simple idea that connects physical conservation laws with SURE-based denoising.  
- Reduces gradient variance by removing stochastic divergence estimation.

### Weaknesses
- **Misleading “unknown σ” claim.** The model is trained with a fixed σ and only tested on slightly different noise levels; there is no random-σ training or blind estimation, so it cannot handle unknown noise.  
- **Extremely narrow scope.** All experiments use grayscale Gaussian noise. There are no color image... The method is not validated on **Poisson**, **Gamma**, or real-world datasets, although its principle directly relates to unbiased risk estimators like **PURE (Poisson Unbiased Risk Estimator)**.  
- **High computational cost.** The paper itself states that inference is about **9× slower** than DRUNet due to repeated gradient evaluations for enforcing divergence-free constraints.  
- **Limited novelty.** The method merely re-parameterizes the denoising backbone to set div(f)=0, optimizing the same SURE objective as MC-SURE. The observed differences stem from reduced variance but increased structural bias.  
- **Lack of generality.** Divergence-free constraints could naturally apply to flow, optical-flow, or magnetic-field problems, yet the paper confines itself to toy Gaussian denoising.

## Minor
- No visualization verifying that learned vector fields are actually divergence-free.  
- Bias–variance trade-off introduced by the constraint is never quantified.  
- Comparison baselines (Noise2Score, Blind2Unblind, Neighbor2Neighbor, diffusion-based denoisers) are missing.

### Questions
1. How is the “unknown σ” setting defined? Was σ randomly sampled during training or fixed to one value?  
2. Could the same architecture be evaluated on **Poisson denoising** using **PURE** to demonstrate generality beyond Gaussian SURE?  
3. Why not apply divergence-free constraints to other tasks (e.g., optical flow, physics-based fields) where this property is physically meaningful?  
4. Does enforcing div(f)=0 introduce measurable bias compared to the true SURE optimum?  
5. Given the reported 9× inference cost, is the stability improvement practically worthwhile?

### Soundness
3

### Presentation
2

### Contribution
1
