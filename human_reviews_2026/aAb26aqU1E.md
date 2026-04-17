# LearnIR: Learnable Posterior Sampling for Real-World Image Restoration

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Image restoration in real-world conditions is highly challenging due to heterogeneous degradations such as haze, noise, shadows, and blur. Existing diffusion-based methods remain limited: conditional generation struggles to balance fidelity and realism, inversion-based approaches accumulate errors, and posterior sampling requires a known forward operator that is rarely available. We introduce **LearnIR**, a learnable diffusion posterior sampling framework that eliminates this dependency by training a lightweight model to directly predict gradient correction distributions, enabling *Diffusion Posterior Sampling Correction (DPSC)* that maintains consistency with the true image distribution during sampling. In addition, a *Dynamic Resolution Module (DRM)* dynamically adjusts resolution to preserve global structures in early stages and refine fine textures later, while avoiding the need for a pretrained VAE. Experiments on ISTD, O-HAZE, HazyDet, REVIDE, and our newly constructed FaceShadow dataset show that LearnIR achieves state-of-the-art performance in PSNR, SSIM, and LPIPS.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study diffusion-based image restoration, proposing a posterior sampling correction method. They also propose a dynamic resolution module that downsamples/upsamples the input during training, both improving the results and accelerating the training.

The proposed method is evaluated on the task of haze removal and shadow removal, across four different datasets.

### Strengths
- The paper is quite well written, few typos or similar issues.
- The two introduced components, DPSC and DRM, make sense overall and both seem to improve the performance (Table 3, Figure 6).
- The proposed method seems to perform well overall compared to recent baselines.

### Weaknesses
- The proposed method could be presented/introduced better in Section 3. In particular, 3.1 would benefit from more commentary instead of just stacking definitions/theorems, the flow could be improved. The algorithms in 3.3 would also benefit from added descriptions/explanations.
- The experimental evaluation could be more extensive/convincing. Only two different tasks. Results are compared with baselines mostly just in terms of PSNR and SSIM, lacking perceptual metrics. 
- The technical contribution/novelty seems somewhat limited, at least based on my understanding of the proposed approach.

### Questions
Questions/suggestions:
- In the abstract and introduction you write "our new FaceShadow dataset" and "our newly constructed FaceShadow dataset", but then in Section 4.1 you write "three standard datasets: ISTD, [...] and FaceShadow (Zhang et al., 2020)"?
- In Table 1 and 2, why results only in terms of PSNR and SSIM? Why not also LPIPS and/or FID?
- In Section 4.2, ISTD results, it's not clear to me what it means for a method to be mask-free? Also, in the Table 1 caption, should violet and blue be swapped?
- Line 238, "Based on Definitions 1 and 2, Eq. 3 can be expressed": Should this be Eq. 4?
- Could be interesting to evaluate methods also in terms of computational cost during training / at test-time, at least to see the effect of the proposed DRM?
- Which dataset are the images in Figure 5 from?
- Line 260, "As illustrated in Figure 2, the model...": This is not illustrated in a lot of detail in Figure 2 though? Perhaps consider tweaking.






Minor things:
- I think Section 2 could be tweaked to improve the overall flow a bit.
- Figure 4 caption: "(Visual comparisons on the HazyDet datasets)" --> "Visual comparisons on the HazyDet datasets"?
- Line 96, "Experimental results clearly and consistently demonstrate that LearnIR consistently outperform":  "consistently" twice, perhaps reformulate.
- Line 146, "To obtain an form of the": typo.
- Figure 2 caption, "The blue line in Eq. 3 denotes timestep T', computed using the": I don't quite understand what you mean, what blue line?
- Figure 3, "ShadowForme" --> "ShadowFormer"?
- Figure 4, "Dehamer" --> "DeHamer"?
- Line 322, "Hazy" --> "HazyDet"?
- O-Haze is not mentioned at the beginning of Section 4.1? 
- Line 371, "We benchmark against Zhang et al. (Zhang et al., 2020)" --> "We benchmark against (Zhang et al., 2020)", perhaps?

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
This paper proposes the LearnIR framework to address heterogeneous degradations in real-world image restoration, featuring a novel learnable diffusion posterior sampling mechanism. The method trains a lightweight model to directly predict the distribution of gradient correction terms, enabling diffusion posterior sampling correction without requiring a known forward operator. A dynamic resolution module dynamically adjusts image resolution during training, applying large-scale downscaling in early stages to preserve global structures and upscaling later to refine textures. Extensive experiments on multiple real-world datasets demonstrate significant improvements in PSNR and SSIM metrics, particularly in challenging facial shadow removal tasks.

### Strengths
● The learnable posterior sampling correction mechanism eliminates the dependency on a known forward operator, addressing a key limitation of diffusion models in real-world degradation scenarios.
● The DRM effectively balances multi-scale feature extraction and computational efficiency while avoiding the need for a pretrained VAE.
● Extensive validation on multiple real-world datasets, including a newly constructed FaceShadow dataset, provides convincing results.

### Weaknesses
● The proposed method introduces training complexity. Joint training of DPSC and DRM modules requires careful hyperparameter tuning. Also, dynamic resolution switching may introduce training instability, with convergence issues insufficiently addressed.
● The major limitation of diffusion-based posterior sampling method is the scalability to real-world scenarios. However, most current experiments are conducted on synthetic degradation datasets, with the only real-world datasets available being limited to the facial domain. The authors are consider to extend the proposed method to more challenging scenarios such as real-world image super-resolution task and compare with the advanced methods.

### Questions
● Detailed illustrations need to be added for Sec. 3.3.
● The authors are highly recommended to provide image results in supplementary material.

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
4

### Summary
The paper proposes LearnIR, which uses a diffusion posterior sampling correction approach to enable robust real-world image restoration. The proposed approach overcomes the limitation of methods such as DPS which require the explicit modeling of the forward degradation operator. Additionally, LearnIR uses a dynamic resolution module instead of a VAE which allows faster restoration with better performance.

### Strengths
1. The paper is well written and easy to follow.
2. The proposed DPSC method is well designed, intuitive and effective.
3. LearnIR achieves state-of-the-art performance on real-world datasets.

### Weaknesses
1. Lack of ablations regarding the effectiveness of the DRM module (see questions).
2. Limited testing on out-of-domain real datasets to substantiate generalization claim (see questions).

### Questions
1. The proposed DRM module is compared with the frozen SD VAE in Sec. A6.2. However, was the SD VAE frozen and paired with the RDDM model for this experiment? The experiment which needs to be performed would be using the RDDM model with the SD VAE being fine-tuned, as the DRM is trained in LearnIR.
2. In Sec. A6.1, was the DRM also trained with the DiT and DDIM backbones?
3. For the comparisons, were all methods trained on the same datasets as LearnIR?
4. While the authors show results on real-world data, the datasets were used for training LearnIR. To further validate generalization, can the authors provide experiments on other real-world haze datasets (such as [1])?
5. The core approach involves learning to predict score of $p(y|x_t)$. However, this is intractable for real-world degradations and is approximated for learning (Line 144). Could this be considered a limitation of the method and be included as part of limitations?
6. Experiments comparing computation complexities of VAEs and DRM need to be added to substantiate claims in Line 93.

[1] Zhang, Xinyi, et al. "Learning to restore hazy video: A new real-world dataset and a new method." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.

### Soundness
2

### Presentation
4

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
This paper proposes LearnIR, a diffusion-based framework for real-world image restoration that eliminates the need for a known degradation forward operator—a long-standing bottleneck in diffusion posterior sampling (DPS) methods. The key innovation lies in Diffusion Posterior Sampling Correction (DPSC), which introduces a learnable model to predict the gradient correction distribution that replaces the explicit operator used in DPS. This approach allows posterior refinement without requiring analytical knowledge of the forward degradation process. In addition, the authors introduce a Dynamic Resolution Module (DRM) that adaptively adjusts image resolution during training to balance global structure preservation and fine detail generation while avoiding dependence on a pretrained VAE.

The method achieves state-of-the-art results on multiple benchmarks, including ISTD, O-HAZE, HazyDet, and a new FaceShadow dataset introduced by the authors. Quantitatively, LearnIR surpasses recent methods such as ResFusion, ShadowRefiner, and ConvIR in both PSNR and SSIM while maintaining efficiency and generalization to complex real-world degradations

### Strengths
1. The proposed method bypasses the requirement for a known degradation operator in previous works, enabling the application of DPS-related methods in blind restoration tasks. I think this is an important contribution to the research field.
2. The method outperforms recent state-of-the-art methods with a large margin, including ISTD, O-HAZE, HazyDet, and a new FaceShadow dataset.

### Weaknesses
In summary, this is a very sound and impressive work, but with poor presentation. I'm happy to increase my rating based on the writing. 

My main concern lies in the presentation and organization of the paper. While the technical contributions appear sound and potentially impactful, the paper is dense with mathematical formulations, and several notations are either inconsistent or insufficiently explained. These issues make the paper difficult to follow, especially for readers who are not deeply familiar with the DPS-related research line.

1. In Section 3.1, the notation $z_t$ is introduced abruptly without prior explanation. Its definition only appears later in Section 3.2, which disrupts the logical flow and makes it challenging for readers to grasp the progression of ideas. Similarly, the notation $\hat{x}_)$ in Eq. (4) also lacks explanation. I suggest re-checking the presentation so that all key notations are properly introduced before use and accompanied by clear definitions.

2. Theorem 1 claims that Eq. (3) can be expressed as Eq. (16). However, these two equations seem unrelated or at least not directly derivable from one another as currently written. The authors should carefully verify this connection and, if necessary, correct or clarify the statement and its derivation.

3. In step 4 of Algorithm 1, the operator $𝐷^s$ takes $x_0$, $y_0$, and $x^{s-1}$ as inputs, while in other steps (e.g., step 5), it appears to only take $y$ as input. This inconsistency is confusing and suggests either a typographical error or an incomplete explanation of $D^s$'s input structure. A clearer, self-consistent description of the algorithmic steps is required.

Overall, this is a technically solid and promising piece of work with interesting ideas. However, the presentation quality needs significant improvement. The current version suffers from poor organization and inconsistent notation, which hinders readability and comprehension. I would be happy to raise my rating if the authors substantially improve the clarity, consistency, and accessibility of the writing in the revision.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
4
