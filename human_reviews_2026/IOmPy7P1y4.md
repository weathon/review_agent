# Learning Heterogeneous Degradation Representation for Real-World Super-Resolution

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6, 6

## Abstract
Real-World Super-Resolution (RWSR) aims to reconstruct high-resolution images from low-resolution inputs captured under complex, real-life conditions, where diverse distortions result in significant degradation heterogeneity. Many methods rely on degradation representations, yet they struggle with the lack of spatially variant degradation modeling and degradation-content entanglement. We propose Spatially Amortized Variational Learning (SAVL), an implicit framework that models per-pixel degradations as spatially varying Gaussians inferred from local neighborhoods. SAVL couples a conditional likelihood lane (SAVL-LM) with a mutual information suppression lane (SAVL-MIS) to filter out degradation-irrelevant signals, yielding a well-constrained solution space. Both our qualitative visualizations and quantitative analyses confirm that the learned representations effectively capture the spatial distribution of complex degradations while being highly discriminative of diverse underlying degradation factors. Building on these representations, we design a degradation-aware SR network with channel-wise guidance and spatial attention modulation for adaptive reconstruction under heterogeneous degradations. Extensive experiments on real-world datasets demonstrate consistent gains over prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper works on real-world super-resolution and proposes Spatially Amortized Variational Learning (SAVL), which learns pixel-wise degradation representations through spatially varying Gaussian posteriors inferred from local neighborhoods. SAVL integrates two components: SAVL-LM, for conditional likelihood learning via an amortized variational bound, and SAVL-MIS, which suppresses mutual information to disentangle degradation from content. The learned representations guide the downstream SR network via spatial and channel modulation. Experiments on both synthetic and real datasets (e.g., RealSR, DRealSR, SVSR) show moderate improvements over prior methods such as LightBSR and CDFormer.

### Strengths
1. The paper explores the relatively underexplored direction of modeling spatially heterogeneous real-world degradations through pixel-wise variational distributions. Using degradation learning as an amortized variational inference problem and integrating information-theoretic regularization could inspire follow-up work.
2. The proposed SAVL framework and its joint training strategy are well-formulated and explained in detail.
3. The authors provide experiments and visual analyses to demonstrate both the quality of the learned degradation representations and their impact on super-resolution performance.

### Weaknesses
1. The comparisons rely mostly on older baselines (e.g., HAT, StableSR, CDFormer) and do not include recent stronger diffusion-based super-resolution models. It therefore remains unclear whether the proposed method truly advances the state of real-world super-resolution.
2. While qualitative heatmaps and t-SNE plots are shown, quantitative validation of spatial degradation modeling is limited. The experiments appear to rely on simplified or toy setups, and the actual benefit of the per-pixel Gaussian modeling requires more investigation.
3. The evaluation primarily uses reference-based metrics (PSNR, SSIM, LPIPS), without sufficient inclusion of no-reference metrics (e.g., MANIQA, MUSIQ, CLIP-IQA) that are more appropriate for real-world settings.
4. The paper’s presentation quality could be improved. For instance, improving figure readability (fonts, labels) and providing clearer, more descriptive captions.

### Questions
1. It is better to provide more evidence on synthetic spatially varying degradations (e.g., blur kernels, noise maps) with quantitative evaluation of SAVL’s per-pixel degradation accuracy.
2. Include additional visual comparisons with recent diffusion-based SR methods to better assess SAVL’s performance against current state-of-the-art approaches.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper suggests framework for learning degradation representation which effectively captures the spatial distribution of complex degradations and also can be decoupled with contents of the image. Using the proposed degradation representation, paper outperforms the previous baseline with clear margin.

### Strengths
* The paper suggests the good motivation. Previous degradation representation learning methods didn't consider that 1) degradations are not uniformly distributed within an image and 2) the degradation should be decoupled with the content. These two limitations are clearly shown in the Fig 1. and the experiments.

* Paper achieves fair contribution to mitigate aforementioned issues with mathematical demonstration.

* Experiments and ablation study is thorough, demonstrating the effectiveness of paper's framework.

### Weaknesses
* In Tab. 1, what does 'CLUB' mean here? It is not mentioned in the previous section. Which part of the method section is related with this?
* The explanation about the experiment results is too brief. There is only one sentence for each table from Tab.1 to Tab. 2 which is like "the proposed method outperforms the previous methods".
* In the right hand side of the image, I can't find clear improvement of the paper compared to the baseline.

### Questions
Refer to the weakness section.

### Soundness
3

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
This paper introduces a novel framework called Spatially Amortized Variational Learning (SAVL) to address some challenges of Real-World Super-Resolution (RWSR), which are spatial variance and entanglement between degradation and image content. SAVL models per-pixel degradations as spatially varying Gaussian distributions inferred from local neighborhoods. It combines a conditional likelihood module (SAVL-LM) with a mutual information suppression module (SAVL-MIS) to disentangle degradation from content. This yields a well-constrained latent space that effectively captures degradation heterogeneity. The learned representations are then integrated into a degradation-aware super-resolution network using channel-wise and spatial attention modulation.

### Strengths
1. SAVL introduces a per-pixel Gaussian representation that captures spatially varying degradations across an image. This allows the model to handle real-world scenarios where degradation is non-uniform and locally complex, outperforming traditional global or patch-based approaches.
2. By incorporating a mutual information suppression term into its variational objective, SAVL explicitly filters out content-related signals from the degradation representation. This results in a latent space that is highly discriminative of degradation factors while minimizing entanglement with image semantics.
3. The presentation of the ideas is relatively clear and easy to follow. The illustraions are good, especially figure 1, though there are some formats issues.

### Weaknesses
1. There exists severe format issues in Figure 2, which is very improfessional and impacts reviews' understand of the content. For the details, please see the question part. 
2. The paper claims it outperforms state-of-the-art methods, but the most recent SOTA approach in the comparison was published on CVPR 2024, which was 2 years ago. I suggest it should include more updated works, perhaps published on CVPR 2025 and ICCV 2025, in both quantitative and qualitative comparisons.
3. Qualitative comparisons of SR results (Fig. 6 and 8) are all conducted on old methods, no visual comparisons with more up-to-date methods such as DiffBIR, StableSR, and CDFormer.

### Questions
1. There are many latex encoding errors in Figure 2, displaying as a "?" with bounding box below the text "Amortized Estimator", after the text "Per-pixel Gaussian sample", and above several arrors. I don't understand the original notation at these places.
2. In Section 3.4, the paper claims "the posterior variance guides spatial attention,while the posterior mode (mean) guides channel attention". Please provide more explanation and analysis on this design.
3. The proposed SAVL aims to solve both spatial variance of degradation and entanglement between degradation and image content, but I only see experimental verification of the later but no justification for the former. Do you provide any numerial or visual evidence to verify that your proposed spatially varying Gaussian modeling solves this chanllenge?

### Soundness
3

### Presentation
2

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
This paper addresses the challenge of modeling complex, spatially-variant degradations and the common issue of "degradation-content entanglement" in Real-World Super-Resolution (RWSR). The proposed Spatially Amortized Variational Learning (SAVL) framework models per-pixel degradation as a spatially varying Gaussian distribution. SAVL uniquely couples two components: A conditional likelihood lane (SAVL-LM) learns the degradation representation by maximizing the Evidence Lower Bound (ELBO). A mutual information suppression lane (SAVL-MIS) explicitly disentangles this representation from image content. Building on this disentangled representation, the learned Gaussian's mode (mean) and variance are leveraged to guide a degradation-aware SR network through channel-wise and spatial attention modulation, respectively. Extensive experiments demonstrate that this approach successfully captures the spatial distribution of complex degradations, leading to consistent, state-of-the-art performance on real-world datasets.

### Strengths
1. The paper identifies degradation–content entanglement as RWSR’s key bottleneck, especially under spatially varying degradations, and addresses it with an optimization objective under an information-theoretic constraint.
2. The SAVL framework offers a principled, theoretically-grounded solution for disentanglement. By modeling degradation as a probability distribution, it naturally captures uncertainty and "severity" (via the variance) and cleverly uses the posterior's mode and variance in a dual-guidance mechanism for channel and spatial modulation.
3. Extensive experiments demonstrate that the proposed approach accurately captures the spatial distribution of complex degradations and achieves consistent state-of-the-art performance on real-world datasets.
4. This paper is well-written and well-structured, with clear motivation.

### Weaknesses
1. While the posterior variance is intuitively interpreted as degradation "severity," the meaning of the posterior mean (the "mode") remains less explored. The paper suggests it "characterizes the degradation type," but the 64-dimensional latent space is treated as a black box. A deeper analysis, for instance, investigating whether specific dimensions or subspaces correlate with distinct degradation factors (e.g., blur, noise, compression), would further strengthen the paper's claims and provide valuable insights into the learned representation.
2. The SAVL framework is trained on paired data, which in practice is generated by applying a synthetic degradation pipeline (from Real-ESRGAN) to high-quality images. While this is a standard practice, the generalization capability of the learned representation is ultimately bounded by the diversity and fidelity of this synthetic pipeline. Generalization may be limited when real-world degradations are mismatched with or underrepresented in the synthetic training set.
3. Although the GFLOPs reported in Table 2 are competitive, the SAVL framework introduces additional complexity through variational inference and mutual information regularization. A brief discussion on training stability, convergence speed, and the overall training time/cost compared to simpler implicit representation learning methods would be beneficial.

### Questions
1. Have the authors conducted any experiments to probe the learned 64-D mean space? For example, by systematically varying a single degradation parameter (e.g., the kernel size of a Gaussian blur) while keeping others constant, is it possible to observe a consistent, predictable change along certain dimensions of the mean vector?
2. The final objective (Eq. 12) simplifies the mutual information (MI) upper bound into a KL divergence term by aligning the priors. Did the authors experiment with using other direct MI estimators, such as CLUB (which was used in the ablation study), as a regularizer in the main objective instead of the VIB-style upper bound? How would integrating such an estimator directly into the SAVL framework compare to the current formulation?
3. How does the model perform on entirely out-of-distribution degradation types not seen during training (e.g., motion blur, atmospheric haze, or specific sensor noise patterns)? Does the probabilistic nature of SAVL provide enhanced robustness to such domain shifts compared to deterministic methods?
4. In Section 3.4, it is stated that the variance map guides spatial attention by re-weighting the scores in SW-MSA. Could the authors provide more specific details on this operation? Is it a simple element-wise multiplication of the attention scores with the (processed) variance map, or is a more complex fusion mechanism employed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper aims to tackle two crucial issues in Single-Image Super-Resolution: inability to model spatially variant degradation and difficulty of degradation-content decoupling. To this end, it proposes Spatially Amortized Variational Learning (SAVL) algorithm, which models per-pixel degradations as spatially varying Gaussians inferred from local neighborhoods. Meanwhile, SAVL imposes an explicit mutual information constraint to suppress the degradation-content entanglement.

### Strengths
1. The paper is motivated well and modeling of spatially variant degradation is indeed crucial for single image super resolution. 
2. The proposed Spatially Amortized Variational Learning (SAVL)  framework models per-pixel degradations as spatially varying Gaussians, which is a novel design.

### Weaknesses
1. More visual comparisons for ablation study are favorable to show the effectiveness of the proposed SAVL in 1) modeling the spatially variant degradations and 2) mitigating the degradation-content entanglement.

2. How to perform quantitative evaluation on the effectiveness of SAVL on degradation-content decoupling? It is indeed not straightforward to conduct such evaluation whilst it is important.

Minor issues:
Figure 1 and Figure 5 are too small to deliver a good presentation.

### Questions
Check the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
