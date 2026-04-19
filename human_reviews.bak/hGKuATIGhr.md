# DDIL: Improved Diffusion Distillation with Imitation Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 5, 3

## Abstract
Diffusion models excel at generative modeling (e.g., text-to-image) but sampling requires multiple denoising network passes, limiting practicality. Diffusion distillation methods have shown promise by reducing the number of passes at the expense of quality of the generated samples but suffer from lack of diversity, quality, etc. . In this work we identify co-variate shift as one of reason for poor performance of multi-step distilled models from compounding error at inference time. To address co-variate shift, we formulate diffusion distillation within imitation learning **DDIL** framework and enhance training distribution for distilling diffusion models on both data distribution (forward diffusion) and student induced distributions (backward diffusion). Training on data distribution helps to diversify the generations by *preserving marginal data distribution* and training on student distribution addresses compounding error by *correcting covariate shift*. In addition, we adopt reflected diffusion formulation for distillation and demonstrate improved performance, stable training across different distillation methods. We show that DDIL and reflected diffusion formulation consistency improves on baseline algorithms of progressive distillation **(PD)**, Latent consistency models **(LCM)** and Distribution Matching Distillation **(DMD2)**

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper tackles the challenge of lengthy inference times in diffusion models caused by numerous denoising steps.  While prior works like Progressive Distillation and Latent Consistency Matching address this, they often compromise generation quality. The authors attribute this degradation to covariate shift, where the distribution of noisy input latents encountered by the student model during training deviates from the distribution observed during inference.  To mitigate this, they propose DDIL, a method that enhances the training distribution by incorporating both the data distribution (forward diffusion) and student-induced distributions (backward diffusion), drawing inspiration from the DAgger algorithm in imitation learning. DDIL hypothesizes that training on the data distribution preserves diversity by maintaining the marginal data distribution, while training on the student distribution corrects compounding errors by addressing covariate shift. 

I particularly liked the well crafted intuition and parallels to imitation learning, as well as the clear differentiation with prior works in Table 1. Besides, Figure 2 provides a clear and intuitive understanding of the method by demonstrating the teacher and student rollouts and how the losses are computed. The authors tested their method across a couple of prior works in diffusion distillation and have impressive results (improved FID and CLIP score) showing the effectiveness of their approach. While the experiments are only for image generation, the approach is general and could also potentially be applicable to other domains such as audio and video generation thus accelerating research in real-time media generation. The paper could have been improved with some more details and discussion around limitations and future work. Overall, I think the work is novel and well articulated. I would be in favor of accepting the paper with a few asterisks / improvements in the paper that I have listed in the following sections.

### Strengths
The paper proposes a novel approach drawing parallels from Imitation learning and specifically Dataset Aggregation (DAgger) inspired training distribution augmentation, which is well tested in literature both theoretically and empirically. The paper conducts thorough experiments on standard benchmarks, comparing DDIL against multiple established distillation techniques. The authors also show that the approach is not just applicable on one method but across a few diverse methods showing it is generally applicable. The results demonstrate consistent improvements in FID and CLIP scores, supporting the effectiveness of the proposed approach. The paper is well-written and organized. The motivation, methodology, and results are presented clearly. The figures are effective to illustrate key concepts. The problem of addressing covariate shift in diffusion model distillation is crucial for improving the efficiency and practicality of these models.

### Weaknesses
* **Missing details on Sampling Priors (β):** 
   * The algorithm mentions user-defined sampling priors (β_frwd, β_teach_bckwrd, β_student_bckwrd) to control the selection of intermediate latent variables. However, it does not specify how these priors are defined, updated, or scheduled during training. This is a crucial detail for understanding the practical implementation of DDIL and its behavior. The authors should provide more detail / ablations are needed on how these are initialized and potentially adjusted during training (e.g., based on a schedule, the student's performance, etc.) on how changes in these priors impact results.
* **Lack of qualitative discussion:**
    * Figure 1, though included, is not referenced in the paper. The authors should consider elaborating and explaining the qualitative improvements achieved by DDIL across the different scenarios depicted in the figure, focusing on a few specific examples, highlighting how DDIL improves specific aspects of image quality or fidelity compared to the baseline methods. This would strengthen the understanding of its effectiveness and justify the figure's inclusion.
* **Reflected diffusion and Thresholding:**
    * The paper mentions the use of "reflected diffusion" and thresholding for stability. While the concept is explained in the text, the algorithm itself does not show where and how the thresholding is applied (e.g., to the teacher's score estimates, the student's predictions, or both).
        * It would be better if it is also reflected and explained in Algorithm 1 on how it fits into the overall method. While the concept is introduced, the practical details of thresholding are unclear. What specific threshold values are used? How are they chosen or adjusted? The authors should add more details about these questions in the paper and Algorithm 1.
    * While the authors mention improved stability and performance, they lack specific quantitative results isolating the impact of thresholding on covariate shift mitigation. (e.g., measuring gradient variance, loss fluctuations)
* **Limited analysis of diversity preservation:** 
    * In Table2, the diversity / LPIPS score for PD does not seem to degrade but that of LCM degrades a bit with DDIL.
    * Is there any insight or empirical results showing if that is always the case for PD?
    * Are these reported results statistically significant?
    * There is a small section in the end talking about diversity vs. quality but I'd encourage the authors to elaborate more on their diversity results including statistical significance tests and how they relate to this tradeoff along with any future considerations, and discussion of why DDIL might affect diversity differently for PD and LCM. 



**General suggestions for improvements:**
  * Expanding the discussion on the limitations of DDIL and potential future research directions would improve the completeness of the paper.
  * In Table 2, the authors could break down individual ablations and show delta gains to highlight the effectiveness and changes from their experiments better. Specifically, introducing another small delta column and breaking down the LCM experiments in Table 2 separating them with the PD results below through lines and highlighting the delta with base.
  * The authors should also consider citing: “Photorealistic text-to-image diffusion models with deep language understanding.” during the discussion of reflection / thresholding which was an earlier work to alter the diffusion sampling process through thresholding.



**Things that could be improved but did not impact the score:**


* Typos & Spelling Errors:

    * Line 63: "a white robot, a red robot and a black robot standing together"  ->  "a white robot, a red robot, and a black robot standing together" (missing comma before "and")
    * Line 80:  "motorcyle" -> "motorcycle"
    * Line 110: "on on both" -> "on both" (duplicate word)
    * Line 177: "excarbates" -> "exacerbates"
    * Line 184: "prounced" -> "pronounced"
    * Line 314: "threshold-ed" -> "thresholded"
    * Line 449: "Ss discussed" -> "As discussed"

* Grammar & Clarity Suggestions:

    * Line 42:  "when the number of denoising steps are low" -> "when the number of denoising steps is low"
    * Line 174: "With in iterative" -> "Within iterative"
    * Line 178: "in accumulation of error" -> "in an accumulation of errors"
    * Line 226-233: This paragraph about "Mode Seeking" is somewhat confusingly worded. Rephrasing to focus on the core problem of reduced diversity and its cascading effects would improve clarity.
    * Line 376: "In this work we consider" -> "In this work, we consider"

### Questions
* The authors have drawn parallels more towards imitation learning, were any other or more recent SOTA imitation learning methods tried? If not, why? Some considerations are: 
    * SafeDAgger, 
    * Generative Adversarial Imitation Learning (GAIL): GAIL and its variants (e.g., AIRL (Adversarial Inverse Reinforcement Learning)) that use a discriminator to distinguish between expert and student trajectories. This adversarial training can lead to more robust policies and better generalization. While the paper mentions adversarial losses in the context of distribution matching, applying a GAIL-style approach directly within the imitation learning aspect of DDIL could be interesting.
* Missing details on the sampling priors and how were they chosen?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work addresses the limitations of the score distillation of diffusion models. Specifically, the authors identify the imbalance of mode-covering and mode-seeking behavior due to the current design of the distillation methods, resulting in trade-offs of performance and inference NFEs.

To counter this, they propose Diffusion Distillation in Imitation Learning (DDIL), which trains on both the data reference distribution to preserve diversity and the student-induced reference distribution to address balance of mode-covering and mode-seeking. By combining these distributions, DDIL enhances performance across various distillation methods, including Progressive Distillation (PD), Latent Consistency Models (LCM), and Distribution Matching Distillation (DMD2). The results show that this approach enhances both quality and diversity, providing a more practical solution for efficient diffusion distillation.

### Strengths
- The proposed method is overall simple, straighforward but effective. The idea of imitation learning naturally construct the student-induced reference distribution to enhance the mode-seeking behavior in the distillation.

- The paper is overall well-written and easy to read. 

- The proposed DDIL method can be naturally introduced in a variety of existing distillation methods.

### Weaknesses
- The discussion of co-variate shift in section 3 is fairly weak and the authors lack of corresponding justification of this argument in the following experiments. It is not clear where the co-variate shift exists in each score distillation methods and how does it affect the final performance. Moreover, it is confusing how DAgger approach is related to DDIL and how the use of total variation distance connects the loss function in Eq (2). 

- The training on the backward trajectory also requires additional sampling of the trajectory. The authors should discuss the additional cost in this part. Moreover, it is non-trivial to discuss or conduct ablation studies on how the sampled trajectory affects the distillation.

- Considering the default of using only forward or reverse KL divergence in the distillation, there are previous works that inspired by other distribution measure such as combining forward KL with GANs (Kim et al, 2023), Fisher divergence (Zhou et al, 2024). The authors may need to carefully address the differences and connections with these previous methods, and also include them into comparison. 

```
Ref:

Kim, Dongjun, et al. "Consistency trajectory models: Learning probability flow ode trajectory of diffusion." arXiv preprint arXiv:2310.02279 (2023).

Zhou, Mingyuan, et al. "Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation." Forty-first International Conference on Machine Learning. 2024.
```

- The evaluation is only conducted on text-to-image generation, while the performance on unconditional/class-conditional generation remains unknown.


-  The improvement of using DDIL on top of Progressive Distillation and LCM in terms of the generation quality, and the improvement on top of DMD2 in terms of diversity is marginal.

### Questions
Please see Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper tries to solve the diffusion models’ limitation of multiple denoising network passes and identify the co-variate shift as one of the main reasons. To address co-variate shifts and achieve a better performance of diffusion models, the authors proposed Diffusion Distillation with Imitation Learning (DDIL). DDIL draws inspiration from the Dataset Aggregation (DAgger) method and alleviates the co-variate shift by incorporating the diffusion distillation process into an imitation learning framework. It trains the student model on the data distribution (forward diffusion) and the distribution generated by the student model (backward diffusion). Besides, the authors introduce the reflected diffusion formulation and apply thresholding to the score estimates of both teacher and student models.

### Strengths
The research direction is relatively interesting, and the review of related work is thorough.

### Weaknesses
1. In Line 38, the authors claim that 'Multi-step student models offer a promising approach in balancing quality and computational efficiency. However, they often face a critical challenge: covariate shift.' However, there is no theoretical or experimental evidence provided to substantiate the covariate shift issue. Please validate the 'covariate shift issue', such as visualizing the potential distribution of different time steps or measuring the distribution offset between training and inference, etc. Besides, the experimental results only show improvements in quantitative metrics for generated images with the proposed method, without directly addressing or resolving the covariate shift problem.

2. Commonly used metrics, such as the Inception Score, and additional datasets beyond COCO, such as CIFAR100 and ImageNet, etc., should be employed to validate the experimental conclusions drawn in this paper.

3. I can not conduct a comprehensive review of the technological accuracy of this paper, as it presents an empirical rather than theoretical approach, and the implementation code is not provided (particularly the implementation code of Algorithm 1).

### Questions
Please refer to the Weaknesses and Questions above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper identifies covariate shift along the sampling trajectory as a significant challenge in diffusion model distillation. To address this issue, the authors propose DDIL, a method that enhances the training distribution of noisy latents through three distinct sources: (1) forward diffusion of real images, (2) denoised latents produced by the teacher model, and (3) latents generated by the student model. By incorporating teacher feedback into the student trajectories, DDIL effectively mitigates covariate shift that may result from errors or suboptimal score estimates of the student model, particularly during the initial steps.

### Strengths
- Novelty: The paper highlights that errors accumulated during the early stages of sampling can exacerbate covariate shift and introduce potential biases. Section 3.2 and Figure 2 effectively illustrate this issue and demonstrate a solution through querying the teacher model during the reverse sampling process. The construction of a teacher-student symbiotic training dataset is a novel approach that could generate beneficial synergies.

- Quantitative Results: The effectiveness of DDIL is validated using three state-of-the-art distillation models—LCM, Progressive Distillation, and DMD2—that represent commonly used groups in distillation modeling, whether directly predicting the ODE endpoint or the endpoint of sub-intervals.

### Weaknesses
- Lack of Empirical Analysis & Ablation Study: While the proposed “Teacher Backward” mechanism represents a novel component in Algorithm 1, the use of (2) real images and (3) student backward methods are less innovative, with student backward resembling a consistency constraint. Thus, an ablation study isolating the impact of Teacher Backward is essential to validate the framework. For instance, what results emerge when comparing the use of (2) real images + (3) student backward against the combined approach of (1) Teacher Backward + (2) + (3) under a fixed computational budget? Although there is some analysis related to covariate shift in Tables 2, 3, and 4 in the appendix, it primarily addresses stochastic mixing of teacher and student outputs during inference. Additionally, these tables are difficult to interpret due to conflicting trends between FID and CLIP scores; the authors prioritize CLIP scores, basing claims on their trends. A more thorough examination of the Teacher Backward component in DDIL is warranted.

- Metrics & Experimental Settings: Given that traditional metrics like FID and CLIP scores may not fully capture human preference or image quality, recent distillation studies often incorporate alternative metrics for assessing high-frequency details or human preference (e.g., Image reward [1], Patch FID [2], etc.). Such metrics are required here, as the DDIL improvements is apparently marginal in the CLIP and FID trends. Furthermore, the qualitative results (e.g., Figure 1 and related figures in the appendix) do not convincingly demonstrate improvement. Additional comparisons and a large-scale user study with uncurated samples would further support the findings.

- Writing Quality: The overall writing quality is subpar, with numerous typos, incomplete discussions, and key quantitative results relegated to the appendix despite available space in the main text. For instance, the discussion of the diversity-quality tradeoff in Line 447 is vague—does unrolling with the teacher improve or impede diversity? It appears that DDIL reduces diversity based on $LPIPS_{Diversity}$, despite using a teacher network. Additionally, baseline (student-only) results are missing in Tables 3 and 4 in the appendix, yet the authors assert that teacher intervention improves quality. A precise mathematical definition of covariate shift would help reader comprehension, as interpretations may vary from dataset-related shifts to time-dependent sampling path discrepancies.

---
**References**

[1] Xu, Jiazheng, et al. "Imagereward: Learning and evaluating human preferences for text-to-image generation." Advances in Neural Information Processing Systems 36 (2024).
[2] Lin, Shanchuan, Anran Wang, and Xiao Yang. "Sdxl-lightning: Progressive adversarial diffusion distillation." arXiv preprint arXiv:2402.13929 (2024).

### Questions
Overall, the paper presents a novel concept with promising quantitative results; however, the quality and structure could be refined. Below are some example questions highlighting areas of weakness and corresponding suggestions (please refer to weakness part for full details):

- Ablation Study: How significantly does the Teacher Backward mechanism enhance distillation performance? What is the associated increase in computational cost, and does this approach reduce the diversity of outcomes?

- Metric Trends: Have you considered evaluating performance using additional metrics, such as ImageReward or Patch FID? Can you explain why CLIP and FID trends are contradictory in some tables?

- Alternative Approaches: The authors utilize teacher intervention to improve the training distribution. Have you explored the possibility of employing pre-trained distillation models, such as DMD2 or SDXL-lightning, in place of the teacher for unrolling? Since student models often approximate the integral over sub-intervals, they could also potentially rectify sampling trajectories and mitigate covariate shift but at a lower computational cost.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper aims to address the covariate shift issue in diffusion distillation by employing data augmentation techniques. The authors propose utilizing samples from the data distribution, as well as from the teacher and student model distributions, during the distillation process. Additionally, a thresholding technique from prior work is incorporated to enhance training stability. While the proposed approach is straightforward, the empirical results do not demonstrate significant improvements.

### Strengths
1. The proposed method is conceptually simple and easy to understand.

### Weaknesses
1. **Figure 1 Analysis**: The visualization in Figure 1 does not convincingly show the superiority of the proposed method. The color scheme and styling are too similar, making it difficult to distinguish improvements. Additionally, the minor differences in output images could potentially be explained by different random seeds.
2. **Covariate Shift Justification**: The explanation regarding covariate shift due to the accumulation error in the backward diffusion process is unconvincing. The paper would benefit from citing relevant literature that supports this claim.
3. **Method Complexity**: The proposed method, which introduces data augmentation into the diffusion distillation process, lacks sufficient novelty for an ICLR submission. The primary contribution—utilizing samples from the data, teacher, and student distributions—resembles more of a practical trick than an innovative approach.
4. **Empirical Results**: The empirical performance is underwhelming, with improvements that are often marginal. Although the paper claims enhanced generation diversity, the results frequently indicate a decline in diversity.
5. **Writing Quality**: The manuscript contains numerous typos and grammatical errors, indicating a lack of careful proofreading.

typo:
1. Abstract: "We show that DDIL **consistency** improves on baseline"
2. Line 108: We propose a novel DDIL framework which enhances training distribution of the diffusion distillation within the dataset aggregation ‘DAgger’ framework by performing distillation **on on** both the data distribution (forward) ...
3. Line 176: which is classic feedback loop **[1]?** in imitation learning
4. ...

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
