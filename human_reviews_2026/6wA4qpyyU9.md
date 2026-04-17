# Directional Textual Inversion for Personalized Text-to-Image Generation

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Textual Inversion (TI) is an efficient approach to text-to-image personalization but often fails on complex prompts. We trace these failures to embedding norm inflation: learned tokens drift to out-of-distribution magnitudes, degrading prompt conditioning in pre-norm Transformers. Empirically, we show semantics are primarily encoded by direction in CLIP token space, while inflated norms harm contextualization; theoretically, we analyze how large magnitudes attenuate positional information and hinder residual updates in pre-norm blocks. We propose Directional Textual Inversion (DTI), which fixes the embedding magnitude to an in-distribution scale and optimizes only direction on the unit hypersphere via Riemannian SGD. We cast direction learning as MAP with a von Mises-Fisher prior, yielding a constant-direction prior gradient that is simple and efficient to incorporate. Across personalization tasks, DTI improves text fidelity over TI and TI-variants while maintaining subject similarity. Crucially, DTI's hyperspherical parameterization enables smooth, semantically coherent interpolation between learned concepts (slerp), a capability that is absent in standard TI. Our findings suggest that direction-only optimization is a robust and scalable path for prompt-faithful personalization. Code is available at https://github.com/kunheek/dti.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the text-to-image customization problem, leveraging a textual inversion method. 

The authors showcase 2 aspects about textual inversion: the OOD magnitude of the inverted vector and the direction importance for the semantic meaning. Specifically, norm inflation or direction drift causes poor text alignment or object's poor fidelity in generated images. 

As a solution, the authors propose to fix the magnitude of the inverted vector (aligned with the average in the vocabulary) and optimize only the direction.

### Strengths
1. Explanations for some of the known textual inversion problems (Effect I and Effect II).
2. Theoretical support of the claims (Proposition 1 and Corollary 1).
3. Despite some theoretical complexities, the method implementation is simple and flexible. No heavy finetuning required for diffusion models.

### Weaknesses
1. Qualitative comparisons are mostly or even only done with the baseline Textual Inversion method. Many improved or advanced methods are neglected in qualitative comparisons.
2. The proposed method's quantitative results are very close to the results of advanced textual inversion methods (NeTI, P+, etc), and the improvement is marginal.
3. There is no a limitations section or failed examples.

### Questions
1. Would you please provide more details about the norm inflation plot in Figure 1 (dataset, text encoders, etc.)?
2. In Proposition 1, there is a condition when $$||x^{(0)}|| > S_{L}$$ 
What if this condition is not met? What is the guarantee that the initial embedding's norm will be greater than the sum of all B's?
3. There is no description of S^{d-1} in Expression 1. Would you provide more details and descriptions about it?

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
5

### Summary
This paper focuses on enhancing personalized text-to-image generation by addressing critical limitations in Textual Inversion (TI), specifically the issue of embedding norm inflation that leads to poor prompt fidelity in complex scenarios. The authors first identify this bottleneck through empirical observations, such as the semantic drift and out-of-distribution magnitudes in learned tokens, and provide a theoretical analysis demonstrating how large norms attenuate positional information and cause residual stagnation in pre-norm Transformers. Building on this, they propose Directional Textual Inversion (DTI), a novel framework that fixes embedding magnitudes to in-distribution scales and optimizes solely the directional component on the unit hypersphere via Riemannian SGD, incorporating a von Mises-Fisher prior for semantic regularization.

### Strengths
The paper exhibits several notable strengths that enhance its scholarly contribution. 

First, the analysis of Textual Inversion (TI) limitations is particularly insightful, as it introduces a new perspective by identifying and rigorously examining embedding norm inflation—a previously underexplored issue. This is supported by empirical evidence, such as the demonstration of out-of-distribution norms and semantic drift, and complemented by theoretical foundations that explain how large magnitudes disrupt Transformer dynamics, providing a comprehensive understanding of TI's failures.

Second, the technical pathway of Directional Textual Inversion (DTI) is well-justified and methodologically sound, building logically on the identified problems. By decoupling magnitude and direction, employing Riemannian SGD for hyperspherical optimization, and incorporating a von Mises-Fisher prior within a MAP framework, the approach effectively addresses the core issues, demonstrating a coherent and innovative design.

Third, the paper is exceptionally well-written, with clear organization and rigorous experimentation across diverse settings, including various personalization tasks, and comprehensive evaluations through metrics and human assessments.

### Weaknesses
The paper's introduction emphasizes TI's poor performance with complex prompts, yet the subsequent technical analysis in Section 2 appears decoupled from this specific issue. The theoretical framework primarily explains general performance degradation due to norm inflation but does not explicitly model or analyze the distinct challenges of compositional prompts, creating a slight disconnect between the stated motivation and the core technical solution.

While the paper claims TI is an "efficient alternative," this claim warrants critical examination. The requirement of approximately 7 minutes per subject for SDXL and 30 minutes for SANA, as mentioned in the reproducibility statement, along with TI's inherent limitation to optimizing a single concept per embedding, challenges its practicality for real-time applications or scenarios requiring multi-concept personalization. The efficiency argument is relative to full model fine-tuning but may not hold against more recent parameter-efficient methods.

The qualitative results, while generally supportive, exhibit inconsistencies that merit discussion. For instance, in the first column of Figure 2, the generated bears show noticeable variation in appearance, compromising subject consistency. Furthermore, the intended "wizard costume" attribute is not convincingly realized in the outputs, highlighting a potential limitation in capturing specific, fine-grained attributes through directional optimization alone.

A more comprehensive quantitative comparison is needed to fully situate DTI within the current landscape. The evaluation primarily benchmarks against standard TI and CrossInit. Including comparisons with a wider range of contemporary methods—such as other TI variants (e.g., P+, P*, NeTI), lightweight fine-tuning approaches like LoRA, or even full fine-tuning methods like DreamBooth on key metrics—would provide a clearer picture of DTI's relative advantages and trade-offs in terms of both performance and computational cost.

The theoretical analysis, though valuable, relies on the assumption of bounded sub-layers (Section B.1). The practical validity and tightness of these bounds in modern, large-scale text encoders are not empirically verified. Exploring how these bounds behave in practice could strengthen the theoretical claims or reveal conditions under which the effects of norm inflation are more or less pronounced.

The selection of the vMF prior's concentration parameter κ, while justified via a grid search, is presented as a fixed value (1e-4). The paper does not explore whether an adaptive κ, potentially based on training dynamics or the semantic specificity of the target concept, could lead to further improvements. This fixed hyperparameter approach might limit optimal performance across diverse personalization tasks.

### Questions
How does DTI's performance scale with the complexity and number of concepts in a single prompt?

Could the directional optimization framework be extended to multi-token representations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper is related to the task of text-to-image personalisation generation. The authors use Textual Inversion as the base method and claim that one problem with this method is the high norm of embedding of learned concepts, which leads to semantic drift. To address this issue, the authors propose finding such embeddings on a unit hypersphere. Additionally, they propose using a von Mises–Fisher distribution as a prior to model the distribution of these embeddings.

### Strengths
1) The idea behind the presented method is clear, and the text is well written.
2) The paper provides a solid theoretical basis for the proposed modifications.
3) The authors demonstrate the additional capabilities of their method.
4) The paper contains many experiments and an extensive ablation study.

### Weaknesses
1) The work contains a lot of theoretical discussion, but most of it is based on asymptotical estimations (e.g. Lemmas 1 and 2, Proposition 1). The problems described do not seem obvious to me and I am not sure they could arise in practice. (questions 1, 2)
2) I am unsure about the correct evaluation of TI and CrossInit on SDXL (Table 2). It seems that the methods are slightly overfitted. (question 3)
3) Some parts of the work (e.g. Fig. 1, Tab. 7) lack details on how they were obtained. (questions 4, 5, 6)

### Questions
1) Could you demonstrate Effect 1 in practice and explain how your method deals with it? For example, you could train a classifier that takes the trained embedding after the Layer Norm of a particular concept as input and tries to predict the position of that embedding. You could train such a classifier for the base model, TI and DTI, and demonstrate the distribution of the resulting accuracies for different concepts.
2) Could you demonstrate Effect 2 in practice and explain how your method deals with it? For example, you could fix a concept and calculate the angle between the trained embedding and its initialization. You could show the distribution of different concepts for TI and your method. 
3) Could the authors present the metrics for the earlier TI and CrossInit checkpoints from Table 2?
4) Could you please explain how you obtained Fig. 1b? What hyperparameters did you use to train the methods, and what was the checkpoint step?
5) Could you clarify the results in Table 7? Why are the results for your method all in bold when, as I can see, they are not always the best?
6) Could you clarify which settings were used to generate the images in Figure 2?

### Soundness
2

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
4

### Summary
This paper proposes DTI, based on its key motivation that the norm of the special token inflates with training. They analyze the impact of this inflation by providing theoretical analysis on positional embedding and residual updates within the transformer. To mitigate this, they propose to learn only the direction with a vMF prior, while fixing the magnitude. The proposed method is claimed to yield improvements in text fidelity of TI and its variants.

### Strengths
1. It is useful that the norm inflation dilutes the impact of positional embedding and the residual updates within the transformer.
2. The claims are supported with theoretical justification.
3. The writing is clear and easy to follow.

### Weaknesses
1. The baselines are limited to heavily outdated methods. Further comparison is needed [1-3].
2. The norm inflation of the learned token had already been explored, not only in personalization [4], but also in VLM prompt tuning [5].
The quantitative comparison with [4] is provided, it is not clearly explain how the proposed approach differ conceptually.
3. Fig.3 - Is SLERP interpolation specifically available to the DTI? For fair comparison, SLERP should also be applied for TI for concept interpolation.
4. The choice of m* (average norm of the vocaubulary) is not justified. For example, m* can be the norm of the common adjectives e.g., yellow, fluffy, or the norm of the corresponding prior concept e.g., dog, cat.
5. Further comparison with recent optimizers e.g., Adam, AdamW should be provided to validate the effectiveness of the RSGD.


[1] Direct Consistency Optimization for Robust Customization of Text-to-Image Diffusion Models, ICLR'24
[2] Identity Decoupling for Multi-Subject Personalization of Text-to-Image Models, Neurips'24
[3] Controlling Text-to-Image Diffusion by Orthogonal Finetuning, Neurips'23
[4] Cross Initialization for Personalized Text-to-Image Generation, CVPR'24
[5] Nemesis: Normalizing the Soft-prompt Vectors of Vision-Language Models, ICLR'24

### Questions
1. The paper explains how the large norm disrupts the transformer's contextualization ability, but I'm curious why this happens. It would be more helpful to know why this phenomenon arises.
2. Should the norm always be excluded from learning? Could this be learned, but with regularization?

### Soundness
2

### Presentation
2

### Contribution
2
