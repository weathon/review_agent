# CASteer: Cross-Attention Steering for Controllable Concept Erasure

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Diffusion models have transformed image generation, yet controlling their outputs for diverse applications, including content moderation and creative customization, remains challenging. Existing approaches usually require task-specific training and struggle to generalise across both concrete (e.g., objects) and abstract (e.g.,4 styles) concepts. We propose CASteer (Cross-Attention Steering), a training-free framework for controllable image generation using steering vectors to influence a diffusion model’s hidden representations dynamically. CASteer precomputes concept-specific steering vectors by averaging neural activations from images generated for each target concept. During inference, it dynamically applies these vectors to modify outputs only when necessary, either removing undesired concepts from images where they appear or adding desired concepts to images where they are absent. This selective activation ensures precise, context-aware adjustments without altering unaffected regions. This approach enables precise control over a wide range of tasks, including removing harmful content, interpolating between desired attributes, replacing objects, all without model retraining. CASteer outperforms state-of-the-art techniques while preserving unrelated content and minimising unintended effects.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes CASteer, an activation steering method for concept erasure in diffusion models. It extracts steering vectors from cross-attention layers in the diffusion model for the concept to be erased and applies them during inference for erasure. Experiments are conducted for erasure of abstract concepts (nudity, violence) and objects (snoopy, mickey) on I2P benchmark and COCO dataset.

### Strengths
The proposed activation steering method for concept unlearning is training-free. 

Adapting the method to novel concepts is straightforward and does not require retraining. 

Experiments on I2P benchmark for SDv1.4 model shows the effectiveness of the proposed method over a comprehensive set of prior methods.

### Weaknesses
The proposed method is not new as it applies the idea of activation steering which is well-established in LLMs to concept unlearning in diffusion models. So, the novelty is limited. 

There is no theoretical justification on why operating in the latent space of diffusion models is better. It does not fully justify why unlearning in the cross-attention activation space (latent) is fundamentally better than guidance or prompt-based methods. 

Is it sensitive to the noise seed of the diffusion model? Are the steering vectors extracted for random noise seeds?

How does the method work if there is a mismatch between the number of steps used in steering vector extraction and during inference?

Missing ablation study on steering with only one vector per CA layer in SDv1.4.

Concept erasure based on guidance or steering is susceptible to adversarial attacks via concept addition or subtraction. Is this robust to adversarial attacks based on concept arithmetic [1] ?

Results are only reported for few concepts (nudity, violence, snoopy, mickey). Missing erasure results on art styles (e.g., van gogh) or more abstract concepts (e.g., summer, mosaic style etc).

Comparisons are only reported for SDv1.4 model. Missing comparisons to prior methods on more recent models such as SDXL, SANA or SD3.5

Paper can be better organized. Important ablations from supplementary need to be included in the main paper or the appendix instead of supplementary paper.

[1] Petsiuk et. al. Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models, ECCV 2024

Minor: 
Reference to tables in main paper (e.g., table 15) is missing.

### Questions
See weaknesses above.

How are the multiple prompts generated?

Do the steering vectors generated for SDv1.4 work for SD v1.5 across different checkpoints of the same model?

### Soundness
2

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
The paper proposed Cross-Attention Steering (CASteer), a training-free framework for concept erasure in diffusion models using steering vectors to influence hidden representations dynamically. Specifically, CASteer cleverly designs an algorithm for constructing steering vectors for new concepts and leverages these steering vectors to suppress unwanted image features without retraining. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
* The writing is fluent and logically coherent, exhibiting strong readability.
* The proposed method is highly efficient, requiring no training or fine-tuning while achieving excellent performance.
* The proposed method exhibits strong generalization and is applicable to various text-to-image models that incorporate cross-attention mechanisms.
* Comprehensive qualitative and quantitative experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
* As shown in Figure 1, CASteer computes a steering vector for the output of every cross-attention layer at every timestep in the generation process and applies a correction. Could this be somewhat excessive? Would it be possible to experimentally analyze whether the number of corrected timesteps and CA layers can be reduced to improve efficiency?
* The reviewer examined the examples provided in the main paper as well as the prompt set in the appendix, and observed that the subjects used when constructing the prompts are relatively simple. This may lead to high variability in the generated outputs, which in turn could make the steering vector estimation unstable. For example, in Figure 1, when using “dog” as the subject, the model generates a brown dog as the positive sample and a white dog as the negative sample; this could cause the steering vector to incorrectly treat other attributes of the dog as part of the target concept. Would using more specific subjects yield a more stable and robust steering vector?
* It would be very helpful to provide some positive visual examples of erased abstract concepts, such as “Van Gogh style.”

### Questions
See 'Weaknesses'.

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
The paper introduces CASteer, a training-free method for concept erasure in text-to-image diffusion models. Given an input prompt and a concept prompt, CASteer guides image generation to follow the input prompt while avoiding visual content associated with the concept prompt. The method computes steering vectors from the difference between cross-attention outputs when prompting the model with and without the concept. These vectors are applied at inference to steer generation in all the cross attention layers of the diffusion transformers. Experiments show that CASteer significantly outperforms prior baselines and is able to remove both abstract (e.g. nudity, inappropriate content) and concrete concepts (e.g. Mickey, Spongebob) and to generalize across models such as Stable Diffusion v1.4, SDXL, and SANA.

### Strengths
1. The works presents a novel *training-free* approach to solve the problem of content erasure which improves over existing baselines both in term of performances and of practicality (no need to run an ad-hoc training). 
2. I find the idea of using steering vectors to *remove* concepts rather than *adding* them very interesting.
3. The paper is well written and easy to follow.

### Weaknesses
I only have some minor concerns:

1. The *extension to multiple concepts* (L272–L276) is not experimentally validated. It would be useful to assess performance as the number of erased concepts increases and/or their individual steering vectors substantially differ from each other.
2. Equation 4 introduces *cosine-similarity weighting* between the text prompt and the erased concept; the paper does not analyze how crucial this weighting is in practice. An ablation could clarify whether simpler weighting performs comparably.
3. Some *figures* could be improved. Figures 2 and 4 could benefit from clearer labeling of rows/columns and spacing can be increased among the sub captions of Figure 3.

### Questions
1. How does CASteer perform when erasing multiple unrelated concepts or when steering vectors are different from each others?
2. How critical is cosine-similarity weighting? Could fixed or learned scalars achieve similar results?

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
4

### Summary
This paper introduces CASteer, a training-free framework for erasing specific concepts from diffusion models. The core idea is to compute steering vectors that represent the direction of an unwanted concept in the model's cross-attention layers. During inference, these vectors are subtracted from the model's activations without requiring any model retraining. The authors did comprehensive empirical experiments to demonstrate the effectiveness of the proposed method across various concepts, prompts, and model architectures.

### Strengths
- a training-free framework using precomputed steering vectors to remove concepts from diffusion models.
- comprehensive empirical evaluations across various concepts and models.

### Weaknesses
- limited novelty regarding method, with strong assumptions on steering vectors’ linear compositionality.
- ad-hoc parameter selection - steering vector scaling hyperparameters are fixed empirically.
- no discussion of computational or memory trade-offs when constructing per-layer, per-step steering vectors.
- somewhat limited improvements, for example, CASteer with clipping only surpassing second-based model Receler by 1.42%, which is not significant.
- did not compare with unlearning baselines, such as [1,2]. 

[1] Wu, Yongliang, et al. "Unlearning concepts in diffusion model via concept domain correction and concept preserving gradient.

[2] Alberti, Silas, et al. "Data unlearning in diffusion models." arXiv preprint arXiv:2503.01034

### Questions
- the hyperparameter $β=2$ is used for all experiments. How does the method's performance and stability vary with this parameter, especially across different concepts or model architectures?
- the paper uses a large number of prompt pairs (50 for concrete, 196 for abstract concepts). How sensitive is the method to the quality and diversity of these prompt pairs? poorly chosen set of prompts could lead to an imprecise steering vector.
- for multi-concept erasure, how do you resolve conflicting steering directions? is orthogonality assumptions still valid?
- the results on using steering vectors for adding concepts (and style transfer) are rather limited, can you discuss why certain applications fails with the proposed method?

### Soundness
3

### Presentation
2

### Contribution
2
