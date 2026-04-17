# Learning to Remove, Not Repeat: Robust Object Removal in Cluttered Scenes using Diffusion Models

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
Object removal, a key image inpainting task, aims to erase specified objects and plausibly fill the resulting region. Although recent diffusion models excel at generating realistic content, when employed for the removal task, they often fail in cluttered scenes by replicating nearby objects or hallucinating semantically similar ones, an artifact of their powerful, yet context-agnostic, generative priors. To address this, we introduce a robust framework that Learns to Remove, Not Repeat (LRNR). Our approach has three key components. First, we propose the Scatter-Tile Object Removal (STORe) dataset, a large-scale synthetic dataset with unique scatter and tile configurations designed to make models robust to object replication. Second, we employ an efficient fine-tuning strategy that combines Low-Rank Adaptation (LoRA) with a learnable task prompt, which internalizes the concept of removal, thereby eliminating the need for manual text guidance. Third, we introduce Mask-Aware Scheduled Guidance (MASG), a training-free inference technique that spatially and temporally modulates classifier-free guidance to enhance inpainting quality and preserve background integrity. Our evaluations demonstrate that LRNR outperforms state-of-the-art approaches, particularly in terms of removal success rate in challenging scenes prone to object replication, leading to more reliable and semantically correct results. Our dataset, source code, and trained models will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a challenging case of object removal where multiple similar objects existed in the image. To tackle this issue, the authors construct a synthetic dataset by simulating images containing several similar objects and then removing one of them. Their model is based on PowerPaint with LoRA fine-tuning, and they further propose a dynamic CFG scale for the masked region during denoising. The method achieves significantly better object removal performance than previous approaches.

### Strengths
The approach to construct inpainting pairs containing multiple similar objects is inspiring.

### Weaknesses
Overall, the contributions of this paper are not very convincing:

a) The proposed dataset appears more like a specialized data augmentation technique rather than a new dataset. It is designed specifically to handle inpainting cases where multiple similar objects exist in an image. However, the samples in Figure 3 look unrealistic and may even harm large-scale inpainting training. The scatter method simply pastes objects without considering lighting, shadows, or physical constraints, and the tile method also produces unrealistic images.
In essence, the model trained on this dataset might simply learn to rely on adjacent areas to fill the masked region, rather than understanding global image context. While this is a straightforward way to avoid hallucinating similar objects, it raises concerns about whether training on such unrealistic data might degrade generation quality.
This can be verified by training only on the proposed STORe dataset and comparing results. Such an experiment would also help clarify whether the improvement shown in Table 2 comes from the data construction method or simply from having more training samples. Please clarify this point.

b) The proposed method seems incremental, as it essentially builds on PowerPaint with LoRA fine-tuning.

c) Although the method successfully removes objects, the inpainted regions in Figure 1 appear to have unrealistic colors compared with the surrounding areas. Specifically, compared with RoRem-Mixed, the colors look less natural in the first, third, and last rows. Is this caused by the proposed MASG module?

### Questions
1. Table 2 requires further analysis:

    a) It is confusing that using a single learnable token significantly improves the success rate, while using more tokens fails to remove the object effectively. This seems inconsistent with the motivation of "capture the semantics of object removal across diverse visual scenarios". The learning of these tokens seems problematic, as there is no consistent setting that performs best on both success rate and FID. This cannot be explained by a trade-off between object removal capability and generation quality, since using a single learnable token combined with fixed text tokens yields the best removal results, while using all learnable tokens gives the best generation quality. Please analyze why one learnable token performs better than multiple tokens in terms of success rate.

    b) What does "effectively decoupling the removal and generation tasks" mean in this context?

    c) The bolded CLIP score in Table 2 appears to be mislabeled.

2. In Tables 2 and 3, the MASG module consistently lowers the CLIP score while improving other metrics. What causes this contradiction?

3. It seems odd that the paper cites recent works from the 2020s when introducing image inpainting and object removal. Please respect the foundational works in this field and include proper historical references.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new synthetic dataset named STORe (Scatter-Tile Object Removal). Generated via scatter and tile methods to include numerous repeated objects, STORe is designed to address the issue where models tend to generate similar objects when removing duplicate ones from images. To prevent the degradation of generation quality caused by training on small-scale datasets, the authors introduce MASG (Mask-Aware Scheduled Guidance), a Classifier-Free Guidance approach guided by masks and adaptive to diffusion timesteps. Specifically, it increases conditional guidance at high timesteps to enhance object removal, while reducing conditional guidance at low timesteps to preserve the model’s natural generation capability.

### Strengths
1. The proposed STORe dataset effectively targets the problem of models generating similar objects during duplicate object removal, filling a gap in existing training data for this specific challenge.
2. MASG well mitigates the quality degradation issue that arises when training large models on small-scale datasets, striking a balance between object removal accuracy and generation fidelity.
3. Comprehensive qualitative comparisons are provided, which sufficiently demonstrate the effectiveness of the proposed method in practical scenarios.

### Weaknesses
1. The object placement of the scatter method in STORe is inconsistent with real-world scenarios. As shown in the dataset samples, HQ-SAM objects are placed in a dispersed and regular way, which deviates from the common dense and random arrangements of objects in practical environments. This idealized placement may limit the model’s adaptability to unstructured cluttered scenes.
2. The dataset construction has limitations in physical plausibility. The scatter method may result in objects being unnaturally placed (e.g., floating in the sky), while the tile method suffers from unrealistic background repetition. Training on such dataset may potentially hinder the model’s ability to generalize to real-world, physically consistent scenes.
3. The advantages in quantitative metrics are not prominent. Only the human-annotated "removal success rate" achieves state-of-the-art performance, and even this metric shows no significant gap compared to RoRem, failing to fully validate the method’s superiority in objective evaluation.

### Questions
See weaknesses.

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
This paper tackles object removal in image editing, especially in cluttered scenes where object repetition occurs. The authors present LRNR, a diffusion-based framework featuring a dedicated anti-repetition dataset (STORe), parameter-efficient adaptation, and Mask-Aware Scheduled Guidance. LRNR achieves superior object removal and image quality compared to prior methods in complex scenarios.

### Strengths
- The problem selection is both relevant and insightful: repetition and regular duplication of objects is ubiquitous in real-world imagery, and context-driven repetition remains a key challenge in image editing.

- Both dataset and model design are original. The STORe dataset’s construction, especially the integration of Scatter and Tile strategies, is innovative and effectively balances data diversity with training complexity. The MASG approach skillfully leverages the frequency-specific generative characteristics of diffusion models at different timesteps, achieving a balance between structural consistency and natural textures.

- The experimental study is thorough, with comprehensive ablations validating the individual contributions of each module. Both quantitative and qualitative comparisons are extensive, showing LRNR’s clear superiority in success rate and generalizability across diverse visual domains (e.g., cartoons, animation, LEGO).

### Weaknesses
- More comprehensive comparisons with GAN-based or other generative paradigm methods are recommended. While the work focuses on diffusion models, it would benefit from including benchmarks against recent, high-performing GAN or Transformer-based inpainting approaches.

- The synthetic nature of the dataset may introduce bias. Although STORe’s construction is creative, it relies on recomposing content from HQ-SAM, RORD, and similar sources. This could introduce compositing artifacts or edge inconsistencies that limit the model’s generalization to real-world scenarios.

- The analysis of failure cases is insufficient. Although the paper showcases many successful results, it lacks a systematic investigation of the situations where LRNR still struggles (e.g., extreme occlusion, complex lighting, or non-rigid objects), which limits understanding of its practical applicability.

### Questions
1. Could the model’s sensitivity to mask precision, size, and shape be systematically tested?

2. Can the “success rate” metric be further refined by introducing more granular evaluation, such as edge consistency, lighting consistency, or structural plausibility? Relying solely on subjective “successful removal” may introduce bias.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new dataset and sampling strategy to remove objects in cluttered scenes were many other, similar objects, might be present. In this case, existing removal methods usually don't remove the given object but instead inpaint a similar object given the surrounding clues. By creating a specific dataset simulating this scenario and finetuning a model on this dataset the paper shows that the resulting method, coupled with an updated sampling strategy, can successfully remove objects even from cluttered scenes.

### Strengths
The paper creates a new dataset specifically for the use case of cluttered object removal. By simulating this challenging task the resulting model shows clear improvements both qualitatively and quantitatively. The user study also shows that training on this dataset, coupled with the updated sampling strategy, leads to clear improvements.

The ablation study shows how the individual parts of the pipeline, from dataset, trainable parameters, and sampling strategy, all meaningfully contribute to the improved performance.

### Weaknesses
There is limited novelty in the overall approach.
The paper basically shows that creating a dataset for a specific use-case and then finetuning a model on it leads to improved results. The dataset creation itself also seems to be relatively straight forward, mostly consisting of overlaying objects on background images.
The updated sampling strategy is mostly a hyperparameter tuning approach of cfg which also does not seem to be novel. 
The results themselves are good but only on a very specific subset of object removal tasks in cluttered environments with repeated objects. In other, more general settings, the approach seems to be slightly worse than other approaches.

### Questions
It would be interesting to also see some results of the approach on non-cluttered "default" scenes to compare the quality on normal object removal cases compared to some of the baselines.

### Soundness
3

### Presentation
3

### Contribution
2
