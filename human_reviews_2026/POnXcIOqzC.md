# One-for-All: Towrads Human-Centric Multi-Subject Customization from Single-Subject Examples

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Human-centric multi-subject customization remains a key challenge in the field of subject-driven image synthesis. A primary obstacle lies in the curation of paired multi-subject data, which is labor-intensive and often introduces subject inconsistencies that hinder effective model learning. In this paper, we introduce One-for-All, a framework that pioneers a new paradigm by learning multi-subject consistency from only real-world, single-subject examples, breaking the dependency on curated multi-subject data. Building upon this, we unlock the full potential of this paradigm shift by introducing two key designs that ensure robust multi-subject consistency. Firstly, a Center-Aligned Cross-Modal Position Association module is proposed to guide the interaction between visual references and their textual descriptions. This interaction facilitates intra-subject semantic grounding among cross-modal conditions and improves their synergistic contributions for subject consistency. Secondly, to alleviate the attention dilution caused by the increased tokens of multiple subjects, a Dynamic Attention Modulation mechanism is introduced. This design maintains multi-subject consistency by dynamically predicting and applying token-wise attention weights, ensuring focus remains on critical features. Comprehensive experiments demonstrate that our method, even trained exclusively on single-subject data, exhibits robust generalization across varying numbers of reference subjects, and surpasses all baseline methods trained on curated multi-subject data pairs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the challenge of acquiring high-quality multi-agent data for human-centric multi-agent customization, this paper proposes a One-for-All framework that establishes multi-agent consistency by learning solely from real-world single-agent examples.
The proposed framework introduces a Center-Aligned Cross-Modal Position Association (CCPA) module to facilitate interaction between visual references and textual descriptions, and incorporates a Dynamic Attention Modulation (DAM) mechanism to adaptively adjust attention weights, effectively mitigating issues of semantic grounding ambiguity and attention dilution.
Extensive experiments, including evaluations on HumanBench, demonstrate that One-for-All, even when trained exclusively on single-agent data, surpasses all baseline methods in multi-agent consistency, natural human pose generation, and visual fidelity, while exhibiting strong generalization capabilities.

### Strengths
1. This paper is well-written and presented. 

2. The paper proposes One-for-All, a new framework that achieves multi-subject consistency using only real-world single-subject data. Through category-specific position conditioning, the model assigns unique conditional IDs and enforces position-aware learning, enabling robust multi-subject generation without relying on costly multi-subject datasets.

3. The Center-Aligned Cross-Modal Position Association (CCPA) module resolves semantic grounding ambiguity by explicitly linking textual descriptions to corresponding visual concepts through 3D RoPE-based position encoding. This centroid-aligned design ensures accurate text–image correspondence and enhances overall visual consistency.

4. The Dynamic Attention Modulation (DAM) mechanism alleviates feature competition among multiple subjects by dynamically adjusting token-wise attention weights through a learnable temperature generator. This adaptive control sharpens attention on relevant features and suppresses interference, improving multi-subject coherence and fidelity.

### Weaknesses
1. The proposed CCPA module lacks clear novelty, as similar position re-encoding strategies have been explored in prior works such as OmniControl, OmniControl-2, and EasyControl. The idea of using positional embeddings to handle multi-object reference conflicts is already well-established, making this component less distinctive in contribution.

2. The DAM mechanism appears relatively generic, relying solely on adaptive attention modulation without introducing explicit multi-object constraints or losses. It remains unclear whether such adaptive weighting alone can effectively mitigate feature competition among multiple subjects, and further analysis or ablation would be needed to validate its actual impact.

### Questions
1. How does the proposed CCPA module differ from prior works such as OmniControl, OmniControl-2, and EasyControl, which also employ position re-encoding to resolve multi-object reference conflicts? It would be helpful to clarify what specific innovation or design improvement CCPA introduces beyond these established approaches.

2. The DAM mechanism relies on adaptive attention scaling to alleviate feature competition, but it is unclear whether such self-learned modulation is sufficient to handle multi-object conflicts, especially when multiple subjects occupy overlapping spatial regions. How does the model ensure effective disentanglement or balance of attention in such cases without explicit multi-object constraints or loss terms?

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
This paper introduces Onefor-All, a framework that pioneers a new paradigm by learning multi-subject consistency from only real-world, single-subject examples, breaking the dependency on curated multi-subject data. Building upon this, it unlocks the full potential of this paradigm shift by introducing two key designs that ensure robust multisubject consistency.

### Strengths
1. This paper presents a novel framework that learns robust multi-subject consistency by leveraging only real-world, single-subject data.
2. Two novel mechanisms are proposed to adapt to paradigm shift and ensure multisubject consistency.
3. Extensive comparisons with state-of-the-art methods on our HumanBench demonstrate the superiority of the proposed method.

### Weaknesses
1. Can you show some attention maps or other visualizations to prove that it's correctly 'routing' the features for each subject to the right place in the image?
2. What happens when you try to combine even more subjects, like 6 or 8 at once? Does the quality start to drop?
3. How does your model learn realistic interactions, like how a shirt wrinkles when a person wears it, or how a bag strap presses into a jacket?
4. How generalizable is the method? For example, if we change the scene to cartoon characters, will it still work?
5. In your comparison images, it would be super helpful if you could add arrows or circles to point out exactly where your method is better.
6. Have you compared your results to the closed-source models?

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

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
This paper proposes One-for-All, a novel framework designed to address the challenge of human-centric multi-subject customization in image synthesis without relying on curated multi-subject data. It solves the problem of subject inconsistency by learning multi-subject consistency from single-subject real-world examples. Firstly, the paper introduces a Center-aligned Cross-Modal Position Association (CCPA) module, which aligns visual references with their textual descriptions by using center-based positional encoding. This design enhances semantic grounding across modalities and strengthens intra-subject consistency. Secondly, the authors propose a Dynamic Attention Modulation (DAM) mechanism that predicts token-wise attention weights to counteract attention dilution from multiple subjects. This ensures that critical features receive sufficient focus and preserves multi-subject consistency. Finally, extensive experiments on the proposed HumanBench demonstrate that the effectiveness of the proposed methods.

### Strengths
1. The task is interesting.
2. The motivation somewhat makes sense.
3. The proposed method is simple yet effective.
4. The framework learns multi-subject consistency from single-subject data, which is much easier to obtain.

### Weaknesses
1. Limited Input Scalability. The proposed framework supports only four input types (face, upper garment, lower garment, and bag), with the restriction that each type can appear only once and must be encoded using its corresponding positional embedding. This design limits the method's scalability, making it unsuitable for scenarios involving more diverse subjects, such as multiple faces, multiple tops, several bags, or additional accessories.
2. The paper repeatedly refers to a small set of internal data, yet the scale and details of this dataset are not disclosed. This lack of transparency makes it difficult for other researchers to reproduce the results or conduct fair comparisons.

### Questions
**Major Comments**
1. Is the proposed CAM module trained only on the single-subject dataset? From Fig. 5, it seems that the input is limited to Q. How does the module infer the required weights solely from the target image? Furthermore, how does it generalize to a larger number of reference items? Are the re-weighting scores computed independently for each item?
2. Please provide more details on the internal dataset used in the paper. Clarifying its specific characteristics would benefit the community as a whole.
3. To strengthen the argument, it would be more convincing to include objective results that validate the effect of the position-aware conditioning strategy. I may have overlooked it, but I only found visual comparisons in Figure 9.

**Minor Comments**
1. The visual comparison figures  (e.g.,  Fig. 6, Fig. 9, Fig. 10) do not display the corresponding input text of the generated images.
2. In several visual comparisons (e.g., Fig. 9, Fig. 10), the differences between methods are not very clear. I suggest highlighting the key differences directly in the figures for better readability.
3. The proposed human-centric multi-subject customization task is very closely related to the multi-garment virtual try-on task. Therefore, it would be beneficial to discuss more related works in this area.

- [1] Zhu L, Li Y, Liu N, et al. M&m vto: Multi-garment virtual try-on and editing[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 1346-1356.
- [2] He Z, Ning Y, Qin Y, et al. VTON 360: High-fidelity virtual try-on from any viewing direction[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 26388-26398.
- [3] Li Y, Zhou H, Shang W, et al. Anyfit: Controllable virtual try-on for any combination of attire across any scenario[J]. Advances in Neural Information Processing Systems, 2024, 37: 83164-83196.
- [4] Velioglu R, Bevandic P, Chan R, et al. MGT: Extending Virtual Try-Off to Multi-Garment Scenarios[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025: 6039-6048.
- [5] He Z, Chen P, Wang G, et al. Wildvidfit: Video virtual try-on in the wild via image-based controlled diffusion models[C]//European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024: 123-139.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents One-for-All, which is a framework for human image generation that combines multiple subjects like face, clothes, and bags using only single-subject training data. It introduces a position-aware learning strategy that gives each subject a fixed spatial location to keep them consistent. Two key designs help the model work well: a cross-modal position module to align text and image meaning, and a dynamic attention module to balance focus among subjects. The experiment results show that One-for-All produces more consistent and realistic results than the existing methods.

### Strengths
1. The proposed position-aware learning and attention design help keep all subjects well aligned and consistent in appearance.
2. The paper builds a new benchmark and shows strong improvements over several state-of-the-art methods in both quality and consistency.
3. The visualization of the attention after adding DAM demonstrates the reason for improvement which is intuitive.

### Weaknesses
1. The paper relies only on automatic metrics to measure image quality and consistency. Without human evaluation, it is unclear whether the results align with real user perception.
2. he study of the position-aware conditioning strategy shows only qualitative results, with limited images and no quantitative scores, which makes the contribution less convincing.
3. The experiments focus mainly on faces and clothes, without testing on other human-related or non-human subjects, so the generalization of the method remains uncertain.

### Questions
Since the paper focuses on human-centric generation, perceptual quality is important. Have the authors considered conducting a user study or pairwise preference test to verify whether their results are visually preferred by human observers?

### Soundness
3

### Presentation
3

### Contribution
2
