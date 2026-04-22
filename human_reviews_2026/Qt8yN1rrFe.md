# ISAC: Training-Free Instance-to-Semantic Attention Control for Improving Multi-Instance Generation

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6, 6

## Abstract
Text-to-image diffusion models excel at synthesizing single objects but frequently fail in multi-instance scenes, producing merged or missing objects. We show that this limitation arises because instance structures emerge before semantic features during denoising, making early semantic guidance unreliable. To address this, we propose \textbf{I}nstance-to-\textbf{S}emantic \textbf{A}ttention \textbf{C}ontrol (ISAC), a training-free and hierarchical inference objective that first enforces non-overlapping instance formation with self-attention and then aligns semantics through cross-attention. ISAC introduces a maximum pixel-wise overlap (MPO) criterion to strictly decouple instances and can be applied either as latent optimization or latent selection. Experiments on T2I-CompBench, HRS-Bench, and a new similar-object benchmark show that ISAC substantially improves both multi-class and multi-instance fidelity, achieving up to 52\% multi-class accuracy and 83\% multi-instance accuracy without external supervision. Our findings highlight the importance of aligning control with diffusion dynamics for faithful and scalable multi-object generation. The code will be made available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ISAC (Instance-to-Semantic Attention Control) to address the failure of text-to-image (T2I) diffusion models in generating multiple instances in complex scenes. By analyzing the temporal dynamics of diffusion models, the authors observe that semantic signals are underdeveloped before instance structures form. Based on this, they design a two-phase control objective: Phase 1 forms instance structures using structural signals, and Phase 2 binds semantics to these structures. The method is evaluated on multiple benchmarks (T2I-CompBench, HRS-Bench) and a custom similar-object scenario, demonstrating strong performance. Overall, the work is technically sound, innovative, and well-supported by experiments.

### Strengths
1. Methodological novelty: ISAC introduces a dynamic-aware two-phase objective that decouples instance structure formation and semantic binding, effectively addressing missing or merged instances in multi-instance generation.

2. Technical completeness: The method includes global foreground mask creation, K-means clustering for instance structure, Maximum Pixel-wise Overlap (MPO) for strict separation, and semantic binding, making it a comprehensive and practical approach. It can be applied as a loss function or verifier and is compatible with latent optimization and latent selection.

3. Extensive experiments: Evaluation on multiple benchmarks and scenarios, with metrics covering both instance separation and attribute binding, as well as qualitative analysis, supports the effectiveness of ISAC.

### Weaknesses
1. The description of the method in the paper needs to be clearer.

2. The examples presented in the paper are relatively limited; they are mostly cats and dogs, with only a few examples of other categories (SD1.5) shown in the appendix.

3. This training-free approach generally tends to degrade image quality, and the method lacks evaluation of image quality.

4. The method introduces significant time and memory overhead, increasing inference memory from 23GB to 75GB and substantially prolonging inference time.

5. Some formatting issues exist in the paper, such as the page number on page 6 being enclosed within a citation box.

### Questions
1. Could the authors provide more intuitive explanations or visual illustrations for some of the formulas in the method section?

2. Could a relatively complex example be included? For instance, if one wants to generate a person wearing many accessories and equipment, can such overlapping cases be generated correctly?

3. Most of the examples shown in the paper are cats and dogs. Could the authors include examples from other domains? Additionally, it would be helpful to show more results using stronger generators, such as SD3.

4. Does the proposed method affect the quality of the generated images? Could metrics such as FID or aesthetic scores be provided to evaluate potential quality degradation?

5. The method introduces substantial memory overhead. Compared to directly using existing L2I models or RL-finetuned models, is this overhead justified? Can the method achieve significantly better results?

6. There are some formatting issues in the paper, such as the page number on page 6 being enclosed within a citation box.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a training-free instance-to-semantic attention control for improving multi-instance generation. It identifies that instance features are formed first followed by the semantic features which are then bound to the instances. Based on this it proposes a new MPO loss for separating instances and class propagation loss for binding attributes to classes. Experiments are conducted on T2I-Compbench and HRS-Bench for multi-class and multi-instance setups.

### Strengths
The proposed method works across UNet and Diffusion Transformer models and can be combined with layout-guided methods.

It introduces a new Maximum pixel-wise Overlap criterion to enforce mutually exclusive boundaries between instances based on the observation that instance structures form early in the diffusion timesteps.

### Weaknesses
The evaluation of the proposed method is insufficient contrary to the title and claims made in the paper. It needs to be evaluated with other similar instance generation methods that can handle more complex tasks such as MIGC and InstanceDiffusion since the proposed method can be combined with those.


The multi-instance evaluation setup is weak as it only uses instances of the same class and does not evaluate multiple instances of two classes such as “A photo of two dogs and three cats”

In L210, it mentions that the instances should be mutually exclusive, how does this method work for small objects. All qualitative results are shown only for objects that cover a significant portion of the image.

Quantitative evaluation of the proposed method is only performed on simple object classes like animals and not on persons or indoor objects. Section C.3 mentions that these are omitted due to the difficult nature of the objects and so the effectiveness of the method is limited beyond a certain set of classes. The qualitative results in Appendix Figure 13 for different objects needs to be compared with prior works as they seem to be cherry picked.


The proposed latent optimization method is slow, sensitive to the noise seeds and more importantly does not work robustly for all cases, sharing the inherent limitations of prior inference-based latent optimization techniques.


The presentation of the paper is not clear and it can be improved further. The results for Algorithm 2 are not discussed in the paper and so it can be moved to Appendix or removed. Table 13 in the Appendix including the latency and memory costs should be moved to the main paper. 

Minor: 
Typo in L10 of Algorithm 1 and L1200

### Questions
See weaknesses above

### Soundness
2

### Presentation
1

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
This paper proposes the ISAC framework, which employs a two-stage objective function: the first stage leverages self-attention to achieve instance-structure separation, and the second stage utilizes cross-attention to accomplish semantic binding. The approach incorporates the MPO metric to optimize spatial exclusivity and semantic correspondence in multi-instance generation. Experiments conducted on multiple mainstream text-to-image models demonstrate that the framework significantly improves the quality of multi-instance generation.

### Strengths
1. Based on the observation that in the denoising process of diffusion models, spatial instance structures emerge before clear semantics materialize, the proposed stage-wise separation algorithm is more reasonable and better aligned with this generative process compared with prior work.
2. The effectiveness of the method is validated across multiple popular text-to-image models (e.g., SD1, SD2, SD3, SDXL, PixArt), showing consistent improvements in multi-instance generation quality.

### Weaknesses
1. Compared with the original approach, the proposed method may introduce additional computational overhead. The inclusion of latent optimization and VLM models requires larger VRAM and increases inference time.
2. The work lacks comparative experiments with LLM + Layout methods.

### Questions
1. Potential Reduction in Layout Diversity. Given the conclusion stated in Line 210 that “A total of N instances should occupy mutually exclusive, N distinct regions”, I am concerned that such a constraint may limit layout diversity. For example, if the input prompt describes “a cup and three ice cubes”, the proposed method may tend to generate images where the ice cubes appear outside the cup, rather than depicting them inside the cup. This could potentially reduce the natural compositional variety expected in certain prompts. Could the authors clarify whether the framework allows flexibility for compositional layouts where overlap or containment is required?
2. Overlapping Semantic Regions. For prompts such as “a cat standing behind a transparent glass”, a particular image region may need to visually represent both the cat and the glass simultaneously. In such cases, the assumption that each instance should occupy a distinct, non-overlapping region could hinder accurate rendering, especially when transparency is involved. How does the proposed method address these complex spatial relationships while preserving semantic accuracy?
3. Baseline Selection and Efficiency Comparison. Since the proposed framework leverages an LLM to identify each instance within a textual description, it seems natural to compare against a baseline where an LLM is used to parse scene layouts and is subsequently combined with layout-to-image generation methods. In addition, direct efficiency comparisons (e.g., VRAM usage, inference time) with these layout-based approaches would make the evaluation more comprehensive.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work presents Instance-to-Semantic Attention Control (ISAC), a framework designed to enhance multi-instance generation in text-to-image diffusion models. It addresses a key limitation of current models—their difficulty in handling scenes with multiple distinct instances. ISAC introduces a hierarchical, training-free approach that separates instance formation from semantic binding during the denoising process. By tackling the challenges of instance separation and semantic alignment, the framework significantly improves the accuracy and fidelity of generated images in complex, multi-object scenarios.

### Strengths
The proposed approach effectively addresses common failures in multi-object generation by decoupling instance formation from semantic binding, enabling more accurate and coherent image synthesis in complex scenes.

The proposed method is model-agnostic and can be complementary to fine-tuned models and also enhance existing layout-guided models.

The paper is well-structured, and the supplementary material contributes meaningfully to both understanding and reproducibility of the proposed approach.

### Weaknesses
While the paper introduces some interesting ideas, its contribution is somewhat limited by the fact that object count and layout consistency in multi-object image generation are already being actively explored in the literature. Given the growing number of existing approaches, the novelty of the proposed method appears incremental, and broader comparisons with related works could further clarify its distinct advantages.

The conclusion section is relatively underdeveloped; it lacks a thorough articulation of the significance and impact of the contribution. Additionally, the vision for future work is only addressed in the supplementary material, which limits its visibility and integration into the main narrative of the paper.

### Questions
Please, see the previous comments.

Several existing works explicitly address the challenges of object counting and coherent layout generation in multi-object image synthesis. Notable examples—some of which are already cited in the paper—include CountGen, COUNTLOOP, and IMAGHarmony. These contributions reflect an active research landscape in this area. Extending the discussion to consider such or similar approaches would help better situate the proposed method within the broader context and clarify its unique contributions.

CountGen. Make It Count: Text-to-Image Generation with an Accurate Number of Objects (CVPR2025)
COUNTLOOP. Iterative Agent Guided High Instance Image Generation
IMAGHarmony. Controllable Image Editing with Consistent Object Quantity and Layout

### Soundness
3

### Presentation
3

### Contribution
3
