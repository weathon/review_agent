# Struct2Real: A Systematic Framework for Accurate and Efficient Structure-Grounded Object Image Generation

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 6, 4, 4

## Abstract
Recent advances in image generation have enabled the creation of high-quality visual content with impressive semantic fidelity. However, generating object images under fine-grained structural constraints, particularly preserving topology and spatial layout,  remains an open challenge. We propose Struct2Real, a novel framework for structure-grounded object image generation that combines explicit structural control with photorealistic generation, consisting of twofold. 1) we develop a novel structure modeling system that enables users to create a 3D structural representation named StructMap — an object structure abstraction composed of geometric primitives and their spatial layouts. 2) We design a modular image generation algorithm and combine this algorithm with multimodal large language models (MLLMs), harnessing their superior performance to generate realistic object images under structural constraints encoded in StructMap.
Extensive experiments demonstrate that Struct2Real achieves strong performance in structure-grounded object image generation while ensuring low user effort required for this task, highlighting the practicality and effectiveness of our method. Please refer to more details in our appendix and supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Struct2Real, a novel framework for generating photorealistic object images under stringent structural constraints, specifically preserving object topology and spatial layout. The framework consists of two main modules: 1) A structure modeling system centered around StructMap, an explicit 3D representation composed of geometric primitives that encode the object's topology and spatial layout. 2) A modular image generation algorithm leveraging Multimodal Large Language Models (MLLMs). Extensive qualitative and quantitative experiments demonstrate that Struct2Real surpasses text, lineart, and scribble-based controllable generation methods in both visual realism and structural fidelity, while requiring lower creation effort.

### Strengths
1. Struct2Real introduces a cognitively inspired, part-based 3D abstraction framework named StructMap, which models objects through geometric primitives and their spatial relationships. The design achieves a strong balance between expressiveness and usability, enabling users to specify complex structures intuitively. Moreover, the authors provide a visual design and interaction interface that allows users to construct and inspect StructMaps directly, greatly enhancing interpretability and accessibility.

2. The integration of the StructMap representation with the structure-consistency feedback loop offers a principled mechanism to enforce explicit topology and spatial layout. This feedback-driven process effectively guarantees structural fidelity even under complex geometric configurations, overcoming a key limitation faced by traditional sketch- or layout-based approaches.The modular condition-augmentation and consistency-discriminator design exploits MLLM reasoning to maintain geometric correctness.

3. The overall framework is modular, combining condition augmentation, image generation, and consistency discrimination into a unified process. By leveraging the reasoning capability of multimodal large language models, the system maintains high geometric consistency while synthesizing photorealistic textures, achieving both controllability and realism in structure-grounded image generation.

### Weaknesses
1. The method relies heavily on commercial MLLMs such as GPT-4o, which raises reproducibility concerns and may limit open evaluation.

2. The dataset covers only about 30 object categories with 3,000 samples, which restricts scalability and domain generalization. Broader testing on complex scenes or organic, non-rigid objects would strengthen the conclusions.

3. The StructMap representation is inherently constrained by its predefined primitive library. The current examples focus mainly on simple mechanical or geometric shapes, leaving its expressive capacity for more complex structures insufficiently demonstrated.

4. The proposed feedback loop requires multiple rounds of MLLM inference, which could introduce significant computational overhead. A quantitative analysis of runtime and system cost would clarify the method’s practical feasibility.

5. The quantitative evaluation focuses mainly on comparisons with large text-to-image models. It should include comparisons with other structure-aware or 3D-conditioned generation methods that address similar tasks.

### Questions
1. Could StructMap be automatically learned or inferred from existing 3D data or multi-view images, rather than being manually constructed?

2. How scalable is the StructMap creation process when modeling complex or deformable objects such as plants, animals, or articulated human figures?

3. What is the computational cost of the full feedback loop? Please quantify the average number of iterations required by the discriminator for convergence.

4. Could the authors quantify the contribution of each component in the modular pipeline through ablation or error propagation analysis?

5. How well does Struct2Real generalize across viewpoints or lighting conditions? Can a single StructMap be used to render consistent multiview outputs, potentially linking to 3D-aware diffusion pipelines?

### Soundness
4

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
4

### Summary
This paper proposes Struct2Real, a framework for structure-grounded object image generation. It introduces a new 3D structural representation called StructMap, composed of geometric primitives encoding object topology and spatial layout. The framework includes:
(1) a structure modeling system allowing users to assemble StructMaps via an interactive interface, and
(2) an image generation algorithm that combines StructMaps with multimodal large language models (MLLMs) to produce photorealistic images faithful to the provided structure.
Experiments compare Struct2Real with text, lineart, and scribble-based conditioning under various baselines (OmniGen, ControlNet++, T2I-Adapter, etc.), evaluated by FID and human MOS ratings.

### Strengths
1. This work presents a clear motivation for structure-grounded control in object image generation, supported by cognitive inspiration (Recognition-by-Components theory).


2. This work introduces StructMap, a clean, interpretable representation that enables explicit structural input.

3. Demonstrates broad evaluation—multiple baselines, diverse conditioning modalities, and human studies on both realism and structural alignment.

4. Visual examples are compelling and show genuine improvements in both realism and structure control.

### Weaknesses
1. The paper relies solely on MOS for assessing structure–image alignment (Sec. 4.1 → A.2.5). This weakens objectivity.


2. In Fig. 4, the iterative consistency-checking process is central, yet there is no evidence on iteration counts, failure cases, or convergence stability.

3. The claimed “3000 samples, 30 categories” dataset is newly built but lacks public availability or validation diversity. Examples of StructMap complexity (number of primitives, topology variety) are missing.

4. The proposed StructMap indeed provides a more accurate and explicit representation of object structures. However, since it relies on specific geometric priors and requires a dedicated software interface for creation, its applicability remains somewhat limited in terms of flexibility and accessibility, especially when compared to more lightweight and widely usable inputs such as text or scribble conditions.

### Questions
1. How is the “structure consistency discriminator” implemented—purely via LLM reasoning or also via visual feature comparison? How many regeneration rounds are typically required before convergence?

2. Could StructMap be automatically extracted from existing CAD or mesh data? If yes, how scalable is the user-creation interface beyond toy-level examples?

3. Have you evaluated Struct2Real on multi-object scenes or non-rigid categories? Does the geometric-primitive abstraction generalize beyond single rigid objects?

4. Is there a possibility to learn a StructMap-to-latent alignment (e.g., via adapter or encoder) instead of relying entirely on prompting?

5. Will you release the structure-prior dataset and interface tools for reproducibility?

### Soundness
4

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
3

### Summary
The paper introduces Struct2Real, a framework for structure-grounded object image generation that leverages explicit 3D structural priors (called StructMap) and integrates multimodal large language models (MLLMs) for photorealistic image synthesis under topology and spatial layout constraints. The work addresses a long-standing challenge in controllable generation — maintaining structural fidelity while achieving high realism. The proposed system includes (1) a structure modeling interface for creating StructMaps, (2) a condition augmentation and reasoning pipeline using MLLMs, and (3) a structure-consistency discriminator for iterative refinement. Experiments show consistent improvements over text, lineart, and scribble-based baselines in both realism (FID, MOS-R) and structure alignment (MOS-A).

### Strengths
1. a 3D composition of geometric primitives encoding topology and layout is conceptually elegant and practically useful. It provides a middle ground between coarse 2D conditions (e.g., lineart) and complex 3D CAD models.
2. qualitative evaluations shows good performance in both realism and structure preservation.

### Weaknesses
1. while StructMap is new, the image generation algorithm primarily relies on prompting existing MLLMs (e.g., GPT-4o). There is little discussion of model-specific innovations or learnable components beyond prompt design.

### Questions
1. Line 192-193 "these conditions are often coarse-grainedor ambiguous,making it difficult to accurately reflect the object’s structure" do you have an exmple? can you explain?
2. the paper talk a lot on 3D structure, it can generate novel view? 
3. can you suggest new way of controllability? in manner no one done before?
4. Could Struct2Real generalize to articulated or deformable objects? new concepts objects?

### Soundness
3

### Presentation
1

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
This paper addresses the problem of generating high-quality, photorealistic object images under fine-grained structural constraints—specifically preserving an object’s topology and spatial layout. Unlike existing controllable image generation methods that either offer only coarse structural guidance or require high professional skills, this work focuses on balancing precise structural control, visual realism, and user-friendliness, ensuring low effort for users while maintaining structural fidelity.
To tackle this challenge, the authors propose Struct2Real, a two-module framework integrated with multimodal large language models (MLLMs), featuring:
- a structure modeling system centered on "StructMap"—a 3D structural representation built from geometric primitives and their adjustable properties . StructMap explicitly encodes an object’s topology and spatial layout, and the accompanying interactive interface lets users assemble StructMaps without specialized skills;
- a modular image generation algorithm that works with MLLMs to translate StructMaps into photorealistic images, including three core components:
  1. a Condition Augmentation Module that converts StructMap images into detailed textual descriptions to bridge the gap between structural inputs and language-preferred generative models, highlighting key structural details like part count, connections, and spatial arrangement;
  2. an Image Generator that uses MLLMs to produce images conditioned on both StructMap images and their textual descriptions, preserving structural constraints while adding realistic textures, materials, and fine details. Users can also add optional style prompts to customize appearance;
  3. a Structure Consistency Discriminator that forms a feedback loop—MLLMs compare generated images with StructMaps to identify structural inconsistencies, provide reasoning for mismatches, and guide the generator to regenerate until the image aligns with the input structure;
- compatibility with multiple MLLMs to ensure generality across different multimodal model backends.
Experiments on a manually constructed "structure-prior dataset" show this method outperforms state-of-the-art baselines: it delivers better image realism and structural alignment than text, lineart, and scribble-based methods; StructMap balances accessibility and performance; and ablation studies confirm the framework’s generality across MLLMs and the necessity of each component for strong results.

### Strengths
- The paper is well written and easy to follow. The figures are clear, visually appealing, and effectively support the explanations.
- The experimental results are impressive and convincingly demonstrate the effectiveness of the proposed framework.
- The overall approach is conceptually sound and logically consistent—it makes good sense and aligns well with the problem formulation.

### Weaknesses
- Although the authors claim that their 3D structural representation is easy to obtain, this holds mainly for objects with relatively simple geometry. The method becomes less practical for complex shapes, which limits its applicability to more intricate or detailed structures.
- While the system design is reasonable and well engineered, it lacks strong conceptual novelty. The work feels more like a comprehensive engineering integration rather than a fundamentally new algorithmic contribution.
- The proposed pipeline, though effective, seems somewhat over-engineered for the task. The overall process could be viewed as unnecessarily complicated relative to the problem’s scale.
- Some of the line-art examples used in the experiments appear to be of relatively low quality. When high-quality line-art conditions are used, the performance gap between baseline methods and the proposed approach becomes much smaller, which raises questions about the fairness of the dataset and evaluation setup.

### Questions
1. The generated objects in your paper are mostly structurally simple. Could you provide more challenging examples that better demonstrate the capability and generalization of your method? While your structural control is indeed impressive, the simplicity of the examples limits the practical significance. Designing such simple shapes (e.g., cups) may not justify the relatively complex pipeline you propose—in some cases, manually sketching might even be more efficient.
2. In Figure 5, don’t you think some of your geometry conditions are excessively detailed, while certain line-art conditions appear overly coarse? Regardless of how difficult these inputs are to obtain, could you show what would happen if one directly traced the geometric components to create line-art conditions? This would clarify whether the gap stems from representation quality rather than the generation method itself.
3. Could you provide a quantitative estimate of the time required to construct geometric conditions? For instance, how long would it take to build the geometry condition for a simple object like a cup? And how would this time scale with more complex shapes—does the modeling effort grow significantly with geometric complexity?

### Soundness
3

### Presentation
3

### Contribution
3
