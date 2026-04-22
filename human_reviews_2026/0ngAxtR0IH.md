# HMFusion: Hierarchical Multi-Modality Fusion for CAD Representation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Computer‑Aided Design (CAD) generation, which plays a vital role in product iteration and virtual simulation, has been of great interest in modern industry. Existing deep learning-based methods for CAD generation have achieved remarkable success. However, these studies either require lengthy domain-specific prompts or multiview sketches. Though effective, they exhibit limitations in ensuring consistency in geometric representation and rely on multiple inputs. To address these challenges, we propose HMFusion: Hierarchical Multi-Modality Fusion for CAD Representation, which incorporates a cross-modal geometric prior with hierarchical embeddings for consistent and faithful CAD generation. Specifically, our method introduces a prompt-enhancement module that transforms minimal user prompts into professional CAD-oriented descriptions containing structural and dimensional details. To improve the consistency of geometric representations, we tightly fuse textual and geometric information through a CAD-aware hierarchical alignment between visual and textual semantics in a hyperbolic space. Extensive experiments demonstrate that our proposed framework achieves effective geometric accuracy and semantic fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel multimodal framework for generating accurate CAD models from minimal inputs—a brief text prompt and a single sketch or image. It enhances user prompts into detailed CAD descriptions using a fine-tuned LLM. Also it proposes a hierarchical multi-modality fusion module to fuse textual and visual features into a hyperbolic space, in order to ensure geometric consistency and semantic fidelity.

### Strengths
1. Main idea presentation of this paper and pipeline figure are easy to understand.

2. The proposal of learning relationship between primitives and entire CAD models sounds interesting in Line 259-260.

### Weaknesses
Major:

1. This paper may not be well completed in terms of writing, as Figure 3—which illustrates the core pipeline—is never referenced in the main text. Moreover, the "Modal Enhancement" module shown in the top-right of Figure 3 is never elaborated upon in the main body or even in the appendix.

2. While the idea sounds interesting, the claim in lines 259–260 regarding learning the relationship between primitives and entire CAD models does not seem fully justified. This relationship is only learned in the “image + text” fusion part, and doesn't include command sequence. Primitives such as loop, profile, and solid can exist in the command sequence, according to HNC-CAD:

- Hierarchical Neural Coding for Controllable CAD Model Generation, Xu et al. (https://arxiv.org/abs/2307.00149)

This explicit primitive information from command sequence is not used.  The fusion module claims to learn relationship between primitives and entire CAD models, but only relies on “image + text”. The reviewer is concerned about the correctness of this claim. How to prove the correctness of this claim if so?

3. The motivation to utilize hyperbolic space technique for multi-modal fusion in this paper is not clearly elaborated in Abstract and Introduction.

4. In Table.1, the evaluation result of "Ours", with "image + expert text" as input, seems missing. The reviewer understands the paper trys to propose use minimal text prompt. The reviewer still thinks it's essential to present the result of  "image + expert text".

5. In Table.1, the multi-modal input of DeepCAD and "Ours" do not align, which may introduce improper comparison.

6. In Table 2, the “Modality Enhancement” ablation appears inconsistent with the rest of the paper: the “Modal Enhancement” module shown in the top-right of Figure 3 is never described in the main text or the appendix, and this component is not explicitly listed among the paper’s contributions.

Minor:

1. Math definition in L236 has problem: $g_M(M_k) = \exp_{\mathbf{0}}^{\kappa}(f_M(M_k)$ 

Should complete the bracket here:  $g_M(M_k) = \exp_{\mathbf{0}}^{\kappa}(f_M(M_k))$

2. The citation formatting seems a bit off. For example, in line 113:

> such as point cloudsMa et al. (2024); Wu et al. (2021); ...

There should be a space before the citation, i.e., “point clouds (Ma et al., 2024; Wu et al., 2021)”. The same issue appears throughout the paper.

3. Some recent works on CAD generation are missing from the related work section. This area is evolving rapidly, particularly with the rise of LLM-based methods, so including a more comprehensive coverage would strengthen the paper’s survey:

- CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation With MLLM, Xu et al. (https://arxiv.org/abs/2411.04954)

- CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation, Li et al. (https://arxiv.org/abs/2505.04481)

- CAD-Coder: Text-to-CAD Generation with Chain-of-Thought and Geometric Reward, Guan et al. (https://arxiv.org/abs/2505.19713)

- CADmium: Fine-Tuning Code Language Models for Text-Driven Sequential CAD Design, Govindarajan et al. (https://arxiv.org/abs/2507.09792)

### Questions
1. What is the real effectiveness of the hyperbolic space technique in multi-modal fusion? This is the reviewer's core concern about this paper's contribution. It is recommended to include a comparison against standard multi-modal fusion methods to showcase the essentiality of this module and enhance the paper's technical soundness. And also to highlight the motivation of utilizing hyperbolic space for multi-modal fusion.

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
4

### Summary
This paper proposes HMFusion, a framework for generating CAD models from minimal inputs: a brief text prompt and a single sketch. It addresses the limitations of existing methods, which either require expert-level prompts or lack dimensional accuracy. Specifically, a LoRA-tuned LLM enhances brief prompts into CAD-specific descriptions, while a hyperbolic space fusion module aligns linguistic and geometric semantics through contrastive and entailment learning. The fused representation guides a Transformer-based CAD decoder to produce accurate and consistent 3D models. Experiments show notable improvements in geometric accuracy, command precision, and structural fidelity over prior text or image approaches.

### Strengths
1. The model’s ability to generate manufacturable CAD models from concise prompts and simple sketches has strong industrial  interaction implications.

2. The paper is exceptionally well-written and easy to follow. Figures 2 and 3, in particular, provide a clear and intuitive overview of the complex architecture and data flow.

3. The paper tackles a well-defined and highly practical problem in generative design. The goal of enabling high-fidelity CAD generation from minimal, non-expert inputs (a simple sketch and a brief sentence)  is a significant step forward in usability compared to existing methods.

### Weaknesses
1. The method relies on the DeepCAD representation, which is limited to sketch and extrusion commands. It only represents the simple CAD model that is far from real-world CAD design . The paper does not discuss how the proposed hierarchical fusion would scale to more complex CAD operations (e.g.,sweeps, fillets, chamfers) 

2.It is currently unclear how the model handles non distributed shapes or non mechanical CAD categories (OOD problem, such as free-form designs).

3.Considering the current model's design, which uses an SFT-LLM to expand the basic text, this raises the question of whether the user-prompted basic text must adhere to a specific format or paradigm. Furthermore, if the format of the basic text differs from what the SFT-LLM was trained on, how would the model's performance be affected? An in-depth discussion on this point would help to strengthen the paper's contribution.

4. Some descriptive analyses do not align with the tables. For instance, in the Ablation Study of Modality Enhancement section, it mentions, “Expert-SFT improves the parameter accuracy on average by more than 3%... ”but in Table 2, Ablation Study 2, the results for ”w/ SFT LLM“ do not seem to be listed.

### Questions
1. Have authors considered the performance of feeding expert text directly to the model in such an experiment? This comparison can further help demonstrate the significance of basic expansion in SFT-LLM.

2. How does the model perform when given conflicting information between the sketch and the text? Which one would be considered first?

3. What is the minimum length of basic text, and has the model been tested to produce the shortest description required for valid generation?

4. What would the effect be if the text expanded by the SFT-LLM were fed into Text2CAD?

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
3

### Summary
This paper presents HMFusion, a multimodal CAD generation framework that combines text and image inputs. The method integrates three components: (1) a LoRA-tuned LLM for prompt enhancement that transforms short natural descriptions into detailed CAD-specific language; (2) a visual encoder that extracts geometric priors; and (3) a hierarchical fusion module operating in hyperbolic space to align visual and textual embeddings. The approach aims to improve both semantic fidelity and geometric consistency of CAD generation, addressing limitations of unimodal (text-only or image-only) systems.

### Strengths
The paper tackles the cross-modal alignment between text and image inputs, a relevant and underexplored problem in CAD generation. The motivation is clearly articulated: text-to-CAD systems often lack spatial grounding, while image-to-CAD lacks parametric precision. The proposed multimodal fusion framework provides a compelling attempt to balance these complementary weaknesses.

The work demonstrates substantial engineering effort and a non-trivial combination of components—text expansion, hyperbolic-space embedding, and CAD-aware contrastive and entailment losses. The experiments are thorough, including ablations across modalities, enhancement stages, and loss components. The visual results consistently support the claim of improved geometric completeness and semantic consistency. The approach seems technically sound and well-implemented, with quantitative gains over baselines in Chamfer distance and parameter accuracy.

### Weaknesses
The framework indeed involves several heavy modules (CAD autoencoder, LoRA-tuned LLM, GLIP-based contour extractor, hierarchical contrastive learning). This complexity may hinder reproducibility and scalability, especially since each submodule requires specialized training.

The experimental comparisons are acceptable. However, the authors claim that their baselines represent state-of-the-art methods. In my view, this is inaccurate, as DeepCAD and Text2CAD are no longer considered SOTA. In the text-to-CAD domain, more advanced frameworks such as CAD-Llama [1] and CADFusion [2] have demonstrated superior performance. While this may not be an intentional oversight, the phrasing is misleading as it disregards recent progress in the field.

Minor problems: 
1. Use `\citep` if you want to create citations in the parentheses.
2. Annotation misalignment: \mathcal{L}{pair} and L{pair} are interchanged and can be confusing.

[1] CAD-Llama: Leveraging Large Language Models for Computer-Aided Design Parametric 3D Model Generation. CVPR 2025.

[2] Text-to-CAD Generation Through Infusing Visual Feedback in Large Language Models. ICML 2025.

### Questions
How are the text embeddings computed after segmentation? Are SpaCy-extracted phrases encoded via the same text encoder used in fusion, or is there a separate mapping network?

Have the authors evaluated against recent multimodal baselines (e.g., CADFusion, CAD-LLaMA)? If not, could they discuss performance expectations relative to those?

Since it is a multi-modal task, recent VLMs are also competitive baselines. Can the authors test the performance of some of these models (e.g. GPT-4o or later models) on it?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers a task of text+sketch 2 - CAD generation. This is an active research area, fusion of 2 tasks - text2CAD and image2CAD. The new multi-modal method is proposed for this task. The model evaluation is performed on DeepCAD dataset.

### Strengths
1) The proposed methods improves in accuracy over several competitors like Text2CAD and DeepCAD

### Weaknesses
1) Limited evaluation
- The proposed method evaluation only on DeepCAD dataset. 
- IoU metric is not used for evaluation
- The number of competitors is limited, e.g. Cadrille (https://arxiv.org/abs/2505.22914) shows better results even for single modality (image or text) 

Overal this limited evaluation raises question regarding impact and significance of the proposed method. If this approach is valid and promising for futher research?

2) How robust is the proposed method for view angle changes? What will happen if sketch is generated from a different angle? Has the model been tested for it?

### Questions
Please, address the weaknesses #1&2

### Soundness
3

### Presentation
3

### Contribution
1
