# Domain-Specific Text-to-Image Generation: Planning, Merging, and Replacing with Training-free LLMs

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Diffusion-based techniques, such as Stable Diffusion, exhibit remarkable capabilities in text-to-image synthesis and editing. However, general text-to-image diffusion methods frequently fail to accurately generate domain-specific components, such as particular electrical elements in schematic circuit diagram. Lacking domain-specific knowledge, rules, and sufficient data,  existing methods may struggle with resource-consumption model training. To address these limitations, we propose a novel, training-free framework for mastering domain-specific text-to-image generation, namely Planning, Merging, and Replacing (PMR).  Specifically, PMR precisely generates domain-specific elements and their configurations, enabling schematic circuit diagram generation without requiring model fine-tuning. 
Based on the establishment of a knowledge base, PMR employs large language models (LLMs) to plan inter-component connectivity according to the requirements provided by users. 
PMR further utilizes LLMs to spatially arrange symbolic blocks (representing components) and their connecting wires. Subsequently, PMR has a fine-grained positional control and generates symbolic blocks and wires at designated locations.  Extensive experiments demonstrate that PMR outperforms existing methods in domain-specific generation.
Our work opens a potentially new avenue of automated domain-specific text-to-image generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper tackles the failure of text-to-image models in generating circuit diagrams. It proposes PMR, a novel training-free framework. PMR uses LLMs to plan component connectivity and layout, then guides a diffusion model for fine-grained, spatially-controlled generation, avoiding the need for resource-intensive fine-tuning.

### Strengths
Targeting electrical circuit generation, the authors' proposed method of Planning, Merging, and Replacing advances beyond several existing T2I baselines.

### Weaknesses
1.  The title of the paper claims the contribution is "Domain-Specific", yet the methodology and experiments focus exclusively on a single domain (e.g., electrical circuit generation). This is confusing and potentially misleading. If the method's generality has not been substantiated, the title and claims should be narrowed to specifically reflect its application to electrical circuit generation.

2.  Section 3.4 is confusing to me; can you explain the complete denoising process? Please also clarify the role of the softmax operation and the meaning of the function \(\phi\) in Equations 9 and 10. Furthermore, have you compared your method with layout-to-image diffusion models (e.g., LayoutDiffusion [1], LayoutDM [2])? Such a comparison would help strengthen the persuasiveness of your work.

3.  It seems the experimental dataset used for evaluation is not clearly specified. Please clarify whether a publicly available benchmark or a private dataset was used.

**References**

[1] Zheng, G., et al.: LayoutDiffusion: Controllable diffusion model for layout-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)

[2] Inoue, N., et al.: LayoutDM: Discrete diffusion model for controllable layout generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2023)

### Questions
Please see Weaknesses section

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the problem of text-based circuit diagram generation.

The main contribution is to propose a training-free solution to the problem, which utilizes domain-specific knowledge derived from historical circuit diagram examples and the reasoning abilities of pretrained large language models (LLM) to guide the image generation of pretrained diffusion models.

### Strengths
1. Domain-specific text-to-image generation is an important problem to study.

2. The proposed merge regional diffusion is shown to be effective.

### Weaknesses
1. The paper is claimed to focus on domain-specific text-to-image generation, as indicated by many places in the paper, such as the title, the last sentence of the abstract, the second last sentence of the second paragraph in the introduction. However, the components of the proposed method are highly specialized for circuit diagram design. It is not clear how the method can be adapted to solve text-to-image generation in other domains. To provide strong evidence for the paper’s claim, it would be necessary to provide several examples of how the proposed method can be applied to other domains. 

2. The effectiveness of the proposed method is not adequately validated. In particular, the evaluation of the full generation method (introduced in Section 3) is missing in the experiments, and only a single component (i.e., the merge regional diffusion) is tested.

### Questions
1. How are the contextual examples used in the second step (Section 3.3) created?

2. How many input text prompts are used in the experiments (Section 4)?

### Soundness
1

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
The paper proposes PMR (Planning, Merging, Replacing), a training-free framework for domain-specific text-to-image generation, exemplified by circuit diagrams. It builds a knowledge base and uses LLMs to plan connectivity, arrange component layouts, and enforce fine-grained positional control, enabling accurate rendering without finetuning. Experiments report superior component and topology fidelity vs. baselines with lower training cost.

### Strengths
The work effectively leverages chain-of-thought (CoT) ideas to operationalize LLMs for domain-specific generation of circuit diagrams. Compared with large models without domain-specific finetuning, the proposed approach shows clear advantages in this task. The paper is well written and technically complete, with clear organization and presentation.

### Weaknesses
The paper does not clearly define domain-specific text-to-image generation. Although the framework is described as training-free with respect to LLMs, the training workload is shifted to an object recognition module, which diminishes the core contribution. Moreover, the work lacks comparisons with this year’s state-of-the-art methods, making the claimed effectiveness insufficiently substantiated.

### Questions
1.	In Related Work, the subsection “Specialized Diffusion Models” should cover Domain-Specific Diffusion Models; currently it does not. The comparisons also lack domain-specific diffusion baselines.
2.	The paper lacks experiments validating the method’s effectiveness on other domains.
3.	Captions for Figs. 2, 3, and 6 should briefly explain the method, rather than only providing a title.
4.	The paper lacks ablation experiments.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces a method to solve the detailed schematic circuit diagram generation by leveraging the LLM for planning and Diffusion model for the diagram generation. Specifically, this work’s pipeline is constructed with three stages: Planning, Merging, and Replacing.

In the Planning stage, this work first plans the component relationship through the CoT process of LLM with knowledge base of schematic circuit diagram preprocessed from diagram images. Then it leverages LLM to plan the regions (positions and sizes) and lines. This pipeline then merges and replaces the latents of each planned region to form the final generation.

### Strengths
1. It successfully uses PMR (Planning, Merging, and Replacing) to achieve training-free generation of schematic circuit diagrams.
 2. It successfully utilizes pretrained models for practical applications.
 3. It proposed a stable and reliable method to generate circuit diagrams.

### Weaknesses
1. This paper should either consider using this pipeline as a syntactical data pipeline and fine-tuned (lora) the Flow Matching model (Flux) with syntactical data for an end-to-end model or test this method for more other domains than the circuit schematic as stated in the title.
2. It does not have qualitative results shown, for example, some sample generated circuit diagram, although it has some generated black blocks.
3. The method this work uses highly correlated to the major backbone of this paper (Yang et al.)

[1] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering textto-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024.

### Questions
1. Could the authors please show the actual generated circuit figure instead of just black boxes?
2. It is highly recommended that the authors do more types of domain specific text-to-image generation other than circuit one?

### Soundness
2

### Presentation
2

### Contribution
2
