# VoxelPrompt: A Vision Agent for End-to-End Medical Image Analysis

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
We present VoxelPrompt, an end-to-end vision system that tackles composite radiological tasks. Given a user prompt, VoxelPrompt integrates a language model that generates executable code to invoke a novel, jointly-trained vision network. This adaptable network can integrate any number of volumetric (3D) inputs across heterogeneous real-world clinical modalities to segment and characterize diverse anatomy and pathology. Predicted code employs this network to carry out analytical steps to automate practical quantitative pipelines, such as measuring the growth of a tumor across visits, which often require practitioners to painstakingly combine multiple specialized but brittle tools. We evaluate VoxelPrompt using diverse neuroimaging tasks and show that it can delineate hundreds of anatomical and pathological features, measure complex morphological properties, and perform open-language analysis of lesion characteristics. VoxelPrompt performs these objectives with an accuracy similar to that of specialist single-task models for image analysis, while facilitating a broad range of biomedical workflows.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work focuses on improving generalization of language-vision medical image analysis to diverse practical clinical use cases across real-world lesions types. Specifically, VoxelPrompt is an agent-based approach that coordinates outputs from vision and language models to produce executable code that performs tasks such as ROI measurements, image manipulation, language characterization, and analyses across multiple ROIs, multiple acquisitions, and multiple visits. The authors also propose a CNN model that integrates information from language prompts and processes images in native resolution. Furthermore, they propose a procedure for constructing a dataset that that improves robustness to lesion types across different datasets.

### Strengths
- The use of agents to produce code seems like a promising approach to generating interpretable operations on medical images. 
- The evaluations seem to consider a broad range of relevant use cases
- The lesion synthesis procedure seems to be a novel approach. 
- This work also highlights an important need for free-form workflow benchmarks that capture common practitioner use-cases

### Weaknesses
- **Unclear notation**: 
    - Section 3.1: The notation that is introduced is not precise. What exactly is $V, p, a, \Omega, W$, etc.. Are these vectors, scalars, matrices, functions, etc.? 
        - Line 186: $\phi$ is defined as “image-specific latent instruction embedding”. However, there is no further detail about what this object is or where it comes from. Similarly, where does $\phi_s$ (line ~209) come from?
        - Line 210: The shape notation $\mathbb{R} ^{S,c}$ is not standard. I think you mean $\mathbb{R} ^{S \times c}$, but it is not clear in the text.
- **Unclear description of model architecture**: 
    - Line 207: I’m not sure I understand the role of “streams”. How are these different than the standard intermediate activations from a transformer layer, where inputs are processed separately before fusion with an attention mechanism? 
    - Line 215-222: The section on native space processing is unclear. What is the base architecture and what is the upsampling and downsampling arm referring to? It would also be helpful to elaborate on what “common geometry” (line 222) mean in this context. What exactly is being updated to adapt the model for different resolutions? What mechanism allows enables the sharing of model weights across these different resolutions? 
- **Unclear description of Dataset generation procedure**: 
    - Overall, section 3.2 is quite vague. In general, it is not clear what the exact tasks are, what their inputs and outputs are, how they are generated, or what are the parameters that can be varied during generation. Concretely, here are a few examples:
        - Line 244-250: In section “training code for quantitative ROI processing”, it is unclear if these tasks are considered separately or if they are combined together. How are multiple tasks sampled during training?
        - Line 252: what exactly are the relevant metrics? Please be specific. What is the objective function that you are optimizing?
        - Line 260: In section “training code for question answering”, Its not clear how correct natural language text response is generated. 
        - Line 263: how did you construct these templates? Can you provide some examples of what these templates look like?
        - Line 266: How do you make sure that the generated prompt doesn’t produce invalid combinations?
- **Unclear description/motivation for the evaluation**:
    - Line 311: It is unclear what zero-shot lesion segmentation is intended to evaluate. Could you clarify what type of generalization the held-out pathology datasets are designed to test? Are we testing for generalization to unseen diseases or same diseases but unseen populations or unseen tasks, or something else?
    - Section 4.2. What is specialist network referring to? Can you provide a citation here? 
    - Could the authors clarify how the evaluations in Section 4 validate the use cases presented in Figure 1? At present, it is not clear whether VoxelPrompt successfully accomplishes its intended use cases.
- **Insufficient baselines**. For certain evaluations, the authors evaluate compare against only a single baseline which is insufficient in evaluating how the proposed approach compares to the existing literature.
    - Fig 3D: Why not compare performance against other VLMs? My intuition says that other VLMs will achieve a similar runtime improvement when compared against FreeSurfer.  
    - Fig 3E: Same comment here, why did you only choose to benchmark against SynthSeg? Why not compare against other segmentation or VLM models?

### Questions
Please see weakness section for the majority of questions. Here are a few additional questions:
1. In line 198, How do you guarantee that the feedback procedure does not result in an infinite loop?
2. How well does this approach generalize to new tasks, especially more complex multi-step tasks? It seems like the training procedure is quite specific to simple tasks that are provided during training. Is the dataset generation approach scalable? 
3. What is the effect of synthetic lesion training? Can you do an ablation to show that the proposed approach is effective?

### Soundness
2

### Presentation
2

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
This paper presents VoxelPrompt, a system that combines a language model agent with a jointly-trained vision network to perform complex neuroimaging analysis tasks.

### Strengths
The joint training of language and vision components for code-based workflow generation is interesting. The code generation approach provides transparency and interpretability compared to black-box vision-language models, which is important for clinical applications.

### Weaknesses
1. "End-to-end" is claimed throughout, but most quantitative evaluations are on segmentation subtasks, not complete clinical workflows.
2. Training the language model from scratch on synthetic template-based prompts is a critical limitation explicitly acknowledged: "limits their utility when given entirely unseen prompts". The system can only use predefined library functions, limiting true flexibility.
3. Missing quantitative evaluation of the complex multi-step workflows shown in Figure 1. And small sample sizes for some tasks (e.g., n=12 subjects for vascular territory classification).
4. There are some Baseline Comparison Concerns. Different prompts for different baselines (shown in Table on p.19) is unclear if BiomedParse v2 and SAT received optimal prompting.
5. Fine-tuning a pretrained language model (even small ones like CodeLlama, StarCoder) would likely improve prompt generalization significantly. The choice to train from scratch seems inefficient and limits performance.
6. Some notation inconsistencies (e.g., E used for both encoder output and feature encodings).
7. The longitudinal FreeSurfer comparison is somewhat unfair. FreeSurfer performs full cortical reconstruction, not just segmentation.

### Questions
1. What percentage of queries result in code execution errors?
2. How does the system handle code execution failures, invalid operations, or edge cases?
3. What happens when users provide prompts significantly different from training templates?
4. Can the approach extend to other body regions (chest, abdomen) or modalities without retraining from scratch?
5. For RadFM, what performance would be achieved with the full 32-layer model? Could other memory optimization strategies enable fair comparison?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper propose a code agent framework to assist radiologist of free-form radiological tasks. Experimental reuslts show the current system seems work well in some author defined tasks.

### Strengths
The paper is well-written in structure and readers are easy to follow.

### Weaknesses
1. An Inefficient and "Weak" Agent: The paper's most significant limitation is that its agent is trained from scratch on a curated, domain-specific dataset. This results in an agent that is, by definition, "weaker" and less capable in its reasoning, language understanding, and code-generation abilities than any modern, general-purpose foundation model (such as the Gemini or GPT series). The field has largely demonstrated that the emergent reasoning and planning capabilities of large-scale models are a prerequisite for robust agentic behavior.

2. Ignores the Superior LGM-as-Agent Paradigm: As you noted, the tasks described (e.g., "calculate progressive signal reduction") are complex, multi-step analytical workflows. The current, established approach for this is to use a powerful, pre-trained foundation model as a central "agent" or "orchestrator." This agent then intelligently calls upon a suite of specialized tools—which could include the VoxelPrompt vision network itself—via APIs. The paper's design, which builds a weak, custom agent instead of leveraging a powerful general one, seems to solve the wrong problem.

3. Lack of Verifiable Generalizability: The "from-scratch" approach makes the agent "brittle." Because the vision and language models are co-trained on this specific neurological dataset, the agent's "intelligence" is inextricably tied to this single domain. There is no evidence it could generalize to any other task (e.g., analyzing chest X-rays, a slightly different MRI protocol). This lack of zero-shot capability makes it impossible to verify the agent's generalizability, as its performance is completely coupled with its training data.

### Questions
See Weaknesses for details.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces VoxelPrompt, an end-to-end system that combines language-driven code generation with a unified vision model. The method combines a language model and a cnn by allowing the language model to augment the cnn with code, and thus perform unconstrained analysis of the input.

### Strengths
The proposed method stands out for its originality, offering a fundamentally new approach in both functionality and design compared to existing medical image frameworks. Its flexibility and performance are highly compelling.

* The innovative use of code as an output of a language model, enabling segmentation to serve its true role as an intermediate step toward downstream clinical or analytical objectives.
* A thorough and convincing evaluation that clearly demonstrates the method’s effectiveness.
* The joint training of a language model and a segmentation model from scratch, a novel and effective strategy.
* The ability to process scans at native resolution is useful and non-trivial with cnns. 
* The used datasets are interesting, however underspecified. In particular the q&a pairs, would, if released, represent a contribution to the community.

### Weaknesses
The primary limitation of the paper lies in the presentation of the proposed method. Its exact capabilities and constraints are not clearly defined. Crucial details regarding the architecture, datasets, and evaluation are relegated to the appendix, forcing the reader to consult supplementary material to grasp the approach.

* The use of the term “zero-shot” is misleading, as the model has been exposed to the same ROIs during training; the evaluation therefore reflects domain transfer rather than true zero-shot performance.

* The evaluation section omits essential information about the baseline specialist models, preventing a sound assessment of comparative performance. Similarly, the available tools and interfaces accessible to the language model are not specified.

* The term “agent” is used without sufficient justification.

* Key procedural details are missing regarding the generation of Q&A pairs: it is unclear whether templates are used, what their variation is, and whether the language model generalizes beyond them.

* An ablation study examining the language model’s ability to correctly identify and interpret the intended task is also absent, limiting interpretability of the reported results.

* Lastly, the paper lacks important information on the tools available for the agent to use -- i.e. which functions can be called when and in what order.

### Questions
* Can the proposed network be zero-shot adapted to arbitrary tasks, or is adaptation limited to tasks represented in the training dataset?
* How many distinct regions of interest (ROIs) are included in the dataset?
* In this context, does “zero-shot” refer to ROIs unseen during training?
* What are the specialist models referenced in Figure 4A, and do they represent current state-of-the-art baselines?
* Do all evaluated tasks involve segmentation, or are other modalities or objectives included?
* Will the datasets used in the study, particularly the Q&A pairs, be publicly released?
* What is the solution space available to the language model? Which functions can be called when and in what order?

### Soundness
3

### Presentation
2

### Contribution
4
