# ReSpace: Text-Driven 3D Indoor Scene Synthesis and Editing with Preference Alignment

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Scene synthesis and editing has emerged as a promising direction in computer graphics. Current trained approaches for 3D indoor scenes either oversimplify object semantics through one-hot class encodings (e.g., 'chair' or 'table'), require masked diffusion for editing, ignore room boundaries, or rely on floor plan renderings that fail to capture complex layouts. LLM-based methods enable richer semantics via natural language (e.g., 'modern studio with light wood furniture'), but lack editing functionality, are limited to rectangular layouts, or rely on weak spatial reasoning from implicit world models. We introduce ReSpace, a generative framework for text-driven 3D indoor scene synthesis and editing using autoregressive language models. Our approach features a compact structured scene representation with explicit room boundaries  that enables asset-agnostic deployment and frames scene editing as a next-token prediction task. We leverage a dual-stage training approach combining supervised fine-tuning and preference alignment, enabling a specially trained language model for object addition that accounts for user instructions, spatial geometry, object semantics, and scene-level composition. For scene editing, we employ a zero-shot LLM to handle object removal and prompts for addition. We further introduce a voxelization-based evaluation capturing fine-grained geometry beyond 3D bounding boxes. Experimental results surpass state-of-the-art on addition and achieve superior human-perceived quality on full scene synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a next-token prediction pipeline for language-driven indoor scene editing.  Specifically, a preference optimization with reinforcement learning with verifiable rewards is used for finetuning. Additionally, a voxel-based loss function metric is used for capturing geometric interactions beyond bounding boxes. Experiment results show better results on object addition in indoor scene synthesis.

### Strengths
This paper has proposed several interesting modules for object addition in given indoor scenes, which includes: 
1. a compact structured scene representation with explicit room boundaries that enables asset-agnostic placement
2. voxelization-based Loss rather than bbox constraint

### Weaknesses
1. It is not clear how Group Relative Policy Optimization (GRPO) is used for training the AR model (L211-212). There is no motivation of using this and advantage $A_i$ is not explained either, which is claimed to be the contribution of the paper. 
2. Computation cost comparison is missing. 
3. I do not find where the overall loss fuction is designed,  since there is envoved with several loss functions. 
4. How to enable the proposed OOB and MBL loss functions in training/optimization the model? The details are not clearly explained. 
5. Missing discussion and comparison with other related autoregressive based indoor scene synthesis/editing works such as: 
[1] FOREST2SEQ: Revitalizing Order Prior for Sequential Indoor Scene Synthesis, https://arxiv.org/abs/2407.05388
[2] CASAGPT: Cuboid Arrangement and Scene Assembly for Interior Design, https://arxiv.org/abs/2504.19478

### Questions
Some parts are not explained clearly: 
1. the modeling of rectilinear polygons in L161
2. What is the differences among A, B and L in L303?
3. How to calculate PMS when ATISS and Mi-Diff do not support language-driven editing? 
4. Curious about whether the model support scene rearrangement. 
5. Any visual limitations?

### Soundness
2

### Presentation
2

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
The paper proposes ReSpace, a text-driven framework for 3D indoor scene synthesis and editing built around a compact Structured Scene Representation (SSR), a specialized SG‑LLM for object addition, and a zero-shot LLM that decomposes user instructions and performs object removal via direct SSR edits. The system adds objects autoregressively, decouples asset selection from layout via a probabilistic sampler, and evaluates layouts with a new Voxelization‑Based Loss (VBL) that counts out‑of‑bounds voxels and mesh‑mesh overlaps.
The SG-LLM is trained in a dual-stage pipeline: first with Supervised Fine-Tuning (SFT) and then with preference alignment (using GRPO), where the VBL metric serves as a verifiable reward signal. Experiments show that ReSpace achieves state-of-the-art results on object addition and, despite mixed results on rendering-based metrics (FID/KID), achieves superior human-perceived quality for full scene synthesis, as validated by a large-scale user study.

### Strengths
1. Editable Generative Representation: The SSR (a JSON format) and SG-LLM successfully frame 3D scene generation as an editable, next-token prediction task.
2. Accurate, Rewarded Alignment: The framework leverages GRPO, using the novel VBL metric as a verifiable reward, to fine-tune the model for more geometrically accurate placements.
3. Superior Performance: ReSpace achieves state-of-the-art results on object addition and, more importantly, is rated by human evaluators as having superior perceived quality for full scene synthesis.

### Weaknesses
1. Lacks other baselines: The other LLM-based baselines, such as LayoutVLM, which also fine-tune the LLM for scene configuration generation, are not included in the paper. Besides, for the scene synthesis task, at least one end-to-end method needs to be compared in the paper, like InstructScene.
2. Modest Impact of GRPO: The paper highlights the SFT+GRPO pipeline as a key contribution. While Table 1 shows GRPO improves quantitative VBL metrics, the appendix (A.6) reveals a crucial finding: a second user study comparing SFT-only vs. SFT+GRPO found no statistically significant human preference (51% vs 49%). This significantly weakens the claim that preference alignment is a key driver of perceived quality. The authors also note "training fragility" and "reward hacking", suggesting this component is difficult to implement and its benefits may be marginal.
3. Limited Editing Capability: The framework is marketed for "synthesis and editing," but the editing functionality is split and weak. Object removal is handled by the ZS-LLM with low accuracy (75.2% on the ‘liv’ split). More complex edits like "move," "rotate," or "resize" are not supported at all, as the authors admit.

### Questions
1. The removal accuracy is a major bottleneck. Is this failure primarily due to semantic ambiguity (e.g., two "chairs" in the scene), or does the ZS-LLM also fail at the task of correctly manipulating the JSON string?
2. For full scene synthesis, why does the model handle the objects one by one? Can we directly generate the SSR for the full scene?
3. What is the per‑object addition latency (SG‑LLM + retrieval + VBL check) and how does BoN scale in interactive settings?

### Soundness
3

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
3

### Summary
This paper proposes a text-driven scene editing method using LLM. The proposed system employs dual-LLM architecture: a primary LLM acts as a natural language interface to translate user instructions into specific prompts, and a second, specially fine-tuned Scene-Graph LLM (SG-LLM) generates and edits a Structured Scene Representation (SSR) based on those prompts. The SG-LLM is trained using supervised fine-tuning followed by preference alignment. The authors demonstrate state-of-the-art performance on the 3D-FRONT dataset.

### Strengths
The authors introduces a text-to-indoor scene editing method by leveraing LLMs to autoregressively predict next object with preference alignment.
The paper proposes a new evaluation metric Voxelization-Based Loss (VBL) to measure fine-grained geometric interactions among the room boundaries and 3D objects.

### Weaknesses
Lack of Representation Justification: The paper claims to use a Structured Scene Representation (SSR), but it does not sufficiently justify its advantages over other contemporary scene representations formulated as language, such as those in LayoutGPT [1], SceneScript [2], or SpatialLM [3]. Furthermore, encoding scene boundaries is not novel, as methods like Ctrl-Room [4] explicitly model complex walls as part of the generative process. In contrast, this work treats boundaries only as a fixed input. A comparative analysis of SSR's expressiveness and efficiency is needed.

Unmotivated Dual-LLM Architecture: The rationale for the two-LLM pipeline is unclear. A critical question is whether the SG-LLM, trained specifically on SSR data, is prone to overfitting and fails to generalize to direct natural language input, thus necessitating the first LLM as an intermediary. This relates to a core challenge in domain-specific LLMs. The authors should analyze this potential issue and strengthen their experimental validation by including more complex and realistic datasets like Structured3D[5] or the one used in SpatialLM [3] to test the model's robustness. 

Limited Experimental Scope: the chosen baselines are not the most recent, and newer methods like InstructScene[6] and SceneWeaver[7] also support intuitive, text-driven editing. More importantly, the 3D-FRONT dataset is known for its simplicity in room diversity and layout complexity. To fully establish the method's efficacy, experiments on more challenging datasets with greater object count and layout variety are encouraged.

Some minor weaknesses:
•Inconsistency in Claims: The authors criticize previous works for being limited by simplified object categories and fixed floor plans. However, the proposed method itself operates under similar constraints, as it starts from a partial scene and retrieves objects from a fixed dataset. The authors should more clearly articulate how their method advances beyond these limitations.
•Clarification of Formulation: Equation 1 requires clarification.
1) What does the p_i represent ?
2) Is \mathcal U_i equivalent to Tok(S_i) ? The relationship between these terms should be explicitly stated.
•Clarification of Prompt Bank: The role and composition of the prompt bank \mathcal P(o) are not well-explained. How does it relate to the sequence modeling process defined in Equation 1?


[1] LayoutGPT: Compositional Visual Planning and Generation with Large Language Models 
[2] SceneScript: Reconstructing Scenes With An Autoregressive Structured Language Model 
[3] SpatialLM: Training Large Language Models for Structured Indoor Modeling 
[4] Ctrl-Room: Controllable Text-to-3D Room Meshes Generation with Layout Constraints 
[5] Structured3D: A Large Photo-realistic Dataset for Structured 3D Modeling
[6] InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior

### Questions
See my discussion in weakness

### Soundness
2

### Presentation
2

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
The paper introduces ReSpace, a framework for autoregressive indoor scene generation and editing from natural language prompts, representing the current scene with a Structured Scene Representation (SSR). The framework is designed to perform three main tasks. For (a) object removal, a zero-shot LLM directly edits the SSR, and for (b) full scene generation, the same zero-shot LLM produces object prompt list, which is fed to SG-LLM. SG-LLM is trained for (c) single object addition with SFT+GRPO, which predicts the next object placement given the SSR and an object prompt. Additionally, a voxelization-based loss is introduced to capture fine-grained geometric details that bounding-box metrics fail to reflect.

### Strengths
1) Prior work on scene synthesis predominantly employs global optimization, which is not well-suited to scene editing. Sequential synthesis is a natural way to support editing tasks.
2) The paper introduces a voxelization-based loss that captures fine-scale details and evaluates spatial arrangements more accurately than bounding-box metrics.
3) Preference alignment via GRPO is novel, and seems to achieve good results.

### Weaknesses
1) Missing comparison with recent baselines: ATISS (NeurIPS 2021), LayoutGPT (NeurIPS 2023), and Mi-Diff (2024) are relatively old for full scene synthesis evaluation, making it difficult to assess the proposed model’s performance in the current landscape. More recent baselines such as LayoutVLM [2] report stronger results than those selected here.
2) Generation time: Autoregressive generation is likely slower than in-context learning methods like LayoutGPT. However, end-to-end generation time for a full scene is not reported.
3) Lack of global re-optimization/correction: Although an autoregressive approach is natural for editing, as stated in L460–462, it is not directly suitable without a global correction/re-optimization step (e.g., when there is no space for a large object). A global layout solver or feedback mechanism applied before each insertion could mitigate these limitations and achieve the best of both worlds.

[1] Sun, F. Y., Liu, W., Gu, S., Lim, D., Bhat, G., Tombari, F., ... & Wu, J. (2025). Layoutvlm: Differentiable optimization of 3d layout via vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 29469-29478).

### Questions
All of my concerns are listed in the weaknesses section, and I may adjust the rating if they are well addressed.

### Soundness
3

### Presentation
3

### Contribution
3
