# Energy-Guided Prompt Optimization for Controllable Cross-Architectural Diffusion Models

- Decision: Reject
- Scores: 2, 4, 6, 0

## Abstract
Diffusion models are central to text-to-image synthesis, yet enforcing semantic constraints such as exclusion and negation remains challenging across architectures. We propose a unified, training-free intervention that combines diagnostic instrumentation with a principled sampling-time optimizer to improve constraint adherence without modifying pretrained denoisers. The diagnostic module uses latent attribution and Jacobian analysis to reveal sensitivity to textual conditioning and guide conservative hyperparameter initialization. The optimizer shapes a smooth semantic potential on the CLIP manifold and applies Hamiltonian updates with mild stochasticity, enabling manifold-aware corrections and a distributional interpretation via a semantic Fokker–Planck equation. Experiments on multiple diffusion variants and datasets show that inference-time energy shaping significantly improves negative-prompt compliance while preserving perceptual quality. Our approach advances controllable generation by integrating model introspection and theoretically grounded constrained sampling into a lightweight, architecture-agnostic procedure.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces EGP, a training-free framework for improving semantic constraint enforcement in DMs, particularly for negation and exclusion prompts. The approach combines two main components: (1) a Jacobian-based diagnostic tool for analyzing how different model architectures respond to constraints, and (2) an energy-guided optimization method that reshapes the latent space during sampling to avoid unwanted concepts. The authors evaluate EGP across multiple diffusion model architectures (SD-2.1, SD-3.5, SD-XL, Flux) and demonstrate improvements in constraint adherence (Neg-ACC) while maintaining image quality. The method operates by adding gradient-based correction steps during DDIM sampling, using CLIP embeddings to measure similarity between generated images and negative prompts.

### Strengths
- The EGP algorithm (Algorithm 1) is well-documented with step-by-step details, making the method reproducible. The mathematical formulations are generally precise and the energy function design is well-motivated.
- The paper includes thorough experiments across multiple diffusion architectures (SD-2.1, SD-3.5, SD-XL, Flux), multiple datasets (COCO, MedicalX, ComicArt, AbstractPrompt), and multiple evaluation metrics (FID, LPIPS, CLIPScore, Neg-ACC). Human evaluations add credibility to the quantitative results.
- The method's ability to work across different model architectures without requiring retraining is a practical advantage. This makes it broadly applicable to existing pretrained models.
- The connection to energy-based models and constrained sampling provides theoretical grounding for the approach.

### Weaknesses
- What is the specific purpose of the Jacobian diagnostic in your framework? The authors mention it's proposed to analyze different existing models and adapting EGP without training (lines 071-075), but I don't see experiments demonstrating this (nothing provided in the main paper, only providing some contexts in appendix C). How does the Jacobian analysis inform the EGP method? Can you provide clear examples of how the diagnostic guides model selection or parameter tuning?
- The main technical contribution: using ReLU thresholding with energy-based guidance, is a relatively incremental modification over existing negative prompting techniques (e.g., SLD [R1]). While theoretically motivated, the practical novelty feels limited.
- The paper emphasizes being "cross-architectural" but all evaluated models are text-to-image DMs with similar underlying architectures (latent diffusion with U-Net variants). It's unclear what architectural diversity is actually being addressed. What specific architectural diversity are you addressing? Would the method work on fundamentally different architectures (e.g., autoregressive models, GANs, or diffusion models with different parameterizations)?
- The paper doesn't clearly explain what base model EGP uses or how it builds upon existing models. Does it use a specific SD variant as foundation? Does it combine different pretrained components? The strong performance of EGP could be partially due to the underlying base model capabilities rather than purely the EGP contribution.

[R1] Schramowski, Patrick, et al. "Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models." CVPR. 2023.

### Questions
- The method adds ~40% latency and ~46% more FLOPs compared to baseline (Table 7). While this overhead is mentioned, the paper doesn't adequately discuss whether this cost is justified or explore more efficient alternatives in the main experiments.
- Since EGP is training-free, why not position it as a complementary module that can enhance existing models rather than as a standalone method? This would make it clearer that EGP adds value on top of any base architecture. Your current framing makes it seem like a separate model competing with SD variants.
- Have the authors conducted ablations showing that the Jacobian diagnostic actually improves EGP's performance? If the diagnostic is a key contribution, it would be helpful to see experiments demonstrating its utility.
- How sensitive is the method to the choice of threshold τ = 0.25? How should practitioners set this threshold for new concepts or domains? Is there a principled way to select it?
- Some guidance-based negative prompting methods for DMs [R1, R2, R3] are specifically designed for safety applications, particularly NSFW content detection and mitigation. Have you considered or evaluated EGP for this important use case? Given that your method is training-free and focuses on constraint enforcement, it seems like a natural and practically important application. If you have explored this, what are the results? If not, could you discuss whether your approach would be suitable for safety-critical constraint enforcement, and what modifications (if any) might be needed?

[R1] Schramowski, Patrick, et al. "Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models." CVPR. 2023. \
[R2] Yoon, Jaehong, et al. "Safree: Training-free and Adaptive Guard for Safe Text-to-Image and Video Generation." ICLR. 2025. \
[R3] Liu, Zhili, et al. "Implicit Concept Removal of Diffusion Models." ECCV. 2024.

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
3

### Summary
The paper aims to improve the semantic constraint enforcement of text-to-image diffusion models.

The paper proposes a training-free energy-based optimization technique that corrects the latent at each timestep in the generation process with gradient-based optimization of an energy function, in order to suppress the generation of negative concepts.

### Strengths
1. Enhancing the controllability of text-to-image models is an important research problem.

2. The results of the proposed energy-based optimization method are promising.

### Weaknesses
1. The paper is claimed to contribute two major components, including 1) Jacobian-based diagnostic and  2) energy-guided optimization, in the abstract and introduction parts, where they are posed as almost equally important. However, the Jacobian-based diagnostic is only briefly introduced in Section 3.2, and empirical evidence on its utility is not provided in the main paper, but rather in the appendix. In Appendix C, the paper only uses a comparison between two guidance methods (DNG and EGP in Table 6) to show the correlation between Jacobian changes (measured by $L_F$ and $R_\sigma$) and  constraint adherence (measured by Neg-ACC). More comprehensive experimental evidence on a wider range of model architectures is missing, which should be added to make the conclusion more convincing. In addition, a preliminary experiment on how the Jacobian-based analysis can help architectural selection and design for better constraint adherence should be provided, from which findings would be much more interesting and inspiring to the community. Such an experiment, unfortunately,  remains missing in the current paper. The above issues would raise concerns about the significance of the Jacobian-based diagnostic and the amount of contribution involved in the paper.

2. The design choice of $\Delta J_t$ is not well justified and explained. What motivates the Jacobian-based definition of $\Delta J_t$? Why can $\Delta J_t$ quantify the architectural differences between two models?

3. Section 3 is difficult to follow, mainly because some concepts suddenly appear without any context, some concepts are not used after being introduced, and connections between some concepts are not clear.  For example, what are unconstrained ($\mathcal{E}_u$) and constrained ($\mathcal{E}_c$) energy functionals in Section 3.4? What are their relationships to the energy functional ($\mathcal{E}$) in Section 3.3? Where is the objective $\mathcal{L}$ in Eq. (13) used? What is the relationship of $\mathcal{L}$  to the update equation in Eq. (14)?

4. Experimental results are incomplete. The visual comparison of images generated by different models is lacking in Section 4.2.2, and should be added. The results on Representation Balance Index (RBI) are missing in the paper.

### Questions
1. In the experiments, which diffusion model is the proposed EGP method based on?

2. What is the detailed setup for the experiment in Section 4.2.3? What are inputs to different models? Why is the EGP compared with only SD-2.1 and SD-XL?

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
3

### Summary
The paper introduces Energy‑Guided Prompt Optimization (EGP), a method that enforces negative prompts during image synthesis to suppress unwanted content. At each DDIM reverse‑diffusion step the latent trajectory is adjusted so that it satisfies the negation constraints while preserving the Markov property of the diffusion process . Experimental results show that EGP attains the highest negative‑prompt accuracy, while achieving comparable or lower FID values.

### Strengths
- The method enforces negative prompts through an energy formulation, so it improves adherence without any model retraining.
- Experiments show that EGP attains the highest Neg‑ACC score while maintaining (or slightly improving) CLIPScore and FID, indicating superior handling of negations without sacrificing visual fidelity.

### Weaknesses
- It is not described what the qualitative assessment (Sec. 4.2.2) is based on
- Although the introduction claims a new latent‑space attribution metric, the metric is never evaluated or used in the experiments.
- Overall, the writing of the paper could be polished further:
	- The related‑work section reads like a series of disconnected paragraphs, which is why it is a bit difficult to follow the paper.
	- The experimental section seems to be a collection of independent paragraphs. More transitions and further explanations/analyses would be nice and help the reader to follow and understand the paper better.
- In 4.4, the table is placed in the middle of the text, disrupting the reading flow.


Minor things:
- Oftentimes, after an equation, the sentence is ended with a full stop. However, in the next sentence, it is continued with "Where ...". Either the full stop has to be changed to a comma, or the second sentence has to be edited.

### Questions
Q1: How is the runtime affected? Is the inference slower when applying EGP?  
Q2: Since the authors did not include an ethics statement: Could this technique also be used for guiding towards malicious concepts?  
Q3: What is the quality assessment in Section 4.2.2 based on? 
Q4: Where is the latent-space attribution metric used?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper proposes an Energy-Guided Prompt Optimization (EGP) framework aimed at improving the controllability and cross-architecture consistency of text-to-image diffusion models. Specifically, it introduces an energy-based correction mechanism applied during the sampling process to enhance the effectiveness of negative prompts and align outputs across different diffusion architectures (e.g., Stable Diffusion 2.1, XL, and Flux). The authors further propose a Jacobian-based diagnostic tool to analyze model sensitivity to textual conditioning, arguing that such differences explain inconsistencies between architectures. Experimental results are reported on several datasets, suggesting that EGP improves negative prompt adherence and semantic alignment without retraining. The paper positions its contributions as a unified, training-free approach to controllable text-to-image generation and cross-model consistency analysis.

### Strengths
The paper addresses an underexplored aspect of diffusion-based generative models—cross-architecture consistency and negative prompt reliability—which reflects an effort to formalize controllability issues that are often treated heuristically. The introduction of an energy-guided optimization framework at inference time represents a creative adaptation of energy-based modeling concepts to prompt control without additional training. The Jacobian-based diagnostic for analyzing model sensitivity provides an interpretable, theoretically motivated lens for comparing architectures, which could inspire future research on representational alignment in generative models. The manuscript is generally clear in its high-level organization and conveys the intuition behind energy-guided correction in a way accessible to readers familiar with diffusion processes.

### Weaknesses
The paper presents serious issues in clarity, structure, and overall scholarly presentation, which significantly detract from its readability and impact. The writing often lacks grammatical and logical precision—for instance, the very first sentence in the introduction is syntactically broken and semantically unclear, making it difficult to understand the problem being introduced. Many passages are written in a way that feels disconnected or overly verbose, with inconsistent use of terminology and weak linkage between motivation, method, and results. The manuscript also raises questions about its adherence to standard conference formatting conventions—for example, the inclusion of a “Keywords” section seems unusual for this venue. From a technical standpoint, while the idea of energy-guided prompt optimization is conceptually interesting, the theoretical grounding remains vague, and the motivation for enforcing cross-architecture consistency is not convincingly justified.

### Questions
1. The introduction lacks grammatical precision and conceptual clarity, particularly in the opening sentence, which makes it difficult to grasp the main motivation and scope of the work. Clarification of the introduction framing and problem definition would help establish a clearer context.

2. The purpose and value of pursuing cross-architecture consistency remain uncertain. Further explanation is needed to clarify whether such consistency is a scientifically meaningful objective or an engineering consideration, and how it contributes to broader progress in diffusion modeling.

3. The distinction between the proposed energy-guided prompt optimization and prior inference-time control methods is not clearly demonstrated. A more explicit comparison—both conceptually and experimentally—would help assess the novelty of this approach.

4. Certain formatting choices, such as the inclusion of a “Keywords” section and unconventional section organization, raise questions about adherence to the conference submission format. Clarification of whether these reflect intentional stylistic decisions or formatting oversights would be helpful.

### Soundness
2

### Presentation
1

### Contribution
2
