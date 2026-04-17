# Visual Prompt-Agnostic Evolution

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Visual Prompt Tuning (VPT) enables effective adaptation of a frozen Vision Transformer (ViT) to downstream tasks by inserting a small number of learnable prompt tokens into the token sequence at each layer. However, we observe that existing VPT variants often suffer from unstable training dynamics, characterized by gradient oscillations. A closer layer-wise analysis reveals that shallow-layer prompts tend to stagnate early, while deeper-layer prompts exhibit high-variance oscillations, leading to a cross-layer mismatch. These issues contribute to slower convergence and degraded final performance. To address these challenges, we propose the Prompt-Agnostic Evolution ($\mathtt{PAE}$) method, which can strengthen vision prompt tuning by explicitly modeling the dynamics of learnable prompts. From a frequency-domain perspective, we initialize prompts in a task-aware direction by uncovering and propagating frequency shortcut patterns that the backbone inherently exploits for recognition. To ensure coherent evolution across layers, we further employ a shared Koopman operator, which imposes a global linear transformation rather than uncoordinated, layer-specific updates. Finally, inspired by Lyapunov stability theory, we introduce a regularizer that constrains error amplification during evolution. Extensive experiments demonstrate that using $\mathtt{PAE}$ with VPT variants not only accelerates convergence with an average 1.41$\times$ speedup but also yields 1–3% gains on 25 datasets with multi downstream tasks. Beyond performance, $\mathtt{PAE}$ remains prompt-agnostic and lightweight, and it integrates seamlessly with diverse VPT variants without backbone modification or inference-time changes, providing a practical and scalable solution for advancing prompt tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Visual cue adaptation effectively adapts frozen pre-trained models by inserting a small number of learnable cue tokens into each layer of ViT. However, existing VPT variants often suffer from dynamic instability in training, manifested by gradient oscillations, early stagnation of shallow-layer cues, and high-variance oscillations of deep-layer cues, leading to slow convergence and ultimately degraded performance. To address these challenges, this paper proposes Prompt-Agnostic Evolution (PAE), which enhances VPT by explicitly modeling the dynamic evolution of learnable cues.

### Strengths
1. PAE is lightweight and independent of prompts, a significant decoupling advantage.
2. It performs well, especially on classification tasks.
3. The derivation is robust, with no major issues.

### Weaknesses
This method introduces multiple new hyperparameters, increasing the complexity of model tuning. Ablation experiments in the paper show that model performance is sensitive to the choice of these parameters. This means that in real-world applications, tedious search and fine-tuning for different tasks may be required to achieve optimal results, which somewhat limits its plug-and-play usability.

The experimental validation in the paper focuses primarily on the specific architecture ViT-Base and the task of image classification. While achieving convincing results on the FGVC and VTAB-1k datasets, the effectiveness of this method on a wider range of visual tasks, such as object detection and semantic segmentation, has not yet been verified. This raises questions about its general applicability.

### Questions
The paper simplifies the evolution of cues across layers to a global linear transformation (the Koopman operator) in a shared latent space. While this linear assumption effectively promotes inter-layer consistency and stabilizes training, does it limit the model's expressive power, especially when dealing with complex tasks or very deep networks with highly nonlinear inter-layer relationships? Exploring nonlinear dynamical models or introducing layer-wise adaptive evolutionary operators might yield further improvements in model performance at the expense of a small degree of simplicity.

Reframing cue tuning as a dynamical system is a novel perspective. In addition to stabilizing training and accelerating convergence, can this framework also be used to improve model interpretability? For example, by analyzing the eigenvalues ​​and eigenvectors of the learned Koopman operator, can we gain insight into how the model adjusts its internal representation to adapt to new tasks? Or can we identify the "dynamic patterns" that are crucial for specific tasks, thereby providing theoretical guidance for more effective cue design?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Visual prompt tuning enables parameter-efficient fine-tuning. However, existing VPT variants often suffer from unstable training. To address this challenge, the authors propose the Prompt-Agnostic Evolution (PAE) by explicitly modeling the dynamics of learnable prompts.

### Strengths
1. The problem of unstable VPT training is well defined. The observations on shallow- and deep-layer prompts are interesting. The clear mismatch for gradient oscillations is something researchers might be interested in. 

2. The paper is easy to follow, and the problem-solving is practical.

3. The ablation study is sufficient.

### Weaknesses
1. The masking then project idea sounds similar to projection-based (a.k.a instance-aware) prompt tuning [1-2], where these papers use input projection directly to guide prompt training. The authors need to discuss them and clearly separate their differences. 

2. The format in conclusion is a little bit weird. Please fix it.

3. The motivation of this paper can be clearer, for example, why the authors want to discover frequency shortcuts. I understand the observations; however, their motivations are unclear to me.

[1] Visual Instance-aware Prompt Tuning

[2] All You Need is One: Capsule Prompt Tuning with a Single Vector

### Questions
My question mainly focuses on the discussions on projection-based prompt tuning methods. Other than that, this paper looks good to me.

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
3

### Summary
This paper addresses a core instability issue in Visual Prompt Tuning (VPT), a fine-tuning method for Vision Transformers (ViTs). The authors observe that many VPT variants suffer from unstable training processes, characterized by gradient oscillations and a cross-layer mismatch phenomenon, where prompts in shallower layers stagnate while those in deeper layers oscillate to compensate. This ultimately slows down convergence and hinders optimal performance. The paper identifies two root causes for this problem: task-agnostic prompt initialization and the independent, uncooperative optimization of prompts at each layer. To resolve this, the authors propose Prompt-Agnostic Evolution (PAE), a framework that treats prompt tuning as a dynamical system. PAE consists of two novel components. The first is Modal Pre-Alignment (MPA), which provides a task-aware initialization by identifying the most discriminative frequency shortcuts for a given task and using them to generate initial prompts. The second is the Koopman-Lyapunov Discrete Dynamical System (KLD), which governs the prompt optimization. It uses a shared Koopman operator to enforce a coherent linear evolution for prompts across multiple layers within a shared latent space, and a Lyapunov-style regularizer to ensure this evolution remains stable. In experiments on the FGVC and VTAB-1k benchmarks, applying PAE to various VPT methods consistently improved accuracy by 1-3% and accelerated convergence by an average of 1.48x. As a prompt-agnostic module, PAE can be integrated into existing methods without modifying the model's backbone and has zero overhead at inference time, making it a practical and effective solution.

### Strengths
The primary strength of this research is its novel conceptualization of prompt tuning as a dynamical system, providing a principled framework to address the observed training instabilities. By applying the Koopman operator and Lyapunov stability theory, it moves beyond empirical heuristics and introduces an explicit mechanism to coordinate prompt updates across layers, directly tackling the optimization mismatch problem. Another key innovation is the Modal Pre-Alignment (MPA) strategy. This method effectively solves the cold start problem in prompt tuning by using a task-aware initialization based on frequency-domain analysis. By identifying the frequency shortcuts already utilized by the pre-trained backbone, MPA provides initial prompts that are well-aligned with the task objective from the outset, which emerged as the single largest contributor to performance gains. Finally, the PAE framework demonstrates remarkable robustness and practicality. Its prompt-agnostic design allows it to be seamlessly integrated as a plug-and-play module into numerous state-of-the-art VPT variants, consistently improving performance in all cases. This proves that PAE addresses a fundamental weakness in the VPT paradigm. The comprehensive empirical validation, which includes not only accuracy metrics but also insightful loss landscape and Grad-CAM visualizations, provides strong evidence for its effectiveness. The fact that there is zero inference-time overhead further solidifies its value for real-world applications.

### Weaknesses
Assumptions of the KLD Framework: The Koopman-Lyapunov Discrete Dynamical System (KLD) assumes that the prompt dynamics can be effectively modeled by a single, global, linear operator. This could create a representational bottleneck for complex tasks where different dynamics in shallow and deep layers might be more beneficial. The framework's success also hinges on the assumption that prompt evolution is approximately linear in the learned latent space, which has not been validated across diverse model architectures and scales. The Lyapunov-style stability constraint, while effective, might be overly restrictive, potentially preventing the model from exploring optimal solutions that require a temporary increase in complexity. Consequently, although performance has been demonstrated, these are results from a single model, and the effectiveness of these simplifications (a single global operator and regularization) may diminish as conditions become more complex. 

Limited Experimental Scope: The paper's empirical validation is limited to a single backbone architecture (ViT-Base/16), so it remains unverified whether PAE's effectiveness generalizes to other architectures. It also does not investigate how performance varies with model scale (e.g., larger or smaller ViTs). Lastly, the absence of a direct comparative analysis with other major PEFT (Parameter-Efficient Fine-Tuning) families, such as LoRA, makes it difficult to fully assess the pros and cons of PAE-enhanced VPT within the broader PEFT landscape. An exploration of its orthogonality and complementarity with these methods would be beneficial.

In conclusion, the study has demonstrated the success of the KLD framework's simplifying assumptions within the specific context of ViT-Base/16. However, it has not been verified whether this success will hold under more complex conditions, such as with more intricate architectures, much larger-scale models, or when combined with other PEFT techniques like LoRA. In other words, a key limitation of this research is that the possibility that the success of this simplification is a coincidence within the limited experimental scope cannot be ruled out.

### Questions
I would appreciate your response to the points raised in the "Weakness" section.

Additionally:
- I would like to know how the convergence speed varies with changes in the hyperparameters alpha and beta.
- I am interested in the performance differences based on intra-class variance. It would be helpful to see the difference in performance gains between the best- and worst-performing classes or class groups in the dataset.

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
3

### Summary
This paper introduces Prompt-Agnostic Evolution (PAE), a framework designed to stabilize and accelerate training in Visual Prompt Tuning (VPT) for Vision Transformers. The authors identify that existing VPT variants suffer from unstable gradients, including shallow-layer stagnation and deep-layer oscillations. To address this, they propose two key components: 1) Modal Pre-Alignment (MPA): A frequency-domain initialization that aligns prompts with task-relevant frequency “shortcuts.” 2) Koopman-Lyapunov Discrete (KLD) system: A shared dynamical model where prompts evolve across layers under a Koopman operator with Lyapunov-based regularization for stability.
Experiments on FGVC and VTAB-1k benchmarks show 1–3% accuracy improvements and 1.5× faster convergence in terms of the number of required epochs.

### Strengths
1. Novel formulation: Reframing prompt tuning as a dynamical system using Koopman theory and Lyapunov stability is novel and mathematically grounded.

2. Comprehensive analysis: The paper clearly diagnoses VPT training instability through layer-wise gradient visualizations and supports it with quantitative results.

3. Strong empirical performance: PAE consistently improves various VPT baselines, showing strong generalization across tasks and benchmarks.

4. Prompt-agnostic applicability: The method is modular, adding no inference-time overhead and requiring no backbone modification.

5. Clear ablations: Ablation studies demonstrate the complementary roles of MPA and KLD, and verify robustness against random initialization and batch selection.

### Weaknesses
- Motivation clarity: While the dynamical-system framing is novel and interesting, the necessity of such complexity for solving gradient oscillation may be overstated. Simpler temporal regularization could have been compared. For example, could simpler smoothing (e.g., temporal moving average across layers) achieve comparable stability?

- Dependence on frequency bias: MPA relies on identifying “frequency shortcuts,” which may not exist or be stable in non-natural image domains, limiting transferability.

- Missing comparison on across-layer effects: The paper does not cite prior work such as [ref1], which examines how prompts interact across layers. A more direct comparison and analysis between that study’s findings and the proposed PAE framework would strengthen the discussion.

[ref1] Improving Visual Prompt Tuning for Self-supervised Vision Transformers, ICML 2023

- Questionable real-world efficiency: Although Fig. 1(a) and Table 1 show faster convergence in terms of epochs, the overall efficiency claim may be overstated. When accounting for MPA initialization overhead and the extra hyperparameters (α, β, K, w, r) that expand the tuning space, the total wall-clock time might actually increase. Hence, the practical benefit in large-scale or hyperparameter-sensitive scenarios remains uncertain.

### Questions
Please see above

### Soundness
3

### Presentation
3

### Contribution
3
