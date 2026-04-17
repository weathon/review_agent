# AURA: Visually Interpretable Affective Understanding via Robust Archetypes

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Text-driven vision--language methods (e.g., CLIP variants) face three persistent hurdles in affective computing: (i) limited support for continuous regression (e.g., Valence--Arousal), (ii) brittle reliance on language prompts, and (iii) the absence of a unified paradigm across expression classification, action-unit detection, and affective regression. We introduce AURA, a prompt-free framework that operates directly in a frozen CLIP visual space via visual archetypes. AURA comprises two components: (1) self-organized archetype discovery, which adaptively allocates the number of archetypes per affective state, assigning denser archetype sets to complex or ambiguous states for fine-grained interpretability, and (2) archetype contextualization, which models interactions among the most relevant archetypes and semantic tokens to enhance structural consistency while suppressing redundancy. Inference reduces to cosine matching between projected features and fixed archetypes. Across six datasets, AURA consistently surpasses prior state-of-the-art while remaining highly efficient. Overall, AURA unifies classification, detection, and regression under a single visual-archetype paradigm, delivering strong accuracy, cognitively aligned interpretability, and excellent training/inference efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes AURA, a prompt-free framework for affective understanding that operates via self-organized visual archetypes in a frozen CLIP visual space. It aims to unify classification (FER), detection (AU), and regression (VA) tasks, claiming SOTA performance, improved interpretability, and high efficiency across six benchmarks.

### Strengths
1. Originality: Introduces a unified visual-archetype paradigm, moving away from brittle text prompts.

2. Significance: A well-motivated approach with potential to influence affective computing and interpretable ML.

3. Experiments: Comprehensive evaluation across three tasks; ablation studies support design choices.

### Weaknesses
1.  **Misleading Claim of "Direct" Operation in CLIP Space:** The paper repeatedly states it operates "directly in a frozen CLIP visual space." However, the core of the method relies on a **trainable Visual Archetype-Space Projector (VAS)**, `ℱ(·)`, which transforms the original CLIP features into a new, task-specific "archetype space." This is not a direct operation but a learned adaptation. This phrasing is conceptually misleading and should be clarified to avoid confusion.

2.  **Insufficient Detail on Adaptive Archetype Mechanism:** While the "self-organized archetype discovery" is a key contribution, the process for adaptively determining the final number of archetypes per state is underspecified. The text mentions it "adaptively discovers around 100 archetypes" but provides no concrete criteria, convergence conditions, or initialization strategy for this process. The lack of reproducibility details here is a significant methodological weakness.

3.  **Inadequate Explanation for Feature Granularity Results:** The ablation study (Table 2) shows that patch-level features, intended for local AU detection, slightly outperform global features on RAF-DB (a global FER task). This counter-intuitive result is not discussed or analyzed. The authors should investigate whether this indicates that local features are unexpectedly beneficial for certain expression categories or if it reveals a more complex relationship between feature granularity and task performance that is not yet understood.

### Questions
1.  Please clarify the methodological description: The framework learns a projection from the CLIP space to an archetype space. Revising the "operates directly" claim would improve conceptual accuracy.
2.  What are the specific, concrete criteria or algorithm for determining the final number of archetypes `K` in the self-organizing process? How is it initialized and when does the adaptation converge?
3.  How do you explain the superior performance of patch-level features on the global FER task (RAF-DB) in your ablation study? Please provide a hypothesis or further analysis.
4.  Please provide all supplementary material (Appendices A, C) for a complete assessment of implementation details and reproducibility.

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
5

### Summary
The paper proposes AURA (Affective Understanding via Robust Archetypes), a prompt-free framework for affective computing that operates in the frozen visual embedding space of CLIP. AURA unifies facial expression recognition (FER), action unit detection (AUD), and valence-arousal estimation (VAE) through a shared paradigm based on learnable visual archetypes. These archetypes are discovered via a self-organized mechanism that adaptively allocates more prototypes to affective states with higher visual or semantic complexity. Predictions are made by cosine matching between projected features and fixed archetypes, enabling efficient inference. The method is evaluated across six benchmarks and reports state-of-the-art results with low computational overhead.

### Strengths
1. A new framework is presented that integrates three different affective tasks into a unified architecture.
2. The model is lightweight and computationally efficient, with a low number of parameters and floating-point operations (FLOPs), making it suitable for practical use.
3. An adaptive prototype allocation mechanism helps determine the number of prototypes for each class or state, based on the data, reflecting the emotional complexity of the task.
4. Task-aware regularization techniques are introduced for both classification and regression tasks, including score-based attraction and repulsion for continuous labels.
5. Evaluation is conducted across multiple datasets and tasks, such as FER, AUD, and VAE, with a clear comparison to state-of-the-art (SOTA) methods.

### Weaknesses
1. The use of only the visual branch of CLIP for purely visual tasks is a reasonable design choice, but this approach is widely adopted in recent literature and does not represent a conceptual novelty. The core mechanism relies on learnable prototypes, a well-established technique in metric and few-shot learning.
2. While the introduction of the term "archetype" may create the impression of conceptual novelty, the main technical difference seems to be the adaptive density of prototypes based on label complexity - a useful but incremental extension.
3. Interpretability is primarily demonstrated through post-hoc archetype visualizations and error diagnosis, but it is unclear how this approach can explain individual model decisions like attention-based or saliency-based methods. The current form of interpretability is more useful for dataset curation rather than understanding per-sample model behavior.
4. While archetype visualizations highlight annotation errors, the paper does not analyze cases where the model’s prediction is wrong despite correct labels, which would better test the robustness of the archetype-based representation.
5. The framework is applied separately to each task, using task-specific archetypes and heads. This limits the claim of "unified modeling", as there is no shared representation or joint optimization across FER, AUD, and VAE.

### Questions
1. The authors could explain why they use the term "archetype" beyond stylistic or motivational reasons. Is there a specific reason why this term is necessary for their approach? Is there any formal distinction between their approach and other methods of adaptive prototype learning?
2. How would AURA explain a correctly labeled but misclassified sample?
3. It would be interesting to see if it is possible to train a shared set of archetypes for FER, AUD, and VAE using multi-task learning. If this is not possible, what are the fundamental reasons for this?

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
This paper introduces AURA, a prompt-free framework for affective understanding that operates directly on frozen CLIP visual embeddings. The core problem it addresses is the threefold limitation of existing VLM-based methods in affective computing: (i) incompatibility with continuous regression tasks (like Valence-Arousal), (ii) reliance on brittle and labor-intensive text prompts, and (iii) the lack of a unified paradigm across Facial Expression Recognition (FER), Action Unit Detection (AUD), and VA regression (VAE). AURA's main contribution is a "Self-organized Archetype Discovery" module, which learns a set of visual archetypes that serve as perceptual anchors.

### Strengths
1. The shift from a text-dependent, "text-image matching" paradigm to a "visual-archetype matching" one is a significant and compelling contribution. This prompt-free approach directly solves a major, practical bottleneck in applying VLMs to specialized domains like affective computing.

2. The ability of a single framework to successfully unify classification (FER), detection (AUD), and continuous regression (VAE) is a major strength. The paper's design choices, using global-level embeddings for FER/VAE and patch-level embeddings for AUD, combined with distinct regularization strategies, are well-motivated and empirically validated.

3. The paper achieves SOTA performance across all three task categories on six different benchmarks. It achieves this with the lowest parameter count and FLOPs compared to other SOTA methods. The inference-time design (omitting the contextualization module and using only a projector and cosine matching) makes it extremely practical for deployment.

### Weaknesses
1. While inference is exceptionally simple, the training process appears highly complex. The total loss is a weighted sum of three major components. $\mathcal{L}^{Arc}$ is itself a composite of three other losses 23, one of which ($\mathcal{L}^{Reg}$) has two entirely different formulations depending on the task. The paper provides little to no discussion on the sensitivity to the various weighting coefficients ($\lambda_{Proj}$, $\lambda_{Arc}$, $\lambda_{Contx}$) or the margins ($m$) used in the regularization losses. This complexity could be a significant barrier to reproducibility.

2. The paper claims to evaluate on a video benchmark (DISFA). However, the AURA architecture appears to be a frame-based processor. It uses patch-level features for AUD, but there is no mechanism described for modeling temporal dependencies across frames. Therefore, its excellent results on DISFA are for frame-level AU detection, not "video-level" analysis in a temporal sense.

### Questions
The "visual archetype" paradigm is very promising. Do the authors believe this approach could be generalized to other domains beyond affective computing, where VLMs also struggle with prompt-brittleness, such as fine-grained visual categorization (FGVC) or other visual regression tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents AURA (Affective Understanding via Robust Archetypes), a visually interpretable framework for emotion understanding built on a frozen CLIP visual space. Instead of relying on textual prompts, AURA models emotions through adaptive visual archetypes that self-organize and contextualize within the embedding space, enabling unified handling of facial expression recognition, action unit detection, and valence-arousal regression. The approach combines efficiency and interpretability by matching features to archetypes through cosine similarity during inference, achieving competitive or superior results on six benchmarks with lower computational cost.

### Strengths
1. The paper proposes a clear and well-motivated framework that unifies multiple affective understanding tasks (FER, AU, VA) within a single, interpretable visual archetype space.
2. AURA achieves strong empirical performance across six benchmarks while maintaining low computational cost and providing visual interpretability through archetype-based reasoning.

### Weaknesses
The maximum number of archetypes ($K_{\max}$) is fixed (e.g., 98) without any accompanying sensitivity or stability analysis. This omission leaves uncertainty regarding how different choices of $K_{\max}$ affect model performance, convergence behavior, and the balance between representation granularity and computational efficiency.

### Questions
Since the paper is developed within the CLIP-based visual space, it would be interesting to further explore how AURA might behave when paired with a non-CLIP encoder. Such an investigation could offer additional insight into whether the framework’s strengths arise primarily from the archetype design or from the representational characteristics of CLIP features.

### Soundness
4

### Presentation
4

### Contribution
4
