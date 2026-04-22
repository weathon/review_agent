# Exploring Cross-Modal Flows for Few-Shot  Learning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Aligning features from different modalities is one of the most fundamental challenges for cross-modal tasks. Although pre-trained vision-language models can achieve a general alignment between image and text, they often require parameter-efficient fine-tuning (PEFT) for further adjustment. Today’s PEFT methods (e.g., prompt tuning, LoRA-based, or adapter-based) always selectively fine-tune a subset of parameters, which can slightly adjust either visual or textual features, and avoid overfitting. In this paper, we are the first to highlight that all existing PEFT methods perform one-step adjustment and are insufficient for complex (or difficult) datasets, where features of different modalities are highly entangled. To this end, we propose the first model-agnostic multi-step adjustment approach by learning a cross-modal velocity field: Flow Matching Alignment (FMA). Specifically, to ensure the correspondence between categories during training, we first utilize a fixed coupling strategy. Then, we propose a noise augmentation strategy to alleviate the data scarcity issue. Finally, we design an early-stopping solver, which terminates the transformation process earlier, improving both efficiency and accuracy. Compared with one-step PEFT methods, FMA has the multi-step rectification ability to achieve more precise and robust alignment. Extensive results have shown that FMA can consistently yield significant performance gains across various benchmarks and backbones, especially on difficult datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a new framework called Flow Matching Alignment (FMA) to improve feature alignment between visual and textual modalities in few-shot learning. The authors observe that current parameter-efficient fine-tuning (PEFT) methods—such as prompt tuning, adapter tuning, and LoRA—perform only one-step feature adjustments, which are insufficient for complex datasets where image–text features are highly entangled. FMA leverages the multi-step rectification ability of flow matching by learning a cross-modal velocity field that iteratively transforms image features toward corresponding text embeddings, achieving finer alignment. To ensure stability and correctness, the method incorporates three key designs: coupling enforcement (to maintain class correspondence), noise augmentation (to mitigate data scarcity), and an early-stopping solver (to prevent over-transformation during inference). Experiments across 11 benchmarks and multiple backbones show that FMA consistently outperforms existing PEFT methods.

### Strengths
1. FMA introduces flow matching to few-shot learning. By formulating traditional PEFT methods as one-step updates, FMA enables more precise iterative alignment between visual and textual features. As argued by athe uthors, FMA better handles entangled multimodal distributions, especially in challenging datasets.
2. The framework is architecture-independent and can be integrated with various pre-trained vision-language models (e.g., CLIP, CoOp, LoRA) without altering their internal structures.

### Weaknesses
1. The multi-step flow matching process requires iterative training and inference, which increases computational cost compared to traditional one-step PEFT methods, potentially limiting scalability for large datasets or real-time applications.
2. The method relies on carefully chosen parameters such as the number of inference steps, step size, and noise schedule. Suboptimal tuning can lead to degraded performance or instability during alignment. Especially when flow matching originates from generative modeling, and its adaptation to supervised classification tasks lacks rigorous theoretical grounding, in terms of convergence and optimal stopping criteria.

### Questions
1. As discussed in the weakness part, is there any theoretical guarantee of convergence and stability of FMA? Given that flow matching originates from generative modeling, what are the theoretical conditions under which FMA ensures convergence to the correct class-aligned distribution in supervised learning settings?

### Soundness
3

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
The paper reframes PEFT as a one-step adjustment problem that fails on difficult datasets and proposes Flow Matching Alignment (FMA), which learns a velocity field to iteratively transport image features toward their ground-truth text features. It introduces coupling enforcement to preserve class correspondence, noise augmentation to combat data sparsity and manifold collapse, and an early-stopping solver (ESS) that classifies from intermediate states to avoid late-stage drift. FMA is plug-and-play across CLIP and multiple PEFT backbones and shows consistent gains on 11 benchmarks, especially on difficult datasets, with ablations supporting each component.

### Strengths
The diagnosis of one-step PEFT limitations is convincing; the method is simple, modular, and effective across backbones; experiments are comprehensive; and the early-stopping insight is well-supported by empirical phenomena.

### Weaknesses
Lack of formal guarantees for coupling assumptions, reliance on validation to set inference steps, missing comparisons with higher-order ODE solvers, coarse difficulty metric, and limited analysis of compute trade-offs and failure modes.

### Questions
1. Can ESS be made adaptive without validation tuning (e.g., stop on sufficient logit margin, small velocity norm, or diminishing logit gains)?
2. How is σ(xt) chosen in practice, and how sensitive are results to the noise magnitude/schedule? Any benefit from uncertainty- or density-adaptive noise?
3. Did you try higher-order or adaptive ODE solvers (Heun/RK) to reduce truncation error and improve margins at the same step budget?
4. Is fixed pairing too rigid for multi-modal classes? Would transporting toward multiple positive prototypes or a class subspace help?
5. Have you considered adding a discriminative loss on intermediate states (e.g., contrastive/margin) to align transport with classification throughout the trajectory?
6. Can you report detailed overhead (velocity net size, train/infer time, average ESS steps) and scaling with number of classes?

### Soundness
3

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
4

### Summary
This paper addresses the challenge of achieving precise cross-modal alignment in vision-language models (VLMs) for few-shot learning. The authors argue that existing parameter-efficient fine-tuning (PEFT) methods—such as prompt tuning, adapter-based, and LoRA-based approaches—perform only a "one-step" adjustment of features, which is insufficient for complex datasets where modalities are highly entangled. To overcome this limitation, the authors propose Flow Matching Alignment (FMA), a model-agnostic framework that leverages flow matching theory to enable multi-step feature transformation. FMA incorporates three key designs: coupling enforcement to preserve class correspondence, noise augmentation to mitigate data scarcity, and an early-stopping solver for efficient and accurate inference. Extensive experiments on 11 benchmarks show that FMA consistently improves performance, especially on challenging datasets, and integrates seamlessly with various backbones and PEFT methods.

### Strengths
1. Novel application of flow matching to cross-modal alignment in few-shot learning, moving beyond generative tasks.
2. Effective design choices (e.g., early-stopping solver, noise augmentation) that address practical challenges in training and inference.

### Weaknesses
1. No analysis of computational overhead or inference latency introduced by multi-step transformation.
2. Ablation studies do not explore the sensitivity of performance to hyperparameters like inference steps.
3. The early-stopping strategy uses a fixed step count rather than a sample-adaptive criterion, which may limit optimality.

### Questions
1. Could the author provide more detailed information across different datasets in section 4.2 GENERALIZATION ABILITY, where only average performance was given?
2. Was any exploration done into adaptive early-stopping criteria (e.g., based on feature discriminability) rather than a fixed stepsize?
3. How does FMA perform in cross-modal retrieval or other downstream tasks beyond classification, given its alignment-focused design?
4. Could the authors provide more intuition or theoretical insight into why coupling enforcement preserves class-level correspondence in high-dimensional feature spaces?

### Soundness
3

### Presentation
2

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
This paper proposes a method to improve alignment performance between modalities in cross-modal models.
It claims that existing methods fail to align well on challenging datasets because they attempt one-step alignment, 
and proposes a multi-step approach to align the embedding vectors of the two modalities.
Specifically, it performs flow matching to transform from images to the distribution of text embedding vectors.

### Strengths
- This paper is well-written and well-structured.
- Alignment of embedding vectors across multi modalities is an important research topic.
- Flow-matching alignment seems novel. However, its necessity is questionable, and it might be just a combination of new techniques.
- In experiments, the proposed method outperforms baselines on class classification tasks. However, as written in Weaknesses, it is unclear whether the evaluation is well-designed to confirm the claims.

### Weaknesses
- The motivation for multi-step adjustment is unclear. 
First, the definition of one-step adjustmentfor poor performance in existing methods is ambiguous.
For example, is the claim that PEFT's optimization objective function is inappropriate, or that optimization is insufficient due to difficult learning?
Figure 2 discusses PEFT's characteristics compared to LP, but the validity of using LP as a baseline for this discussion is unclear. 
It is unclear how this connects to the statement: “these methods try to adjust their general aligned multi-modal distribution towards the golden distribution by one rectification step.”

- The experimental setups are insufficiently described, resulting in a lack of reproducibility. 
For example, it states that velocity networks are learned, but I could not find a description of the specific structure of the velocity networks.
There is no definition of $\sigma^2(\cdot)$. There seems also no report of the number of steps M for the proposed method across each dataset.
In addition, there is no evaluation of statistical significance.

- The baseline varies depending on the evaluation. 
While Table 1 compares against 8 baselines, Table 2 has one baseline and Table 3 has five.
Specifically, the baseline compared in Table 2 is one of the weaker baselines among those appearing in Table 1.
Although there are practical limitations on the number of experiments, comparing against the strongest baseline yields more convincing results.

- The proposed method seems computationally expensive. 
It requires preparing velocity networks and performing multiple updates during inference (Algorithm 2).
How does the computational cost compare to the CLIP-Adapter with two linear layers? How does it compare to PEFT?
Since the performance improvement over CLIP-LoRA is only 0–2%, the heavy inference cost makes the proposed method less useful.

- Minor issues
- The space is filled, making it difficult to read.
 The absence of a single line of space before and after figures and tables, such as the caption for Figure 4, violates the template.

### Questions
- What is the definition of one-step adjustment? Does it mean that the objective function is set once or that the optimization is only one step? Are Fig. 1(b)-(d) optimal embeddings in some sense?

- In Fig. 2, isn't it a bit simplistic to conclude that PEFT is weak on challenging datasets based on LP?
Couldn't one also conclude that LP is strong on more challenging datasets?

- What happens if stronger methods are used as baselines in Table 2? Also, did you check the standard deviation of the results and their statistical significance?

- How about the comparison of computational cost?

### Soundness
2

### Presentation
3

### Contribution
2
