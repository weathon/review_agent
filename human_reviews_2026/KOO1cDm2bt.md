# Tug-of-War No More: Harmonizing Accuracy and Robustness in Vision-Language Models via Stability-Aware Task Vector Merging

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
Foundation Vision-Language Models (VLMs) excel across benchmarks yet remain vulnerable to adversarial attacks. While adversarial fine-tuning improves robustness, attaining a desirable clean–robust performance trade-off typically requires costly hyperparameter searches with multiple retraining runs. A promising alternative is to merge task vectors (i.e., parameter displacements from pre-trained models) to balance accuracy and robustness without retraining. However, we find that naive task-vector merging produces a near-linear trade-off, as it equally weights all coordinates and fails to distinguish weights that aid both objectives from those that create conflicts. To overcome this limitation, we propose a prediction stability-aware merging framework that composes task vectors from off-the-shelf naturally and robustly fine-tuned VLMs. Our key insight is that prediction stability serves as a proxy for cross-objective compatibility, enabling us to favor perturbation-invariant parameters while attenuating those with high cross-objective impact. Specifically, we estimate per-parameter stability from gradients under both objectives, building complementary masks that retain jointly stable coordinates while suppressing counterpart-sensitive ones. We further refine these masks along adversarial parameter trajectories, with steps weighted by a prediction-sensitivity index.  Our theoretical analysis shows that the masks provably contract first-order cross-objective interference, and the prediction criticality index tracks curvature, biasing the merge toward flatter minima and better generalization. Extensive experiments across benchmarks and scenarios demonstrate our method consistently achieves superior clean–robust trade-offs over prior approaches, with the learned balance transferring effectively to downstream tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper investigates a training-free way to merge a naturally fine-tuned CLIP and a robustly (adversarially) fine-tuned CLIP using gradient-informed and prediction-stability masks over task vectors. Instead of uniformly adding task vectors (which they show yields an almost linear clean-vs-robust trade-off), the authors (1) estimate per-parameter stability from gradients of the *other* objective, (2) cap unstable coordinates, and (3) refine these masks along *adversarial parameter trajectories* weighted by a Prediction Criticality Index (PCI) that tracks curvature (Hessian trace). Empirically, PISTOLE bends the frontier and improves the sum of clean+robust performance across 14 datasets. Theoretically, the authors show the masks contract cross-objective first-order interference.

### Strengths
1. This paper provides an interesting and clear problem formulation and empirical evidence, where vanilla task vector merging fails to improve the trade-off, yet the proposed method effectively addresses this issue.
2. The paper shows clear and logical writing along with a novel methodology.
3. To the best of my knowledge, this is the first paper to address the accuracy-robustness trade-off without costly adversarial fine-tuning, generalizing across tasks.
4. The theoretical justification matches intuition: masks contract first-order interference, while PCI tracks curvature.
5. The strong ablations and analyses demonstrate the SOTA performance of the proposed method associated with improved efficiency.

### Weaknesses
1. The proposed method requires access to both natural and robust fine-tuned models. Does it mean that the quality also depends on them?
2. The “Prediction Criticality Index” is well-motivated mathematically, but the intuition appears late (after Eq. 10). An earlier conceptual explanation could help readers.
3. Recent VLM robustness techniques (e.g., prompt ensembling, token-level defenses) should be discussed in the Related Works.

### Questions
1. Could the authors briefly clarify the meaning of kappa in Eq. (6)? It would help if the notation were reintroduced when referenced later in the paper.
2. The authors introduce the term Prediction Criticality Index (PCI) in Section 3.4, but the intuition only becomes clear later. Could you add a short intuitive explanation when it’s first mentioned?
3. Sections 3.3 and 3.4 are mathematically dense. If possible, could you add a brief ‘takeaway’ sentence at the end of each subsection summarizing the main intuition?”

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PISTOLE (Prediction STability-aware mOdeL mErging), a training-free framework designed to reconcile the long-standing accuracy–robustness trade-off in CLIP.
Instead of costly adversarial fine-tuning, PISTOLE merges two off-the-shelf models—one naturally fine-tuned and one adversarially fine-tuned—directly in parameter space using task vectors.

To avoid the near-linear degradation observed in naïve merging, the authors propose gradient-informed complementary masks and a prediction criticality index (PCI) that approximates curvature and identifies stable parameters for selective fusion.
Theoretically, PISTOLE is shown to reduce the conflict between natural and robust objectives and guide the merged model toward flatter, more stable parameter regions, thereby improving both accuracy and robustness; empirically, it achieves superior clean–robust trade-offs and transfers effectively to downstream tasks such as captioning, VQA, hallucination detection, and reasoning.

### Strengths
1.**Motivation is Clear:** The paper is well-motivated, shifting the focus from model retraining to direct parameter merging, which effectively leverages pretrained foundation models to achieve robustness–accuracy balance with minimal cost.

2.**Method is reasonable:** The approach is conceptually sound, integrating gradient-informed and curvature-aware weighting into the model merging framework, and theoretically demonstrating that (i) complementary masks provably contract cross-objective interference, and (ii) PCI quantitatively approximates curvature through the Hessian trace.

3.**Experiment proves the claim:** Experiments are comprehensive, covering zero-shot classification, captioning, VQA, hallucination, and CoT reasoning, and consistently showing improvements across both clean and robust performance metrics.

### Weaknesses
1. PISTOLE assumes access to both naturally and adversarially fine-tuned models, which are themselves costly to obtain. Thus, the “training-free” claim is conditional—effective only once such models are available—and may not reduce the overall computational cost in practice.

2. The paper lacks analysis on failure cases. While quantitative results are strong, does this method works all the time with all the cases? It remains unclear under what conditions the proposed method may fail or underperform.

3. The experimental section could be improved by including more comparisons and discussions with similar task-vector-based methods to better contextualize the contribution.

### Questions
In your work, PISTOLE assumes that task vectors can be linearly combined in parameter space 
(e.g., $\boldsymbol{\tau}_{\text{add}} = \sum_i \boldsymbol{\tau}_i$). 
However, in multimodal encoders, task-specific update directions may not be linearly independent, 
and direct addition could introduce interference due to the nonlinearity of the representation space.  
Could you clarify on that?

Figure 5 shows that PISTOLE reduces loss curvature during model merging.  
Could you clarify how this empirical result quantitatively supports Theorem1 and Theorem2?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes PISTOLE, a training-free way to merge a naturally fine-tuned CLIP and an adversarially fine-tuned CLIP by masking task vectors with stability cues. Instead of adding task vectors uniformly (which yields a nearly linear clean vs. robust balance), the proposed method builds complementary masks so each branch down-weights coordinates that the other loss wants to change. Furthermore, the introduced method refines those masks by accumulating gradients along adversarial weight trajectories, where each step is weighted to capture high-curvature regions. The theory proves first-order interference contraction and connects PCI to the Hessian trace. Experiments on 14 datasets plus multiple backbones and PEFT show improved performance and better downstream transfer (captioning, VQA, hallucination, and CoT).

### Strengths
1. The paper is well-written and organized. The figures are illustrative and intuitive.
2. The proposed idea is well-motivated by the systematic analyses of the linear trade-off between clean and robust accuracy.
3. The theoretical analysis in this paper provides additional justification of the method's effectiveness. For example, it links the prediction criticality index to the model's curvature.
4. Component analyses and comparisons with prior approaches prove the efficacy of PISTOLE. This method can also be transferred to diverse downstream tasks in a plug-and-play way.

### Weaknesses
1. It appears that the per-layer normalization and \kappa sharpening are important. Although the paper has provided sensitivity analyses, additional evidence that the method is stable across diverse scenarios should be provided.
2. Most experiments focus on the vision encoder. It would be better to discuss further regarding the task vector for the text encoder to give more insights for future researchers.
3. The discussion regarding model interpolation and soups beyond task vectors is missing. They are also closely related to this paper.
4. (Minor) The figure readability needs to be improved. For example, I suggest increasing font sizes in figures.

### Questions
1. Does the text-side parameter merging benefit from the proposed merging method? The authors can briefly validate whether applying PISTOLE to the text encoder yields additional gains or harms alignment. 
2. When referencing Eq. (6) later, can you briefly remind readers what \kappa controls (sharpening vs. sparsity)?
3. Is mask capping (q-quantile) applied globally or per-layer? More details should be provided.

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
4

### Summary
This paper proposes PISTOLE, a novel prediction stability-aware model merging framework designed to reconcile the trade-off between clean accuracy and adversarial robustness in vision-language models (VLMs). Instead of retraining, the method merges task vectors from naturally and adversarially fine-tuned models using gradient-informed masks and adversarial parameter trajectories. The approach leverages prediction sensitivity and curvature estimates to selectively retain stable parameters and suppress conflicting ones. Theoretical analysis and extensive experiments demonstrate improved clean-robust trade-offs and transferability to downstream tasks.

### Strengths
1. The idea of merging task vectors from conflicting objectives (natural vs. robust) is novel and well-motivated.

2. The paper provides clear mathematical reasoning (Theorem 1–2) showing that the complementary masks contract cross-objective interference and PCI correlates with curvature — adding interpretability and credibility.

3. Extensive experiments across datasets, backbones (ViT-B/L/H), and downstream tasks (captioning, VQA, hallucination, reasoning) demonstrate consistent improvement and generality.

### Weaknesses
1. The role of PCI versus GISM (gradient-informed stability mask) feels somewhat overlapping; clarifying their unique contributions would help.

2. Although the method claims to avoid retraining, the process of computing path-integrated gradients and PCI-based weighting still requires multiple forward–backward passes. A brief complexity analysis would strengthen the practicality argument.

### Questions
Clarify in the introduction whether the merging operates solely on the vision encoder or also on the text encoder.

### Soundness
3

### Presentation
4

### Contribution
4
