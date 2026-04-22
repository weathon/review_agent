# Beyond the Linear Separability Ceiling: Aligning Representations in VLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
A challenge in advancing Visual-Language Models (VLMs) is determining whether their failures on abstract reasoning tasks, such as Bongard problems, stem from flawed perception or faulty top-down reasoning. To disentangle these factors, we introduce a diagnostic framework centered on the Linear Separability Ceiling (LSC), the performance achievable by a linear classifier on a VLM's raw visual embeddings. Applying this framework to state-of-the-art VLMs, we uncover a pervasive ``alignment gap'', where most models fail to generatively outperform the linear separability of their own representations. We find that the few models surpassing this ceiling do so via two mechanisms: by further refining visual representations into a more linearly separable format or by executing non-linear decision logic. We demonstrate that this bottleneck is not a fundamental limitation but a solvable visual alignment issue. By augmenting standard next-token prediction with a contrastive objective, our method restructures the visual manifold into a more one-dimensionally linear geometry, improving image-to-image comparison and enabling models to significantly surpass the LSC on abstract binary classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper investigates why VLMs fail on abstract visual reasoning tasks such as the Bongard problems. The paper proposes the Linear Separability Ceiling (LSC), a diagnostic measure of how well a simple linear classifier performs on a VLM’s visual embeddings. It serves as a baseline to test whether a model’s reasoning pipeline adds non-linear reasoning beyond what its visual features already provide.
The authors evaluate several vLMS (e.g. Gemma, Pixtral, Phi, Qwen, InternVL) and realize that most models fail to exceed their own LSC, implying that their reasoning components are not effectively leveraging visual representations.
They also identify two strategies by which the VLMs can surpass the LSC
1) Enhanced linear separability: improving internal representations to be more discriminative
2) Non-linear decision logic: leveraging deeper reasoning pathways beyond linear readouts 

The paper also proposes a finetuning method that combines next-token prediction and a contrastive loss to improve representation alignment. They show that this object allows models to surpass LSC and improve in-domain and cross-domain reasoning.

### Strengths
1) The concept of an “alignment gap” between perception and reasoning reframes common VLM failures through a geometric and statistical lens
2) The paper’s decomposition of reasoning into linear vs. non-linear computational mechanisms (perceptual refinement vs. reasoning) provides conceptual clarity that could generalize beyond VLMs

### Weaknesses
1) The experimental validation is limited to binary image-to-text retrieval variants of Bongard-style tasks (Bongard OpenWorld and Bongard HOI), where the model must choose between two options given a query image. While these are well-defined tests for abstract reasoning, they represent a highly specific and simplified evaluation setup. It remains unclear whether the proposed framework and fine-tuning method would generalize to broader vision-language reasoning tasks such as open-ended visual question answering, caption generation. The improvements demonstrated may partially stem from optimizing for the same binary discriminative signal introduced by the contrastive objective, rather than from a genuinely general enhancement of multimodal reasoning.
2) Because the proposed contrastive objective directly optimizes for improved separability between positive and negative examples, part of the observed performance gain could be attributed to alignment with the evaluation metric itself (linear separability), rather than improved reasoning. Additional experiments on independent tasks not directly linked to the contrastive loss would help verify generalization.
3) The paper notes that the combined objective (L_combined) can induce catastrophic forgetting and prompt-format overfitting. It is unclear whether the performance of the finetuned model on general vision-language benchmarks (e.g. VQA benchmarks) is preserved after finetuning.
Furthermore, the paper provides little quantitative analysis of how the weighting between the next-token prediction and contrastive loss terms affects this trade-off. A sensitivity study or learning dynamics analysis would strengthen the claims.

### Questions
1) The paper mentions LSC could be used as a “live diagnostic” during model training. Could the authors outline how such an online LSC metric might be integrated into training pipelines (e.g., as a stopping criterion or auxiliary signal)?
2) How sensitive are the results to the relative weights w_m and w_c in L_combined? Does a small contrastive component already improve alignment, or is a strong contrastive signal necessary?
3) Have the authors evaluated whether the LSC framework predicts reasoning performance on tasks that are less abstract (e.g., visual question answering or commonsense reasoning)? How consistent is the alignment gap in these settings?

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
This paper investigates failures of VLMs on abstract visual reasoning tasks like Bongard problems, questioning whether the bottleneck lies in visual perception or higher-level reasoning. The authors introduce a diagnostic framework centered on the Linear Separability Ceiling (LSC): the maximum performance achievable by a linear classifier on the VLM's initial visual embeddings. Applying this framework, they discover an alignment gap: most state-of-the-art VLMs fail to generatively outperform their own LSC, suggesting their reasoning capabilities are poorly aligned with their visual representations. They propose a fine-tuning method using LoRA with a combined objective, adding a contrastive loss to the standard next-token prediction loss. This method improves the linear separability of final embeddings, successfully allowing models to systematically surpass the LSC and achieve higher performance on abstract reasoning tasks.

### Strengths
1. The graphs and tables are clear and easy to understand.
2. Experiments are thorough, covering multiple VLMs, datasets, PEFT methods, objectives, and generalization scenarios.

### Weaknesses
1. While effective, the LSC relies solely on linear separability. It's possible that representations hold complex non-linear structures useful for reasoning that the LSC metric fails to capture.
2. The core observation that VLM generative performance often fails to surpass a linear probe on its visual features is not entirely new. Similar gaps between representation quality and end-to-end performance have been previously studied, showing VLMs can underperform linear probes on classification or generally overlook information in their visual representations.
Some related works like:
[Why are Visually-Grounded Language Models Bad at Image Classification?](https://arxiv.org/pdf/2405.18415.pdf)
[Hidden in plain sight: VLMs overlook their visual representations](https://arxiv.org/pdf/2506.08008.pdf)

### Questions
1. Can the fine-tuning approach be successfully applied to improve VLM performance on other challenging reasoning domains beyond Bongard problems, such as VQA or complex instruction following?

### Soundness
3

### Presentation
3

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
This paper investigates why Visual–Language Models (VLMs) often fail on abstract visual reasoning tasks and whether such failures arise from perception or reasoning deficits. To diagnose this, the authors propose the Linear Separability Ceiling (LSC) framework, which quantifies the performance achievable by a linear probe on a model’s visual embeddings. Analyses across state-of-the-art VLMs reveal a pervasive alignment gap, where models rarely exceed their own LSC. To address this, the study augments next-token prediction with a contrastive objective that enhances the linear structure of visual representations. Fine-tuned models consistently surpass the LSC, achieving human-level accuracy on Bongard-OpenWorld and significant gains on relational reasoning benchmarks, demonstrating that reasoning limitations stem from misalignment rather than intrinsic capacity.

### Strengths
1)	The paper introduces the Linear Separability Ceiling framework to disentangle perception and reasoning in VLMs.
2)	Through large-scale analysis, the authors reveal a pervasive alignment gap: most VLMs fail to outperform their own LSC, highlighting a fundamental but previously unmeasured bottleneck in vision–language reasoning.
3)	The approach attains or surpasses human-level accuracy on OpenWorld and narrows the gap on HOI reasoning, demonstrating that the limitation in current VLMs stems from misalignment, not innate capacity.

### Weaknesses
1)	While the Linear Separability Ceiling is intuitively defined, the paper lacks a rigorous theoretical justification for why linear separability should represent the upper bound of perceptual quality. A more formal link between LSC and model capacity or information-theoretic limits is missing.
2)	The claim that failures arise from “alignment gaps” rather than perception deficits is mostly correlational. The experiments show association but not causal evidence that reasoning misalignment causes underperformance.
3)	The evaluation focuses mainly on Bongard-style reasoning and a single compositional benchmark. Broader validation on diverse abstract reasoning or real-world multimodal tasks would strengthen the generality of conclusions.
4)	The “nonlinear decision logic” mechanism is described conceptually but not visualized or quantitatively analyzed. Without feature attribution or attention-map evidence, the interpretation remains speculative.
5)	This paper is lengthy and conceptually dense. Core ideas like “alignment gap” and “surpassing the ceiling” could be more precisely illustrated. Figures (e.g., Fig. 2) lack clear axis descriptions, and some tables overflow with statistical detail without clear takeaway messages.
6)	The paper’s structure is somewhat diffuse, with diagnostic and intervention sections interleaving and key concepts repeated across sections. This weakens the logical flow from problem to solution and makes the main argument harder to follow.

### Questions
Please refer to the weak points.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the LSC to diagnose whether VLMs fail on abstract reasoning tasks due to perception or reasoning. The authors identify an alignment gap where reasoning fails to surpass the model’s own representational limit. They further proposed a contrastive fine-tuning objective that can close this gap on Bongard-style reasoning tasks.

Overall, the paper is thoughtfully motivated but overinterprets linear-probe diagnostics as mechanistic evidence for reasoning non-linearity. The statistical evidence is somewhat weak, and the causal link between representational geometry and reasoning behavior remains speculative. I will consider moving up the score, depending on how the authors clarify statistical robustness and theoretical grounding during discussion.

### Strengths
1. The paper is well motivated and identifies an important problem: diagnosing perceptual–reasoning misalignment in VLMs.
2. The proposed LSC is a clear, interpretable metric that operationalizes representational quality in a meaningful way.
3. The paper is well written and organized, with 2 dataset on 8 models coupled with various promptings.
4. The paper introduced a contrastive fine-tuning objective that simultaneously improves generative accuracy and final-layer separability.

### Weaknesses
1. The paper defines “non-linearity” as “cases where linear probes fail.”  The non-linearity can be an artifact of your measurement of cosine similarity of euclidean averaged embeddings, not a measured representational property. The claim would be stronger with direct evidence of curvature or manifold structure.
2. Important evaluation details are underspecified—for instance, how generative accuracy is computed relative to the probe-based classification accuracy. 
3. Some of the results (e.g., Section 7.2) appear cherry-picked without consistent statistical treatment. A group-level comparison across models or prompt conditions would increase confidence in the claims.
4. The causal interpretation—that contrastive fine-tuning resolves misalignment—remains speculative. The improvements may simply reflect more linearly organized feature geometry rather than deeper mechanistic reasoning.
5. The paper notes catastrophic forgetting as a limitation but does not fully analyze why L_{combined} causes this — an area that could benefit from ablation or regularization experiments.

### Questions
1. How sensitive are your results to pooling strategy (mean pooling vs. attention pooling vs. CLS token)?
2. How stable is the LSC metric across different random seeds or mini-batch samplings?
3. How exactly is the statistical comparison between generative accuracy and LSC performed in Fig. 2—are these paired comparisons over test trials or aggregated accuracies? For models that pass LSC, the advantage of generative accuracy is low.
4. Could this framework be applied to tasks requiring fine-grained perceptual reasoning (e.g., gaze direction or social interaction), rather than Bongard tasks that suit better for linear separation?

### Soundness
2

### Presentation
3

### Contribution
3
