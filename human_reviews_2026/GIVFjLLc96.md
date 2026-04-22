# Model Merging with Functional Dual Anchors

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Model merging is an efficient post-training strategy for integrating knowledge from multiple finetuned checkpoints of a shared foundation model. Existing methods operate in the parameter space, combining task vectors to mitigate conflicts, but remain constrained by parameter inconsistencies. We propose Functional Dual Anchors (FDAs), a framework that instead models the input–representation space. FDAs are synthetic inputs whose induced gradients align with task vectors, capturing task-specific functional shifts relative to the pretrained model. This perspective bridges joint multi-task training and post-hoc merging, offering both robustness and flexibility. We further introduce a principled initialization scheme and show that FDAs are complementary to parameter-centric model merging. Comprehensive experiments demonstrate the effectiveness of FDAs in model merging.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a method called Functional Data Alignment (FDA) for model merging.
Instead of directly averaging or interpolating model parameters, the authors first find a set of representative input features for each layer and align the models so that their outputs remain unchanged on these features.
The intuition is that aligning model behavior in function space allows smoother merging and better preservation of learned knowledge.

### Strengths
Performing model merging via function alignment rather than parameter interpolation is conceptually interesting.
This paper is well-structured and easy to follow.

### Weaknesses
The FDA construction process relies on gradient descent, but it is unclear whether these gradients are computed per layer or over the entire network. If gradients are restricted to individual layers, the global inter-layer dependencies may be ignored. If gradients are computed for the full network, the computational cost would become prohibitively large, especially for deep models. The authors should clarify this trade-off and the precise optimization scope.

The convergence of FDA optimization is not well analyzed.
Since the gradient descent outcome depends heavily on random initialization, the authors should provide:
Results over multiple random seeds,
Mean and standard deviation of performance,
A short convergence analysis to justify robustness.
Without this, the reproducibility and reliability of FDA are questionable.

It is unclear whether the same optimization setup (learning rate, steps, loss weights) is applied uniformly across layers. Given that shallow and deep layers encode very different semantic information, a uniform optimization configuration may be inappropriate. The paper should discuss whether the FDA optimization strategy varies by layer or is shared globally, and why.

The FDA process requires layer-wise gradient-based optimization, which could be expensive for deep models such as Qwen3 or other large-scale transformers.
The paper should report or estimate. The computational time and memory cost of FDA, such as a simple simulation of time or memory cost on a large model (if you can not find fine-tuned checkpoint during rebuttal). Without such analysis, it is difficult to assess the practicality and scalability of the method.

The proposed method shows small gains in CV tasks but large improvements in NLP tasks. This discrepancy lacks explanation.

The story of merging in function space in this paper is good. However, it may be verbose and occasionally for some readers not faimilar with model merging. I think 'finding a set of representative input features for each layer and aligning the models so that their outputs remain unchanged on these features' may be a better way for understanding.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper propose a framework named Functional Dual Anchors (FDAs) to optimize synthetic inputs whose induced gradients align with task vectors, capturing task-specific functional shifts relative to the pretrained model. In addition, the authors introduce a principled initialization scheme and show that FDAs are complementary to parameter-centric model merging. Comprehensive experiments show the effectiveness of FDAs.

### Strengths
This paper provide a new way to understand the merging process. By reinterpreting task vectors as gradients induced by synthetic inputs, FDAs bridge the gap between multi-task learning and post-hoc model merging, offering a new functional perspective for knowledge consolidation. The discussions are sound and insightful.

### Weaknesses
1. In lines 55-56, the authors claimed that 'we shift the merging process into the input space, where representations can naturally capture task-specific variations.' Why does the 'merging process in the input space' can 'naturally' capture task specific variantions?

2. The whole pipeline seems to be computationally intensive. What's the exact learning time? Compared to baselines, will the performance increase be enough to offset the increase in computing complexity?

3. In line 53, the authors claimed that 'FDAs capture the analogous knowledge in the input space through their induced gradients', which sounds novel, but why modeling knowledge in such a way? Is it necessary?

4. Are the learned anchors transferable? Or we need to learn such anchors model by model?

5. Will the number of models in a merging process influence the performance of your method? To what degree does it influence the performance?

6. Will this method still work well when the models in a merging process do not share a same starting point? i.e., when they are not tuned from a same model checkpoint.

7. The anchors are model level anchors (e.g., see your Algorithm 1, lines 221-227, for each model i you optimized several anchors), which means as the number of models increase, the learning time will also increase. In addition, how to make sure that there's no conflict between different anchors, and how to ensure the diversity of the optimized anchors?

### Questions
See weaknesses. I'm welling to increase the rating once my concerns are well addressed.

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
3

### Summary
This paper presents an input-representation space model-merging framework that projects task-specific knowledge from fine-tuned checkpoints into synthetic inputs. These inputs are optimized so that their induced gradients on the pretrained model align with task vectors (parameter deltas). Experiments on vision (CLIP ViTs), language (RoBERTa), and autoregressive (LLaMA-2) models show consistent improvements over state-of-the-art baselines, with ablations validating key design choices such as initialization schemes and distance functions. However, the paper lacks a clear motivation and a solid theoretical foundation for the proposed method.

### Strengths
1)New approach to merging: Proposes a method that projects task knowledge into the input–representation space rather than directly manipulating parameter vectors. This connects joint multi-task training (input-centric) with post-hoc weight averaging (parameter-centric), providing an alternative design perspective.
2)Theoretically grounded initialisation: Derives closed-form dynamics for a linear encoder and shows that tail eigen-energy of the task vector slows convergence. Two simple initialisation strategies (weight-row sampling and scaled Gaussian) fall out of the theory and yield the fastest loss decrease.
3)Comprehensive empirical validation: Evaluated across multiple vision datasets, NLP benchmarks, and large language models, covering both encoder and decoder architectures. FDA-enhanced TA consistently improves performance across these task.

### Weaknesses
1)The motivation is unclear, and the paper lacks a clear explanation of why input representations can be used to replace task vectors in model merging.
2)Limited theoretical justification beyond linear case: All convergence claims rest on a single-layer linear encoder (Sec. 2.2); no analysis for non-convex deep nets or layer interactions. No guarantee that gradient-aligned synthetic points transfer to real-data loss basins.
3)Missing statistical significance: Reported numbers are single-run means without variance estimates; it is impossible to judge whether +0.3% gains are systematic or noise (all tables). 
4)Optimisation steps vs. quality: Fig. 10 stops at 1200 steps without showing whether performance saturates or collapses later. 
5)Anchor shape: token-num ablation on RoBERTa shows drop beyond 5 tokens but explanation is deferred to “closer to real shape” hypothesis without measuring actual data token statistics (Sec. 5.2; Table 8).
6)Distance-metric sensitivity under-explored: Only cosine, L1, L2 are tested; manuscript does not explain why cosine is optimal or whether the choice interacts with architecture (Sec. 5.3; Fig. 9).

### Questions
1)Explain why inputs can replace task vectors?
2)Explain the meaning of equation (1). What does \phi(\theta,x) and Dist(.) in equation (1) mean?

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
3

### Summary
This paper introduces Functional Dual Anchors (FDAs), a novel framework for model merging that shifts the focus from the conventional parameter space to the input-representation space. The core idea is to construct a set of synthetic inputs (FDAs) for each downstream task, such that the gradients they induce on the pretrained model align with the corresponding task vector. This approach allows FDAs to encode task-specific knowledge in the input space, serving either as a standalone merging strategy or as a complementary refinement for existing parameter-centric methods.

### Strengths
1. The paper provides a novel and insightful perspective on model merging. 
2. The proposed method is well-motivated by a solid theoretical analysis.
3. The experimental validation is comprehensive and robust, convincingly demonstrating FDA's effectiveness.

### Weaknesses
1. The FDA construction process involves a nested optimization problem that requires computing second-order gradients, leading to a significant computational overhead. Although the layer-wise strategy makes it tractable, the method is inherently more expensive than one-shot approaches like TA or WUDI, and the paper lacks a quantitative analysis of this extra cost, which raises concerns about its practical utility.
2. While FDAs show the ability to enhance existing methods, the performance gains on strong state-of-the-art baselines like WUDI are sometimes marginal.
3. The framework introduces a considerable number of new hyperparameters, including the number of anchors, token numbers, the scaling coefficient for initialization, optimization steps, and the choice of distance function. This complexity can make the method more difficult to tune and apply in practice compared to simpler merging algorithms.

### Questions
1. Could you provide a quantitative comparison of the computational overhead? For instance, what is the total wall-clock time or the number of FLOPs required to merge a set of models using your method (including both construction and optimization stages) compared to baselines like TA, TSV, and WUDI under the same hardware setup?
2. Have you attempted to visualize the constructed FDAs for vision tasks (e.g., the input to a ViT)? It would be highly insightful to see whether they resemble noisy natural images, abstract textures, or something else entirely, as this could provide a more intuitive understanding of the knowledge they capture.
3. It is noted that the performance gains on strong baselines like WUDI can be marginal. This raises a practical question regarding the tuning effort: was this small improvement achieved using a default set of hyperparameters, or did it necessitate extensive, baseline-specific tuning? More broadly, how would you advise a practitioner to reasonably select hyperparameters like the number of anchors (n) and token_num in a real-world scenario, given that the tuning process itself is computationally expensive?

### Soundness
4

### Presentation
3

### Contribution
3
