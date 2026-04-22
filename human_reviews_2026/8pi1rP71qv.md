# FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning

- Avg Score: 5.60
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4, 10

## Abstract
General continual learning (GCL) challenges intelligent systems to learn from single-pass, non-stationary data streams without clear task boundaries. While recent advances in continual parameter-efficient tuning (PET) of pretrained models show promise, they typically rely on multiple training epochs and explicit task cues, limiting their effectiveness in GCL scenarios. Moreover, existing methods often lack targeted design and fail to address two fundamental challenges in continual PET: how to allocate expert parameters to evolving data distributions, and how to improve their representational capacity under limited supervision. Inspired by the fruit fly's hierarchical memory system characterized by sparse expansion and modular ensembles, we propose FlyPrompt, a brain-inspired framework that decomposes GCL into two subproblems: expert routing and expert competence improvement. FlyPrompt introduces a randomly expanded analytic router for instance-level expert activation and a temporal ensemble of output heads to dynamically adapt decision boundaries over time. Extensive theoretical and empirical evaluations demonstrate FlyPrompt's superior performance, achieving up to 11.23%, 12.43%, and 7.62% gains over state-of-the-art baselines on CIFAR-100, ImageNet-R, and CUB-200, respectively. Our source code is available at https://github.com/AnAppleCore/FlyGCL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FlyPrompt, an expert-based framework for General Continual Learning (GCL), where each task is associated with a prompt expert. FlyPrompt proposes two key contributions: a novel strategy named REAR (Random Expanded Analytic Router), which leverages random projection to identify suitable experts at inference, and Task-wise Experts with Temporal Ensemble, designed to track distributional drift. The paper is supported by detailed experiments across standard benchmarks and thoughtful ablation analyses.

### Strengths
- The decomposition into expert routing and expert competence is empirically well-supported.
- The results are consistent across a broad range of pre-trained models (Sup-21K, iBOT, DINO, MoCo), rather than being limited to a single backbone.
- The paper is further strengthened by extensive ablation studies and a comprehensive hyperparameter analysis.
- The writing is clear, and the overall structure is well-organized.

### Weaknesses
- The paper does not discuss the number of parameters (different from memory in Appendix F) of the proposed method compared to the baselines, which could be a significant concern.
- In the GCL setting, task boundaries are unknown during both **training** and inference [1]. The methodology described in L216–217, which states that "we associate each task $t$ with a corresponding expert $E_t$," appears to assume that task identities are known during training, thereby allowing the assignment of a new expert. This assumption fundamentally contradicts the definition of GCL. Consequently, the problem being addressed may be closer to Class-Incremental Learning with overlapping classes rather than true task-agnostic GCL.
- If task boundaries are indeed assumed to be known during training, then several recent SOTA methods like HiDe-Prompt [2], NoRGa [3], and SD-LoRA [4] could be straightforwardly implemented. The paper lacks a comparison to these highly relevant methods, which weakens its claims of superiority.
- The REAR component, used for identifying task identity, is a powerful module in its own right and is functionally similar to the full RanPAC method [5]. Comparing FlyPrompt (which includes REAR) to methods that employ much simpler routing strategies may therefore be unfair. A more rigorous comparison would involve incorporating REAR into other methods (e.g., evaluating “HiDe-Prompt + REAR”) to properly ablate its contribution. A similar argument applies to the Task-wise Experts with Temporal Ensemble; this technique could also be combined with other baselines for a fairer assessment.
- In L184-186, the authors motivate their second component by claiming that "even with perfect routing, previous methods still exhibit inferior performance... highlighting... the limited competence of individual experts.". However, the paper does not sufficiently diagnose the source of this inferior performance. It remains unclear whether the issue lies in representation drift within the expert prompts $f_\theta(\cdot, p_t)$ or in catastrophic forgetting within the final classification head $g_\psi$. An experiment measuring the representation drift of each expert (e.g., by analyzing the similarity between representations from correct and incorrect experts) would be necessary to clarify and validate this motivation.
- The “Task Experts as Temporal Ensembles” component is presented as a mechanism to enhance expert competence. However, the core prompt expert is trained using a standard cross-entropy loss (Equation 6). Thus, the observed novelty and performance gains appear to stem from ensembling multiple classification heads rather than from improving the representational quality of the prompt expert itself.


[1] Dark Experience for General Continual Learning: a Strong, Simple Baseline, NeurIPS 2020

[2] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality, NeurIPS 2023

[3] Mixture of Experts Meets Prompt-Based Continual Learning, NeurIPS 2024

[4] SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class Incremental Learning, ICLR 2025

[5] RanPAC: Random Projections and Pre-trained Models for Continual Learning, NeurIPS 2023

### Questions
Please see the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets General Continual Learning (GCL), characterized by single-pass, non-stationary data streams without clear task boundaries. Identifying limitations in existing Parameter-Efficient Tuning (PET) methods, it decomposes GCL into expert routing and expert competence improvement. Inspired by the fruit fly's memory system, the paper proposes the FlyPrompt framework with two core components: (1) A Random Expanded Analytic Router (REAR) using fixed random projections and a closed-form solution for gradient-free, rapid input-to-expert (prompt) assignment. (2) Task-wise Experts with Temporal Ensemble (TE$^2$) employing multiple EMA heads with different decay rates within each expert to dynamically refine decision boundaries over time. FlyPrompt achieves strong performance across various GCL benchmarks.

### Strengths
[S1] The proposed Random Expanded Analytic Router (REAR) uniquely employs a closed-form solution rather than iterative gradient updates to assign inputs to experts. This is advantageous for the strict online, single-pass constraints of GCL, offering a theoretically grounded and computationally efficient alternative to traditional routing methods.

[S2] The framework effectively tackles the complexity of GCL by decomposing it into two manageable subproblems: routing and competence improvement. The Task-wise Experts with Temporal Ensemble (TE²) addresses the latter by leveraging multi-timescale EMA heads, which significantly enhances expert robustness against non-stationary data streams.

### Weaknesses
[W1] Initializing the prompt for a new task by averaging previous prompts may not be beneficial and is likely to show degraded performance when subsequent tasks come from significantly different domains.

[W2] While REAR outperforms gradient-based routers, comparisons against simpler non-learning baselines in the expanded feature space (e.g., k-NN routing) are absent, making it hard to gauge the benefit derived from the analytic ridge regression complexity.

[W3] Experiments primarily use the default Si-Blurry configuration. Performance under more extreme imbalance, higher task overlap, or different types of distribution drift needs further investigation.

[W4] While neuro-inspired, drawing direct equivalences between specific algorithmic components (e.g., EMA heads and KC subtypes) and biological counterparts might be an oversimplification.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes FlyPrompt, a framework for GCL inspired by the neural circuitry of the fruit fly mushroom body. It decomposes GCL into two subproblems: expert routing and expert competence improvement. For expert routing, the authors introduce the REAR, which uses fixed random projections and closed-form updates for feedforward expert selection. For competence improvement, they propose TE2, which integrates knowledge across multiple timescales via EMA heads with different decay rates. The method reports SOTA performance on benchmarks.

### Strengths
1. The separation of GCL into routing and competence subproblems offers a structured approach to tackling its challenges.
2. The use of principles from fruit fly olfactory memory introduces a novel interdisciplinary perspective to CL.
3. The paper provides both informal and formal theoretical bounds on routing error and EMA parameter error, enhancing methodological credibility.

### Weaknesses
1. It resemble an ad hoc combination of existing techniques. The proposed FlyPrompt framework appears to be largely a composition of well-established components rather than a fundamentally novel algorithm. Specifically, the REAR combines fixed random projection with ridge regression (a paradigm already explored in prior CL works and analytic class-incremental learning). Similarly, the TE2 employs EMA with multiple decay rates, a standard technique in online learning and model stabilization (e.g., SWA, temporal ensembling). 
2. While the paper provides a biologically inspired narrative grounded in the fruit fly’s mushroom body, the mapping between neurobiological mechanisms and algorithmic design remains largely metaphorical. The performance gains reported (e.g., +11.23% auc on CIFAR-100) are primarily attributable to the strong supervised pretraining (Sup-21K) and the inherent benefits of random feature expansion, rather than the proposed routing or ensemble mechanisms. The ablation study (Table 2) further reveals that a RanPAC-like baseline already achieves 82.17% auc. This suggests that FlyPrompt’s contribution is incremental engineering rather than a necessary or uniquely effective solution.
3.  While GCL as a research direction is valid, the specific formulation and assumptions in this work appear tailored to a controlled benchmark rather than a pressing real-world problem. The number of tasks $T$ (and thus the number of prompt experts) is assumed known and fixed a priori, which contradicts truly open-world or task-agnostic streaming environments. The evaluation protocol assumes access to task-level metadata during training (e.g., expert identity per session), which may not hold in fully unsupervised or user-generated data streams.

### Questions
How about the practical scalability and efficiency? Although Table 5 reports only marginal increases in total parameters and per-batch latency, it omits the auxiliary memory and compute burden of $G$ and $Q$

### Soundness
2

### Presentation
3

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
Existing parameter-efficient tuning methods struggle in General Continual Learning (GCL) because they cannot effectively allocate expert parameters or improve representations in single-pass, boundary-free data streams. Inspired by the fruit fly's brain, this paper proposes FlyPrompt, a framework that decomposes GCL into expert routing and expert competence improvement. FlyPrompt uses a random analytic router to activate experts and a temporal ensemble of output heads to adapt, significantly outperforming state-of-the-art baselines on key benchmarks like CIFAR-100 and ImageNet-R.

### Strengths
**Biologically Inspired Foundation**: The framework is grounded in the neurobiological principles of the fruit fly's brain, offering a novel approach to solving complex GCL challenges.

**Addresses Core GCL Problems**: It effectively tackles two fundamental challenges in GCL: "expert routing" (selecting the right parameters) and "expert competence improvement" (adapting to new data) under difficult, realistic constraints (single-pass data, no task boundaries).

**Novel and Efficient Components**: It introduces two key innovations:

 - A randomly expanded analytic router for non-iterative (fast and efficient) expert selection.

 - A temporal ensemble of expert heads to ensure the model robustly adapts to data changes over time.

**Proven Performance**: The method is backed by both strong theoretical analysis and excellent empirical results, demonstrating superior performance and scalability across multiple GCL benchmarks.

### Weaknesses
**Notational Clarity**: There appears to be a notational inconsistency in Equations 2 and 3, where the symbols $\Phi$ and $\varphi$ seem to be transposed or confused.

**Comparative Analysis**: The paper would be significantly strengthened by a direct comparison of FlyPrompt against other prominent methods (such as LoRA, Adapters, and MoE). This comparison should explicitly analyze key metrics:

- Parameter efficiency (total and new parameters)

- Computational overhead (training and inference time)

- Backward Transfer (BWT)

**Novelty of Application**: Given that Random Projection is a well-established technique, what is the specific novelty of its application within the FlyPrompt framework? How does its integration into the "randomly expanded analytic router" differ from standard implementations and what unique advantages does this specific application provide for the GCL setting?

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
the paper introduces a neuro-inspired framework for General Continual Learning that learns online without task labels or replay. It breaks the problem into two parts: routing and expert competence. The Random-Expanded Analytic Router uses random feature projections and closed form ridge updates to select experts efficiently -- no  backpropagation. The Temporal-Ensemble Experts module maintains several EMA classifier heads with different decay rates and it is combining them to balance plasticity and stability. Theoretically REAR is shown to approximate batch ridge regression, and TEE achieves an almost optimal bias–variance trade-off. Experiments on CIFAR-100, ImageNet-R, and CUB-200 under the Si-Blurry protocol demonstrate consistent state-of-the-art results with minimal trainable parameters and runtime overhead.

### Strengths
1. The paper identifies two core challenges in GCL a) expert routing and b)expert stability. it addresses each with a distinct, principled component: REAR and TEE.
2. REAR does not do backpropagation. Instead it maintains online sufficient statistics and is solving a closed-form ridge regression, which is much faster/simpler.
3. the authors show consistent state-of-the-art results on CIFAR-100, ImageNet-R, and CUB-200 under the Si-Blurry protocol.
4. I like the math analysis linking random-feature expansion to generalization (thm 1) and characterizing the bias–variance trade-off in temporal ensembling (thm 2)
5. Less than a million trainable parameters 
6. I very much like the analogy to multi-timescale synaptic adaptation and biologically plausible interpretation of the design

### Weaknesses
1) Thm 1 relies on a pairwise concentration lemma but omits a full matrix-concentration argument and a margin assumption needed to link regression risk to routing accuracy. This may be fixable in the rebuttal period. 

2)  Thm-2:  the derivation connecting EMA bias to temporal drift is approximate. The claim of “near-optimal adaptation” is not formally proven. Needs to be clarified

3) Despite the task-free framing that the paper emphasizes, in my opinion REAR initialization and label accumulation still assume known session starts and one-hot session indicators. True?

4) Maintaining and inverting the (M \times M) Gram matrix can be memory-intensive for large random expansions

5) The method’s logit mask may leak boundary information but this is not analyzed against Si-Blurry baselines such as MVP.

### Questions
1. plz clarify how REAR maintains the inverse of (G+ lambda I) online. Is inversion recomputed per batch or updated incrementally?

2. Is it that each sample contributes once to G and  Q or multiple times across the three online iterations per batch? If repeated, the estimator corresponds to weighted ridge regression - not the exact form proven in Lemma 3.

3. In Lemma 1 the jump from pairwise concentration to operator-norm bounds on \hat \Sigma - \Sigma is not rigorous. I think this is fixable though.  

4. The paper states a random-feature error rate O(\log N/M) but this is  is inconsistent with the lemma’s \tilde O(\sqrt (\log N/M)) bound, right? Plz correct or provide an argument for the stronger rate.

5. Thm-1 gives a routing-accuracy guarantee  but it does not make any margin assumption between expert scores. Introduce this assumption and carry the margin constant into the final bound.

6. Thm-2 makes the bias step explicit. But can you show formally that \sum_j \alpha^j \Delta_{t-j+1} \le L P_t --  and quantify the constant C_2?

7. the claim that a geometric EMA bank achieves near-optimal performance should be supported with a short covering-ratio argument showing the error factor in terms of grid spacing r

9. If there is time in the rebuttal phase, plz evaluate the effect of removing or randomizing the logit mask to confirm that improvements are not due to boundary information.

10. Again, if there is time, it would be good to compare REAR with RanPAC under identical settings to clarify the unique contribution of routing versus analytic classification

11. I would appreciate a list of Assumptions (data i.i.d., λ > 0, bounded feature norms, fixed number of experts) at the start of the theory section

### Soundness
4

### Presentation
4

### Contribution
4
