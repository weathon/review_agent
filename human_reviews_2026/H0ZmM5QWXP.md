# Faster-VPS: Accelerating Object-Level Interpretation of Multimodal Foundation Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Attribution is essential for interpreting object-level foundation models, yet existing methods struggle with the trade-off between efficiency and faithfulness. Gradient-based approaches are efficient but imprecise, while perturbation-based approaches achieve high fidelity at prohibitive cost. Visual Precision Search (VPS) represents the current state-of-the-art, but its greedy search requires a quadratic number of forward passes, severely limiting practicality. We introduce Faster-VPS, which replaces VPS’s greedy search with a novel Phase-Window (PhaseWin) algorithm. PhaseWin combines phased pruning, windowed fine-grained selection, and adaptive control mechanisms to approximate greedy attribution with near-linear complexity. Theoretically, Faster-VPS retains approximation guarantees under monotonous submodular conditions. Empirically, it achieves over 95\% of VPS’s faithfulness using only 20\% of the computational budget, and consistently outperforms all other attribution baselines on tasks such as object detection and visual grounding with Grounding DINO and Florence-2. Faster-VPS thus establishes a new state-of-the-art in efficient and faithful attribution.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on accelerate the previous method called Visual Precision Search (VPS), which is designed to explain results by foundation model-based object detectors, such as GroundingDINO and Florence-2. The accelerated VPS is called Faster-VPS, replacing VPS's greedy search with a proposed Phase-Window (PhaseWin) algorithm. PhaseWin combines phased pruning, windowed fine-grained selection, and adaptive control mechanisms. The paper provides theoretical analysis by drawing an analogy to theories related to submodular functions. It conducts experiments on COCO, LVIS, and RefCOCO datasets to validate the proposed method.

### Strengths
Below are notable strengths of this paper.
- The dedication to accelerating a good method is appreciated.
- The datasets and foundation models used in this paper are standard that follow previous works.

### Weaknesses
The main challenge in the paper is the readability issue, making it hard for me to fully understand the paper. This could cause me to misunderstand parts of the paper. Nevertheless, I list some notable weaknesses below.

- The paper is hard to follow as it extensively uses terminologies without definitions. For example, it is unclear what "faithfulness" and "Insertion AUC" mean in Figure 1. The paper does not define them before using them. Due to this, I cannot understand the terms and their relations, e.g., how "Insertion AUC" measures "faithfulness".

- Line 182 explains "for maximizing the ordered insertion-AUC objective" but it does not define "insertion-AUC objective" in the paper. 

- Figure 2 is confusing. For example, the right panel means to illustrate "Phasewin Internal Loop" but the diagram does not display loops.

- Line 075 mentions "a dynamic supervision policy" and Line 208 formally introduces "two policies". Are these policies designed manually or by learning?

- Line218 states "Greedy search is both a curse and a shackle in the development of submodular function maximization algorithms". Can authors clarify what "a curse" means? What "a shackle" means? Do authors mean that greedy search is an inferior method here?

- The theories in Section 3.4 are either from (Nemhauser et al., 1978; Fujishige, 2005) or based on them with an assumption that "the objective F is monotone submodular" (Line235). However, the paper does not justify whether the objective F is monotone submodular. Refer to Line 260 which mentions "the submodularity assumption".

- The paper lacks important details or explanations on the crucial modules. For example, Algorithm 1 uses two functions "PartitionCandidates" and "WindowSelection" but neither of them is sufficiently explained in terms of implementations. As the difference of this paper and the VPS paper is whether to use Algorithm 1 or greedy search, the reader cannot fully understand the technical contributions without the important details.

- The visual results are expected to discuss how the method can be used to explain failures of foundational detectors, beyond their success. To note, the VPS paper contains such. Moreover, the visual results are hard to read. For example, the texts in Figure 3 are too small to identify. The colors in this figure are also confusing.

### Questions
The reviewer asks the authors to address each point in weaknesses listed above and does not repeat them in this Questions box.

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
The authors propose an improvement to the VPS attribution method by introducing a new algorithm to improve the search space constraints of the original work. Their algorithm results in greatly improved runtime with little to no reduction in resulting attribution faithfulness.

### Strengths
There is a need for the algorithm presented, as the research community prefers real-time attributions that could be used to safeguard real model deployments. As such, this paper’s contribution is relevant to the area. 

The algorithm seems well crafted from a computer science perspective and it achieves a significant runtime percentage improvement. 

The quantitative evaluation is exhaustive.

### Weaknesses
The novelty of this paper is, in a sense, limited to the algorithm employed. This is both a minor and significant complaint because it takes a high-quality method and does make it more realistic for real-time deployment (which is valuable), but it does not reveal any new information about a model or interpretability as a whole. It cannot be denied that this is a less significant contribution overall than the baseline VPS work.  

Many figures and tables are too small or hard to read. The tables are overly aggressive in their use of text resizing. Heatmaps in Figure 3 are not presented in a very interpretable color. It is hard to differentiate regions. I recommend the use of the matplotlib “jet” cmap as used in multiple other papers. 

More of the paper should be spent on section 3.3. It is challenging to parse what all of the steps of the algorithm actually do. Figure 2 is not all that intuitive. The text helps, but the algorithm seems like it should be quite simple to explain and yet it feels convoluted. I am not confident that I could reproduce the algorithm from what is presented.

### Questions
There are obvious runtime improvements, but what is the time cost, in seconds, of this method? Is it approaching a speed that could be used in real-time deployments? 

My rating can be increased to a 6 if the question is addressed and the weaknesses receive a proper response. I do think there is value in making an optimization-based approach work in real time. However, I do not think I could rate this paper higher than a 6 due to my concerns for the novelty and overall contribution to how we think about interpretability.

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
This paper proposes Faster-VPS, an accelerated variant of Visual Precision Search (VPS) for object-level attribution in multimodal detectors (e.g., Grounding DINO, Florence-2). The key idea is the Phase-Window (PhaseWin) search, which alternates between (i) picking a strong “anchor” region, (ii) pruning candidates via adaptive thresholds, and (iii) doing windowed fine-grained selection with a dynamic early-exit rule and an annealed deferral strategy. This approximates greedy search with near-linear evaluation complexity in practice while keeping faithfulness close to VPS. Empirically, the proposed method retains ≥95% of VPS’s faithfulness using about 20% of the computation overhead across MS-COCO, RefCOCO, and LVIS.

### Strengths
1. The idea is technically sound and has theoretical proofs.
2. The empirical results demonstrate the effectiveness of the proposed method.
3. The paper is well organized and easy to follow.

### Weaknesses
1. The proposed method relies on (local) submodularity. I appreciate that the authors openly acknowledge that this is an assumption and provide an insightful discussion in Appendix F, showing that the acceleration is more pronounced for models like Grounding DINO (which behaves more submodularly) than for Florence-2 (which behaves more supermodularly). Although this is not a fatal flaw, it is a fundamental limitation of the method. The paper could be strengthened by a more prominent discussion of this dependency in the main text, as it defines the boundaries of the method's applicability.
2. While the overall method is shown to be effective, a more detailed ablation study on the individual components of PhaseWin would provide deeper insight into what drives the performance gains. For example, the contribution of the annealing delay vs. the dynamic supervision, and the impact of different window policies from Table 7.

### Questions
1. The hyperparameters for PhaseWin (window size, $\tau$, $m_\mathrm{active}$, etc.) are crucial for its performance. Could you discuss the sensitivity of the results to these parameters and provide more guidance on how to set them effectively for a new model or dataset? Was any formal hyperparameter optimization performed?
2. In the failure interpretation (Sec 4.4), Faster-VPS sometimes slightly surpasses VPS in certain metrics (e.g., Table 4, 100-region setting). Can you hypothesize why the accelerated method might outperform the original greedy search?

### Soundness
3

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
This paper tackles the trade-off between faithfulness and efficiency in attribution methods for object-level foundation models such as Grounding DINO and Florence-2. It proposes an efficient variant of Visual Precision Search (VPS), Faster-VPS. The key contribution is the Phase-Window algorithm, which approximates the greedy search in VPS through phased pruning, windowed fine-grained selection, and adaptive control. The method theoretically maintains a near-greedy approximation bound under submodular conditions, and empirically achieves about 95% of VPS’s faithfulness using only ~20% of the computational cost across detection and grounding benchmarks.

### Strengths
+ VPS’s quadratic cost has indeed limited its scalability, and the focus on computational efficiency is well justified.
+ The phased pruning and windowed selection resemble a practical relaxation of the greedy search, and the dynamic supervision and annealed deferral mechanisms are clearly explained.
+ The paper is generally well organized, with clear figures and detailed methodological explanations.

### Weaknesses
- The problem formulation and, most importantly, the core scoring function $\mathcal{F}$ are borrowed from the original VPS paper. The main novelty lies in a new reordering and pruning strategy for VPS rather than a fundamentally new algorithmic principle. Much of the method appears as an engineering improvement over VPS, not a conceptual advance in attribution theory or optimization. 
- There is a disconnection between the theoretical analysis and the practical implementation. The paper's approximation guarantees (Theorem 3.1) are derived under the assumption that the objective function $\mathcal{F}$ is monotone submodular. However, the authors correctly note in Section 3.2 that their chosen scoring function "is not strictly submodular." This discrepancy raises questions about the applicability of the theoretical guarantees to the actual algorithm. The term "local submodularity" is used to bridge this gap, but it is not formally defined, making the theoretical foundation of the method unclear.
- The proposed solution trades one form of complexity (computational) for another (algorithmic and hyperparameter). The PhaseWin algorithm (Algorithm 2) introduces a considerable number of new hyperparameters (e.g., $w$, $m_{active}$, $\alpha_{sel}$, $\beta_{del}$, $\theta_t$, $\tau$). The paper currently lacks ablation studies to demonstrate the sensitivity of the method to these parameters. This makes it difficult to assess the method's robustness and practicality, as it may require extensive, expert-level tuning to replicate the reported results.

### Questions
(1) Could you please provide an ablation study or analysis on the sensitivity of PhaseWin to its key hyperparameters, such as the window size ($w$) and the pruning thresholds ($\alpha_{sel}$, $\beta_{del}$)? This would help readers understand the method's robustness.

(2) Could you provide a more formal definition or empirical characterization of "local submodularity"?

(3) How general is the PhaseWin algorithm? Is it highly tuned for the specific properties of the VPS scoring function, or could it be applied as a general-purpose accelerator for other greedy, perturbation-based attribution methods?

### Soundness
3

### Presentation
3

### Contribution
2
