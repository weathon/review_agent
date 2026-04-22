# Towards Robust Graph Unlearning via Gradient Consistency Control

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Recent graph unlearning models, which aim to efficiently remove undesired data by optimizing a unified objective of forget and retain losses, exhibit a critical vulnerability: their efficacy is severely compromised by inference-time noise attacks on node features. We are the first to diagnose that this fragility stems from a fundamental \textbf{gradient inconsistency} problem. Specifically, we theoretically and empirically demonstrate that within the unified optimization objective of graph unlearning, conventional robustness techniques such as adversarial smoothing are counterproductive: they exacerbate the \textbf{directional conflict} between the forget and retain gradients, leading to negative interference and failed optimization. To address this, we propose RUNNER, a novel framework for \textbf{R}obust graph \textbf{UN}learning via gradie\textbf{N}t consist\textbf{E}ncy Cont\textbf{R}ol. RUNNER resolves this conflict through a principled decoupling strategy comprising two core innovations: (1) a decoupled regularization scheme that independently stabilizes gradients from both the forget and retain losses against perturbations, and (2) a gradient alignment objective that penalizes inconsistent gradient between the two losses. Extensive experiments conducted on four real-world datasets demonstrate that RUNNER significantly enhances robustness against noise attacks while maintaining the model’s performance under noise-free conditions. Codes are available at \href{https://anonymous.4open.science/r/RUNNER-2FD7}{https://anonymous.4open.science/r/RUNNER-2FD7}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets a neglected threat in graph unlearning: tiny inference-time feature noise can undo forgetting in two-loss (forget + retain) objectives. It diagnoses the root cause as gradient inconsistency: standard robustness tricks make the forget/retain gradients point in opposing directions, derailing optimization. To fix this, the authors propose RUNNER with two parts: GSC decouples and stabilizes each loss’s gradients under weight perturbations; HA penalizes misaligned directions more than aligned ones. Across four datasets, RUNNER yields markedly better AUC/AP under multiple noise types and keeps performance under noise-free tests.

### Strengths
1. The paper tackles a real and neglected risk: feature noise can undo graph unlearning. It offers a clear diagnosis and a practical fix (RUNNER).

2. The empirical study is thorough and convincing: four datasets, multiple noise types and levels, clear comparisons to strong baselines, and stepwise ablations that isolate each module’s contribution.

### Weaknesses
1. Robustness is validated with synthetic noise injections; the study lacks evaluations on naturally noisy graph data, which may differ in structure, spectrum, and correlation patterns.

2. The paper does not compare to a denoise-first pipeline (maybe using traditional graph feature denoising methods) prior to unlearning.

### Questions
1. Can you test RUNNER on naturally noisy graphs to demonstrate robustness? 

2. Before running graph unlearning method, what if adding a feature-denoising step? Does this help GNNDelete and RUNNER or is it unnecessary?

3. Could you add a few good-case qualitative examples where baselines fail under  noise but RUNNER succeeds, to make the benefits concrete and interpretable?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work investigates the robustness of approximate graph unlearning methods (those using a unified forget-and-retain loss) against inference-time noise on node features. The authors claim that standard robustness techniques like smoothing introduce a "gradient inconsistency" between the forget and retain objectives, affecting the optimization. They propose RUNNER, a framework to stabilize gradients independently and a "human-aware alignment" (HA) loss to penalize inconsistencies during optimization. Experiments suggest RUNNER improves robustness against noise attacks compared to baseline unlearning and naive smoothing, while preserving utility on clean data.

### Strengths
1. **Problem:** Addresses the relevant and under-explored problem of robustness in graph unlearning against inference-time noise.
2. **New insights:** Provides empirical evidence (Figure 2) suggesting standard smoothing techniques can interact negatively with the two-loss unlearning objective.
3. **New framework:** Proposes a concrete framework combining decoupled regularization and an alignment-inspired loss (HA) to mitigate the identified issue. Demonstrates empirical improvements in robustness against various noise types compared to baseline unlearning and naive smoothing application (Table 1, Figure 3a).

### Weaknesses
1. **Problem Framing:** The "gradient inconsistency" might be overstated as a unique phenomenon caused by smoothing, potentially being just a byproduct of the conflicts in the two-loss objective that smoothing doesn't resolve well. Stronger evidence is needed to differentiate it from standard multi-objective optimization challenges.
2. **Theoretical Support:** The theoretical analysis relies on strong assumptions (local linearity) and provides intuition rather than rigorous guarantees about RUNNER's effectiveness or the necessity of its components.
3. **Complexity of Solution:** The HA loss component adds complexity with extra hyperparameters ($\gamma_1, \gamma_2, \alpha$) requiring careful tuning (Appendix H), potentially making the framework less practical.
4. **Narrow Scope:** Focuses primarily on inference-time noise for edge unlearning (link prediction) with GNNDelete. Robustness against other perturbations (ex., noisy unlearning requests, training noise) or for other tasks/models is not thoroughly explored.
5. **Limited Baselines:** Comparison against adversarial training methods adapted for the two-loss setting is missing. I would like to see Cognac [1] also included.

---

*[1] Kolipaka, Varshita, Akshit, Sinha, Debangan, Mishra, Sumit, Kumar, Arvindh, Arun, Shashwat, Goel, Ponnurangam, Kumaraguru. "A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks." Proceedings of the 42nd International Conference on Machine Learning (ICML).*

### Questions
1. Can you provide more (apart from PubMed) experiments or analysis to more clearly distinguish the "gradient inconsistency exacerbated by smoothing" from the baseline level of gradient conflict inherent in optimizing the forget and retain losses simultaneously *without* smoothing? Does smoothing consistently make the cosine similarity *more* negative than without smoothing, or does it just fail to improve it?
2. The HA loss formulation seems complex. Have you experimented with simpler gradient alignment techniques, such as directly adding a penalty term like $-\cos(g_f, g_r)$ or projecting gradients to remove conflicting components (as in some multi-task learning literature)? How does HA compare?
4. How does RUNNER perform when applied to other approximate unlearning methods besides GNNDelete and INPO (MEGU, GIF, Cognac)? Does the gradient inconsistency issue manifest similarly across different two-loss formulations?
5. Could the observed robustness improvements be achieved simply by very careful tuning of the standard smoothing hyperparameters (like $\gamma$ in CR) without needing RUNNER's specific components, perhaps accepting a slight trade-off in clean performance?

### Soundness
2

### Presentation
2

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
The paper diagnoses a lack of robustness of recent two-loss graph unlearning methods (e.g., GNNDelete / INPO) to inference-time feature noise and proposes RUNNER, which (i) separately regularizes forget/retain gradients and (ii) adds a gradient-alignment penalty. Empirically, RUNNER shows strong improvements on link-prediction (edge-unlearning) tasks on four datasets. The paper contains theoretical claims about why naive smoothing increases gradient inconsistency.
The topic is important and the core idea (control gradient interactions between competing unlearning objectives) is sensible and has empirical merit. However, the work has several weaknesses to address. Therefore, I am currently on the borderline and look forward to a strong rebuttal by the authors, following which I am happy to revise my score.

### Strengths
- The problem setup defined is interesting, and is a refreshing change from the privacy preservation objective of unlearning.
- The motivation is clear. The gradient-inconsistency phenomenon is demonstrated empirically (cosine similarity plots) and used as the basis for the method.
- The experiments study multiple noise families and ablate components.
- The proposed method is simple and elegantly solves the proposed problem.

### Weaknesses
1. I would like to see more of a discussion on the paradigm of corrective unlearning introduced in [1]. The motivation of noise injection is similar to adversarial attacks, and could warrant a deeper discussion and also an inclusion as a baseline, as [1] has similar experiments.
2. The experiments are severely lacking in many areas.
    - Breadth and validity of datasets. There are only 4 datasets discussed, and all of them are old datasets which the community has been warning against [2]. I feel in general the experimental breadth is not enough to support the paper's claims. PubMed should absolutely not be used to present the main findings of the paper. Similarly, node unlearning results are deferred to the appendix and are only compare with GNNDelete on two datasets. I understand the motivation of the paper is focused on two-loss methods, however comparison with a wide variety of methods is necessary to evaluate if this problem is actually a problem in practice. 
    - Questionable performance for retrain. Retraining is usually the gold standard for unlearning. Why is the performance low in Table 1? Is this due to the added noise? Also, retraining performance is not present everywhere. Please be consistent with the methods reported.
    - Lack of error bars. The paper reports the mean performance, but does not report the standard deviation in the runs, making it hard to identify the statistical significance of the work.
   - Comparison with newer methods. Why have newer methods like [1, 3] not been evaluated? In general the evaluation seems limited. It would be nice to see if newer methods also suffer from robustness issues and would make the paper's claims stronger.
   - Runtime analysis. Please provide a runtime analysis for all methods in your main results, not just GNNDelete, which is at this point a very old baseline. 
3. Is robustness only a problem for two-loss methods? It would be nice to see if other methods suffer from it or not, particularly [1] since it operates on adversarial attacks. 
4. It would help the paper's merit if the authors can discuss attacker capabilities and whether the defense generalizes to stronger adversarial attacks (worst-case adversarial perturbations crafted with full model knowledge) as opposed to random noise. If the method only helps against random noise but not adversarial perturbations, that should be clearly stated.

[1] https://arxiv.org/abs/2412.00789

[2] https://arxiv.org/abs/2502.14546

[3] https://arxiv.org/abs/2401.11760

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
