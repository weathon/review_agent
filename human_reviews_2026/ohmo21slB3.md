# Understanding the Role of Rehearsal in Continual Learning under Varying Model Capacities

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
Continual learning, which aims to learn from dynamically changing data distributions, has garnered significant attention in recent years. However, most existing theoretical work focuses on regularization-based methods, while theoretical understanding of the rehearsal mechanism in continual learning remains limited. In this paper, we provide a closed-form analysis of adaptation, memory and generalization errors for rehearsal-based continual learning within a linear-Gaussian regression framework, covering both underparameterized and overparameterized regimes. We derive explicit formulae linking factors such as rehearsal size to each error component, and obtain several insightful findings. Firstly, more rehearsal does not always better for memorability, and there exists a decreasing floor for memory error when tasks are similar and noise levels are low. Secondly, rehearsal enhances adaptability under underparameterization, but can be provably detrimental under overparameterization. Moreover, enlarging the rehearsal size can raise peaks in generalization error when slightly overparameterized, and may further degrade generalization when tasks are dissimilar or noise is high. Finally, numerical simulations validate these theoretical insights and we further extend the analysis to neural networks on MNIST, CIFAR-10, CIFAR-100 and Tiny-ImageNet. The empirical curves closely follow with the predicted trends, indicating that our linear analysis captures phenomena that persist in modern deep continual learning models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents theoretical analysis of rehearsal-based methods for continual learning in terms of memory, adaptation, and generalization errors, and shows the effectiveness of the formulation with experiments on the four benchmark datasets. The analysis implies some interesting observations that more rehearsal is not always beneficial and the rehearsal affects model adaptability differently in underparameterized and overparameterized cases.

### Strengths
- Overall, a sound theoretical formulation is provided in a logical way.

- The experiments as well as a thorough derivation at the appendix support the main findings.

### Weaknesses
- The assumption of the theoretical formulation is too restricted to draw any interesting conclusion for practical algorithms based on rehearsal mechanism: the Gaussian assumption and linear regression problem for each task.

- The experiments are limited to the relative easy datasets, and more updated methods for rehearsal should be used for the comparative study.

### Questions
- How much effective is the proposed analysis for the practical rehearsal-based continual learning?

- Are the assumptions for the formulation reasonable for general rehearsal-based continual learning?

- How do you expect the results for the larger, practical datasets with recent advanced rehearsal-based methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors theoretically analyze adaptation, memory, and generalization errors for rehearsal based continual learning within a linear Gaussian regression framework, covering both underparameterized and overparameterized regimes. They also validate their theoretical insights on several benchmarks.

### Strengths
1. The theoretical and experimental analyses are comprehensive. In particular, the paper studies rehearsal based continual learning in both the underparameterized and the overparameterized regimes, and evaluates on commonly used continual learning benchmarks to verify the theoretical results.
2. The paper offers several interesting insights, for example that more rehearsal does not always improve memorability, and that rehearsal enhances adaptability under underparameterization but can be provably detrimental under overparameterization.

### Weaknesses
1. The theoretical results rely heavily on the linear model and the Gaussian assumption. We do not deny the value of these assumptions: the linear Gaussian regime yields many elegant results with practical guidance. Nevertheless, adding a clear discussion of their limitations would make the paper more complete.
2. The theoretical novelty is limited, and parts of the results overlap with prior work (e.g., Banayeeanzade2024, Zheng2024). Nevertheless, the authors provide a relatively comprehensive discussion and accompanying experiments.

### Questions
1. If the Gaussian assumption is relaxed, do the theoretical results still hold? If so, could you add a simple synthetic data experiment to verify this?
2. The insights in lines 282--289 and 335--341 are derived for the two task case. Do these insights extend to $T>2$?
3. In the theoretical analysis, each update assumes using the entire buffer, whereas in standard training one typically draws random mini batches from the buffer. How is rehearsal implemented in your experiments? Do you sample mini batches from the buffer at random, or do you train with the full buffer at each step? We expect that drawing random mini batches from the buffer may lead to different phenomena and thus deserves discussion.

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
The paper studies rehearsal-based continual learning (CL) through a linear–Gaussian regression lens and provides closed-form expressions for adaptation, memory, and generalization errors under both underparameterized and overparameterized regimes. The core technical results (Theorems 1–3) quantify how rehearsal size s, parameter dimension p, sample size n, inter-task similarity, and noise σ affect these three errors. 

Key takeaways include: 

(i) more rehearsal is not always better that is there can be a decreasing floor in memory error and a degradation of adaptation/generalization when slightly overparameterized; 

(ii) rehearsal helps adaptation when n+s>p+1 (underparameterized) but can hurt when p>n+s+1 (overparameterized) and 

(iii) generalization can worsen with larger s in slight overparameterization but improve in heavy underparameterization. 

The authors back the analysis with simulations and DNN experiments on MNIST, CIFAR-10/100, and Tiny-ImageNet. The empirical trends broadly match theory, and ablations cover buffer sizes and simple sampling strategies (Random/Reservoir/Herding).

### Strengths
- Clear, unified decomposition of CL performance into adaptation/memory/generalization with closed-form expectations. It was asy to reason about levers like s,p,n,σ (Sec. 3–4; Eqs. 2–4, 5–14).

- Main insights include: The rehearsal is helping underparameterization but harming in slight overparameterization. The existence of a decreasing floor for memory error is nice. An analysis of non-monotonic generalization vs. s given by Theorems 1–3 and the discussion under Fig. 2–3. 

- The authors present simulations plus DNN studies across four datasets. The trends (e.g., adaptation worsening with larger buffers; memory non-monotonicity) qualitatively match the linear analysis. (Figs. 4–5; Tables 1–4). 

- Positioning vs. recent theory on CL and rehearsal (e.g., comparisons to overparameterized analyses and multi-task links) seems valid. It shows awareness of the literature and clarifies scope (Related Work).

### Weaknesses
- Modeling assumptions might narrow external validity. Gaussian i.i.d. features, equal per-task n/σ, and the reliance on min-norm solutions limit transfer to realistic non-Gaussian, structured vision features or varying task sizes/noise. I would sugest that the authors consider relaxing Assumption 2 and see if they can can add/report at least partial results beyond equal n,σ (Assumptions 1–2).

- Empirical baselines are somewhat thin. The DNN section primarily varies buffer size/strategy and architecture depth but does not compare against strong non-rehearsal and hybrid CL baselines (e.g., EWC, LwF/Distillation, DER/ER-ACE, prompt-based CL). Without these, it is harder to assess practical implications of “harmful rehearsal” when practitioners could switch methods.

- Sampling strategy analysis seems not full. Table 1 covers adaptation for Random/Reservoir/Herding, but memory and generalization under these strategies are not reported. Also no class-balanced or distance-aware modern coresets are examined. As reported, it kind of weakens the claims about rehearsal “harms” in practice (might be strategy-dependent). 

- Clarity on inter-task similarity questionable. The main text often appeals to “task similarity” but operationalizes it variously (parameter distances in theory; class overlap in DNNs). A single main-text definition plus a mapping between theory-side || w_k^∗ − w_j^∗ ||, ||w_k^∗ − w_j^∗||  and dataset-side similarity might be helpful here. (Secs. 3–5, Fig. 4c–f).

- Figures lack uncertainty. Many curves lack confidence intervals. Some axes are log while others are linear without explicit callouts, making visual comparisons harder. Simple recommendation would be to add error bars and consistent scaling. (Figs. 2–5).

- Reproducibility details light. The paper notes multiple runs and averages, but hyperparameters, optimizers, learning-rate schedules, buffer management, and training protocols for each dataset/architecture are mainly in the appendix. A dedicated reproducibility checklist and code link would help. (Sec. 5; Appendix H).

### Questions
- Scope beyond equal n, σ. Can the theorems extend to unequal per-task sample sizes and noise? If so, what changes in Eqs. (5–14) qualitatively (e.g., replace n with n_t)? A brief corollary or remark would boost applicability.

- From min-norm to practical SGD: Appendix A argues GD/SGD converges to the min-norm interpolant under zero-loss. In noisy DNN training (non-interpolating), which part of the theory do the authors expect to persist, and what breaks? Do the authors maybe have any experiments with early stopping / weight decay to probe this?

- Sampling strategies: Could the authors show/report memory/generalization errors for Random/Reservoir/Herding (Table 1 only shows adaptation) and include a stronger, recent coreset selection (e.g., gradient-based or diversity-aware) to test whether “harmful rehearsal” is mitigated by better buffers?

- Baselines: It might make sense to add 1–2 strong non-rehearsal CL baselines (e.g., EWC, LwF/distillation, adapter/prompt-based) to the DNN section to contextualize how the outlined rehearsal insights translate into method choice?

- Where is the turning point? The theory predicts non-monotonicity vs. s in slight overparameterization. Can the authors provide a simple practitioner rule (in terms of p,n,s,σ or proxy measures) to detect when increasing s starts to harm? A small diagnostic might add more value to the paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper provides a theoretical investigation of rehearsal-based continual learning, focusing on how model capacity influences adaptation, memory, and generalization performance. Using a linear Gaussian regression framework, the authors derive closed-form expressions for different error components under both underparameterized and overparameterized settings. Theoretical findings show that underparameterized models benefit from rehearsal, while overparameterized ones may experience degradation. Simulations and neural network experiments on MNIST, CIFAR-10/100, and Tiny-ImageNet confirm that the analytical trends hold in practice, suggesting the theoretical results capture key behaviors of real-world continual learning systems.

### Strengths
The theoretical analysis of continual learning with rehearsal is an important and meaningful topic. The paper is clearly presented, combining theoretical results derived from a linear model with empirical validation on deep neural networks.

### Weaknesses
A major limitation of this paper is the lack of novelty in its theoretical results. Much of the analysis appears to replicate existing work rather than introduce new insights. In particular, compared with the recent work [Deng et al., 2025], the setup (including the linear regression framework, Gaussian feature assumptions, rehearsal mechanism, and even the mathematical expressions in the theorems) appears nearly identical.  

*Reference:*  
Junze Deng, Qinhang Wu, Peizhong Ju, Sen Lin, Yingbin Liang, and Ness Shroff. *Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective*, 2025.

### Questions
1. As mentioned in the weaknesses, the comparison with related literature is not sufficiently careful. The citation of [Deng et al., 2025] in the last paragraph of Section 2 is too brief ("While some of these studies also use the linear Gaussian model, they focus on different aspects of continual learning"). A detailed comparison is needed to clarify which results are novel and which have already been established.  
2. The font size in Figures 2 and 3 is too small to read comfortably. Please enlarge the text for clarity.  
3. In the last paragraph of Section 3, the authors discuss striking a balance among adaptability, memorability, and generalizability. However, the paper does not provide any quantitative metric to measure or verify this balance. How can one determine whether such a balance is achieved in practice?

### Soundness
3

### Presentation
3

### Contribution
2
