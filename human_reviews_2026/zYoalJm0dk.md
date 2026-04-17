# Diversity-Driven Offline Multi-Objective Optimization via Bi-Level Pareto Set Learning

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Multi-objective optimization (MOO) has emerged as a powerful approach to solving complex optimization problems involving multiple objectives. In many practical scenarios, function evaluations are unavailable or prohibitively expensive, necessitating optimization solely based on a fixed offline dataset. In this setting, known as offline MOO, the goal is to find out the Pareto set without access to the true objective functions. This setting suffers from an out-of-distribution (OOD) issue, where the surrogate model is not accurate for unseen designs. Due to OOD issue, surrogate errors may cause the optimizer to select solutions that do not lie on the true Pareto front and are biased toward its extremes. To address this, this paper proposes Diversity-driven Offline Multi-Objective Optimization (DOMOO), which aims to find out a diverse and high-quality set of solutions. Firstly, DOMOO incorporates an accumulative risk control module that estimates the potential risk of candidate solutions and alleviates OOD issue between the training data and the generated solutions. In addition, a bi-level Pareto set learning (PSL) strategy is proposed to jointly learn preference and PSL parameters, then optimize them, enabling adaptation to diverse Pareto front geometries. To further enhance solution quality, we design a diversity-driven selection strategy that extracts a representative and well-distributed set of final solutions. To achieve this strategy, we propose $\text{IGD}_\text{offline}$, a tailored indicator for the offline setting that considers both diversity and convergence, and avoids the bias of hypervolume indicator. Extensive experiments on synthetic and real-world benchmarks, such as neural architecture search, show that, on average across benchmarks, DOMOO achieves a 1.38× improvement in convergence and diversity over comparable methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies the problem of Offline Multi-Objective Black-Box Optimization. Unlike online settings, new evaluations of expensive objective functions are unavailable, leading to significant out-of-distribution (OOD) challenges. To mitigate this issue, the paper proposes DOMOO, a method that initializes the Pareto Set Model using the offline Pareto front and preference vectors, and then continually trains it with preferences sampled from a Dirichlet distribution alongside surrogate model feedback, similar in spirit to PSL-MOBO. A key novelty of DOMOO lies in incorporating an energy model signal into the gradient update to reduce OOD risk during optimization.

### Strengths
- The paper clearly presents the problem setup and the OOD challenges in offline multi-objective optimization.
- Experimental evaluations are comprehensive, with comparisons to relevant baselines.
- The integration of the energy model signal into the optimization process is an interesting design choice.

### Weaknesses
- Figure 2 has a lot of awkward dots like Morse code, which makes it difficult to interpret. A clearer visualization or explanation is needed.
- The “Bi-Level PSL” terminology is confusing. The proposed framework first pre-trains the Pareto Set Model $h_{\phi}$ using the offline Pareto front  and preference vectors, before entering the exploration phase. This deviates from conventional bi-level optimization. If the term “bi-level” is to be retained, the formulation should better align with standard bi-level structures or be renamed to avoid confusion.
- The energy model is a crucial component, yet its motivation and role are insufficiently explained. Most details are deferred to Appendix B, while only brief mentions appear in the main text (e.g., lines 271–272). The paper would benefit from a clearer, better-motivated presentation of this component and its connection to OOD generalization.
- The overall writing flow could be improved for clarity and coherence, especially in the methodological section.
- Most of improvements come from the Synthetic and RE tasks, while results are marginal in the other tasks.

### Questions
- How does the performance of DOMOO vary with different sizes of the offline dataset?
- How sensitive is the framework to the changes of the energy model?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper tackles offline multi‑objective optimization, where one must recommend a set of Pareto‑optimal solutions using only a fixed dataset and where surrogate models suffer from OOD errors that distort the learned front toward extremes. The authors propose DOMOO, a framework combining: (1) Bi‑level Pareto Set Learning (PSL) that jointly updates preferences and the PSL parameters; (2) A diversity‑driven selection of the final recommendation set using a new offline inverse generational distance metric IGD_offline. Experiments on the Off‑MOO‑Bench show improved average HV rank and IGD_offline rank.

### Strengths
1. The offline MOO setting and its OOD failure modes are well‑motivated and illustrated, and the paper clearly explains why surrogate over‑optimism can collapse diversity on the predicted front.
2. The risk‑aware bi‑level PSL is a sensible way to couple preference exploration with risk‑controlled learning, and the use of an energy model to compute R(x) follows prior work while adapting it to multi‑objective PSL updates.
3. The proposed IGD_offline is a pragmatic metric for the offline regime and, combined with HV, gives a reasonable two‑stage selection policy to trade off coverage and convergence.

### Weaknesses
1. Missing comparison to ParetoFlow. The paper cites ParetoFlow[1] but does not compare against it. Given that ParetoFlow also converts multi‑objective problems into preference‑guided single‑objective generation with flow‑based models, it is a highly relevant baseline.
2. Related work mentions PSL‑MOBO, EPS, and CDM‑PSL, but the empirical suite omits these closer preference‑conditioned generators that could be adapted to the offline setting. Even though the authors mentioned that "When applied to offline optimization, they often encounter severe OOD issues.", I still think an empirical experiment is needed to support this claim.

[1] Yuan, Ye, et al. "Paretoflow: Guided flows in multi-objective optimization." arXiv preprint arXiv:2412.03718 (2024).

### Questions
Please see my main concerns above.

1. IGD_offline uses an offline front (estimated from data) and a shifted reference y'. While practical, can this metric advantage methods that extrapolate near existing data?
2. The authors state that guarantees are “established” after Eq. (3), but the main text does not present a theorem/assumptions, and I did not see a clear formal result. Please make any guarantee explicit (statement, assumptions, proof location) or soften the claim.
3. I'm curious why the authors chose a different set of tasks for the computational cost analysis in Appendix E and the hyperparameter analysis in Appendix J

### Soundness
2

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
This paper proposes DOMOO, a risk-aware offline multi-objective optimization (MOO) method designed to handle the out-of-distribution (OOD) issue in surrogate-based offline optimization. The framework combines three core components: an accumulative risk control module, a bi-level Pareto set learning strategy to jointly learn preferences and Pareto parameters, and a diversity-driven solution selection strategy integrating IGD_offline and hypervolume (HV). Experimental results show improvements in convergence and diversity compared to baselines.

### Strengths
1. The paper addresses a relevant problem: offline MOO under distribution shift.
2. The proposed framework is conceptually comprehensive, integrating risk estimation, Pareto set learning, and diversity-aware selection.
3. The idea of modifying IGD for offline evaluation is novel.

### Weaknesses
1. Although the method is claimed to mitigate OOD risks, none of the experimental questions (Q1–Q4) or results explicitly evaluate robustness under OOD scenarios. All benchmarks appear to share the same distribution as the training data, making the central motivation (OOD alleviation) insufficiently supported.
2. The technical novelty of the diversity-driven selection strategy is somewhat incremental. While IGD_offline is a nice adaptation, the combination with HV remains heuristic without clear theoretical guarantees (e.g., Pareto compliance or convergence bounds).
3. The offline IGD definition (Eq. 4) introduces a “shifted reference front,” but its effect and sensitivity are not thoroughly analyzed or justified.
4. The accumulative risk control module is only briefly described and seems to follow conventional uncertainty-weighted or ensemble-based ideas without substantial innovation.
5. The paper would benefit from a more formal treatment of algorithmic complexity and selection efficiency, as well as a deeper analysis of hyperparameter choices (e.g., Dirichlet, shift magnitude).
6. Some minor presentation issues exist, e.g., long paragraph structures and notation inconsistencies, which slightly reduce readability.

### Questions
1. How is the proposed DOMOO empirically validated under true OOD conditions?
2. How sensitive is IGDoffline to the choice of the shift value? Could different scaling factors significantly alter the evaluation outcome?
3. Is there a way to formally show that the combined IGD_offline + HV selection process maintains Pareto diversity or monotonic convergence properties?
4. Could the authors clarify whether the candidate set generated by both the surrogate and the Pareto set model is necessary, or if one of them suffices in practice?
5. What is the computational overhead of DOMOO compared to existing methods?

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
This paper introduces DOMOO, a novel framework for offline multi-objective optimization
(MOO). The core problem it addresses is the out-of-distribution (OOD) issue, where surrogate
models trained on a fixed offline dataset produce unreliable predictions for new solutions,
leading to poor convergence and diversity on the true Pareto front. DOMOO tackles this with a
three-part strategy: (1) an accumulative risk control module to suppress unreliable OOD
predictions, (2) a bi-level Pareto set learning (PSL) strategy to adapt to various Pareto front
geometries, and (3) a diversity-driven selection strategy using a novel indicator to mitigate the
biases of the standard Hypervolume (HV) indicator.

### Strengths
● Paper is very well written and easy to understand
● The paper correctly identifies the OOD problem as a major and "largely unexplored"
challenge in offline MOO. The illustration in Figure 1, showing how surrogate errors can
lead to a severely imbalanced Pareto front, is a clear and effective motivation for the
work.
● The combination of three distinct ideas - bi-level PSL, accumulative risk control, and
diversity driven solution selection - is novel

### Weaknesses
● The authors admit the method is less effective on highly discrete tasks and. The reason
given is that one-hot encoding for high-cardinality categorical variables creates an
extremely sparse, high-dimensional input space that is challenging for the model. This is
a significant limitation, as many real-world problems (like NAS) are inherently discrete.
● The paper mentions that DOMOO takes longer than some baselines due to the risk
control module. While the authors argue this is a worthwhile trade-off, the complexity of
training multiple surrogate models, an energy model, and then running a bi-level
optimization loop (Table 7) makes it a very heavy offline method compared to simpler
approaches

### Questions
● The paper proposes a novel Diversity-Driven Solution Selection (DDSS) strategy in
Section 4.3, which first selects 128 solutions before filling the remaining slots with HV.

Could the authors provide more intuition or an ablation study on how this number was
chosen?

### Soundness
3

### Presentation
3

### Contribution
3
