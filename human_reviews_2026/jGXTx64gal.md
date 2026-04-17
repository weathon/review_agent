# FERD: Fairness-Enhanced Data-Free Adversarial Robustness Distillation

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Data-Free Robustness Distillation (DFRD) aims to transfer the robustness from the teacher to the student without accessing the training data. While existing methods focus on overall robustness, they overlook the robust fairness issues, leading to severe disparity of robustness across different categories. In this paper, we find two key problems: (1) student model distilled with equal class proportion data behaves significantly different across distinct categories; and (2) the robustness of student model is not stable across different attacks target. To bridge these gaps, we present the first Fairness Enhanced data-free Robustness Distillation (FERD) framework to adjust the proportion and distribution of adversarial examples. For the proportion, FERD adopts a robustness guided class reweighting strategy to synthesize more samples for the less robust categories, thereby improving robustness of them. For the distribution, FERD generates complementary data samples for advanced robustness distillation. It generates Fairness-Aware Examples (FAEs) by enforcing a uniformity constraint on feature-level predictions, which suppress the dominance of class-specific non-robust features, providing a more balanced representation across all categories. Then, FERD constructs Uniform-Target Adversarial Examples (UTAEs) from FAEs by applying a uniform target class constraint to avoid biased attack directions, which distribute the attack targets across all categories and prevents overfitting to specific vulnerable categories. Extensive experiments on three public datasets demonstrate that FERD achieves state-of-the-art worst-class robustness and NSD under all adversarial attacks. For instance, FERD improves worst-class robustness by up to 11.3% and reduces NSD by 0.077 compared to the optimal baseline on CIFAR-10 with MobileNet-V2. Our code is available at: [https://github.com/mayaobuduyao/FERD](https://github.com/mayaobuduyao/FERD).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new framework, FERD, to address the robustness and fairness issues in data-free adversarial robustness distillation. The authors find that, even with uniformly sampled synthetic data, the robustness of the student model in traditional DFRD methods still varies significantly across categories, and the success rate of adversarial attacks also varies depending on the target class. FERD addresses this issue through two strategies: Robustness-guided class reweighting: This increases the generation of samples for weakly robust classes; Fairness-aware sample and uniform target adversarial sample generation: This improves the class coverage of adversarial samples by uniformly constraining feature levels and uniformizing attack targets.

### Strengths
1. This paper is easy to follow.

2. The proposed method is novel. A class reweighting strategy is proposed to enhance robustness against weak classes. FAEs and UTAEs are designed to improve fairness at both the feature and attack target levels.

3. The experimental result demonstrates the proposed method can effectively alleviate fairness issues in data-free ARD.

4. This paper also provides an explanation from a theoretical level.

### Weaknesses
1. This paper only selected one teacher model for one data set. I am curious whether the method is effective when different teacher models are selected.

2. The experimental result in newer architectures, such as VIT, can further verify the effectiveness.

### Questions
See the weakness

### Soundness
3

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
4

### Summary
This paper identifies that data-free robust distillation (DFRD) methods create models with unfair robustness, meaning they are robust for some class but non-robust for others. The authors find this is because DFRD trains on an equal number of samples for all classes and uses attacks that are not diverse. They propose FERD, which fixes this by generating more synthetic data for the non-robust classes and creating special adversarial attacks that target all classes uniformly. Experiments show this method improves the robustness of the weakest classes, leading to better overall fairness in the context of DFRD.

### Strengths
1. The paper is the first to investigate and address the problem of robust fairness specifically within the context of Data-Free Robustness Distillation (DFRD), an area where fairness considerations have been largely overlooked.
2. The proposed class reweighting strategy introduces a mechanism to generate intentionally imbalanced synthetic data tailored for fairness improvement, adapting generation proportions based on class vulnerability within the data-free constraint.

### Weaknesses
1. The paper claims state-of-the-art robust fairness in DFRD but fails to compare against a crucial baseline: combining a standard DFRD data generation method with established fairness-aware adversarial distillation techniques like Fair-ARD or ABSLD. It is plausible that simply applying existing fairness distillation losses to student training, using data from a generic DFRD generator, could yield comparable fairness improvements. Without this comparison, the paper does not demonstrate that FERD's specific components offer benefits beyond a straightforward integration of known DFRD generation and fairness AD methods.

2. Robust fairness methods (e.g., ABSLD) demonstrate their effectiveness by showing significant improvements in worst-case robustness or NSD while maintaining comparable or slightly improved average robustness relative to strong baselines. This isolates the contribution specifically to fairness. However, FERD's results show a large increase in average robustness alongside the worst-case improvement (e.g., Table 1, RN-18/CIFAR-10: Avg. AA +3% vs. Worst AA +1% compared to DFHL). This large average gain makes it difficult to determine if the worst-case improvement is a direct outcome of the fairness mechanisms or simply a byproduct of the model becoming significantly stronger overall. While NSD improves, the paper does not isolate the fairness contribution by comparing against a baseline with similar average robustness, thus obscuring whether FERD primarily enhances fairness or just general robustness, which happens to also lift the worst-case performance.

3. As a pioneering work in DFRD fairness, the paper's evaluation could be strengthened by providing additional context. Including teacher's own fairness profile and student performance under data-available adversarial distillation—using both standard (e.g., RSLAD) and fairness-aware (e.g., ABSLD) methods—would offer valuable reference optimal points. Without these comparisons, and alongside the significant average robustness gains (Weakness 2), it remains somewhat challenging to fully gauge the significance of FERD's fairness improvements specifically within the constraints of the data-free setting.

### Questions
-	While Section 2.1 discusses several DFRD methods (DFARD, DERD, DFHL), your experiments only include DFHL as a direct DFRD baseline, primarily comparing against adapted DFKD methods. Could you explain the rationale for omitting DFARD and DERD from the empirical comparison and for this specific choice of baselines?
-	For the adapted DFKD baselines, why did you use standard PGD for attack generation instead of the inner maximization objective proposed in the original RSLAD paper? Could this choice underestimate the adapted baselines' potential robustness?

This paper tackles the important problem of DFRD fairness with strong results. However, concerns about baselines and context limit the current evaluation. I am open to revising my score based on the rebuttal.

### Soundness
3

### Presentation
2

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
The work offers a valuable perspective on robust fairness under data-free settings and provides convincing experimental evidence, though the methodological and theoretical originality are limited.

### Strengths
The paper addresses the intersection of data-free robustness distillation and robust fairness, which is an emerging and practically important direction for fairness-aware model compression and deployment on edge devices.

The experiments are extensive and demonstrate consistent improvements in worst-class robustness and normalized standard deviation (NSD) across CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

Ablation studies, hyperparameter analyses, and visualization of synthetic samples (Fig. 6, Table 3) enhance the reproducibility and credibility of results.

### Weaknesses
Although the paper includes proofs of conjectures (Appendix A.1–A.2), they are mostly intuitive restatements of empirical observations and lack rigorous formalism. 

The fairness claim relies mainly on NSD and worst-class robustness. Introducing additional fairness metrics would make the evaluation more convincing.

The manuscript is lengthy and includes numerous large figures and dense equations that affect readability.

### Questions
Although the paper includes proofs of conjectures (Appendix A.1–A.2), they are mostly intuitive restatements of empirical observations and lack rigorous formalism. 

The fairness claim relies mainly on NSD and worst-class robustness. Introducing additional fairness metrics would make the evaluation more convincing.

The manuscript is lengthy and includes numerous large figures and dense equations that affect readability.

### Soundness
3

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
4

### Summary
This paper studies **robust fairness in Data-Free Robustness Distillation (DFRD)** — transferring robustness from a teacher to a student model without original data.

The authors identify two fairness issues:

(1) class-wise robustness disparity under uniformly sampled synthetic data, and

(2) target-dependent vulnerability under adversarial attacks.

They propose **FERD**, which enhances fairness at both the *proportion* and *distribution* levels:

- **Robustness-guided Class Reweighting** generates more synthetic data for weakly robust classes.
- **Fairness-Aware / Uniform-Target Adversarial Examples (FAEs / UTAEs)** apply uniformity constraints on feature and target spaces to prevent biased robustness transfer.

Experiments on **CIFAR-10, CIFAR-100, and Tiny-ImageNet** show that FERD achieves higher worst-class robustness (+15.1% under FGSM) and lower NSD than prior DFRD methods, suggesting improved fairness and stability.

### Strengths
- Novel and well-defined problem setup: fairness within DFRD.
- Dual-level solution addressing both sample imbalance and attack-target bias.
- Consistent empirical results across datasets and attack variants.
- Ablation studies verify component contributions.

### Weaknesses
- **Limited theoretical rigor:** The design is intuitive but lacks formal analysis — no explicit robustness bounds, convergence arguments, or assumption statements are provided.
- **Restricted evaluation scope:** Experiments are confined to small datasets (CIFAR-10/100, Tiny-ImageNet) and a single perturbation budget (ε = 8/255), leaving its generality under stronger or diverse attacks unexplored.
- **Hyperparameter opacity:** Key parameters (τ, γ, λ) are fixed without justification or sensitivity analysis, creating uncertainty about robustness across configurations.

### Questions
- Could you include **ε-sweep results (ε = {4,6,8,10}/255)** to test whether fairness improvements persist under stronger perturbations?
- Have you explored the **stability of results** under different τ, γ, and λ settings?

### Soundness
3

### Presentation
4

### Contribution
3
