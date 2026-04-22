# DomED: Redesigning Ensemble Distillation for Domain Generalization

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Domain generalization aims to improve model performance on unseen, out-of-distribution (OOD) domains, yet existing methods often overlook the crucial aspect of uncertainty quantification in their predictions. While ensemble learning combined with knowledge distillation offers a promising avenue for enhancing both model accuracy and uncertainty estimation without incurring significant computational overhead at inference time, this approach remains largely unexplored in the context of domain generalization. In this work, we systematically investigate different ensemble and distillation strategies for domain generalization tasks and design a tailored data allocation scheme to enhance OOD generalization as well as reduce computational cost. Our approach trains base models on distinct subsets of domains and performs distillation on complementary subsets, thereby fostering model diversity and training efficiency. Furthermore, we develop a novel technique that decouples uncertainty distillation from the standard distillation process, enabling the accurate distillation of uncertainty estimation capabilities without compromising model accuracy. Our proposed method, $\textit{Domain-aware Ensemble Distillation}$ (DomED), is extensively evaluated against state-of-the-art domain generalization and ensemble distillation techniques across multiple benchmarks, achieving competitive accuracies and substantially improved uncertainty estimates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DomED, a domain-aware ensemble distillation framework for domain generalization. Two ideas are central. First, complementary domain allocation trains each teacher on a single source domain and distills the student on complementary domains, which is compared against five alternative allocation schemes with consistent advantages in accuracy and calibration. Second, the loss decouples accuracy and uncertainty by combining a standard distillation objective with a Dirichlet NLL regularizer using a small weight. The method is evaluated on PACS, OfficeHome, VLCS, TerraIncognita, and DomainNet using the DomainBed protocol. Results show stable accuracy gains over ERM and clear calibration benefits, with single model inference that approaches test-time ensembles on several reliability metrics.

### Strengths
- **Design aligned with DG structure.** Complementary domain allocation is systematically compared against multiple alternatives and yields small but consistent accuracy gains together with stronger calibration.

- **Uncertainty-preserving distillation.** The decoupled objective $L_S + beta * L_Dir$ avoids the accuracy degradation observed with pure Dirichlet training and improves *ECE* and *NLL*.

- **Pragmatic efficiency observation.** Student performance stabilizes early, which motivates training each teacher for only a fraction of the nominal steps and supports a practical efficiency narrative.

### Weaknesses
- **Lack of student-side quantification for teacher count and domain granularity**
Appendix B shows that single-domain teachers exhibit higher diversity than dual-domain teachers. However, the manuscript does not report how this diversity translates to the student when the number of teachers *M* or the granularity of domain splits changes. 

- **TerraIncognita requires a dataset-specific analysis**
In the DomainBed summary, the improvement on TerraIncognita is modest relative to ERM and notably smaller than on OfficeHome. In addition, on TerraIncognita the ROC-AUC for DomED is slightly below that of a three-model ensemble. The paper provides a mechanism-level explanation for OfficeHome but presents only aggregate figures for TerraIncognita. A brief, dataset-specific analysis for TerraIncognita would complete the narrative.

- **Related work on parameter-efficient fusion is under-covered**
The manuscript situates DomED relative to weight averaging and ensembling but does not cover parameter-efficient expert aggregation that is conceptually adjacent to DomED’s goal of integrating multiple experts. A concise subsection contrasting DomED with representative lines such as AdapterFusion and Mixture-of-Adapters, LoRA-based fusion and merging strategies, and prompt or prefix compositions would sharpen the novelty boundary. 

- **Wording relative to test-time ensembles should strictly match Table 3**
Table 3 indicates that on PACS the EoA ensemble outperforms DomED on ERR, NLL, and PRR, while DomED achieves a lower ECE. Hence, the text should not suggest that DomED broadly approaches ensemble performance.

### Questions
- **Ablation on dark knowledge retention**
As teachers become sharper during late training, non-target probabilities shrink and dark knowledge diminishes. Did you evaluate higher distillation temperatures or entropy-preserving constraints in the late phase to retain dark knowledge, and does this materially improve DomED’s accuracy or calibration under complementary allocation

- **Why Dirichlet NLL instead of simpler or alternative proper scoring rules**
Your calibration gains are attributed to the Dirichlet regularizer. Can the same gains be achieved with strong but simpler single-model calibrators such as temperature scaling, label smoothing with tuned schedules, or focal-style cross entropy, when used in the same complementary allocation regime Further, did you compare against other strictly proper scoring rules for probabilistic classification such as Brier score with class-wise weighting or energy-based calibration penalties

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
5

### Summary
This paper proposes DomED (Domain-aware Ensemble Distillation), a tailored ensemble–distillation framework for domain generalization (DG) that (i) trains domain-specific teachers, each on a single source domain to promote diversity; (ii) performs complementary distillation, where teachers distill only on unseen (complementary) source domains to encourage cross-domain generalization; and (iii) uses decoupled uncertainty-preserving distillation, separating accuracy and uncertainty transfer via a cross-entropy loss and a Dirichlet-prior loss, respectively, to address the trade-off between calibration and accuracy.

### Strengths
1. Clear motivation and formulation. The paper clearly identifies the overlooked role of uncertainty estimation in DG and provides a principled way to integrate ensemble distillation with uncertainty transfer.
2. Novel decoupled uncertainty distillation. The split between accuracy-driven and uncertainty-driven loss terms is conceptually clean and empirically validated.
3. Comprehensive evaluation. The experiments span multiple DG benchmarks and cover both accuracy and uncertainty metrics.

### Weaknesses
1. Limited novelty beyond recombination of known components. The ideas of domain-specific ensembling and uncertainty-aware distillation are incremental extensions of existing work [1]
2. Insufficient theoretical analysis. The method is motivated intuitively but lacks theoretical grounding for why complementary data allocation improves generalization or why the decoupling of uncertainty loss preserves accuracy.
3. Scalability concerns. Despite claiming efficiency, training multiple domain-specific teachers (especially on DomainNet) can be computationally heavy. The paper does not report compute/efficiency metrics for training M domain-specific teachers plus the student, nor cost-normalized accuracy/calibration comparisons against single-model or weight-averaged baselines; scalability with M remains unclear
4. Modest accuracy gains. Improvements over strong baselines.
5. he evaluation under-represents recent knowledge-distillation–based DG methods [2][3][4]

[1] 2022 - ECCV - Cross-domain ensemble distillation for domain generalization

[2] 2021 - ACM MM - Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation

[3] 2023 - ICCV - A Sentence Speaks a Thousand Images: Domain Generalization through Distilling CLIP with Language Guidance

[4] 2025 - IJCAI - Balancing Invariant and Specific Knowledge for Domain Generalization with Online Knowledge Distillation

### Questions
See Weakness.

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
The paper proposes Domain-aware Ensemble Distillation (DomED), a framework of efficient ensemble distillation for domain generalization (DG). DomED investigates and optimizes data allocation schemes for training teacher models on different source domains, followed by complementary domain distillation to a student model. The authors also propose a decoupled uncertainty-preserving distillation objective without sacrificing predictive accuracy. Extensive experiments on several datasets demonstrate DomED's competitive performance and substantial improvements in uncertainty quantification over selected baselines.

### Strengths
1. The thorough evaluation of data allocation schemes is insightful, covering all possible variations with single model architecture and data of distinct domains.

2. The rigorous ablations and clear analyses validate the importance of teacher diversity, the effect of each loss component, and the choice of hyperparameters.

3. The writing and organization of the paper is commendable, highlighting the improved efficiency and uncertainty quantification while maintaining the accuracy.

### Weaknesses
1. The empirical section lacks a direct comparison with a highly pertinent work XDED [1],  and work [2] should at least be mentioned in related work 4.2, both for completeness.

2. The discussion on the scalability with respect to the number of domains is missing. Since the teacher diversity is critical, it remains unknown whether DomED will degrade significantly when the number of teacher decreases.

3. There are a few minor issues can be improved, e.g., tables could be resized denser, and the notation $\tau_m$ in Equation (3) could be revised to $\tau^m$ for consistency.

[1] *Cross-Domain Ensemble Distillation for Domain Generalization.* in ECCV 2022.

[2] *Ensemble Distillation for Out-of-distribution Detection.* in ICPADS 2023.

### Questions
1. Have the author compared DomED’s computational cost in terms of training wall-time and memory usage with other single-model DG baselines such as DNA?

2. Is there the risk of negative transfer when some teachers are 'amateur', i.e., trained on domains with very limited data? At what point might teacher diversity become detrimental?

Please respond to both the weaknesses and the questions for rebuttal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new ensemble distillation method for domain generalization. By combining standard distillation loss and Dirichlet NLL loss, the distilled model can achieve both good accuracy and epistemic uncertainty estimation. The authors provide comprehensive analysis on data allocation strategies for ensemble distillation and compare with multiple baselines on standard benchmarks.

### Strengths
1. This paper studies an important problem in ensemble learning for domain generalization. The authors focus on both accuracy and uncertainty estimation.

2. The authors provide comprehensive analysis on the data allocation strategies for ensemble distillation. This is valuable.

3. The authors recognize the problems in Dirichlet-based uncertainty estimation method and propose Decoupled Uncertainty Distillation to address it.

### Weaknesses
1. Training different models on different domains is relatively straightforward. The main novelty lies in the complementary distillation and decoupled uncertainty learning.

2. By distilling into a single student model, the method loses the flexibility to adaptively weight ensemble members differently for different test domains. This may create potential challenges for handling diverse distribution shifts since one model cannot fit all.

3. The authors claim the Dirichlet objectives can conflict with accuracy, forcing the model to compromise. But what the authors actually do is very straightforward: just adding the standard distillation loss to enforce accurate mean prediction. But this will make the uncertainty estimation not accurate, right?

### Questions
1. You add L_S to keep accuracy. But doesn't this hurt uncertainty quality? How do you verify the uncertainty is still accurate?

2. DiWA requires models trained on all domains and uses 60 checkpoints for weight averaging. Your teachers are trained on single domains only. Do you use the same base models for comparison? If not, how can we know if the performance difference comes from the method or the training setup?

3. Different test domains may require different ensemble weights for each model. How does your single distilled student handle this limitation?

### Soundness
3

### Presentation
3

### Contribution
2
