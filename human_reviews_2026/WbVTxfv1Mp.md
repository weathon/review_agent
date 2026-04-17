# Rectified Decoupled Dataset Distillation: A Closer Look for Fair and Comprehensive Evaluation

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Dataset distillation aims to generate compact synthetic datasets that enable models trained on them to achieve performance comparable to those trained on full real datasets, while substantially reducing storage and computational costs. Early bi-level optimization methods (e.g., MTT) have shown promising results on small-scale datasets, but their scalability is limited by high computational overhead.
To address this limitation, recent decoupled dataset distillation methods (e.g., SRe$^2$L) separate the teacher model pre-training from the synthetic data generation process. These methods also introduce random data augmentation and epoch-wise soft labels during the post-evaluation phase to improve performance and generalization. However, existing decoupled distillation methods suffer from inconsistent post-evaluation protocols, which hinders progress in the field. In this work, we propose **R**ectified **D**ecoupled **D**ataset **D**istillation (RD$^3$), and systematically investigate how different post-evaluation settings affect test accuracy. We further examine whether the reported performance differences across existing methods reflect true methodological advances or stem from discrepancies in evaluation procedures. Our analysis reveals that much of the performance variation can be attributed to inconsistent evaluation rather than differences in the intrinsic quality of the synthetic data. In addition, we identify general strategies that improve the effectiveness of distilled datasets across settings. By establishing a standardized benchmark and rigorous evaluation protocol, RD$^3$ provides a foundation for fair and reproducible comparisons in future dataset distillation research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper revisits the evaluation of *decoupled dataset distillation (DDD)* methods and introduces Rectified Decoupled Dataset Distillation (RD3), a unified benchmark framework for fair and comprehensive comparison.
The authors identify that prior DDD papers (e.g., SRe2L, CDA, DWA, EDC, G-VBSM, D4M) have used inconsistent post-evaluation settings—varying batch sizes, loss functions, schedulers, soft label configurations, and augmentations—making their reported performance incomparable.
RD3 standardizes all these evaluation aspects (batch size, loss, augmentations, optimization schedule, and model architectures), and re-evaluates the main DDD methods under identical conditions.
The results show that previously reported gaps of up to 27% shrink to ~6%, indicating that most gains in prior work were due to inconsistent settings rather than true algorithmic improvements.
Beyond unification, RD3 also provides diagnostic analyses, new metrics (efficiency, cross-architecture generalization), and an open-source benchmark for reproducible comparison.

### Strengths
* **Timely and valuable contribution:** RD3 addresses a real and growing problem in the dataset distillation community: inconsistent evaluation practices that lead to misleading comparisons.
* **Thorough analysis:** The paper systematically disentangles how batch size, loss function, learning rate scheduler, and soft-label design affect performance, providing empirical evidence that evaluation bias dominates algorithmic differences.
* **Unified and transparent benchmark:** All post-evaluation protocols are standardized, with clear rationale (e.g., KL loss for soft-label consistency; batch size = 50 as a fair trade-off).
* **Broader impact on reproducibility:** The open benchmark and diagnostic framework can serve as a long-term reference for fair evaluation of future DDD methods.
* **Clarity:** The paper is well-structured and transparent about design decisions and their effects.

### Weaknesses
* **Limited coverage of cutting-edge methods:**
  While RD3 re-evaluates several major DDD baselines, it currently does not include more *high-performing or near-lossless* distillation frameworks such as DATM and related trajectory-based optimization methods.
  Extending RD3 to evaluate these *lossless distillation* regimes would significantly strengthen its comprehensiveness and demonstrate that the framework remains valid even for next-generation algorithms that achieve nearly full-data performance.

* **Scope restricted to image datasets:**
  The current benchmark focuses solely on vision data. Enabling RD3 to handle other modalities—such as **graph distillation**  or multimodal datasets—would greatly expand its applicability and impact.

* **No new algorithmic contribution:**
  The contribution is primarily evaluative rather than methodological, which is appropriate for a benchmark paper but may limit its novelty for readers seeking new distillation algorithms.

### Questions
1. Could the authors integrate **trajectory-matching-based distillation methods**, to strengthen coverage and demonstrate RD3’s scalability to newer paradigms?
2. How feasible is it to extend RD3 to **non-image domains**, particularly graph data distillation (e.g., GCond[1], GEOM[2])? Would the same evaluation protocol and metrics apply?
3. Are there plans to host RD3 as a public leaderboard or benchmark platform?

[1] Graph condensation for graph neural networks. ICLR 2021.

[2] Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching. ICML 2024.

### Soundness
4

### Presentation
3

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
The paper proposes RD3, a standardized evaluation framework for decoupled dataset distillation. It argues and shows empirically that divergences in post-evaluation choices (batch size, LR schedule smoothing, data augmentations, label formulations, training epochs) drive a large share of performance differences across methods, not the synthesis procedure itself. The authors unify settings across optimization-based, selection-based, and generation-based approaches; evaluate on six datasets (CIFAR-10/100, TinyImageNet, ImageNet-1K, ImageNette/Woof); and test cross-architecture generalization (ResNet family, MobileNet-V2, EfficientNet-B0, Swin-T, ViT-B). They further emphasize efficiency (data generation time) and explore general techniques (alternative initialization, hybrid soft labels, MSE-GT loss) that materially shift outcomes under the same data.

### Strengths
1. **Clear empirical finding with real stakes for the field.** Under a consistent setup, the headline gap between methods shrinks dramatically (27.3% → 6.7% on ImageNet-1K @ IPC=10), reframing prior “SOTA” claims as largely protocol effects.


2.**Unified, well-documented evaluation protocol .** The paper spells out aligned epochs (400), batch size (BS=50 for R-18), smoothing cosine LR (ζ), and a fixed augmentation recipe, and shows incremental effects (Fig. 2). This is useful to both reproduce and compare. 

3.The paper does a comphrhensive analysis and isolates factors many works quietly vary: initialization can help/hurt (e.g., noise harms DWA/EDC; RDED init boosts several methods), and hybrid soft labels help at low IPC regardless of method family.1

### Weaknesses
1.The augmentation suite is taken from RDED and applied universally. That is reasonable for standardization, but it may advantage methods whose images/labels interact best with that augmentation mix. A short sensitivity analysis (swap in SRe2L/CDA augmentations) would address perceived bias. Similarly, fixing BS=50 and ζ (e.g., ζ=1 for ResNets) is pragmatic, yet different families could prefer distinct regimes. The appendix defends ζ choices, but a small grid over {BS, ζ} across families would strengthen neutrality claims.

2.The efficiency plot excludes classifier training time for optimization/selection methods. I’d prefer an end-to-end accounting.

### Questions
1.For MSE-GT, you adopt γ from G-VBSM. Could you provide a small γ sweep (e.g., {0, 0.01, 0.025, 0.05}) across two families to verify that conclusions don’t hinge on one borrowed value? 

2.In Appendix F, do you have analyses (e.g., attention maps, calibration, failure clusters) to explain the >40% ViT drop at IPC=10 and why it recovers at higher IPC? Any indications that label entropy or image diversity drives the effect?

3.In Table1, you standardize on a single ResNet-18 teacher for soft labels. What happens if the teacher is ViT-B or an ensemble (distinct from hybrid labels used at relabel time)? I am curious about different teacher models' effects. (I checked the result in Appendix F, but it doesn’t test ViT-B or ensembles as the teacher)

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper argues that evaluation across existing dataset distillation methods are inconsistent. In consequence, it is unclear whether performance gains from new methods is due to improvement from the distillation mechanism or auxiliary improvement in the downstream model training. To overcome this, the paper introduces Rectified Decoupled Dataset Distillation (RD3) to systematically investigate how different evaluation settings affect test accuracy. RD3 find that previous observed performance increase of over 27% reduces to under 7% with the same evaluation setting.

### Strengths
1. The paper point out a concerning trend with current dataset distillation evaluation protocol that can misrepresent the capabilities of existing algorithm.
2. The paper provides a rather thorough analysis towards understanding the capabilities of current dataset distillation methods, revealing the performance gap between the baseline SRe2L and RDED is a lot smaller than expected.
3. To the best of my knowledge, this is the first work that provide comprehensive training time, generalization, and initialization analysis, which are important considerations for people that use these dataset distillation algorithm in practice.

### Weaknesses
1. The problem of different evaluation procedure exist is many subfields of machine learning and not unique to dataset distillation. More discussion on the unique challenges that exist in dataset distillation would strength the overall motivation of the paper. 
2. The presentation of many arguments in the paper can substantially be improved (see question 1-3).
3. Minor: Figure 3 can substantially be improved. There are eight points on the graph that differs only in color. This makes it difficult to read and not color-blind friendly.

### Questions
1. The paper argues "to ensure fairness, all methods should adopt a unified training setting, regardless of their specific motivations" but it is unclear from the paper why this matters or ensures fairness. Why does using the same evaluation procedure enforce fairness? For instance, method A, compared to method B, can benefit more in from training procedure X than Y. Is it fair to evaluate with procedure X or Y? To a partitioner, the will only care about the final performance on the downstream task. 
2. Figure 2 argues that different techniques leads to different performance impact by why is this the case? The accuracy gains from the technique isn't the same but the relatively ranking between the two method is consistent with the augmentations. Hence, the same conclusion can be drawn. 
3. Section 4.2 argues "in comparison to marginal differences in test accuracy, computational efficiency should be considered a more critical criterion for evaluating different methods". What is considered marginal? For example, RDED takes 1 hour to train and achieve 61.5% but D4M takes 52 hours to train and achieve 63.2%. To a practitioner, the 1-2% improvement in accuracy maybe well worth the 50x increase in training time. 
4. What is the total storage size (including soft labels) across the different methods? 
5. How does this differ from DD-Ranking?

Li, Zekai, et al. "Dd-ranking: Rethinking the evaluation of dataset distillation." arXiv preprint arXiv:2505.13300 (2025).

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper introduces a unified and systematic evaluation framework for dataset distillation to compare decoupled distillation methods.

### Strengths
1. The evaluation is systematic and practical, providing valuable guidance for empirical method comparison and design.

### Weaknesses
1. Lacks fundamental analysis—most observations are based on empirical results and hypotheses.
2. Uses a fixed hyper-parameter set for all methods. As an evaluation framework, this approach is unconvincing since different methods may have different optimal hyperparameters.
3. Some design choices require additional justification (see questions below).

### Questions
1. Should convergence speed be considered a characteristic of a method? If some methods converge faster than others, using longer training epochs may negatively affect their performance.
2. The cross-architecture results are interesting. Would replacing the soft label network with a different network change the results?

### Soundness
2

### Presentation
2

### Contribution
2
