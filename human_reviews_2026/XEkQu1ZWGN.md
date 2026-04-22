# ChemBOMAS: Accelerated Bayesian Optimization for Scientific Discovery in Chemistry with LLM-Enhanced Multi-Agent System

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Bayesian optimization (BO) is a powerful tool for scientific discovery in chemistry, yet its efficiency is often hampered by the sparse experimental data and vast search space. Here, we introduce ChemBOMAS: a large language model (LLM)-enhanced multi-agent system that accelerates BO through synergistic data- and knowledge-driven strategies. Firstly, the data-driven strategy involves an 8B-scale LLM regressor fine-tuned on a mere 1% labeled samples for pseudo-data generation, robustly initializing the optimization process. Secondly, the knowledge-driven strategy employs a hybrid Retrieval-Augmented Generation approach to guide LLM in dividing the search space while mitigating LLM hallucinations. An Upper Confidence Bound algorithm then identifies high-potential subspaces within this established partition. Across the LLM-refined subspaces and supported by LLM-generated data, BO achieves the improvement of effectiveness and efficiency. Comprehensive evaluations across multiple chemical benchmarks demonstrate that ChemBOMAS set a new state-of-the-art, accelerating optimization efficiency by up to 5-fold compared to baseline methods. Additionally, a real wet-lab campaign with strong early-round gains validated the practical relevance of ChemBOMAS.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces ChemBOMAS, a novel framework integrating large language models (LLMs) into Bayesian optimization (BO) for chemical reaction optimization tasks. The approach tries to address the challenges of data scarcity and inefficiency in vast search spaces. ChemBOMAS employs a knowledge-driven module to decompose the search space and a data-driven module for generating pseudo-data to enhance optimization performance. The framework is evaluated through extensive experiments indicating improvements in yield and speed compared to several baselines.

### Strengths
1. Their framework shows integrating LLMs improves BO for chemical optimization
2. The framework was validated through wet-lab experiments

### Weaknesses
1. Presentation and Clarity:
  *  The metrics (like best found, object value max Iter, and Initial) used in the paper are not well explained, neither in the main paper, nor in the Appendix. So it's hard for the reviewers to identify their significance when comparing BO performance.
  * RAG is introduced in the knowledge-driven strategy. How is it implemented, and how does it affect the results?
  * The Notations are confusing. In Line 165, $x_t$ represents the $t$-th token, while in line 169, $x$ is the reaction configuration that contains $R, P,c$. 

2. Rigour of Experiments and Conclusion:
  * As the proposed method has introduced data augmentation and different data initialization conditions for different datasets, it's unclear whether the proposed method is fairly compared to the baselines. Especially in Figure 2, the improvement is more obvious on Suzuki while not too significant for other datasets. If we use the same percentage of datasets as the initial points, as Suzuki has more data points, then the use of it for pretrain and SFT will be more stable.
  * In Table 4, the experimental results are reported without variance; the knowledge module and data module do not always have a positive impact on the results, where the best iteration for Arylation is obtained when neither of these modules is imported.
  * All the results are reported based on five random seeds; this is not reliable for heuristic approaches.
3. The model's performance relies heavily on fine-tuning data volume, suggesting that substantial effort is needed to optimize the model for specific datasets.
4. The integration of knowledge-driven and data-driven modules seem complex and convoluted

### Questions
1. How does ChemBOMAS handle scalability issues, especially in scenarios involving extremely large and complex chemical spaces?
2. What's the cost for the knowledge-driven process that combines RAG and general LLMs? Did you query for space partition at every BO iteration? If so, it should be quite expensive. 
3. What mechanisms are in place to adapt ChemBOMAS effectively under varying initial conditions concerning data volume?

### Soundness
2

### Presentation
1

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
The paper proposes ChemBOMAS, an LLM-enhanced multi-agent framework that accelerates Bayesian optimization for experimental chemistry by combining a knowledge-driven RAG module that partitions the design space with a data-driven module that uses a fine-tuned 8B LLM regressor to produce pseudo-labels for more informative initialization. A UCB-guided subspace selection policy coordinates the two modules in a closed loop, after which BO is run within promising regions. The authors evaluate on standard reaction-optimization benchmarks and a materials-style task, and they report faster early-stage convergence and higher best-of-budget outcomes than strong baselines. Importantly, they provide supplementary materials with code and data that enable replication, and they include an experimental wet-lab validation on a previously unpublished reaction system.

### Strengths
1. Reproducibility and transparency. The authors provide supplementary materials with code and data, including implementation details sufficient to rerun core experiments and inspect prompts and baselines.
2. Practical validation. The approach is tested in a real wet-lab setting with strong early-round gains, demonstrating operational relevance beyond simulation.
3. Clear algorithmic scaffold. The closed-loop interplay between RAG-based subspace construction, UCB-guided selection, and BO within selected regions is well specified and easy to follow.
4. Broad and relevant evaluation. Benchmarks cover standard reaction datasets and a materials-style task, with ablations that probe the value of pseudo-data and partitioning.
5. Sensible problem framing. The method directly addresses cold-start and high-dimensionality pain points that routinely limit BO in chemistry.

### Weaknesses
1. Statistical reporting. Claims such as up to 5-fold acceleration and several percent gains over baselines are not consistently accompanied by confidence intervals, variance across multiple seeds, or significance tests. This is especially important for stochastic BO and LLM-generated pseudo-labels.
2. Safety and feasibility constraints. The current loop does not appear to enforce domain safety or feasibility filters to prevent proposing hazardous or impractical conditions. Incorporating rule-based checks, surrogate predictive models of feasibility, or expert-validated constraints would improve readiness for autonomous labs.
3. Pseudo-label calibration and bias. The LLM regressor can introduce bias if mis-calibrated. There is limited analysis of reliability, label-noise sensitivity, or robustness when pseudo-data conflict with early measurements.
4. RAG partitioning auditability. The retrieval corpus, filtering criteria, and sensitivity of subspace splits to retrieval noise are under-documented. Without this, it is hard to assess whether partitions encode prior bias that drives observed gains.
5. Baseline alignment and coverage. While strong baselines are included, details on kernel choices, acquisition settings, restart budgets, batch sizes, and runtime are scattered. Broader comparisons to recent hybrid methods and a clearer accounting of computational cost would strengthen fairness and scope.
6. Generality claims. The materials-style task is promising but under-described. More context on its construction and accessibility would help calibrate the claim of cross-domain applicability.

### Questions
1. Can you provide means, standard deviations, confidence intervals, and significance tests for all benchmarks and wet-lab studies?
2. How sensitive is the performance of the Bayesian optimization loop to noise or mis-calibration of the LLM-generated pseudo-labels?
3. Is the LLM regressor kept fixed after initial fine-tuning or updated online, and what safeguards prevent confirmation bias in that process?
4. What retrieval corpus and filtering criteria are used for the RAG partitioning, and how often do retrieval errors lead to sub-optimal subspace splits?
5. Are safety or feasibility constraints (e.g., chemical hazard filters) integrated into the proposal generation loop, and if not, how might this affect deployment in autonomous labs?
6. For each reaction dataset and pre-trained component, has overlap between pre-training data and evaluation sets been checked to rule out data leakage?
7. Could you clarify the materials-style “LNP3” task: define its dataset, availability, and whether a minimal open-version will be released for benchmarking?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents a well-designed and innovative pipeline, combining LLM and Bayesian optimization, for chemical optimization.

### Strengths
- The paper is very well written and structure. The flow chart cleanly demonstrates the design of the algorithm while the results are also clearly communicated.
- It appears that the resulting language model si very capable of chemical optimization.
- The optimization trajectories showcase the potential utility of this model in real-world chemical optimization tasks.
- The problem addressed here is exciting and topical.

### Weaknesses
- The design space of BO, in terms of acquisition functions and policies, is explored insufficiently. I also understand that this is not the focus of this paper.
- In the beginning paragraph, the aim of the paper seems to be addressing chemical discovery in general, but only synthesis is considered. I would tune down the claimed scope.
- It seems that the performance gap between the proposed method and LLMs trained on labeled data is not too huge.

### Questions
- Have you considered generalizing this method to other modalities of chemical discovery, such as prediction tasks such as physical property predictions?

### Soundness
3

### Presentation
4

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
This paper presents ChemBOMAS, a multi-agent system using Large Language Models (LLMs) to accelerate Bayesian optimization (BO) for scientific discovery in chemistry. It addresses the common BO challenges of sparse experimental data and vast search spaces. The system combines two strategies: a data-driven approach, using an LLM to generate pseudo-data for robust initialization , and a knowledge-driven approach, using an LLM to intelligently partition the search space, allowing BO to focus on high-potential areas.

### Strengths
It proposed a novel multi-agent system-based Bayesian optimization method.

### Weaknesses
**Lack of LLM regression baseline**

The paper proposed their LLM regression model, but the experimental evaluation is insufficient due to a lack of comprehensive baselines[8~9]. It is difficult to assess the proposed method's true contribution without comparing it against a wider range of existing or even simpler alternative methods.

**About Optimization results**

The results in Figure 2 are concerning and require clarification.
It is unclear why the objective values differ at iteration 0 for the various methods being compared. For a fair comparison, all methods should presumably start from an equivalent initial state.
Following this, the optimization curves for the proposed method appear almost entirely flat after the initial iteration. This suggests that minimal, if any, optimization or "search" is occurring. This raises a critical question: Is the method's performance simply an artifact of finding a good initial data point, rather than a demonstration of an effective optimization process?

**Lack of optimization baselines**

A notable omission is the comparison against LBO (Latent Bayesian Optimization)[1~7]. Given the nature of the problem, LBO methods seem like a highly relevant and potentially strong baseline. The authors should justify this exclusion or provide results for this comparison.

**Reference**

[1] Tripp, Austin, Erik Daxberger, and José Miguel Hernández-Lobato. "Sample-efficient optimization in the latent space of deep generative models via weighted retraining." Advances in Neural Information Processing Systems 33 (2020): 11259-11272.

[2] Maus, Natalie, et al. "Local latent space bayesian optimization over structured inputs." Advances in neural information processing systems 35 (2022): 34505-34518.

[3] Lee, Seunghun, et al. "Advancing bayesian optimization via learning correlated latent space." Advances in Neural Information Processing Systems 36 (2023): 48906-48917.

[4] Chu, Jaewon, et al. "Inversion-based latent bayesian optimization." Advances in Neural Information Processing Systems 37 (2024): 68258-68286.

[5] Moss, Henry B., Sebastian W. Ober, and Tom Diethe. "Return of the latent space COWBOYS: Re-thinking the use of VAEs for Bayesian optimisation of structured spaces." arXiv preprint arXiv:2507.03910 (2025).

[6] Grosnit, Antoine, et al. "High-dimensional Bayesian optimisation with variational autoencoders and deep metric learning." arXiv preprint arXiv:2106.03609 (2021).

[7] Lee, Seunghun, et al. "Latent bayesian optimization via autoregressive normalizing flows." arXiv preprint arXiv:2504.14889 (2025).

[8] Song, Xingyou, et al. "Omnipred: Language models as universal regressors." arXiv preprint arXiv:2402.14547 (2024).

[9] Kristiadi, Agustinus, et al. "A sober look at LLMs for material discovery: Are they actually good for Bayesian optimization over molecules?." arXiv preprint arXiv:2402.05015 (2024).

### Questions
See weakness section above

### Soundness
2

### Presentation
2

### Contribution
2
