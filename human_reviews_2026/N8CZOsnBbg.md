# Nature-Inspired Population-Based Evolution of Large Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 6

## Abstract
Evolution, the engine behind the survival and growth of life on Earth, operates through the population-based process of reproduction. Inspired by this principle, this paper formally defines a newly emerging problem---the population-based evolution of large language models (LLMs)---and introduces a novel framework. Starting with a population of parent LLMs, our framework enables the population to evolve through four key operations: (i) crossover, merging the weights of different parents to create offspring LLMs, (ii) mutation, introducing small, random changes to model weights to foster diversity, (iii) selection, prioritizing high-performing models, and (iv) succession, transferring the learned experience from parent to offspring LLMs. With only 200 samples per new task, the LLM population evolves rapidly to adapt to the task at hand, without any gradients. Experiments on 12 datasets show that our framework consistently outperforms existing multi-LLM merging and adaptation methods, achieving relative accuracy gains of up to 54.8% over the best LLM in the initial population. Moreover, our framework allows for the evolution of LLMs across multiple new tasks simultaneously, scaling effectively with populations of up to 40 LLMs, and even zero-shot generalization to unseen held-out tasks. We have open-sourced the code on GitHub enabling reproduction of our proposed framework using just a single 4090 GPU with 24GB memory, without any performance degradation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces GENOME and GENOME+, two algorithms for population-based evolution of LLMs, inspired by genetic algorithms. The framework defines genetic operators (crossover, mutation, selection) for model weight merging and introduces two additional operations: succession (inspired by particle swarm optimization) and ensemble (to combine predictions from top individuals). Both methods aim to evolve a population of LoRA-based LLM experts without backpropagation, using only small validation sets (200 samples). Experiments across 12 datasets show consistent performance improvements over prior baselines like Model Swarms[1] and LoRAHub[2].

### Strengths
- Formal definition of the population-based LLM evolution problem.
- Extensive empirical validation across 12 datasets, 7 baselines, and 2 foundation models (Gemma 2 2B and Llama 3.1 8B).
- Clear empirical gains, especially on reasoning-heavy tasks (DROP, MGSM).
- Good scalability up to 40 models, reproducible on consumer hardware (Nvidia 4090 GPU).
- Open-source codebase.

### Weaknesses
1. Only LoRA adapters are evolved: unclear generalization to full-weight finetuning.
2. No comparison to Evolutionary Model Merging[3] or MERGE³[4] despite major conceptual overlap.
3. GENOME+ shows major improvements but with higher computational overhead, not quantified beyond wall-clock time.
4. “Succession” is effectively a simplified PSO update; not fully original.

### Questions
1. Have you tested GENOME and GENOME+ under the same setting as Evolutionary Model Merging[3] or MERGE³[4]?
2. Have you explored extending GENOME(+) to full-weight finetuned populations rather than LoRA adapters?
3. Can you provide FLOP-level estimates of GENOME+ vs. GENOME (given that GENOME+ introduces additional experience and ensemble operations)?

### References

[1] Feng, S., Wang, Z., Wang, Y., Ebrahimi, S., Palangi, H., Miculicich, L., Kulshrestha, A., Rauschmayr, N., Choi, Y., Tsvetkov, Y., Lee, C. & Pfister, T.. (2025). Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence. _ICML_ (2025) .

[2] Huang, C., Liu, Q., Lin, B., Pang, T., Du, C., & Lin, M. (2023). LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition. _COLM_ (2024).

[3] Akiba, T., Shing, M., Tang, Y., Sun, Q., & Ha, D. (2024). Evolutionary Optimization of Model Merging Recipes. _Nat Mach Intell_ **7**, 195–204 (2025).

[4] Mencattini, T., Minut, A.R., Crisostomi, D., Santilli, A., & Rodolà, E. (2025). MERGE$^3$: Efficient Evolutionary Merging on Consumer-grade GPUs. _ICML_ (2025).

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
4

### Summary
The paper introduces GENOME, a new evolutionary-based post-training techinque that extends ideas from evolutionary model merging. Model merging is a post-training strategy that constructs new models by combining previously fine-tuned models sharing a common pretrained backbone, allowing the reuse of specialized expertise across domains. In contrast, evolutionary algorithms are natural-evolution-inspired, gradient-free optimization methods that evolve candidate solutions through crossover, mutation, and selection. GENOME unifies these two paradigms by treating each fine-tuned model as an individual in a population whose parameters evolve toward higher task fitness via biologically inspired mechanisms (crossover → simple merging, mutation → adding noise). Building upon this foundation, GENOME+ enriches the process with succession,  a mechanism for transferring knowledge from high-performing models to others via another merging step, and ensemble inference, which aggregates outputs from the top-performing individuals for more robust predictions via an ensemble. The framework is evaluated using the Gemma-2-2b-it model across twelve diverse datasets, demonstrating consistent performance gains (even though, it is hard to judge them significant due to lack of std report). However, the paper fail to acknowledge more recent methods in model merging and evolutionary optimization, therefore omitting crucial baselines for data-intensive merging pipelines.

### Strengths
- **Extensive Evaluation:** The paper presents an impressively thorough empirical study of the proposed merging method on **Gemma-2-2b-it**, covering a wide range of datasets and evaluation settings. Such comprehensive testing is rare in the model merging literature, where evaluations are often limited or approximate. The authors deserve particular credit for this, despite making the significance of these findings is tricky to interpret without reporting the standard deviation.
    - I particularly appreciated the ablation, it is very indepth. Nevertheless, it has the same problem that I mentioned before (its lack of standard deviation reporting remains a limitation for understanding its actual validity)
- **Original Evolutionary Steps:** The **succession** and **ensemble** components introduced in **GENOME+** represent genuine novelties compared to prior model merging methods, enabling knowledge transfer and collective inference within the evolving model population. Nevertheless, the broader evolutionary framework itself is not conceptually new, as it closely parallels established approaches such as [1, 2], and mainly adapts well-known genetic algorithm principles to the context of large language model merging rather than introducing a fundamentally new post-training optimization paradigm.
- **Relevant Results:** The paper demonstrates consistent improvements over **Model Swarms**, with GENOME+ achieving higher average performance across multiple benchmarks and task types. This suggests a meaningful advancement in population-based model adaptation. However, because the results are reported only as averages across random seeds without standard deviations, it is difficult to fully assess the reliability or statistical strength of these gains. Providing variance estimates would have made the comparison more convincing and transparent. Furthemore, it completely miss baseline against other SOTA method such as Akiba or MERGE3 [1,2].

- *The code* is well done and it is a contribution

[1] Akiba, Takuya, et al. "Evolutionary optimization of model merging recipes." *Nature Machine Intelligence* (2025): 1-10.
[2] Mencattini, Tommaso, et al. "MERGE3: Efficient Evolutionary Merging on Consumer-grade GPUs." Proceedings of The Forty-Second International Conference on Machine Learning (ICML) (2025).

### Weaknesses
- **Presentation quality.** The paper’s presentation could be improved, as the overall layout, and particularly the teaser in **Figure 1,** appears overly busy and difficult to follow.
    - Moreover, the way performance gains are reported can be somewhat misleading: for instance, the claimed **10.75% improvement** over Model Swarms is expressed as a *relative* rather than *absolute* increase, which may exaggerate the perceived impact. *Given that the true gain is likely within the single-digit range, statistical testing or at least reporting standard deviations would be necessary to judge the real significance of these improvements.*
- **Overstated Novelty.** One of the paper’s main claims is the introduction of an **LLM population-based evolution** framework. However, this concept is **not novel**. While Akiba et al. [1] may have offered a narrower perspective (focusing e.g. on CMAES), similar ideas of LLM evolution through model merging have already been explored, notably in **MERGE3 [2]**, which the authors do not adequately acknowledge. This omission hides a reduced originality of the contribution: the paper does not truly introduce the first formulation of population-based LLM evolution, but rather a **variation of previously established approaches**. Although the inclusion of an ensemble step arguably makes the method somewhat more general than prior evolutionary model merging frameworks, its specific role and added value remain unclear (see below). Moreover, the other three evolutionary operations, crossover, mutation, and selection, are simply merging step with gaussian noise, so they are not a true novel contribution to evolutionary merging of LLMs.
- **Poor Baseline Selection:** A major limitation of the paper is the **lack of appropriate comparison**, which raises concerns about the validity of the reported improvements. Because key prior work is not fully considered, the evaluation omits comparisons with the most relevant methods, most notably **MERGE3**, which also implements an evolutionary merging framework compatible with consumer-grade GPUs, and **Akiba et al.**, whose approach shares conceptual overlap with GENOME+.
    - Combined with the **absence of standard deviations** in performance reporting and therefore the unclear magnitude of gains over **Model Swarms**, it becomes difficult to assess whether the observed improvements are statistically or practically meaningful.
    - Furthermore, the experiments rely exclusively on **ad hoc fine-tuned experts** rather than publicly available or standardized fine-tuning checkpoints (e.g., from Hugging Face), further weakening the robustness and comparability of the evaluation. As a result, the current experimental setup does not convincingly demonstrate a clear or generalizable advantage over existing baselines.
- **Result Significance:** A broader issue throughout the paper concerns the **lack of statistical significance analysis and standard deviation reporting**, which undermines confidence in the reported results. Almost all performance comparisons are presented as mean values without any measure of variance, making it difficult to assess whether observed differences are consistent or simply due to random fluctuations. This limitation affects both the main results and the ablation studies, where conclusions about the contribution of individual components may not be statistically reliable. Incorporating statistical significance tests (e.g., t-tests or confidence intervals) **or at least standard deviation** would be essential to substantiate the claimed improvements and to better demonstrate the robustness and reproducibility of the proposed approach.

In conclusion, the work presents an interesting framework: I found GENOME+ definitely interesting, and I would like to try in the future. But I suggest **rejecting** because the claimed novelty is overstated relative to prior evolutionary model-merging, which are ignored. Without that, the paper proposes a method that builds on others, and therefore should show clear improvement. Nevertheless, the empirical validation lacks key baselines, variance/significance reporting, and clear evidence that gains are reliable or generalizable.

### Questions
Q1) I do not fully understand how the ensemble mechanism operates. Is the ensembled model subsequently considered as part of the population? If so, how is this “ensembled individual” integrated into the standard GENOME operations, such as mutation and crossover?

Q2) To fairly compare data-free merging methods with data-based merging approaches, it would be necessary to include a baseline using a random or Bayesian hyperparameter search. Specifically, have you conducted an equivalent FLOP-budgeted random or Bayesian search for DARE_TIES or similar baselines to ensure a fair comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper formalizes LLM population-based evolution and proposes GENOME and GENOME+, which evolve populations of LoRA adapters using fitness-weighted crossover, masked Gaussian mutation, elite + fitness-proportional selection, and (for GENOME+) a succession (experience vector) update plus top-k ensembling at inference. Experiments across 12 datasets report notable gains over several dynamic model-merging baselines and claim scalability up to 40 models and feasibility on a single RTX 4090.

### Strengths
The paper addresses a timely, practical problem: reusing many fine-tuned LoRA experts without full gradient fine-tuning. As the performance of open source models improves, this provides a valuable direction for combining complementary strengths.

It relies on simple, implementable evolutionary operators (linear mixing, masked mutation, selection) that are attractive as a gradient-free approach.

Broad empirical ambition: diverse tasks (12 datasets), multi-task and zero-shot experiments, and population-scaling studies, base models of two different practical sizes.

The authors indicate code release and include many appendix details, which could support reproducibility and adoption.

### Weaknesses
In my opinion there are two missing critical baselines: no gradient-based (multi-task) LoRA fine-tuning on the same 200 samples and no simple top-k ensemble of original experts to separate ensemble gains from evolution.

It would be appreciated if the authors could improve on the presentation of key algorithmic details, e.g. the exact per-layer/parameter merging for LoRA adapters, mask sampling granularity (per-parameter vs per-matrix), mutation application, and successive normalization are unclear.

There is some risk of overfitting to the small 200-sample validation set used for evolution and selection; it would interesting to investigate the robustness to the specific validation choice and size is not demonstrated.

The compute and hyperparameter tuning details appear incomplete: how many tuning runs were used per method, were baselines given the same hyperparameter budget, and how many seeds were used?

Finally, all results consider two base models Gemma-2-2B-it, Llama-3.1-8B. In order to demonstrate general applicability, the authors should potentially look into larger base models, e.g. >10B, etc. Additionally, why were the Llama results not explicitly discussed in the main text?

### Questions
Can you provide direct baselines: (a) LoRA gradient-based fine-tuning on the same 200 validation samples with a comparable compute budget; (b) top-k ensemble of the original experts (no evolution); (c) naive averaging of top experts. Report these on main datasets with identical prompts and compute constraints.

Additionally, can you add low-level implementation details for parameter merging: how are LoRA matrices combined elementwise or per-layer? How are low-rank decompositions preserved?

Could you demonstrate robustness to validation set choice and size: repeat adaptation/evolution with multiple disjoint 200-example validation samples and with other sizes (e.g., 50, 500) and report stability of selection and final test performance.

Also, can you provide ablations that isolate ensemble vs evolution: (i) evolve without ensemble and report best individual performance; (ii) ensemble-only of original experts (no evolution); (iii) random merging + ensemble. Report statistical significance.

Can you please extend the analysis to one more large base model (>10B)?

Finally, how does this work relate and contrast to CycleQD ([1], Kuroki, et al., 2024) especially with regards to the multi-task setup?

[1] Kuroki, S., Nakamura, T., Akiba, T. and Tang, Y., 2024. Agent skill acquisition for large language models via cycleqd. arXiv preprint arXiv:2410.14735.

### Soundness
3

### Presentation
4

### Contribution
3
