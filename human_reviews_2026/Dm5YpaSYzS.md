# LLMs as In-Context Meta-Learners for Model and Hyperparameter Selection

- Avg Score: 3.20
- Decision: Reject
- Scores: 2, 4, 2, 4, 4

## Abstract
Model and hyperparameter selection are critical but challenging in machine learning, typically requiring expert intuition or expensive automated search. We investigate whether large language models (LLMs) can act as in-context meta-learners for this task. By converting each dataset into interpretable metadata, we prompt an LLM to recommend both model families and hyperparameters. We study two prompting strategies: (1) a zero-shot mode relying solely on pretrained knowledge, and (2) a meta-informed mode augmented with examples of models and their performance on past tasks. Across synthetic and real-world benchmarks, we show that LLMs can exploit dataset metadata to recommend competitive models and hyperparameters without search, and that improvements from meta-informed prompting demonstrate their capacity for in-context meta-learning. These results highlight a promising new role for LLMs as lightweight, general-purpose assistants for model selection and hyperparameter optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes the application of LLMs to the Combined Algorithm Selection and Hyperparameter (CASH) problem. The authors present two prompting strategies: a "Zero-Shot" approach, which relies on a description of the dataset, and a "Meta-Informed" approach, which includes examples of previously solved tasks in the prompt. The performance of these methods is evaluated on a synthetic task and a benchmark of 22 real-world Kaggle datasets, with the goal of demonstrating that LLMs can act as efficient, non-iterative assistants for AutoML.

### Strengths
1. Well-presented manuscript: The paper is clearly written, well-structured, and easy to follow. The authors do an excellent job of explaining their methodology and presenting their results in an organized fashion, which aids in understanding the work.

2. Comprehensive benchmark collection: A great effort was clearly invested in collecting and evaluating the method on a diverse set of 22 Kaggle challenges. This provides a broad and transparent, albeit unflattering, view of the proposed method's performance across different real-world tabular data problems.

### Weaknesses
1. Lack of novelty.
At its core, this paper does not introduce a new method. It is a straightforward application of standard zero-shot and few-shot prompting to an existing problem. Describing this as "in-context meta-learning" is a generous reframing of what is. The paper offers no new algorithms, no architectural insights, and no novel techniques. Besides, LLM has been applied for AutoML in a more comprehensive way in literature. It is an application study that fails to contribute meaningfully to the advancement of either AutoML or the science of LLMs.

2. The problem scenario is ill-suited for an LLM and the performance is not good enough.
The paper fails to justify why the CASH problem, particularly for a single dataset, is a task that requires an LLM. An experienced data scientist can typically select a strong model and a reasonable hyperparameter search space in minutes by applying common heuristics. The LLM's reasoning traces confirm that it is merely replicating these same simple heuristics (e.g., "CatBoost is good for categorical features"). More importantly, the results are underwhelming. No significant performance gap can be witnessed between applying LLM and some simple baselines like Context-Random. Moreover, no baseline with expert's knowledge nor AutoML in the same search space are compared, which are expected to be likely show better performance than the proposed method.

3. The "Meta-Informed" approach is impractical.
This is the most critical flaw of the paper. The "Meta-Informed" strategy, which drives the best results, requires providing the LLM with "high-performing configurations" (theta*) from past tasks. The authors state these ground-truth solutions were obtained by "running extensive hyperparameter search." 
In real-world scenarios where facing a new ML problem,  the access of high-quality examples are not practical. 
To use this method, a user must have already invested the time and resources to collecting highly-related and generalizable other tasks/datasets and and get theta* for each of them using exhausting search.
It is a solution that requires you to have already solved the problem many times over, while such scenario may not need the proposed method.

### Questions
Please refer to Questions

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the use of Large Language Models (LLMs) as in-context meta-learners for the Combined Algorithm Selection and Hyperparameter optimization (CASH) problem. The authors propose converting tabular datasets into interpretable textual metadata, which is then fed into an LLM to recommend a complete ensemble of models and their hyperparameters in a single pass. They evaluate two strategies: a "zero-shot" mode relying only on the LLM's pretrained knowledge, and a "meta-informed" mode augmented with in-context examples of high-performing configurations from past tasks. A synthetic ridge experiment demonstrates scale-dependent emergence of in-context meta-learning (with Qwen2.5 variants), and real-world tests on 22 Kaggle tabular challenges show Meta-Informed outperforming several HEBO-based HPO baselines at comparable training budgets using private leaderboard percentile rank.

### Strengths
1. The paper introduces a compelling "search-free" paradigm for tackling the CASH problem. Unlike traditional AutoML methods that rely on expensive, iterative search (e.g., Bayesian optimization, evolutionary algorithms), this approach generates a strong set of candidate models in a single forward pass. This has significant practical implications for efficiency and accessibility.
2. The synthetic Ridge regression experiment (Section 4) is a key strength. By using Random Matrix Theory to derive a closed-form, analytic test error, the authors cleverly isolate and validate the LLM's in-context meta-learning capability without the confounding variable of cross-validation noise
3. The evaluation on 22 real-world Kaggle datasets is robust. The choice of private leaderboard percentile rank ($p_{rank}$) as a metric is excellent.

### Weaknesses
1. Assumption of access to high-quality “Context Blends” from prior tasks to populate Meta-Informed prompts may be unrealistic in cold-start or non-tabular settings, and may partly bake strong expert/HEBO search into the context rather than purely demonstrating generalization.
2. (my key concern) The paper claims to address the CASH problem, but it has limited model family coverage (CatBoost, LightGBM, XGBoost, scikit-learn MLP), constraining claims of generality across CASH, omitting linear models, tree ensembles beyond GBMs, kNN, and modern deep tabular architectures.
3. There is a growing literature on recommending models in an unsupervised way, which is completely ignored. For example,
[1] Guha, Neel, et al. "Smoothie: Label free language model routing." Advances in Neural Information Processing Systems 37 (2024): 127645-127672.
[2] Jeong, Daniel P., Zachary C. Lipton, and Pradeep Ravikumar. "Llm-select: Feature selection with large language models." arXiv preprint arXiv:2407.02694 (2024).
[3] https://openreview.net/pdf?id=gWi4ZcPQRl
[4] Jiang, Chumeng, et al. "Beyond Utility: Evaluating LLM as Recommender." Proceedings of the ACM on Web Conference 2025. 2025.

### Questions
1. In Fig. 1, it is not clear why the graph for log mean goes down with increasing k. This behavior means that the true configuration is actually closer to the mean of the configurations of other tasks. Is there any relation between the k tasks (in context) and the task in hand? From the problem setting, I would not have expected any smooth pattern for the log mean curve. 
2. In Fig. 1, except for the Qwen2.5 7B model, all other curves look almost monotonically declining. Any reason for the distinct behavior of a smaller model?
3. How sensitive are results to the exact metadata schema, and which fields matter most; can an ablation quantify the marginal utility of each metadata component and the effect of richer metafeatures (e.g., simple statistics per feature group, target skew/kurtosis, leakage checks)?
4. How well does the approach extend beyond tabular tasks and the four model families used here, and what breaks when adding neural tabular architectures or broader pipelines (e.g., preprocessing, feature engineering, and metric-specific loss surrogates)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates an innovative AutoML paradigm by employing Large Language Models (LLMs) as "In-Context Meta-Learners" to address the computationally expensive and expert-reliant Combined Algorithm Selection and Hyperparameter optimization (CASH) problem. The authors propose that by converting datasets into metadata and prompting an LLM, the model can recommend high-quality configurations in a "one-shot" manner, bypassing traditional iterative search. The study (utilizing both Zero-Shot and Meta-Informed strategies) demonstrates that LLMs, particularly large-scale ones, can not only leverage pre-trained knowledge but also perform effective meta-learning from in-context "support examples," achieving performance superior to traditional HPO baselines under a low-budget setting.

### Strengths
1. **Novelty of Approach:** The paper introduces a new paradigm for the CASH problem, treating the LLM as an in-context meta-learner. This "one-shot" recommendation (rather than iterative search) is an important and insightful exploration in contrast to traditional HPO and AutoML methods (e.g., Bayesian Optimization).

2. **Significant Efficiency and Performance:** Under a fixed low-budget (10 model configurations), the Meta-Informed strategy's average performance across 22 Kaggle tasks significantly surpasses several strong baselines (including Random-Hyperopt and MaxUCB-Hyperopt). This demonstrates the method's strong practical value and efficiency advantage in "cold-start" scenarios with limited computational resources.

3. **Strong Evidence for In-Context Meta-Learning:** The paper clearly demonstrates the meta-learning capabilities of LLMs through two experimental dimensions. First, the synthetic experiment (Fig. 2) proves this capability is scale-dependent. Second, on real-world tasks, the Meta-Informed strategy's performance (Fig. 3) significantly exceeds the Context-Random baseline, providing compelling evidence that the LLM is performing effective reasoning and adaptation rather than merely replicating the context.

### Weaknesses
**1. Generality of Claims:** The paper's main conclusions are drawn from evaluations on Kaggle tabular tasks and a limited set of four (primarily tree-based) model families. This represents a significant simplification of the full CASH problem. The authors should discuss the challenges of extending this method to broader task domains (e.g., time series, NLP/CV) and more diverse model libraries (e.g., linear models, deep tabular models), as the generality of the current claims is questionable.

**2. Fairness of Evaluation:** The comparison of a one-shot generation of 10 configurations against the first 10 rounds of traditional HPO may be unfair. HPO methods (like HEBO) often require a cold-start phase, and a 10-round budget is likely insufficient to reflect their true performance potential.

**3. Attribution of Gains:** The source of the observed performance gains is unclear. Is the benefit derived from the "quality" of the LLM's recommendations, or merely from the "quantity" of ensembling 10 candidates? The authors should provide ablation studies (e.g., comparing 1, 5, or 20 candidates). Furthermore, the method relies heavily on high-quality metadata, but its robustness in the presence of noisy or incomplete metadata—a common real-world scenario—is not tested.

**4. Lack of Iterative LLM Baselines:** The comparison is currently limited to traditional HPO (HEBO/MaxUCB) and Context-Random. The authors may argue that their "one-shot CASH" setting is distinct from iterative HPO, but the paper lacks a direct comparison against representative *iterative LLM4HPO* methods (e.g., LLAMBO [1], AgentHPO [2], AutoML-Agent [3]). We believe that **while the task settings (CASH vs. HPO) differ slightly, these methods represent the SOTA for LLMs in optimization. A comparison is essential to highlight the true value and boundaries of this work's one-shot meta-learning approach.**

**5. Potential Pre-training Contamination:** A critical concern is overlooked: the public Kaggle tasks used for evaluation are very likely to have been included in the LLMs' pre-training corpora. If so, the model's strong performance (especially in the Zero-Shot mode) may not stem from novel "meta-learning reasoning" but rather from "memory retrieval" of known solutions. The authors must address or conduct experiments (e.g., using private datasets) to rule out this possibility.

**References:**

> Liu, Tennison, et al. "Large Language Models to Enhance Bayesian Optimization." ICLR 2024
>
> Liu, Siyi, et al. "AgentHPO: Large language model agent for hyper-parameter optimization." CPAL 2025
>
> Trirat, Patara, et al. "AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML." ICML 2025

### Questions
None

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
The paper proposes using LLMs as in-context meta-learners to address the CASH problem by prompting an LLM with human-interpretable metadata for each dataset to directly recommend model families and full hyperparameter configurations. It studies two prompting regimes: a Zero-Shot mode that relies only on target-task metadata, and a Meta-Informed mode that augments the prompt with pairs of prior tasks' metadata and their well-performing configurations, thereby enabling cross-task generalization without iterative validation feedback. The system instructs the LLM to output a structured (JSON) ensemble of candidate models drawn from a predefined, family-specific grid (CatBoost, LightGBM, XGBoost, MLP), with lightweight post-processing to project any out-of-grid values, and uses these recommendations in a downstream training and blending pipeline. Overall, the results indicate that LLMs can leverage task metadata to make competitive, search-free recommendations, with the Meta-Informed strategy improving over Zero-Shot, suggesting a practical role for LLMs as efficient assistants within AutoML workflows.

### Strengths
1. The proposed method is simple and easy to understand, and as shown in the Experiments section, it demonstrates strong performance compared to other baselines for the CASH problem.

2. The synthetic ridge regression experiment appears reasonable and serves as an effective validation of the motivation behind the proposed methodology.

3. The authors evaluated the proposed method on a sufficiently large dataset -- 22 Kaggle tabular challenges spanning both regression and classification -- which provides a solid empirical basis for the study.

### Weaknesses
1. The paper appears to lack novelty. The proposed *Meta-informed* approach seems to be a straightforward application of an in-context learning method to the CASH problem. It would be helpful to more clearly articulate what distinguishes this approach from existing *zero-shot* or *in-context learning (ICL)* paradigms.

2. The paper does not sufficiently explain the rationale, intuition, or assumptions underlying why the proposed method works. For example, examining the model’s internal mechanisms from an interpretability perspective could offer valuable insights, as could an analysis from a data perspective. The absence of such deeper analyses weakens the overall contribution.

3. In addition, the paper does not address well-known phenomena in ICL research, such as *example selection* or *order sensitivity*. It would strengthen the work to examine whether these phenomena also arise in the context considered by the paper and to discuss how they can be effectively handled in this task setting.

4. The methodology involves multiple components -- e.g., elements in the task metadata, ensemble models, and the defined search space -- but lacks an ablation study to evaluate the importance of each. Such an analysis would help clarify which components are most critical to the method’s effectiveness.

### Questions
1. Have you tried fine-tuning the LLMs using the same data split? It would be great if you could provide the corresponding experimental results.

2. It is well known that ICL is sensitive to factors such as the type and order of in-context examples. Conducting additional experiments to analyze this sensitivity would strengthen the work.

3. Could you elaborate further on the logistic regression described in line 192?

4. In Figure 2, the 32B model appears to underperform compared to the 7B and 14B models. Could you explain why this occurs? To maintain consistency with the claim in Section 4, larger models within the same family should exhibit better performance, but this trend is not observed.

5. How did you define the search space used as input to the LLMs?

6. I would suggest adopting a constrained decoding strategy to prevent malformed output generation.

7. If there is a recent state-of-the-art (SOTA) method for the CASH problem, please mention it and compare your approach against it.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper frames CASH (algorithm + hyperparameters) as in-context meta-learning via prompting. Each dataset is summarized into a compact metadata block and an LLM is prompted to output model configs (chosen from a small family) in a single pass. Two modes: (i) Zero-Shot (only current task metadata) and (ii) Meta-Informed (prepend a few (metadata, best-config) exemplars from prior tasks) are proposed to get the final output. The method is compared against 4 other baseline methods and generally demonstrates improvements.

### Strengths
S1. Simple and practical setup: A Clean metadata template makes outputs trainable and reproducible.

S2. Single-pass: Gets a set of candidates at once without multiple prompting, easy to utilize.

S3. The performance on diverse types of tasks is promising.

### Weaknesses
W1. Mechanism ambiguity (meta-learning vs. knowledge bank: The evidence doesn’t separate genuine meta-learning from retrieval-ish pattern-matching. The paper just uses a structured task description as the context and lets the LLM figure out the details of the best hyperparameters. This just looks like an LLM being used as a knowledge bank, and a more comprehensive test requires an experiment to determine whether the LLM learns mappings rather than parrots "best" configs from memory.

W2. Narrow model/space coverage: The approach is restricted to four families and a fixed grid, which aids engineering but limits generality. The authors themselves note that extending coverage is essential and miss ablations on family set, grid granularity, K (ensemble size), and metadata richness make it hard to assess robustness.

### Questions
Refer Weakness.

### Soundness
2

### Presentation
2

### Contribution
1
