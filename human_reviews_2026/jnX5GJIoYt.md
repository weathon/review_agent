# EIP: Weighted Ranking of LLMs by Quantifying Question Difficulty

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Benchmarks establish a standardized evaluation framework to systematically assess the performance of large language models (LLMs), facilitating objective comparisons and driving advancements in the field. However, existing benchmarks fail to differentiate question difficulty, limiting their ability to effectively distinguish models' capabilities. To address this limitation, we propose Empirical Interaction Propagation (EIP), a novel framework designed to quantify both question difficulty and model competency. EIP introduces difficulty as the primary criterion for differentiation, enabling a more fine-grained evaluation of LLM capabilities. EIP's core mechanism facilitates bidirectional score propagation between models and questions. The core intuition of EIP is that a model earns a competency score when it correctly answers a question, while a question's difficulty score increases when it challenges a model. Using this framework, we evaluate 30 models on 35,550 questions across multiple domains. EIP achieves 90\% agreement with human judgments and consistently outperforms strong baselines such as IRT. It also exhibits strong stability, fast convergence, and high computational efficiency, making it a practical solution for large-scale, difficulty-aware LLM evaluation. Code is available at https://github.com/Leozz04/EIP.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces RankLLM, a difficulty-aware framework for evaluating Large Language Models (LLMs) by jointly estimating question difficulty and model competency. The core mechanism is a bidirectional score propagation process over a directed bipartite graph of models and questions, using model successes and failures to iteratively reinforce difficulty and competency scores. The approach is non-parametric, scalable, and designed for use with large evaluation pools. Empirical studies are conducted on 30 LLMs across 35,550 questions spanning several benchmarks. The authors report that RankLLM exhibits strong alignment with human judgments of question difficulty, outperforms standard baselines like Item Response Theory (IRT), converges rapidly, and scales to large datasets. The framework is further analyzed for stability, extensibility, and computational cost, with robustness claims under dataset/model perturbations.

### Strengths
Principled, mathematically sound methodology: The central mechanism relies on a well-formulated ergodic Markov chain on a bipartite graph, with mutually reinforced score propagation between questions and models.
Comprehensive empirical validation: Evaluation extends to 30 models over 35,550 questions. Large-scale empirical studies support many intuitive claims, clearly presented and visualized.
Human judgment: This is a great source to measure the goodness of fit for item difficulty.
Open and reproducible claims: Claims are signposted as non-parametric and hyperparameter-light, and computational infrastructure (including licensing) is disclosed.

### Weaknesses
No major weakness, however see my questions

### Questions
1. Are you implementing IRT yourselves or using a well-established package? If you implement IRT yourselves, did you carefully validate your output with a widely validated package (say, an R package)? In Table 3, it is reported that 1PL IRT takes ~30 mins to fit. I happened to work on IRT for AI evaluation; my 1PL IRT implementation takes ~30 seconds to fit on ~180 LLMs and ~80,000 questions (on an A100 GPU, though).

2. Given that different benchmarks have different measurement objectives, is there a reason you fit all benchmarks together?

3. IRT is well established in psychology; Is your method thoroughly studied in any other fields? Does it have an original reference in psychology or measurement science, or other fields? If so, is there previous work that reaches the same conclusion as you do (i.e., outperforming IRT)? I have heard about Elo-rating systems involving test takers and items (which can be shown to be equivalent to 1PL IRT in theory), but I am not familiar with the details.

4. IRT enables computerized adaptive testing (CAT), which can reduce evaluation compute for new LLMs. To create a leaderboard, the computational cost of querying benchmark questions to LLMs seems much larger than the difficulty/competence update step. Can your method similarly support computerized adaptive testing?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The core of this paper lies in addressing a major flaw in current evaluation methods for Large Language Models (LLMs): existing benchmarks (such as MMLU and MATH) typically only calculate overall accuracy, treating all questions equally without considering their actual difficulty levels. This approach may lead to distorted evaluation results. For instance, a model that answers many simple questions correctly might appear stronger than one that answers fewer but extremely difficult questions correctly.

To solve this problem, the authors propose a new evaluation framework called RankLLM.

The core idea of RankLLM is to move beyond evaluating models or questions in isolation; instead, it quantifies both "model capability" and "question difficulty" simultaneously. It is based on an intuitive interaction logic:

- If a model can correctly answer a widely recognized difficult question, its capability score should be higher.
- If a question is answered incorrectly even by widely recognized strong models, its difficulty score should be higher.

Based on this logic, RankLLM constructs a "bipartite graph" between models and questions. It then uses an iterative algorithm called "bidirectional score propagation" to calculate the final scores. This process iterates continuously until the model capability scores and question difficulty scores reach a stable equilibrium.

In conclusion, I think this is a simple but interesting work.

### Strengths
By introducing "question difficulty" as a core variable, the RankLLM framework significantly outperforms traditional evaluation models that prioritize "accuracy above all else".

1. Finer Differentiation (Beyond Accuracy)
In traditional methods, models receive the same score for answering a simple question and a difficult question correctly. RankLLM breaks this limitation by assigning a higher "capability score" to models that solve difficult questions.

- As shown in the simulation experiments in Section 4 of the paper, when two models have the same overall accuracy, RankLLM can accurately identify the model that solves more difficult questions (M1 > M2), while traditional accuracy-based methods fail to make such a distinction.

2. High Alignment with Human Judgment (Strong Consistency)
The "gold standard" for an evaluation framework is whether it aligns with the judgments of human experts.

- Section 3.3 of the paper demonstrates that the ranking of question difficulty generated by RankLLM matches the consensus of human experts by up to 90%. This is far higher than that of mainstream statistical models such as Item Response Theory (IRT), confirming the validity and reliability of its evaluation results.

3. Efficiency, Scalability, and Robustness (Engineering Practicality)
- **High Efficiency**: On a dataset with 35,000 questions and 30 models, the algorithm converges in only 0.006 seconds (Section 3.4). This allows it to be easily deployed in large-scale leaderboards that require frequent updates.
- **Strong Robustness**: Experiments prove that even if a large number of models (up to 15) are randomly removed from the evaluation pool, the relative rankings of the remaining models and the difficulty rankings of questions remain highly stable (Section 3.5, Table 5).

4. Automated Difficulty Quantification (No Manual Annotation Required)
Traditional benchmarks (such as MATH) require human experts to pre-label subjective difficulty levels (e.g., Level 1~5).

- RankLLM, by contrast, is fully automated. It "emerges" a definition of difficulty from the performance of the model group, without relying on any manual prior knowledge, making it more objective.

### Weaknesses
1. The core of RankLLM lies in the "relative" relationship between models and questions. The difficulty of a question is defined "relative to" the pool of models participating in the evaluation. This gives rise to an issue: if the model pool itself is biased, the resulting difficulty scores and capability rankings may also be biased.
- The paper itself demonstrates this in Figures 7 and 8: if only a "large-scale" model pool is used, 58% of the questions will be classified as "excessively easy"; if only a "small-scale" model pool is used, 30% of the questions will instead be classified as "impossible".
- While the paper proves that "removing" models from the current pool is robust (Table 5), it does not explore what would happen to the overall ranking if a completely new "super model"—with capabilities far exceeding all existing models in the pool is added. Theoretically, the addition of this new model would "lower" the difficulty scores of many previously "difficult questions", potentially causing drastic changes to the entire ranking list, especially for models in the upper-middle tier.
2. This framework ultimately computes a single capability score ($\pi_m$) for each model and a single difficulty score ($\pi_q$) for each question, which represents an oversimplification of the actual scenario.  

- Model capabilities are inherently multi-dimensional. For example, a model might demonstrate exceptional performance in "mathematical reasoning" while showing significant weaknesses in "historical knowledge". RankLLM tends to "average out" these varied capabilities, yielding a moderate overall score. This averaging effect can lead to a "well-rounded" model and a "specialized (or lopsided) expert" model receiving similar overall rankings, ultimately masking the true capability profile of each model.  
- Question difficulty also exhibits multi-dimensional characteristics. A single question may simultaneously demand proficiency in "knowledge retrieval", "multi-step reasoning", and "spatial imagination". RankLLM lacks the ability to decouple these composite difficulty components, making it unable to accurately capture the nuanced difficulty structure of such questions.

### Questions
1. Regarding the dependence on the "evaluation model pool": The abstract mentions that "using a diverse (mixed-scale) model pool yields the most accurate difficulty estimates." I am curious about how the authors define a "sufficiently diverse" pool. For a new user looking to adopt this framework, how many models—and of what types—would be required to obtain a reliable (stable) difficulty ranking?  

2. Regarding the 90% "human agreement": This is an impressive figure (mentioned in Section 3.3). I would like to delve into the details of this experiment: How were the "human experts" selected? How was "agreement" specifically calculated? Additionally, what was the level of "agreement" among the human experts themselves (i.e., their internal consensus)?  

3. Regarding the limitation of "single-dimensional" capability: This method seems to generate a single "capability score" ($\pi_m$) for each model. In practice, however, a model might excel in mathematics but perform poorly in writing. I am interested to know whether the authors investigated this "unbalanced capability" phenomenon. Alternatively, can the framework be extended to generate a multi-dimensional "capability radar chart" (e.g., a mathematics difficulty score and a writing difficulty score) instead of just an overall ranking?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The core contribution of this paper is the proposal and validation of an evaluation framework named RankLLM. This framework utilizes an iterative algorithm to jointly estimate question difficulty and model competency. 
This process is modeled as a bidirectional score propagation on a bipartite graph connecting models and questions. The introduction of a damping factor ensures the algorithm converges to a unique stationary distribution, yielding final competency scores for models and difficulty scores for questions. The authors validate the framework's effectiveness, robustness, and scalability through large-scale experiments on 6 popular benchmarks, involving 30 models and over 35,000 questions.

### Strengths
1. The proposed method of jointly modeling question difficulty and model competency via score propagation on a bipartite graph is highly novel and intuitive. Treating question difficulty and model competency as interdependent and co-evolving variables is more dynamic and sound than traditional static methods based on accuracy or IRT.
2. The paper provides extensive validation across multiple mainstream benchmarks and 30 models. The results are convincing, particularly highlighting the method's outstanding performance in aligning with human judgment and its computational efficiency.
3. The study not only proposes a new method but also uses it to reveal several empirical findings that are insightful and of practical value to the LLM field.

### Weaknesses
1. The paper proves that the algorithm converges for any α ∈ (0, 1) and shows its effect on convergence speed in the appendix. However, it does not thoroughly investigate the impact of the choice of α on the final scores and rankings for model competency and question difficulty. It is unclear whether different values of α could lead to changes in model rankings. The paper lacks a discussion on a principled method for selecting an optimal α or a sensitivity analysis of the final results to this parameter.
2. If a specific model family (e.g., the Llama series) shares a common 'blind spot' for a certain type of reasoning, questions targeting this weakness could be erroneously labeled as 'extremely difficult.' Consequently, when a new model with a different architecture correctly answers these questions, its resulting competency score boost might be constrained by the initial competency landscape dominated by the biased model family. 
3. While the method is innovative, the definition of 'difficulty' fundamentally remains dependent on model performance. The essence of difficulty is still derived from model failure rates, and the paper does not introduce external criteria or a theoretical framework to independently validate the soundness of this difficulty definition.

### Questions
please see Weaknesses

### Soundness
2

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
3

### Summary
This paper proposes RankLLM, a difficulty-aware evaluation framework that jointly estimates question difficulty and model competency via a damped random walk on a model–question bipartite graph, yielding a unique stationary distribution for both scores. The method operationalizes difficulty through model failures and propagates scores bidirectionally between questions and models. Experiments span 30 models over 35,550 questions from six benchmarks, with reports of strong alignment with human difficulty judgments and robustness analyses.

### Strengths
1. The motivation is clear and timely: moving beyond flat accuracy to a difficulty-sensitive ranking that better separates closely matched models. 

2. The method is simple yet principled—formulated as a damped Markov chain with a uniqueness/convergence guarantee—and scales to large pools and datasets.
 
3. The paper is well written and easy to follow, with a clean derivation and a clear pipeline figure that makes inputs, transitions, and stopping criteria explicit.

### Weaknesses
1. Rankings are sensitive to who is in the pool: adding many weak models inflates failure mass, makes those items look “hard,” and artificially boosts any model that solves them. This causes rank shifts without any change in per-item accuracy and breaking cross-study comparability.
2. The score scale itself changes with the participating models and the domain mix, so adding or swapping peers alters the baseline. There is no common unit across model clusters or data clusters. Results cannot be compared across studies or over time.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2
