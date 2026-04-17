# Who Routes the Router: Rethinking the Evaluation of LLM Routing Systems

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
The growing ecosystem of Large Language Models (LLMs) with diverse capabilities and costs has motivated the need for LLM routing systems that dynamically select the most appropriate model for each query. Evaluating these routing systems is important yet inherently challenging due to the complex interplay of multiple factors: the selection of representative input queries, the composition of the model pool, and the definition of comprehensive evaluation metrics for optimal routing decisions. Through extensive analysis of existing benchmarks, we identify critical limitations that may lead to incomplete results and/or misleading conclusions about router performance: 
(1) limited task diversity, (2) imbalanced model pools, and (3) oversimplified evaluation methodologies. To address these limitations, we propose a novel evaluation framework that incorporates diverse task distributions (33,337 queries across 68 categories), a balanced model pool of 85 models with complementary model strengths, and multi-faceted metrics that reflect real-world deployment scenarios. 
We implement this framework as an open-source benchmark, enabling researchers to rigorously assess routing strategies under realistic conditions. The code and dataset are shared anonymously at: https://anonymous.4open.science/r/rethinking-routing-evaluation-DE30

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors present flaws (redundant tasks, redundant and dominant models, lack of multi-faceted evaluation) with current router evaluation practices. They then present a remastered evaluation of the methods in embedLLM and some binary routing methods.

### Strengths
**Each of the identified issues with LLM evaluation are (to me) valid problems with how we measure router performance** 

* Many Routing datasets are simple blends of general reasoning, knowledge and math benchmarks which leaves out many important applications.
* Often routers are trained on datasets where one model can dominate. For example in CARROT, o3 mini is by far the best model at most tasks.
* As far as I am aware, routing benchmarks have not looked into issues such as OOD generalization.

**The remastered llm evaluation addresses some of these issues**

* The authors subsample EmbedLLM tasks to highlight those that benefit from non-generalist models (the authors also remove duplicate LLM queries)
* The authors inject pseudo-specialist models
* The authors hold out certain categories of query to test for ood generalization

### Weaknesses
* The set of ``remastered" evaluations feels limited. A re-analysis of EmbedLLM is completed, as is a redone analysis of binary routing. It would be interesting to see how other predictive routing evaluations such as those in CARROT or routerbench are effected by the identified flaws. CARROT in particular has its own data set, which may have more variety than that in embedLLM.
* Certain ``solutions" to the proposed flaws feel limited. For example, the injected specialist models are just normal models with artificially boosted scores, not models trained for specific domains that are then incorporated in the evaluation.
* Most critically, its not clear to me how much the proposed changes affect evaluation results. The main text of the paper should include an analysis of how each the changes to embedLLM (question diversity, expert ) effect the performance of each routing method (and in particular the ordering of each method).

### Questions
* How does the analysis extend to other predictive multi llm routers?
* How do the proposed changes effect the original embedLLM results? Which of the faults is most important to reduce?

### Soundness
3

### Presentation
3

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
This work studies an interesting problem: how to choose the proper LLM to use given a user query. Although there has been a line of research and analysis of existing benchmarks, this paper identifies some critical limitations that may lead to incomplete results and misleading conclusions, such as the existing benchmarks have limited task coverage or skewed on some specific tasks, and the model pools are too large and imbalanced. To address this issue, this work proposes a new evaluation framework. Empirical results have been provided to showcase the limitations of existing evaluation strategies and the advantages of the new benchmarks proposed in this work.

### Strengths
1. This work is overall well written and easy to follow.
2. The empirical results are quite complete and coherent. 
3. This work studies an important problem: how to fairly evaluate the existing LLM routers, and I think this line of research is important.

### Weaknesses
I do not find any major weaknesses of this work, while I am not very familiar with existing literature so I may refer to other reviewers' opinions.

1. I feel it might be better to also consider the influence of the embedding models on the final performance of the routers. 
2. It is also quite surprising that the kNN-based methods outperform the trained MLPs, especially on OOD tasks. I suspect this is because the retraining dataset is relatively limited, preventing the MLPs from fully converging. To verify this hypothesis, it might be useful to include a simple parametric baseline such as linear regression as a router. If the linear model were to outperform the MLPs, it would suggest that the training dataset is indeed insufficient for proper convergence. Could you please elaborate on this issue?
3. The pseudo model is not very realistic, and it is better to use some fine-tuned specific models instead for each small task.

### Questions
1. I had a hard time understanding Figure 5 (a), as the X-axis is the model parameter. Can you explain that to me? And how model parameters affect your models' accuracy, as shown in the curves.

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
This paper studies the limitation of previous LLM routing benchmarks from three perspectives: (1) limited task diversity, (2) imbalanced model pools, and (3) oversimplified evaluation methodologies. Given that, this paper proposes a novel evaluation framework that incorporates diverse task distributions, a balanced model pool of 85 models with complementary model strengths, and multi-faceted metrics.

### Strengths
1. How to properly evaluate LLM routing approaches is an important topic in current research on LLM serving systems.
2. This paper proposes a novel benchmark with 33k queries across 68 categories, which serves as a concrete foundation for effective LLM evaluation.
3. This paper illustrates the technical developments with sufficient justification.

### Weaknesses
1. Some advanced routing work is neither compared nor discussed. For example,
    1. Ding, Dujian, et al. "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute." Forty-second International Conference on Machine Learning.
2. This paper primarily leverages binary label & BARTscore as the quality metrics for LLM responses, which seems limited for open-ended conversation — the mainstream LLM service scenarios.
3. In the evaluation section, the performance results are primarily reported on Llama2 models (fig 3 & 6) which are often considered as outdated models given the presence of Llama3 herd and Llama4 family.

### Questions
1. Majority of the evaluation examples are just one-turn query. However, the daily LLM usage typically happens with multi-turn conversation. Can the proposed benchmark to evaluate LLM routing performance in the multi-turn conversation scenarios?

### Soundness
3

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
The paper addresses fundamental flaws in current benchmarks for evaluating systems that dynamically route queries to the most suitable large language model (LLM). Existing frameworks often suffer from limited task diversity, imbalanced model pools, and oversimplified metrics, leading to misleading conclusions about router performance. To overcome these issues, the authors propose RouterBench+, a comprehensive evaluation framework featuring (1) a diverse and realistic task distribution of 33,337 queries across 68 categories, (2) a balanced pool of 85 models with complementary strengths, and (3) modified metrics that better capture the cost-performance trade-offs and out-of-distribution robustness.

### Strengths
1. The paper accurately identifies important flaws in existing router evaluation pipelines. Lack of task diversity, dominance by a single model, and poor assessment of OOD performance are all major issues that make existing evaluation pipelines unsuitable for real-world scenarios.

2. The solutions are clearly and systematically presented, with experiments and plots illustrating their efficacy, and overall, the paper is easy to follow.

### Weaknesses
1. The removal of duplicate queries leads to very small improvements for learning based routers and causes a drop in performance in for clustering-based routers. Thus, the claim in lines 268-269 that "duplicate queries with conflicting labels can mislead routers" is not well substantiated by the results.

2. The role of the pseuod-specialist models in Section 4.3 is not clear. If they are not meant for deployment, how does adding them provide any benefit? If a router is selected based on its performance in the presence of pseuod-specialist models, but does not have access to those models when it is deployed, how will it be able to replicate that performance?

3. It is not clear how the binary routing evaluation paradigm in Fig 3 is helpful in evaluating routers in multi-LLM settings.

4. It is not clear how the benchmark incorporates cost-performance trade-offs, latency constraints and reliability, as claimed in line 420

### Questions
1. Please give some examples of common-sense and domain-specific tasks to better illustrate the difference between the two.

2. How is the generalist model and the set $\mathcal{M}_{\text{non-gen}} $ chosen from a given set of models? 

3. The expression in line 237 suggests that the score is the difference between specialist and generalist accuracy but the caption of figure 2 says it is the difference between the specialist and the heuristic router. Which is it?

4. What is the dimensionality of the embeddings $\mathbf{e}_i$, $\mathbf{e}_j$ in line 263? A high dimensionality may lead to inconsistent results due to the curse of dimensionality.

5. How will sub-sampling tasks in EmbedLLM help (Section 5.1) if the dataset has very few specialist tasks to begin with?

### Soundness
2

### Presentation
3

### Contribution
2
