# DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
The performance of an LLM depends heavily on how well the training data matches the downstream evaluation task. However, in many practical settings, we typically do not know the data in the evaluation task (e.g., conversations between a chatbot and users are end-to-end encrypted). We refer to such tasks as unseen evaluation tasks. We can only deploy the LLM on these unseen evaluation tasks to gather multiple rounds of feedback on how well the model performs (e.g., gathering user ratings from a chatbot). In addition, this feedback can be noisy. How can we exploit such noisy feedback efficiently to optimize the LLM training data-mixture? Our paper presents DUET, a novel global-to-local algorithm that optimizes training data mixtures by interleaving data selection with Bayesian optimization to exploit coarse and noisy feedback from a downstream evaluation task. DUET is flexible enough to incorporate different data selection methods, each with different performance-compute tradeoffs. By analyzing DUET's cumulative regret, we theoretically show that DUET converges to the optimal training data mixture even without any fine-grained data information from an unseen task. Finally, our experiments across a variety of language tasks demonstrate that DUET attains substantial performance improvements over existing data selection and mixing methods in the unseen-task setting. Our library, which is flexible enough to optimize different LLM training ingredients, can be found at  https://github.com/chenzhiliang94/BO-for-LLM.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Optimizing the deployment of the model without an instance-wise loss function but only at a model-wise loss function is an important problem. The paper formalizes the setting, focuses on optimizing only data selection, and shows that this problem can be divided into two steps: optimizing a data mixture ratio and optimizing a dataset meeting this ratio. The author introduces DUET to use BO for the first optimization problem and IF-driven estimators for the second problem. Cumulative regret results for DUET is provided. DUET performs better than other baselines on LLM benchmarks with small LMs (<=8B).

### Strengths
- The problem formulation is clear and the writing is good.
- The introduction of the two components of DUET to solve the problem is very natural.
- Theoretical results on cumulative regrets are provided.

### Weaknesses
- The theoretical results don't seem very novel and more like applying previous results in the literature in a new setting.

### Questions
- Can we extend the set of optimization variables to be more than just the data mixing ratio (e.g., optimizing a small set of parameters of the LLM) or is the setting too noisy to reliably do so? I don't see an explicit dependence on $n$, the dimension of $r$ in Theorem 4.1.
- Can we get Theorem 4.1 without an IF-driven estimator? If so, having that somewhere would help showing how IF helps DUET theoretically?
- Does the unseen evaluation task setting mean that you train on all datasets except for that task? Also, is it true the evaluation is just one feedback (good/bad model) at the model level without access to the exact evaluation loss? These details should be clearer. You should not point to LLM-harness and be done with it.

### Soundness
4

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
2

### Summary
This paper presents DUET, a method that combines data mixture optimization and data selection, for LLM fine tuning. The authors propose a two stage optimization, in which the inner problem takes a mixture ratio as input and samples data satisfying the mixture ratio via data selection, and the outer problem solves the mixture ratio taking the optimization of the inner problem as a blackbox. In particular, the outer problem is solved by Bayesian optimization.

The authors conduct extensive experiments to demonstrate the effectiveness of the proposed method.

### Strengths
1. The paper is generally well-written, clear, and easy to follow. The proposed method is elegant and easy to implement.

2. The empirical results strongly support the effectiveness of the proposed method.

### Weaknesses
1. The proposed method is a simple combination of data selection and Bayesian optimization.

2. The reviewers have some questions about the empirical results.

### Questions
1. The authors highlight that the experiments are done with sampling size of k = 1 in each BO iteration. However, the results seem too good to be true, given that the optimization is done over a degree-8 simplex. Note that almost all experiments in Figure 4 show >10-point improvements within 10 total samples. Moreover, in 4(a), there are >10-point improvements even with only 1 sample, and if the reviewer understands correctly how Bayesian optimization works, with only 1 sample, Bayesian optimization cannot provide useful suggestions. The reviewer may be missing something here. Could the authors help clarify?

2. Related to 1 above, could the authors provide the mixture found by the proposed methods for each iteration?

3. It seems that this paper mainly applies Bayesian optimization to the data mixture optimization problem (along with data selection), which limits the technical contributions from this paper. Are there technical difficulties the authors could highlight? In particular, are there modifications needed for Bayesian optimization in order to address challenges from "noisy feedback from unseen evaluation tasks"?

### Soundness
2

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
This paper proposes DUET, a novel global-to-local data mixture optimization algorithm for LLM finetuning, especially when the target evaluation task is unseen and with noisy performance feedback. Theoretically, the authors applies Bayesian Optimization (BO) for global (inter-domain) mixture rate search, combined with a local (inner-domain) sample-level search by influence function based methods. Empirically, the proposed method demonstrates an improvements on the performance on in-domain and out-of-domain tasks, compared to several existing baselines.

### Strengths
1. The paper targets at a critical and challenging problem of data mixture optimization, under a practical and real-world scenario, when the target task is unseen and with noisy feedbacks.
2. The paper proposes a bayesian optimization based method for data mixture search, which serves as a reliable estimation with coarse feedbacks from the target, black-box-like tasks.
3. The proposed DUET method obtains a significant performance gain, outperforming several existing baselines.

### Weaknesses
1. The computational cost and overhead have not been discussed. Both the retraining involved in the BO process and the influence function computation can be computational intensive, especially when the model gets scaled-up (e.g. 70B+). can you provide an analysis on computation and memory overhead caused by the BO and BO+IF methods?

2. To compute the influence scores (IF), do you require the validation set from the target distribution? if so, how do you ensure the IF scores are reliable under the unseen and noisy evaluation environment? Also, the data may not be available if the evaluation system is a black-box.

3. The effectiveness of BO depends on how well the Gaussian Process models the objective function.

### Questions
1. can you provide an analysis on computation and memory overhead caused by the BO and BO+IF methods?

2. How does the BO method scales by the number of data domain $k$? 

3. To compute the influence scores (IF), do you require the validation set from the target distribution? if so, how do you ensure the IF scores are reliable under the unseen and noisy evaluation environment? Also, the data may not be available if the evaluation system is a black-box.

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
The paper proposes DUET, a data selection method for optimizing the data mixture used in language model fine-tuning. DUET leverages feedback from downstream evaluation tasks to guide its selection process. The method combines Bayesian optimization with influence function to identify the optimal data composition. The authors validate DUET's effectiveness through comprehensive empirical evidence and theoretical analysis.

### Strengths
1. Optimizing the data mixture with "evaluation aware"  for language model fine-tuning is a practical and important field, primarily because the ideal data distribution is rarely known and language models are intrinsically multitask learners.
2. The authors propose a novel method to find an optimal data mixture, using Bayesian optimization for the outer optimization loop and influence functions for the inner optimization.
3. The authors support their approach with sound theoretical analysis and comprehensive experimental results.

### Weaknesses
1. The method is computationally intensive, requiring multiple separate LLM fine-tuning runs to gather feedback for the Bayesian Optimization loop.
2. The reliance on IF is questionable, as some of works point out IF for LLM is not very reliable [1]. 
2. The experiments are confined to fine-tuning. The approach's feasibility for large-scale pre-training, especially for computationally heavy estimators like IF, is not demonstrated. 

[1] Basu, Samyadeep, Philip Pope, and Soheil Feizi. "Influence functions in deep learning are fragile." arXiv preprint arXiv:2006.14651 (2020).

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2
