# Efficient Estimation of Kernel Surrogate Models for Task Attribution

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Modern AI agents such as large language models are trained on diverse tasks---translation, code generation, mathematical reasoning, and text prediction---simultaneously. A key question is to quantify how each individual training task influences performance on a target task, a problem we refer to as task attribution. The direct approach, leave-one-out retraining, measures the effect of removing each task, but is computationally infeasible at scale. An alternative approach that builds surrogate models to predict a target task's performance for any subset of training tasks has emerged in recent literature. Prior work focuses on linear surrogate models, which capture first-order relationships, but miss nonlinear interactions such as synergy, antagonism, or XOR-type effects. In this paper, we first consider a unified task weighting framework for analyzing task attribution methods, and show a new connection between linear surrogate models and influence functions through a second-order analysis. Then, we introduce kernel surrogate models, which more effectively represent second-order task interactions. To efficiently learn the kernel surrogate, we develop a gradient-based estimation procedure that leverages a first-order approximation of pretrained models; empirically, this yields accurate surrogate estimates with less than 2% relative error without repeated retraining. Experiments across multiple domains---including mathematical reasoning in transformers, in-context learning, and multi-objective reinforcement learning---demonstrate the effectiveness of kernel surrogate models. They achieve a 25% higher correlation with the leave-one-out ground truth than linear surrogates and influence-function baselines, enabling more accurate and scalable task attribution. When used for downstream task selection, kernel surrogate models further yield a 40% improvement in demonstration selection for in-context learning and multi-objective reinforcement learning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies the problem of task attribution: quantifying how each training task affects performance on a target task. The authors propose KernelSM, a kernel surrogate model over subset-indicator vectors trained via kernel ridge regression to capture higher-order task interactions. The method aims to address two limitations of existing approaches: the computational cost of leave-one-out evaluation and the inability of linear surrogate models to capture nonlinear interactions. Experiments on modular arithmetic, in-context learning, and multi-objective RL show that KernelSM achieves higher correlation with leave-one-out estimates and improves downstream task selection.

### Strengths
* Task attribution is an important and broadly applicable problem, as demonstrated by the diverse experimental settings.
* Modeling task interactions with kernel methods is a reasonable and well-motivated approach.

### Weaknesses
* In my opinion, the issue of capturing nonlinear interactions is relatively straightforward and has already been explored in feature or data-selection settings using linear surrogate methods such as LIME - KernelLIME. This limits the contribution, as the paper essentially replaces linear surrogates with kernel-based ones.
* There is no theoretical support for the gradient-based estimator obtained by linearizing the model in Section 3.3. It is surprising that this approach works well for highly nonlinear models such as LLMs. The authors should provide more intuition for why this estimator is effective and whether it generalizes to larger model sizes.
* Baselines: the paper doesn’t discuss and compare with several recent works beyond TRAK, such as [1, 2], which address related problems in data and task attribution.

[1] Bae, Juhan, et al. "Training data attribution via approximate unrolling." Advances in Neural Information Processing Systems 37 (2024): 66647-66686.
[2] Kreer, Philipp Alexander, et al. "Bayesian Influence Functions for Scalable Data Attribution." High-dimensional Learning Dynamics 2025.

### Questions
In tasks like in-context learning, the order of examples can substantially affect the final outcomes. Could the authors clarify how their method accounts for or mitigates this issue?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies "task attribution," which is a task to quantify the influence of individual training tasks on a model's performance for a given target task. 

The paper aims to become less computationally expensive compared to leave-one-out techniques while being more accurate than linear models which only capture first-order effects.

This work introduces the use of kernel surrogate models to capture nonlinear task interactions better. The authors develop a gradient-based estimation procedure for fitting these kernel models, leveraging first-order approximations to avoid repeated retraining. 

Finally, the proposed approach is validated on different tasks showing improved accuracy.

### Strengths
Computational efficiency (comparable to linear surrogate runtime) while being much more accurate (<1% relative error)

Clearly explain and support the fact about why linear surrogates work well only when the task does not have interaction (in section 3.1)

Simple and smart method to get rid of multiple retraining of the model in different input subsets.

### Weaknesses
Minor point: you didn't clearly define KernelSM. Although it is obvious that it stands for Kernel Surrogate Model.

The discussion in the results is not well organized and is confusing. For instance, table 3 is not referenced and discussed anywhere

Some components are not explained and well discussed (e.g. SAM and NSO) or they are in the appendix although it would be better to be in the main body

### Questions
There is a paper called "Register Always Matters" where they find how different genres of data in the pretraining affect the performance of the model on the test samples. I am wondering if one can use your method to validate their results? No need to answer if you don't know or etc.

### Soundness
2

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
The paper presents a framework for estimating how each training task impacts performance on a target task. By extending beyond linear models, to capture some of the non-linear interactions between tasks, significant empirical performance gains are found to emerge. This kernel based approach is validated with a range of domains and found to correlate more favourably with the ground truth LOO approach.

### Strengths
The manuscript is well written and clearly presented.

The approach presented in the paper is novel, well motivated, and timely.

A broad range of experiments were performed with a suitable set of baselines, and empirical performance of the proposed approach is consistently strong.

### Weaknesses
Motivations for the kernel hyperparameters could be more clearly presented.
 
Figures 3 and 4 appear to be lacking uncertainty estimates

### Questions
What motivated the choice of setting gamma to 10^-5? (section C4) 
If a sweep of hyperparameters was run, how sensitive are the empirical results to these values?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces KERNELSM, a kernel-based surrogate modeling approach for task attribution in multi-task learning. The method captures nonlinear relationships between training tasks and model performance, going beyond the linear assumptions of prior approaches. The paper provides (1) a theoretical framework connecting influence functions and linear surrogate models, clarifying their limitations; (2) an efficient gradient-based algorithm for estimating kernel surrogate models without retraining; and (3) comprehensive experiments across arithmetic reasoning, in-context learning, and multi-objective reinforcement learning, showing consistent improvements in attribution quality.

### Strengths
1. The paper is well-structured and clearly written. 
2. The theoretical discussion is solid; it connects influence functions and linear surrogate models, offering valuable insights into their relationship and the limitations of linear approximations.
3. The method seems to perform well against all baselines, and the evaluation tasks are comprehensive (modular arithmetic reasoning, in-context learning, multitask RL).

### Weaknesses
In Table 4, the comparison seems to exclude the task-level adapted versions of TRAK and TracIn, and the paper also does not specify their hyperparameter settings, which may reduce the credibility of the results reported in Table 3.

Although the evaluation covers a diverse and comprehensive set of tasks (including arithmetic reasoning, in-context learning, and multi-objective reinforcement learning), the actual number of training tasks in each setting remains relatively small (only 10–50).

### Questions
1. While Appendix C.4 provides an empirical comparison among polynomial and RBF kernels, could the authors offer more theoretical justification or intuitive reasoning for why the RBF kernel performs best in this method?

2. Is this method sensitive to the choice of hyperparameters (λ and γ)? Are there any guidelines or heuristics for selecting these hyperparameters in practice? I am asking this because some other data attribution methods (e.g., TRAK) have already had practical heuristics for hyperparameter selection. If KERNELSM is highly sensitive to λ and γ, the additional hyperparameter search could increase the computational cost of applying the method in practice.

### Soundness
2

### Presentation
3

### Contribution
3
