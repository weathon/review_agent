# Distributed Gradient Descent with Many Local Steps in Overparameterized Models

- Decision: Reject
- Scores: 5, 5, 6, 3

## Abstract
In distributed training of machine learning models, gradient descent with local iterative steps is a very popular method, variants of which are commonly known as Local-SGD or the Federated Averaging (FedAvg). In this method, gradient steps based on local datasets are taken independently in distributed compute nodes to update  the local models, which are then aggregated intermittently. Although the existing convergence analysis suggests that with heterogeneous data, FedAvg encounters quick performance degradation as the number of local steps increases, it is shown to work quite well in practice, especially in the distributed training of large language models. In this work we try to explain this good performance from a viewpoint of implicit bias in Local Gradient Descent (Local-GD) with a large number of local steps. In overparameterized regime, the gradient descent at each compute node would lead the model to a specific direction locally. We characterize the dynamics of the aggregated global model and compare it to the centralized model trained with all of the data in one place. In particular, we analyze the implicit bias of gradient descent on linear models, for both regression and classification tasks. Our analysis shows that the aggregated global model  converges exactly to the centralized model for regression tasks, and converges (in direction) to the same feasible set as centralized model  for classification tasks. We further propose a Modified Local-GD with a refined aggregation and theoretically show it converges to the centralized model in direction for linear classification. We empirically verified our theoretical findings in linear models and also conducted experiments on distributed fine-tuning of pretrained neural networks to further apply our theory.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper investigates the performance of Local Gradient Descent (Local-GD) in distributed training environments, particularly focusing on overparameterized models. It discusses the dynamics of Local-GD under conditions of high local iteration counts and heterogeneous data distributions, areas where traditional convergence analyses have been pessimistic about the performance of such approaches. The authors argue that despite these concerns, Local-GD performs well in practice and attribute this to the implicit bias inherent in gradient descent techniques.

### Strengths
1. The paper dives deep into the theoretical aspects of Local-GD for overparameterized linear models, providing an analytical framework that helps explain its effectiveness even with a high number of local steps.
2. The theoretical claims are backed by empirical evidence.

### Weaknesses
1. Viewing that the complex, non-linear models dominate modern machine learning applications, the focus on overparameterized linear models restrict the relavance of the results.
2. The **LocalUpdate** methods in this paper are based on gradient descent (GD), which may limit the applicability of the algorithms since stochastic gradient descent (SGD) are much more widely used in distributed and federated learning.
3. The theoretical analysis presents only convergence results, while the rates of convergence and sample complexity are unclear.

### Questions
1. The **LocalUpdate** methods seem to perform sufficiently many local steps until convergence (line 290). How does the numerical error incurred by each round of **LocalUpdate** affect the outer-loop convergence at the server side?
2. How to set the values of the parameters $L$ and $\eta$ and how do they affect the overall convergence?
3. In Lemma 2, the model $w_0^k$ converges to $\bar{w}_0^k$ *in direction*. Even if $w_0^k$ tends to be in line with $\bar{w}_0^k$, they can be far away from each other. How does this result make sense?
4. Algorithm 1 line 7: Should it be $w_i^{k+1}$?
5. Constraint in Eq. (13): $\bar{w} \rightarrow w$?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper attempts to provide theoretical explanation on why FedAvg works in practice when learning overparameterized models with many local steps.

### Strengths
The paper is well-written with clear motivation and clean structure.

### Weaknesses
1. The analysis, theoretical results, and experiments in this paper focus only on linear regression and linear classification. This is a reasonable starting point but is clearly a severe limitation. This limitation should be stated explicitly in the title and abstract.

2. The statement that "Although the existing convergence analysis suggests that with heterogeneous data, FedAvg encounters quick performance degradation as the number of local steps increases, it is shown to work quite well in practice" in the abstract is questionable. In many recent literatures, FedAvg has shown to stagnate early compared to a centralized approach and the difference increases when the number of clients, number of local steps, and data heterogeneity increases, which motivates the development of various schemes (FedLin, FedSVRG, FedVARP, FedMoS, etc) to mitigate client drifts. Of course, these work do not consider linear models and thus may not contradict to the results shown in this paper. However, I suggest the authors should take these results into account and revise the relevant statements. As for the LLM results with FedAvg, could the authors provide references in which FedAvg is compared to a centralized approach in LLM training? Without comparisons to a centralized baseline, I don't think the fact that LLMs can be trained with 500 local steps is supportive to the statement above.

3. The analysis does not seems to take the stochasticity in SGD into account. The authors could consider discussing potential extensions to the SGD case.

4. The heterogeneous data case considered in the experiment is rather naive. The authors should consider reporting the test results for data distributed based on the Dirichlet distribution as considered in, for example, https://arxiv.org/pdf/2309.01275 and https://arxiv.org/pdf/1909.06335 . This approach allows for a description of level of heterogeneity based on the Dirichlet distribution parameters.

### Questions
None.

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
2

### Summary
This manuscript solves the contradiction between the results that current theoretical research showed that FedAvg with heterogeneous data encounters quick performance degradation as local updates increase and the fact that FedAvg works well in practice under such conditions. The authors illustrate their theory on the examples of linear regression and linear classification problems from a viewpoint of implicit bias, showing how the result obtained by Local-GD converges with respective to the result of the centralized model. Furthermore, experiments are conducted to validate their theoretical findings.

### Strengths
- From the perspective of novelty, this is the first work to analyze the implicit bias of the gradient descent in the distributed setting. The authors propose a distributed algorithm with a refined aggregation method, which shows the exact convergence direction to the centralized model.

- For the linear regression task, the paper presents the closed form solution. Although it is hard for the classification task to get the closed form solution, the authors use other techniques to show the results obtained by Local-GD converge in direction to one point in the global feasible set.

- Experiments are not only conducted on linear regression and linear classification tasks, but also perform fine-tuning of pretrained neural networks.

### Weaknesses
- Typos: line 887.

- This manuscript doesn't provide an explicit convergence rate. While there exist works on the convergence rate of Local-SGD on overparameterized model "Faster Convergence of Local SGD for Over-Parameterized Models". Can the authors compare their derived theoretical results with this paper?

- With a large amount of data, stochastic methods might be more popular. Can the theory extend to the Local-SGD to reveal the convergence performance with heterogeneous for overparameterized models, will there be similar convergence results for Local-SGD?

### Questions
See Weakness.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In the paper, the authors analyze the FedAvg algorithm for overparameterized linear regression and classification problems when the local nodes take a large number of local updates before communicating with the central server. The authors model a large number of local updates by assuming that the local nodes exactly solve their corresponding problems. The authors show that for the regression problem, the aggregation of local models and the centralized model converge to the same solution while for the classification problem, the two solutions converge to the same set. Further, the authors propose a different aggregation rule that guarantees that the centralized model converges in direction to the aggregated local models. Finally, the authors corroborate their theoretical findings via a set of numerical experiments.

### Strengths
**Strengths.** Here, I list the major strengths of the paper. 

- The paper considers an important problem of analyzing the implicit bias of the FedAvg algorithm which is popular for training large-scale models. 
- The attempt to characterize the usefulness of a large number of local updates is useful and of interest to the research community. 
- The experiments corroborate the presented theoretical results. 
- The paper is well-written and easy to follow.

### Weaknesses
**Weaknesses.** Although the paper considers an important problem, I have major doubts about the quality of the paper and the presented results. I list my major concerns in the following. 

**Major.**
- I do not understand the argument before equation (8) where the authors state that "We replace $y_i$ in the local model (4)...". Can the authors explain how and why they can substitute $X_i w_c = y_i$ in equation (4)? I do not believe that we should be able to do that since equation (4) is defined for the FedAvg algorithm which is a different algorithm and for FedAvg we have $X_i w_i=y_i$ rather than $X_i w_c = y_i$ which holds for the centralized setting. If I am right equations (8) and (9) will not hold and therefore I have major doubts about the presented results. 
- The algorithm for the classification problem should not be called FedAvg or local SGD since it is solving a different regularized problem. 
- The result of Theorem 2 utilizes Lemma 3, however, I do not believe the assumptions stated in Lemma 3 to hold true. Specifically, Lemma 3 states that one of the feasible sets is bounded, however, none of the feasible sets in equation (13) are bounded. So how do the results stated in Theorem 2 hold?
- Theorem 2 states that the solution of the centralized model and the distributed algorithm will converge to the same set, however, this result does not convey much since the two points can still be far apart. 
- I have a question about the setting of heterogeneous data generation, specifically, after equation (2) the authors mention that $w_i^\ast$ can be different across local nodes and these $w_i^\ast$'s model the heterogeneity across the network. However, in my understanding, the goal of FL is to learn a global model under the setting that the data is generated from a global model and the heterogeneity originates from different noise distributions.  Can the authors please comment on this setting? 
- The authors have missed a whole line of work on analyzing overparameterized FL algorithms. Please see [1-3], these should be added to the literature. There may be more, please check. 

    [1] Huang, B., Li, X., Song, Z., and Yang, X. Fl-NTK: A neural tangent kernel-based framework for federated learning analysis. In International Conference on Machine Learning, pp. 4423–4434. PMLR, 2021.

    [2] Deng, Y. and Mahdavi, M. Local SGD optimizes overparameterized neural networks in polynomial time. arXivpreprint arXiv:2107.10868, 2021.

    [3] Song B, Khanduri P, Zhang X, Yi J, Hong M. Fedavg converges to zero training loss linearly for overparameterized multi-layer neural networks. In International Conference on Machine Learning PMLR, 2023.


**Minor.**
- When the authors mention in the abstract that the solution to the "classification problem converges (in direction) to the same feasible set as the centralized model" is confusing since reading the statement seems that the two solutions converge to the solutions that have the same direction which is not true.
-   In the introduction, when the authors mention "number of local steps $L$ should not be very large, rather vary as $O( \sqrt{T})$ where $T$ is the total number of iterations". They should clarify the setting under which this statement holds. I believe this does not hold for training non-convex functions but only for strongly-convex smooth problems. 
- Once Algorithm 2 is stated, there is no need to re-state Algorithms 3 and 4 since the only difference between the three algorithms is the definition of local functions.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
