# Covariate-Guided Clusterwise Linear Regression for Generalization to Unseen Data

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
In many tabular regression tasks, the relationships between covariates and response can often be approximated as linear only within localized regions of the input space; a single global linear model therefore fails to capture these local relationships. Conventional Clusterwise Linear Regression (CLR) mitigates this issue by learning $K$ local regressors. However, existing algorithms either optimize latent binary indicators, (i) providing no explicit rule for assigning an $\textit{unseen}$ covariate vector to a cluster at test time, or rely on heuristic mixture of experts approaches, (ii) lacking convergence guarantees. To address these limitations, we propose $\textit{covariate-guided}$ CLR, an end-to-end framework that jointly learns an assignment function and $K$ linear regressors within a single gradient-based optimization loop. During training, a proxy network iteratively predicts coefficient vectors for inputs, and hard vector quantization assigns samples to their nearest codebook regressors. This alternating minimization procedure yields monotone descent of the empirical risk, converges under mild assumptions, and enjoys a PAC-style excess-risk bound. By treating the covariate data from all clusters as a single concatenated design matrix, we derive an $F$-test statistic from a nested linear model, quantitatively characterizing the effective model complexity. As $K$ varies, our method spans the spectrum from a single global linear model to instance-wise fits. Experimental results show that our method exactly reconstructs synthetic piecewise-linear surfaces, achieves accuracy comparable to strong black-box models on standard tabular benchmarks, and consistently outperforms existing CLR and mixture-of-experts approaches.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes Covariate-Guided Clusterwise Linear Regression (CG-CLR), which jointly learns a covariate-based assignment rule and multiple local linear regressors in an end-to-end fashion. A feedforward neural network predicts a per-sample proxy regressor using only the covariate, which are quantized to the nearest of 𝐾 learned regressors. The paper provides convergence and generalization guarantees and introduces an F-test-based method to select 𝐾. Experiments on synthetic and tabular datasets show that CG-CLR achieves superior accuracy comparable to other models that have "large coverage".

### Strengths
1. Novel end-to-end CLR framework: The paper introduces a novel way to partition the covariate space using a feedforward neural network and to assign a regressor to these partitions simultaneously. The key ideas are captured clearly in Equations (3) and (4).

2. Because the model uses local linear regressors, it maintains a level of interpretability – each cluster’s behavior can be understood via a simple linear formula. The learned “codebook” of regressors is compact, making the overall model relatively easy to inspect and reason about. This is an advantage over complex black-box models and aligns with the paper’s motivation of balancing accuracy and interpretability.

3. Theoretical rigor: The paper proves monotonic descent of the empirical risk and even linear convergence to an optimum under certain conditions. Additionally, it derives a PAC-style excess risk bound.

4. The paper offers a principled way to choose the number of clusters  K via an F-test.

5. Empirical performance: The model demonstrates strong performance on several benchmarks compared to “large-coverage” models.

6. Clarity: The paper is clearly written and very well-organized, making both the methodology and theory easy to follow.

### Weaknesses
1. The assumptions in the theoretical proofs could be quite restrictive. It would be helpful to discuss why these assumptions are reasonable and under what conditions they might fail.

2. The framework may fail when groups have very different regression vectors but very similar covariate distributions, since the assignment is guided only by covariates.

3. The clusters are not necessarily linear or interpretable; the learned partitions may be arbitrary or data-driven without clear meaning.

4. Simpler methods such as Random Forest, XGBoost, or CatBoost often outperform CG-CLR, suggesting it is not clearly superior to strong baselines.

Minor Comment:
typo in line 184

### Questions
1. It would be interesting to see results for higher-dimensional synthetic experiments (dimension > 2) and for more practical covariate distributions, such as Gaussian mixtures.

### Soundness
4

### Presentation
4

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
The paper introduces covariate-guided Clusterwise Linear Regression (CLR), a framework that jointly learns cluster assignments and local linear models through end-to-end gradient-based optimization. By combining a proxy network with hard vector quantization, the method ensures convergence, provides theoretical risk guarantees, and adapts smoothly from global to instance-wise regressions. Experiments show it accurately reconstructs piecewise-linear functions and outperforms existing CLR and mixture-of-experts approaches on tabular benchmarks.

### Strengths
- The paper tackles an important and well-motivated problem in modeling local linear relationships in tabular regression tasks.

- The proposed approach effectively integrates a simple yet elegant idea—using a proxy network (similar to a hypernetwork) to predict regression coefficients—into an end-to-end optimization framework.

- The method is theoretically well justified, offering convergence guarantees, a PAC-style excess-risk bound, and a principled F-test analysis of model complexity.

- The approach demonstrates strong empirical performance, accurately recovering piecewise-linear structures and outperforming existing CLR and mixture-of-experts baselines.

### Weaknesses
- The proposed approach appears somewhat incremental, primarily combining elements of hypernetworks with an expectation–maximization-style framework using hard assignments. While effective, the conceptual novelty over existing methods may be limited.

- The paper’s presentation is weak—the writing is often unclear, and the main contributions focus more on the properties of the proposed method rather than its broader impact or significance. Although the method itself is relatively straightforward, the paper is difficult to follow, and important implementation details—particularly regarding how the model is applied at test time—are missing.

- While the paper provides some theoretical assumptions about the mapping function used for cluster assignments, it remains unclear how these properties are enforced during training when using an MLP. It would also be useful to discuss whether other architectures could replace the MLP and how such substitutions might affect convergence or theoretical guarantees.

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes Covariate-Guided Clusterwise Linear Regression (CG-CLR), an end-to-end framework that jointly learns (i) a response-free assignment rule mapping a covariate vector to one of $K$ local linear regressors and (ii) the $K$ regressors themselves. The system uses a proxy network that outputs instance-wise coefficient vectors; a hard vector-quantization step assigns each input to the nearest codebook regressor; and training alternates between updating the proxy and the codebook via a composite “fit + alignment” objective. The authors show monotone descent of a Lyapunov-style objective with linear convergence under assumptions, a PAC-style excess-risk bound, and a nested-model $F$-test to tune $K$. Empirically, CG-CLR recovers synthetic piecewise-linear structure and is competitive with tree ensembles on several tabular benchmarks.

### Strengths
* The paper is well-organized and clearly written. The paper has clean and good illustrations for their takeaways and empirical findings.
* The paper has a good mix of both theoretical and empirical evidence. The method proposed is also generic enough. I must admit I'm not an expert in this field, but the results in general strike me as interesting.

### Weaknesses
Major comments:
* The paper claims one of the strength is that prediction is accurate without ever observing the true response $y_{i'}$---however, there is really nothing surprising there. This is basically generalization in conventional machine learning setup, and is not special to this particular task. The authors should make it clear that the actual strength is the proxy is label independent, which means the algorithm is unsupervised in natural. I'm also not sure how much ``the inner minimization over $j$ cannot be evaluated at test time because the response $y_i'$ associated with an unseen covariate vector $x_i′$ is unknown at prediction time'' is undesirable. We can always estimate generalization error by cross validation/hold-out set if needed. 
* The linear convergence result requires a uniform lower bound on the proxy network's Jacobian, strong convexity and smoothness of the alignment loss. Those conditions seem strong and probably unverifiable in reality. If for some toy model or the empirical setup in the paper, those conditions can be justified (even numerically), they would be much more convincing.

Minor comments:
* P7. "form" should be "from"

### Questions
* My main questions are in the weaknesses section.
* Overall I think the paper provides a useful engineering tool, while the discussion about its theoretical results, especially on the assumptions are limited. How would the author justify their assumptions empirically? How should those requirements be tested?

### Soundness
3

### Presentation
4

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
The paper studies the Clusterwise Linear Regression (CLR) problem in which the goal is to find $K$ regressors for a given dataset such that for each point there is one regressor out of the $K$ which has a low loss prediction. Previous methods either used optimization to find the $K$ linear regressors but provided no way to assign a new unlabeled point to a regressor during inference, or leveraged a mixture of experts or decision trees but did not have good convergence. In this work, the authors try to address these problems by introducing a proxy network which predicts a good regressor for each input point along with a dual loss which optimizes the network as well as a codebook of candidate regressors simultaneously. Under some smoothness, convexity and cluster separation assumptions, the authors prove convergence as well as generalization error bounds for their CG-CLR method. Included experiments on synthetic and real datasets show performance gains as compared to some previous approaches.

### Strengths
1. The paper proposes a new technique for CLR which optimizes a proxy predictor network whose predictions at test time can be used to assign each point to a candidate regressor. 
2. The paper provides theoretical guarantees under some reasonable assumptions.

### Weaknesses
1. The proposed method hinges on the assumption that a good proxy network can be trained, however this means that $W_\phi(\textbf{x}_i)^\textsf{T}\textbf{x}_i$ is a good label predictor. This might be a fairly strong assumption which simplifies the problem. In particular, the assumption means that the label can essentially be predicted by the proxy network.  Why not use the proxy network itself for inference instead of the set of linear regressors?
2. The experiments seem a bit inadequate: the proposed method is somewhat of an extension to the methods of [Ghosh-Mazumdar, ICML’24] and so should be compared against them. Also, the experiments should measure how well the proxy network itself is predicting the labels (as mentioned above). 
3. The paper, though well written for the most part, has some parts which need clarification (see questions to authors).

### Questions
1. In the CG-CLR algorithm, why are the proxy network and the codebook updated separately and not together?
2. The empirical risk mentioned on page 6 should be formally defined (or a reference to the definition in the appendix included).
3. Assumption 4 seems unclear. The condition that $d \gg K(p+1)$ can trivially be achieved by adding dummy parameters. The statement of the assumption should be made more formal.
4. What are CG-CLR (Proxy) and CG-CLR (Codebook) mentioned on line 408?

### Soundness
2

### Presentation
2

### Contribution
2
