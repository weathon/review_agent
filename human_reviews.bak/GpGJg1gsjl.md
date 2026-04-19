# Uncertainty for Active Learning on Graphs

- Decision: Reject
- Scores: 3, 3, 5, 6

## Abstract
Active learning (AL) is a promising technique to improve data efficiency of machine learning models by iteratively acquiring data labels during training. While Uncertainty Sampling (US) - a strategy that labels data points with the highest uncertainty - has proven effective for independent data, its implications for interdependent data, such as nodes in graphs, remain under-explored. In this work, we
propose the first extensive study of US for node classification. Our contribution is threefold: **(1)** We are the first to provide a benchmark for US approaches beyond predictive uncertainty. We highlight a performance gap between conventional AL strategies for graphs and US. **(2)** We develop novel ground-truth Bayesian uncertainty estimates in terms of the data-generating process. We both theoretically prove and empirically confirm their effectiveness in guiding US toward high-quality label queries. **(3)** Based on our analysis, we highlight pitfalls in modeling uncertainty and relate them to contemporary uncertainty estimators for node classification.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a comprehensive study of applying Uncertainty Sampling (US) within the Active Learning (AL) framework for node classification within graphs. The authors provide a benchmark for AL and evaluate the performance of AL baselines using real word datasets. Additionally, they propose novel Bayesian uncertainty estimation methods based on the ground truth labels, and illustrate their effectiveness using synthetic CSBM dataset.

### Strengths
The paper offers a thorough evaluation of AL performance through a series of well-conducted experiments and a qualitative analysis.

### Weaknesses
The proposed ground truth uncertainty is not so useful and the uncertainty sampling US based on it is not practical. During prediction procedure, the ground truth label remains unknown and therefore it is inappropriate to define an uncertainty based on it. 

US with knowledge of ground truth label would benefit from the information leakage and so the good performance in the CSBM dataset is not achievable in real-world datasets. For example, in traditional AL algorithms, it's difficult to select a node for query when the classifier gives the ground truth label of the node a low predictive probability, although querying such node would provide the classifier a lot information. Take, for instance, a scenario where  p(ground truth class| y_i ) = 0.1 and p(incorrect class| y_i ) = 0.9. Typically the prediction to incorrect class of y_i might be considered confident and AL algorithm will not choose y_i for querying, and such error will cause general AL algorithm not perform as good as random sampling. But for US based on ground truth uncertainty, the epistemic uncertainty is large and the node will be selected. Therefore, the good performance in the CSBM dataset is not practical.

### Questions
Please explain the practical application of the ground truth uncertainty.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors establish a benchmark for uncertainty sampling based active learning approaches for graph data. The paper also proposes a Bayesian uncertainty estimation to actively select the node. This estimation is based on the knowledge of data-generating process. The authors validate the effectiveness of their approach with both theoretical analysis and empirical experiments.

### Strengths
a. This paper studies the active learning problem with graph data from an interesting perspective--uncertainty sampling strategy and propose a new Bayesian uncertainty estimation.

b. The authors provide both theoretical analysis and empirical results to show the effectiveness of the proposed estimation.

### Weaknesses
a. Theoretical contributions in this paper appear to be somewhat limited. The proposed uncertainty estimation is based on the posterior probability given the ground-truth label of the unobserved nodes. However, the essence of active learning lies in addressing this problem without access to ground-truth information, which remains inadequately addressed.

b. The experimental results provided are restricted to synthetic data, and the method's reliance on knowledge of the true data generation process probabilities pose practical challenges. How to approximately compute the estimation remains unclear.

c. In the empirical evaluation, the compared baselines are only random queries and other uncertainty-based methods, the state-of-the art methods are missing, e.g. SEAL[1] and IGP[2]. 

d. The paper's presentation could be improved. For instance, when introducing concepts like aleatoric and epistemic uncertainty, the authors provide limited explanations and intuitions, potentially causing readers unfamiliar with these terms to struggle to follow the paper.

[1] Li Y, Yin J, Chen L. Seal: Semisupervised adversarial active learning on attributed graphs[J]. IEEE Transactions on Neural Networks and Learning Systems, 2020, 32(7): 3136-3147. 
[2] Zhang W, Wang Y, You Z, et al. Information Gain Propagation: a new way to Graph Active Learning with Soft Labels[J]. arXiv preprint arXiv:2203.01093, 2022.

### Questions
a. How can the proposed uncertainty estimation be computed in practical scenarios where true data generation probabilities are unknown? Are there methods or approaches to approximate this estimation without relying on ground-truth knowledge?

b. What's the performance of non-US based active learning methods on CSBMs?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work is an empirical study of a typical Active Learning method, Uncertainty Sampling (US) for node classification on graphs. The authors present an extensive benchmark for US methods that goes beyond predictive uncertainty, revealing that, the US employing modern uncertainty estimators struggles to outperform random queries consistently. The authors establish ground-truth Bayesian uncertainty estimates for a Bayesian classifier based on the underlying graph generative process, providing formal evidence of the alignment between US and AL. When they apply their approach using a Clustered Stochastic Block Model (CSBM), they empirically confirm the effectiveness of US when uncertainty estimates are accurately disentangled into aleatoric and epistemic uncertainty while considering all available graph information.

### Strengths
- This work provides an empirical study for US with node classification on graphs, highlighting both its efficacy and potential limitations.

- An important finding is that the existing AL methods cannot outperform random sampling benchmarks.

### Weaknesses
- The study primarily concentrates on a specific graph type, the CSBM, which might not fully represent the characteristics of all real-world graphs.

- Novelty concern: undoubtedly, this work offers an extensive exploration of uncertainty-based Active Learning (AL) within the context of graphs. However, it does not introduce any novel methods for active learning in the graph domain.

### Questions
- In the experimental results, such as Figure 3, the curves depicting acquired labels versus accuracy exhibit significant fluctuations. Did the authors conduct repeated trials to mitigate these fluctuations in the model's performance?

- In the Introduction section, the authors dedicated an extensive portion of the text to explain uncertainty sampling. This level of detail might be excessive as uncertainty sampling is a straightforward concept. It would be more beneficial to present the essential formulations and allocate additional space to elaborate on active learning in graph-related tasks.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This article studies the application of AL to graph data, with a focus on the approach of uncertainty sampling. The authors demonstrated through an extensive empirical analysis that many AL strategies, uncertainty-based or not, failed to surpasse random sampling on graph data. A curious observation is that uncertainty estimators which distinguish the reducible uncertainty caused by the randomness of training data from the irreducible uncertainty due to the underlying data generating process and use only the reducible uncertainty to guide the label queries work well on i.i.d. data but not on graph data. Motivated by this observation, the authors proved theoretically that, under a Contextual Stochastic Blockmodel (CSBN) with known parameters, minimizing the reducible uncertainty leads to an optimal AL strategy. This remark is  confirmed on simulated data of (CSBN).

### Strengths
- This work is well guided with a series of inquiries, starting with open questions in literature review, conducted with empirical observation, theoretical investigation, ending with experimental confirmation.

- The thorough empirical analysis and the original theoretical insights are of interest to the scientific community.

### Weaknesses
- The theoretical investigation, which is a major contribution of the article, not only considers a specific model (which is perfectly acceptable), but also assumes the full knowledge of the parameters underlying the model. As in practice the model parameters are rarely known and have to be estimated from data, their estimation error will contribute to the reducible uncertainty. Therefore defining the reducible uncertainty while assuming the model parameters to be pre-known seems to be problematic and needs at least to be discussed.

- It should be made clear earlier in the article (e.g. in the abstract or introduction) that the proposed uncertainty measure is not directly applicable in practice, and rather of theoretical interest.

### Questions
My questions are related to the first point of Weaknesses:

- How will the reducible uncertainty change without the knowledge of model parameters ?

- Will the conclusion regarding the optimality of using the reducible uncertainty to guide AL stay the same ?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
