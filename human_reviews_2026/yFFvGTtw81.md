# Whom to Trust? Adaptive Collaboration in Personalized Federated Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Data heterogeneity poses a fundamental challenge in federated learning (FL), especially when clients differ not only in distribution but also in the reliability of their predictions across individual examples. While personalized FL (PFL) aims to address this, we observe that many PFL methods fail to outperform two necessary baselines, local training and centralized training. This suggests that meaningful personalization only emerges in a narrow regime, where global models are insufficient, but collaboration across clients still holds value. Our empirical findings point to two key ingredients for success in this regime: adaptivity in collaboration and fine-grained trust, at the level of individual examples. We show that these properties can be achieved within federated semi-supervised learning, where clients exchange predictions over a shared unlabeled dataset. This enables each client to align with public consensus when it is helpful, and disregard it when it is not, without sharing model parameters or raw data. As a concrete realization of this idea, we develop FEDMOSAIC, a personalized co-training method where clients reweight their loss and their contribution to pseudo-labels based on per-example agreement and confidence. FEDMOSAIC consistently outperforms strong FL and PFL baselines across a range of non-IID settings, and we prove convergence under standard smoothness, bounded-variance, and drift assumptions. In contrast to many of these baselines, it also consistently outperforms local and centralized training. These results clarify when federated personalization can be effective, and how fine-grained, trust-aware collaboration enables it.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work finds that many personalized FL (PFL) methods fail to outperform two necessary baselines, local training and centralized training.  FEDMOSAIC addresses these challenges with two core mechanisms: dynamic loss weighting and confidence-based aggregation. The authors verify its effectiveness via empirical experiments and theoretical analysis.

### Strengths
1. The proposed FEDMOSAIC approach provides a flexible framework for fine-grained trust, allowing clients to adaptively decide when and whom to trust at a per-sample level.

2. Compared with the traditional parameter-sharing method, FEDMOSAIC significantly reduces uplink communication costs by only exchanging hard labels and scalar expertise scores.

3. The method's effectiveness is validated through comprehensive experiments across diverse non-IID settings, including label skew, feature shift, and challenging hybrid scenarios.

### Weaknesses
1. FEDMOSAIC requires on a shared, unlabeled public dataset. While the authors mentioned that this dataset can be found in many domains, the distribution and reliability of this dataset are hard to know, undermining the FEDMOSAIC's practicality.

2. The exploration of the expertise score is limited. The paper relies on a simple class-frequency heuristic that performed similarly to an uncertainty-based score, leaving its optimal design in various scenarios underexplored.

3. The server-side aggregation scales with the number of clients and the public dataset size, but its computational feasibility is untested beyond 30 clients, which is a small scale for typical FL scenarios.

### Questions
1. The method's performance and computational viability in large-scale FL settings with thousands or more clients remains an open question.

2. In the provided code, which `args.method` does FEDMOSAIC correspond to? I didn't find the searching result of `FEDMOSAIC`.

3. How does the introduction of differential privacy mechanisms  empirically affect the accuracy and convergence of the confidence-based aggregation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents FedMosaic, an adaptive collaborative training scheme, that augments federated co-training (Abourayya et al., 2025) to strike a balance between collaboration and personalization by adaptively tuning how much each client relies on the collective expertise. Technical improvements with respect to federated co-training include leveraging uncertainty (or expertise) scores to enhance the labeling quality of the consensus unlabeled data, and reweighing the local and global training loss at each training step to avoid collaboration when not needed. The paper presents a theoretical convergence result, a brief privacy accounting argument and an empirical evaluation including comparisons with state-of-the-art Personalized Federated Learning (PFL) methods.

### Strengths
The research problem is relevant. Deriving algorithms that allow collaborative training agnostic to the local training algorithms, while improving efficiency by avoiding expensive model sharing is quite interesting. The empirical findings of the paper are quite promising, showcasing that FedMosaic outperforms other personalized federated learning baselines in different heterogeneity scenarios. FedMosaic is also shown to consistently outperform local learning and global federated learning.

### Weaknesses
- The bounded objective drift assumptions is not standard in the distributed optimization literature. The authors do provide an explanation of why they expect this assumption to hold in practice (line 300), however, no guarantee on $\delta$ is provided. Given that this term appears additively in the convergence result, this is a key limitation that needs to be addressed.

- The optimization results involves upper bounding $||\nabla L_t^i  (\theta_t)||^2 $, but this is not really a relevant quantity, as by design the algorithm optimizes $L_t^i$ at each step, but the relevant quantity would be for instance the local test loss (if we care about personalization performance) or global test loss. 
For a fair theoretical comparison with the other PFL methods, generalization bounds are preferable (e.g APFL (Deng et al., 2020))

- The gradient estimator $g_t^i$, presented in the assumptions, does not appear in the algorithm. 

- The theoretical analysis does not take into account the quality of the self-evaluated expertise vectors, which can also be a key weakness of the algorithm. 

- As acknowledged by the author, another limitation of this work is assuming the existence of the unlabeled dataset which is assumed to be from the global distribution. In practice, this would mean clients should send part of their own data which limits the privacy (or use private synthetic data as stated in the conclusion). In any case, this unfairly favors FedMosaic when comparing it to baselines. 

- For the experiments, the number of clients remains quite low, which is likely to favor good local training performance, as each client has enough data.

### Questions
- Can the authors address the issues of the theoretical part raised in the Weaknesses section above ? 
- In the experiments, what is used as $A_i$ ? 
- How does the wall-clock compare with the baselines ?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper developed an algorithm, FedMosaic, for personalized federated learning. Instead of sharing model parameters, each client communicates with the server its predictions and confidence scores of a public unsupervised dataset. The local training balances between the local objective and the global objective based on the public dataset with pseudo-labels. Theoretical analysis on communication cost and convergence are provided. Empirically, the algorithm outperforms not only the baselines of local/global training, but also (personalized) federated learning algorithms.

### Strengths
1. The writing is clear and easy to follow. 

2. It provides theoretical analysis on both communication cost, convergence and privacy.

3. It achieves superior performance compared to many other algorithms using standard datasets with strong skewness.

### Weaknesses
1. The confidence scores are not explicitly explained in the paper. L205 discusses two different methods to calculate the confidence scores. However, they are not explained in detail anywhere. Only their performances are shown in the experiments. This should be addressed before it is ready for publication.

2. The algorithm relies on the availability of a public unlabelled dataset, which may not exist in practice. Moreover, the FL baselines in the experiments are not utilizing the public data, which makes the comparison unfair to some extent. It would make more sense to compare to baselines that do use the public dataset (Li & Wang, 2019; Huang et al., 2022; Yu et al., 2022)

Reference

- Li, D. and Wang, J., 2019. Fedmd: Heterogenous federated learning via model distillation. *arXiv preprint arXiv:1910.03581*.
- Huang, W., Ye, M. and Du, B., 2022. Learn from others and be yourself in heterogeneous federated learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition* (pp. 10143-10153).
- Yu, S., Qian, W. and Jannesari, A., 2022. Resource-aware federated learning using knowledge extraction and multi-model fusion. *arXiv preprint arXiv:2208.07978*.

3. Experiments require further discussion.

- What are the numbers in the parentheses for the tables? How many runs/repeats are used?
- The number of clients is relatively low (15 for Table 2 and 5 for Table 4). It would be helpful to see how FedMosaic performs when there are many clients.
- Why not apply both confidence mechanisms to Tables 2 and 3? As it is written right now, only one of the mechanisms is used for each table.

Minor comments

- It is unclear why the algorithm is named FedMosaic
- L54, L365 and many other places left quotation should be `` instead of '' or "
- L69 essential
- L194 $L_t^i$ should be defined here instead of L209.
- L197 diag and $C$ undefined.
- L320 there is no actual footnote.
- L425, L427 no need to use abs

### Questions
Please see the weaknesses above.

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
4

### Summary
This paper introduces a novel mechanism for cross-client training that preserves data privacy and accuracy, inspired by the concept of federated co-training. Unlike conventional federated learning, where local models are distributed and aggregated on a central server, the proposed approach requires clients to share only the prediction results of their local models on a global unlabeled dataset, along with their per-sample confidence scores. Based on these predictions, the server dynamically assigns pseudo-labels to all samples in the global dataset. To enhance local model performance, each client trains its model using both its private local data and the globally pseudo-labeled data, employing a weighted loss combination strategy. Experimental results demonstrate that the proposed method significantly outperforms the baseline in terms of accuracy in many different settings.

### Strengths
S1:The proposed co-training approach is novel and offers numerous opportunities for further improvement. In this framework, neither local data nor local models are shared, which enhances privacy preservation and reduces communication costs with the server.
S2: The main contribution of the authors is the introduction of a dynamic mechanism that combines local loss functions and adaptively updates the labels of samples in the global dataset U. This mechanism provides flexibility and allows the proposed method to adapt to various learning scenarios.

### Weaknesses
W1. TFrom Algorithm 1, it appears that the authors assume all clients participate in the co-training process at every time step (or communication round). This assumption differs from that of conventional federated learning and may not be feasible in real-world applications. Furthermore, even under this assumption, the proposed mechanism faces scalability challenges. As the number of clients increases (e.g., hundreds or thousands) and/or the size of the global dataset U grows, the communication and computation overhead at the server can become substantial. The authors should discuss potential strategies to address this issue or clarify how the proposed method would perform if the aforementioned assumption does not hold.
W2. The experimental results are promising; however, several important details are missing. Please clarify: (1) how the model accuracy is evaluated — for instance, is it measured using a global testing dataset or by averaging the accuracies across all clients? (2) why local training outperforms other methods in Table 1 and certain results in Table 2. Moreover, the datasets used are relatively small and contain a limited number of classes. It would strengthen the paper if the authors could include results on other types of tasks or larger-scale datasets. (3) please also clarify which model architecture was used in the experiments, as this information is missing from both the main text and the appendix.

### Questions
See above

### Soundness
4

### Presentation
3

### Contribution
3
