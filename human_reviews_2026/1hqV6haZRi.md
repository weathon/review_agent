# Target Before You Perturb: Enhancing Locally Private Graph Learning via Task-Oriented Perturbation

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Graph neural networks (GNNs) have achieved remarkable success in graph representation learning and have been widely adopted across various domains. However, real-world graphs often contain sensitive personal information, such as user profiles in social networks, raising serious privacy concerns when applying GNNs to such data. Consequently, locally private graph learning has gained considerable attention. This framework leverages local differential privacy (LDP) to provide strong privacy guarantees for users' local data. Despite its promise, a key challenge remains: how to preserve high utility for downstream tasks (e.g., node classification accuracy) while ensuring rigorous privacy protection. In this paper, we propose TOGL, a Task-Oriented Graph Learning framework that enhances utility under LDP constraints. Unlike prior approaches that blindly perturb all attributes, TOGL first targets task-relevant attributes before applying perturbation, enabling more informed and effective privacy mechanisms. It unfolds in three phases: locally private feature perturbation, task-relevant attribute analysis, and task-oriented private learning. This structured process enables TOGL to provide strict privacy protection while significantly improving the utility of graph learning. Extensive experiments on real-world datasets demonstrate that TOGL substantially outperforms existing methods in terms of privacy preservation and learning effectiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies graph neural networks under local differential privacy and, for the first time, introduces a task-relevant optimization mechanism. The authors compare their approach with existing LDP methods that protect node features and demonstrate a better trade-off between privacy and utility.

### Strengths
1. In terms of novelty, the paper is the first to propose a multi-stage perturbation mechanism guided by task-relevant feature selection.
2. The proposed method achieves a superior privacy–utility trade-off compared to existing approaches.
3. The paper is well-structured, clearly written, and the experimental results are easy to follow.

### Weaknesses
1. The proposed method includes an additional server-side aggregation step that merges results from two rounds of perturbation, whereas the baselines do not. Therefore, it is unclear whether the observed improvement in the privacy–utility trade-off arises from the proposed LDP mechanism itself or from the aggregation process on the server.
2. The authors compute task-relevant features based on the first-round perturbed data and then perturb these features again in the second round. Intuitively, this means the most important features are perturbed twice, and the noise magnitude is larger than that of existing single-shot mechanisms. The authors should clarify why this design leads to higher utility rather than degradation.
3. The paper lacks a clear definition of what aspects are protected under LDP in the main text, only stating in experiments that feature privacy is considered. Since GNNs may involve protecting features, edges, or labels, the lack of explicit scope may cause confusion.
4. The paper does not evaluate resistance to feature inference attacks, which is an important aspect of verifying practical privacy protection. Such experiments are strongly recommended.

### Questions
See the issues discussed in the “Weaknesses” section above.

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
This paper presents a new locally private graph learning framework from a task-oriented graph learning perspective (TOGL). It contains three phases: locally private feature perturbation, task-relevant attribute analysis, and task-oriented private learning. Extensive experiments demonstrate TOGL's substantial utility improvements over existing baselines.

### Strengths
1. Well-structured and clearly written.
2. This paper emphasizes the urgent need to connect local differential privacy (LDP) with downstream tasks to achieve better utility, and empirically demonstrates its importance.
3. This paper provides fundamental theoretical proof and analysis, showing the correctness of its use of LDP.

### Weaknesses
1. This paper does not contribute to the LDP part, only designing a task-oriented attribute selecting mechanism in the server to benefit downstream tasks. Phase I is a one-time perturbation, no different from LPGNN (Sajadmanesh & Gatica-Perez, 2021).
2. The presentation of Phase III in Figure 2 is misleading. According to Algorithm 2, the selected attributes $S^*$ and hyperparameter $\rho$ do not directly affect the LDP, but utilize the LDP's post-processing invariance properties, ensuring strict privacy guarantees for subsequent processing.
3. There is no summary of task-oriented methods. Is LPGNN a task-oriented method?
- If not, why? And what special adjustments are needed for different tasks (node classification and link prediction) compared to the baselines?
- If it is, then the contribution of this paper will be diminished. Overall, the method in this paper is similar to LPGNN in its approach, as both utilize embedding and labels to constrain task performance.
4. The LDP mechanisms of PM, MB, and SW lack an explanation of the coefficient $\delta$ $, which is only described in the Gaussian mechanism.
5. The accuracy in Figure 6 was normalized, which may overemphasize the differences between methods. It is recommended to show actual ablation study results.
6. The interpretation in Figure 9 is weak, casting doubt on the method's utility. The results show that random feature selection achieves near-suboptimal results when $\rho$=0, indicating that random diversity is more helpful. However, when $\rho$=1, the algorithm relies entirely on task-relevant effects (approximately 30%), almost losing its inference ability for downstream tasks, indicating that this module contributes little.
7. Attack experiments are lacking to demonstrate that the method's privacy guarantees are not compromised to address the second challenge in line 88.
8. The lack of open-source code and insufficient reproducibility reduce the credibility of this work.

### Questions
1. How is Equation 7 used in Algorithm 2 to represent the SMA mechanism?

2. Are the six LDP mechanisms implemented by changing the perturbation mechanism based on the LPGNN framework? Please clarify.

3. Do the LDP mechanisms share the same set of parameters in the same dataset? For example, $K$, $\rho$, etc.

4. Which mechanism is described as the state-of-the-art (SOTA) in Figures 4, 5, and 6?

5. Why can the analysis of parameter $K$ be an ablation study in Figure 7?

### Soundness
3

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
This paper presents Task Oriented Graph Learning (TOGL) framework for locally private graph learning under Local Differential Privacy (LDP) constraints. The paper advocates that instead of considering random dimensions of node attributes for perturbation, which provides privacy but at a cost of utility one should consider identifying task-specific features. To this end, the authors introduce the notion of “target then perturb” for LDP. TOGL follows a three-stage pipeline: in the first stage, node features are perturbed locally using LDP to satisfy privacy requirements, and the server then denoises the perturbed features through neighborhood aggregation. In the second stage, the server identifies the top-m task-relevant feature dimensions from the denoised representations using either Fisher Discriminant Analysis (FDA) or Sparse Model Attribution (SMA). Finally, in the third stage, a second round of LDP perturbation is performed to balance privacy and utility. The authors have performed evaluation on 6 small to medium scale datasets in the main paper and 2 additional in the appendix.

### Strengths
1. The flow of the introduction, along with the motivation for the proposed framework, is good. Overall, the paper is well-motivated and nicely written.

2. The three-stage framework is intuitive and easy to understand.

3. TOGL demonstrates strong utility improvements compared to baseline LDP methods.

4. The method performs well across various GNN architectures.

### Weaknesses
1. I believe the authors should at least mention experiments on large-scale datasets and robustness evaluations in the main text.

2. The method relies on access to task-specific signals, which may not always be practical in real-world scenarios.

3. The motivations for using FDA and SMA as feature-selection modules should be discussed, along with an analysis of how sensitive the algorithm is to this choice.

4. Could the authors also provide fairness evaluations for the baselines in Table 6?

5. There should be a discussion on the selection of the hyperparameter $\rho$ for practical deployment.

6. The neighborhood aggregation used for denoising may be detrimental for heterophilic datasets.

### Questions
See weakness.

### Soundness
3

### Presentation
4

### Contribution
3
