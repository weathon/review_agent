# Energy Guided Smoothness to Improve Robustness

- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
Graph Neural Networks (GNNs) perform well on graph classification tasks but are notably susceptible to label noise, leading to compromised generalization and overfitting. We investigate GNNs' robustness, identify generalization failure modes and causes, and prove our hypotheses with three robust GNN training methods. Specifically, GNN generalization is compromised by label noise in simpler tasks (few classes), low-order graphs (few nodes), or highly parameterized models. Focusing on graph classification, we show the link between GNN robustness and the smoothness of learned node representations, as quantified by the Dirichlet energy. We show that GNN learns smoother representations with decreasing Dirichlet energy during training, until the model fits on noisy labels, adding high-frequency components to the representations. To verify our analysis, we propose three robust training strategies for GNNs: (a) a spectral inductive bias by enforcing positive eigenvalues in GNN weight matrices to demonstrate the link between smoothness and robustness; (b) a Dirichlet energy overfitting control mechanism, which relies on a noise-free validation set; (c) a noise-robust loss function tailored for GNNs to induce smooth representations. Crucially, our methods do not degrade performance in noise-free data, reinforcing our central hypothesis that GNNs’ smoothness bias defines their robustness to label noise.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper examines the robustness of GNNs to label noise in the graph classification task. They empirically demonstrate that there is a correlation between the increase of Dirichlet energy when overfitted to noisy labels. Based on this finding, the paper proposed three training strategies: enforcing positive eigenvalues to focus on low-pass signals, explicit regularization of the Dirichlet energy, and a noise-robust loss function tailored for GNNs. Experimental results by comparing distinct noise levels further strengthen the contribution of this work.

### Strengths
- The paper tackles the challenge of noisy labels in graph classification, a problem that has received relatively limited attention compared to node classification.
- Through extensive empirical analyses in Section 4, the authors provide insights into the conditions under which GNNs fail to maintain robustness, and in Section 5, they establish a novel connection between this phenomenon and the Dirichlet Energy, demonstrating notable originality and potential for future research.
- Comprehensive experiments under both symmetric and asymmetric noise settings further substantiate the effectiveness and robustness of the proposed approach.

### Weaknesses
- The adopted GNN model primarily focuses on capturing low-pass signals. It would be beneficial to include a broader range of GNN architectures that can capture both low- and high-pass components, such as [1] and [2], to validate the generality of the findings.
- Figures 1 and 2 present results only under the noisy-label setting. If similar accuracy improvements are observed on clean graphs, the interpretation and discussion in Section 4 may require reconsideration.
- The claim that noise predominantly corresponds to high-frequency signals could be further supported through additional references or empirical evidence, which would help reinforce the justification of Method 1.
- Table 3 shows minor improvements compared to existing works.

Minor errors:
- Some expressions in the paper (maybe because of the usage of \eqref) should be modified (line 373, 849, and so on).
- The abbreviation GCOD should be explicitly defined upon its first appearance.
- Inconsistent expression in line 267 and equation 3, $\mathcal{G}_i$ and $\mathcal{G}^i$.


[1] GREAD: Graph Neural Reaction-Diffusion Networks, ICML 2023
[2] From Trainable Negative Depth to Edge Heterophily in Graphs, Neurips 2023

### Questions
1. Could the authors provide a more detailed explanation of how symmetric and asymmetric noise are defined, including the specific noise injection procedure used in the experiments?
2. In case of utilizing class-specific bounds for method 2, does the number of graphs for each class in the validation set affect the overall performance?
3. Rather than tuning hyperparameters on a clean graph, how would the performance trends change if the optimal hyperparameters were selected based on the noisy graph?
4. In Table 2, do other datasets exhibit a similar performance pattern between CE + W2 and the fixed, class-specific method as observed in the PPA dataset?

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
4

### Summary
The paper presents an interesting connection between noisy label (model robustness) and Dirichlet energy under graph classification task. Specifically, from experiments, the author observes that models overfitting to noisy labels tend to have high Dirichlet energy rising and decreasing Dirichlet energy could provide more robust graph classification model. The author attemps with three different strategies to improve robustness of models under noisy labels by constraining learnable weight to positive eigenvalue only, adding Dirichlet energy as part of smoothness regularization loss, and a method inspired from image classification called GCOD loss. It is observed that despite not directly related to decreasing Dirichlet energy, GCOD loss still provide robust model and decrease Dirichlet energy.

### Strengths
1.The paper shows a clear observation between the connection of Dirichlet energy and label noise.  
2.The paper provides comprehensive experiments on when models are overfitting to the noisy data and conclude with high dirichlet energy observation and propose three methods to alleviate it.

### Weaknesses
1. The proposed method show no clear difference between existing methods such as OMG as shown in Table 3.
2. The GCOD method description is too sparse in the main text, lacking detailed descriptions.
3. Why detailed results are only reported for PPA dataset with other baselines?

### Questions
1. I hate to ask but I have to ask again: why is the figure 3 legend still has the same typo? Is this something that is hard to change even after someone explicitly points it out?
2. See weakness.

### Soundness
3

### Presentation
1

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
This paper focuses on the problem of overfitting and generalization degradation in graph classification tasks with noisy labels. The authors propose using Dirichlet energy as an observable signal to assess the degree of overfitting in GNNs and design three complementary strategies to enhance robustness. Experiments show that these methods significantly improve classification accuracy under label noise on several datasets, including PPA.

### Strengths
- The paper is well-written and easy to follow.

- The insights derived from the pre-experiments are compelling and directly motivate the three proposed methods.

### Weaknesses
- Both the pre- and main experiments rely on synthetic symmetric/asymmetric noise, rather than real-world noisy labels, limiting the external validity and generalizability of the conclusions.

- Dirichlet energy is a global measure for the whole dataset, while the proposed methods optimize at the sample level. This mismatch raises questions about scalability and effectiveness when applied to larger, more complex real-world graphs.

- The GCOD loss depends on training accuracy as feedback, which may introduce bias or instability during training. The paper does not thoroughly investigate or ablate this potential issue.

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the robustness of GNNs to noisy labels in graph classification tasks. Specifically, the authors investigate how GNNs leverage smooth representations as an inductive bias for generalization and noise robustness—an effect that can be disrupted by noisy labels. And then, three ways of avoiding GNN fitting on the noise are discussed and evaluated.

### Strengths
1. This work is the first to study the link between label noise robustness and the spectral dynamics of Dirichlet energy in graph classification.
2. It shows that three different approaches to improving robustness can be understood from an energy-based perspective that constrains harmful high-frequency energy.
3. Experimental results verify the effectiveness of the proposed GCOD loss.

### Weaknesses
1. Based on the definition of Dirichlet energy in Equation (2), the discussion in the paper is restricted to homophilic graphs, which makes the proposed improvement less broadly applicable.
2. A wider range of real-world datasets and a more detailed analysis of the results are required to validate the generality of the smoothness assumption and to clarify what the proposed loss function contributes to specific graph structures.

### Questions
See the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1
