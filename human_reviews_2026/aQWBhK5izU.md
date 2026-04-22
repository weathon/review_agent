# Credal Graph Neural Networks for Robust Uncertainty Quantification

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Uncertainty quantification is essential for deploying reliable Graph Neural Networks (GNNs), where existing approaches primarily rely on Bayesian inference or ensembles. In this paper, we introduce the first credal graph neural networks (CGNNs), which extend credal learning to the graph domain by training GNNs to output set-valued predictions in the form of credal sets. To account for the distinctive nature of message passing in GNNs, we develop a complementary approach to credal learning that leverages different aspects of layer-wise information propagation. We assess our approach on uncertainty quantification in node classification under out-of-distribution conditions. Our analysis highlights the critical role of the graph homophily assumption in shaping the effectiveness of uncertainty estimates. Extensive experiments demonstrate that CGNNs deliver more reliable representations of epistemic uncertainty and achieve state-of-the-art performance under distributional shift on heterophilic graphs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to predict credal sets for node classification from the concatenated embeddings of a node in a GNN. It combines established paradigms from UQ for GNNs with credal learning to provide uncertainty estimates that are roughly competitive with existing approaches.

### Strengths
- The paper is easy to follow and provides an extensive and visual explanation of the methods.
- The proposed approach is well-motivated as it follows principles from credal learning (CL) and uncertainty quantification (UQ) for GNNs.

### Weaknesses
- The paper's novelty is limited as it applies a well-established framework for credal learning to the joint representations in a GNN, a paradigm that has recently been studied to inform UQ for heterophilic graphs. There are also no novel theoretical insights for applying credal learning to GNNs using the embeddings from all layers to compensate.

- The empirical performance of the approach is not convincing: The proposed epistemic uncertainty only performs well on 2/6 graphs. Additionally, the predictive performance of `CredalLJ` deteriorates notably on many datasets (see Table 3). This makes the method impractical. 

- The evaluation misses several relevant baselines: Graph PostNet [2] is a well-established method for UQ, and also recent advancements like GEBM [3] or (ICML paper)[4] are not considered.

- For a paper whose contribution is limited to combining existing methods, the experimental studies are limited. The authors ablate three variants of their approach. Following [4], it would be interesting to see if there are certain layers beyond the final one that drive UQ performance. Additionally, it could be studied if the credal GNNs capture homophily / heterophily patterns for the respective graphs by predicting smooth / unsmooth credal sets between neighbours. There is also no study on using different GNN backbones to validate the applicability of the approach beyond GCN and SAGE -- and even for these two, there is no comprehensive study on if there are performance differences between the backbones.

- The structure of the paper, especially regarding related work, can be improved: The related work section extensively discusses UQ in general but neglects both credal learning as well as works that focus on UQ for GNNs [GPN, GEBM, The recent ICML paper]. Some other references are spread throughout other sections (like JLDE [5] or some prior work on CL), but a bit misplaced. For example, the section starting at L. 198 ff. elaborates on theoretical concepts that have little impact on the proposed approach, which just concatenates latent embeddings. 

- The evaluation misses some well-established datasets in node classification, like CoraML, PubMed, Citeseer, as well as recent heterophilic benchmarks like Roman Empire and Tolokers [1]. 

## References
[1] Platonov, Oleg, et al. "A critical look at the evaluation of GNNs under heterophily: Are we really making progress?." arXiv preprint arXiv:2302.11640 (2023).
[2] Stadler, Maximilian, et al. "Graph posterior network: Bayesian predictive uncertainty for node classification." Advances in Neural Information Processing Systems 34 (2021): 18033-18048.
[3] Fuchsgruber, Dominik, Tom Wollschläger, and Stephan Günnemann. "Energy-based epistemic uncertainty for graph neural networks." Advances in Neural Information Processing Systems 37 (2024): 34378-34428.
[4] Fuchsgruber, Dominik, et al. "Uncertainty Estimation for Heterophilic Graphs Through the Lens of Information Theory." arXiv preprint arXiv:2505.22152 (2025).

### Questions
- The interval SoftMax is typically proposed in terms of $0.5 (a_v^{U_k} + a_v^{L_k})$ [5]. Why do the authors use $a_v^{U_k}$ and $a_v^{L_k}$ alone respectively in Eq. (4) instead?
- Regarding the heterophilic datasets: Did the authors use the revised versions of these datasets as proposed in [1] to avoid data leakage?


## References
[1] Platonov, Oleg, et al. "A critical look at the evaluation of GNNs under heterophily: Are we really making progress?." arXiv preprint arXiv:2302.11640 (2023).
[5] Wang, Kaizheng, et al. "Credal deep ensembles for uncertainty quantification." Advances in Neural Information Processing Systems 37 (2024): 79540-79572.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces Credal Graph Neural Networks, a novel framework for robust uncertainty quantification in Graph Neural Networks. The core contribution is extending credal learning to the graph domain, allowing the model to output set-valued predictions that formally disentangle aleatoric and epistemic uncertainty. The key model, CredalLJ, addresses the limitations of standard GNNs on heterophilic graphs by leveraging a concatenated joint latent representation from all layers of the GNN backbone.

### Strengths
1. This is the first work to apply the robust framework of credal learning to Graph Neural Networks, providing a principled alternative to purely Bayesian or ensemble-based UQ methods
2. The proposed CredalLJ architecture is well-motivated by the theoretical limitations of message passing in GNNs, especially the challenge of heterophily.
3. The credal set formulation effectively provides a separate and interpretable measure for aleatoric and epistemic uncertainty, which is shown to be crucial for successful OOD detection

### Weaknesses
1. The CredalLJ approach uses a joint latent representation by concatenating all layer embeddings. For deep GNNs or large-feature inputs, this significantly increases the input dimensionality to the final Credal Layer, potentially introducing a considerable computational and memory overhead that is not fully quantified or discussed.
2. In the analysis section, visually comparing the credal sets output for a few representative in-distribution and OOD nodes would intuitively reinforce the benefit of the architecture.

### Questions
Beyond the OOD detection experiments (which test test-time shifts), did the authors investigate the robustness of CGNNs when the training data itself contains noise or mislabeled nodes, a common issue in real-world graph data?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Credal Graph Neural Networks (CGNNs), which extend credal learning to the graph domain by training GNNs to output set-valued predictions represented as credal sets. The model quantifies both aleatoric and epistemic uncertainties through two output heads that learn the lower and upper bounds of class probabilities. It further aggregates latent representations from all GNN layers to capture information across message-passing steps. The evaluation is conducted on out-of-distribution (OOD) detection tasks.

### Strengths
1. The paper is clearly written and well structured.
2. It is the first work to apply credal learning to graph neural networks for uncertainty estimation.
3. The use of a Distributionally Robust Optimization (DRO) loss is well motivated, encouraging robustness under distributional shifts by combining optimistic (upper-bound) and pessimistic (lower-bound) predictions.

### Weaknesses
1. The model simply concatenates embeddings from all layers, which may not adaptively capture homophily or heterophily information.
2. The set of baselines is limited; evidential methods (e.g., GPN [1], SGCN) and diffusion-based uncertainty models (e.g., [2, 3]) are missing.
3. The proposed approach shows a substantial drop in classification accuracy, which limits its practicality, as uncertainty estimation is often tied to reliable class prediction.
4. The model selection procedure relies on validation OOD detection performance, which is inappropriate for OOD detection tasks where OOD samples should remain unknown during training and validation.
5. The separation between aleatoric and epistemic uncertainty is not clearly validated. For instance, Table 1 shows cases where aleatoric uncertainty outperforms epistemic uncertainty in OOD detection, which contradicts the common expectation that epistemic uncertainty should dominate in such scenarios [1].

[1] Stadler, Maximilian, et al. "Graph posterior network: Bayesian predictive uncertainty for node classification." NeurIPS 2021.

[2] Fuchsgruber, Dominik, et al. "Uncertainty Estimation for Heterophilic Graphs Through the Lens of Information Theory." ICML 2025.

[3] Fuchsgruber, Dominik, Tom Wollschläger, and Stephan Günnemann. "Energy-based epistemic uncertainty for graph neural networks." NeurIPS 2024

### Questions
See Weaknesses

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
3

### Summary
This paper introduces Credal Graph Neural Networks (CGNNs), which extend credal learning to graphs by training GNNs to produce set-valued (credal) predictions that capture epistemic uncertainty. The authors propose a layer-wise information propagation framework suited to message passing, and demonstrate through experiments on node classification under distribution shift that CGNNs provide more reliable uncertainty estimates and outperform existing methods, especially on heterophilic graphs.

### Strengths
1. This paper studies a challenging scenario in UQ of GNNs where the homophily assumption weakens and propose a solution to address it.

2. Good presentation, such as Figure 2, illustrate the main idea of the work clearly.

3. Experiments on extensive datasets shows the effectiveness of the proposed approach.

### Weaknesses
1. There is no experimental comparison between post-hoc credal learning and modification of the base GNN model with a credal layer.

2. Some important baselines, such as [1], are not included.

[1] Wang, Xiao, et al. "Be confident! towards trustworthy graph neural networks via confidence calibration." Advances in Neural Information Processing Systems 34 (2021): 23768-23779.

### Questions
1. Can you further explain how eq.(2) is related to Figure 1. It seems that AU and EU are defined differently in the two places.

2. Can your method be extended to link prediction?

### Soundness
2

### Presentation
3

### Contribution
3
