# The Final Layer Holds the Key: A Unified and Efficient GNN Calibration Framework

- Decision: Reject
- Scores: 6, 6, 4, 8

## Abstract
Graph Neural Networks (GNNs) have demonstrated remarkable effectiveness on graph-based tasks. However, their predictive confidence is often miscalibrated, typically exhibiting _under-confidence_, which harms the reliability of their decisions. Existing calibration methods for GNNs normally introduce additional calibration components, which fail to capture the intrinsic relationship between the model and the prediction confidence, resulting in limited theoretical guarantees and increased computational overhead. To address this issue, we propose a simple yet efficient graph calibration method. We establish a unified theoretical framework revealing that model confidence is jointly governed by class-centroid-level and node-level calibration at the final layer. Based on this insight, we theoretically show that reducing the weight decay of the final-layer parameters alleviates GNN under-confidence by acting on the class-centroid level, while node-level calibration acts as a finer-grained complement to class-centroid level calibration, which encourages each test node to be closer to its predicted class centroid at the final-layer representations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper provides a theoretical and practical framework (SCAR) for calibrating GNNs by adjusting final-layer weight decay and introducing a node-level calibration step. It proves that reducing final-layer regularization enlarges class-centroid distances, alleviating under-confidence. A post-hoc correction then nudges node embeddings toward their predicted class centroids, further improving calibration.

### Strengths
1. Strong theoretical grounding linking weight decay to confidence underestimation.
2. Dual-level calibration (centroid + node) unifies model-intrinsic and post-hoc methods.
3. Training-free node-level adjustment ensures efficiency and interpretability.
4. Extensive experiments show lower ECE and runtime than prior calibrators

### Weaknesses
I haven't seen too much weakness. But I am wondering whether node-level adjustment may amplify overconfidence for misclassified nodes.

### Questions
1. How does the method behave when the predicted class centroid is incorrect?
2. Can the framework generalize to heterophilous or dynamic graphs? (Optional)
3. How sensitive are results to the final-layer λ schedule?
4. Could centroid regularization be integrated during training for joint optimization?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SCAR, a nonparametric GNN calibration framework that analyzes miscalibration of GNNs in model perspective. SCAR theoretically shows that reducing a weight decay in the final layer leads to higher class separation, which can mitigate underconfidence of GNNs. It further performs node-level calibration to make the test node’s representation closer to the corresponding label’s centroid.

### Strengths
- This paper provides a theoretical connection between underconfidence of GNNs and final layer’s weight decay, which is valuable given the lack of theoretical analysis in GNN calibration literature.
- The proposed method is simple yet effective, avoiding the need to train additional calibration networks as required by many existing methods.
- Extensive experiments shows that SCAR substantially reduces ECE compared to prior baselines, as well as maintaining original classification accuracy of GNNs.

### Weaknesses
- The proposed node-level calibration assumes that pushing test nodes toward their predicted class centroids improves confidence, which may not hold under settings such as out-of-distribution (OOD) conditions. For instance, in OOD graphs, pushing test nodes toward centroids learned from training data can degrade calibration.
- If the original GNNs are trained with zero weight decay, the proposed method may be partially inapplicable.
- While SCAR is efficient, it needs to search the optimal configuration over three hyperparameters. Although the authors offer practical heuristics in the appendix (e.g., $\alpha$ should be lower than $\beta$), it is not guaranteed that such heuristics hold universally.

### Questions
- Could the authors show the performance of SCAR on OOD graphs?

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
5

### Summary
In this paper, the authors conduct a comprehensive analysis of confidence calibration in Graph Neural Networks (GNNs). They first theoretically demonstrate that weight decay applied to the final-layer parameters exacerbates under-confidence by collapsing class centroids toward the origin, thereby reducing class separability. To address this, the authors propose reducing the final-layer weight decay to enhance inter-class distinction and improve confidence calibration at the class-centroid level. Additionally, they introduce a node-level calibration strategy as a fine-grained complement, which encourages each test node to move closer to its predicted class centroid while distancing itself from others in the final-layer representation space, thus improving individual calibration. Finally, they develop a unified theoretical framework that shows model confidence is jointly governed by both class-centroid-level and node-level calibration, underscoring the completeness and coherence of their approach. Extensive experiments demonstrate that the proposed method consistently outperforms state-of-the-art techniques in terms of both effectiveness and efficiency across various datasets and settings.

### Strengths
1. The authors are the first to theoretically show that final-layer weight decay aggravates GNN under-confidence, and they mitigate this by reducing the decay.

2. They propose a training-free node-level calibration method as a fine-grained complement to class-centroid-level calibration.

3. They develop a unified theoretical framework showing that both calibration levels jointly govern model confidence, and validate the method’s superiority across diverse settings.

### Weaknesses
1. Missing important related work: Given that the paper focuses on confidence calibration, it is concerning that several key papers in the area of uncertainty estimation or calibration for GNNs are not cited or discussed [1-4].

2. Limited baselines: The experimental comparisons would benefit from the inclusion of recent calibration methods [5]

3. Restricted backbone models: The authors only evaluate their method on GCN and GAT. While these are classical models, they are no longer sufficient to represent the landscape of modern GNN architectures. Including additional backbones like GraphSAGE would strengthen the empirical claims and validate the method’s generality.

[1] Uncertainty quantification over graph with conformalized graph neural networks. NeurIPS 2023

[2] Energy-based Epistemic Uncertainty for Graph Neural Networks

[3] Uncertainty Aware Semi-Supervised Learning on Graph Data. NeurIPS 2020

[4] Calibrate Automated Graph Neural Network via Hyperparameter Uncertainty. CIKM 2022

[5] GETS: Ensemble Temperature Scaling for Calibration in Graph Neural Networks. ICLR 2025

### Questions
see Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the problem of confidence miscalibration in graph neural networks. The authors observe that GNN confidence is influenced by two factors in the final layer, namely Class-Centroid-Level Calibration and Node-Level Calibration. Building on this insight, they propose the SCAR framework, which unifies these two calibration components into a single theoretical framework, enabling more effective confidence calibration in GNNs.

### Strengths
1. This is the paper's most significant strength. It moves beyond heuristic-based calibration by providing a rigorous theoretical analysis.

2. The proposed SCAR method consistently outperforms a wide range of strong baselines across multiple datasets.

### Weaknesses
1. The node-level calibration is refined in Eq. 10 to account for the structural bias of GNNs (nodes closer to training data get more similar representations). While this is a thoughtful addition, its evaluation is limited. An ablation study showing the performance gain of using two parameters $\alpha$ and $\beta$ over a single one would have strengthened this claim.

2. The details of the high-order neighbors of the training node is not well specified.

3. Sensitivity analysis on hyper-parameter $\lambda^{(k)}$ is not provided.

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
3
