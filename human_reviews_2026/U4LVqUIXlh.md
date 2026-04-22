# FedSal: Enhancing Federated Graph Classification Through Saliency Aware Client Clustering

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Graph Neural Networks (GNNs) are essential for analyzing structured data but face significant challenges in federated learning (FL) environments, where non-IID client distributions and structural heterogeneity impede convergence and performance. To address these issues, we introduce Federated Saliency Aggregation Learning (FedSal), the first framework to apply saliency maps in GNN-based FL on graph classification tasks. FedSal replaces full-gradient uploads with compact saliency activations, enabling dynamic clustering of clients via simple thresholds (\(\epsilon_{\mathrm{mean}}, \epsilon_{\max}\)) and cluster-wise model averaging. We further propose FedSal+, which augments node features with positional and random-walk encodings to inject structural priors without exposing raw graph data. Extensive experiments on thirteen molecular, protein, and social-network benchmarks under extreme non-IID splits show that FedSal and FedSal+ achieve higher accuracy, converge faster, and reduce communication cost compared to state-of-the-art methods. These results demonstrate the SOTA performance of saliency-driven clustering for personalized, robust, and communication-efficient federated graph classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a federated learning model that exploits saliency maps/scores for client clustering and aggregation.

### Strengths
The paper uses saliency maps to perform clustering of clients in a federated learning setting before applying the standard FedAvg method in each cluster. The use of saliency maps seems to be novel in the context of federated learning.

### Weaknesses
1. The novelty of the work mainly lies in the use of saliency maps, which is a common concept from various branches of machine learning like computer vision. The definition of this important saliency map is also unclear.

1. The assumptions required for the theoretical results to hold are very limiting.

1. Baseline benchmarks are not up to date.

### Questions
1. It is unclear why the proposed approach is restricted to graph federated learning when the approach does not actually utilize anything specific to graphs, except for FedSal+, which integrates features from FedStar. It is curious why the authors chose to position the paper on federated graph learning.

1. Assumptions made for the theoretical results should be stated upfront and discussed. At the minimum, readers should have been alerted to these instead of having to find out from the appendix that the results are actually extremely limited. For example, the requirement that the feature matrix has full column rank makes the results impractical. In small graphs, this is automatically not satisfied. In big graphs with feature dimensions smaller than the number of nodes, it is not obvious we will have full column rank. Numerical evidence based on common graph datasets can be provided to check this.

1. I am unclear why Prop. 3 implies that FedSal can capture task heterogeneity when it says the differences between saliency maps are bounded. Don't we want the saliency maps to be very different if the tasks are different? Won't similar saliency maps lead to wrong clustering?

1. To compute the saliency value in Section 2.1.3, is there an implicit assumption that the loss is differentiable w.r.t. the sample value $x$ and $x$ is a continuous variable? How do you deal with the case where $x$ has categorical components? Furthermore, it is unclear how this saliency can be computed easily. The formula as written is not explained properly: does $|\cdot |$ denote determinant or elementwise absolute value?

1. Does the feature $x$ have to be the same dimensions for all clients? If so, does this not mean that local client models have the same architecture?

1. The comparison is missing many important baselines like FedPer, pFedGraph, FedSheafHN, Flow, HeteroFL, FedRolex, etc.

### Soundness
3

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
4

### Summary
The paper proposes FedSal (Federated Saliency Aggregation Learning), a GNN-based federated learning framework designed to mitigate performance degradation under non-IID and structurally heterogeneous conditions. FedSal replaces full gradient uploads with saliency maps, and employs two thresholds (ϵ_mean, ϵ_max) to trigger dynamic client clustering and intra-cluster aggregation. An extended version, FedSal+, further incorporates positional and random-walk encodings to inject structural priors. Experiments on thirteen molecular, protein, and social-network datasets show that FedSal/FedSal+ achieve smoother convergence and an average accuracy improvement of about +1.5-2.0pp under extreme non-IID settings, though the overall gain is limited.

### Strengths
1. Novel concept: Introduces saliency maps into federated graph learning and proposes a threshold-triggered dynamic clustering mechanism.

2. Broad experimental coverage: Evaluated on thirteen diverse datasets and compared against multiple strong baselines.

3. Stable performance under non-IID conditions: Shows smoother and faster convergence trends when data heterogeneity is high.

### Weaknesses
1. The average gain (+1.5–2.0pp) is small and comparable to the reported standard deviation. Only accuracy (ACC) and communication time are reported; lacks F1, AUC, or Recall for a more comprehensive evaluation.

2. The paper does not provide any time or space complexity for the main procedures, including similarity graph construction, Stoer–Wagner min-cut, and recursive clustering.

### Questions
1. Could the authors provide a theoretical convergence analysis or at least an empirical justification of the claimed “faster and smoother convergence”?

2. What is the computational complexity of the clustering and aggregation process as a function of client number and feature dimension?

3. Have the authors considered adding statistical significance tests (e.g., t-test) to verify whether the observed accuracy gains are meaningful?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes FedSal, a novel framework that uses saliency maps to cluster clients in Federated Graph Neural Networks (FedGNNs), addressing non-IID data challenges that cause instability and slow convergence in federated learning. Evaluated on 13 graph classification tasks (molecular, protein, and social networks), FedSal and its enhanced version outperform state-of-the-art methods in accuracy, convergence speed, and communication efficiency.

### Strengths
Using saliency maps for clustering in FedGNNs is a novel approach that enhances interpretability and robustness. The framework effectively tackles non-IID data challenges, improving model convergence.

### Weaknesses
1.FedSal reduces communication costs versus some SOTA methods but has higher latency than simpler approaches like FedAvg.
2.Experiments focus on classification; generalization to other graph tasks (e.g., regression, link prediction) is unexplored.
3.Dynamic clustering requires careful tuning of thresholds (ϵ-mean, ϵ-max), posing a barrier for users lacking parameter-tuning expertise.

### Questions
1. How does the framework performs in large-scale, real-world deployments, considering real-time adaptation of the model to constantly changing client data distributions?
2. How does the framework address potential privacy issues in detail, even though it deals with federated learning, which is often implemented in privacy-sensitive environments?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This manuscript proposes FedSal and FedSal+, two novel federated graph neural network frameworks that leverage saliency activation maps for client clustering to address non-IID data distribution and structural heterogeneity challenges in federated graph learning. The work is well-motivated, with rigorous theoretical analysis, comprehensive experiments across 13 benchmarks, and clear articulation of technical innovations. The proposed methods demonstrate superior performance in accuracy, convergence speed, and communication efficiency compared to state-of-the-art baselines, making a valuable contribution to the federated graph learning field.

### Strengths
1. The use of saliency activation maps (instead of raw gradients) for client clustering is a novel and effective design. Saliency maps capture feature importance for predictions, providing more stable and representative client summaries than noisy raw gradients, which directly mitigates the impact of non-IID data and structural heterogeneity. 
2. Extensive experiments not only compare accuracy but also analyze communication overhead and convergence speed, providing a holistic assessment of the proposed methods’ performance. Ablation studies further validate the necessity of saliency maps and dynamic clustering.
3. The manuscript provides solid theoretical support, including proofs for the structural, feature, and task sensitivity of saliency maps, ensuring the method’s theoretical soundness. Additionally, FedSal+ extends the core framework with positional and random-walk encodings, demonstrating the flexibility and scalability of the proposed architecture to integrate structural priors without exposing raw data.

### Weaknesses
1. The manuscript mentions aggregating per-sample saliency maps for each client but does not explicitly clarify how saliency is computed for graph-structured data (e.g., node-level vs. graph-level saliency aggregation, handling of edge features). More details on this implementation would improve reproducibility.
2. The baselines in this manuscript are mainly from 2024 or earlier. I am not very familiar with this field, but I believe that new baselines have likely been proposed in 2025. The authors should include comparisons with these recent methods to demonstrate that their approach remains state-of-the-art.
3. While the hyperparameter study ($ϵ_{mean}$, $ϵ_{max}$) shows optimal mid-range values, the analysis is limited to specific datasets. It would be valuable to discuss how these thresholds generalize across different graph types (e.g., sparse vs. dense graphs) or provide adaptive tuning strategies for real-world applications.

### Questions
Note: Here are some questions I have after reading the manuscript. Since I am not very familiar with this field, the authors may choose to answer selectively, focusing on issues related to the manuscript's contributions. I will also take into account comments from other reviewers when forming my final rating.

1. For graph-level classification tasks, how is the per-sample saliency map aggregated to form the client-level saliency summary? Is there a weighting mechanism for different graphs (e.g., based on graph size or classification confidence)?

2. The manuscript mentions that FedSal+ injects structural priors via positional and random-walk encodings. How do these encodings interact with the original node features, and is there any redundancy or interference between them? Could the authors provide quantitative analysis of this interaction?

3. In the multi-dataset setting, how does FedSal handle domain shifts between different types of graphs (e.g., molecular vs. social network graphs)? Does the clustering mechanism automatically separate domain-specific clients, and if so, how is this verified?

4. For minority-label clients, the manuscript states that the adaptive protocol provides significant benefits. Could the authors elaborate on how the clustering thresholds (ϵmean, ϵmax) specifically protect minority clients from being marginalized in the aggregation process?

### Soundness
3

### Presentation
3

### Contribution
3
