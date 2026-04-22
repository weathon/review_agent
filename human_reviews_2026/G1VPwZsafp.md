# CITED: A Decision Boundary-Aware Signature for GNNs Towards Model Extraction Defense

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Graph neural networks (GNNs) have demonstrated superior performance in various applications, such as recommendation systems and financial risk management. However, deploying large-scale GNN models locally is particularly challenging for users, as it requires significant computational resources and extensive property data. Consequently, Machine Learning as a Service (MLaaS) has become increasingly popular, offering a convenient way to deploy and access various models, including GNNs. However, an emerging threat known as Model Extraction Attacks (MEAs) presents significant risks, as adversaries can readily obtain surrogate GNN models exhibiting similar functionality. Specifically, attackers repeatedly query the target model using subgraph inputs to collect corresponding responses. These input-output pairs are subsequently utilized to train their own surrogate models at minimal cost. Many techniques have been proposed to defend against MEAs, but most are limited to specific output levels (e.g., embedding or label) and suffer from inherent technical drawbacks. To address these limitations, we propose a novel ownership verification framework CITED which is a first-of-its-kind method to achieve ownership verification on both embedding and label levels. 
Moreover, CITED is a novel signature-based method that neither harms downstream performance nor introduces auxiliary models that reduce efficiency, while still outperforming all watermarking and fingerprinting approaches. Extensive experiments demonstrate the effectiveness and robustness of our CITED framework. Code is available at: \url{https://anonymous.4open.science/r/CITED}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CITED, a unified framework for defending against GNN model extraction attacks (MEAs) via decision boundary-aware signatures. It enables ownership verification at both embedding and label levels, addressing limitations of existing watermarking (label-level only) and fingerprinting (embedding-level only) methods. CITED generates signatures from boundary sensitive nodes, uses Wasserstein distance and prediction accuracy for verification, and achieves good performance across seven datasets and five GNN backbones without harming downstream utility or requiring auxiliary models.

### Strengths
1. Avoids task-irrelevant triggers, maintaining or even improving model performance on node classification tasks.

2. Eliminates auxiliary model training, reducing computational overhead, and retains effectiveness under MEAs and removal attacks (pruning, fine-tuning).

2. Provides probabilistic bounds for embedding similarity and prediction agreement.

### Weaknesses
1. Regarding the necessity of integrating the embedding and label levels, this seems more like a technical assumption. Could the authors provide some insight into why this integration is necessary? Watermarking or fingerprinting methods, for example, do not usually focus on both embedding and label levels. Moreover, labels can be viewed as embeddings passed through a classifier, so the nature of embeddings and labels may not differ significantly.

2. In line 245, the authors claim that “as the decision boundary captures the most distinctive and model-specific behavior.” However, nodes near the decision boundary are typically hard to distinguish. The repeated assumption in the paper seems questionable and may be hard to justify.

3. Can this method be extended to regression models or graph classification datasets? The current approach appears to be limited to node classification.

4. In line 157, the authors mention real-world applications, citing citation networks. However, MEA attacks are not particularly relevant for citation networks. In contrast, fields like molecular structure prediction are more common MLaaS scenarios, as mentioned in the paper. The authors should provide experiments on such datasets; For now, the practical applicability seems limited.

5. The theoretical conclusions appear to have issues. In Theorem 1, when $\lambda = \Delta G$, substituting into Equation 8 gives $\Pr(D_p < \lambda) \geq 0$ instead of 1. This raises concerns about the validity of the derivation and why there seems to be this truncation effect.

### Questions
See weakness.

### Soundness
3

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
3

### Summary
This paper proposes a decision boundary-aware signature framework, CITED, for defending graph neural networks against model extraction attacks. The paper introduces a method for extracting task-relevant signatures from nodes near the model's decision boundary, enabling unified embedding/label-level verification without the need for auxiliary models. Experiments demonstrate improved performance of verification over standard watermarking and fingerprinting methods.

### Strengths
- The paper proposes a novel signature-based ownership verification mechanism that operates at both embedding and label levels, addressing a limitation in prior work.  
- This paper presents an efficient verification framework, improving scalability over conventional fingerprinting methods.  
- The proposed method achieves good verification performance  across various datasets and GNN architectures while providing probabilistic guarantees on signature preservation.

### Weaknesses
- The framework's experimental validation is confined to a single threat model; testing it across multiple MEA scenarios would more effectively demonstrate its robustness.  
- While the method shows efficiency advantages, its scalability to graphs with millions of nodes remains unverified; its signature generation (involving boundary node identification and multi-metric scoring) and verification workflows may face computational or memory bottlenecks in such scenarios.  
- The framework’s reliance on the target model’s decision boundaries for signature generation may potentially lead to instability or diminished distinguishability when the model undergoes fine-tuning or domain adaptation.

### Questions
I have the following questions regarding the practicality of CITED:  

- Is the signature mechanism applicable in real-world settings with noisy labels or significant class imbalance, and does its reliance on the decision boundaries maintain robustness under such scenarios?  
- Can the signature set be compressed or structurally optimized to reduce storage or computational demands without compromising verification reliability?  
- Is there a minimum number of signature nodes required to ensure verification confidence, and how does this threshold vary with graph scale or model capacity?  
- The rating will be adjusted accordingly if the above concerns are properly addressed.

### Soundness
3

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
3

### Summary
This paper addresses the problem of protecting proprietary GNNs deployed under the Graph Machine Learning as a Service paradigm, where models are accessed through APIs but are vulnerable to model extraction attacks (MEAs). Existing defenses, such as watermarking and fingerprinting, are limited because they operate only at either the label level or the embedding level, making them ineffective when attackers extract models through different output interfaces. 

To overcome this limitation: 

* The paper proposes **CITED**, a unified ownership verification framework based on a decision boundary–aware signature that functions across both embedding-level and label-level outputs. The signature arises naturally from the model’s decision boundary and can be transferred to surrogate models through extraction, allowing ownership to be verified without inserting task-irrelevant triggers. 
* The framework also introduces an efficient verification protocol that avoids training additional auxiliary models by extending the ARUC metric with the Wasserstein distance.
* Experiments demonstrate that CITED consistently outperforms existing MEA defense methods and enables robust ownership verification across diverse output settings.

### Strengths
The technical contribution of this paper is solid. The proposed signature effectively addresses the challenge of achieving unified ownership verification at both the embedding level and the label level under model extraction attacks. Moreover, the theoretical analyses provided in the paper are generally sound and offer clear justification for using the 2-Wasserstein distance as the metric for verification.

### Weaknesses
The experimental evaluation in this paper is somewhat limited. For example, only one model extraction attack (GNNStealing) is adopted as the threat model, and the study focuses solely on the classical node classification task. Additionally, I suggest improving the presentation by introducing an overview figure to provide a clearer illustration of the CITED framework.

### Questions
Here are some questions:

* In Section 5.2, how is the defense model trained? The current description lacks sufficient detail.

* The performance of CITED on the Photo dataset appears noticeably weaker compared to other datasets. Could you provide an explanation for this discrepancy?

* I don’t quite understand the motivation for introducing **Heterogeneity**, and it does not seem to provide clear benefits (as shown in Table 3).

### Soundness
3

### Presentation
3

### Contribution
3
