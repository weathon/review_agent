# FedPAC: Consistent Representation Learning for Federated Unsupervised Learning under Data Heterogeneity

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Federated unsupervised learning enables collaborative model training on decentralized unlabeled data but faces critical challenges under data heterogeneity, which often leads to representation collapse from weak supervisory signals and semantic misalignment across clients. Without a consistent semantic structure constraint, local models learn disparate feature spaces, and conventional parameter averaging fails to produce a coherent global model. To address these issues, we propose Federated unsupervised learning with Prototype Anchored Consensus (FedPAC), a novel framework that establishes a consistent representation space via a set of learnable prototypes. FedPAC introduces a dual-alignment objective during local training: a semantic alignment loss that steers local models towards a prototype-anchored consensus to ensure cross-client semantic consistency, coupled with a representation alignment loss that promotes the learning of discriminative and stable features. On the server, prototypes are aggregated by an optimization-based strategy that preserves semantic knowledge and ensures the prototypes remain representative. We provide a rigorous convergence analysis for our method, formally proving its convergence under mild assumptions. Extensive experiments on benchmarks including CIFAR-10 and CIFAR-100 demonstrate that FedPAC significantly outperforms state-of-the-art methods across a wide range of heterogeneous settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes FedPAC to address the critical challenges of representation collapse and semantic misalignment in federated unsupervised learning. FedPAC introduces a dual-alignment objective during local training: a semantic alignment loss that steers local models towards a prototype-anchored consensus to ensure cross-client semantic consistency, coupled with a representation alignment loss that promotes the learning of discriminative and stable features. On the server, prototypes are aggregated by an optimization-based strategy that preserves semantic knowledge and ensure the prototypes remain representative. Extensive experiments CIFAR-10 and CIFAR-100 demonstrate the effectiveneses of proposed method.

### Strengths
1. This work proposes a synergistic architecture that combines a dual-alignment learning objective for clients’ local unsupervised learning and a prototype aggregation strategy refining global prototypes on the server.
2. This work provides a convergence analysis of proposed method.
3. Extensive experiments on CIFAR-10 and CIFAR-100 show the effectiveness of proposed method.

### Weaknesses
1. Many existing works explore the challenges the authors claim to have solved. What are the main distinctions between the proposed method and existing federated unsupervised learning methods? It is suggested that the authors add a comparative table to directly illustrate these distinctions. And it is recommended that the authors incorporate more recent and state-of-the-art works into Section 2.2.
2. In the convergence analysis, the definition of the prototype parameters $\rho$ conflicts with the strength/regularization parameter $\rho$ used in line 243. Furthermore, the prototype parameters appear to be updated using an SGD-like approach in every communication round. This requires a more detailed explanation of the update mechanism.
3. Is Assumption 6 commonly used in related literature? What is the practical meaning and implication of this assumption?
4. In Section 4.2.1, line 234, the authors claim that Unbalanced Optimal Transport (UOT) can accommodate scenarios where clients lack certain classes. However, there is no experimental evidence to support this claim. It is suggested that the authors add an experiment demonstrating performance under a pathological non-IID distribution.
5. In line 305, why is the number of local client prototypes (I) typically larger than the number of global prototypes (K), i.e., $I\geq K$?
6. The rigor of the presentation can be improved. Line 239 states that Equation (2) can be solved using Sinkhorn iterations, while line 247 refers to using a Sinkhorn-like iteration. The authors must use consistent and precise terminology to describe the solution method.

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FedPAC, a novel federated unsupervised learning framework designed to address two critical challenges under data heterogeneity: representation collapse and semantic misalignment. FedPAC introduces a dual-alignment objective on the client side, including a semantic alignment loss (enforcing consistency with global prototypes) and a representation alignment loss (promoting discriminative and stable features). On the server side, a prototype optimization strategy refines global prototypes to preserve semantic diversity and maintain cross-client coherence. The paper provides theoretical convergence guarantees and extensive experiments on CIFAR-10 and CIFAR-100, showing clear improvements over prior federated unsupervised learning methods.

### Strengths
1. The paper effectively integrates semantic alignment (with global prototypes) and representation alignment (across augmented views), encouraging clients to learn representations that are both locally discriminative and globally consistent. This dual-loss design directly addresses data heterogeneity in a principled manner.

2. By optimizing global prototypes on the server, FedPAC maintains prototype diversity and captures the distribution of client representations more effectively. This prototype-centered aggregation approach is rarely explored in prior federated unsupervised learning literature and contributes a fresh perspective on maintaining global semantic consistency.

3. The paper provides a detailed convergence analysis proving sufficient decrease and global convergence bounds under mild assumptions.

### Weaknesses
1. Experiments are conducted only on CIFAR-10 and CIFAR-100 using Dirichlet sampling to simulate data heterogeneity. The evaluation could be strengthened by including additional datasets (e.g., MNIST, Tiny-ImageNet, or DomainNet) and broader forms of heterogeneity (e.g., domain shift, quantity-based label imbalance). This would help assess the generalization of FedPAC in more realistic federated scenarios.

2. The paper does not examine how FedPAC performs under varying numbers of clients, client participation ratios, or communication budgets. These factors are crucial in practical federated environments and could significantly impact convergence and accuracy.

3. There are small typographical errors and redundancies (e.g., repeated “sample sample” at line 225). A careful proofreading pass would improve the overall presentation quality.

### Questions
1. Can the proposed method maintain its performance advantage under other types of data heterogeneity, such as domain shift scenarios where the data distribution varies in style or modality across clients, rather than only under label distribution heterogeneity (i.e., Dirichlet-based label shift)?

2. The experiments only use ResNet-18 as the encoder. Would the method still hold its superiority when using different backbones such as ResNet-50, ViT, or MobileNet, particularly given their different representational capacities?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a federated unsupervised learning framework named FedPAC, designed to address representation learning challenges under non-IID data distributions. FedPAC introduces a prototype-anchored consensus mechanism, incorporating dual alignment objectives during client-side local training. On one hand, the representation alignment loss ensures invariance across augmented views; on the other hand, the semantic alignment loss aligns local representations with globally shared prototypes, thereby achieving global semantic consistency. The server aggregates local prototypes via an optimization strategy to update the global prototypes, enhancing cross-client semantic consensus. Experimental results demonstrate that FedPAC outperforms existing methods on the CIFAR-10 and CIFAR-100 benchmark datasets, achieving superior performance.

### Strengths
(1) The FedPAC method integrates dual alignment objectives in client-side local unsupervised learning with a prototype aggregation strategy for server-side global prototype optimization, achieving a balance between local representations and global consistency.

(2) This paper conducts a mathematical analysis of FedPAC's convergence, theoretically demonstrating that the method achieves stable convergence within a finite number of communication rounds.

### Weaknesses
(1) The related work section cites studies that are not sufficiently recent. Additionally, the paper includes only 22 references, with just 3 published in the last three years (2023 and beyond), which does not provide readers with a comprehensive overview of the broader topic.

(2) The FedPAC employs the Unbalanced Optimal Transport (UOT) method to relax the prototype marginals, thereby preventing spurious assignments. In Section 3, FedPAC assumes that all clients utilize the same augmentation function $\mathcal{T}$, without accounting for the potential impacts of feature shift or noise on model performance.

(3) After completing local training on the clients, the model parameters and local prototypes are uploaded to the server, which may introduce privacy leakage risks. However, the paper does not provide sufficient discussion on this issue.

(4) The FedPAC introduces multiple hyperparameters (such as the entropy regularization parameter $\epsilon$, KL penalty strength $\rho$, loss balancing coefficient $\lambda$, uniformity trade-off coefficient $\beta$, and repulsion sharpness $\gamma$, among others). The paper merely describes their roles without providing specific values in the experimental setup.

(5) The paper lacks an analysis of the time complexity and space complexity of FedPAC, and does not provide a comparison of computational efficiency with baseline methods.

(6) The experimental section relies exclusively on CIFAR-10 and CIFAR-100 as benchmark datasets. Although these two datasets differ in class granularity, they are fundamentally similar in nature. It is advisable to use a broader set of datasets to enhance the persuasiveness of the experimental results.

(7) The baseline methods cited in the paper are somewhat outdated, with the most recent being from 2024 (FedU2), while the majority stem from 2022 or earlier. It is recommended to incorporate additional recent comparative methods for a more comprehensive evaluation.

### Questions
(1) The FedPAC method employs UOT to relax prototype marginals in order to prevent spurious assignments, yet it assumes that all clients use the same augmentation function $\mathcal{T}$. How does it address the potential impacts of feature shift or noise? What is the robustness of this mechanism under extreme non-IID scenarios?

(2) The process of clients uploading model parameters and local prototypes to the server may introduce privacy leakage risks. Have the authors considered incorporating protective measures?

(3) The paper lacks an analysis of FedPAC's time and space complexities, as well as a comparison of computational efficiency with baseline methods. How do the authors evaluate the method's practical deployability, particularly in resource-constrained federated environments?

(4) The majority of the baseline methods compared are from 2022 or earlier, with the most recent being FedU2 from 2024. Could the authors extend the comparisons to include more recent methods to enhance the comprehensiveness of the evaluation?

### Soundness
3

### Presentation
3

### Contribution
3
