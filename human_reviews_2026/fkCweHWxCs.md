# Federated Learning with Unlabeled Clients: Personalization Can Happen in Low Dimensions

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Personalized federated learning has emerged as a popular approach to training on devices holding statistically heterogeneous data, known as clients. However, most existing approaches require a client to have labeled data for training or finetuning in order to obtain their own personalized model. In this paper we address this by proposing FLowDUP, a novel method that is able to generate a personalized model using only a forward pass with unlabeled data. The generated model parameters reside in a low-dimensional subspace, enabling efficient communication and computation. FLowDUP’s learning objective is theoretically motivated by our new transductive multi-task PAC-Bayesian generalization bound, that provides performance guarantees for unlabeled clients. The objective is structured in such a way that it allows both clients with labeled data and clients with only unlabeled data to contribute to the training process. To supplement our theoretical results we carry out a thorough experimental evaluation of FLowDUP, demonstrating strong empirical performance on a range of datasets with differing sorts of statistically heterogeneous clients. Through numerous ablation studies, we test the efficacy of the individual components of the method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes FLowDUP to train personalized models for clients participating in federated learning with and for even clients that have unlabeled data using only a forward pass. FLowDUP trains a hypernetwork that takes the unlabeled dataset and outputs parameters of a personalized predictive model that allows them to train in a subspace that is much smaller than the actual underlying personalzied model. They provide intuition behind their method by giving a generalization bound with multi-task PAC-Bayesian. The paper includes experimental evaluation of FLowDUP

### Strengths
- The paper investigates an important problem in FL where clients without labels can often exist and not be able to gain anything out of federated learning. 
- The paper provides theoretical insights into their proposed algorithm's loss function regarding using labeled clients to benefit the unlabeled client's personalized models. 
- The paper is clearly written.

### Weaknesses
- FLowDUP relies largely on the labeled clients' data, and due to this I have a couple of concerns. First, if the labeled clients data does not match well of the unlabeled clients' data, which is often the case in federated learning due to data heterogeneity and cannot really be controlled since they are clients' private data, the performance I believe will largely suffer. Moreover, it is rather unfair for the labeled clients to provide their training resource/data for the benefit of training personalized models for these unlabeled clients. In the perspective of incentives and fairness of clients the method also is not really convincing. 
- In the evaluation, the model used and the dataset used seems rather outdated, and easy ones and not sure how convincing the results actually are. Moreover, as mentioned above it seems like the data heterogeneity across clients will matter quite a lot here and the fact that this is not really explored in either in the theory or experiment section is concerning.

### Questions
Please address the weaknesses.

### Soundness
2

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
3

### Summary
This paper addresses a critical limitation of existing personalized federated learning (PFL) methods—their reliance on labeled client data to generate personalized models—by proposing FLowDUP (Federated Low Dimensional Unlabeled Personalization), a framework enabling personalized model generation for unlabeled clients via a single forward pass on unlabeled data. FLowDUP trains a hypernetwork h that takes unlabeled client data as input and outputs low-dimensional subspace parameters \(v_i\). Full model parameters \(\theta_i\) are derived via \(\theta_i = \theta_0 + Pv_i\) (where P is a fixed random expansion matrix and \(\theta_0\) is an initial parameter), avoiding high-dimensional parameter transmission. The objective combines a loss L (evaluated on labeled clients to measure model quality) and a learnable regularizer \(\Omega\) (computed on all clients, including unlabeled ones, to prevent overfitting).

### Strengths
1-FLowDUP directly addresses a well-identified pain point of PFL (unlabeled client incompatibility) with a non-trivial solution. Unlike prior hypernetwork-based methods (e.g., Amosy et al. 2024; Scott et al. 2024), it uses low-dimensional subspaces to enable on-device hypernetwork deployment and leverages unlabeled clients for training—two key innovations that distinguish it from related work.

2-It explicitly connects the regularizer \(\Omega\) and loss L to generalization performance, providing a mathematical justification for FLowDUP’s objective. This contrasts with many PFL works that lack theoretical guarantees for unlabeled clients.

3-Low-dimensional parameters reduce communication costs, the hypernetwork is lightweight for edge devices, and no client data is transmitted. This aligns with the core goals of federated learning (privacy, efficiency).

### Weaknesses
1-FLowDUP fails when conditional distributions \(D_{i|Y|X}\) (label given input) differ across clients but marginal distributions \(D_{i|X}\) (input alone) are insufficient to infer predictive models—e.g., subjective tasks like product recommendations or sentiment analysis. The authors acknowledge this but provide no mitigation strategies, restricting the method’s real-world applicability.

2-The default \(k=10^4\) is chosen empirically, with no theoretical analysis of how k relates to dataset heterogeneity, model architecture, or performance. While experiments show k correlates with accuracy, the lack of a principled way to select k (e.g., via the generalization bound) reduces FLowDUP’s usability.

3-All experiments use the \(pfl-research\) simulation library—there is no validation on actual edge devices (e.g., smartphones, IoT sensors). The authors claim low computational cost, but simulated efficiency may not translate to real hardware (e.g., memory constraints, latency).

### Questions
1-FLowDUP fails when conditional distributions \(D_{i|Y|X}\) (label given input) differ across clients but marginal distributions \(D_{i|X}\) (input alone) are insufficient to infer predictive models—e.g., subjective tasks like product recommendations or sentiment analysis. The authors acknowledge this but provide no mitigation strategies, restricting the method’s real-world applicability.

2-The default \(k=10^4\) is chosen empirically, with no theoretical analysis of how k relates to dataset heterogeneity, model architecture, or performance. While experiments show k correlates with accuracy, the lack of a principled way to select k (e.g., via the generalization bound) reduces FLowDUP’s usability.

3-All experiments use the \(pfl-research\) simulation library—there is no validation on actual edge devices (e.g., smartphones, IoT sensors). The authors claim low computational cost, but simulated efficiency may not translate to real hardware (e.g., memory constraints, latency).

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
4

### Summary
The paper proposes a PFL method called FLowDUP that generates a client-specific model using only unlabeled data. FLowDUP employs a hypernetwork to produce low-dimensional subspace parameters, which are expanded via a fixed random matrix to yield the full model. Its training objective combines a labeled-client loss and an unlabeled-client regularizer, theoretically supported by a new transductive multi-task PAC-Bayesian generalization bound. Experiments across serveral widely-used datasets show improved performance compared with baselines such as FedAvg, FedProx, and ATP.

### Strengths
1. Addressing personalization for unlabeled clients is a relatively unexplored area in PFL.
2. The paper introduces a new transductive multi-task PAC-Bayesian bound that motivates the objective and gives its proof.
3. The low-dimensional subspace formulation has the potential to reduce communication and computation costs in federated settings.

### Weaknesses
1. The proposed method is built upon the foundations of regulation-based and hypernetwork-based personalized federated learning (PFL). However, the related work section does not provide an adequate review or a clear comparison between prior methods and the proposed approach.
2. The paper uses the hypernet for personalization which is not a new idea. Besides, the authors emphasize that the proposed method can handle federated learning with labeled data, yet there is no direct comparison with existing hypernetwork-based PFL approaches, such as [1] and [2].
3. If FedAvg cannot be trained without labels, how was it used for comparison? If FedAvg was trained with labeled data but the proposed method used unlabeled data, how is this comparison fair and meaningful?
4. The paper claims that the proposed approach is computationally and communication efficient, yet no experiments or quantitative evidence are presented to substantiate this claim.
5. The writing and structural focus of the paper could be improved. While substantial space (including Appendix A) is devoted to PAC-Bayes theory, its direct connection to the proposed method is not clearly articulated and should be elaborated. Conversely, the discussion of basic federated learning concepts is overly detailed and could be condensed.

[1] Shamsian, A. et al. “Personalized Federated Learning Using Hypernetworks.” ICML, 2021.

[2] Yang, Z. et al. “Hypernetwork-Based Physics-Driven Personalized Federated Learning for CT Imaging.” IEEE Transactions on Neural Networks and Learning Systems, 2023.

### Questions
1. What is the detailed form of the regularization term $\psi_r$ in implementation, will the authors make their code public?
2. The paper claims that FLowDUP supports training with unlabeled data, but all experimental datasets are public labeled datasets. How were the results in Tables 1 and 2 obtained, were the training data labeled, unlabeled, or partially labeled?
3. What was the data splitting scheme between labeled and unlabeled clients?

### Soundness
3

### Presentation
2

### Contribution
3
