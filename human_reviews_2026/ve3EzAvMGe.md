# DeepAFL: Deep Analytic Federated Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Federated Learning (FL) is a popular distributed learning paradigm to break down data silo. Traditional FL approaches largely rely on gradient-based updates, facing significant issues about heterogeneity, scalability, convergence, and overhead, etc. Recently, some analytic-learning-based work has attempted to handle these issues by eliminating gradient-based updates via analytical (i.e., closed-form) solutions. Despite achieving superior invariance to data heterogeneity, these approaches are fundamentally limited by their single-layer linear model with a frozen pre-trained backbone. As a result, they can only achieve suboptimal performance due to their lack of representation learning capabilities. In this paper, to enable representable analytic models while preserving the ideal invariance to data heterogeneity for FL, we propose our Deep Analytic Federated Learning approach, named DeepAFL. Drawing inspiration from the great success of ResNet in gradient-based learning, we design gradient-free residual blocks in our DeepAFL with analytical solutions. We introduce an efficient layer-wise protocol for training our deep analytic models layer by layer in FL through least squares. Both theoretical analyses and empirical evaluations validate our DeepAFL's superior performance with its dual advantages in heterogeneity invariance and representation learning, outperforming state-of-the-art baselines by up to 5.68%-8.42% across three benchmark datasets. Related code is available at https://github.com/tangent-heng/DeepAFL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents DeepAFL, a framework designed to achieve invariance to data heterogeneity in federated learning (FL) while reducing computational cost. Unlike existing analytical federated learning (AFL) methods that are limited to a single layer, DeepAFL extends the idea to multiple nonlinear layers, thereby enabling richer representation learning. Specifically, building upon pre-trained features, DeepAFL updates the model in a layer-wise manner, where it learns an analytical classifier for each layer and subsequently learns a transformation to adapt the layer’s features for improved alignment with the class labels. In the FL setting, local clients transmit information matrices only once to the server, which then aggregates the final model. Experimental results show that DeepAFL outperforms both gradient-based and analytical FL approaches, demonstrating strong invariance to data heterogeneity and improved computational efficiency.

### Strengths
- The paper tackles a well-motivated problem that constitutes a significant and timely advancement for analytical federated learning. 

- The proposed DeepAFL demonstrates promising performance compared to existing gradient-based and analytical FL methods, while maintaining strong computational efficiency and invariance to data heterogeneity. 

- The experiments are comprehensive and examine multiple aspects, including hyperparameter settings, activation functions, and ablation analyses.

### Weaknesses
- To obtain an analytical solution, DeepAFL relies on the mean squared error (MSE) loss; however, MSE can be sensitive to noise in the data when applied to classification tasks. In realistic federated learning (FL) scenarios, such as hospital applications, clients may contain data annotated by different labelers, leading to slight inconsistencies due to varying labeling standards. The experiments in the paper primarily use clean and well-organized datasets, such as CIFAR and TinyImageNet. Therefore, it remains unclear whether DeepAFL can achieve competitive performance under noisier, real-world conditions compared to gradient-based FL approaches that employ cross-entropy loss.

- Although the paper highlights the importance of representation learning for analytical FL, it lacks visualizations or analyses to demonstrate whether the learned features are semantically meaningful. Moreover, the generalization ability of these features remains unexplored, for example, their cross-domain generalization capability.


- The gradient-based FL baselines used for comparison are mostly from studies prior to 2023, lacking evaluation against more recent and advanced approaches such as FedAWA [1] and one-shot FL methods like FedLPA [2].


[1] 2025 CVPR FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors


[2] 2024 NeurIPS FedLPA: One-shot Federated Learning with Layer-Wise Posterior Aggregation

### Questions
Besides the weakness shown in the above section, please also see the following questions: 

Q1: DeepAFL uses ResNet-18 as a pre-trained backbone and trains $T$ additional layers on top of it. Do the gradient-based approaches also follow a similar setup, or do they fine-tune the ResNet-18 backbone? If not, would gradient-based methods potentially perform better, or be more stable, if they likewise kept the pre-trained backbone fixed and only trained additional layers? 

Q2: Since each intermediate layer in DeepAFL is trained with an auxiliary classifier that shares the same objective and class labels, all layers are encouraged to align closely with the same targets without awareness of how subsequent layers will utilize their features. Could this design lead to redundant representations across layers? For instance, if a given layer already achieves linear separability, the following layer may have little room to learn additional useful information. Would it be possible to incorporate a reconstruction-based objective, such as an autoencoder with MSE, to encourage more diverse and informative feature learning?

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
This paper proposes DeepAFL, an extension of Analytic Federated Learning (AFL) that introduces analytic residual blocks and deeper analytic layers to enable federated representation learning without gradient calculation and communication. The method alternates between solving for the classifier and transformation matrices​ in closed form, using aggregated feature–label correlation matrices across clients. This paper provides theoretical proofs for heterogeneity invariance and monotonic empirical risk reduction, and the analysing of privacy and efficiency. Experiments evaluate DeepAFL on CIFAR-10, CIFAR-100, and Tiny-ImageNet under Non-IID settings, showing 5–8% accuracy gains over gradient-based FL baselines (FedAvg, FedGen, FedDisco, etc) and the original AFL.

### Strengths
1. The paper is well-written and clearly structured, with equations and derivations easy to follow. The figures and notations are well explained.

2. Extending analytic learning to a deeper residual form and formulating closed-form residual updates is technically neat and easily understood.

3. This work provides sound theoretical analyses to support the heterogeneous invariance and representation learning capabilities.

4. Due to no backpropagation, this work achieves quite optimization cost reduction, which is demonstrated by experiments.

### Weaknesses
### **1. The experimental setup is methodologically unfair.**

DeepAFL aggregates global feature–label statistics and computes a closed-form model that the authors explicitly state is identical to centralized analytic learning, hence it is inherently immune to data heterogeneity. In contrast, gradient-based baselines (FedAvg, FedProx, MOON, etc.) naturally degrade under Non-IID settings. Evaluating all methods on strongly heterogeneous partitions, therefore, gives DeepAFL an artificial advantage: the method bypasses heterogeneity by design while the baselines must deal with it. 

A fair comparison should instead be conducted under IID conditions or against a centralized analytic baseline, so that any remaining advantage would reflect DeepAFL’s genuine strengths in efficiency or analytic formulation rather than the absence of gradient-related heterogeneity effects.

### **2. The representation learning ability is not convincing.**

While the paper repeatedly emphasizes that *DeepAFL achieves deep analytic representation learning*, the method it introduces does not actually justify that claim. According to Section 3.1, each residual analytic block is defined as  
$g_{t+1}(\Phi_t) = \sigma(\Phi_t B_t)\Omega_{t+1},$  
where $B_t$ is a random projection matrix, $\sigma(\cdot)$ is a nonlinear activation, and $\Omega_{t+1}$ is computed analytically via least squares. This design simply reprojects and reweights the *fixed backbone features* rather than creating new or more abstract representations. All transformations occur within the same frozen feature space extracted by the pretrained backbone, without modifying or adapting the backbone itself. Consequently, DeepAFL indeed enhances the expressiveness of the analytic mapping in this fixed space, but it does not improve the representational capacity or transferability of the underlying model. In other words, the pretrained backbone remains the real bottleneck of representation learning, and DeepAFL does not attempt to address it.  

Moreover, the theoretical and empirical evidence provided does not demonstrate genuine feature learning. Theorem 2 merely proves that the empirical risk decreases monotonically with layer depth, a sign of better fitting, not of learned representations. Similarly, the “representation analyses” in Section 4.2 only report accuracy gains with increasing layers, which reflect improved regression capability but not new feature semantics or cross-domain adaptability.  

### **3.The overall problem setting is conceptually misguided.**
DeepAFL assumes that all clients share a frozen, pre-trained backbone such as an ImageNet-trained ResNet-18.
However, if this backbone has never been trained on data related to the clients’ tasks, its extracted features may already be poorly aligned with the actual data distributions.
This feature-domain mismatch is exactly the challenge that federated learning is meant to address—adapting a shared model to diverse client data without violating privacy.
The authors themselves note that “a domain shift between the backbone’s pre-training data and the FL training data can further impact AFL’s performance”, yet the proposed method and theory ignore this issue.
All experiments assume that the backbone provides perfectly relevant and transferable features, effectively sidestepping the main difficulty that would justify using FL.
To be meaningful, DeepAFL should be analyzed and validated in settings where the backbone features are imperfect or mismatched with client data, since that is where its claimed advantages would actually matter.

### Questions
1. Could the authors clarify the real motivation and use case of DeepAFL? If all clients already share a strong frozen backbone, what problem is this method actually addressing? Would it make more sense to frame DeepAFL as a collaborative fine-tuning approach, where clients adapt a shared public backbone to private data with different feature distributions?

2. The current experiments assume that the backbone provides perfect and well-aligned features. Could the authors evaluate DeepAFL under domain-gap conditions, where the backbone has never seen the clients’ data or task, to test whether the proposed analytic layers can compensate for feature misalignment?

3. Please include results under IID partitions or with a centralized analytic baseline to verify whether DeepAFL’s improvements come from genuine analytic efficiency rather than its immunity to Non-IID effects.

4. What exactly does “representation learning” mean in this work? If the backbone is frozen and the residual blocks only reweight existing features, are new representations actually being learned? Some analysis of intermediate features, such as layer-wise visualization, separability metrics, or similarity, would help substantiate this claim.

5. Theoretical analyses currently assume ideal, fixed backbone features. Do the invariance and convergence results still hold when the features are imperfect, noisy, or mismatched with the clients’ data? If not, discussing or extending the theory to such realistic cases would greatly improve the paper’s relevance.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents DeepAFL, a deep analytic federated learning framework that extends the original Analytic Federated Learning (AFL) approach from a single linear layer to multiple residual analytic layers. Each layer is trained via a closed-form “sandwiched” least-squares solution, completely avoiding gradient-based optimization. The authors provide theoretical proofs showing (1) invariance to data heterogeneity when random projections are shared and (2) a monotonic reduction of empirical risk as the network depth increases. Experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate consistent accuracy improvements and notable reductions in computational and communication costs.

### Strengths
1. The paper introduces the first federated learning framework that achieves deep representation learning without gradients while maintaining closed-form analytical updates.
2. The idea of stacking analytic layers in a residual manner is a meaningful and elegant extension of AFL.
3. Reported efficiency gains-around 99% reduction in computation and 50% reduction in communication compared to gradient-based methods-are impressive.

### Weaknesses
1. While theoretical invariance is well-supported, the practical scalability to very large models such as ViTs or LLMs has not been evaluated.
2. The related work section omits recent multimodal and representation-based FL methods like FedRep [1] and FedU² [2].
3. No wall-clock runtime, GPU-hour, or FLOP comparisons are provided to substantiate the efficiency claims.

### Questions
1. Clarify theoretical assumptions, including shared random seeds, the impact of regularization, and the influence of finite precision.
2. Provide quantitative computational profiling (runtime, FLOPs, GPU-hours) to verify the claimed efficiency improvements.
3. Include comparisons with other analytic or gradient-free deep learning methods beyond AFL.
4. Add a clear visual diagram explaining the “sandwiched” least-squares process for improved readability.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper aims to address the shortcoming of analytic federated learning: lack of representation learning capability. The idea is to incorporate skip connection into the multi-layer version of the analytic federated learning. The initial feature is extracted from a pre-trained backbone, and then randomly projected and activated to get the zero-th layer feature. The feature is refined at the next layer by adding a residual block consisting of a nonlinear feature transformation and a learnable weight. The nonlinear feature transformation is the same as those for the initial feature, i.e., random projection for stochasticity and activation for nonlinearity. Given the feature at each layer, both the classifier and weight have a closed-form solution to the regularized least-squares. This way, the classifier keeps improving and is shown to converge eventually along the chain of layers. This idea is then adapted to the federated learning setting with only auto-correlation and cross-correlation matrices of relevant client data uploaded to the server, which ensures the invariance of the global model to data heterogeneity as analytic federated learning does. It is mentioned that it is hard to infer client information from the communicated data and the data encryption technique can be integrated to enhance privacy. Extensive experiments are conducted to demonstrate the superiority of the proposed algorithm in terms of both accuracy and efficiency across various settings.

### Strengths
1. The idea of adding skip connection to the multi-layer version of analytic federated learning is great.
2. The design of the residual block is nice.
3. The framework addresses well the representation learning issue of the analytic federated learning while keeping all advantages like closed-form solutions and invariance to data heterogeneity.
4. The algorithm is guaranteed to converge
5. Extensive experiments are provided to demonstrate the potential of the algorithm.

### Weaknesses
The privacy argument seems weak. Once data encryption techniques are introduced, the computational cost will be increasing as well.

Minor: 
In Line 373-374, "and AFL (He et al., 2025). Furthermore, we include the analytic learning-based method AFL (He et al., 2025) as a baseline to further highlight our advantages", where "AFL (He et al., 2025)" is repeated.

### Questions
1. what if only partial participation of clients?

### Soundness
3

### Presentation
3

### Contribution
3
