# InfoScissors: Defense against Data Leakage in Collaborative Inference through the Lens of Mutual Information

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5, 5

## Abstract
Edge-cloud collaborative inference empowers resource-limited IoT devices to support deep learning applications without disclosing their raw data to the cloud server, thus protecting user's data. Nevertheless, prior research has shown that collaborative inference still results in the exposure of input and predictions from edge devices. To defend against such data leakage in collaborative inference, we introduce InfoScissors, a defense strategy designed to reduce the mutual information between a model's intermediate outcomes and the device's input and predictions. We evaluate our defense on several datasets in the context of diverse attacks and offer a theoretical robustness guarantee.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers a scenario in which an edge device, due to its resource constraints, hosts only small parts of an NN (e.g., the first layer and the head), whereas a cloud server hosts the rest. 

Namely,  in this scenario, the edge device sends to the cloud server not the raw inputs but their mid-representation and receives not the outputs of the model but the extracted features.

While this scenario provides some privacy, more is needed since carefully designed attacks (e.g., inversion) can reconstruct approximate inputs and labels, compromising clients' data privacy.

This paper proposes INFOSCISSORS, a defense strategy aiming to reduce the mutual information between a model's inputs and predictions to the intermediate outcomes (i.e., mid-representations and extracted features). 

INFOSCISSORS uses a carefully designed training objective to regularize the model that takes into account not only the model's ML performance (e.g., accuracy) but also the mutual information objectives.

The paper provides some theoretical and empirical results showing that INFOSCISSORS offers better accuracy/privacy tradeoffs than some previous defense mechanisms (e.g., DP, MID).

### Strengths
1. The considered problem of a hybrid deployments is very timely and important.

2. Minimizing mutual information between the input / output and intermediate outcomes is an interesting and intriguing concept.

3. There is theoretical derivation that support the proposes defense mechanism.

### Weaknesses
The paper, in its current form, has some significant gaps:

1. In a realistic scenario, the server may have access to lots of similar data (which is also growing over time) it can correlate with the inputs and outputs of the model. Thus, it is not a "single inference" problem. It is unclear how the amount of data available to the server over time affects the privacy guarantees. For example, in the evaluation, the sever is assumed to have 40 and 400 labeled samples to conduct KA and MC attacks for CIFAR10 and CIFAR100 respectively. What is the server has 50% of the data (e.g., 25,000) or 90%?

2. The threat model is not convincing. Why would a strictly untrusted server would strictly follow the collaborative inference protocol? 

3. The evaluation is not sufficient. Why a simple image classification problem over a network with 11M parameters would require a hybrid deployment? To make the case convincing a much more significant evaluation should be carried out with larger architectures and diverse tasks. Moreover, even for the CIFAR10 and CIFAR100 over ResNet18, the reported accuracies are significantly lower than an unsecure baseline model (approximately, -15 for CIFAR10 and -25 for CIFAR100). Why such a significant degradation would be acceptable? given that CNNs are fairly robust to noise and secondary objectives, what degradation one should expect in harder tasks (e.g., LLMs)?

### Questions
See weaknesses 1,2 and 3.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposed a unified approach to defend both label and feature leakage problems at inference time during collaborative learning scenarios. It addressed an important underlining problem which has promising applications in practice. By implementing mutual information regularization, the proposed work achieves superior defense results than other methods for label and feature attacks, individually.

### Strengths
1. The paper proposed a MI-based protection approach to defend both label and feature attacks, which are important problems in collaborative learning scenarios.

2. The experimental results indicate that the proposed method shows better utility-defense tradeoff.

3. Related works are sufficient.

### Weaknesses
Major weaknesses:

1. Limited Novelty: The proposed method is based on Mutual information (MI) regularization, which has been proposed in MID (Zou et al., 2023) . The major difference is that the paper used a different method proposed in (Cheng et al, 2020) for MI estimation, yet is also limited to inference attacks, not training attacks compared to MID. Therefore it appears that the novelty and contribution are limited. It is suggested that the paper highlights its contributions by explaining in-depth the differences between this work and other MI-based works such as MID and [1], and provide more insights on key factors to achieve superior defense results.

[1]  Tianhao Wang, Yuheng Zhang, and Ruoxi Jia. Improving robustness to model inversion attacks via mutual information regularization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 11666–11673, 2021.

2. Insufficient Experimental Evaluations: The paper evaluated only two feature reconstruction methods and one label leakage attacks (MC). It is also unclear how and why the experiments are performed separately for feature and label attacks when the model is trained end-to-end in Eq.10. For example, is label MI regularization turned off when feature attack is evaluated? It is unclear how the hyperparameters in Eq. 10 are chosen for each of the evaluations. It is suggested that the importance of each component in Eq.10 should be studied to understand their importance. It is also unclear exactly what hyperparameters are used to carry out experiments in Figure 7 and how their values impact the performance, which would provide valuable insights on how to reproduce and use the proposed method in practice. It is also suggested  to compare the results of integrated approach with other methods. 

3. Overclaim: The paper states that "this is the first paper to systematically defend against data
leakage in collaborative inference, encompassing both input leakage and prediction leakage." However, papers like MID (Zou et al. 2023) also defend both feature and label leakages, in both inference and training stage. 



Minor points:

1.  Fig.1 lacks proper explanation of notions, e.g. x, x', y, y' and \theta^h..., which makes it hard to understand.

2.  The approach proposed and evaluation is limited to collaborative inference only.

### Questions
What are the exact hyperparameters and settings for conducted label and feature attacks separately? How are the choice of these hyperparameter combinations impact the results? In other words, how would you suggest implementing both label and feature protections in practise?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces InfoScissors, a defense mechanism designed to protect both data and prediction privacy during collaborative inference. The approach aims to provide a good trade-off between privacy and utility by minimizing the mutual information between the output representation of the head model and the input data/label information.

### Strengths
1. The paper explores an important and timely topic of privacy in collaborative inference.
2. The proposed method demonstrates superior performance over existing baseline defenses, successfully achieving a better trade-off between utility and privacy concerns in both data and predictions.
3. The paper is well-organized and easy to follow.

### Weaknesses
1. The novelty of the proposed mutual information framework remains ambiguous. Many relevant works have utilized mutual information to balance privacy and utility. It is unclear what is the novel design in the proposed defense. How is the proposed mutual information framework distinguished from these works? How does it address the challenges in the existing works? For example, the paper claims that (Mireshghallah et al., 2020; Wang et al., 2021; Zou et al., 2023) “only achieve decent results when the head model on the edge device is deep.” How does InfoScissors address this limitation? In addition, the paper claims that “(Makhdoumi et al., 2014; Rassouli & Gündüz, 2019; Wu et al., 2022) apply mutual information on input space to protect input data, but their methods are only feasible with limited input dimension.” How does InfoScissors increase the input dimension?
2. The need for protecting data and prediction in edge computing needs further motivated. It would be great to discuss some potential edge applications that require such data protection.
3. The experiment is insufficient. First, five defense baselines are introduced in the evaluation, but the result of PPDL is not reported. Second, recent defenses, such as (Singh et al., 2021) and [1], are not compared in the paper. 
4. I assume that the highest accuracy shown in Figure 3 indicates the case where no defense is applied. (Correct me if I am wrong) Then this clean accuracy on the CIFAR10 and CIFAR100 datasets is pretty low (less than 80% and 50%). This indicates the models might not be well-trained.
[1] Yang, Mengda, Ziang Li, Juan Wang, Hongxin Hu, Ao Ren, Xiaoyang Xu, and Wenzhe Yi. "Measuring Data Reconstruction Defenses in Collaborative Inference Systems." Advances in Neural Information Processing Systems 35 (2022): 12855-12867.

### Questions
Please clarify the novelty of the proposed design.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes InfoScissors, a defending method against data leakage in collaborative inference. InfoSicssors aims to reduce the mutual information between edge and cloud. The authors discuss the robustness in theory and evaluate the defense results on several attack secnarios for CIFAR10 and CIFAR100 datasets.

### Strengths
InfoSisccor aims to protect user data in a hybrid computing environment by reducing mutual information, thereby enhancing user privacy, especially in edge-cloud IoT devices.

InfoSisccor offers a theoretical robustness guarantee for their defense strategy (sec 4.1) and proves that with such a strategy, the cloud server will not be able to extract original features and labels.

Furthermore, the authors empirically evaluate the performance on CIFAR10 and CIFAR100 datasets with different attack scenarios. The experimental results show that InfoScisccor can protect private user data while maintaining performance (0.5-0.1% drop).

### Weaknesses
The authors claim in their paper, "InfoScissor is the first to systematically defend...", but there have been several published papers that have already explored this area [1]. The authors neither cite their own work in the submission nor compare it in the experiments. Can you clarify the difference and original contribution of InfoScissor?

When hosting models in a cloud-edge hybrid format, it is important to know the distribution of computations within the model. For example, which layers are placed on the cloud and which are on the edge. It is known that the latter features are obstacles and hard to attack, but the authors do not clearly discuss their setting nor compare the defense effectiveness in different scenarios.

Using CIFAR10 and 100 as evaluation tasks alone may not provide a holistic view of the ability of InfoScissor. Including more evaluations such as Stanford Cars, PETs, and CUB-200 would make the experiments more solid.

[1] Measuring Data Reconstruction Defenses in Collaborative Inference Systems; NeurIPS'22

### Questions
See comments above

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper highlights the importance of edge-cloud collaborative inference in enabling IoT devices with limited resources to participate in deep learning applications while preserving user data privacy. It also acknowledges that existing research has identified a potential vulnerability in this approach, as it can still lead to the exposure of input and prediction information from edge devices. To address this, the authors introduce InfoScissors, a defense strategy aimed at minimizing the mutual information between the model's intermediate outcomes and the device's input and predictions. The effectiveness of InfoScissors is demonstrated through evaluations of various datasets and under different attack scenarios.

### Strengths
- The significance of this study lies in its relevance to the progress of deep learning and its application on real-world devices with limited computational capabilities. 
- The motivation for the problem and presentation is clear and easy to follow.

### Weaknesses
- The originality of this work is my primary concern. The proposed solution heavily relies on the mutual information estimator from (Cheng et al. 2020), and the idea of DL inference partition is not new.

- While the paper describes deploying the network's first and last few layers on the edge device, it must be clarified if the number of layers is fixed across devices. Further clarification on this aspect is needed.

- In DL networks, specific layers require a broader context of information from the entire input. Partitioning could disrupt this information flow, potentially leading to suboptimal performance. Knowing how the authors have addressed this challenge in their proposed solution would be beneficial.

- The paper only considers a scenario with one edge device and suggests that the proposed defense can be extended to multiple edge devices. However, it would be helpful to provide experimental results supporting this assertion to lend more credibility to this claim.

### Questions
- Given that the effectiveness of your defense relies on minimizing mutual information, have you explored or experimented with alternative information-theoretic measures to assess their suitability for achieving similar objectives?

- Can you discuss any potential limitations or trade-offs associated with applying InfoScissors, such as computational overhead, scalability, or compatibility with different DL models?

- In practical deployment scenarios, how feasible is the implementation of InfoScissors on edge devices with varying hardware capabilities? Are there specific requirements or constraints?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
