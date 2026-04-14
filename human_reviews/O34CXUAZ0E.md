# MARS: A Malignity-Aware Backdoor Defense in Federated Learning

- Decision: Reject
- Scores: 3, 6, 6, 5

## Abstract
Federated Learning (FL) is a distributed paradigm aimed at protecting participant data privacy by exchanging model parameters to achieve high-quality model training. However, this distributed nature also makes FL highly vulnerable to backdoor attacks. Notably, the recently proposed state-of-the-art (SOTA) attack, 3DFed (SP2023), uses an indicator mechanism to determine whether the backdoor models have been accepted by the defender and adaptively optimizes backdoor models, rendering existing defenses ineffective. In this paper, we first reveal that the failure of existing defenses lies in the employment of empirical statistical measures that are loosely coupled with backdoor attacks. Motivated by this, we propose a Malignity-Aware backdooR defenSe (MARS) that leverages backdoor energy (BE) to indicate the malicious extent of each neuron. To amplify malignity, we further extract the most prominent BE values from each model to form a concentrated backdoor energy (CBE). Finally, a novel Wasserstein distance-based clustering method is introduced to effectively identify backdoor models. Extensive experiments demonstrate that MARS can defend against SOTA backdoor attacks and significantly outperforms existing defenses.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper propose a novel backdoor defense mechanism in federated learning. This work first identifies deficiencies of existing defense methods, which are loosely coupled with backdoor attacks, resulting in low detection rate. This work proceeds to propose MARS, which first extract backdoor information by computing the backdoor energy (BE). Extracted BE values are then amplified by identifying the most prominent BE values. A Wasserstein distance-based clustering algorithm is finally applied to identify backdoor models. Authors further conduct experiments on MNIST, CIFAR10 and CIFAR100 to demonstrate the performance of the proposed method.

### Strengths
1. The paper is well-written with clear structure.
2. This work proposes a novel concept named backdoor energy. BE could represent the malignity of uploaded models.

### Weaknesses
1. The paper of BackdoorIndicator discover the strong connection between the learning rate adopted by adversaries and the detection ability of statistical backdoor defense methods. Authors should consider to specify the adversarial settings concerning different malicious learning rates and the number of training rounds.
2. The selection of counterpart poisoning algorithm is insufficient. Model replacement attack, which directly scales up poisoned updates, will obviously cause distinct difference between backdoor updates and benign ones. MRA could also be easily defended by clipping update norm to an previously agreed bound. 3DFed is designed specifically for breaking defense mechanisms with the component of OOD detection and consistency detection. However, MARS does not have either of the component. Thus, i would expect that the attack performance of 3DFed reduces to vanilla backdoor training algorithms. Authors should consider other backdoor training methods, like Neurotoxin [1] and Chameleon [2], which are proved to be more stealthy against various backdoor detection mechanism in the BackdoorIndicator paper.
3. Authors should also move the discussion part on the non-IID degree to the main paper, as it is a important aspect for advanced backdoor defense mechanisms.

[1]Zhang, Zhengming, et al. "Neurotoxin: Durable backdoors in federated learning." International Conference on Machine Learning. PMLR, 2022.

[2]Dai, Yanbo, et al. "Chameleon: Adapting to Peer Images for Planting Durable Backdoors in Federated Learning." International Conference on Machine Learning. PMLR, 2023.

### Questions
1. Could MARS maintain good performance under different adversarial settings (learning rates)?
2. Could MARS detect backdoor updates trained using more advanced algorithms (Neurotoxin and Chameleon)?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a malignity-aware defense that estimates the backdoor energy (BE) to quantify each neuron's maliciousness, and clustering algorithms to reject attackers at the server. In contrast to traditional defenses that rely on generic statistical measures, this method relaxes such assumptions and requires no access to clean datasets to estimate BE values. Extensive experiments validated the effectiveness of the proposed defense.

### Strengths
1. The paper proposes a defense method that does not rely on traditional statistical measures and relaxes the need for clean datasets to detect malicious users.

2. The logic of the paper is very clear and easy to follow.

3. The paper considers the scalability of the proposed work (e.g., ratio of attacks)

### Weaknesses
1. While Equations 2 and 3 are well-explained and motivated, It is unclear how to get the Lipschitz constants for different layers. 

2. It is still possible that all selected clients were malicious during the sampling process. In this case, the authors should provide more explanations of the assumption that “when the cluster is low, all local models are benign” (Page 7).

3. The proposed defense outperforms the other defenses over 3DFed, yet shares similar metrics as other defenses. Detailed explanations should have been given.

### Questions
Thank the authors for their work, please see my questions as follows.

1. Given the motivation and explanations of Eq 2 and 3, the Lipschitz constants across different layers are the key to quantifying the malicious clients. Could the authors provide specific details on how they compute or estimate these constants for different layer types (e.g., convolutional, fully-connected) in their experiments? This would enhance reproducibility and clarify a crucial technical aspect.

2. It is a norm that only a subset of clients will be selected for each FL round, in this case, when you have a high attacker ratio (as one of the key points mentioned in the paper), how to make sure the selected clients are not all malicious? If fail to guarantee, then does the clustering method still work? Please discuss potential limitations or failure modes if this assumption does not hold, or analyze the impact on the defense mechanism if this assumption is violated.

3. It is evident that MARS outperforms other defenses against attacks, especially the 3DFed attack. Yet, for the other two attacks, there are defenses that work similarly to MARS (metric-wise). Why is this the case? The authors should provide more insights.

Minor: There are some grammar issues such as “a L-layer…”, please proofread thoroughly.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper addresses the challenge of backdoor attacks in federated learning (FL) by introducing a novel defense mechanism called MARS (Malignity-Aware Backdoor Defense). Unlike existing defenses, MARS effectively identifies backdoored models by calculating backdoor energy (BE) to quantify each neuron’s level of malignancy and then applies Wasserstein distance-based clustering to isolate these models. Through extensive experiments on four datasets—MNIST, CIFAR-10, CIFAR-100, and ImageNet—the authors show that MARS outperforms current defenses in resisting state-of-the-art backdoor attacks.

### Strengths
- The use of five evaluation metrics provides a comprehensive assessment of the defense mechanisms, offering a detailed analysis of the proposed method.
- MARS identifies specific weaknesses in existing defenses, underscoring the limitations of current approaches and the need for more robust defense mechanisms.
- Toy examples effectively illustrate MARS’s capability to detect backdoored models, enhancing the clarity of the proposed defense approach.

### Weaknesses
- A comparison with optimized backdoor attacks, such as A3FL[1] and IBA[2], is missing, which would better demonstrate MARS’s resilience against sophisticated adversaries. How does MARS perform against these advanced attacks?
- There is limited discussion on the generalizability of MARS to other data types beyond image classification. How applicable might MARS be to FL settings involving text or tabular data?
- Computing BE for each neuron could be computationally expensive for larger models, raising concerns about scalability as model complexity increases.
- The experiments assume an attacker rate of 20%, a condition that might not reflect realistic scenarios. How does MARS perform with varying attacker ratios and data poisoning rates?

**References:**

[1]. Zhang, Hangfan, et al. "A3FL: Adversarially adaptive backdoor attacks to federated learning." Advances in Neural Information Processing Systems 36 (2024).  
[2]. Nguyen, Thuy Dung, et al. "IBA: Towards irreversible backdoor attacks in federated learning." Advances in Neural Information Processing Systems 36 (2024).

### Questions
- Does MARS require a high proportion of malicious clients to be effective, or can it detect backdoored models with a small number of attackers?
- What are the specific attack settings in the experiments? How many rounds do attackers participate in, and what is the data poisoning rate?
- Can FLIP [3] be used as a benchmark for comparison with MARS in terms of defense performance?
- In some cases, the backdoor energy of benign models may be higher than that of malicious models. How does MARS handle this scenario?
- Under what circumstances does Wasserstein distance-based clustering fail to detect backdoored models, and how might MARS be improved to address these cases?
- Which version of ImageNet is used in the paper? Is it Tiny ImageNet (200 classes) or the full ImageNet (1000 classes)?

**References:**

[3]. Zhang, Kaiyuan, et al. "Flip: A provable defense framework for backdoor mitigation in federated learning." International Conference on Learning Representations (2023).

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a new backdoor defense method called MARS for federated learning (FL). Traditional defenses rely on empirical statistical measures, which fail against advanced attacks like 3DFed due to their loose coupling with backdoor attacks. MARS overcomes this by introducing a concept called Backdoor Energy (BE) to assess neuron malignancy. The authors propose Concentrated Backdoor Energy (CBE) and a Wasserstein distance-based clustering approach to detect and filter backdoored models accurately. Experiments on multiple datasets (MNIST, CIFAR-10, CIFAR-100) show MARS's effectiveness, even under advanced adaptive attacks.

### Strengths
* I like the idea of BE and CBE because they offer a more direct method for evaluating neuron malignancy compared to empirical statistical measures.
* Extensive experiments demonstrate that MARS effectively counters SOTA backdoor attacks, maintaining high model accuracy and low attack success rates.
* MARS remains effective even when attackers adjust parameters to evade detection.

### Weaknesses
* My primary concern is the computational overhead incurred by the BE and CBE calculations, along with Wasserstein-based clustering. Also, the success of BE calculation relies on model parameters, which may vary across neural network architectures, affecting generalizability.
* In real-world scenarios, attackers may vary their strategies across rounds, models, and clients, making the BE metric less reliable for identifying such dynamic attacks.
* The paper assumes all clients use the same model architecture, allowing consistent BE and CBE calculations. However, in practical FL setups, clients may use slightly different architectures due to varying hardware capabilities and use cases,
* The performance of MARS depends on hyperparameters like the top percentage of BE values (κ) and the Wasserstein distance threshold (ϵ). The paper provides some sensitivity analysis, but more extensive exploration could strengthen the claims.
* While the paper demonstrates effectiveness on standard datasets, I expect to see evaluations on larger, more complex datasets like ImageNet to assess scalability.

### Questions
- How does MARS handle scalability in large federated networks (e.g., with thousands of clients)?
- What are the computational and communication overheads of deploying MARS in a real-world FL system?
- How sensitive is MARS to hyperparameter changes in clustering thresholds and the BE calculation process?
- Could the BE/CBE method be extended to detect other attack types, such as data poisoning or Byzantine attacks?

### Soundness
3

### Presentation
2

### Contribution
2
