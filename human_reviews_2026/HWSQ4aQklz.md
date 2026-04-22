# Demystifying Latent Forgetting in Federated Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Federated Learning (FL) enables collaborative model training across decentralized, isolated clients in a privacy-preserving manner, but at the cost of limited control over the data and the training procedure. One of the key challenges in FL is the spatial data heterogeneity, which is due to the stratified nature of the underlying data distributions between clients. In addition, FL systems also undergo periods of time in which certain features disappear from the training data pool, resulting in the less studied but critical problem of temporal-spatial data heterogeneity. Such non-uniformity in training data across time introduces a new feature-level latent forgetting that is fundamentally different from the well-studied task-level catastrophic forgetting in continual learning. 
This latent forgetting, if not detected and mitigated timely, can result in poor model performance, especially for certain learning features.
The privacy requirements and temporal-spatial data heterogeneity of FL make the detection and mitigation of latent forgetting challenging.
In this paper, we analyze latent forgetting and propose FedMemo, a privacy-preserving FL framework to control its impact. FedMemo  employs an automated detection mechanism to detect latent forgetting in real time with preserved privacy. FedMemo further introduces a proxy-based 2-step aggregation approach to mitigate the impact of latent forgetting. We evaluate FedMemo in a diverse set of vision and language classification tasks in various FL settings, and show that it outperforms state-of-the-art methods by up to $20.06\%$

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to solve the spatial data heterogeneity in federated learning (FL), a feature-level latent forgetting.This paper proposes FedMemo, a privacy-preserving framework that employs an automated detection mechanism to detect latent forgetting in real-time. Extensive experiments show the benefits of FedMemo in language and vision tasks.

### Strengths
1. This work targets a less studied problem, temporal-spatial data heterogeneity, which is a critical problem in real-world applications.

2. While using the proxy data, the proposed method FedMemo ensures the privacy-preserving feature of FL, providing a secure way to use proxy data.This method also incurs minimal additional communication costs compared with previous approachs.

3. The experiments cover various tasks including vision and language tasks. The improved performance on these tasks show the effectiveness of the proposed method.

### Weaknesses
1. This work prooposes two methods, including client-side and server-side FedMemo. They have different benefits in critical learning period and near convergence, posting the problem when to use the client-side or server-side method. In practice, it is hard to identify which period the learning process is in.

2. The quality of proxy data has critical impact on the final model performance of FL. This implies the difficulty of training a GAN model in this setup. In practice, the proxy data is hard to obtain or has low quality or different distributions.

3. The experiments compare the method with three methods: SCAFFOLD and FedProx are not specially designed for the spatial heterogeneity problem, and MFCL is designed for vision tasks. The lack of baselines weakens the benefits of the proposed method.

### Questions
1. Appendix E provides the GAN architecture for the vision task, how about the language task?

2. If there are some proxy data, can we directly use other approachs without training GAN to improve the model performance? I believe these should be strong baselines for this work.

3. Using DP can enhance the privacy protection of FedMemo for sure, but could authors provide an analysis on the privacy leakage of FedMemo without DP?

### Soundness
2

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
This paper proposes FedMemo, a privacy-preserving federated learning framework designed to detect and mitigate feature-level latent forgetting caused by temporal–spatial data heterogeneity. The method introduces an automated forgetting detection mechanism based on global weight updates and a proxy-based two-step aggregation strategy to alleviate performance degradation from missing features. This work introduces a new concept of “latent forgetting,” but there is still room for improvement in the aspects of method design, experimental design, and overall writing quality.

### Strengths
1. The paper presents a new perspective by introducing latent forgetting caused by feature unavailability, which differs from catastrophic forgetting in continual learning.
2. It provides a certain level of theoretical analysis to explain the phenomenon of latent forgetting.
3. The overall writing structure is clear and well organized, but some parts of the content are not clearly described.

### Weaknesses
1. The authors describe two types of latent forgetting, but the method section does not clearly explain how the proposed approach effectively addresses both. Evidence based solely on accuracy results in the experiments is insufficient to support this claim.
2. In FedMemo, latent forgetting is detected through Weight Update Variance. However, relying only on Weight Update Variance is inadequate, since its decrease can result from various factors such as convergence, and does not necessarily indicate latent forgetting.
3. Figure 1 illustrating FedMemo is not sufficiently clear, particularly regarding how the generative model operates differently on the server and client sides.
4. The proposed “2-step aggregation” appears to be a simple proxy-based fine-tuning procedure and lacks substantial innovation.
5. The paper conflates feature unavailability with sample unavailability. Specifically, most experiments simulate “feature unavailability” by removing certain classes, which in fact represents class or sample missingness, not true feature unavailability as defined.
6. The notation system is overly complex and unclear. For example, while $P_i^{t}(x, y)$ is compared with $P_i^{ref}(x, y)$, the paper never defines how unavailable features are identified from actual data.
7. The rationale behind the selection of specific classes and clients for simulating “temporal feature unavailability” in the experiments is not provided.
8. The paper lacks ablation studies to verify the independent contribution of the proposed “2-step aggregation” compared with single-step aggregation.
9. The analysis of privacy and communication overhead is insufficient. Although the paper claims “minimal overhead,” it provides no quantitative comparison of communication volume, training time, or FLOPs.
10. While the authors discuss potential privacy-preserving mechanisms, no experiments are conducted to validate their effectiveness.

### Questions
1. How much of a decrease in Weight Update Variance triggers latent forgetting detection?
2. In the paper, $\cos(\theta) > 0$ is merely a sign judgment rather than a practical threshold. How is it selected during training? Moreover, how is this angle relationship measured or used to decide whether to continue using proxy samples?
3. Why are the latest FCL methods not included for comparison? MFCL is from 2023, yet the paper still labels it as “SOTA.”
4. In the sentence “the task is the same, the temporally or permanently unavailable features during training is usually unexpected and uncontrollable,” how should the “unavailable features” be specifically interpreted? 

I look forward to the authors' rebuttal to address some of my remaining concerns.

### Soundness
2

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
4

### Summary
This paper introduces FedMemo, a novel framework to address the problem of "latent forgetting" in Federated Learning (FL). The authors identify and formalize a critical challenge in FL: temporal-spatial data heterogeneity, where features can become temporarily or permanently unavailable across clients and training rounds, leading to a feature-level "latent forgetting". FedMemo proposes a two-pronged approach: 1) a privacy-preserving detection mechanism for latent forgetting based on weight update variance, and 2) a two-step aggregation method that leverages synthetic proxy data (generated either on the server or clients by GAN) to mitigate the detected forgetting. The method is evaluated on both vision (CIFAR-10, SVHN) and language (GLUE) tasks, demonstrating significant performance improvements over strong baselines, including state-of-the-art methods adapted from Federated Class-Incremental Learning (FCL).

### Strengths
1. Clear motivation and presentation. The studied problem "temporal-spatial data heterogeneity" is challenging (but not novel) in the domain of federated continual learning. 
2. Theoretical grounding: The paper provides a theoretical analysis (Propositions 3.1 and 3.2) that formally connects feature unavailability to performance degradation (latent forgetting) and explains how the proposed 2-step aggregation can mitigate it. This strengthens the methodological foundation. 
3. Experimental results validate the better performance of the proposed approach in tackling the latent forgetting caused by temporal-spatial data heterogeneity, compared with several baselines and a state-of-the-art method.

### Weaknesses
1. Novelty is insignificant. 
While the dynamic detection of latent forgetting in FL runtime is novel, the main idea for tackling the detected forgetting is replaying, which has been well-studied in the domain of (federated) continual learning.
2. The usage of the proposed FedMemo is not clear. 
The paper proposes server-side and client-side FedMemo, but does not provide instructions on how to choose from these two variants. The necessity of client-side variant is not clear. This variant cannot ensure a superior performance in all FCL scenarios, compared with than the server-side variant. Additionally, it incurs extra computation burden on the clients.
3. Clarity of the 2-Step aggregation process needs further improvement.
While the high-level idea is clear, the exact sequence and timing of the two steps could be described more precisely in the main text. For instance, it is not clear how the server generates w_p^{t+1} in Equation (1) when the client-side mode is triggered. A more detailed algorithmic pseudo-code in the main paper would greatly improve clarity.
4. Experiment results are not illustrative. 
(1) The effectiveness of client-side FedMemo is validated on only vision datasets. The authors do not provide any explanation on the missing results of client-side FedMemo on NLP tasks. 
(2) The results supporting the effectiveness of weight update variance (WV) in detecting latent forgetting are reported only on one dataset (CIFAR-10). Therefore, it is hard to believe that the proposed WV is a general effective metric for detecting latent forgetting. 
(3) The compared baselines are limited. Previous works have proposed a large number of FCL methods. Besides replaying-based methods, the authors should also test the other types of methods (e.g., regularization-based EWC, distillation-based CFeD, etc.) under the studied temporal-spatial data heterogeneity. See “Federated Continual Learning: Concepts, Challenges, and Solutions” for more details about the representative types of solutions to FCL and select more baselines from each type to compare.
(4) The experimental setups (e.g., the number and index of classes removed in different FL tasks) lack reasonability. Ablation studies with different experimental setups (e.g., setting the number of classes removed during learning to be 1,3,5,7, resepectively) are needed for validating the generality of the proposed FedMemo. 
5. Extra computational overhead. 
Although the paper argues that the client-side computation overhead is small in Appendix, a quantitative analysis (e.g., extra time or FLOPs compared to standard FedAvg) would make this claim more concrete, especially for resource-constrained edge devices.

Minor issues
- Typos: (1) “however,” in the line 115; (2) “MFCL Babakniya et al. (2023) and Yu et al. (2025) in FCL considers…, they assumes” in lines 257—258; (3) “and and” in the line 709; (4) “Client sideFedMemo” in Table 2; The format of “Client-side” and “Server-side” used in this paper is not unified. (5) “temporal-spatialheterogeneity,” in the line 408; (6) The sentence in the line 866 is not completed.

### Questions
1. Can you provide clear instructions on when to use the client-side FedMemo and when to use the server-side mode? Is there any cost-effective method to determine whether the clients’ generated synthetic data are more effective for tackling latent forgetting than the server’s generated data? Can the two modes be triggered simultaneously?
2. Once the 2-step aggregation is triggered, is it used until the end of learning, no matter whether the decreasing trend of WV is disappeared or not?
3. Why the results of "Client-side FedMemo" are not shown in Table 4? If client-side variant is not applicable to NLP tasks like GLUE, why do we need such a variant? 
4. Can your proposed approach remain effective, if the experimental setup parameters (e.g., the number/indexes of removed classes) are changed? 
5. Are the other types of FCL methods, like LWC, CFedD, applicable to your studied FCL scenarios (i.e., scenarios with temporal-spatial data heterogeneity)? If applicable, can you include them in the comparative experiments?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the FedMemo framework, which detects forgetting using a weight-update variance signal computed from successive global updates, and mitigates it via a proxy-based two-step aggregation. The method is applicable on both server and client sides.

### Strengths
* articulation of temporal-spatial heterogeneity and why this yields latent forgetting.
* variance-based detection of forgetting.
* the pipeline is clearly described.

### Weaknesses
* several strong FL methods that mitigate forgetting and drift (e.g., Flashback, FeGAN, FL‑distillation approaches: FedNDT, FedDF) are mentioned in the related works but are not directly compared empirically.
* requires proxy data to train the server-side GAN (used 5000 samples). Client-side generator costs are also not measured.
* non-standard metrics for GLUE (accuracy for CoLA/MRPC/QQP), reduce comparability to prior work.
* the modeling of the temporal data heterogeneity excludes data points from a given client, which is not very realistic. Stronger work like Oort and REFL (which are not referenced) modeled client availability over time, which is a more realistic approach to vary the data distribution based on client availability.

### Questions
* does the proxy cover all classes, and with what class balance? What is the performance of a model trained only on the proxy (no FL)?
* in GLUE setup, clarify whether prefix parameters are applied only to the final layer or all layers, and why accuracies are low?

### Soundness
2

### Presentation
2

### Contribution
2
