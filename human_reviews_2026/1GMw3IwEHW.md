# Protection against Source Inference Attacks in Federated Learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Federated Learning (FL) was initially proposed as a privacy-preserving machine learning paradigm. However, FL has been shown to be susceptible to a series of privacy attacks. Recently, there has been concern about the Source Inference Attack (SIA), where an honest-but-curious central server attempts to identify exactly which client owns a given data point which was used in the training phase. Alarmingly, standard gradient obfuscation techniques with Differential Privacy have been shown to be ineffective against SIAs, at least without severely diminishing the accuracy.

In this work, we propose a defense against SIAs within the widely studied shuffle model of FL, where an honest shuffler acts as an intermediary between the clients and the server. First, we demonstrate that standard naive shuffling alone is insufficient to prevent SIAs. To effectively defend against SIAs, shuffling needs to be applied at a more granular level; we propose a novel combination of parameter-level shuffling with the residue number system (RNS). Our approach provides robust protection against SIAs without affecting the accuracy of the joint model and can be seamlessly integrated into other privacy protection mechanisms.

We conduct experiments on a series of models and datasets, confirming that standard shuffling approaches fail to prevent SIAs and that, in contrast, our proposed method reduces the attack’s accuracy to the level of random guessing.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates source inference attacks (SIAs) in federated learning, where an honest-but-curious server aims to identify which client owns a given training sample, and proposes both new reconstruction attacks and a defense mechanism under the shuffle model. The authors first demonstrate that standard model-, layer-, and parameter-level shuffling are insufficient to prevent SIAs by designing effective remapping attacks. To counter these, they introduce a parameter-level bit-wise shuffling strategy combined with residue number system (RNS) encoding, which theoretically ensures that only aggregated information is revealed to the server and empirically reduces attack accuracy to random guessing while maintaining model performance. Extensive experiments on MNIST, CIFAR-10, and CIFAR-100 validate the approach, showing strong protection with reasonable communication overhead.

### Strengths
1. The paper targets the relatively unexplored source inference attack (SIA) in federated learning, extending beyond traditional membership or gradient inversion attacks, and contributes both new attack formulations and corresponding defenses within the shuffle model framework.
2. The authors provide a systematic exploration of reconstruction attacks at different granularities (model-, layer-, and parameter-level), clearly demonstrating the limitations of naive shuffling and motivating the need for a more fine-grained defense.
3. Experimental results on multiple benchmarks (MNIST, CIFAR-10, CIFAR-100) show that the proposed method consistently reduces SIA success rates to the level of random guessing, while maintaining comparable accuracy and reasonable communication overhead.

### Weaknesses
1. The reconstruction attacks heavily rely on a shadow dataset that is directly obtained from the attacked clients or from data sources with a similar distribution. This assumption is impractical in most real-world scenarios, and moreover, it introduces severe privacy leakage risks in sensitive domains such as healthcare or finance. The paper should also provide a comprehensive analysis of the impact of shadow data quality, for example, when the shadow dataset does not come from the specific attacked client or exhibits significant distributional bias, or contains noisy data.
2. The proposed attack and defense assumptions are overly idealized. The attacker is assumed to be an honest-but-curious server with a small shadow dataset, while the defense relies on the existence of a fully or partially trusted MixNet shuffler. Both assumptions seem contradicted or are difficult to guarantee in realistic federated learning settings.
3. The limitations of numerical precision in encoding may significantly affect both model convergence and potential information leakage, especially in complex models or datasets. For example, in the CIFAR-100 experiments, achieving lossless precision requires several times higher communication overhead.
4. All experimental datasets and models are designed for vision tasks, which limits the generalizability of the proposed approach to other modalities, such as medical tabular data.
5. The proposed defense is only applicable to sum-based aggregation scenarios, which represents a major limitation given the existence of numerous non–sum-based secure aggregation algorithms designed to counter adversarial attacks.
6. The aggregation formula of FedAvg mentioned around Line035 lacks mathematical rigor regarding aggregation weights.
7. The overall paper structure could be improved. For example, Section 6.1 appears as the only subsection under Section 6.

### Questions
1. How can the authors clearly differentiate the real contributions of this work from the previously submitted (under-review) reference [3]?
2. Could the authors provide a quantitative analysis of the communication cost required to achieve lossless precision on the CIFAR-100 dataset?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper studies source inference attacks (SIA) in cross-silo FL and argues that naïve shuffling at different granularity (model/layer/parameter) does not protect against a curious server equipped with a small shadow dataset. The authors then propose a hybrid encoding–shuffling defense: fixed-point quantization, RNS decomposition, unary encoding, and per-bit shuffling, aiming to leak only the aggregated sum. Experiments on CNN/ResNet over MNIST/CIFAR claim to suppress SIA to random chance without model degradation.

### Strengths
1. Clearly identifies that basic shuffling in FL still leaks client identity.

2. Proposes a new bit-level encoding + shuffling defense, not just adding noise.

3. Shows strong privacy improvement with almost no accuracy loss.

4. Provides both attack and defense experiments to support claims.

### Weaknesses
1. Shadow-dataset assumption is strong; needs sensitivity analysis under distribution shift.

2. Relies on a trusted shuffler, not obvious in many deployments.

3. No evaluation on text/LLM/tabular medical/time-series with only CV toy setups.

4. Multi-round leakage not addressed (momentum, clipping signals, correlated updates).

5. Communication claims hinge on compression + trust assumption, which is not apples-to-apples vs secure aggregation.

### Questions
1. What if the shadow data distribution does not match exactly?

2. Can a server correlate updates across rounds and break anonymity?

3. Comparison to threshold secure aggregation under the same drop-rate constraints?

4. Does this work for Transformers/LLMs or non-image workloads?

5. How realistic is the trusted shuffler assumption? Any decentralized variant?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses Source Inference Attacks (SIAs) in Federated Learning (FL)—attacks where a central server identifies which client owns specific training data. It highlights that traditional defenses (e.g., Differential Privacy, regularization) and conventional shuffling (model-level, layer-level, parameter-level) fail to block SIAs without harming model accuracy.
A new defense combining parameter-level shuffling and Residue Number System (RNS) is proposed: parameters are scaled, RNS-encoded, unary-encoded, bit-wise shuffled, then decoded/aggregated by the server. Validated on MNIST/CIFAR-10/CIFAR-100 with CNN/ResNet-18, it reduces SIA accuracy to random guessing, preserves model performance, integrates seamlessly into shuffle-model FL, and has controllable communication costs.
Key Contributions:
1. Identifies vulnerabilities of conventional shuffling via 3 reconstruction attacks.
2. Proposes the first shuffle-model FL defense to neutralize SIAs (random-guess accuracy).
3. Extends defense to resist Data Reconstruction Attacks.

### Strengths
1. First systematic defense against Source Inference Attacks (SIAs) in Federated Learning, introducing a novel parameter-level shuffling and RNS-based mechanism that reduces SIA accuracy to random guessing.
2. Well-structured “problem–proposal–verification” format with clear explanations, visual aids, and comprehensive appendices.
3. Addresses a core privacy challenge in cross-silo FL, offering compatible, low-cost protection against both SIAs and DRAs. Expands FL privacy theory and establishes a new paradigm for noise-free privacy amplification.

### Weaknesses
1. Lack of Discussion on Detailed Synergistic Optimization Between the Mechanism and Differential Privacy (DP)：
The paper claims that the proposed mechanism can be "seamlessly integrated with other privacy mechanisms such as DP" (meeting Specification S.2), yet it fails to verify the actual performance after integration or provide a specific integration scheme. DP requires adding noise to protect privacy, but the RNS encoding of the mechanism may interact with the noise distribution (e.g., noise could cause parameters to exceed the RNS encoding range). Additionally, the balance among "privacy gain, accuracy loss, and communication cost" after integration has not been quantified. As a result, this feature remains at the theoretical level and lacks practical guiding significance.
2. Improvement: Design an integrated "RNS + DP" scheme where clients first add DP noise to parameters, followed by RNS encoding and shuffling. Test the SIA defense effect, model accuracy, and communication cost under different DP noise intensities (ε= 1, 2, 5) on the MNIST dataset to identify the optimal integration parameters (e.g., when ε= 2, the mechanism + DP can reduce the SIA accuracy to below 10% while maintaining 95% model accuracy).

### Questions
It is suggested to supplement the "quantitative correlation analysis between RNS modulus selection, communication cost, and model accuracy" and clarify the decision-making basis for the optimal modulus combination under different scenarios.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigate the defense against source inference attacks in FL in the shuffle model. On a high level, if the server knows a subset of the target client's training data, it can use that to identify the target client's share from the shuffled results it receives. To address this, the authors propose to do bit-level shuffling for each parameter of the client. The authors argue that this means that after reconstructing the server knows nothing more than the aggregated result. Empirical results demonstrate the effective of such defense.

### Strengths
1. Interesting setting and important problem to study.
1. Theoretical results on the security of the proposed shuffling algorithm.
1. Comprehensive discussions on different trust models and different variations of the proposed method.
1. A lot of experiments of different settings.
1. Comparison to secure aggregation.

### Weaknesses
1. Unclear settings for the experiments in Section 7.
1. I would like more clarification on the trust model of the shuffler.
1. Discussions of the security of the proposed method beyond SIA.

### Questions
1. In the experiments, how are the coprimes set? Does this affect security?
1. There are different levels of trust for the shuffler. This should be clarified and explained in a clearer way. If it is completely trusted, then we can send the model weights or bits in plaintext to the shuffler and trust it to perform shuffling. We can also only trust it to perform shuffling and then hide the plaintext weights or bits from it, i.e., it is a trusted shuffling router of encrypted messages (like Cloudfare). It can further be malicious where we would need ZKPs. In section 7, is the shuffler trusted? i.e., are the bits/models sent to the shuffler in plaintext and it shuffles without verification and sends the shuffled results to the server? I understand that we need an honest shuffler for the compression technique to work, but for the other experiments, is the shuffler also trusted? If we are encrypting messages to the shuffler, then doing more granular shuffling would means more overhead for the client and the server.
1. Compared to secure aggregation, I understand that each client's message size do not increase with the number of clients, and the assumptions on the server/shuffler can be different. However, is the security guarantees the same? Does the shuffling with encoding scheme provide cryptographic semantic security? If so, its power shouldn't be limited to SIAs.

### Soundness
3

### Presentation
3

### Contribution
3
