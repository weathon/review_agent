# Backdoor Federated Learning by Poisoning Backdoor-Critical Layers

- Avg Score: 6.00
- Decision: Accept (poster)
- Scores: 6, 6, 6, 6

## Abstract
Federated learning (FL) has been widely deployed to enable machine learning training on sensitive data across distributed devices. However, the decentralized learning paradigm and heterogeneity of FL further extend the attack surface for backdoor attacks. Existing FL attack and defense methodologies typically focus on the whole model. None of them recognizes the existence of backdoor-critical (BC) layers-a small subset of layers that dominate the model vulnerabilities. Attacking the BC layers achieves equivalent effects as attacking the whole model but at a far smaller chance of being detected by state-of-the-art (SOTA) defenses. This paper proposes a general in-situ approach that identifies and verifies BC layers from the perspective of attackers. Based on the identified BC layers, we carefully craft a new backdoor attack methodology that adaptively seeks a fundamental balance between attacking effects and stealthiness under various defense strategies. Extensive experiments show that our BC layer-aware backdoor attacks can successfully backdoor FL under seven SOTA defenses with only 10% malicious clients and outperform the latest backdoor attack methods.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors introduce two layer-specific backdoor attack methods that aligns with the framework of federated learning attackers.
To recognize backdoor-critical layers, they provide a layer substitution method including Local Training, Forward Layer Substitution and Backward Layer Substitution.
Their evaluation, conducted across different models (ResNet18, VGG-19, etc.) and datasets (CIFAR-10, Fashion-MNIST, etc.), demonstrates that the newly proposed layer-wise backdoor attack techniques surpass the performance of existing non-layer-wise methods for backdoor attacks.

### Strengths
1. Extensive experiment results
2. Clear presentation of the proposed method

### Weaknesses
1. It appears that the approach is tailored to a specific model. I'm curious about its performance when applied to other models within the Resnet and VGG families.

2. For the LP attack, the attacker must possess knowledge of the benign workers' model parameters, and for the LF attack, they need to ascertain whether the defense modifies the sign of these parameters.

3. To ensure a fair comparison, the authors may want to take into account layer-specific defenses such as norm-clipping specific to each layer and add noise to only specific layers. 

4. Instead of solely replacing the BC layers after their identification, it might be more optimal for the authors to consider freezing the other layers and fine-tuning the BC layers using poisoned data.

### Questions
1. During the earlier stages and as FL training nears convergence, do the characteristics of the BC layer change, or is it primarily determined by the model architecture?

2. It's noticeable that all BC layers in the provided experiments are fully connected (fc) layers. Is there a specific rationale or intuition behind this? Given that BC layers are often the first fc layer, it could be intriguing to explore adjustments in their structure, size, and hyperparameters. Additionally, is there a possibility that within a single BC layer, certain parameters or neurons are more crucial than others? If this is the case, why not consider selecting critical parameters or neurons across multiple layers instead of exclusively focusing on critical layers?

3. What is the reasoning behind the sequence of Local Training, Forward Layer Substitution, and Backward Layer Substitution, as opposed to individually training each layer using poisoned data and subsequently sorting them?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a backdoor attack in Federated Learning (FL) by identifying Backdoor-Critical (BC) layers, a subset of layers crucial for model vulnerabilities. The paper verifies and exploits these layers, devising a new backdoor attack strategy that balances effectiveness and stealthiness. The introduced Layer Substitution Analysis algorithm and two layer-wise backdoor attack methods, LP and LF Attack, minimize model poisoning while successfully infiltrating FL systems, outperforming recent methods even with a lower percentage of malicious clients.

### Strengths
+ Layer Substitution Analysis can identify the existence of backdoor-critical layers.
+ Utilize the knowledge of backdoor-critical layers to craft effective and stealthy backdoor attacks with minimal model poisoning.
+ Demonstrates effectiveness in bypassing state-of-the-art defense methods and injecting backdoors into models with a small number of compromised clients.

### Weaknesses
The core concept of this paper centers around identifying critical parameters within the model to facilitate effective backdoor insertions. While the strategy of embedding persistent backdoors through carefully identifying key parameters has been previously explored in [1] and [2], the authors must outline the novelty of the proposed method when compared to the techniques presented in [1] and [2].

The paper does not provide any comparisons between the proposed method and the recent state-of-the-art backdoor insertion technique presented in [3].

The primary assumption for identifying the critical layer for backdoor insertion hinges on the global model reaching saturation. However, given the continuous nature of federated learning, the timeline to model saturation can be extensively prolonged based on dataset distributions and the application domain, thereby escalating the adversary's complexity. Furthermore, the persistent nature of the injected backdoor amidst the continuous updates in federated learning remains unclear. The authors need to provide further clarifications on this aspect.

In a paper focused on federated learning, it is expected to see some experimental evaluations conducted on the LEAF [4] benchmark dataset.

[1] Z Zhang et al., "Neurotoxin: Durable Backdoors in Federated Learning", ICML 2022.
[2] M Alam et al., "PerDoor: Persistent Backdoors in Federated Learning using Adversarial Perturbations", IEEE COINS 2023.
[3] H Li et al., "3DFed: Adaptive and Extensible Framework for Covert Backdoor Attack in Federated Learning", IEEE S&P 2023.
[4] https://leaf.cmu.edu/

### Questions
1. How does the proposed method diverge from the techniques presented in [1] and [2] in terms of novelty?
2. Provide comparative analysis between the proposed method and the state-of-the-art backdoor insertion technique outlined in [3].
3. How does the injected backdoor maintain its persistence amidst the continuous updates inherent in federated learning?
4. Provide experimental evaluations on the LEAF [4] benchmark dataset.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes the concept of "backdoor-critical" (BC) layers, which are a small subset of model layers that dominate the model's vulnerability to backdoor attacks.
They introduce a method called Layer Substitution Analysis to identify BC layers from the attacker's perspective. This involves substituting layers between a benign model and malicious model and evaluating the impact on backdoor success rate.
Based on the identified BC layers, they design two new backdoor attack methods - layer-wise poisoning attack and layer-wise flipping attack. These precisely poison only the BC layers to inject backdoors while minimizing detectability. Experiments on CIFAR-10 and Fashion-MNIST datasets show their attacks can successfully bypass state-of-the-art defenses like Multi-Krum, FLAME, and RLR. The attacks achieve higher backdoor success rates and main task accuracy compared to prior attacks

### Strengths
1.	Designs two highly targeted poisoning attacks (layer-wise poisoning and flipping) that precisely exploit BC layers to inject backdoors. Requires minimal model modification.
2.	Comprehensive experiments show the BC layer-aware attacks can bypass state-of-the-art defenses like Multi-Krum, FLAME, RLR etc. Achieves higher attack success rate and main task accuracy.
3.	Analysis of BC layers provides new perspectives for future research into vulnerabilities of federated learning models and development of more robust defenses.
4.	Well-written paper with clear motivation, technical approach, extensive experiments and analysis. Meaningful insights for both attacks and defenses in federated learning.

### Weaknesses
1.	More analysis needed on why certain layers tend to be BC layers, and how factors like model architecture, data, triggers etc. influence this.
2.	Layer substitution analysis to identify BC layers has high computational overhead since it requires retraining models multiple times.
3.	How diversity across clients' data affects BC layer analysis needs more investigation. Paper assumes attacker has representative clean and poisoned data.
4.	No ablation study on key components of the layer-wise attacks like malicious model averaging, adaptive layer control etc.

### Questions
How do different trigger types (pixel, semantic, hardware-based) influence the BC layers? Can triggers be designed to target specific layers?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper highlights a previously underexplored aspect of FL security: backdoor-critical (BC) layers within neural networks. Unlike conventional attacks that target the entire model, focusing on BC layers can lead to equally damaging outcomes with a significantly lower probability of detection by SOTA defense mechanisms.

The contribution is an in-situ approach Layer Substitution Analysis that enables attackers to identify and verify these BC layers. With this knowledge, the authors design 2 attacks: layer-wise poisoning attack and layer-wise flipping attack.

The proposed methodology is tested against SOTA FL defense strategies, demonstrating that even with as few as 10% of the participants in the FL system being malicious, the BC layer-aware attacks can successfully implant backdoors into the model.

The experiments show that these BC layer-aware attacks not only succeed in evading current defenses but also surpass the performance of the latest backdoor attack methods.

### Strengths
- The paper proposes a novel and interesting analysis to precisely identify backdoor-critical layers.
- The paper provides a comprehensive evaluation to show the effectiveness and stealthiness.

### Weaknesses
- The design only consider backdoor success rate, without considering the clean accuracy of the layer substitution, which is not reasonable.
- No discussion on the limitations.

### Questions
1. The proposed method only consider backdoor success rate, why not also consider the clean accuracy of the layer substitution? What if the layer substitution leads to a significant drop in clean accuracy?

2. Could the author explain the intuition behind the layer substitution? Compared with existing attacks, even with less backdoor-related layer, it keeps the backdoor behavior (even enhances it). Why can it be stealthier, surpassing the defense methods?

3. For sensitivity analysis for threshold $\tau$ in Section 4.5, the BSR increase with the increase of $\tau$, which is reasonable. But I am interested in how the MAR changes with the increase of $\tau$. This can illustrate how the number of substituted layer affects the stealthiness. Is there any trend?

4. For Step 2 in Section 3.2, I am curious what is the typical value range of $\Delta BSR_{b2m(l)}$? Is any statistics?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
