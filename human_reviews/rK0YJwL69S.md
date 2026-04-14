# Protecting against simultaneous data poisoning attacks

- Decision: Accept (Poster)
- Scores: 3, 6, 6, 6, 6, 6

## Abstract
Current backdoor defense methods are evaluated against a single attack at a time. This is unrealistic, as powerful machine learning systems are trained on large datasets scraped from the internet, which may be attacked multiple times by one or more attackers. We demonstrate that multiple backdoors can be simultaneously installed in a single model through parallel data poisoning attacks without substantially degrading clean accuracy. Furthermore, we show that existing backdoor defense methods do not effectively defend against multiple simultaneous attacks. Finally, we leverage insights into the nature of backdoor attacks to develop a new defense, BaDLoss (**Ba**ckdoor **D**etection via **Loss** Dynamics), that is effective in the multi-attack setting. With minimal clean accuracy degradation, BaDLoss attains an average attack success rate in the multi-attack setting of 7.98% in CIFAR-10, 10.29% in GTSRB, and 19.17% in Imagenette, compared to the average of other defenses at 63.44%, 74.83%, and 41.74% respectively. BaDLoss scales to ImageNet-1k, reducing the average attack success rate from 88.57% to 15.61%.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes BaDLoss, a defense method designed to detect and mitigate multiple simultaneous backdoor attacks in machine learning models. By analyzing the loss dynamics of individual training samples, BaDLoss identifies and filters out poisoned data, effectively lowering attack success rates while preserving model accuracy. The approach is evaluated across multiple datasets and attack types, demonstrating strong performance and adaptability, making it a robust solution for complex, real-world backdoor attack scenarios.

### Strengths
The paper presents a novel approach to defending against simultaneous multi-backdoor attacks, addressing a challenging scenario that previous work has largely overlooked. The proposed BaDLoss method effectively utilizes loss dynamics to detect poisoned data, reducing attack success rates while preserving model accuracy. Furthermore, the paper provides a comprehensive evaluation across multiple datasets and attack types, showcasing the method's robustness and adaptability in diverse training environments.

### Weaknesses
1. The paper’s threat model could be clarified. If attackers control the training data, it would be helpful to understand why defenders would still need to detect and retrain on filtered data post-contamination, rather than directly identifying and removing poisoned samples before the training process.
2. The paper states that no current defenses can detect multiple backdoor samples. However, "Towards A Proactive ML Approach for Detecting Backdoor Poison Samples" presents a proactive method that addresses various backdoors. Including a comparison or discussion of this approach could provide a more comprehensive evaluation.
3. The paper does not evaluate the proposed defense method on large-scale datasets such as ImageNet. Evaluating on larger datasets would help demonstrate the method's scalability and practical effectiveness, as backdoor detection can be more challenging with increased data volume and complexity.

### Questions
1. How scalable is BaDLoss when applied to large-scale datasets like ImageNet, and what are the computational costs associated with tracking loss dynamics across extensive data? 
2. How effective is the proposed method in defending against dynamic backdoor attacks, where attackers modify the poisoning strategy to mimic clean data loss dynamics?

### Soundness
2

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
3

### Summary
The author considers an interesting scenario where multiple attackers simultaneously poison a dataset. The author finds that traditional poisoning methods fail in this context. Therefore, a defense method called BaDLoss, based on loss trajectories, is proposed to be robust against multiple simultaneous backdoor attacks.

### Strengths
1. I think simultaneously poison is an interesting topic.

2. A well-organized presentation and clear article structure

### Weaknesses
1. Although the method is relatively simple, I find it acceptable.

2. While the experiments are comprehensive, they lack analysis.

### Questions
1. Although the author provides a brief overview of why existing methods fail under simultaneous poisoning, I believe this is insufficient. A detailed theoretical explanations of why each defense fails under multiple attacks is needed.

2.The reasons for the failure of some methods are not always applicable. For example, the author mentions that Neural Cleanse fails because too many other classes are attacked, but this scenario is not guaranteed to occur. The author should provide a sensitivity analysis showing how Neural Cleanse's performance changes as the number of attacked classes increases. This would give readers a clearer understanding of when this defense starts to break down in multi-attack scenarios.

3. I'm wondering why STRIP was not included if there are specific reasons, the author should include STRIP in their experimental comparisons.

4.I'm wondering why BaDLoss performs differently across various datasets. This inconsistency in defense effectiveness between datasets requires further analysis. The authors should  provide a more detailed analysis of how dataset characteristics (e.g., number of classes, image complexity, dataset size) might influence BaDLoss's performance.

Typos:
In line 370, there is an extra "with".

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
The paper proposes BadLoss, a backdoor defense that removes poisoned samples by measuring the distance of samples' loss trajectories from the trajectories of a clean set. The defense is validated on poisoned training data containing multiple backdoor attacks that use patch, blend and frequency triggers. The paper compares the efficacy of the defense against a set of standard defenses and observes that existing defenses are ineffective for the examined setting.

### Strengths
- The writing is clear and easy to follow.
- The defense is well-motivated and the method makes sense.
- The defense requires significantly less clean data than defenses that require fine-tuning.
- Defending against simultaneous data poisoning attacks is an important and understudied topic.

### Weaknesses
- Studying the multi-attack setting on small benchmarks requires very high poisoning rates and results in very odd training dynamics. This is especially concerning when evaluating a defense that relies so heavily on training dynamics.  

- The motivation for the analysis in section 3.3 is not clear. The classic motivation for not degrading clean accuracy is to ensure the stealthiness of the backdoor [1]. However, if the poisoning is visible from the erratic loss/accuracy curves of the model, the point is largely moot. An analysis on that would have been more valuable.  

- Figure 8 provides important information on loss trajectories, but is cluttered and difficult to interpret. The legend is covering a lot of the figure. Maybe it could be split into multiple figures?  

- The clean accuracy degradation and required rejection rate is quite high. As shown in table 3, the defense requires 40% rejection rate to completely remove a patch attack from CIFAR-10.  

- Some phrasing in the threat model is vague, particularly: "The defender has complete control over the training process, while the attacker has neither **knowledge** nor control.'' What is knowledge of the training process? The model architecture? What classes are in the dataset? What specific attacker capabilities are being included/excluded here?

- Potentially missing references on the multi-attack setting: [2] explores the detection setting for multiple backdoors on image classifiers. [3] corroborates the patch boosting effect observed by the authors.

### Questions
- The above weaknesses section contains some questions. Also, were the hyperparameters of the defense tuned against the same set of attacks you evaluate against? It would be interesting to see how hyperparameters tuned to defend against a patch attack work for other types of attacks.  
  

 - Do you expect that the unstable training dynamics would persist when training multi-attacks on a larger benchmarks where the required poisoning rate would be lower?
    
- In figure 8 it looks like the test loss is trending up throughout the entire training run? Could the defender have prevented successful backdoor via early stopping?

References

[1] Gu, Tianyu, et al. "Badnets: Evaluating backdooring attacks on deep neural networks." (2019).  

[2] Xiang, Zhen, David J. Miller, and George Kesidis. "Post-training detection of backdoor attacks for two-class and multi-attack scenarios." (2022).  

[3] Schneider, Benjamin, Nils Lukas, and Florian Kerschbaum. "Universal Backdoor Attacks." (2023).

### Soundness
2

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
3

### Summary
This paper proposed a new defense, namely BaDLoss, to simultaneously defend against multiple backdoor attacks to machine learning classifiers. The proposed defense is evaluated on multiple backdoor attacks and is compared with multiple baselines on three benchmark datasets.

### Strengths
1. A machine learning model can be corrupted by multiple backdoor attacks. How to defend against multiple backdoor attacks simultaneously is largely unexplored in the research community. 

2. The problem is well-motivated. Existing defenses are shown to be insufficient for multiple attacks, which motivates the authors to propose a new defense for this scenario. Overall, I feel the setting considered in this paper is interesting.

### Weaknesses
1. It is assumed that multiple attacks cannot target the same image. As acknowledged by the authors, this assumption may not hold in practice. For instance, an attacker can perform multiple backdoor attacks on the same image with the same target label. 

2. The insights of the proposed defense can be discussed. For instance, the authors may consider providing insights on why backdoored samples have a different training trajectory from the clean samples. 

3. Will the proposed method be effective for clean-label backdoor attacks?

4. Rejecting 40% of training samples can hurt performance in certain application scenarios, especially when the task is complex. For instance, based on Figure 6, the performance drop on Imagenette is larger than the other two datasets. The potential reason is that Imagenette is a more complex classification task.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the problem of defending against multiple simultaneous data poisoning (backdoor) attacks during the training of deep learning models and proposes a defense method called BaDLoss. Unlike previous studies, the authors point out that in real-world scenarios, models may suffer from multiple attacks simultaneously, while most existing defense methods can only handle single attacks. BaDLoss detects potential backdoor samples by tracking the loss trajectories of individual samples during training and excludes these samples from the dataset during retraining, thereby improving defense against multiple attacks without significantly degrading model performance. Through experiments on CIFAR-10, GTSRB, and Imagenette datasets, BaDLoss demonstrates superior performance compared to existing defense methods in multi-attack scenarios.

### Strengths
1.The paper addresses the issue of multiple simultaneous backdoor attacks, which is an important problem that has been largely overlooked in the existing literature. With the growing application of deep learning models and the complexity of training data sources, multiple attackers may tamper with the dataset simultaneously, making multi-attack scenarios a more realistic threat. 

2.The paper proposes BaDLoss defense method identifies anomalous samples by tracking the loss trajectories of individual samples during training. This method fully exploits training dynamics to detect backdoored samples, regardless of whether these samples exhibit unusually high or low losses during training. Compared to traditional defense methods, BaDLoss significantly reduces the attack success rate in multi-attack scenarios while maintaining high clean data accuracy.

3.The paper validates BaDLoss on three datasets with different characteristics: CIFAR-10, GTSRB, and Imagenette, covering variations in image size, class distribution, and more. The experimental results show that BaDLoss significantly outperforms other defense methods in multi-attack scenarios.

### Weaknesses
1.Although the paper points out the deficiencies of existing defense methods in multi-attack scenarios, it does not provide an in-depth analysis of the specific reasons for their failures. For example, methods like Neural Cleanse and Spectral Signatures perform well in single-attack settings but fail in multi-attack scenarios.

2.The experiments in the paper assume that each image can only be affected by one type of attack, which simplifies the experimental setup. However, in real-world scenarios, a single image may contain multiple attack features, which could have a greater impact on model performance.

3.BaDLoss identifies backdoored samples by tracking the loss trajectories of all training samples, which requires storing and analyzing a large amount of intermediate data at each training stage. This could introduce additional computational overhead, especially in large-scale datasets or model applications. The authors are encouraged to provide a detailed analysis of the computational complexity of this method and discuss its feasibility and optimization strategies for practical deployment.

### Questions
1.In real-world applications, multiple different types of attack features may be superimposed on the same image. Can BaDLoss effectively handle such situations? Is it possible to improve the analysis of loss trajectories to further enhance the method's adaptability to complex attack scenarios?

2.In the experiments, BaDLoss outperforms other defense methods significantly under multi-attack scenarios, but its performance in single-attack settings is comparable to or slightly inferior to other methods. Have the authors considered making some adjustments to BaDLoss, such as optimizing the detection strategy for specific attack types, to improve its performance in single-attack scenarios?

3.The paper mentions that BaDLoss identifies potential backdoored samples by rejecting those farthest from clean sample loss trajectories. Could this strategy mistakenly filter out some minority class samples? In cases of imbalanced class distribution, is there a mechanism to ensure that minority class data is not mistakenly removed due to anomalous trajectories?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses the challenge of defending machine learning systems against multiple simultaneous data poisoning attacks, which is increasingly relevant in real-world applications where large datasets are involved. The authors highlight the limitations of existing defenses that are effective in single-attack scenarios but fail under multi-attack conditions. The primary contributions of the paper are identification of the multi-attack threat and introduction of BaDLoss.

### Strengths
1. The paper presents an advancement in the field of machine learning security by addressing the novel challenge of defending against multiple simultaneous data poisoning attacks. While prior research has largely focused on single-attack scenarios, this work recognizes the complexity of real-world applications and the increased risks associated with multi-attack settings.
2. The research demonstrates high methodological quality through rigorous experimentation and robust validation of the proposed defense mechanism. The authors carefully design experiments that evaluate BaDLoss in both single-attack and multi-attack contexts, providing empirical evidence of its effectiveness.
3. The logical flow from the introduction of the problem to the presentation of BaDLoss and its evaluation ensures that readers can easily follow the authors' arguments and findings.
4. The significance of this work lies in its potential impact on the field of machine learning security. By successfully demonstrating a method that defends against multiple data poisoning attacks with minimal loss in model utility.

### Weaknesses
1. Limited exploration of poisoning ratio effects. While the paper acknowledges the impact of poisoning ratios on detection effectiveness, it could benefit from a more comprehensive exploration of this dimension. Specifically, the authors could conduct experiments that systematically vary poisoning ratios across different attack types to better understand the thresholds at which BaDLoss operates effectively. 
2. Minority class considerations. The methodology’s tendency to mark minority-class data as anomalous could exacerbate existing imbalances in the dataset. The authors could address this issue by implementing strategies to ensure that the filtering process accounts for class representation. 
3. Detailed evaluation of  counter-attacks. Although the paper briefly discusses the potential for adaptive counter-attacks, it lacks an in-depth analysis of how an informed attacker might exploit the defense mechanism. Providing case studies or simulated scenarios where attackers adapt their strategies in response to BaDLoss would enrich the discussion. 
4. Limited real-world application discussion. The implications of the findings for real-world applications are somewhat underexplored. A more thorough discussion on how BaDLoss could be integrated into existing machine learning workflows or what practical challenges might arise during implementation would enhance the paper's relevance. 
5. It is more reasonable to use an adaptive threshold to determine the reject ratio than a fixed ratio.

### Questions
1. In the context of this paper, how many levels of multiple attacks are there? Does the number of attack levels affect the performance of the model?
2. The BaDLoss proposed in this paper does not perform particularly well in a single-attack environment, with its clean accuracy being lower than that of most single-defense solutions. Does this imply that the model is only effective in scenarios involving multiple attacks?
3. Figure 2 is inconsistent with the accompanying textual description.

### Soundness
3

### Presentation
2

### Contribution
3
