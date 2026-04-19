# CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception

- Decision: Reject
- Scores: 5, 3, 5, 5, 8

## Abstract
Collaborative perception (CP) is a promising method for safe connected and autonomous driving, which enables multiple connected and autonomous vehicles (CAVs) to share sensing information with each other to enhance perception performance. For example, occluded objects can be detected, and the sensing range can be extended. However, compared with single-agent perception, the openness of a CP system makes it more vulnerable to malicious agents and attackers, who can inject malicious information to mislead the perception of an ego CAV, resulting in severe risks for the safety of autonomous driving systems. To mitigate the vulnerability of CP systems, we first propose a new paradigm for malicious agent detection that effectively identifies malicious agents at the feature level without requiring verification of final perception results, significantly reducing computational overhead. Building on this paradigm, we introduce CP-GuardBench, the first comprehensive dataset provided to train and evaluate various malicious agent detection methods for CP systems. Furthermore, we develop a robust defense method called CP-Guard+, which enhances the margin between the representations of benign and malicious features through a carefully designed mixed contrastive training strategy. Finally, we conduct extensive experiments on both CP-GuardBench and V2X-Sim, and the results demonstrate the superiority of CP-Guard+.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposed a defense method, named CP-Guard+, against the adversarial attack on intermediate-fusion collaborative perception for connected and autonomous vehicles. Unlike the prior work deploying the hypothesize-and-verify approach to identify the malicious attacker, CP-Guard+ leverages contrastive learning to project benign and malicious feature maps to feature vectors, and then use a classification model to achieve attack detection. Meanwhile, the paper proposed a benchmark of the colalborative perception defense solutions, named CP-GuardBench.

### Strengths
* The paper is well written and easy to follow.
* The contastive learning based classification outperforms prior approaches.

### Weaknesses
* The threat model can be improved to make the attack definition clearer.
* The evaluation did not clarify whether one trained CP-Guard+ is universally usefull for different (unknown) attacks.
* The benefit and functionality of CP-GuardBench is somehow unclear.
* It misses certain related work discussion.

### Questions
I enjoy reading the paper. It is built on a safety-critical problem of collaborative perception, which is not yet deployed in scale but is a critical vehicle application widely discussed. The approach of contrastive learning based classfication is straightforward and brings performance gains, in terms of both defense success rate and computation overhead. However, I have several questions that make me uncertain about the validity of the approach, mainly from a security aspect.

First, the thread model, the adversarial attack on collaborative perception is not clearly defined. I understand the attacker solves an optimization to degrade the victim vehicle's perception performance, but the loss function (attack objective) is a key design choice which the paper did not explain clearly. Are there different choices of the loss function? I think so because I found a security paper on collaborative perception [1] defines a different attack objective. Will the choice of loss function or attack objective affects the effectiveness of CP-Guard+? It is fine to focus on a certain scope of attacks but it must be clearly defined.

Second, which is perhaps my misunderstanding, I am unclear if all experiments are using the same one trained CP-Guard+ or different CP-Guard+ instances given different attack parameters. The later sounds not realistic as a good defense should be universally useful for different variants of attacks and even unknown attacks. In other words, the training details of CP-Guard+, especially the training data to use, is not well defined.

The paper clearly claims CP-GuardBench as one of the major contributions. However, I did not fully understand the use of CP-GuardBench. The benchmark lacks flexibility; it stores the feature space data, which will change on any modification on the perception model or attack methods. As assessing security of a system always needs to consider adaptive attacks, such record of fixed attack methods can hardly be useful for state-of-the-art evaluation for a long time. I appreciate the effort to put these implementation together, but I would recommend to label this as a coding framework, not benchmark. I did not see artifact or code repository links either. Also, what is the difference between CP-GuardBench and the V2X-Sim experiments, given CP-GuardBench is also built from V2X-Sim?

The contrastive learning approach sounds valid to me, but I did not see significant innovation in the algorithm level except using the existing components. The math formulas in the paper are also basic definition of contrastive learning or classification itself. What is the technical contribution besides moving contrastive learning to a new application? At least it deserves a related work section for this.

Lastly, a frequent question for any defense paper: the adaptive attacks. What if the PGD attack uses the classification as feedback during the attack optimization? It could be acceptable to leave it as future work but at least there should be a discussion on such attack oppotunities.

[1] Zhang, Qingzhao, et al. "On data fabrication in collaborative vehicular perception: Attacks and countermeasures." 33rd USENIX Security Symposium (USENIX Security 24). 2024.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a new approach, CP-GUARD+, to detect attacks against collaborative perception (CP), mainly in autonomous driving scenarios. Their method leverages a mixed contrastive training strategy to detect attacks at the feature level and addresses the limitations of prior works in computation cost and latency. This work also constructed a new dataset, CP-GuardBench, which is the first dataset for malicious agent detection in CP systems. Their methods outperform prior work (MADE and ROBOSAC) on the V2X-Sim dataset.

### Strengths
The paper designs a valid method to detect malicious agents in CP at the feature level motivated by the limitations of prior works: computation-intensive and time-consuming. To evaluate the effectiveness of their method, this paper constructed a new dataset,  CP-GuardBench. On the V2X-Sim dataset, their methods outperform the existing methods (MADE and ROBOSAC).

### Weaknesses
I have the following major concerns in this paper:

### Threat model is not clear

I am not fully convinced of their threat model about what kind of attacks they assume and how their defense works on them. This paper assumes a white-box threat model in which adversaries can inject malicious perturbations into their own intermediate features. However, the collaborative perception of autonomous driving has to be real-time.   When does the adversary generate the attacks, e.g., C&W and PGD. As Eq. (1)-(6) assume that the adversary can know the intermediate features of other agents, it should also need communication time. As the CAV is the main target of this paper, this paper should provide more rigorous discussions about their threat model, i.e., how attack and defense works

### No comparison with generic anomaly detection methods

I appreciate their evaluation of MADE and ROBOSAC, which are the recent CP defenses. Meanwhile, this paper should also compare their method not only defense specialized for CP with more generic anomaly detection methods. Their motivation, feature-level detection, makes sense, but this may also remove the motivation to consider CP domain-specific requirements. As the encoder can work to absorb the domain differences, we should apply generic methods for the features because these are already "features". For example, there are many prior anomaly detection methods leveraging contrastive learning [a, b, c]. These are just examples, but this paper should provide more sufficient discussions of why existing works are not suitable for their research or demonstrate that their method has significant advantages via quantitative evaluation.

[a] Cho, Hyunsoo, Jinseok Seol, and Sang-goo Lee. "Masked contrastive learning for anomaly detection." IJCAI 2021
[b] Guille-Escuret, Charles, et al. "CADet: Fully Self-Supervised Out-Of-Distribution Detection With Contrastive Learning." Advances in Neural Information Processing Systems 36 (2024).
[c] Reiss, Tal, and Yedid Hoshen. "Mean-shifted contrastive loss for anomaly detection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 37. No. 2. 2023.

### No comparison with existing methods in their proposed dataset

This paper should provide benchmark evaluation results of existing methods on their proposed dataset. CP-GuardBench. This paper claims that the dataset creation is one of their contributions. In this case, this paper should provide how representative prior methods work on their methods along with their method. Otherwise, we cannot deny the possibility that the dataset could be cherry-picked. If prior work is over-performant or under-performant on their dataset, this paper should provide further cause analysis. As the dataset will be publically used, this paper should provide official benchmark results.

### Building upon an unpublished work 

This paper mentions their unpublished work, CP-Guard, as a prior work. Research paper should not cite the unavailable reference. It also could potentially break the review anonymity. If this paper wants to mention it, they should have published it as a preprint and cited it as a third-party paper. Based on this, I feel that this paper is not ready to be reviewed.

### Questions
- Could you elaborate more about the attack timeline (e.g., when we can get data from ego and helping CAVs, when the attack is generated,  and when will it be deployed)?
- How representative is the CP system described in Fig. 2 in the autonomous driving domain?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses security vulnerabilities in Collaborative Perception (CP) systems for autonomous vehicles, which share information among connected vehicles to enhance perception capabilities. This openness, however, exposes CP systems to potential attacks from malicious agents, which can inject adversarial data to disrupt perception outcomes. To mitigate these risks, the authors propose CP-Guard+, a novel feature-level malicious agent detection framework. Unlike traditional output-based verification methods, CP-Guard+ detects malicious agents by analyzing intermediate features, reducing computational overhead. The authors also introduce CP-GuardBench, the first dataset explicitly designed for training and evaluating malicious agent detection methods in CP systems. CP-Guard+ employs a mixed contrastive training strategy to increase the feature separation between benign and malicious agents, improving detection performance.

### Strengths
1. The work is well-motivated. It is important to have an efficient feature-level detection method for collaborative perception because it is a real-time and safety-critical task by nature. The paper is generally well-written and easy to follow.

2. It is appreciated that the author also publishes the dataset and their methods to generate and annotate the data, which help build a standard benchmark for future works.

### Weaknesses
1. The method (CL) itself is not novel or has not made any adaptation tailored for the collaborative perception tasks. In previous works such as "Driver Anomaly Detection: A Dataset and Contrastive Learning Approach" (WACV 21), "Learning Representation for Anomaly Detection of Vehicle Trajectories" (IROS 23), they both adopt the CL methods to detect anomaly/adversarial attacks in the feature level. At least the author should discuss these works.

2. In anomaly detection, it is important to make sure the methods are well generalized to unseen attacks/anomaly. However, in the methodology, it is unclear whether the model is trained with multiple types of attacks together or only on simple patterns of attacks (e.g. PGD). In the evaluation, it is unclear how the intermediate feature and detector can be generalized (e.g. trained on certain attacks but tested on unseen patterns of attacks).

3. According to some literature, it is not clear whether the assumption "the anomalous feature should cluster together". In anomaly detection, it is always a single-class classification problem and we can assume the benign samples clusters but it is hard to say this applies to the anomalies.

### Questions
Please kindly refer to the weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a new dataset CP-GuardBench as the first one for malicious agent detection in collaborative perception systems and then it proposes CP-Guard+ as a robust malicious agent detection method. Compared with the previous hypothesize-and-verify paradigm, CP-Guard+ can detect malicious agents directly at the feature level in one pass. Experiments on CP-GuardBench and V2X-Sim demonstrate the performance of CP-Guard+.

### Strengths
1. The proposed method is straightforward and easy to understand.

2. The proposed method is compared with two state-of-the-art CP defense methods on both accuracy and time in experiments.

3. The performance of the proposed method is good enough.

### Weaknesses
1. For each collaborative detector, CP-GUARD+ need to train the corresponding binary classifier to detect the malicious agents which has less expansibility.

2. Table 2 is hard to read for comparing the results of different methods. The results of your CP-Guard+ are not highlighted.

3. No experimental results on real-world dataset. The V2X-Sim is the data generated by simulator. It would be better to have experiments on real-world datset.

### Questions
1. In Section 3.1, five attacks are used to generate the attack data. How would the performance of the trained model would be if it meets other types of attacks (not these five attacks)? It is common that the attacker creates new types of attack technology to defeat the defenses. The method should be able to work well even in this situation.

2. Can you consider the out-of-distribution attack? For example, train your binary classifier on only three or four types of attacks and then test on the rest one or two types of attacks.

3. In Table 2, why there is no results on the other two types of attacks? I think this table is the most important to show your performance, the results should be as many as possible.

4. Are the compared baselines trained on your CP-GuardBench? The training data should be the same for fair comparison.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes (1) CP-GuardBench, a pipeline to generate collaborative perception scenarios under malicious attack. The authors built CP-GuardBench by applying 5 different deep learning adversarial generation methods on top of V2X-Sim simulated scenarios. To the best of the reviewer's knowledge, this is the first benchmark in this area. (2) CP-Guard+, a deep learning model to differentiate malicious perception features from benign ones on the encoded feature. The author assumes the cooperative perception is done via a sensor-encode-transmission-fusion-decode pipeline. The author designed this model to intake encoded per-CAV feature maps (both benign and malicious) and outputs a benign/malicious classification by training the model with a contrastive loss to maximize the benign-malicious difference. 

The authors performed experiments using (1) and compared to other methods MADE, ROBOSAC and demonstrated improved performance in detection precision and runtime.

### Strengths
The topic of attack-robust cooperative perception has existed and there are other works that seek to identify attacks, but differentiating the attack on the feature level is new. Thus this work has high originality.

Overall the paper is clear and easy to follow. The authors' methods are well explained and the experiments with competitor methods are sufficient.

### Weaknesses
- There is at least one related paper that the author needs to discuss as a related work: Cooperative Perception for Safe Control of Autonomous Vehicles under LiDAR Spoofing Attacks (Proceedings Inaugural International Symposium on Vehicle Security and Privacy).
- The author made an assumption that an adversarial attacks occurs in the form of a perturbation to the encoded feature map by using the PGD, BIM, C&W, FGSM, GN tasks. In vision detection tasks, such attacks are effective at misleading a model to produce a wrong class label. However, in a cooperative perception task where the object state is more of an interest, such attacks might not be the prevailing form. According to Figure 7, such perturbations will result in hallucinated false objects spreading over the scene. But in reality, an attacker might as well want to inject just one or two fake objects at designated locations in the scene (for example injecting just a single object in front of the ego to force it to stop). In such cases, the perturbation on the feature map will not look like a noise. How will the model respond?

### Questions
- Could the author clarify the `encoder` and `decoder` choice in section 2.1?
- Does the cooperative perception system consider transmission delays? (This might be helpful for the ego CAV can differentiate attacks because it has a no-delay advantage)

### Soundness
3

### Presentation
4

### Contribution
3
