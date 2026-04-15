# Label Privacy Source Coding in Vertical Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 5, 5

## Abstract
We study label privacy protection in vertical federated learning (VFL). VFL enables an active party who possesses labeled data to improve model performance (utility) by collaborating with passive parties who have auxiliary features.  Recently, there has been a growing concern for protecting label privacy against semi-honest passive parties who may surreptitiously deduce private labels from the output of their bottom models. However, existing studies do not remove the prior label information in the active party's features from labels in an offline phase, thus leaking unnecessary label privacy to passive parties.
In contrast to existing methods that focus on training-phase perturbation, we propose a novel offline-phase data cleansing approach to protect label privacy without compromising utility. Specifically, we first formulate a Label Privacy Source Coding (LPSC) problem to remove the redundant label information in the active party's features from labels, by assigning each sample a new weight and label (i.e., residual) for federated training. We give a privacy guarantee and theoretically prove that gradient boosting efficiently optimizes the LPSC problem. Therefore, we propose the Vertical Federated Gradient Boosting (VFGBoost) framework to address the LPSC problem. Moreover, given that LPSC only provides upper-bounded privacy enhancement, VFGBoost further enables a flexible privacy-utility trade-off by incorporating adversarial training during federated training. Experimental results on four real-world datasets substantiate the efficacy of LPSC and the superiority of our VFGBoost framework.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper studies the problem of protecting label privacy in vertical federated learning. In this setting, training data is vertically split into features and labels. There is one special player named Active party that owns labels and some features; there can be multiple Passive Parties that own disjoint features. Previous works have identified severe privacy issues when applying vanilla vertical federated learning algorithms. This work proposes a novel method of adjusting weight and label of each sample without leaking raw label information. In this paper, the authors make the assumption that the active party has features that are informative of the labels, and use gradient boosting to learn the new label-ID joint distribution. The privacy can be further enhanced with adversarial training. The experiment results show that the proposed method can achieve better privacy-utility trade-offs.

### Strengths
The idea of adjusting sample weights are novel and interesting. The closed form solution of AdaBoost is clean and nice. The experiment results are promising, showing that the proposed method is strong against listed attacks.

### Weaknesses
My biggest concern is that the setting may be too restricted. In particular, the active party must own informative features. Let us look at Theorem 1. In the very extreme case, assume that the KL-divergence of $p_{gt}$ and $p_{act}$ is 0. Then Theorem 1 says that the LPSC does not leak anything. However, in this case we do not need federated learning: the active party can just learn with its own features and no communication with passive parties are needed. It requires more justification for the setting (active party with informative features, label only privacy, etc) considered in this work.

Mutual information is known to be closely related privacy. [1] is an important reference.

[1] Cuff, Paul, and Lanqing Yu. "Differential privacy as a mutual information constraint." Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security. 2016.

### Questions
- It looks like all the datasets are for binary classification. Can LPSC work with non-binary labels?
- According to Eq.5, it seems that in adversarial training, the original labels are used. Will this bring extra privacy issues?
- Is it possible to prove formal privacy guarantees, e.g. differential privacy for LPSC?
- In Appendix B.1, you mentioned " we can use boosting to reduce the bias of the local model by using passive parties’ auxiliary features." I am a bit confused, does LPSC uses information from passive parties?

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose an offline-phase data cleansing approach to protect label privacy without compromising utility. Specifically, the idea is to formulate a Label Privacy Source Coding (LPSC) problem to remove the redundant label information in the active party’s features from labels, by assigning each sample a new weight and label (i.e., residual) for federated training. 

The authors propose the Vertical Federated Gradient Boosting (VFGBoost) framework to address the LPSC problem with a theoretical guarantee. Moreover, given that LPSC only provides upper-bounded privacy enhancement, VFGBoost further enables a flexible privacy-utility trade-off by incorporating adversarial training during federated training. Experimental results on four real-world datasets substantiate the efficacy of LPSC and the superiority of our VFGBoost framework.

### Strengths
+ A unique perspective from the offline phase and present an interesting idea.
+ Provide theoretical guarantee and soundness.
+ Provide comprehensive experiments conducted on four real-world datasets in practical scenarios, such as recommendation and healthcare.

### Weaknesses
- Related works need to be improved. Section 2 contains many repeated contents.
- The experimental findings could be more detailed.

### Questions
1. The authors redefine privacy and introduce a privacy-utility trade-off. In the related work, the authors also mentioned differential privacy, which has a similar trade-off. Could the authors elaborate on the difference between them? 
2. How does adversarial training impact the privacy in your experimental findings? Could the authors explicate the insights/findings?
3. The authors give a comprehensive analysis of newly defined privacy and its leakage. The idea is interesting since it introduces a new perspective on privacy. Actually, I feel a little confused about why Definition 1 and Definition 2 are required. What is the insight/intuition of privacy guarantee? What does the newly defined privacy essentially protect? How do you measure the privacy loss in practice/experiments? Why privacy leakage is defined as mutual information?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a novel framework to defend the data leakage problems in VFL, an important issue in collaborative learning scenarios.   The proposed LPSC approach is based on  mutual information optimization, which is solved by boosting approaches, and is able to reduce label leakage while preserving model utility in the evaluated experimental settings. Combination of LPSC and other methods also demonstrate promising utility-privacy trade-off.

### Strengths
1. The paper proposed a novel approach to defend label attacks in VFL settings and demonstrate experimentally the effectiveness of the method on multiple datasets.

2. Theoretical analysis is conducted to guide the design of the framework. 

3. The proposed approach is able to achieve privacy protection without hurting model utility.

### Weaknesses
1. Notations are inconsistent and confusing, making the paper hard to follow. There are many inconsistent notations such as X_0^loc and X_0, y^loc and y, i^loc and i, both denoting active party's data, p*_{lpsc} and p_{lpsc} etc. In Figure 3, it is unclear whether it illustrates the joint distribution or weight distribution. Eq.4 is also confusing, in which the federated model training is a function of i, the ID. The authors are suggested to proofread the entire manuscript to make notations easy to understand and consistent. 

2. Experimental evaluations are insufficient to support the main claims of the work. The comparison with other methods are only conducted  on PMC attack, when the adversarial loss of the proposed method is trained also against PMC attack. No comparison on other label leakage attacks or feature attacks are provided, whereas feature leakage attacks are also considered in the security definition. 

3. It appears that the effectiveness of proposed method depends on the quality of the local features of the active party. In the extreme case that no features are available on the active party but labels, the method may not apply. The impact of the importance of the local features of the active party needs to be evaluated. 

4. Some implementation details are missing. For example, how is LPSC+DP or LPSC+Marvell trained? why not LPSC+MID?

### Questions
1. How are importance of the local features affect the evaluation?

2. How are the proposed method compared with other approaches on attacks other than PMC? 

3. Can you explain why it is reasonable to assume the conditional distribution to be the same in Theorem 2?

4.how is LPSC+DP or LPSC+Marvell trained? why not LPSC+MID?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents an approach to enhancing label privacy in vertical federated learning (VFL). The paper first introduces a privacy notion based on the concept of mutual information, then formalizes the privacy protection task into the Label Privacy Source Coding (LPCS) problem, and demonstrates that gradient boosting is a suitable method for solving the LPCS problem. Subsequently, the Vertical Federated Gradient Boosting framework is proposed to efficiently optimize the LPCS problem.

### Strengths
1. The proposed solution is interesting.

2. The experimental evaluation is extensive.

### Weaknesses
1. The new privacy definition needs to be more formally and rigorously defined. The adversary model and the privacy guarantee provided by the new privacy definition should be clearly delineated. Without a clear description of the adversary's capabilities, the privacy definition's theoretical foundations remain unclear. 

2. The paper does not provide a formal analysis of the new privacy definition's properties. In particular, it remains unclear whether this definition adheres to the axioms laid out in https://www.cse.psu.edu/~duk17/papers/axioms.pdf. The paper would be greatly strengthened by a discussion on this aspect, identifying any axioms that are not met and providing a reasoned argument for why such deviations are acceptable within the context of the problem being tackled.

3. An in-depth discussion of why VFGBoost surpasses label DP would be beneficial.

### Questions
See the weaknesses

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
