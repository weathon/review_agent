# Data-Centric Defense: Shaping Loss Landscape with Augmentations to Counter Model Inversion

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 3, 3, 3

## Abstract
Machine learning (ML) models are widely used but vulnerable to privacy attacks, such as model inversion. Current defense techniques are mostly model-centric, involving modifying the model training or inference process. However, these approaches require model trainers' cooperation, are computationally expensive, and often result in a significant privacy-utility tradeoff.
To address these limitations, this paper proposes a novel data-centric approach to mitigate model inversion attacks. Compared to traditional model-centric techniques, our approach offers the unique advantage of decentralization, enabling individual users to control their data's privacy risk. Specifically, we introduce several privacy-focused data augmentations that modify the private data uploaded to the model trainer. These augmentations will shape the resulting model's loss landscape, making it challenging for attackers to generate private target samples. Additionally, we provide theoretical analysis to support our approach and explain how data augmentation can reduce the risk of model inversion. We evaluate our approach against state-of-the-art model inversion attacks and demonstrate its effectiveness and robustness across various model architectures and datasets. Specifically, in standard face recognition benchmarks, we reduce face reconstruction success rates to less or equal to 5%, while maintaining high utility with only a 2% classification accuracy drop, surpassing state-of-the-art model-centric defenses by 90%. This is the first study to propose a data-centric approach for mitigating model inversion attacks, showing potential for decentralized privacy protection.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a novel defense mechanism for (generative) model inversion attacks (MIAs). The basic intuition behind the approach is to add additional, unrelated surrogate samples to the training data and label them with the same label as the class to be defended. The training procedure itself is not changed in any way. To push the MIAs' optimization away from the true samples and towards the surrogate samples, a fraction of the true samples is mislabeled to increase their training loss. Since this step degrades the model's utility, a Gaussian noise-based augmentation is used to create more variations of the same sample and then only mislabel a share of those augmented versions. The defense approach is theoretically motivated by using the capture theorem of Bersekas and empirically evaluated on various common MIAs.

### Strengths
- By introducing a quite simple, yet effective data-centric defense mechanism for MIAs, the paper opens an interesting direction for model defense. Compared to existing defenses, no adjustments to the model's architecture are required, which makes the approach more universal compared to, e.g., BiDO and MID.
- The theoretical foundation of the approach is well-written and clearly motivates the defense mechanism.
- The empirical evaluation is extensive and uses various attack and defense approaches, as well as different architectures and datasets. This makes the experimental protocol basically convincing (even if I think the evaluation could be improved, see the Weaknesses).

### Weaknesses
- I expect that by introducing a significant number of misleading samples that look nothing like the true classes in the training data, there will be some undesired side effects. For example, I expect the model's robustness to adversarial examples to be reduced by the defense method. Particularly due to the Gaussian Noise augmentation in combination with mislabeling the resulting samples the model already has a natural adversarial perturbation pattern learned, which could then be exploited by a standard adversarial example algorithm. So the method probably trades robustness for privacy. Unfortunately, the paper does not investigate this potential drawback but focuses only on the prediction accuracy of the target model.
- While the evaluation investigates a wide range of attacks, architectures and datasets, I think that additional metrics are required. Related MIAs also compute, for example, the feature distance in the evaluation model or use a pre-trained FaceNet model for it. I think such a metric would be beneficial to better assess the similarity between reconstructed samples and the true training samples. Moreover, I think the attack accuracy can be rather brittle since the evaluation model can be susceptible to adversarial features incorporated in the attack results.
- Also, the qualitative samples seem to be cherry-picked. A clarification of how those samples were selected would help here. Also, stating a larger number of attack results in the Appx. helps to qualitatively analyze the results and make sure that the defense indeed works as promised.
- I like the overall ablation and sensitivity analysis and think it really helps to demonstrate the effectiveness of the approach. However, I do not think that the GTSRB dataset used in the main paper is the best choice here since the classes are quite easy to separate. Conducting these analyses on a more fine-grained dataset, e.g., CelebA, would offer more informative insights. Also, I think that the investigation of "How to choose surrogate samples", which only uses four target identities, is not statistically reliable. More target identities are required here to draw a reliable conclusion. Similarly, the evaluation of the number of protected samples in Fig. 4 in the Appx. is important but using GMI as the evaluation algorithm seems not to be a good choice since various MIAs have stated low performance for this attack. Using Mirror-B and PPA would make the results much more convincing.


Small Remark:
- There is a typo in Lemma 2: "defined in Eq. equation 1"

### Questions
- How does the approach work for classes with only a small number of training samples available? Datasets like CelebA usually have something like 20-30 samples per class available. What if the number decreases to only 5 or 10?
- Defending a subset of the targets and adding additional augmented samples leads to a class unbalance between defended and undefended classes. Can this be a problem for training the model and its resulting predictive performance?
- Why is a larger pi (pi_1=0.3) used for PPA compared to the other attacks?
- Is there any side effect on the unprotected classes? Are the attacks similarly effective for those when protecting other classes with DCD or increase/decrease the privacy leakage of those?

### Soundness
3 good

### Presentation
3 good

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
This work presents a novel approach of mitigating model inversion attacks, which is centred on the data rather than the model. Authors influence the loss landscape by introducing carefully crafted client-side data augmentations.

### Strengths
This work addresses a very important topic of MI vulnerability in ML and is aimed at the users (and their data), making it very applicable to many real-world deployments. The work is well-structured, has a strong motivation and many empirical results. What distinguishes this work from many similar ones in the field is the theoretical analysis of these findings as well as extensive evaluation on many computer vision dataset.

### Weaknesses
I do have a number of concerns, however.

Firstly, using data augmentations against inference attacks is hardly a novel [1] (or a particularly interesting) idea. While I understand that these can reduce adversarial advantage, this is not universally the case for different attack types (even within the realm of model inversion alone) and can pose challenges when taking other privacy attacks into account (more on that later).

The problem with this approach is that it is also difficult to quantify and does not have any robust guarantees attached to it (unlike DP, for instance, which you critique on page 3).
Additionally, authors do not report any evidence to suggest that this approach is scalable to other types of MI: gradient-based attacks or reconstruction from weights. I would like the authors to either discuss this limitation of present evidence that their method can be used against these adversaries as well. Same goes for scalability to other modalities: authors position this work as a data-centric defence against MI attacks, not against computer vision MI attacks. I would like to see evidence that attacks on other modalities also see a similar reduction in performance, as otherwise the applicability of this method is severely limited.

One potential criticism I have is the loss-controlled method you use to protect the target class. You directly mislabel a number of target class representatives (page 4), which leads to a higher loss and should, therefore, reduce the adversarial risks. What this also does, however, is more memorisation for these target representatives ([3,4]), making them more vulnerable to other privacy attacks (such as [5]), thus potentially compromising their privacy. Memorisation on its own is a big tangential topic, but this seems like a limitation of this method in my eyes, as you do not consider other (very related [2]) attackers and how your method can affect their results.

### Questions
Another concern I have is the ease of adversarial sample generation. Since these are much easier crafted close to the decision boundary (i.e. they are more likely to have higher loss associated with them), this method can potentially make the learning setting more vulnerable to these attacks too. Can authors offer any evidence that this is not the case?

I also could not find any performance metrics: how does the proposed method affect the training time, time-to-convergence etc.

Minor: some abbreviations were not properly introduced (e.g. GMI, which could mean either generative or gradient model inversion).

Overall, I am not fully convinced that this work proposes a robust novel solution to the issue of model inversion. One of my main criticisms is that by using a loss-based defence, there is a possibility that some of the samples become more vulnerable to other adversarial attacks. 
I would be happy to change the score, however, if the authors address my comments above (particularly on other attack types).


[1] - Kaya, Yigitcan, and Tudor Dumitras. "When does data augmentation help with membership inference attacks?." International conference on machine learning. PMLR, 2021.
[2] - Yeom, Samuel, et al. "Privacy risk in machine learning: Analyzing the connection to overfitting." 2018 IEEE 31st computer security foundations symposium (CSF). IEEE, 2018.
[3] - Feldman, Vitaly, and Chiyuan Zhang. "What neural networks memorize and why: Discovering the long tail via influence estimation." Advances in Neural Information Processing Systems 33 (2020): 2881-2891.
[4] - Zhang, Chiyuan, et al. "Understanding deep learning (still) requires rethinking generalization." Communications of the ACM 64.3 (2021): 107-115.
[5] - Carlini, Nicholas, et al. "The privacy onion effect: Memorization is relative." Advances in Neural Information Processing Systems 35 (2022): 13263-13276.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper considers the problem of model inversion attacks on machine learning models, which aim to infer/ reconstruct private training samples for a particular class/ identity. The paper proposes a novel Data-Centric Defense (DCD) against model inversion attacks that uses a combination of techniques (i) (manually) injecting surrogate samples, (ii) mislabeling target samples, and (iii) augmenting data using Gaussian noise. These methods alter the loss landscape perceived by attackers, forcing them to recover samples from the injected surrogate data instead of protected samples.

### Strengths
1) This paper is written well and it is easy to follow.
2) To my knowledge, this is the first work to explore data centric defense in the context of model inversion attacks, although similar ideas have been applied for membership inference attacks [A].

[A] Heo, G., & Whang, S. E. (2023). Redactor: A Data-Centric and Individualized Defense against Inference Attacks. Proceedings of the AAAI Conference on Artificial Intelligence, 37(12), 14874-14882. https://doi.org/10.1609/aaai.v37i12.26737

### Weaknesses
1) Serious security Implications due to Surrogate Sample Injection. The proposed method of using surrogate samples raises serious security concerns. Specifically, the process of (manually) identifying surrogate classes, relabeling these samples as target identities, and incorporating them into model training could lead to a serious security vulnerability. The risk of the model classifying surrogate samples as the target identity needs to be thoroughly investigated. This issue could lead to unauthorized access/ bypassing the classifier if public surrogate data is used. This paper should consider analysis on potential misclassification rate for surrogate identities, which is a critical oversight. Techniques such as mislabeling and curvature-controlled injection could further aggravate this problem, thereby compromising the model’s security and utility. This trade-off between model security and resilience to model inversion attacks needs to be thoroughly examined and discussed.

2) Non-Standard Experiment Setups, especially lower number of identities attacked compared to state-of-the-art white box attacks [B, C]. Why do authors significantly reduce the number of attacked identities to 5, 8 and 10 for GMI, PPA and MIRROR respectively? The standard MI setup on CelebA uses 300 identities for Model Inversion Attacks. 

3) It is unlikely that the proposed DCD scheme would scale to a large number of identities similar to contemporary Model inversion attacks. I believe that if the authors extend their setup to attack all 1000 identities in CelebA, the model performance would severely suffer due to surrogate injection and mislabelling target identities.

4) Additional evaluation metrics for model inversion attacks required. Currently the paper only considers Attack Accuracy, but it is important to include additional metrics such as KNN Distance [B, C]  and user studies [MIRROR] to understand the efficacy of the proposed method.

5) Error bars/ Standard deviation for experiments are missing, especially in Table 1 (although the paper claims that Table 1 experiments were repeated 3 times). 

6) Missing results for LOMMA model inversion attack [ B ].

Overall I enjoyed reading this paper. But in my opinion, the weaknesses of this paper significantly outweigh the strengths. But I’m willing to change my opinion based on the rebuttal. 

==

[A] Heo, G., & Whang, S. E. (2023). Redactor: A Data-Centric and Individualized Defense against Inference Attacks. Proceedings of the AAAI Conference on Artificial Intelligence, 37(12), 14874-14882. https://doi.org/10.1609/aaai.v37i12.26737

[B] Nguyen, Ngoc-Bao, et al. "Re-thinking Model Inversion Attacks Against Deep Neural Networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[C] Yuan, Xiaojian, et al. "Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network." AAAI 2023 (2023).

[MIRROR] An, Shengwei et al. MIRROR: Model Inversion for Deep Learning Network with High Fidelity. Proceedings of the 29th Network and Distributed System Security Symposium.

### Questions
Please see Weaknesses section above for a list of all questions.

Remark : I managed to study only the main text. Unfortunately, I only skimmed through the supplementary due to its length / my time-constraints.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an approach called data-centric defense to prevent deep neural networks against model inversion (MI) attacks. The main idea is to define a surrogate class for each class in the private dataset, and then guide the optimization in MI to recover the surrogate class instead of the true class. 
For this purpose they have used three major components: i) injecting the surrogate samples for each class of the private dataset (labeled as the true class), ii) reducing the loss for surrogate samples by deliberately mislabeling some samples from the true class (so the loss on surrogate samples become slightly less than the loss for true samples), iii) shaping the loss landscape by employing the Gaussian augmentations through adding Gaussian noise.
The experimental analysis uses several studies, with different MI attacks and comparisons to different MI defense approaches.

### Strengths
The paper is written well, and it is easy to get the main ideas. Focusing on the data used for training instead of model architecture for preventing MI attacks also seems to be orthogonal to the literature.

### Weaknesses
I have some major concerns regarding the scalability of the idea to protect all classes, the validity of the setup used for experiments, the security issues introduced by this approach, and the failure to discuss the related works. Please see my detailed comments below:

1. Even though this paper manipulates the training data, the term Data-Centric might be misleading for a lot of researchers in the filed. The data-centric machine learning aims to engineer the data to improve the robustness and the generalization performance of the machine learning models for a particular task. However, this work seems to be more like a data pollution approach, where we pollute the data by some surrogate classes/samples to prevent the attackers from revealing the information about target classes/ identities. Note that this pollution is expected to affect the model performance (will discuss this later).

2. The experimental setup in this paper is unconventional and non-standard, and the authors are not following the standard setup used in the literature:
- The number of target classes used for GMI and PPA attacks are 5 and 10, respectively. However, in the literature, usually, 300 target classes are used for GMI/ KEDMI, and 1000 target classes are used for PPA. This gives the impression that the proposed method is not scalable to defend a larger number of classes against MI attacks, as by increasing the data pollution percentage (injecting more surrogate classes), the performance of the target model on the primary task (e.g., face recognition) can degrade severely. Note that all previous defense methods discuss protecting a large number of classes from MI attacks.
- The standard deviation information is not provided in Table 1 which is a critical component to observe the behavior of the proposed method and gauge the reliability of the reported numbers.
- The datasets used in experiments do not make sense to me. I cannot understand the connection between the Traffic Sign Recognition (GTSRB) dataset and the privacy thread in MI attacks.

**All these problems in experimental setups and analysis, give the impression that the proposed method is not effective for protecting private data against MI attacks in a real-world scenario.**


3. The proposed data pollution (data-centric) approach **creates another major security issue** while aiming to defend against MI attacks. Re-labeling the samples of the surrogate class into the related target class trains the target classifier in a way to label the samples of the surrogate class as the target class during test/ utilization time. Considering face recognition as an example, now the surrogate person can access the system as the related target class and its activity is left completely uncontrolled. As an example, figure 3 shows how different the surrogate person and related true identity could be. Therefore, there is a major security issue with the models trained with this approach, and these models can not be utilized safely.


4. There are some related works in the literature [a,b] (e.g., membership inference attack) that use similar ideas, but the proposed method fails to discuss these works and highlight the differences. This may undermine the significance of their contribution in this work. References:

- [a] "When does data augmentation help with membership inference attacks?" ICML 2021.

- [b] "Truth serum: Poisoning machine learning models to reveal their secrets" ACM SIGSAC 2022.

### Questions
Please refer to the weaknesses.

### Soundness
1 poor

### Presentation
3 good

### Contribution
2 fair
