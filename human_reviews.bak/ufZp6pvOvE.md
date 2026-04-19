# AROID: Improving Adversarial Robustness through Online Instance-wise Data Augmentation

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3, 6

## Abstract
Deep neural networks are vulnerable to adversarial examples. Adversarial training (AT) is an effective defense against adversarial examples. However, AT is prone to overfitting which degrades robustness substantially. Recently, data augmentation (DA) was shown to be effective in mitigating robust overfitting if appropriately designed and optimized for AT. This work proposes a new method to automatically learn online, instance-wise, DA policies to improve robust generalization for AT. This is the first automated DA method specific for robustness. A novel policy learning objective, consisting of Vulnerability, Affinity and Diversity, is proposed and shown to be sufficiently effective and efficient to be practical for automatic DA generation during AT. Importantly, our method dramatically reduces the cost of policy search from the 5000 hours of AutoAugment and the 412 hours of IDBH to 9 hours, making automated DA more practical to use for adversarial robustness.
This allows our method to efficiently explore a large search space for a more effective DA policy and evolve the policy as training progresses. Empirically, our method is shown to outperform all competitive DA methods across various model architectures (CNNs and ViTs) and datasets (CIFAR10/100, Imagenette, ImageNet, SVHN). Our DA policy reinforced vanilla AT to surpass several state-of-the-art AT methods regarding both accuracy and robustness. It can also be combined with those advanced AT methods to further boost robustness.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a novel adversarial training by introducing a policy network that selects, for each instance, the data augmentation procedure that could, potentially, maximize the robustness of the model. The experimental results show the algorithm is able to achieve good results compared with the current state-of-the art.

### Strengths
**originality**: The main contribution of this paper is the addition of a policy network that selects the best data augmentation procedure to maximize the network's robustness. The idea is interesting, and seems to be novel.

**quality**:  The experimental section shows promising results against current state-of-the-art techniques.

**clarity**: The idea is sound, and the authors put high emphasis in describing the motivation of each decision made, specially regarding to the loss terms.

**significance**: works that target network's robustness with lossless accuracy are of special interest for the community.

### Weaknesses
**originality**: Besides the policy network, the rest of the paper is a combination of well-known techniques. Although, I must say they are combined in a sound and clever way.

**quality**: The experimental section results are confusing. If I understood it correctly, the baseline results consider the network training without any adversarial techniques. Thus, it is known that the accuracy of either WRN and ViT algorithms are higher than those provided by the authors (over 95% in CIFAR10 and 80% in CIFAR100 for the WRN network, for instance). I also think Table 2 should include the training  time of every state-of-the-art algorithm.

**clarity**: Some hyper-parameter default values appear to be missing ($l$ and $u$ bounds in the Diversity penalty. I suggest the authors to include all hyper-parameter values in the main paper.

**significance**: Due to the experimental section problem, it is not possible to conclude that this algorithm can provide a robustness increase with lossless accuracy performance.

### Questions
- Why the baseline methods have lower accuracy scores than the ones provided by the authors of the network models?
- The diversity penalty (Eq. 7) seems to be quite overcomplicated. If the authors aim to obtain values close to an uniform distribution, should the authors use a more usual entropy loss? $$\mathcal{L}^h_{div}(\mathbf{x}) = \sum_i p^h_i(\mathbf{x}; \mathbf{\theta}) \log(p^h_i(\mathbf{x}; \mathbf{\theta})) $$
- Can the authors provide an ablation study regarding to the different penalties applied to the policy network?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes an automated data augmentation method, AROID, to address the generalization issue in adversarial training. Specifically, the authors argue that tailored image transformations should be applied for each training sample and at each stage of training. They introduce the periodic learning of the data augmentation policy model during adversarial training. Based on the results of this policy model for each image, combinations of image transformations are applied during adversarial training. The experiments demonstrate the efficiency and effectiveness of the proposed method compared to other automated data augmentation techniques.

### Strengths
1. This paper is overall clearly clarified and well organized.
2. The proposed method is experimentally demonstrated to be more efficient and effective when compared to other automated data augmentation techniques.

### Weaknesses
1. The proposed method involves many hyperparameters (u, l, beta, lambda, K, policy model hyperparams), and the discovered policy leads to compromised results in different training settings, thereby significantly increasing the cost of adversarial training.
2. It is expected that learning an instance-wise data augmentation policy model would be challenging as the data distribution expands, yet there is no analysis provided on this aspect.
3. Results for CIFAR100+WRN34-10 or ViT-B/16 are missing from Table 1.
4. Rebuffi et al. proposed the use of Cutmix in conjunction with SWA. Therefore, it is necessary to present the results incorporating SWA in Table 1.
5. It is necessary to present the results when used in conjunction with TRADES (Zhang et al., 2019).
6. AutoAugment is a method proposed without considering adversarial training. Therefore, using pre-trained AutoAugment as is would be unfair.
7. It is required to show the results when such approaches are applied alongside the use of more training data [1,2].

[1] Carmon, Yair, et al. "Unlabeled data improves adversarial robustness." NeurIPS 2019.  
[2] Gowal, Sven, et al. "Improving robustness using generated data." NeurIPS 2021.

### Questions
(Copy from Weaknesses)
1. Results for CIFAR100+WRN34-10 or ViT-B/16 are missing from Table 1.
2. Rebuffi et al. proposed the use of Cutmix in conjunction with SWA. Therefore, it is necessary to present the results incorporating SWA in Table 1.
3. It is necessary to present the results when used in conjunction with TRADES (Zhang et al., 2019).
4. It is required to show the results when the proposed method is applied alongside the use of more training data.

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
This paper presents AROID, an adaptive Data Augmentation (DA) method to improve adversarial robustness while avoiding overfitting during Adversarial Training (AT). A common limitation of AT is indeed that it suffers from overfitting and data augmentation is a typical way to overcome this. However, existing DA methods for adversarial robustness have two limitations: 1) they require computing an optimal DA policy, which is expensive as each trial requires an AT run; 2) the found policy is equally applied to all training examples. This paper suggests that there is an opportunity to optimize the policy at each data point, which could lead to lower computational cost and better effectiveness. Evaluation on CIFAR-10, CIFAR-100 and ImageNet reveals that AROID is competitive with other DA methods (and outperforms more of them) while being more efficient than previous similar methods (accuracy-wise) by 3 orders of magnitude.

### Strengths
The paper presents an interesting idea in optimizing DA policy at instance-level rather than globally for all samples.

AROID can be used in offline mode, in which the policy has been previously optimized on a subset of data or on a smaller model.

The paper proposes a thorough experimental protocol that also includes an ablation study, demonstrating the contribution of all components (including the three objectives) to the effectiveness of AROID.

The paper does not point to a replication package to reproduce the results.

### Weaknesses
The paper makes strong overclaims. For example, I disagree that "AROID is the first automated DA method specific to adversarial
robustness"; or that "AROID achieves state-of-the-art robustness for DA methods on the standard benchmarks". 

Indeed, standard evaluation benchmarks like RobustBench (https://robustbench.github.io/#leaderboard) include many robustness methods that rely on data augmentation and many of them outperforms AROID in terms of raw robust accuracy. I appreciate the fact that this may come from some controlled variables, like the used model architectures and parameters, but the lack of comparison (even qualitative) with these established leaderboards/benchmarks makes it difficult to assess the benefits of AROID compared to the available body of knowledge (beyond the few DA methods considered in the paper). More generally, the choice of baselines to compare to lacks proper justification.

Even considering only the baselines reported in the paper, the improvements in robust accuracy are limited to up to 3 percentage points compared to other DA methods (often less than this). Given the number of factors at play when determining empirical robustness, it is hard to establish AROID offers generalizable improvements.

The paper is also unclear in its efficiency evaluation. Indeed, Table 5 reports the time needed by AROID (in online or offline mode?) and other policy optimization methods during the policy search process only. Elsewhere in the paper, it is reported that AROID adds up 43% more time compared to standard AT whereas the "state-of-the-art" AT method named LAS-AT adds up to 52%. First, I would like to understand what is the practical cost of using AROID versus, e.g., IDBH; is this cost limited to policy search? I would think that in online mode, AROID has an additional cost that adds up at each instance (whereas IDBH computes the policy once and for all). Second, 43% more than standard AT is not a minor increase given the already huge cost of AT.

Overall, I have difficulties agreeing that AROID makes a significant step towards effective and efficient adversarial robustness methods, especially given the lack of justification for the chosen comparison baselines and the fact that there exists methods that improve robustness more significantly.

### Questions
- Can you elaborate on how AROID compares with the leading robustness methods reported in RobustBench?

- Table 5 reports the efficiency of AROID compared to similar methods for searching the search space only. Meanwhile, other numbers report that AROID requires 43% more time compared to standard AT while the other methods require 52% more than AT. Therefore, I wonder what is the total computational cost of using AROID versus the other DA methods if one was to using those in practice?

- In Table 5, have you used AROID in offline or online mode?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents an online adaptive data augmentation method called AROID for training a robust classification model. It is an extended method of the IDBH method, which can be viewed as a static method. The main idea is to use a different augmentation method for every input and this strategy evolves over training. The experiments show that the effectiveness of the proposed method.

### Strengths
The online adaptive idea is interesting. The efficiency of the method is much higher than previous data augmentation method including IDBH and AutoAugment. Experiments are extensive.

### Weaknesses
1. The description of the method is not clear. 

The overall pipeline Fig. 1 is complicated, but lacks interpretation. In fact, some notations are not defined at all such as f_{aft}, f_{tgt} and f_{plc}. I can guess the meaning of the last two, but cannot guess the first one. 

It is claimed that we need to perform bilevel optimization, alternate between target and policy models. I'm confused how the adversarial examples are obtained because to optimize eqn (9) we need go generate adversarial examples (see the function of rho). It seems that we have three optimization problems instead of two. Then how do we optimize them? 

Table 2 reports the results of the proposed AROID method combined with different AT methods. But it is not described how they are combined. 

2. The results on ImageNet are not convincing enough. 

It seems the proposed method works well on small datasets. But the results on the large dataset ImageNet are not convincing enough. In ref [a], it is reported that ConvNeXt-T with random init., basic augment. + strong clean pre-training + heavy augmentations achieved 46.5% robust accuracy (see Table 1 in [a]). But this work only achieved 40.40% robust accuracy. It seems that the proposed data augmentation method is not as effective as the method in [a]. Anyway, I think it is necessary to compare with the method in [a]. 

[a] Naman D Singh, Francesco Croce, Matthias Hein, Revisiting Adversarial Training for ImageNet: Architectures, Training and Generalization across Threat Models. NeurIPS 2023 (arXiv:2303.01870).

### Questions
About the ImageNet experiment, if the experimental settings in [a] are used, e.g., run experiments upon (random init., basic augment. + strong clean pre-training), can AROID show advantage over heavy augmentations proposed in [a]?

### Soundness
3 good

### Presentation
1 poor

### Contribution
3 good
