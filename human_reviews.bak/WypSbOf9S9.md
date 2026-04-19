# MOREL: Enhancing Adversarial Robustness through Multi-Objective Representation Learning

- Decision: Reject
- Scores: 5, 6, 6, 5

## Abstract
Extensive research has shown that deep neural networks (DNNs) are vulnerable to slight adversarial perturbations—small changes to the input data that appear insignificant but cause the model to produce drastically different outputs. In addition to augmenting training data with adversarial examples generated from a specific attack method, most of the current defense strategies necessitate modifying the original model architecture components to improve robustness or performing test-time data purification to handle adversarial attacks. In this work, we demonstrate that strong feature representation learning during training can significantly enhance the original model's robustness. We propose MOREL, a multi-objective feature representation learning approach, encouraging classification models to produce similar features for inputs within the same class, despite perturbations. Our training method involves an embedding space where cosine similarity loss and multi-positive contrastive loss are used to align natural and adversarial features from the model encoder and ensure tight clustering. Concurrently, the classifier is motivated to achieve accurate predictions. Through extensive experiments, we demonstrate that our approach significantly enhances the robustness of DNNs against white-box and black-box adversarial attacks, outperforming other methods that similarly require no architectural changes or test-time data purification.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a multi-objective representation learning (MOREL) method for enhancing the adversarial robustness of deep neural networks. By focusing on robust feature learning, the proposed method improves both standard accuracy and robust accuracy compared to existing methods like TRADES and MART.

### Strengths
1. The proposed method improves adversarial robustness without modifying the model’s architecture or adopting additional modules during inference. It achieves this by introducing cosine similarity loss and multi-positive contrastive loss, enforcing natural and adversarial inputs to have similar representations. 
2. The effectiveness of methods is evaluated using various adversarial attacks (e.g., PGD-20, PGD-100, and C&W attack) in both white-box and black-box scenarios.

### Weaknesses
1. There is insufficient theoretical or empirical analysis on the effectiveness of the linear layer $L_{e}$ and the class-adaptive multi-head attention module $M_{e}$, making it unclear why they are necessary.
2. The paper is not clearly written overall. For example, line 244 states that the loss is based on $\ell_{2}$-normalized vectors, but the equation (7) defines the $\ell_{2}$-norm of vectors. Similarly, line 252 indicates that outputs of $L_{e}$ are considered in the cosine similarity loss, but equation (8) uses $t_i$ and $t_i^{’}$, which are defined as the outputs of $M_{e}$. Moreover, in section 4.3, adversarial robustness against black-box attacks is evaluated, but there is no explanation provided on how the black-box attacks were conducted.    
3. The methods used for performance comparisons are too limited to adequately demonstrate the robustness of the proposed method. A broader set of comparisons, especially with more recent methods, would provide a stronger basis for evaluating the effectiveness of the proposed approach.

### Questions
1. Can the authors provide theoretical or empirical analysis to validate the necessity of the linear layer $L_{e}$ and the class-adaptive multi-head attention module $M_{e}$. This would help strengthen the motivation and foundation of the proposed method. An ablation study could effectively demonstrate these points. 
2. The authors state that a key contribution of the proposed method is enhancing robustness by aligning natural and adversarial features in embedding space. Can the authors provide experimental results to empirically validate this alignment.
3. Can the authors provide more details on the implementation?
  - How are the reference point $a$, preference vector $k$, and augmentation coefficient $\gamma$ set in Conic Scalarization?
  - The proposed method uses a batch size of 8 during training, while contrastive learning typically benefits from larger batch sizes. Can the authors explain why such a small batch size was chosen and how it might affect the effectiveness of contrastive learning?
4. Can the authors expand their experimental comparisons to include more recent adversarial training methods?

### Soundness
2

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
4

### Summary
This paper introduces MOREL, a multi-objective learning algorithm for training deep neural networks (DNNs) to enhance robustness against both white-box and black-box adversarial attacks. MOREL promotes alignment between natural and adversarial features through the use of cosine similarity and contrastive losses during training, fostering the development of robust feature representations.

### Strengths
1. The writing is clear and easy to understand.
2. The proposed algorithm is novel, utilizing a multi-objective optimization framework to enhance adversarial robustness while preserving high classification accuracy by aligning natural and adversarial features within a shared embedding space during training.
3. The embedding space used in training is subsequently discarded, allowing the model to retain its original structure and computational efficiency during inference.

### Weaknesses
1. Equation (7) requires additional clarification; specifically, the symbol ⨁ is not defined. Furthermore, it is unclear how  $t_i$ in equation (7) differs from  $t_i$ in equation (8).
2. The baselines selected for comparison are relatively outdated. It would strengthen the results to include more state-of-the-art (SOTA) baselines.
3. Performance improvements under adversarial attacks appear to be incremental for the best models
4. Further evaluation on additional datasets, such as ImageNet, would be beneficial.
5. Only one model is used for both training and evaluation. Including more advanced classifiers would improve the performance evaluation.

### Questions
1. In line 174, does $n_y$ refer to the number of features or the number of samples present in class  𝑦?
2. Are k, a, \gamma, and \alpha in line 305-306 parameters or hyperparameters? Can you provide more details behind the choice of these?
3. For MOREL, why does the final model outperform the designated “best” model, even in terms of clean accuracy?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new defense method against adversarial examples. Specifically, the authors point out two objectives of adversarial robustness, i.e., adversarial robustness and high accuracy, and try to achieve better results in both objectives via multi-objective optimization on both prediction accuracy and the similarity (between natural and adversarial samples) in the embedding space. The authors present the design of the embedding space and multi-objective optimization losses. The authors demonstrate defense performance against white-box and black-box attacks with experiments.

### Strengths
1. The paper proposes a novel approach to handle the robustness-accuracy tradeoff in adversarial training.
2. The experimental results demonstrate reasonably good performance of the proposed method.

### Weaknesses
The experiments only demonstrate the defense performance. The authors should have asked more questions regarding the proposed methods.
1. Two datasets were used for the experiments: CIFAR-10 and CIFAR-100. Considering that CIFAR-100 is the same dataset with more detailed labels, the experiments used only one dataset variant.
2. The authors fixed the attack methods (PGD-10) for training. A natural question is, “Would a more powerful attack (PGD-20) provide a better result?”
3. All the attacks used a $l_\infty$-norm constrained attack. The authors can check if the proposed method can generalize to other norm types, e.g., whether training with a $l_\infty$-norm constrained attack can defend against a $l_2$-norm constrained attack.
4. More ablation studies can be performed. The authors could have tested the effect of different hyperparameter settings. For example, performance with $\alpha=0$ would check if the multi-positive contrastive loss is essential to improve the proposed method.

### Questions
1. Is it possible to enlarge the adversarial batch by adding multiple adversarial examples from different attacks? If the adversarial batch from only one attack is enough, would you give me some justification?
2. Are there good justifications about why the improvements in CIFAR-100 are better than those in CIFAR-10?
3. Similarly, the proposed method improved MART results more than TRADES results. Can the authors explain why MART is more suitable for the proposed method?
4. Please consider adding the experiments listed in the Weaknesses, e.g., more datasets, more hyperparameter settings, etc.

### Soundness
3

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
4

### Summary
This paper proposes a multi-objective feature representation learning method called MOREL that is designed to improve deep neural network robustness against adversarial attacks. MOREL improves robustness by aligning natural and adversarial features in an embedding space using cosine similarity and contrastive losses. Experimental results assess the resulting model on white-box and black-box attacks showing that the network's robustness is improved while preserving classification accuracy.

### Strengths
1. Exploring the idea of using multiple objectives on improving robustness sounds worth exploring. 
2. The use of attention module to extract common features within the class seems novel.

### Weaknesses
1. Limited empirical results. There two broad limitations related to the empirical results. 1) experiments were only conducted on CIFAR10 and CIFAR100 datasets. While these experiments are a good start, they are not sufficient to establish the scalability of the approach. I suggest extending the experiments to Imagent or TinyImagenet datasets. 2) the perturbations used in the study are limited. Stronger and more recent perturbations should also be tested. E.g. AutoAttack which has become a strong common benchmark. 

Croce, F. and Hein, M., 2020, November. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In International conference on machine learning (pp. 2206-2216). PMLR.


2. Section 4.3 shows results for blackbox attacks but the attack names parallel those in section 4.2 for whitebox attacks. I'm not sure what is being shown. There are many black box attacks available that may be tested with. Some examples below: 

https://arxiv.org/abs/2102.00029

https://arxiv.org/abs/2406.04998

https://arxiv.org/abs/1912.00049

3. The proposed method consists of several components including architectural choices (the multi-head attention module) and loss functions. However, no ablation experiments are performed to clarify the importance of different components. For example, how important is the use of the attention module or each of the different losses, the effect of choice of hyperparameters like $\alpha$ or the hyperparameters in the CS method 

4. The proposed approach is conceptually similar to some of the prior work using ideas from domain adaptation to improve network robustness. But the background section doesn't cover these papers e.g. [1] and [2] below: 

[1] Chuanbiao Song, Kun He, Liwei Wang, and John E Hopcroft. Improving the generalization of adversarial training with domain adaptation. In ICLR, pages 1–14, 2019.

[2] Bashivan, P., Bayat, R., Ibrahim, A., Ahuja, K., Faramarzi, M., Laleh, T., Richards, B. and Rish, I., 2021. Adversarial feature desensitization. Advances in Neural Information Processing Systems, 34, pp.10665-10677.

5. Section 3.1 is named "adversarial attacks" but only the PGD attack is explained. Consider renaming the section. 

6. Line 121: "small perturbation of at most ϵ > 0" is not meaningful. 

7. Line 205: "This approach takes advantage of the global context understanding property of the attention mechanism". I'm not sure what the attention module is supposed to learn when it is applied on multiple examples of the same class. What is the presumed context given that the samples within a batch are always selected at random. The attention module needs better motivation and its importance needs to be supported by ablation experiments.

### Questions
1. Line 279 mentions the MART loss but it's unclear if that loss was also used in implementing MOREL. Was that used? If yes, in which cases. 
2. Equation 7: can you clarify how the normalization is performed? It's unclear whether each sample is normalized separately or somehow they are normalized by the norm within each group?

### Soundness
2

### Presentation
3

### Contribution
3
