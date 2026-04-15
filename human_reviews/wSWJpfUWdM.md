# FedLAP-DP: Federated Learning by Sharing Differentially Private Loss Approximations

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
This work proposes FedLAP-DP, a novel privacy-preserving approach for federated learning. Unlike previous linear point-wise gradient-sharing schemes, such as FedAvg, our formulation enables a type of global optimization by leveraging synthetic samples received from clients. These synthetic samples, serving as loss surrogates, approximate local loss landscapes by simulating the utility of real images within a local region. We additionally introduce an approach to measure effective approximation regions reflecting the quality of the approximation. Therefore, the server can recover an approximation of the global loss landscape and optimize the model globally. Moreover, motivated by the emerging privacy concerns, we demonstrate that our approach seamlessly works with record-level differential privacy (DP), granting theoretical privacy guarantees for every data record on the clients. Extensive results validate the efficacy of our formulation on various datasets with highly skewed distributions. Our method consistently improves over the baselines, especially considering highly skewed distributions and noisy gradients due to DP. The source code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new federated learning training algorithm. The key idea of the algorithms is to share synthetic datasets instead of sharing the model weights. The authors argue that the method is superior to traditional model-averaging because the server can recover an approximation of the global loss landscape. The authors also demonstrate that the proposed method can be adapted to satisfy differential privacy (DP) by replacing the clean gradients with clipped noisy gradients. Some experiments show that the new method can outperform the traditional one in both private and non-private settings and may have better communication cost tradeoffs.

### Strengths
* The authors identify the problem of the existing weight-averaging FL training methods.
* The proposed method can overcome the limitations of the existing algorithms.
* The proposed algorithm is shown to have better empirical performances than the existing ones.

### Weaknesses
1. Some operations in the proposed algorithm may need better motivation or explanation. Please refer to Questions.
2. The privacy description part of the algorithm can be improved. The description in Section 4.4 is not clear enough and self-contained for a main contribution in the main text. Some descriptions may need to be more precise (e.g., "sanitize" should be explicitly referred to the clipping operation). Since the privacy composition part mainly relies on existing tools, it may be better to provide a brief conclusion about the composition results. Also, the notations in Appendix (T?) are not consistent with Algorithm 1 (R_b?), which requires extra conjecture to parse the result.
3. Some hyper-parameters may need to show how to tune ($R_b$ and $R_l$), because they are so different for private and non-private settings.
4. The experiment setting may need to be more convincing. Please refer to Questions.

### Questions
1. It is mentioned that the synthetic labels are initialized to be a fixed, balanced set, but the experiments have heterogeneous data. This sounds controversial and may deserve more explanation. When a class does not appear in a local dataset, what should we expect the synthetic data of that class to look like? How do those affect global training compared with relatively iid data?
2. How do you decide the synthetic dataset size of each client? Besides the factors mentioned (local dataset size and bandwidth), it seems the data complexity and local data distribution should also be considered when deciding the synthetic dataset size.
3. Section 5.1 mentions that the learning rate is 100. Is it a typo? Otherwise, why we need such a large learning rate may need to be explained.
4. What do the "Fixed, Max, Median, and Min" radius selection mean? Do they matter for whether private or non-private settings? The description needs to be more specific and consistent (i.e., it says r=1.5 or 10 in Section 5.1, but a different wording in Section 5.4).
5. In the experiment setting of comparing the communication cost and epochs, there may be more reasonable settings. For example, when comparing the communication cost, shouldn't we compare the best end-to-end performance with fixed communication costs (e.g., 500MB) by varying the communication rounds and with the best hyper-parameters? 
6. Another interesting aspect not explored in the experiments FedAvg v.s. FedLAP with different sizes of models. Does a model with more parameters benefit from or loss advantage with the proposed method?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents an approach called FedLAP-DP for federated learning. In contrast to previous methods that share point-wise local gradients for global model update, the new approach proposes to perform global optimization at the server by leveraing synthetic samples received from clients. Experiments are conducted to show the performance of the proposed method as well as some baseline methods.

### Strengths
It is interesting to borrow the idea of Dataset Distillation into federated model training to tackel the non-iid data and privacy issues.

The paper is overall clearly structured and easy to follow.

### Weaknesses
Though the idea is interesting, it seems to lack formal guarantees on the approximation achieved for each local synthetic dataset and how approximation level affects the learning performance in theory.

As for the DP side, the explicit trade-off between the privacy loss and the learning utility is also unclear for the proposed method.

In experiment, the performance on different settings with different heterogeneity can be further explored. And since you consider the record-level DP, the setting of privacy budget $\epsilon$ is relatively large even for high privacy regime where $\epsilon$ is set to be 2.79.

### Questions
1. Can you provide some formal theoretical guarantees for the proposed algorithm, e.g., approximation of the synthetic data learnt compared with the optimal one, and the learning performance, privacy-utility tradeoffs, so on.

2. Is there any possiblity to extend the proposed method to client-level DP, which is very important in cross-device scenarios such as collaboration between massive IoT devices. If not, then what is the barrier for doing this?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies federated learning with data condensation. In order to handle data heterogeneity, instead of sending local updates as in FedAvg, this paper proposes a method of sending synthetic data samples. Experiments show that the proposed method can improve model performance as well as reduce communication. To protect privacy, this paper use Gaussian mechanism to enforce record-level differential privacy.

### Strengths
This paper is well written and easy to follow. Extensive experiments are conducted and the results look promising.

### Weaknesses
My biggest concern is that the novelty may be limited. Data condensation + FL has been well studied, e.g. [1,2]. In particular, using condensed datasets in FL has been discussed in [1]. The novelty of this work looks limited. It would be highly appreciated if the authors can show improvements over existing works in terms of communication, performance, etc. 


[1] Liu, Ping, Xin Yu, and Joey Tianyi Zhou. "Meta knowledge condensation for federated learning." arXiv preprint arXiv:2209.14851 (2022).
[2] Behera, Monik Raj, et al. "Fedsyn: Synthetic data generation using federated learning." arXiv preprint arXiv:2203.05931 (2022).

### Questions
For DP, the paper says "we sanitize the gradients derived from real data with the Gaussian mechanism"。 Do you clip the gradients or do something to bound the sensitivity? How will this affect the model utility?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel federated learning (FL) algorithm, namely FedLAP-DP, to address the drawback of bias in global optimization in traditional FL. The approach involves generating synthetic data resembling real data on the client side and substituting local gradients with these synthetic samples during transmission to the central server to approximate the global loss landscape. The central server then iterates using these synthetic samples, thus mitigating the bias in global optimization. Additionally, differential privacy (DP) is employed to protect the privacy of synthetic data of clients. This idea is innovative, and the writing quality is also acceptable.

### Strengths
1. The approach proposed in this paper involves generating synthetic data resembling real data on the client side and then leverages the synthetic data to update the local model,  thereby reducing the negative impact of DP on the training.
2. To control the communication cost, the size of the synthetic dataset is much smaller than the real client dataset. Thus, in the local training, the proposed approach leverages a small dataset to update a local model with a satisfactory performance.
3. In addition to applying this approach to the DP setting, it can also address the issue of data heterogeneity in FL.
4. Extensive experimental results demonstrate that FedLAP-DP outperforms the traditional approaches with faster convergence under the different DP settings.

### Weaknesses
1. I have a question regarding the generation of synthetic data and the iterations performed by the central server. In the case of non-iid data distribution, are the synthetic samples submitted by the clients to the central server consistent with the original data distribution? If so, referring to Algorithm 1, would there still be bias in the iterations conducted by the central server? Could you please provide a detailed explanation of the algorithm design on how the synthetic data is generated?
2. Please further explain the parameters set that appeared in Sec.5 EXPERIMENT Part 5.1.
3. It would be better to remark on each curve in the experimental figures.  Some notations are not clear.
4. The baselines used in this paper are too simple. Some advanced DP-FL baselines should be included in this paper, such as [1], [2], [3] and etc.

The mentioned references are as follows: 
[1] Skellam mixture mechanism: a novel approach to federated learning with differential privacy
[2] Dpis: An enhanced mechanism for differentially private SGD with importance sampling
[3] PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

### Questions
The following comments should be addressed.
1. I have a question regarding the generation of synthetic data and the iterations performed by the central server. In the case of non-iid data distribution, are the synthetic samples submitted by the clients to the central server consistent with the original data distribution? If so, referring to Algorithm 1, would there still be bias in the iterations conducted by the central server? Could you please provide a detailed explanation of the algorithm design on how the synthetic data is generated?
2. Some notations are not clear. For example, I cannot see the effect of indexes j and l in Algorithm 1.
3. Please further explain the parameters set that appeared in Sec.5 EXPERIMENT Part 5.1.
4. It would be better to remark on each curve in the experimental figures. 
5. Some advanced DP-FL baselines should be included in this paper, such as [1], [2], [3] and etc. 

The mentioned references are as follows: 
[1] Skellam mixture mechanism: a novel approach to federated learning with differential privacy
[2] Dpis: An enhanced mechanism for differentially private SGD with importance sampling
[3] PrivateFL: Accurate, Differentially Private Federated Learning via Personalized Data Transformation

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
