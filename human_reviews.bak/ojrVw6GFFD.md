# Personalized Federated Learning of Probabilistic Models: A PAC-Bayesian Approach

- Decision: Reject
- Scores: 3, 8, 5, 6

## Abstract
Federated learning aims to infer a shared model from private and decentralized data stored locally by multiple clients.
Personalized federated learning (PFL) goes one step further by adapting the global model to each client's data, enhancing the model's fit for different clients.
A significant level of personalization is required for highly heterogeneous clients, but can be challenging to achieve especially when they have small datasets.
To address this problem, we propose a PFL algorithm named *PAC-PFL* for learning probabilistic models within a PAC-Bayesian framework that utilizes differential privacy to handle data-dependent priors.
Our algorithm collaboratively learns a shared hyper-posterior and regards each client's posterior inference as the personalization step.
By establishing and minimizing a generalization bound on the average true risk of clients, PAC-PFL effectively combats over-fitting.
Empirically, PAC-PFL achieves accurate and well-calibrated predictions as demonstrated through experiments on a highly heterogeneous dataset of photovoltaic panel power generation and the FEMNIST dataset.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies the problem of federated learning from clients with highly heterogeneous data distribution and small datasets. To achieve this problem, the paper propose a PFL algorithm named PAC-PFL for learning probabilistic models within a PAC-Bayesian framework. The PAC-PFL learns a shared hyper-posterior in a federated manner, which clients use to sample their priors for personalized posterior inference. Both theoretical analysis and empirical results are provided to show the effectiveness of the proposed method.

### Strengths
1. The paper addresses a lot of issues for PFL, such as small datasets, highly heterogeneous data distribution, uncertainty calibration and new clients, which are critical problems.
2. The paper extensively provides both theoretical analysis and empirical results to show the effectiveness of the proposed method.

### Weaknesses
1. The studied problems are not well-driven and illustrated. For example, the descriptopm of uncertainty calibration is insufficient which is unfriendly for new readers. The studied issues, such as small datasets, highly heterogeneous data distribution, uncertainty calibration and new clients, should be further organized and summarized.
2. The novelty of the proposed method PAC-PFL is limited, since it seems that the PAC-PFL only combine some techniques, such as PAC-Bayesian, FedAvg and SVGD.
3. The related works[1-3] for tackling uncertainty calibration and FL for small datasets are omitted. 
[1] Guo C, Pleiss G, Sun Y, et al. On calibration of modern neural networks. ICML, 2017: 1321-1330. 
[2] Minderer M, Djolonga J, Romijnders R, et al. Revisiting the calibration of modern neural networks. NeurIPS, 2021, 34: 15682-15694.
[3] Fan C, Huang J. Federated few-shot learning with adversarial learning. WiOpt, 2021: 1-8.

### Questions
1. For experiments results, almost all FL methods usually outperform the Pooled GP baseline, which is strange and should be further explained.
2. In experiments results of tables and figures, the reported baselines are generally different with each other, which is confusing.
3. The studied problems are not well-driven and illustrated. For example, the description of uncertainty calibration is insufficient which is unfriendly for new readers. The studied issues, such as small datasets, highly heterogeneous data distribution, uncertainty calibration and new clients, should be further organized and summarized.
4. The novelty of the proposed method PAC-PFL is limited, since it seems that the PAC-PFL only combine some techniques, such as PAC-Bayesian, FedAvg and SVGD.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper considers the personalized federated learning (PFL) through the lens of hierachical PAC-Bayes, similar to a previously studied PAC-Bayesian framework for meta-learning. a hyper-posterior is learned by a data-indpendent hyper-prior and the data from all clients, and a personalized posterior for each client is learned by a data-dependent prior sampled from the hyper-posterior and that client's data. To handle the data-dependence of the prior, an assumption on differential privacy is made and verified for optimal hyper-posterior, which is a Gibbs distribution. Based on this framework, a PAC-PFL algorithm is then proposed that updates the hyper-posterior is updaetd via SVGD.

### Strengths
Personalized federated learning is an important task, and as far as I can tell, the technical results are sound.
The experimental results show the proposed algorithm have better personalized performance.

### Weaknesses
There is no particular weakness of the paper. 
The algorithm presented in the paper has too little details, especially how is the hyper-prior/posterior formulated (Gaussian distribution over the parameters?), make it less readable without knowledge of Rothfuss et al.
Though in applications, it is intractable to acheive the optimal Gibbs hyper-posterior, and thus lead to some concerns of the theory. I guess it is still possible to claim differential privacy for finite number of SVGD udpates?

### Questions
Plese see weakness.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Personalized Federated Learning (PFL) tailors a global model to individual clients' data, especially useful for diverse clients. To overcome challenges in PFL with limited client data, PAC-PFL is introduced. PAC-PFL employs a PAC-Bayesian framework and differential privacy, collaboratively learning a shared hyper-posterior while preventing overfitting through a generalization bound. Empirical tests on heterogeneous datasets confirm that PAC-PFL delivers accurate and well-calibrated predictions.

### Strengths
1. PAC-PFL introduces a systematic, non-heuristic regularization of the hyper-posterior, allowing for the training of complex models without falling into overfitting.
2. This  approach accommodates the accumulation of fresh data over time.
3. It can be interpreted through Jaynes' principle of maximum entropy
4. Experiments confirm PAC-PFL's accuracy in heterogeneous and bimodal client scenarios, along with its ability for efficient transfer learning from small datasets.

### Weaknesses
1. As for baslines, only 1 and the latest of them are proposed in 2022, methods that were proposed in 2023 should also be considered.
2. One dataset seems not enough for demonstrate the scalability and generalization of the proposed framework.
3. The theoritical analysis is pretty solid. However, the experiments are not convincing and strong enough in contrast.

### Questions
1. More state-of-the-arts methods should be included in the experiment.
2. More datasets should be performed on to illustrate the generalization of proposed framework.
3. The percentage of experiment should be increased compared with the theoritical analysis.

### Soundness
2 fair

### Presentation
2 fair

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
This paper developed a PAC-Bayes framework for personalized federated learning. The PAC-PFL algorithm imposes a hyper-prior and a hyper-posterior on the server level. Based on a theoretical analysis (Theorem 4.1) similar to that of the client level (Theorem 3.1), the optimal hyper-posterior has a closed-form solution (Corollary 4.2.1). The final algorithm is based on several approximations (see Sec.5). Empirical studies on regression and classification problems show that PAC-PFL can outperform existing methods, especially when the sample size is small.

### Strengths
1. An algorithm developed from a theoretical perspective

2. Reasonable empirical results

3. Writing is mostly clear

### Weaknesses
1. The reason for using two samples per the client remains unclear. The algorithm requires two samples $S_i$ and $\tilde{S}_i$ as mentioned in the first paragraph of Sec.3. However, what they are used for specifically is not very clear. For example, in (8), shouldn’t the first $S_i$ be $S_i\cup\tilde{S}_i$ while the second one be $\tilde{S}_i$?

2. The computation complexity of the algorithm can be very high. For approximating the optimal hyper-posterior using a set of priors (see Sec.5), the communication overhead is increased from one to k, which can be unbearable for large models. It would be useful to see an ablation study on the choice of k.

3. Some experiment details require clarification

    3.1. we can see that PAC-PFL even outperforms the Pooled method in Tables 2 & 3, which is surprising as Pooled is essentially an oracle. Additional explanation would be helpful. Also, the Pooled method is missing for FMNIST.

    3.2. It is common to use Dirichlet partition (Marfoq et al., 2021, Wang et al., 2020) for other image datasets to simulate heterogeneous clients, so it would be more comparable to other baselines if the paper can include such an experiment.

Ref:
- Marfoq, O., Neglia, G., Bellet, A., Kameni, L. and Vidal, R., 2021. Federated multi-task learning under a mixture of distributions. Advances in Neural Information Processing Systems, 34, pp.15434-15447.
- Wang, H., Yurochkin, M., Sun, Y., Papailiopoulos, D. and Khazaeni, Y., 2019, September. Federated Learning with Matched Averaging. In *International Conference on Learning Representations*.

### Questions
Please clarify the questions mentioned in the Weaknesses section above.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
