# Federated Active Learning via Class-adaptive Local–Global Balancing

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Active learning has emerged as a pivotal approach for addressing data scarcity and annotation cost constraints in machine learning systems. However, its implementation in federated learning settings introduces unique challenges, particularly concerning data heterogeneity across clients. Our comprehensive analysis of existing centralized and decentralized methodologies reveals that state-of-the-art federated active learning techniques do not always outperform simpler baselines where centralized techniques are applied independently to clients. We identify a critical trade-off in performance: decentralized approaches excel when inter-client data heterogeneity is minimal, while centralized methods demonstrate superior performance under high heterogeneity conditions. Moreover, we observe a class-dependent variance phenomenon where the efficacy of each approach strongly correlates with the distribution variance of class samples across federated clients, highlighting critical bounds that limit existing methods. To address these limitations, we propose Adaptive Hybrid Federated Active Learning (AHFAL), a novel framework that dynamically integrates centralized and decentralized paradigms based on class-specific distribution characteristics. AHFAL combines enhanced entropy-based sampling with heterogeneity mitigation strategies, adaptively selecting the optimal paradigm per class based on cross-client variance metrics. Experiments across diverse datasets demonstrate that AHFAL outperforms state-of-the-art methods by prioritizing heterogeneity management over traditional uncertainty
sampling, particularly in low-resource and high heterogeneity scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Adaptive Hybrid Federated Active Learning, abbreviated as AHFAL, a method that adaptively chooses between centralized and decentralized active learning depending on the class specific distribution of data across clients. The central idea is that the performance of federated active learning depends on how unevenly classes are distributed among clients. The authors show that centralized querying is more effective when heterogeneity is high, while decentralized querying performs better when data are more uniformly distributed. AHFAL measures class wise variance to decide which strategy to use per class. The framework is designed to integrate naturally with standard federated learning pipelines and includes privacy considerations through local differential privacy and secure aggregation.

### Strengths
The work identifies a fundamental factor in federated active learning, namely that class wise data heterogeneity determines which strategy performs best. This insight is novel and well supported by both analysis and experiments. 

The theoretical part connects bias and variance of entropy estimators to class variance, which gives an intuitive explanation of when local or global models are preferable. 

The proposed method is conceptually simple, yet general enough to apply to different datasets and model architectures. The paper also considers privacy constraints and proposes reasonable mechanisms to preserve data confidentiality. 

It is clearly written, logically structured, and the figures help convey the main findings.

### Weaknesses
The evaluation of the paper does not fully follow best practices in active learning research. As highlighted by Lüth, Bungert, Klein, and Jaeger (2023) [1], several recurring pitfalls appear in the design and evaluation of active learning methods, some of which are present here.
* The evaluation lacks diversity in data distribution settings. The paper does not include experiments on imbalanced datasets, although such conditions are central to realistic federated learning scenarios.
* The evaluation settings are narrow. The paper fixes the query size at five percent and reports results only after seven active learning rounds for CIFAR10 and CIFAR100, and eight rounds for the other datasets. A broader evaluation using different labeling budgets and query schedules would provide a clearer picture of method behavior. In addition, other dataset partitioning schemes could be applied, such as a variant of McMahan et al. (2017) [2], where the level of heterogeneity is controlled by the shard size and the number of shards per client.
* The paper overlooks the role of classifier configuration. There is no distinction between development datasets, used for hyperparameter tuning, and roll out datasets, used for final evaluation. The threshold parameter Tau should be selected on some datasets and tested on others to verify generalization.
* The experiments do not consider alternative training paradigms. Only models trained from scratch are used, without testing pre-trained backbones or other initialization strategies that are standard in current active learning research.

The paper also lacks quantitative discussion of computational and communication overhead introduced by sharing class histograms and computing hybrid uncertainty scores.

Table 2 is hard to read.

[1] Lüth, Bungert, Klein, and Jaeger (2023) propose a systematic evaluation framework for active learning and analyze common pitfalls in AL literature.

[2] McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). Communication-efficient learning of deep networks from decentralized data (AISTATS, JMLR W&CP Vol. 54). arXiv preprint arXiv:1602.05629. Retrieved from https://arxiv.org/abs/1602.05629

### Questions
How would AHFAL perform under imbalanced data distributions, for example when some classes are heavily underrepresented across clients?

Why was a fixed query size of five percent chosen, and how does the method behave under different query budgets or adaptive labeling schedules?

How does AHFAL perform using more than one dataset partitioning scheme, such as variants of the shard based partitioning proposed by McMahan et al. (2017)?

What is the performance with the same hyperparameters on 1-2 new dataset on which the hyperparameters were not tuned?

Instead of using entropy for sampling in Equation (3), have you considered alternative sampling strategies such as margin sampling [3]?

How does AHFAL perform with pre trained backbones, and if not, how might pre training affect the observed trade off between centralized and decentralized sampling?

How many datapoints do you have for Figure 3? The line looks straight, is it just 4 datapoints?

[3] Bahri, Dara; Jiang, Heinrich; Schuster, Tal; Rostamizadeh, Afshin (2022). Is margin all you need? An extensive empirical study of active learning on tabular data. arXiv preprint arXiv:2210.03822.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper focuses on active learning in federated settings with non-iid data distribution. The authors conducted a systematic analysis and reveals three important findings. Based on these, the authors propose a novel federated active learning method and valiate the proposed method on 4 datasets.

### Strengths
1. Generally, this paper is easy to follow. 

2. The authors focuses on an important problem in federated learning, heterogeneous settings.

### Weaknesses
1. While Section 5 offers some theoretical insights into entropy estimation under heterogeneity, the analysis feels a bit surface-level and doesn’t clearly explain the reasoning behind AHFAL’s specific design choices. The link between the MSE analysis and the practical threshold $\tau$ isn’t very clear, which makes the theory section feel somewhat detached from the main algorithm.

2. The authors are supposed to compare the proposed method with more sota methods in the field of federated active learning, such as [1]. In addition, it misses out on more recent adaptive active learning and meta-learning–based selection strategies that might handle heterogeneity in different ways.

[1] Yingpeng Tang, Chao Ren, Xiaoli Tang, Sheng-Jun Huang, Lizhen Cui & Han Yu, "Efficient Heterogeneity-Aware Federated Active Data Selection," in Proceedings of the 42nd International Conference on Machine Learning (ICML'25), 2025.

3. The method involves computing and sending class distributions, generating pseudo-labels for all unlabeled samples, and possibly querying both local and global models. The paper calls the overhead “lightweight,” but there’s no actual measurement to back that up. It’s also unclear how well this would scale to thousands of clients or really large unlabeled datasets.

### Questions
1. Can you provide more rigorous justification for the fixed $\lambda = 1/2$ choice in Eq. (3)?

2. You mention plans to extend the approach to other domains. What do you see as the main challenges in adapting AHFAL to structured output tasks like object detection or segmentation, where the class boundaries aren’t as clear-cut?

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
This paper proposes a new algorithm, Adaptive Hybrid Federated Active Learning (AHFAL), to dynamically integrate centralized and decentralized paradigms for FAL. Based on their three key findings, they develop a new framework by grouping the clients into two disjoint sets and applying different scoring strategy. AHFAL showed good performance across diverse tasks.

### Strengths
- Their method is built on top of reasonable and good findings.
- The performance improvement over various baselines seems also quite good.
- The overall paper is well-written and has good structure.

### Weaknesses
- I'm quite confused with the definition of "centralized" and "decentralized" algorithms. There might be better words for them I believe.
- Could you elaborate Figure 4 experiments more? I have no idea what this figure means exactly and how the finding 3 comes out.
- I'm personally fine with communicating the client distribution to the server (privacy issue may be little), but it would be good to note that which methods use client distribution like this. Afaik, some decentralized strategy like LoGo does not share client distribution using the server.
- Do you have any other results to show compatibility with other FL algorithms than FedAvg?

### Questions
See above Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of Federated Active Learning under data heterogeneity. The authors observe that: the optimal active learning strategy depends on the degree of data heterogeneity. Specifically, decentralized methods excel in low-heterogeneity settings, while simple centralized methods surprisingly outperform them when heterogeneity is high. The paper proposes AHFAL: classes that are "high-variance" (i.e., concentrated on only a few clients) are better served by local-only sampling, while "low-variance" classes (i.e., spread evenly across clients) benefit from a hybrid local-global strategy. The authors provide empirical analysis to motivate their approach and show that AHFAL outperforms several centralized and decentralized baselines, particularly in high-heterogeneity regimes.

### Strengths
1. The central finding that the optimal FAL strategy is not static but should be adaptive at the class level based on distribution variance, is novel and intuitive. This moves beyond the typical FAL approach of finding a single "best" way to combine local and global information.

2. The proposed method is practical. The "cost" is sharing local class histograms, which are low-dimensional vectors. The authors discuss the privacy implications and potential mitigations.

3. The paper compares against a good set of baselines, including both centralized (Entropy, BADGE, Core-Set) and SOTA decentralized (LoGo, KAFAL, FEAL) methods. The codebase is also provided for reproducibility.

### Weaknesses
1. My primary concern is that the algorithmic novelty may not meet the threshold for ICLR. The main algorithmic step appears to be a class-wise recombination of existing paradigms rather than a fundamentally new mechanism. Specifically, the method adopts existing strategies on a per-class basis. Furthermore, the concept of exploiting both local and global models to mitigate heterogeneity has been explored in prior work [1, 2].

2. The core analysis section motivating the method runs only on CIFAR-10. This limited empirical grounding weakens the causal link between the initial findings and the subsequent generalization of the proposed framework.

3. The experimental validation is not sufficiently robust. While CIFAR-10 is analyzed thoroughly, other datasets (CIFAR-100, SVHN, MNIST) are introduced sporadically in figures and tables without clear explanation or systematic analysis. This lack of consistent evaluation across all benchmarks diminishes the robustness of the paper's empirical claims.

[1] Cao, Yu-Tong, et al. "Knowledge-aware federated active learning with non-iid data." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Deng, Zhipeng, et al. "Fedal: An federated active learning framework for efficient labeling in skin lesion analysis." 2022 IEEE International Conference on Systems, Man, and Cybernetics (SMC). IEEE, 2022.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2
