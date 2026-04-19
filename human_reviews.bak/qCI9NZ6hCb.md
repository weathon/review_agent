# Knowledge Is Not Wisdom: Weight Balancing Mechanism for Local and Global Training in Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 1, 5

## Abstract
Federated learning (FL) is a unique approach that typically leverages client-side computing resources and data on edge devices. Data heterogeneity is a primary challenge that makes federated learning complex, and many studies have been conducted to address this issue. In previous studies, solutions were primarily focused on the client side, such as adjusting the weights of the local model or using proxy data from the aggregation server. However, we identified a problem where the global model becomes biased due to averaging the client’s model, depending on the amount of the client’s data or the extent of data sharing. Therefore, we introduce local and aggregation balancers for federated learning (FedBal), which respectively mediate the local training by class distribution and the weight aggregation by specific clients. We employ a local balancer to mitigate biases in favor of specific classes and an aggregation balancer to regulate biases toward certain clients. Remarkably, through experiments applying various existing methods with an aggregation balancer, we found that reflecting the models of marginalized clients more than those of clients with abundant data and classes can improve the accuracy of the global model by 2\%–7\%. FedBal, which combines two Balancers, exhibited an average accuracy improvement of 3\%–4\% compared to all other methods. This study raises several questions for further work to deepen our understanding of the role of the aggregation framework in FL.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This papers aims to construct a better global model compared to that of prior works, by adopting (1) a local balancer and (2) an aggregation balancer. The local balancer reduces the client drift by balancing the logit for each class in the client. The aggregation balancer uses the cosine similarity between the global model before local update and the updated model of each client, to measure the importance score of each client. Experimental results how that the proposed scheme performs better than FedAvg, FedProx, FedNova, Scaffold in various federated learning scenarios.

### Strengths
1. The paper is in general easy to follow.

2. The proposed scheme performs better than various baselines.

### Weaknesses
1. First of all, I feel that the technical novelty of this paper is somewhat lacking. For the local balancer part, FedBal is actually simply adopting existing work from the long-tail / class-imbalanced learning literature. I'm also not very clear with the technical novelty of the aggregation balancer compared with existing works on server-side applied methods.

2. Moreover, the authors are comparing their scheme with only client-side applied methods, even though the authors are proposing both client-side and server-side methods. What is the advantage of FedBal compared with server-side applied methods? Can existing combinations of client-side and server-side methods perform better than FedBal? I believe the authors should address these questions.

3. The authors are considering two datasets, which I believe is not sufficient.

4. Finally, there are several places that makes me feel that the paper is not well-polished or self-contained.
- In section 5.1, the same paragraph is repeating twice.
- In section 5.1, the authors mention that they only focus on cross-silo FL, but also say that 4 out of 20 clients are participating in each round. Maybe the authors are considering cross-device FL?
- How is the outlier balancer $\beta$ determined in the scheme?
- In section 4.1, the authors do not provide details on how $z_j^b$ is actually differs from $z_j$.

### Questions
See the weakness above

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to address the data heterogeneity problem in federated learning. The problem is identified as a result of the biased global model. To mitigate these biases, local and aggregation balancers for FL (FedBal) are developed towards specific classes and specific clients, respectively. Experiments have shown the superiority of the proposed balancers when they are combined with existing methods.

### Strengths
1. It is good to see a related work summary from the client side and the server side, respectively. This brings about a novel viewpoint to existing FL methods.

2. The proposed aggregation balancer has improved the performance of various FL methods, which demonstrates its effectiveness for different optimization schemes. 

3. Experimental settings and training specifications are given in detail.

### Weaknesses
1. The core motivation and idea of the local balancer inherit from the method in (Li et al., 2022a). The authors have also mentioned that the loss in Eq. (4) is from (Li et al., 2022a). I think these have limited the novelty and contribution of the proposed method. There is a lack of an explanation of the unique contribution and novelty of the balancer in this paper and how it differs from that in (Li et al., 2022a).  

2. Although the proposed method addresses the data heterogeneity problem, it seems that only label-level shift heterogeneity is taken into consideration. As for the feature-level shift heterogeneity, e.g., domain shift, the proposed method seems unable to address it well. 

3. From my perspective, more knowledge always leads to a wiser decision, but more data does not lead to more knowledge, considering the data properties can vary a lot in different cases. I think the concepts of knowledge, data, and wisdom can be further clarified.

### Questions
None.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
1: strong reject

### Rating Number
1

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes a novel FL algorithm to mitigate the bias in model aggregation by assigning higher weights to the marginalized local models. Experimental results show the proposed method can improve the test performance of non-IID FL and combine with the existing methods.

### Strengths
The paper attempts to solve an interesting problem, aggregating local models in a more effective way, instead of naively averaging.

### Weaknesses
1. The paper isn't well written and well-organized. For example, there are two colonial paragraphs in the section 5.1, begining from 'Our experiments primarily used a 3-layered ConvNet...'.  And the overall training procedure is not illustrated in the paper and there is no a graph to show the workflow.

2. There is no theoretical guarantee in this paper. The proposed method FedBal is tottally empirical.  However, the method  can not outperform all other baselines in the experiments. For example, in table 5, FedBal shows better results than the existing methods in all cases 
except $\alpha = 0.1$. It is very strange that FedBal works with higher heterogeneity $\alpha = 0.05$ and lower  heterogeneity $\alpha = 0.5, 1$ but does not work with $\alpha = 0.1$.

3. The experiments are not comprehensive. Three benchmark datasets are sufficient for validating the effect of method. And there are some related prior studies not in the baselines. For example, [1][2], using bayesian theorem to aggregate the local models.

4. The number of clients is too small. only 4 clients paricipating in the training each round. The FL system in the experiments is too small.


Reference:

[1] Yurochkin M, Agarwal M, Ghosh S, et al. Bayesian nonparametric federated learning of neural networks[C]//International conference on machine learning. PMLR, 2019: 7252-7261.

[2] Wang H, Yurochkin M, Sun Y, et al. Federated learning with matched averaging[J]. arXiv preprint arXiv:2002.06440, 2020.

### Questions
1. Do all clients have the same number of data? The number of data decides the weight assigned to the local model $p_{i}$. What if a client has larger local dataset but with imbalanced data. Does this client have 'extensive knowledge' or marginalized model? 

2. How do you aggregate the models in the server? The paper just shows the objective function of FedBal in equation (9).

### Soundness
1 poor

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper aims to tackle data heterogeneity in Federated Learning (FL) by introducing Federated Learning Balancer (FedBal). This addresses the bias introduced when servers average a client's model based on data volume or sharing degree. FedBal incorporates two components: the local balancer, regulating preferences in local models, and the aggregation balancer, which calculates cosine similarity to adjust aggregation weights for the global model. It shows an enhancement of 2% to 7% in marginalized clients' models compared to those with abundant data. The experiments demonstrate FedBal can improve global model accuracy by 3% to 4% compared to alternative methods.

### Strengths
1. Originality: Empirical evidence indicates that focusing on marginalized clients during aggregation can prevent model overfitting, suggesting a new perspective. The method's simplicity aids reproducibility and further research.

2. Significance: Balancing mechanisms on both client and server sides show potential in enhancing model accuracy, proposing a universal solution to data heterogeneity in Federated Learning.

3. Clarity and Replicability: The paper's structure and method simplicity contribute significantly to its clarity and replicability.

### Weaknesses
1. Editorial Imperfections: Repetitive statements and editorial issues affect the paper's readability and require careful revision.

2. Limited Novelty: The novelty of the local balancer and aggregation equalizer is constrained. They draw heavily from prior research, limiting their originality.

3. Inconclusive Experiments: The experiment lacks comprehensive contemporary baseline comparisons, limiting the persuasiveness of its findings.

### Questions
1. Figure 1 mentions that in the case of a uniform segmentation of labels, labels that are shared more among customers (0, 1, 6, 7, etc.) usually show a higher accuracy rate. However, some labels, notably label 6, experience a decrease in accuracy, while label 7 maintains a consistent accuracy. Moreover, certain labels (2, 3, 4, 8, 9) also display decreased accuracy, with only labels 0 and 1 showing improvements. These results might not unequivocally support the assertion that FedAvg yields subpar performance.

2. Table 1 presents accuracy data where the Without exp (Original) configuration yields higher accuracy. The interpretation of these results is somewhat perplexing. Does this imply that the absence of the aggregation balancer results in superior performance?
3.	Analyzing Table 2, it is observed that, on the CIFAR10 dataset, as data heterogeneity increases, the local balancer's performance falls behind the baseline. Could the reasons for this performance differential be elucidated?
4.	In Table 4, a notable deterioration in performance is observed when α is 0.1 and the aggregation balancer is employed. Could you provide an explanation for the specific circumstances in which a decrease in performance occurs at certain levels of data heterogeneity?
5.	Could a dedicated ablation study, focusing solely on the use of the aggregation balancer, be provided to validate its efficacy?
6.	The proposed method appears to lack consistency. In some instances, the local balancer may underperform compared to the baseline in specific settings, while the aggregation balancer can also degrade performance in certain scenarios. Paradoxically, the combined use of both balancers in certain settings results in a marked improvement in accuracy, like in CIFAR10 and α is 0.05. Could the underlying reasons for these variations in performance be explained?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
