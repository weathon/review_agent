# Flashback: Understanding and Mitigating Forgetting in Federated Learning

- Decision: Reject
- Scores: 5, 5, 3, 3

## Abstract
In the realm of Federated Learning (FL), the convergence and effectiveness of learning algorithms can be severely hampered by the phenomenon of forgetting—where knowledge obtained in one round becomes diluted or lost in subsequent rounds. Such a challenge is a result of severe data heterogeneity across clients. Although FL algorithms like FedAvg have been pivotal, they often falter in scenarios of high data heterogeneity. This work delves into the nuances of this problem, establishing the critical role forgetting plays in the inefficient learning of FL in the context of severe data heterogeneity. Knowledge loss occurs in both the local update and the aggregation step; addressing one phase without considering the other will not mitigate forgetting. We introduce a novel metric that offers a granular measurement of forgetting at every round while ensuring that the occurrence of forgetting is distinctly recognized and not obscured by the simultaneous acquisition of new class-specific knowledge. Leveraging these insights, we propose Flashback, an FL algorithm that integrates a novel dynamic distillation approach. The knowledge of different models is estimated and the distillation loss is adapted accordingly. This adaptive distillation is applied both at the local and global update phases, ensuring models retain essential knowledge across rounds while also assimilating new knowledge. Our approach seeks to robustly mitigate the detrimental effects of forgetting, paving the way for more efficient and consistent FL algorithms, especially in environments of high data heterogeneity. By effectively mitigating forgetting, Flashback achieves faster convergence to target accuracy outperforming baselines, by being up to 88.5$\times$ faster and at least 4.6$\times$ faster across the different benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates the catastrophic forgetting issue in FL that occurs in both local training and server aggregation. It provides empirical analysis and insights into the forgetting issue and introduces a new method to mitigate this forgetting issue. The proposed method achieves much better convergence than the compared counterparts.

### Strengths
- The forgetting issue in FL is important and the analysis and the introduced method are technically sound.
- The proposed method achieves much better convergence than the compared counterparts.
- The paper is generally well-written and easy to follow.

### Weaknesses
- Some experimental details seem to be missing. e.g., what is the public dataset that is used for experiments on CIFAR, CINIC, and FEMNIST?
- The comparison with other methods may not be fair as the proposed method leverages a shared public dataset in the server while compared methods may not use it. Some papers on FL and KD also use a public dataset.  e.g., [1][2].
    - [1] Ensemble distillation for robust model fusion in federated learning. NeurIPS’20
    - [2] Performance optimization of federated person re-identification via benchmark analysis. ACMMM’20
- A straightforward baseline to consider is fine-tuning with the public dataset in the server, using soft labels from clients or ground truth labels. It would provide more insights into the significance of the proposed method. The reviewer would consider raising the rating if some of the concerns can be addressed.

### Questions
- What is the impact of different selection choices of public datasets? Would the method still work if the data distribution of the public dataset is different from the client’s data distribution?
- What is the backbone used to train CIFAR and CINIC datasets? Is the method robust across different backbones?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper studies the problem of forgetting in FL. Specifically, they show that several standard federated optimization methods can fail in high-heterogeneity settings due to local and global forgetting, which respectively occur during client training and server aggregation. To address this, the authors use distillation at both the server and client during FL. At the server, they distill an ensemble of models from client fine-tuning and the previous server round into a new server model. At the client, they distill the initial (l.e. server) model into the fine-tuned model. The distillation additionally weights the logits based on (aggregated) client label counts.

### Strengths
The paper makes an interesting point about how prior works focus on making either the global or local step robust but fail to consider both. 

The method outperforms a variety of baselines which use regularization / distillation.

### Weaknesses
Please closely examine the claim in Discussion 5.1 about related work. Li et al. 2020 (FedProx) makes no assumptions about public server-side data.

It would be good if you can include an ablation on using distillation only at the server / clients. The paper claims (page 4, above Fig.2) "Moreover, local forgetting and global forgetting are intertwined, which means addressing the issue at only one of the phases will not be sufficient, since it will happen at the next phase, and therefore have a cascading effect into the same phase at the next round." 
I think this statement intuitively makes sense but it could use more support.

More generally, an ablation on various components of Flashback would be helpful. Relative to FedDF there are a lot of things going on, i.e. including the previous round teacher, weighting the logits, and local distillation. Based on the story of the paper I would expect adding local distillation to be the most important factor.

Figure 2 would be more helpful if you use the same initial model for all methods. Also consider only showing one row of methods.

### Questions
FedDF reports very high numbers on CIFAR10 (Table 1 in https://proceedings.neurips.cc/paper_files/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf). Was the evaluation of FedDF too limited? Why can they reach up to 75% in those experiments?

### Soundness
2 fair

### Presentation
2 fair

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
The paper targets the problem of data heterogeneity in the federated setting. The authors introduce two sources of performance degradation in non-iid settings: local forgetting and global forgetting. To mitigate the forgettings, they propose FLASHBACK, which employs weighted knowledge distillation on the client and server sides. Clients use the global model as their teacher, and the server uses all the new updates + the last round model as a set of teachers.

### Strengths
* The evaluations show the superiority of FLASHBACK.
* The evaluations are comprehensive on different metrics.
* Using the global/local forgettings improves the understanding of the underlying problem in heterogenous FL, and it should be considered in this area as well.

### Weaknesses
* The paper assumes that the distribution of the public data is the same as training data (public data is a part of the original dataset). In other words, in the experiments, representative public data is available, which does not usually happen in reality.
* It is unclear if the other baseline methods benefit from the public dataset. Their performance can improve if the server can train on the centralized public dataset as well. 
* The algorithm has two parts: local and global KD. An ablation study on each part needs to be included.
* Using KD in the clients and server and forgetting in federated learning is not new. Plenty of previous works, such as [1], use KD in client and server to mitigate forgetting.
* Sharing label information with the server is not privacy-preserving.

[1] Ma, Yuhang, et al. "Continual federated learning based on knowledge distillation." IJCAI 2022.

### Questions
* How does your method work on more complex datasets or models?
* How does your paper compare with the federated continual learning papers? 
* Please check out the weakness section for the rest of the questions.

### Soundness
2 fair

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
The paper studies the problem of forgetting in federated learning, particularly in contexts where statistical heterogeneity across clients in high. To do this, the manuscript proposes a new metric to measure forgetting both at the client level (after local updates) and at the the global level (after aggregation at the server), and a method to alleviate this phenomenon by further distilling the knowledge of the local models plus the last round's model. Results are presented on three benchmark datasets.

### Strengths
- The problem of forgetting in federated learning is important and timely. 
- The paper is presented in such a way that it builds upon simple ideas that are intuitive and easy to follow. 
- The use of distillation to mitigate forgetting seems natural given the connections of distillation to predictive churn [1]. 

[1] Jiang, H., Narasimhan, H., Bahri, D., Cotter, A., & Rostamizadeh, A. (2021, October). Churn Reduction via Distillation. In International Conference on Learning Representations.

### Weaknesses
- One of the stated contributions of the paper is to "show how and where forgetting happens in FL". I'm not convinced this question is answered by the manuscript. In particular, only two possible causes are explored: local training and server aggregation. Other possible factors are not considered, e.g., the ordering of the clients, the ordering of the data in the clients [2]. I believe the manuscript should be more specific in this statement or, hopefully, perform a more systematic exploration of what really affects forgetting in FL.  
- In the same line, the paper defines, measures and tests local forgetting. Later on, it concludes that some amount of it is necessary for learning. This conclusion is valuable, but this nuance is not reflected in the introduction nor in the motivation of the manuscript. 
- There is little discussion of the public data used by the algorithm until Section 5.1. Even then, I am left with questions regarding how it can affect the forgetting behavior. What distribution does this data need to be drawn from? Can it exacerbate forgetting if drawn from the wrong distribution?
- I am surprised that several baselines did not converge for FEMNIST in Table 1. This is a fairly simple benchmark that should achieve good performance with a CNN. 

[2] Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., & Gordon, G. J. (2018, September). An Empirical Study of Example Forgetting during Deep Neural Network Learning. In International Conference on Learning Representations.

### Questions
- I found Figures 2 and 5 confusing. I'm not sure what the colors refer to, and why Flashback is performing better according to these figures. Please clarify. 
- In line with the connections between distillation and algorithmic churn, and with other possible causes of forgetting, future versions of the manuscript would benefit from studying forgetting at the example level (see [2]) for a given test dataset at the server.

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor
