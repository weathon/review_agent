# Federated Learning Can Find Friends That Are Advantageous

- Decision: Reject
- Scores: 5, 6, 3, 5

## Abstract
In Federated Learning (FL), the distributed nature and heterogeneity of client data present both opportunities and challenges. While collaboration among clients can significantly enhance the learning process, not all collaborations are beneficial; some may even be detrimental. In this study, we introduce a novel algorithm that assigns adaptive aggregation weights to clients participating in FL training, identifying those with data distributions most conducive to a specific learning objective. We demonstrate that our aggregation method converges no worse than the method that aggregates only the updates received from clients with the same data distribution. Furthermore, empirical evaluations consistently reveal that collaborations guided by our algorithm outperform traditional FL approaches. This underscores the critical role of judicious client selection and lays the foundation for more streamlined and effective FL implementations in the coming years.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a novel algorithm called Merit-based Federated Learning (MeritFed), designed to tackle the challenges posed by heterogeneous data distributions in Federated Learning (FL). The authors employ an adaptive selection of aggregation weights for clients based on their data distributions and provide a theoretical framework demonstrating the provable convergence of MeritFed under mild assumptions, indicating that the method can achieve faster convergence compared to existing approaches. The paper conducts experiments comparing MeritFed with other baseline methods like TAWT and FedAdp, highlighting its efficiency and reduced computational complexity, as it does not rely on intensive hypergradient estimations or cosine similarity for weight calculation. Meanwhile, the incorporation of zero-order mirror descent in the algorithm enhances privacy during the training process. Overall, the paper contributes to the advancement of Federated Learning by proposing a method that promotes collaborative learning while addressing the challenges of data heterogeneity and privacy concerns.

### Strengths
S1. The introduction of adaptive aggregation weights based on the data distributions of clients represents a significant advancement in Federated Learning, allowing for more effective collaboration and utilization of diverse datasets. 

S2. The paper provides a rigorous convergence analysis, proving that MeritFed converges under mild assumptions. This theoretical backing enhances the credibility and reliability of the proposed method compared to traditional FL approaches. 

S3. MeritFed is designed to handle outlier clients effectively, which is crucial in real-world scenarios where some clients may have non-representative or adversarial data. This robustness contributes to the overall stability of the learning process. 

S4.The paper discusses the potential for future extensions and enhancements, indicating that MeritFed can be adapted for large-scale FL scenarios. This forward-looking perspective suggests that the method can evolve to meet the demands of diverse applications, making it a valuable contribution to the field.

### Weaknesses
W1. Lack of theoretical or experimental support of the claim about the algorithm discerning optimal collaborators for faster convergence and better generalization.

W2. The paper does not clearly distinguish its approach from traditional heterogeneous federated learning, especially empirical risk-based methods.  

W3. Although communication is identified as a bottleneck in federated learning, the paper does not propose any improvements or solutions to this issue.

W4. Unclear definitions of key terms, such as the term "similar distributions," lead to potential confusion about the method's applicability and assumptions. 

W5. Inadequate experimental evaluation, such as existing federated learning benchmarks or ablation studies. This limits the understanding of the method's performance.

### Questions
Q1. In line 54, the authors claim that “Through this innovative training mechanism, our algorithm discerns which clients are optimal collaborators to ensure faster convergence and potentially better generalization”. Could you please provide theoretical proof or conduct an empirical analysis to validate the improvement of the generalization ability of the proposed method, such as performing a comparing test on unseen clients or other domains/clusters?

Q2. Domain adaptation addresses the issue of data distribution dissimilarity of client datasets, while most are empirical risk-based methods. How does the approach in this study differ in principle from the empirical risk-based domain adaptation methods in principle?

Q3. Communication is a bottleneck in federated learning. I am wondering how the proposed approach impacts communication efficiency.

Q4. In line 208, how to define “similar distributions”? Could you please provide a formal definition of the "similar distributions" or give a quantitative measure of distribution similarity used in this work?

Q5. The paper conducts experiments on different tasks, such as mean estimation, image classifications, and text classifications, including pervasive domain application tasks. Yet, it does not thoroughly evaluate the method's performance against existing federated learning benchmarks. The three selected baseline approaches are old. Please benchmark the proposed approach against the federated learning algorithms published after 2022. I am also wondering about the performance difference between the three methods presented in Section 3, which is worth conducting an experiment and reporting the results.

### Soundness
2

### Presentation
3

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
This work achieves collaborative learning among clients with both similar and different distributions by solving for the optimal weight allocation and provides a theoretical proof of its convergence. Numerous figures in the experiments visualize the dynamic changes in weights, demonstrating the effectiveness of MeritFed in image classification and text classification tasks.

### Strengths
1. The paper is well-written and clearly presents the dynamic process of weight allocation.
2. The analysis of feasible solutions surrounding the weight optimization objective function (Problem 7) is detailed, with extensive experimental evidence for each feasible solution.
3. Sufficient theoretical proofs are provided.

### Weaknesses
1. The computational overhead could limit the scalability of this method, as the experiments only involved 20 clients. As the number of clients increases, the computation for optimal weights will also increase significantly.
2. The paper’s contributions emphasize that benefits can be obtained from clients with both similar and different data distributions; however, this aspect is not explained in the experimental analysis.

### Questions
1. Solving for an optimal set of weights for each client entails significant computational overhead. Providing additional details on computational and communication costs would make the experiments more complete.
2. From the loss iteration graphs in Figures 4-7, it can be observed that larger values of \alpha lead to greater fluctuations in loss. Is this fluctuation due to weight allocation changes? Please analyze.
3. The weight allocation heatmaps in Figures 4-7 show that the weight distribution for MeritFed SMD and MeritFed MD becomes increasingly stable as iterations progress. Could weight optimization be conducted only in the early rounds, with fixed weights used in later stages?
4. Please elaborate on the statement “not all collaborations are beneficial; some may even be detrimental,” and analyze how harmful clients could be assigned smaller weights.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
This paper proposes a merit-based federated learning algorithm (MeritFed) that addresses data heterogeneity in federated learning (FL) by solving an auxiliary minimization problem in each iteration to adaptively select aggregation weights. The core innovation of the algorithm lies in leveraging auxiliary optimization tasks to enhance the model’s generalization capability and robustness, thereby enabling it to better adapt to the diverse data distributions across different clients.

### Strengths
S1: Extensive experimental analysis across multiple tasks.
S2: Sufficient theoretical proofs that ensure the algorithm’s convergence.
S3: A thorough analysis of the limitations of the study and potential future research directions.
S4: Code is open-sourced.

### Weaknesses
W1: The related work looks like methodology.
W2: The paper lacks a detailed explanation of the main research focus (Algorithm 1).

### Questions
Q1: The paper uses the variable x in several formulas, where it represents a d-dimensional vector, but what does it actually signify?
Q2: In Section 2.2, the authors propose three approaches to address the issue in line 7 of Algorithm 1. Do these methods cover most practical application scenarios, thereby making the algorithm more universally applicable?

### Soundness
2

### Presentation
2

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
The paper tackles non-IID problem in federated learning by adjusting the aggregation weights dynamically. For the aggregated gradients at each communication round, the proposed method finds the optimal aggregation weights that can minimize the loss on the target client. Experiments on some designed non-IID scenarios show that the proposed method achieves dynamic similarity-based weight-assigning by assigning large weights to clients with similar data distributions and small weights to clients with distinct data distributions.

### Strengths
1. The targeting problem is an essential challenge to Personalized FL. Discovering friendships or similarities across clients is criticial to advance the research of Personalised FL.

2. The overall component of the paper is comprehensive including problem formulation, theoretical analysis and experimental analsis.

3. The experiment analysis shows the proposed method can indeed find the inherent client distribution structure, which is achieved by dynamically assigning large weights to similar clients and small weights to distinct clients.

### Weaknesses
1. Lack of discussion about the overall FL framework. The proposed PFL objective function only focuses on the targeting client by leveraging the external knowledge from other clients. Eq.1 makes the problem formulation toward a transfer learning task rather than a federated learning. 

2. The introduction section should be enhanced by adding a more detailed discussion about discovering structures among FL clients, thus providing better support to the motivation. In the meantime, the related work is too long with irrelevant information. For example, communication compression or clustered FL is not related to the proposed topic.

3. Missing some related works, e.g. graph/structure-guided aggregation in PFL [1-4]. 

[1] J Ma, et al., Structured Federated Learning through Clustered Additive Modeling, NeurIPS 2023
[2] F Chen, et al., Personalized Federated Learning With A Graph, IJCAI 2022
[3] C Zhang, et al., GPFedRec: Graph-Guided Personalization for Federated Recommendation, KDD 2024
[4] Y Tan, et al., Federated Learning on Non-IID Graphs via Structural Knowledge Sharing, AAAI 2023

### Questions
Please refer to weakness.

### Soundness
3

### Presentation
3

### Contribution
2
