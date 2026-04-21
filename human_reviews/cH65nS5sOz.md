# Subgraph Federated Learning for Local Generalization

- Avg Score: 7.60
- Decision: Accept (Oral)
- Scores: 10, 6, 8, 6, 8

## Abstract
Federated Learning (FL) on graphs enables collaborative model training to enhance performance without compromising the privacy of each client. However, existing methods often overlook the mutable nature of graph data, which frequently introduces new nodes and leads to shifts in label distribution. Since they focus solely on performing well on each client's local data, they are prone to overfitting to their local distributions (i.e., local overfitting), which hinders their ability to generalize to unseen data with diverse label distributions. In contrast, our proposed method, FedLoG, effectively tackles this issue by mitigating local overfitting. Our model generates global synthetic data by condensing the reliable information from each class representation and its structural information across clients. Using these synthetic data as a training set, we alleviate the local overfitting problem by adaptively generalizing the absent knowledge within each local dataset. This enhances the generalization capabilities of local models, enabling them to handle unseen data effectively. Our model outperforms baselines in our proposed experimental settings, which are designed to measure generalization power to unseen data in practical scenarios. 
Our code is available at https://github.com/sung-won-kim/FedLoG

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper introduces FedLoG, a novel framework addressing local overfitting in subgraph federated learning. The key innovation lies in its approach to generating global synthetic data by condensing reliable information from head degree and head class nodes across clients. The framework employs a dual-branch architecture to process head and tail degree nodes differently, with adaptive weighting based on node degrees. Through feature scaling and prompt generation mechanisms, FedLoG enables each client to adaptively learn locally absent knowledge. The method is evaluated under three practical scenarios: unseen nodes, missing classes, and new clients. Experimental results demonstrate FedLoG's effectiveness in preventing local overfitting while maintaining privacy and improving generalization capabilities across various graph datasets.

### Strengths
Originality:
The paper introduces FedLoG, a framework that innovatively addresses the problem of local overfitting in subgraph federated learning by focusing on the adaptability and generalization of models across clients. The originality lies in its method of using global synthetic data generated from the most reliable node types—head degree and head class nodes—thus tackling the challenge of missing classes and mutable graph structures in a novel way.

Quality:
The research is characterized by its high quality, demonstrated through a comprehensive experimental setup that includes diverse real-world datasets and practical scenarios like unseen nodes, missing classes, and new clients. The thoroughness of the ablation studies and the detailed analysis of the results contribute to a robust validation of the proposed method, ensuring that the findings are both reliable and applicable.

Clarity:
The paper is clearly written, with well-organized sections that effectively convey complex ideas. The authors provide detailed explanations of their methodology, supported by illustrative diagrams and thorough descriptions of experimental setups. This clarity facilitates a deeper understanding of the framework's mechanisms and its impact on federated learning.

Significance:
The significance of this work is underscored by its potential to enhance federated learning applications where privacy and adaptability are crucial. By improving the generalization capabilities of local models and addressing the mutable nature of graph data, the paper contributes significantly to advancing federated learning technologies. Its approach to mitigating local overfitting while respecting privacy constraints is particularly relevant in today's data-sensitive environment, offering practical benefits and paving the way for future research in this area.

### Weaknesses
1） The paper seems to lack a thorough discussion on the complexity of the proposed FedLoG framework. Given the dual-branch architecture that separately handles head and tail degree nodes, alongside the mechanisms for feature scaling and prompt generation, a detailed analysis of computational and implementation complexities would be beneficial. Understanding these complexities is crucial for assessing the feasibility and scalability of the method in practical, large-scale applications.

2）Another potential limitation is the assumption that each client has distinct strengths in terms of node types or class distributions. While this assumption aids in leveraging reliable data from head nodes, it may not hold in scenarios where client data distributions are highly homogenous or overlapping. This could limit the framework's effectiveness and generalization in such environments.

### Questions
The datasets used in this paper are all publicly available (i.e. they are not task-oriented) and have been specifically processed to fit the described scenarios. While this approach provides a controlled environment for testing, is there potential for future work to incorporate specialized datasets that more naturally exhibit the challenges addressed by FedLoG? Such datasets could offer deeper insights into the framework's performance and applicability in real-world contexts where these issues are more prevalent.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a novel personalization method to solve the local overfitting problem of the subgraph federated learning. Specifically, they propose a two-branch prototypical network for handling high-degree and low-degree knowledge of each node,  and a global data generation strategy for tackling the limitation of tail nodes in each client.

### Strengths
1.	Sufficient comparison of experimental results demonstrates the performance advantages of the proposed method. 

2.	The figures and language are clear.

### Weaknesses
1. The local fitting process comprises two branches of modules and prototypical inference networks, which may bring expensive inference computational costs.

2. The ablation study about the effectiveness of dividing the inference process into two head and tail branches is required. In my opinion, this design works appearing more using the ensemble of two branches, instead of considering the head or tail degree of nodes. In fact, the effect of degree is only considered by using the weighted average of predictions, $\alpha$. The design of the two branches themselves are not involved to the degree at all. 

3.	The construction of global synthetic data requires the uploading of local graph nodes, which not only brings extra communication costs but also causes privacy leakage. 

4. This technique seems to apply a mixup of data in FL, which has been used in many previous works, e.g., [1,2]. The novelty may be limited.

[1] FedMix Approximation of Mixup under Mean Augmented Federated Learning NIPS 2022

[2] FedMDO: Privacy-Preserving Federated Learning via Mixup Differential Objective TCSVT 2024

5. Some key steps should be included in the maintext. For example, how to generate the edges for the synthetic data is required to be specified in the main text instead of the appendix.

6. In fact, this paper trains personalized models for each client since the learnable nodes and prototypes are customized for each client. Therefore, the personalization baselines should also be included [3].

[3] Federated Learning on Non-IID Graphs via Structural Knowledge Sharing. AAAI 2023.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the subgraph federated learning problem, where the author hopes to generate a global dataset that includes all categories, with s nodes for each category. The goal is to train a node classifier, and the optimization objective is to minimize the total empirical risk of all clients. Each client k's loss function includes both local data loss and global synthetic data loss. This approach allows each client to generalize across all categories, even if those categories are not present in their current client base. The author first discusses criteria for reliable data judgment and then proposes FedLog based on this criterion. FedLog consists of three steps: local training by clients, global aggregation by servers and data generation, followed by further server-side training. Through these three steps, the author solves the local overfitting problem in subgraph federated learning and conducts comprehensive experiments in three scenarios: Unseen Node, Missing Class and New Client.

### Strengths
In terms of methodology, the design of Feature scaling and prompt generator is very creative. Especially the introduction of the prompt generator can provide a graph structure for the generated nodes, which is a valuable idea.

In terms of writing, the content of the paper is introduced in detail, and both writing and presentation are professional.

In terms of experiments, the experimental design is comprehensive and solid, including multiple verification scenarios, and provides sufficient details to ensure reproducibility.

### Weaknesses
The discussion about 4.2.1 is not entirely convincing to me, even though it implicitly contains structural information, is this enough to express complex graph structures? Are there other methods of synthetic data in the Graph Learning field currently, such as graph compression, graph distillation techniques?

Considering that the current method is essentially an aggregation of features and has little to do with the graph structure, this feature aggregation approach is also common in federated learning for general classification tasks. Therefore, the paper should supplement discussions on data synthesis-based methods in the federated learning field in the related work section.

This paper chooses nodes with high degree centrality and class centrality for global data synthesis. How can we ensure that each category can generate $s$ nodes? If a certain category is a tail class or missing class among all clients, how can we guarantee generating high-quality s nodes for this category?

### Questions
The loss function of each client includes the loss of local data and the loss of global synthetic data. Does this design exist in previous methods based on data synthesis? Is it a standard practice?

Regarding the selection of reliability data, are there any other measurement standards besides degree and class frequency? Are there any other works with high recognition in this area?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this article, the proposed method, FedLoG, addresses this issue by mitigating local overfitting through global synthetic data generation, condensing reliable structural information and class representations from clients. By training on these synthetic data, FedLoG adapts to the knowledge gaps in each local dataset, enhancing local models' generalization capabilities to handle diverse, unseen data. Experimental results confirm FedLoG’s superiority in generalization over existing baselines in realistic settings.

### Strengths
As the paper deals with Federated learning, the major focus should be on what and which kind of data and knowledge is aggregated to train the global model. Based on this, there are following strengths observed:
1. The paper posses a good motivation and a limited but fair novelty.
2. The baselines considered are fair enough and the gain is significant.
3. The organization of the paper is quiet good.
4. the methodology containing local and global fitting with aggregations is also fairly novel.

### Weaknesses
1. The novelty is fair but also limited. As it looks like an ensemble of multiple techniques used in earlier literature.
2. The process contains mutliple steps which may result in overhead in terms of computation and storage.
3. There may be privacy issues while doing such proposed aggregation.
4. If any client will be having data bias, it may propagate to the global from local model as it is taking average.

### Questions
1. How can it be handeling the privacy concerns?
2. What complexity it possess in terms of time with respect to the other baselines? Will it also be significant or just comparable?
3. Comment 4 on weekness. How can the authors see that issue?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Federated Learning (FL) on graphs can struggle with local overfitting due to changes in graph data and shifts in label distributions. Existing methods only focus on local data, which limits their ability to generalize to unseen data. The proposed FedLoG method addresses this by generating global synthetic data from reliable class representations and structural information across clients. This synthetic data helps prevent local overfitting, improving the generalization of local models to unseen data. Experimental results show that FedLoG outperforms existing approaches, and the code is publicly available.

### Strengths
- This paper is well-organized and has reasonable motivation.
- The proposed method, FedLoG, is interesting in improving local generalization from overfitting.
- The experiments are solid, and extensive results are shown to support the proposed method.
- The experimental results show that the proposed method can achieve the sota performances across different federated graph datasets against strong baselines.

### Weaknesses
- Since the proposed method relies on synthetic data, it is of particular interest whether FedLoG can protect data privacy. Can differential privacy techniques be compatible with FedLoG to further guarantee privacy issues?
- Some important literature about generalization in vanilla federated learning (not limited to federated graph learning) should be cited and discussed. For instance, FedRoD [1] bridges the gap between global generalization and local personalization. Recently, FedLAW [2] and FedDisco [3] target global generalization from aggregation perspectives, and FedETF [4], which is inspired by neural collapse, shows a potential way of achieving personalization and generalization at the same time.

---
[1] Chen, H. Y., & Chao, W. L. On Bridging Generic and Personalized Federated Learning for Image Classification. In International Conference on Learning Representations 2022.

[2] Li, Z., Lin, T., Shang, X., & Wu, C. (2023, July). Revisiting weighted aggregation in federated learning with neural networks. In International Conference on Machine Learning (pp. 19767-19788). PMLR.

[3] Ye, R., Xu, M., Wang, J., Xu, C., Chen, S., & Wang, Y. (2023, July). Feddisco: Federated learning with discrepancy-aware collaboration. In International Conference on Machine Learning (pp. 39879-39902). PMLR.

[4] Li, Z., Shang, X., He, R., Lin, T., & Wu, C. (2023). No fear of classifier biases: Neural collapse inspired federated learning with synthetic and fixed classifier. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 5319-5329).

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3
